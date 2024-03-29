import math
from functools import partial
from einops import rearrange, reduce, repeat
from torch import nn, einsum
import torch.nn.functional as F
from ex02_helpers import *
from abc import abstractmethod


# Niels Rogge (nielsr) & Kashif Rasul (kashif): https://huggingface.co/blog/annotated-diffusion (last access: 23.05.2023),
# which is based on
# Phil Wang (lucidrains): https://github.com/lucidrains/denoising-diffusion-pytorch (last access: 23.05.2023)

class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):    #  (batch of timesteps, 1) -> (batch of timesteps, dim ), dim = dim of each embedded timestep
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


class WeightStandardizedConv2d(nn.Conv2d):
    """
    https://arxiv.org/abs/1903.10520
    weight standardization purportedly works synergistically with group normalization
    """

    def forward(self, x):
        eps = 1e-5 if x.dtype == torch.float32 else 1e-3

        weight = self.weight
        mean = reduce(weight, "o ... -> o 1 1 1", "mean")
        var = reduce(weight, "o ... -> o 1 1 1", partial(torch.var, unbiased=False))
        normalized_weight = (weight - mean) * (var + eps).rsqrt()

        return F.conv2d(
            x,
            normalized_weight,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )


class Block(nn.Module):
    def __init__(self, dim, dim_out, groups=8):
        super().__init__()
        self.proj = WeightStandardizedConv2d(dim, dim_out, 3, padding=1)
        self.norm = nn.GroupNorm(groups, dim_out)
        self.act = nn.SiLU()

    # Note: This implements FiLM conditioning, see https://distill.pub/2018/feature-wise-transformations/ and
    # http://arxiv.org/pdf/1709.07871.pdf
    def forward(self, x, scale_shift=None):
        x = self.proj(x)
        x = self.norm(x)

        if exists(scale_shift):
            scale, shift = scale_shift
            x = x * (scale + 1) + shift

        x = self.act(x)
        return x


class ResnetBlock(nn.Module):
    """https://arxiv.org/abs/1512.03385"""

    def __init__(self, dim, dim_out, *, time_emb_dim=None, classes_emb_dim=None, groups=8):
        super().__init__()             # vector length of every timestep = time_dim
        self.dim_out = dim_out
        self.time_emb_dim = time_emb_dim
        self.classes_emb_dim = classes_emb_dim
        self.block1 = Block(dim, dim_out, groups=groups)
        self.block2 = Block(dim_out, dim_out, groups=groups)
        self.res_conv = nn.Conv2d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

        full_emb_dim = int(default(self.time_emb_dim, 0)) + int(default(self.classes_emb_dim, 0))  # defaults to 0 embedding in case None is passed


        self.mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(full_emb_dim, self.dim_out * 2)
        ) if exists(self.time_emb_dim) or exists(self.classes_emb_dim) else None

    def forward(self, x, time_emb=None, class_emb=None):    #feeds additional class_emb along with time_emb and noisy images
                        #(batch, time_dim), (batch #classes, cls_dim)
        self.class_emb = class_emb
        scale_shift = None
        if exists(self.mlp) and (exists(time_emb) or exists(class_emb)):
            cond_emb = tuple(filter(exists, (time_emb, class_emb)))  #results in two lists with filtered-out None values
            cond_emb = torch.cat(cond_emb, dim=-1)
            cond_emb = self.mlp(cond_emb)          # conda_emb.shape = (batch, full_emb_dim) -> (batch, dim_out * 2)
            cond_emb = rearrange(cond_emb, 'b c -> b c 1 1')
            scale_shift = cond_emb.chunk(2, dim=1)
        h = self.block1(x, scale_shift=scale_shift)

        h = self.block2(h)

        return h + self.res_conv(x)


class Attention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.scale = dim_head**-0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)
        self.to_out = nn.Conv2d(hidden_dim, dim, 1)

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = map(
            lambda t: rearrange(t, "b (h c) x y -> b h c (x y)", h=self.heads), qkv
        )
        q = q * self.scale

        sim = einsum("b h d i, b h d j -> b h i j", q, k)
        sim = sim - sim.amax(dim=-1, keepdim=True).detach()
        attn = sim.softmax(dim=-1)

        out = einsum("b h i j, b h d j -> b h i d", attn, v)
        out = rearrange(out, "b h (x y) d -> b (h d) x y", x=h, y=w)
        return self.to_out(out)


# Linear attention variant, scales linear with sequence length
# Shen et al.: https://arxiv.org/abs/1812.01243
# https://github.com/lucidrains/linear-attention-transformer
class LinearAttention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.scale = dim_head**-0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)

        self.to_out = nn.Sequential(nn.Conv2d(hidden_dim, dim, 1),
                                    nn.GroupNorm(1, dim))

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = map(
            lambda t: rearrange(t, "b (h c) x y -> b h c (x y)", h=self.heads), qkv
        )

        q = q.softmax(dim=-2)
        k = k.softmax(dim=-1)

        q = q * self.scale
        context = torch.einsum("b h d n, b h e n -> b h d e", k, v)

        out = torch.einsum("b h d e, b h d n -> b h e n", context, q)
        out = rearrange(out, "b h c (x y) -> b (h c) x y", h=self.heads, x=h, y=w)
        return self.to_out(out)


# Wu et al.: https://arxiv.org/abs/1803.08494
class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.GroupNorm(1, dim)

    def forward(self, x):
        x = self.norm(x)
        return self.fn(x)


class Unet(nn.Module):
    def __init__(
        self,
        dim,    #image size -> 32
        init_dim=None,
        out_dim=None,
        dim_mults=(1, 2, 4, 8),     #dimension multipliers
        channels=3,
        resnet_block_groups=4,
        class_free_guidance=False,  
        p_uncond=None,
        num_classes=None,
    ):
        super().__init__()

        # determining dimensions
        self.channels = channels
        input_channels = channels   # adapted from the original source
        self.class_free_guidance = class_free_guidance
        self.p_uncond = p_uncond

        init_dim = default(init_dim, dim)
        self.init_conv = nn.Conv2d(input_channels, init_dim, 1, padding=0)  # changed to 1 and 0 from 7,3

        # list of inp_out channels along U-Net
        dims = [init_dim, *map(lambda m: dim * m, dim_mults)]       # [init_dim, 32, 64, 128, 256]
        in_out = list(zip(dims[:-1], dims[1:]))         # [ (init_dim, 32), (32, 64), (64, 128), (128, 256) ]

        block_klass = partial(ResnetBlock, groups=resnet_block_groups)

        # time embeddings
        time_dim = dim * 4

        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(dim),      # (batch, dim) ,   where dim = vector length of each embedded timestep
            nn.Linear(dim, time_dim),       # dim -> time_dim
            nn.GELU(),
            nn.Linear(time_dim, time_dim),      # returns (batch, time_dim)
        )

        # Implementing a class embedder for the conditional part of the classifier-free guidance & defining a default
        self.classes_emb = nn.Embedding(num_classes, dim)
        self.null_classes_emb = nn.Parameter(torch.randn(dim))
        classes_dim = dim * 4 if class_free_guidance is not False else 0

        self.classes_mlp = nn.Sequential(
            nn.Linear(dim, classes_dim),
            nn.GELU(),
            nn.Linear(classes_dim, classes_dim)     # (num_classes, class_dim)
        )

        # layers
        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        num_resolutions = len(in_out)       # 4 resolutions for encoding as well as decoding

        # Adapting all blocks accordingly such that they can accommodate a class embedding as well
        for ind, (dim_in, dim_out) in enumerate(in_out):  # in_out = [ (init_dim, 32), (32, 64), (64, 128), (128, 256) ]
            is_last = ind >= (num_resolutions - 1)

            self.downs.append(
                nn.ModuleList(
                    [
                        block_klass(dim_in, dim_in, time_emb_dim=time_dim, classes_emb_dim = classes_dim),
                        block_klass(dim_in, dim_in, time_emb_dim=time_dim, classes_emb_dim = classes_dim),
                        Residual(PreNorm(dim_in, LinearAttention(dim_in))),     # Residual(fn)
                        Downsample(dim_in, dim_out)
                        if not is_last
                        else nn.Conv2d(dim_in, dim_out, 3, padding=1),
                    ]
                )
            )

        mid_dim = dims[-1]
        self.mid_block1 = block_klass(mid_dim, mid_dim, time_emb_dim=time_dim, classes_emb_dim = classes_dim)
        self.mid_attn = Residual(PreNorm(mid_dim, Attention(mid_dim)))
        self.mid_block2 = block_klass(mid_dim, mid_dim, time_emb_dim=time_dim, classes_emb_dim = classes_dim)

        for ind, (dim_in, dim_out) in enumerate(reversed(in_out)):  # in_out = [ (init_dim, 32), (32, 64), (64, 128), (128, 256) ]
            is_last = ind == (len(in_out) - 1)

            self.ups.append(
                nn.ModuleList(
                    [
                        block_klass(dim_out + dim_in, dim_out, time_emb_dim=time_dim, classes_emb_dim = classes_dim),
                        block_klass(dim_out + dim_in, dim_out, time_emb_dim=time_dim, classes_emb_dim = classes_dim),
                        Residual(PreNorm(dim_out, LinearAttention(dim_out))),
                        Upsample(dim_out, dim_in)
                        if not is_last
                        else nn.Conv2d(dim_out, dim_in, 3, padding=1),
                    ]
                )
            )

        self.out_dim = default(out_dim, channels)

        self.final_res_block = block_klass(dim * 2, dim, time_emb_dim=time_dim, classes_emb_dim = classes_dim)
        self.final_conv = nn.Conv2d(dim, self.out_dim, 1)


    def forward(self, x, time, classes=None, prob_uncond=None):

        x = self.init_conv(x)
        r = x.clone()
        p_uncond = default(prob_uncond, self.p_uncond)  #probability of replacing class_emb with a null token


        t = self.time_mlp(time)     # t.shape = (batch, time_dim),   where time_dim = 4 * img_size

        # Implementing the class conditioning


#  - for each element in the batch, the class embedding is replaced with the null token with a certain probability during training
        '''keep_prob_mask = torch.zeros(classes_emb.shape[0]).float().uniform_(0, 1)  #uniform distribution between (0,1)

        for idx, _ in enumerate(classes_emb):
            if keep_prob_mask[idx] <= p_uncond:   #returns null_class_emb for p_uncond=1 & entire class_emb for p_uncond=0
                classes_emb[idx] = self.null_classes_emb'''


        classes_emb = self.classes_emb(classes)

        batch = classes_emb.shape[0]
        if p_uncond > 0:
            keep_mask = prob_mask_like((batch,), 1 - p_uncond, device='cuda')
            null_classes_emb = repeat(self.null_classes_emb, 'd -> b d', b=batch)

            classes_emb = torch.where(
                rearrange(keep_mask, 'b -> b 1'),
                classes_emb,
                null_classes_emb
            )

        
        c = self.classes_mlp(classes_emb)





        h = []      #stores the downsampled outputs to be used as skip connections

        #  - analogously to the time embedding, the class embedding is provided in every ResNet block as additional conditioning
        for block1, block2, attn, downsample in self.downs:
            x = block1(x, t, c)
            h.append(x)

            x = block2(x, t, c)
            x = attn(x)
            h.append(x)

            x = downsample(x)

        x = self.mid_block1(x, t, c)
        x = self.mid_attn(x)
        x = self.mid_block2(x, t, c)

        for block1, block2, attn, upsample in self.ups:
            x = torch.cat((x, h.pop()), dim=1)
            x = block1(x, t, c)

            x = torch.cat((x, h.pop()), dim=1)
            x = block2(x, t, c)
            x = attn(x)

            x = upsample(x)

        x = torch.cat((x, r), dim=1)

        x = self.final_res_block(x, t, c)
        return self.final_conv(x)
