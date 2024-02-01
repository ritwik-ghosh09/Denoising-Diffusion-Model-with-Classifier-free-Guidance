
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from ex02_helpers import extract
from tqdm import tqdm
from ex02_model import Unet

def linear_beta_schedule(beta_start, beta_end, timesteps):
    """
    standard linear beta/variance schedule as proposed in the original paper
    """
    return torch.linspace(beta_start, beta_end, timesteps)


def cosine_beta_schedule(timesteps, s=0.008):

    'cosine schedule as proposed in https://arxiv.org/abs/2102.09672'
    # Implementing cosine beta/variance schedule as discussed in the paper mentioned above
    T = timesteps
    t = torch.linspace(0, T, T+1)
    alphas_cumprod = torch.cos(((t / T) + s) / (1 + s) * torch.pi * 0.5) ** 2       # f(t)
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]     # f(t) / f(0)
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])

    #plt.plot(t / T, alphas_cumprod, label='Cosine Scheduler')
    #plt.xlabel('t / T')
    #plt.ylabel('alpha_cumprod_t')
    #plt.savefig('CosineScheduler.png')

    #plt.show()
    return torch.clip(betas, 0.0001, 0.9999)



def sigmoid_beta_schedule(beta_start, beta_end, timesteps):   # sigmoidal beta schedule - following a sigmoid function

    # Implementing a sigmoidal beta schedule. 
    # Note that it saturates fairly fast for values -x << 0 << +x
    T = timesteps
    t = torch.linspace(0, T, T)
    s_lim = 6

    sigmoid = torch.sigmoid(-s_lim + 2*t*s_lim / T) * (beta_end - beta_start) + beta_start

    plt.plot(t / timesteps, sigmoid, label='Sigmoid Scheduler', color='red')
    #plt.savefig('Sigmoid.png')
    plt.show()

    return sigmoid


class Diffusion:

    # Adapting all methods in this class for the conditional case. You can use y=None to encode that you want to train the model fully unconditionally.
                        # 100,       my_scheduler,        32
    def __init__(self, timesteps, get_noise_schedule, img_size, device="cuda"):
        """
        Takes the number of noising steps, a function for generating a noise schedule as well as the image size as input.
        """
        self.timesteps = timesteps  # 100

        self.img_size = img_size    # 32
        self.device = device

        # defining beta schedule
        self.betas = torch.Tensor(get_noise_schedule(self.timesteps))     # (0.001.......100 equally spaced divisions till......0.02)

        # Computing the central values for the equation in the forward pass already here in order to quickly use them in the forward pass.
        # defining alphas
        self.alphas = torch.Tensor(1. - self.betas)
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)    
        alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)    #since the first element will be 1
        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)

        # calculations for diffusion q(x_t | x_{t-1}) and others    # TODO
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - self.alphas_cumprod)


        # calculations for posterior q(x_{t-1} | x_t, x_0)          # TODO
        self.posterior_variance = self.betas * (1. - alphas_cumprod_prev) / (1. - self.alphas_cumprod)


        self.sqrt_betas = torch.sqrt(self.betas)

    @torch.no_grad()
    def p_sample(self, model, x, t, t_index, y=None, p_uncond=None, cfg_scale=None):
        # Implementing the reverse diffusion process of the model for (noisy) samples x and timesteps t. x and t both have a batch dimension
        x = x.to(torch.device('cuda'))
        t = t.to(torch.device('cuda'))
        y = y.to(torch.device('cuda'))

        sqrt_recip_alphas_t = extract(self.sqrt_recip_alphas, t, x.shape)
        betas_t = extract(self.betas, t, x.shape)
        sqrt_one_minus_alphas_cumprod_t = extract(self.sqrt_one_minus_alphas_cumprod, t, x.shape)

        #self.standard_model = Unet(dim=32, class_free_guidance=False, num_classes=10).to('cuda')

        if cfg_scale == 0:  #normal DDPM with class info.
            final_pred_noise = model.forward(x, t, y, 0)

        else:
            final_pred_noise = model.forward(x,t,y, 0) + (cfg_scale * (model.forward(x,t,y, prob_uncond=0) -  model.forward(x,t, y, 1) ))


        model_mean = sqrt_recip_alphas_t * (
        x - (betas_t * final_pred_noise / sqrt_one_minus_alphas_cumprod_t))     # Equation 11 in the paper
        
        
        # Using the model (noise predictor) to predict the mean

        if t_index == 0:
            return model_mean

        else:
            z_noise = torch.randn_like(x)
            posterior_variance_t = extract(self.posterior_variance, t, x.shape)
            return model_mean + (torch.sqrt(posterior_variance_t) * z_noise)  # Returns the image at timestep t-1 -> x_t-1.


    # Algorithm 2 (including returning all images)
    @torch.no_grad()
    def sample(self, model, image_size, batch_size=16, channels=3, y=None, p_uncond=None, cfg_scale=None):     #samples from individual batches
        # Implementing the full reverse diffusion loop from random noise to an image, iteratively ''reducing'' the noise in the generated image.

        x_T = torch.randn(batch_size, channels, image_size, image_size)
        x_gen = []
        time = []

        for idx in tqdm(range(0,self.timesteps)[::-1]):
            x_T = self.p_sample(model, x_T, torch.full((batch_size,), idx), idx, y, p_uncond, cfg_scale)     # x_t-1
            x_gen.append(x_T.cpu().numpy())
            time.append(idx)
        print(f'timesteps: {time}')
        return np.array(x_gen)   # Returning batches of generated images throughout the reverse timesteps


    # forward diffusion (using the nice property)
    def q_sample(self, x_zero, t, noise=None):
        # Implementing the forward diffusion process using the beta-schedule defined in the constructor

        self.noise = noise
        if self.noise is None:    # if noise is None, you will need to create a new noise vector, otherwise use the provided one.
            self.noise = torch.randn_like(x_zero)

        sqrt_alphas_cumprod_t = extract(self.sqrt_alphas_cumprod, t, x_zero.shape)  #gets corr. values for indices in t
        sqrt_one_minus_alphas_cumprod_t = extract(self.sqrt_one_minus_alphas_cumprod, t, x_zero.shape)

        # mean + variance --> x_noisy_images
        return (sqrt_alphas_cumprod_t * x_zero) \
               + (sqrt_one_minus_alphas_cumprod_t * self.noise)     # returns a batch of noisy images x_t for random timesteps

    def p_losses(self, denoise_model, x_zero, t, *, loss_type="l2", y=None, noise=None):   #(Unet, images, t=[batch of randomint between 0-100]  )
                        # Unet,         4D,
       
        # Computing the input to the network using the forward diffusion process and predicting the noise using the model

        inpT_Unet = self.q_sample(x_zero, t, noise)


        noise_pred = denoise_model.forward(inpT_Unet, t, y, 0.2)    #batch of noisy images along with their respective timesteps fed into U-Net


        if loss_type == 'l1':
            loss = F.l1_loss(noise_pred, self.noise)       # TODO (2.2): implement an L1 loss for this task

        elif loss_type == 'l2':
            loss = F.mse_loss(noise_pred, self.noise)       # TODO (2.2): implement an L2 loss for this task

        else:
            raise NotImplementedError()

        return loss
