import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torchvision.transforms import Compose, ToTensor, Lambda, ToPILImage, CenterCrop, Resize
from torchvision import datasets, transforms
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import os

from ex02_model import Unet
from ex02_diffusion import Diffusion, linear_beta_schedule, cosine_beta_schedule, sigmoid_beta_schedule
from torchvision.utils import save_image

import argparse


def parse_args():
    parser = argparse.ArgumentParser(description='Train a neural network to diffuse images')
    parser.add_argument('--batch_size', type=int, default=64, help='input batch size for training (default: 64)')
    parser.add_argument('--timesteps', type=int, default=100, help='number of timesteps for diffusion model (default: 100)')
    parser.add_argument('--epochs', type=int, default=2, help='number of epochs to train (default: 5)')
    parser.add_argument('--lr', type=float, default=0.003, help='learning rate (default: 0.003)')
    # parser.add_argument('--momentum', type=float, default=0.9, help='SGD momentum (default: 0.9)')
    parser.add_argument('--no_cuda', action='store_true', default=False, help='disables CUDA training')
    # parser.add_argument('--seed', type=int, default=1, help='random seed (default: 1)')
    parser.add_argument('--log_interval', type=int, default=100, help='how many batches to wait before logging training status')
    parser.add_argument('--save_model', action='store_true', default=False, help='For Saving the current Model')
    parser.add_argument('--run_name', type=str, default="DDPM")
    parser.add_argument('--dry_run', action='store_true', default=False, help='quickly check a single pass')
    return parser.parse_args()


def sample_and_save_images(n_images, diffusor, model, device, store_path, rtransform):
    # TODO: Implement - adapt code and method signature as needed

    image_classes=torch.randint(0, 10, (n_images,), device = device).long()
    print(f'image classes for sampling:{image_classes}')
    tensor_images = torch.tensor(diffusor.sample(model, diffusor.img_size, batch_size=n_images, y=image_classes, p_uncond=0.2, cfg_scale=4 ))
    #print(tensor_images[0, 0, 0, :, :].size())  #(timestep, batch_size, channels, h, w)

    timesteps = [99, 95, 90, 80, 40, 10, 5, 2, 0]  # timesteps
    rows = n_images  # rows
    columns = len(timesteps)  # columns
    plot_count = 1
    fig = plt.figure(figsize=(50, 50))
    plt.suptitle("Diffusion over timesteps", fontsize=18)

    for row in range(n_images):
        for col in range(columns):
            fig.add_subplot(rows, columns, plot_count)
            plt.title('image: {}, timestep: {}'.format(row, timesteps[col]))
            plt.xlabel(row)
            plt.imshow(rtransform(tensor_images[timesteps[col], row, :, :, :]))
            plt.axis('off')
            plot_count = plot_count + 1

    plt.tight_layout()
    plt.show()
    #print(images.size())

    #save_image(images,store_path)




'''def val_test(model, valloader, diffusor, device, args):
    # TODO: Implement - adapt code and method signature as needed
    batch_size = args.batch_size
    timesteps = args.timesteps
    num_classes = 10
    epoch = args.epochs

    pbar = tqdm(valloader)

    for step, (images, labels) in enumerate(pbar):

        images = images.to(device)

        t = torch.randint(0, timesteps, (len(images),), device=device).long()  # len(images) = batch size
        image_classes = labels.to(device)#labels.to(device)
        #print(f'labels type is: {image_classes}')
        loss = diffusor.p_losses(model, images, t, p_uncond=0.2, loss_type="l2", y=image_classes)

        loss.backward()

        if step % args.log_interval == 0:
            print('Validation Test Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                1, step * len(images), len(valloader.dataset),
                       100. * step / len(valloader), loss.item()))
        if args.dry_run:
            break'''


def train(model, trainloader, optimizer, diffusor, epoch, device, args):    #trains per batch
    batch_size = args.batch_size
    timesteps = args.timesteps
    num_classes = 10

    pbar = tqdm(trainloader)    #assigns additional progress bar to display in runtime

    for step, (images, labels) in enumerate(pbar):

        images = images.to(device)      #images.shape() = 4D
        optimizer.zero_grad()

        # Algorithm 1 line 3: sample t uniformly for every example in the batch
        t = torch.randint(0, timesteps, (len(images),), device=device).long()   #len(images) = batch size
        image_classes = labels.long().to(device)#labels.to(device)

        loss = diffusor.p_losses(model, images, t, loss_type="l2", y=image_classes)

        loss.backward()
        optimizer.step()

        if step % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, step * len(images), len(trainloader.dataset),
                100. * step / len(trainloader), loss.item()))
        if args.dry_run:
            break


#def test(model, testloader, diffusor, device, args):
    # TODO (2.2): implement testing functionality, including generation of stored images.
    pass


def run(args):
    timesteps = args.timesteps
    image_size = 32  # TODO (2.5): Adapt to new dataset
    channels = 3
    epochs = args.epochs
    batch_size = args.batch_size
    device = "cuda" if not args.no_cuda and torch.cuda.is_available() else "cpu"

    model = Unet(dim=image_size, channels=channels, dim_mults=(1, 2, 4,), class_free_guidance=True, p_uncond=0.2, num_classes=10).to(device)
    optimizer = AdamW(model.parameters(), lr=args.lr)

    my_scheduler = lambda x: linear_beta_schedule(0.0001, 0.02, x)
    my_scheduler_cos = lambda y: cosine_beta_schedule(y, 0.008)
    my_scheduler_sigm = lambda z: sigmoid_beta_schedule(0.0001, 0.02, z)
    diffusor = Diffusion(timesteps, my_scheduler_cos, image_size, device)
                            # 100,  (.0001 -to-0.5), 32
    # define image transformations (e.g. using torchvision)
    transform = Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),    # turn into torch Tensor of shape CHW, divide by 255
        transforms.Lambda(lambda t: (t * 2) - 1)   # scale data to [-1, 1] to aid diffusion process
    ])
    reverse_transform = Compose([
        Lambda(lambda t: (t.clamp(-1, 1) + 1) / 2),
        Lambda(lambda t: t.permute(1, 2, 0)),  # CHW to HWC
        Lambda(lambda t: t * 255.),
        Lambda(lambda t: t.numpy().astype(np.uint8)),
        ToPILImage(),
    ])

    dataset = datasets.CIFAR10('/proj/aimi-adl/CIFAR10/', download=False, train=True, transform=transform)
    trainset, valset = torch.utils.data.random_split(dataset, [int(len(dataset) * 0.9), len(dataset) - int(len(dataset) * 0.9)])
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
    valloader = DataLoader(valset, batch_size=batch_size, shuffle=False)

    # Download and load the test data
    testset = datasets.CIFAR10('/proj/aimi-adl/CIFAR10/', download=False, train=False, transform=transform)
    testloader = DataLoader(testset, batch_size=int(batch_size/2), shuffle=True)

    for epoch in range(epochs):
        train(model, trainloader, optimizer, diffusor, epoch, device, args)
        #val_test(model, valloader, diffusor, device, args)

    #test(model, testloader, diffusor, device, args)

    save_path = "<path/to/my/images>"  # TODO: Adapt to your needs
    n_images = 8
    sample_and_save_images(n_images, diffusor, model, device, save_path, reverse_transform)
    #torch.save(model.state_dict(), os.path.join("/proj/aimi-adl/models", args.run_name, f"UNetchckpt.pt"))     #saves model parameters as a checkpoint
    #torch.save(model.state_dict(), os.path.join("Users/Ritwik/ADL SSH/ex02", args.run_name, f"UNetchckpt.pt"))     #saves model parameters as a checkpoint


if __name__ == '__main__':
    args = parse_args()
    # TODO (2.2): Add visualization capabilities
    run(args)
