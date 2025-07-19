import torch

from MyDiffusion.diffusion import Diffusion
from MyDiffusion.utils import print_seq

from torchvision.datasets import MNIST
import torchvision.transforms as transforms

def data_prepare():
    transform = transforms.Compose([
        transforms.Resize(32),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])

    train = MNIST(root='./data', train=True, download=True, transform=transform)
    test = MNIST(root='./data', train=False, download=True, transform=transform)
    
    return train, test

import torch
import argparse

from MyDiffusion.diffusion import Diffusion
from MyDiffusion.utils import print_seq

from torchvision.datasets import MNIST
import torchvision.transforms as transforms

def data_prepare():
    transform = transforms.Compose([
        transforms.Resize(32),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])

    train = MNIST(root='./data', train=True, download=True, transform=transform)
    test = MNIST(root='./data', train=False, download=True, transform=transform)
    
    return train, test

def main(args):
    train, test = data_prepare()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = Diffusion(
        n_timesteps=args.time_steps,
        in_channels=1,
        out_channels=1,
        custom_channel_scale=[128, 128, 256, 256, 512, 512],
        train_set=train,
        test_set=test,
        train_batch_size=args.batch_size,
        test_batch_size=8,
        device=device,
        learning_rate=args.lr
    )

    # model.load('/content/drive/My Drive/models/DDPM_MNIST/MNIST_T1000_E30_S.pt')
    # model.sampling_weights.load('/content/drive/My Drive/models/DDPM_MNIST/MNIST_T1000_E30_W.pt')
    
    history = model.train(
        n_epoch=args.epochs,
        p_uncond=args.p_uncond
    )
    
    print_seq(history)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train Diffusion Model on MNIST')
    parser.add_argument('--time-steps', type=int, default=1000, help='Number of time steps for diffusion')
    parser.add_argument('--batch-size', type=int, default=16, help='Input batch size for training')
    parser.add_argument('--epochs', type=int, default=30, help='Number of epochs to train')
    parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate')
    parser.add_argument('--p-uncond', type=float, default=0.1, help='Probability of unconditional training')
    args = parser.parse_args()
    main(args)
    
    