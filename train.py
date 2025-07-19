import torch
import argparse

from MyDiffusion.diffusion import Diffusion
from MyDiffusion.utils import print_seq
from MyDiffusion.modules.UNet import UNet
from MyDiffusion.modules.DiT import DiT

from torchvision.datasets import MNIST
import torchvision.transforms as transforms


def data_prepare():
    transform = transforms.Compose(
        [
            transforms.Resize(32),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )

    train = MNIST(root="./data", train=True, download=True, transform=transform)
    test = MNIST(root="./data", train=False, download=True, transform=transform)

    return train, test


def main(args):
    train, test = data_prepare()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create the model based on the specified type
    if args.model_type == "UNet":
        model = UNet(
            in_channels=1,
            out_channels=1,
            n_steps=args.time_steps,
            custom_channel_scale=[128, 128, 256, 256, 512, 512],
        )
    elif args.model_type == "DiT":
        model = DiT(
            img_size=32,
            patch_size=4,
            in_channels=1,
            hidden_size=args.hidden_size,
            depth=args.depth,
            num_heads=args.num_heads,
            n_steps=args.time_steps,
        )
    else:
        raise ValueError(f"Unknown model type: {args.model_type}")

    # Initialize the diffusion pipeline with the chosen model
    diffusion_pipeline = Diffusion(
        model=model,
        n_timesteps=args.time_steps,
        train_set=train,
        test_set=test,
        train_batch_size=args.batch_size,
        test_batch_size=8,
        device=device,
        learning_rate=args.lr,
    )

    # model.load('/content/drive/My Drive/models/DDPM_MNIST/MNIST_T1000_E30_S.pt')

    history = diffusion_pipeline.train(n_epoch=args.epochs, p_uncond=args.p_uncond)

    print_seq(history)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Diffusion Model on MNIST")

    # General arguments
    parser.add_argument(
        "--model-type",
        type=str,
        default="UNet",
        choices=["UNet", "DiT"],
        help="Type of model to use",
    )
    parser.add_argument(
        "--time-steps",
        type=int,
        default=1000,
        help="Number of time steps for diffusion",
    )
    parser.add_argument(
        "--batch-size", type=int, default=16, help="Input batch size for training"
    )
    parser.add_argument(
        "--epochs", type=int, default=30, help="Number of epochs to train"
    )
    parser.add_argument("--lr", type=float, default=0.0001, help="Learning rate")
    parser.add_argument(
        "--p-uncond",
        type=float,
        default=0.1,
        help="Probability of unconditional training",
    )

    # DiT specific arguments
    parser.add_argument(
        "--hidden-size", type=int, default=256, help="Hidden size for DiT model"
    )
    parser.add_argument(
        "--depth", type=int, default=6, help="Depth (number of blocks) for DiT model"
    )
    parser.add_argument(
        "--num-heads",
        type=int,
        default=8,
        help="Number of attention heads for DiT model",
    )

    args = parser.parse_args()
    main(args)
