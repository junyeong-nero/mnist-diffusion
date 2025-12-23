import torch
import argparse
import yaml
import matplotlib.pyplot as plt
from pathlib import Path

from MyDiffusion.Diffusion import Diffusion
from MyDiffusion.Utils import print_seq
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


def save_loss_plot(history, model_type, output_dir="assets"):
    """Save training loss plot to file.

    Args:
        history: List of loss values per epoch
        model_type: Type of model (UNet or DiT)
        output_dir: Directory to save the plot
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(10, 6))
    epochs = range(1, len(history) + 1)
    plt.plot(epochs, history, 'b-o', label='Training Loss', linewidth=2, markersize=8)

    plt.title('Epoch vs Loss', fontsize=16, fontweight='bold')
    plt.xlabel('Epoch', fontsize=14)
    plt.ylabel('Loss', fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)

    # Save plot
    output_path = Path(output_dir) / f"{model_type}_loss.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"\nLoss plot saved to: {output_path}")
    print(f"Final loss: {history[-1]:.6f}")
    print(f"Best loss: {min(history):.6f} (Epoch {history.index(min(history)) + 1})")


def main(args):
    # Load config
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    train, test = data_prepare()

    # Set device
    if args.device:
        device = torch.device(args.device)
        print(f"Using device: {device} (manually specified)")
    else:
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
            print(f"Warning: MPS backend may have compatibility issues. Use --device cpu if you encounter errors.")
        else:
            device = torch.device("cpu")
        print(f"Using device: {device}")

    # Create the model based on the specified type from config
    model_config = config["models"][args.model_type]
    general_config = config["general"]

    if args.model_type == "UNet":
        model = UNet(
            in_channels=general_config["in_channels"],
            out_channels=general_config["out_channels"],
            n_steps=general_config["time_steps"],
            custom_channel_scale=model_config["custom_channel_scale"],
        )
    elif args.model_type == "DiT":
        model = DiT(
            img_size=general_config["img_size"],
            patch_size=model_config["patch_size"],
            in_channels=general_config["in_channels"],
            hidden_size=model_config["hidden_size"],
            depth=model_config["depth"],
            num_heads=model_config["num_heads"],
            n_steps=general_config["time_steps"],
        )
    else:
        raise ValueError(f"Unknown model type: {args.model_type}")

    # Initialize the diffusion pipeline with the chosen model
    diffusion_pipeline = Diffusion(
        model=model,
        n_timesteps=general_config["time_steps"],
        train_set=train,
        test_set=test,
        train_batch_size=args.batch_size,
        test_batch_size=8,
        device=device,
        learning_rate=args.lr,
    )

    history = diffusion_pipeline.train(n_epoch=args.epochs, p_uncond=args.p_uncond)

    print_seq(history)

    # Save loss plot
    save_loss_plot(history, args.model_type)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Diffusion Model on MNIST")

    # General arguments
    parser.add_argument(
        "--config", type=str, default="config.yaml", help="Path to the config file"
    )
    parser.add_argument(
        "--model-type",
        type=str,
        default="UNet",
        choices=["UNet", "DiT"],
        help="Type of model to use",
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
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        choices=["cpu", "cuda", "mps"],
        help="Device to use for training (cpu, cuda, or mps). If not specified, automatically selects the best available device.",
    )

    args = parser.parse_args()
    main(args)

