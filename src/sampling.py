import torch
import argparse
import yaml

from MyDiffusion.diffusion import Diffusion
from MyDiffusion.utils import print_digits
from MyDiffusion.modules.UNet import UNet
from MyDiffusion.modules.DiT import DiT


def main(args):
    # Load config
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    # Set device
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
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

    # Initialize the diffusion pipeline
    diffusion_pipeline = Diffusion(
        model=model,
        n_timesteps=general_config["time_steps"],
        device=device,
    )
    diffusion_pipeline.load(args.model_path)
    print(f"Loaded model from {args.model_path}")

    # Prepare for sampling
    test_noise = torch.randn((1, 1, 32, 32)).to(device)
    test_noise = test_noise.repeat(10, 1, 1, 1)
    condition = torch.tensor(list(range(10))).to(device)

    # DDIM Sampling Example
    print("Running DDIM sampling with different step counts...")
    for steps in [10, 20, 50, 100, 200]:
        print(f"Sampling with {steps} steps...")
        test_denoised_image = diffusion_pipeline.decoder.DDIM_sampling(
            test_noise,
            general_config["time_steps"],
            c=condition,
            w=1,
            sampling_steps=steps,
        )
        print_digits(test_denoised_image)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sample from a trained Diffusion Model")
    parser.add_argument(
        "--config", type=str, default="config.yaml", help="Path to the config file"
    )
    parser.add_argument(
        "--model-type",
        type=str,
        required=True,
        choices=["UNet", "DiT"],
        help="Type of model to sample from",
    )
    parser.add_argument(
        "--model-path", type=str, required=True, help="Path to the trained model checkpoint (.pt)"
    )
    args = parser.parse_args()
    main(args)
