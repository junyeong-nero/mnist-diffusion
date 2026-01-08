"""
Streamlit GUI Application for MNIST Diffusion Model Visualization.

This web-based GUI allows users to:
- Input a digit (0-9) to generate
- Select model type (UNet or DiT)
- Choose a checkpoint file
- Visualize the denoising process step by step with animation
"""

import streamlit as st
import torch
import yaml
import os

from MyDiffusion.Diffusion import Diffusion
from MyDiffusion.Utils import image_normalize
from MyDiffusion.modules.UNet import UNet
from MyDiffusion.modules.DiT import DiT


@st.cache_resource
def load_diffusion_model(model_type: str, checkpoint_path: str, device: torch.device):
    """Load the diffusion model and return the pipeline."""
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    model_config = config["models"][model_type]
    general_config = config["general"]

    if model_type == "UNet":
        model = UNet(
            in_channels=general_config["in_channels"],
            out_channels=general_config["out_channels"],
            n_steps=general_config["time_steps"],
            custom_channel_scale=model_config["custom_channel_scale"],
        )
    elif model_type == "DiT":
        model = DiT(
            img_size=general_config["img_size"],
            patch_size=model_config["patch_size"],
            in_channels=general_config["in_channels"],
            hidden_size=model_config["hidden_size"],
            depth=model_config["depth"],
            num_heads=model_config["num_heads"],
            n_steps=general_config["time_steps"],
        )

    diffusion_pipeline = Diffusion(
        model=model,
        n_timesteps=general_config["time_steps"],
        device=device,
    )
    diffusion_pipeline.load(checkpoint_path)

    return diffusion_pipeline


def generate_images(diffusion_pipeline, digit: int, sampling_steps: int, w: float):
    """Generate images and return intermediate steps for visualization."""
    noise_data = torch.randn((1, 1, 32, 32)).to(diffusion_pipeline.device)
    condition = torch.tensor([digit]).to(diffusion_pipeline.device)

    history_images = []

    def handler(current_data, prev_data):
        img = current_data[0].cpu().clone()
        history_images.append(img)

    diffusion_pipeline.decoder.DDIM_sampling(
        noise_data,
        diffusion_pipeline.n_timesteps,
        c=condition,
        w=w,
        sampling_steps=sampling_steps,
        handler=handler,
    )

    history_images.insert(0, noise_data[0].cpu().clone())
    history_images = history_images[::-1]

    return history_images


def main():
    st.set_page_config(page_title="MNIST Diffusion Visualizer", layout="wide")

    st.title("MNIST Diffusion Model - Denoising Visualization")
    st.markdown("Generate digits and watch the denoising process in real-time!")

    if "pipeline" not in st.session_state:
        st.session_state.pipeline = None

    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader("Settings")

        model_type = st.selectbox("Model Type", ["UNet", "DiT"])

        checkpoint_dir = "checkpoints"
        checkpoint_files = []
        if os.path.exists(checkpoint_dir):
            checkpoint_files = [f for f in os.listdir(checkpoint_dir) if f.endswith(".pt")]

        if not checkpoint_files:
            st.error("No checkpoint files found in 'checkpoints/' directory")
            return

        default_idx = 0
        if model_type == "UNet":
            for i, f in enumerate(checkpoint_files):
                if "UNet" in f:
                    default_idx = i
                    break
        else:
            for i, f in enumerate(checkpoint_files):
                if "DiT" in f:
                    default_idx = i
                    break

        selected_checkpoint = st.selectbox(
            "Checkpoint",
            checkpoint_files,
            index=default_idx,
        )
        checkpoint_path = os.path.join(checkpoint_dir, selected_checkpoint)

        digit = st.number_input("Digit (0-9)", min_value=0, max_value=9, value=5)

        sampling_steps = st.slider("Sampling Steps", min_value=5, max_value=100, value=20)

        w = st.slider("Guidance Weight (w)", min_value=0.0, max_value=5.0, value=1.0, step=0.1)

        show_all_steps = st.checkbox("Show All Steps at Once", value=False)

        st.divider()

        generate_btn = st.button("Generate & Visualize", type="primary", use_container_width=True)

    with col2:
        if "current_images" in st.session_state:
            st.subheader(f"Denoising Process - Digit: {st.session_state.get('current_digit', '?')}")

            if show_all_steps:
                st.subheader("All Steps")
                images = st.session_state.current_images
                cols = st.columns(5)
                for i, img in enumerate(images):
                    with cols[i % 5]:
                        normalized = image_normalize(img).numpy()
                        st.image(normalized, caption=f"Step {i}", width=100)
            else:
                step = st.slider("Step", 0, len(st.session_state.current_images) - 1, len(st.session_state.current_images) - 1)
                img = st.session_state.current_images[step]
                normalized = image_normalize(img).numpy()
                st.image(
                    normalized,
                    caption=f"Step {step}/{len(st.session_state.current_images) - 1}",
                    width=300,
                )

                if step < len(st.session_state.current_images) - 1:
                    if st.button("Next Step"):
                        st.session_state.step = step + 1
                        st.rerun()
                elif step > 0:
                    if st.button("Previous Step"):
                        st.session_state.step = step - 1
                        st.rerun()

        else:
            st.info("Configure settings and click Generate to start")
            st.markdown("""
            ### How it works:
            1. Select a **model type** (UNet or DiT)
            2. Choose a **checkpoint** file
            3. Enter the **digit** (0-9) you want to generate
            4. Adjust **sampling steps** and **guidance weight**
            5. Click **Generate & Visualize** to watch the magic!
            """)

    if generate_btn:
        with st.spinner("Loading model..."):
            try:
                if torch.backends.mps.is_available():
                    device = torch.device("mps")
                elif torch.cuda.is_available():
                    device = torch.device("cuda")
                else:
                    device = torch.device("cpu")

                pipeline = load_diffusion_model(model_type, checkpoint_path, device)
                st.session_state.pipeline = pipeline

                with st.spinner("Generating images..."):
                    images = generate_images(pipeline, digit, sampling_steps, w)
                    st.session_state.current_images = images
                    st.session_state.current_digit = digit

                st.success("Generation complete!")
                st.rerun()

            except Exception as e:
                st.error(f"Error: {str(e)}")


if __name__ == "__main__":
    main()
