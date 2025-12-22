# Diffusion Model Toy Project (MNIST)

This is a toy project that trains a diffusion model using the MNIST dataset. It implements core diffusion model techniques such as DDPM, DDIM, and Classifier-Free Guidance.

Beyond a simple implementation, this project has been improved for scalability and maintainability, featuring configuration file-based management, support for various model architectures (UNet, DiT), code formatting, and unit testing.

## ✨ Features

- **Multiple Model Architectures**: Supports both traditional `UNet` and modern `Diffusion Transformer (DiT)` models, easily switchable via command-line arguments.
- **Config-Based Management**: Manages model parameters and training settings via `config.yaml`, allowing for various experiments without code changes.
- **Apple Silicon (MPS) Support**: Automatically detects and accelerates using the GPU (MPS) on Apple M-series chips.
- **Testing & Documentation**: Code stability and readability are enhanced through unit tests using `pytest` and in-code `docstrings`.
- **Scripted Execution**: `scripts/train.sh` allows for easy execution of UNet and DiT model training.

## ⚙️ Setup

1.  **Clone Repository**
    ```bash
    git clone https://github.com/your-username/MyDiffusion.git
    cd MyDiffusion
    ```

2.  **Install Dependencies**
    Install the required libraries listed in `requirements.txt`.
    ```bash
    pip install -r requirements.txt
    ```

## 🚀 Training

Use the `src/train.py` script to train the model. You can select `UNet` or `DiT` via the `--model-type` argument. Detailed model structures are defined in `config.yaml`.

**Example Commands:**

- **Train UNet Model**
  ```bash
  python3 src/train.py --model-type UNet --epochs 30 --batch-size 16 --lr 0.0001
  ```

- **Train DiT Model**
  ```bash
  python3 src/train.py --model-type DiT --epochs 50 --batch-size 4 --lr 0.0002
  ```

### Using Shell Script

The `scripts/train.sh` file contains various training examples. The batch size is optimized for an 8GB RAM environment.

```bash
# Grant execution permission
chmod +x scripts/train.sh

# Run script (Train UNet and DiT sequentially)
./scripts/train.sh
```

Trained models (`*.pt`) and loss history (`history.pt`) are saved in the project root directory.

## 🎨 Sampling

To generate new images using a trained model, use `src/sampling.py`. You must specify the model type and weight file path via `--model-type` and `--model-path` arguments.

**Example Command:**

```bash
# Sample with trained UNet model
python3 src/sampling.py --model-type UNet --model-path "UNet_T1000_E30.pt"
```

## 📊 Result

- DDIM with `len(τ) = 10`
- `w` = 1 (CFG sampling ratio)

![result1](./assets/result.png)
![result2](./assets/result2.png)
![result3](./assets/result3.png)

### Training Loss
![loss](./assets/loss.png)

## 📚 Reference Papers

- [Denoising Diffusion Probabilistic Model](https://arxiv.org/abs/2006.11239)
- [Denoising Diffusion Implicit Models](https://arxiv.org/abs/2010.02502)
- [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598)
- [Scalable Diffusion Models with Transformers (DiT)](https://arxiv.org/abs/2212.09748)
- [High-Resolution Image Synthesis with Latent Diffusion Models](https://arxiv.org/abs/2112.10752)