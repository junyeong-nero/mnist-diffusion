#!/bin/bash

# =================================================================
# Inference Script Examples for Diffusion Model
# Generate samples from trained models
# =================================================================

# ---
# Example 1: Sample from a trained UNet model
# ---
# Make sure you have a trained model checkpoint (e.g., checkpoints/UNet_T1000_E30.pt)
echo "Generating samples with UNet model..."
if [ -f "checkpoints/UNet_T1000_E30.pt" ]; then
    uv run src/sampling.py \
        --model-type UNet \
        --model-path "checkpoints/UNet_T1000_E30.pt" \
        --device cpu
else
    echo "Warning: UNet model checkpoint 'checkpoints/UNet_T1000_E30.pt' not found. Skipping UNet sampling."
    echo "Please train the model first using train.sh or specify the correct model path."
fi

echo ""

# ---
# Example 2: Sample from a trained DiT model
# ---
# Make sure you have a trained model checkpoint (e.g., checkpoints/DiT_T1000_E50.pt)
echo "Generating samples with DiT model..."
if [ -f "checkpoints/DiT_T1000_E30.pt" ]; then
    uv run src/sampling.py \
        --model-type DiT \
        --model-path "checkpoints/DiT_T1000_E30.pt" \
        --device cpu
else
    echo "Warning: DiT model checkpoint 'checkpoints/DiT_T1000_E50.pt' not found. Skipping DiT sampling."
    echo "Please train the model first using train.sh or specify the correct model path."
fi

echo ""
echo "Inference complete. Check the generated images in the output directory."
