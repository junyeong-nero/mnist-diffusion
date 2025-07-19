#!/bin/bash

# =================================================================
# Training Script Examples for Diffusion Model
# =================================================================

# ---
# Example 1: Train with the default UNet model
# ---
# This command trains the model using the UNet architecture with default parameters.
echo "Starting training with UNet model..."
python3 train.py \
    --model-type UNet \
    --epochs 30 \
    --batch-size 16 \
    --lr 0.0001

# ---
# Example 2: Train with the DiT (Diffusion Transformer) model
# ---
# This command trains the model using the DiT architecture.
# DiT-specific hyperparameters like hidden-size, depth, and num-heads can be adjusted.
echo "Starting training with DiT model..."
python3 train.py \
    --model-type DiT \
    --epochs 50 \
    --batch-size 32 \
    --lr 0.0002 \
    --hidden-size 256 \
    --depth 6 \
    --num-heads 8

echo "All training examples are complete."

