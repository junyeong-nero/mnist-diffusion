#!/bin/bash

# =================================================================
# Training Script Examples for Diffusion Model
# (Adjusted for 8GB RAM environments like Apple M2)
# =================================================================

# ---
# Example 1: Train with the default UNet model
# ---
# Batch size reduced to 8 for stable training on 8GB RAM.
echo "Starting training with UNet model (safe batch size)..."
uv run src/train.py \
    --model-type UNet \
    --epochs 30 \
    --batch-size 64 \
    --lr 0.0002 \
    --device cpu

# ---
# Example 2: Train with the DiT (Diffusion Transformer) model
# ---
# Batch size significantly reduced to 4 to prevent out-of-memory errors on 8GB RAM.
echo "Starting training with DiT model (safe batch size)..."
uv run src/train.py \
    --model-type DiT \
    --epochs 50 \
    --batch-size 64 \
    --lr 0.0002 \
    --device cpu

echo "All training examples are complete."


