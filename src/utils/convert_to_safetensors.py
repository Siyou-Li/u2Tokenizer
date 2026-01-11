#!/usr/bin/env python3
"""Convert pytorch_model.bin to safetensors format"""

import torch
from safetensors.torch import save_file
import os

# Path to the model
model_dir = "checkpoint/u2Qwen3-4B-Instruct"
bin_path = os.path.join(model_dir, "pytorch_model.bin")
safetensors_path = os.path.join(model_dir, "model.safetensors")

print(f"Loading checkpoint from {bin_path}...")

# Load the checkpoint
checkpoint = torch.load(bin_path, map_location="cpu", weights_only=False)

print(f"Checkpoint keys: {len(checkpoint.keys())}")
print(f"Sample keys: {list(checkpoint.keys())[:5]}")

# Save as safetensors
print(f"Saving as safetensors to {safetensors_path}...")
save_file(checkpoint, safetensors_path)

# Remove the old .bin file
print(f"Removing old .bin file...")
os.remove(bin_path)

print("✅ Conversion complete!")
print(f"✅ Model saved as: {safetensors_path}")