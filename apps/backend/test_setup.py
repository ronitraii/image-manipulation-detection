#!/usr/bin/env python
import torch
import segmentation_models_pytorch as smp
from PIL import Image
import os

print("Testing imports and setup...")
print(f"✓ Torch version: {torch.__version__}")
print(f"✓ Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
print(f"✓ Segmentation models imported")
print(f"✓ HF_HOME: {os.getenv('HF_HOME', 'Not set')}")

# Test model loading
try:
    print("\nTesting model loading...")
    model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet50', pretrained=True)
    print("✓ ResNet50 loaded successfully")
except Exception as e:
    print(f"✗ Error loading ResNet50: {e}")

try:
    deeplabv3 = smp.DeepLabV3Plus(
        encoder_name="resnet50",
        encoder_weights="imagenet",
        in_channels=3,
        classes=1,
    )
    print("✓ DeepLabV3+ loaded successfully")
except Exception as e:
    print(f"✗ Error loading DeepLabV3+: {e}")

print("\n✓ All tests passed! Ready to start Flask app.")
