"""
ShadowNet – Inference script.

Usage:
    python inference.py mask.png                        # → mask_shadow.png
    python inference.py mask.png -o output.png          # custom output path
    python inference.py mask.png --weights model.pth    # custom weights
    python inference.py mask.png --show                 # display result
"""


import argparse
from pathlib import Paths
import numpy as np
import torch
from PIL import Image
import torchvision.transforms as T

from model import UNet

TARGET_SIZE = (256, 256)


def load_model(weights_path: str, device: torch.device) -> UNet:
    model = UNet().to(device)
    ckpt = torch.load(weights_path, map_location=device, weights_only=True)
    model.load_state_dict(ckpt)
    model.eval()
    return model


def predict(model: UNet, mask_path: str, device: torch.device) -> Image.Image:
    mask = Image.open(mask_path).convert("L")
    orig_size = mask.size  # (W, H)

    mask_resized = mask.resize(TARGET_SIZE, Image.BILINEAR)
    mask_tensor = T.ToTensor()(mask_resized).unsqueeze(0).to(device)

    with torch.no_grad():
        pred = model(mask_tensor)

    shadow = pred.squeeze().cpu().numpy()
    shadow = (shadow * 255).clip(0, 255).astype(np.uint8)
    shadow_img = Image.fromarray(shadow).resize(orig_size, Image.BILINEAR)
    return shadow_img


def main():
    parser = argparse.ArgumentParser(description="ShadowNet – generate shadow from mask")
    parser.add_argument("input", help="Path to input mask image (PNG)")
    parser.add_argument("-o", "--output", default=None, help="Output path (default: <input>_shadow.png)")
    parser.add_argument("--weights", default="model/shadow_unet_256x256.pth", help="Path to model weights")
    parser.add_argument("--device", default=None, help="Device: cuda / cpu (auto-detected)")
    parser.add_argument("--show", action="store_true", help="Display input and output side by side")
    args = parser.parse_args()

    # Device
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load
    model = load_model(args.weights, device)
    print(f"Loaded weights: {args.weights}")

    # Predict
    shadow_img = predict(model, args.input, device)

    # Output path
    if args.output:
        out_path = args.output
    else:
        p = Path(args.input)
        out_path = str(p.parent / f"{p.stem}_shadow.png")

    shadow_img.save(out_path)
    print(f"Saved: {out_path}")

    # Optional visualization
    if args.show:
        import matplotlib.pyplot as plt

        mask = Image.open(args.input).convert("L")
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        axes[0].imshow(mask, cmap="gray")
        axes[0].set_title("Input Mask")
        axes[1].imshow(shadow_img, cmap="gray")
        axes[1].set_title("Predicted Shadow")
        for ax in axes:
            ax.axis("off")
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    main()