"""
ShadowNet – Training script (local GPU/CPU).

Dataset structure:
    data/
    ├── train/
    │   ├── masks/      ← 256×256 grayscale PNGs
    │   └── shadows/    ← 256×256 grayscale PNGs
    └── val/
        ├── masks/
        └── shadows/

Usage:
    python train.py --data_dir data/
    python train.py --data_dir data/ --epochs 200 --batch_size 16 --lr 1e-3
    python train.py --data_dir data/ --resume model/shadow_unet_best.pth
"""

import argparse
import os
import glob

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import albumentations as A
from PIL import Image
import matplotlib

from model import UNet


# ---------- DATASET ----------

class ShadowDataset(Dataset):
    def __init__(self, mask_dir, shadow_dir, augment=False):
        self.mask_paths = sorted(glob.glob(os.path.join(mask_dir, "*.png")))
        self.shadow_paths = sorted(glob.glob(os.path.join(shadow_dir, "*.png")))
        assert len(self.mask_paths) == len(self.shadow_paths), (
            f"Mismatch: {len(self.mask_paths)} masks vs {len(self.shadow_paths)} shadows"
        )
        self.augment = augment

        self.aug = A.Compose(
            [
                A.HorizontalFlip(p=0.5),
                A.ShiftScaleRotate(
                    shift_limit=0.08,
                    scale_limit=0.12,
                    rotate_limit=10,
                    border_mode=0,
                    value=0,
                    p=0.5,
                ),
                A.RandomBrightnessContrast(
                    brightness_limit=0.1, contrast_limit=0.1, p=0.3
                ),
            ],
            additional_targets={"shadow": "image"},
        )

    def __len__(self):
        return len(self.mask_paths)

    def __getitem__(self, idx):
        mask = np.array(Image.open(self.mask_paths[idx]).convert("L"))
        shadow = np.array(Image.open(self.shadow_paths[idx]).convert("L"))

        if self.augment:
            result = self.aug(image=mask, shadow=shadow)
            mask = result["image"]
            shadow = result["shadow"]

        mask = torch.from_numpy(mask).float().unsqueeze(0) / 255.0
        shadow = torch.from_numpy(shadow).float().unsqueeze(0) / 255.0
        return mask, shadow


# ---------- LOSS ----------

def tv_loss(pred):
    """Total Variation loss – encourages smooth gradients."""
    dy = torch.abs(pred[:, :, 1:, :] - pred[:, :, :-1, :])
    dx = torch.abs(pred[:, :, :, 1:] - pred[:, :, :, :-1])
    return (dy.mean() + dx.mean()) * 0.5


# ---------- TRAINING ----------

def train(args):
    # Use non-interactive backend when --no_preview
    if args.no_preview:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Device: {device}")

    # Data
    train_ds = ShadowDataset(
        os.path.join(args.data_dir, "train", "masks"),
        os.path.join(args.data_dir, "train", "shadows"),
        augment=True,
    )
    val_ds = ShadowDataset(
        os.path.join(args.data_dir, "val", "masks"),
        os.path.join(args.data_dir, "val", "shadows"),
        augment=False,
    )
    print(f"Train: {len(train_ds)} | Val: {len(val_ds)}")

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size,
        num_workers=args.workers, pin_memory=True,
    )

    # Model
    model = UNet().to(device)
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device, weights_only=True)
        model.load_state_dict(ckpt)
        print(f"Resumed from: {args.resume}")

    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs, eta_min=1e-6)

    l1_fn = nn.L1Loss(reduction="none")
    bce_fn = nn.BCELoss(reduction="none")

    os.makedirs(args.save_dir, exist_ok=True)
    best_val_loss = float("inf")

    for epoch in range(args.epochs):
        # --- Train ---
        model.train()
        train_loss = 0.0

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            pred = model(x).clamp(1e-4, 1 - 1e-4)

            l1_map = l1_fn(pred, y)
            bce_map = bce_fn(pred, y)
            weight = 1.0 + args.shadow_boost * y

            pixel_loss = (l1_map * weight).mean() * 0.6 + (bce_map * weight).mean() * 0.4
            smooth_loss = tv_loss(pred)
            loss = pixel_loss + args.tv_weight * smooth_loss

            opt.zero_grad()
            loss.backward()
            opt.step()
            train_loss += loss.item()

        scheduler.step()
        avg_train = train_loss / len(train_loader)

        # --- Validate ---
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                pred = model(x).clamp(1e-4, 1 - 1e-4)

                l1_map = l1_fn(pred, y)
                bce_map = bce_fn(pred, y)
                weight = 1.0 + args.shadow_boost * y

                pixel_loss = (l1_map * weight).mean() * 0.6 + (bce_map * weight).mean() * 0.4
                val_loss += (pixel_loss + args.tv_weight * tv_loss(pred)).item()

        avg_val = val_loss / len(val_loader)

        # Save best
        if avg_val < best_val_loss:
            best_val_loss = avg_val
            best_path = os.path.join(args.save_dir, "shadow_unet_best.pth")
            torch.save(model.state_dict(), best_path)

        lr_now = scheduler.get_last_lr()[0]
        if epoch % 5 == 0 or epoch == args.epochs - 1:
            print(f"Epoch {epoch:3d}/{args.epochs}  train={avg_train:.4f}  val={avg_val:.4f}  lr={lr_now:.2e}")

        # Preview
        if not args.no_preview and (epoch % 15 == 0 or epoch == args.epochs - 1):
            x_val, y_val = next(iter(val_loader))
            with torch.no_grad():
                p = model(x_val.to(device)).cpu()

            fig, axes = plt.subplots(1, 3, figsize=(12, 4))
            axes[0].imshow(x_val[0, 0], cmap="gray")
            axes[0].set_title("Mask")
            axes[1].imshow(p[0, 0], cmap="gray", vmin=0, vmax=1)
            axes[1].set_title("Predicted")
            axes[2].imshow(y_val[0, 0], cmap="gray")
            axes[2].set_title("Ground Truth")
            for ax in axes:
                ax.axis("off")
            plt.tight_layout()
            plt.show()

    # Save final
    final_path = os.path.join(args.save_dir, "shadow_unet_final.pth")
    torch.save(model.state_dict(), final_path)
    print(f"\nBest val loss: {best_val_loss:.4f}")
    print(f"Best model:  {os.path.join(args.save_dir, 'shadow_unet_best.pth')}")
    print(f"Final model: {final_path}")


def main():
    parser = argparse.ArgumentParser(description="ShadowNet – train U-Net for shadow generation")
    parser.add_argument("--data_dir", required=True, help="Root dataset directory (with train/ and val/ subdirs)")
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--shadow_boost", type=float, default=4.0, help="Weight multiplier for shadow pixels")
    parser.add_argument("--tv_weight", type=float, default=0.1, help="Total variation smoothness weight")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--save_dir", default="model", help="Directory to save checkpoints")
    parser.add_argument("--resume", default=None, help="Path to checkpoint to resume from")
    parser.add_argument("--device", default=None, help="Force device (cuda/cpu)")
    parser.add_argument("--no_preview", action="store_true", help="Disable matplotlib preview during training")
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
