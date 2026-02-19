# ShadowNet

**A U-Net that generates realistic shadows from vehicle masks.**

![Prediction vs Ground Truth](https://github.com/shokhrukhShom/shadowNet/blob/main/demo_images/pred_vs_gt.png?raw=true)

## Problem

Simple shadows (drop shadows, ellipses) can be faked with math, but complex-shaped objects like vehicles cast shadows that are hard to simulate convincingly. ShadowNet learns shadow patterns from real-world data and generates realistic ground shadows from a binary mask — fast enough for batch processing.

## How It Works

ShadowNet is built on the [U-Net](https://arxiv.org/abs/1505.04597) architecture (Ronneberger et al., 2015), an encoder-decoder network originally designed for biomedical image segmentation. U-Net uses a **contracting path** (encoder) to capture context at progressively lower resolutions, and a **symmetric expanding path** (decoder) to reconstruct spatial detail. Skip connections between corresponding encoder and decoder levels preserve fine-grained features that would otherwise be lost during downsampling.

### ShadowNet Architecture

ShadowNet adapts U-Net for image-to-image translation (mask → shadow) rather than segmentation:

```
Input (1×256×256 grayscale mask)
  │
  ├─ Encoder Level 1: DoubleConv(1→32)   + MaxPool     ──── skip ────┐
  ├─ Encoder Level 2: DoubleConv(32→64)  + MaxPool     ──── skip ──┐ │
  ├─ Encoder Level 3: DoubleConv(64→128) + MaxPool     ──── skip ┐ │ │
  ├─ Encoder Level 4: DoubleConv(128→256)+ MaxPool     ── skip ┐ │ │ │
  │                                                             │ │ │ │
  ├─ Bottleneck:      DoubleConv(256→512)                       │ │ │ │
  │                                                             │ │ │ │
  ├─ Decoder Level 1: ConvTranspose(512→256) + concat(skip) ───┘ │ │ │
  ├─ Decoder Level 2: ConvTranspose(256→128) + concat(skip) ─────┘ │ │
  ├─ Decoder Level 3: ConvTranspose(128→64)  + concat(skip) ───────┘ │
  ├─ Decoder Level 4: ConvTranspose(64→32)   + concat(skip) ─────────┘
  │
  └─ Output: Conv1×1(32→1) + Sigmoid
             → (1×256×256 shadow map)
```

Each `DoubleConv` block is two rounds of Conv2d → BatchNorm → ReLU. The model has ~7.8M parameters (~30 MB). Every `DoubleConv` block consists of two rounds of `Conv2d(3×3)` → `BatchNorm2d` → `ReLU`, which helps the network learn smooth shadow gradients rather than hard edges.

The training loss combines weighted L1 + BCE (shadow pixels are boosted 4×) with a Total Variation term for smoothness.

### Training

The model was trained on **688 vehicle mask/shadow pairs** (256×256) with augmentation (flips, shift-scale-rotate, brightness/contrast). It can be retrained on any object type given mask + ground-truth shadow pairs.

## Results

![No Shadow vs ShadowNet](https://github.com/shokhrukhShom/shadowNet/blob/main/demo_images/shadovsNoShadow.png?raw=true)

## Installation

Tested on **Ubuntu 24.04**, **Python 3.12**.

```bash
git clone https://github.com/shokhrukhShom/shadowNet.git
cd shadowNet

python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate.bat

pip install -r requirements.txt
```

Download the [pretrained weights](https://drive.google.com/drive/folders/1b07y3v-zk3RRB1x7DFTjGNhUe2vz_-Ja?usp=drive_link) and place the `.pth` file in the `model/` folder:

```
shadowNet/
├── model/
│   └── shadow_unet_256x256.pth
├── model.py
├── inference.py
├── train.py
└── requirements.txt
```

## Inference

```bash
python inference.py masks/243.png                 # → mask_shadow.png
python inference.py masks/243.png -o output.png   # custom output path
python inference.py masks/243.png --weights model.pth # custom weights
python inference.py masks/243.png --show          # display result
```

Auto-detects GPU if available, falls back to CPU. Force a device with `--device cuda` or `--device cpu`.

## Training

Download the [dataset](https://drive.google.com/drive/folders/1rtZ1A35Lu5D5lueCrbjqQrRAMImNkdT0?usp=drive_link) and organize it as:

```
data/
├── train/
│   ├── masks/
│   └── shadows/
└── val/
    ├── masks/
    └── shadows/
```

Then run:

```bash
python train.py --data_dir data/
python train.py --data_dir data/ --epochs 200 --batch_size 16 --lr 1e-3
python train.py --data_dir data/ --resume model/shadow_unet_best.pth  # fine-tune
```

Checkpoints are saved to `model/` by default. Use `--no_preview` for headless environments.

## What's Next?

Once you have the generated shadow (`mask_shadow.png`), to composite it onto a scene:

1. **Invert** the shadow image (so shadow areas become dark)
2. Apply **Multiply** blend mode over your background
3. **Resize / transform** to match the perspective and scale of your scene

## Links

- [Pretrained Weights (Google Drive)](https://drive.google.com/drive/folders/1b07y3v-zk3RRB1x7DFTjGNhUe2vz_-Ja?usp=drive_link)
- [Dataset (Google Drive)](https://drive.google.com/drive/folders/1rtZ1A35Lu5D5lueCrbjqQrRAMImNkdT0?usp=drive_link)

## License

MIT — free to use, modify, and distribute for any purpose. See [LICENSE](LICENSE) for details.

## Support

If you find this project useful: [Buy Me a Coffee ☕](https://buymeacoffee.com/shokhrukhShom)

Contact: shokhrukh.shom@gmail.com
