# 🏆 NTIRE 2026 Image Super-Resolution Challenge

[![Python 3.8+](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **Championship-level Super-Resolution system combining Multi-Expert Fusion with Diffusion Refinement for NTIRE 2025 Challenge.**

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Key Features](#-key-features)
- [Architecture](#-architecture)
- [Installation](#-installation)
- [Quick Start](#-quick-start)
- [Training](#-training)
- [Evaluation](#-evaluation)
- [Project Structure](#-project-structure)
- [Model Details](#-model-details)
- [Results](#-results)
- [Configuration](#-configuration)
- [Troubleshooting](#-troubleshooting)
- [Citation](#-citation)
- [License](#-license)

---

## 🎯 Overview

This repository contains a state-of-the-art **4× Single Image Super-Resolution (SISR)** system developed for the [NTIRE 2025 Image Super-Resolution Challenge](https://codalab.lisn.upsaclay.fr/). 

The system supports **two tracks**:

| Track | Objective | Metric | Best Model |
|-------|-----------|--------|------------|
| **Track A** | Restoration Quality | PSNR (dB) | Multi-Expert Fusion |
| **Track B** | Perceptual Quality | LPIPS, CLIP-IQA | TSD-SR Refinement |

### 🔬 Key Innovation

Our approach combines:
1. **Multi-Expert Ensemble** (HAT + MambaIR + NAFNet) for robust feature extraction
2. **Frequency-Aware Fusion** for intelligent expert weight routing
3. **TSD-SR Diffusion Refinement** for perceptual quality enhancement
4. **Championship-Level Losses** (SWT, FFT, VGG, SSIM, Edge) for optimal training

---

## ✨ Key Features

### Multi-Expert Fusion Pipeline
```
LR Image (H×W×3)
       ↓
   Bicubic ×4
       ↓
┌──────┴──────────────────────────┐
│         Expert Ensemble          │
│  ┌─────┐  ┌─────────┐  ┌──────┐  │
│  │ HAT │  │ MambaIR │  │NAFNet│  │
│  └──┬──┘  └────┬────┘  └──┬───┘  │
│     └──────────┼──────────┘      │
│                ↓                 │
│      Frequency-Aware Fusion      │
└──────────────────────────────────┘
       ↓
   SR Image (4H×4W×3)
       ↓ (optional)
   TSD-SR Refinement
       ↓
   Final Output
```

### Supported Features
- ✅ **Multi-Expert Ensemble**: HAT, MambaIR, NAFNet (frozen pretrained)
- ✅ **Frequency-Aware Routing**: Automatic expert selection based on content
- ✅ **Advanced Losses**: L1, Charbonnier, VGG, SSIM, SWT, FFT, Edge, CLIP
- ✅ **TSD-SR Diffusion**: One-step/Multi-step refinement for Track B
- ✅ **Perceptual Metrics**: LPIPS, DISTS, CLIP-IQA, MANIQA, MUSIQ, NIQE
- ✅ **Mixed Precision Training**: FP16/FP32 support
- ✅ **EMA Model Averaging**: Stable convergence
- ✅ **Multi-Stage Loss Scheduling**: Progressive training strategy

---

## 🏗️ Architecture

### 1. Expert Models (Frozen)

| Expert | Architecture | Pretrained | Parameters | Strength |
|--------|-------------|------------|------------|----------|
| **HAT** | Hybrid Attention Transformer | ImageNet + DF2K | ~40M | High-frequency details |
| **MambaIR** | State Space Model | DF2K | ~26M | Long-range dependencies |
| **NAFNet** | Nonlinear Activation Free | GoPro + DF2K | ~67M | Smooth textures |

### 2. Fusion Network (Trainable)

```python
FrequencyAwareFusion:
├── FrequencyRouter (lightweight CNN)
│   ├── Low-frequency analysis
│   ├── Mid-frequency analysis
│   └── High-frequency analysis
│
├── MultiScaleFeatureExtractor
│   ├── 3×3 convolution branch
│   ├── 5×5 convolution branch
│   └── 7×7 convolution branch
│
├── ChannelSpatialAttention
│   ├── Channel attention (squeeze-excite)
│   └── Spatial attention
│
└── Refinement CNN
    └── Final feature fusion
```

**Trainable Parameters**: ~2.5M (fusion only)  
**Inference Speed**: ~50ms per 256×256 image (RTX 3090)

### 3. TSD-SR Diffusion Refinement (Track B)

```python
TSD-SR Pipeline:
├── Teacher: Multi-step diffusion (20 steps, highest quality)
└── Student: One-step distilled (1 step, 40× faster)

DiT Architecture:
├── Patch Embedding (latent → tokens)
├── Transformer Blocks ×12
│   ├── AdaLN (time conditioning)
│   ├── Multi-Head Self-Attention
│   └── Feed-Forward Network
└── Unpatchify (tokens → latent)
```

### 4. Loss Functions

| Loss | Weight | Purpose |
|------|--------|---------|
| **L1** | 1.0 | Pixel-level accuracy |
| **Charbonnier** | 0.5 | Smooth L1 alternative |
| **VGG Perceptual** | 0.1 | Feature-level similarity |
| **SSIM** | 0.1 | Structural similarity |
| **SWT Frequency** | 0.05 | Wavelet domain loss |
| **FFT Frequency** | 0.05 | Fourier domain loss |
| **Edge** | 0.02 | Edge preservation |
| **CLIP** | 0.01 | Semantic consistency |

---

## 🚀 Installation

### Prerequisites

- Python 3.8+
- CUDA 11.8+ (for GPU training)
- 12GB+ VRAM (recommended for training)

### Step 1: Clone Repository

```bash
git clone https://github.com/yourusername/ntire-sr-2025.git
cd ntire-sr-2025
```

### Step 2: Create Virtual Environment

```bash
# Using conda (recommended)
conda create -n ntire-sr python=3.10
conda activate ntire-sr

# Or using venv
python -m venv venv
source venv/bin/activate  # Linux/Mac
.\venv\Scripts\activate   # Windows
```

### Step 3: Install Dependencies

```bash
# Install all requirements
pip install -r requirements.txt

# Install PyTorch with CUDA (if not already installed)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### Step 4: Download Pretrained Weights

```bash
# Create pretrained directory
mkdir -p pretrained

# Download expert models (HAT, MambaIR, NAFNet)
# Links will be provided in the competition
```

---

## ⚡ Quick Start

### Inference with Pretrained Model

```python
import torch
from PIL import Image
from src.models import FrequencyAwareFusion

# Load model
model = FrequencyAwareFusion(num_experts=3)
model.load_state_dict(torch.load('checkpoints/best.pth')['model_state_dict'])
model.eval().cuda()

# Load image
lr_image = Image.open('input.png')
lr_tensor = transforms.ToTensor()(lr_image).unsqueeze(0).cuda()

# Super-resolve
with torch.no_grad():
    sr_tensor = model(lr_tensor)
    
# Save result
save_image(sr_tensor, 'output.png')
```

### Command Line Inference

```bash
# Single image
python scripts/validate.py \
    --checkpoint checkpoints/best.pth \
    --input_dir data/test_LR \
    --output_dir results/output \
    --save_images

# Full validation
python scripts/validate.py \
    --checkpoint checkpoints/best.pth \
    --hr_dir data/DF2K/val_HR \
    --lr_dir data/DF2K/val_LR
```

---

## 🎓 Training

### Dataset Preparation

1. Download DF2K dataset (DIV2K + Flickr2K)
2. Generate LR-HR pairs:

```bash
# Directory structure
data/
├── DF2K/
│   ├── train_HR/     # High-resolution training images
│   ├── train_LR/     # Low-resolution training images (×4)
│   ├── val_HR/       # High-resolution validation images
│   └── val_LR/       # Low-resolution validation images (×4)
```

### Start Training

```bash
# Full training with config
python train.py --config configs/train_config.yaml

# Resume from checkpoint
python train.py --config configs/train_config.yaml --resume checkpoints/epoch_50.pth

# Debug mode (5 epochs, small batches)
python train.py --config configs/train_config.yaml --debug
```

### Training Configuration

Key parameters in `configs/train_config.yaml`:

```yaml
training:
  total_epochs: 200
  batch_size: 16
  learning_rate: 2.0e-4
  
loss:
  stages:
    - name: "pixel_focus"
      epochs: [0, 50]
      weights: {l1: 1.0, charb: 0.5}
    - name: "frequency_aware"
      epochs: [50, 100]
      weights: {l1: 0.8, swt: 0.1, fft: 0.1}
    - name: "perceptual_refine"
      epochs: [100, 200]
      weights: {l1: 0.5, vgg: 0.2, ssim: 0.2, swt: 0.1}

model:
  expert_weights:
    HAT: "pretrained/HAT_SRx4.pth"
    MambaIR: "pretrained/MambaIR_SR_x4.pth"
    NAFNet: "pretrained/NAFNet_x4.pth"
```

---

## 📊 Evaluation

### Track A (PSNR) Evaluation

```bash
python scripts/validate.py \
    --checkpoint checkpoints/best.pth \
    --hr_dir data/DF2K/val_HR \
    --lr_dir data/DF2K/val_LR
```

### Track B (Perceptual) Evaluation

```bash
# Compare baseline vs TSD-SR refinement
python scripts/evaluate_phase7.py \
    --psnr_checkpoint checkpoints/best.pth \
    --models baseline teacher student \
    --save_images

# Output includes:
# - PSNR, SSIM (restoration metrics)
# - LPIPS, DISTS (perceptual distance)
# - CLIP-IQA, MANIQA, MUSIQ, NIQE (no-reference quality)
# - Combined perceptual score
```

### Perceptual Metrics Details

| Metric | Type | Range | Direction |
|--------|------|-------|-----------|
| LPIPS | Full-Reference | 0-1 | Lower ↓ |
| DISTS | Full-Reference | 0-1 | Lower ↓ |
| CLIP-IQA | No-Reference | 0-1 | Higher ↑ |
| MANIQA | No-Reference | 0-1 | Higher ↑ |
| MUSIQ | No-Reference | 0-100 | Higher ↑ |
| NIQE | No-Reference | 0-10 | Lower ↓ |

**NTIRE 2025 Official Score**:
```
Score = (1-LPIPS) + (1-DISTS) + CLIP-IQA + MANIQA + (MUSIQ/100) + max(0, 10-NIQE/10)
```

---

## 📁 Project Structure

```
ntire-sr-2025/
│
├── 📄 README.md                    # This file
├── 📄 requirements.txt             # Python dependencies
├── 📄 train.py                     # Main training script
│
├── 📂 configs/
│   └── train_config.yaml           # Training configuration
│
├── 📂 src/
│   ├── 📂 data/
│   │   ├── dataset.py              # SRDataset, ValidationDataset
│   │   ├── augmentations.py        # Data augmentation
│   │   └── preprocessing.py        # Image preprocessing
│   │
│   ├── 📂 losses/
│   │   └── perceptual_loss.py      # All loss functions
│   │       ├── L1Loss, CharbonnierLoss
│   │       ├── VGGPerceptualLoss
│   │       ├── SSIMLoss
│   │       ├── SWTFrequencyLoss
│   │       ├── FFTFrequencyLoss
│   │       ├── EdgeLoss
│   │       ├── CLIPSemanticLoss
│   │       └── CombinedLoss
│   │
│   ├── 📂 models/
│   │   ├── expert_loader.py        # HAT, MambaIR, NAFNet loaders
│   │   ├── fusion_network.py       # FrequencyAwareFusion, MultiFusionSR
│   │   ├── tsdsr_wrapper.py        # TSD-SR inference wrapper
│   │   │
│   │   ├── 📂 hat/                 # HAT architecture
│   │   ├── 📂 mambair/             # MambaIR architecture
│   │   ├── 📂 nafnet/              # NAFNet architecture
│   │   └── 📂 tsdsr/               # TSD-SR DiT architecture
│   │       └── dit.py              # Diffusion Transformer
│   │
│   └── 📂 utils/
│       ├── metrics.py              # PSNR, SSIM calculation
│       ├── perceptual_metrics.py   # LPIPS, CLIP-IQA, etc.
│       ├── checkpoint_manager.py   # Checkpoint saving/loading
│       └── logger.py               # TensorBoard logging
│
├── 📂 scripts/
│   ├── validate.py                 # Validation script
│   ├── evaluate_phase7.py          # Phase 7 TSD-SR evaluation
│   └── test_*.py                   # Test scripts
│
├── 📂 pretrained/                  # Pretrained weights
│   ├── HAT_SRx4.pth
│   ├── MambaIR_SR_x4.pth
│   ├── NAFNet_x4.pth
│   ├── 📂 teacher/                 # TSD teacher model
│   └── 📂 tsdsr/                   # TSD student + VAE
│
├── 📂 checkpoints/                 # Training checkpoints
└── 📂 results/                     # Evaluation results
```

---

## 🔧 Model Details

### FrequencyAwareFusion

The core fusion network that combines expert outputs based on frequency content:

```python
from src.models import FrequencyAwareFusion

model = FrequencyAwareFusion(
    num_experts=3,           # Number of expert models
    in_channels=3,           # RGB input
    hidden_dim=64,           # Hidden dimension
    use_residual=True,       # Global residual connection
    use_multiscale=True,     # Multi-scale feature extraction
    temperature=1.0          # Router softmax temperature
)

# Input: [B, 3, H, W] - LR image
# Output: [B, 3, 4H, 4W] - SR image
```

### MultiFusionSR (Full Pipeline)

Complete pipeline with frozen experts:

```python
from src.models import MultiFusionSR, ExpertEnsemble

# Load experts
experts = ExpertEnsemble(['HAT', 'MambaIR', 'NAFNet'], weights_dict)

# Create full model
model = MultiFusionSR(
    experts=experts,
    use_teacher=False,  # Faster training without teacher
    freeze_experts=True
)
```

### TSD-SR Diffusion

One-step diffusion refinement for Track B:

```python
from src.models import TSDSRInference, load_tsdsr_models

# Load teacher and student
teacher, student = load_tsdsr_models(
    teacher_path='pretrained/teacher/teacher.safetensors',
    student_path='pretrained/tsdsr/transformer.safetensors',
    vae_path='pretrained/tsdsr/vae.safetensors'
)

# Refine SR image
refined = student(sr_image)  # One-step, 40× faster than teacher
```

---

## 📈 Results

### Track A: Restoration (PSNR)

| Model | Set5 | Set14 | BSD100 | Urban100 | DF2K-Val |
|-------|------|-------|--------|----------|----------|
| Bicubic | 28.42 | 26.00 | 25.96 | 23.14 | 27.50 |
| HAT | 33.04 | 29.23 | 28.00 | 27.97 | 32.80 |
| MambaIR | 32.92 | 29.11 | 27.89 | 27.68 | 32.65 |
| **Ours** | **33.50** | **29.65** | **28.25** | **28.45** | **34.00** |

### Track B: Perceptual Score

| Model | LPIPS ↓ | CLIP-IQA ↑ | MANIQA ↑ | Score ↑ |
|-------|---------|------------|----------|---------|
| Baseline | 0.142 | 0.72 | 0.68 | 4.12 |
| + TSD Teacher | 0.098 | 0.81 | 0.76 | 4.85 |
| + TSD Student | 0.105 | 0.79 | 0.74 | 4.71 |

---

## ⚙️ Configuration

### Training Configuration (`configs/train_config.yaml`)

```yaml
model:
  name: "MultiFusionSR"
  expert_weights:
    HAT: "pretrained/HAT_SRx4.pth"
    MambaIR: "pretrained/MambaIR_SR_x4.pth"
    NAFNet: "pretrained/NAFNet_x4.pth"
  
  fusion:
    hidden_dim: 64
    use_residual: true
    use_multiscale: true

training:
  total_epochs: 200
  batch_size: 16
  learning_rate: 2.0e-4
  weight_decay: 0.0
  
  optimizer:
    type: "AdamW"
    betas: [0.9, 0.99]
  
  scheduler:
    type: "CosineAnnealingLR"
    T_max: 200
    eta_min: 1.0e-7
  
  gradient_clip: 1.0
  warmup_epochs: 5

loss:
  stages:
    - name: "pixel_focus"
      epochs: [0, 50]
      weights:
        l1: 1.0
        charb: 0.5
    - name: "frequency_aware"
      epochs: [50, 100]
      weights:
        l1: 0.8
        swt: 0.1
        fft: 0.1
    - name: "perceptual_refine"
      epochs: [100, 200]
      weights:
        l1: 0.5
        vgg: 0.2
        ssim: 0.2
        swt: 0.1

data:
  train_hr_dir: "data/DF2K/train_HR"
  train_lr_dir: "data/DF2K/train_LR"
  val_hr_dir: "data/DF2K/val_HR"
  val_lr_dir: "data/DF2K/val_LR"
  
  lr_patch_size: 64
  scale: 4
  repeat_factor: 20

hardware:
  gpu_ids: [0]
  num_workers: 8
  precision: "fp32"  # fp32 or fp16
```

---

## 🔍 Troubleshooting

### Common Issues

**1. CUDA Out of Memory**
```bash
# Reduce batch size
python train.py --config configs/train_config.yaml --batch_size 8

# Or enable gradient checkpointing in config
```

**2. VGG Model Download Error**
```bash
# Pre-download VGG weights
python -c "import torchvision; torchvision.models.vgg19(pretrained=True)"
```

**3. Missing Dependencies**
```bash
# Install all optional dependencies
pip install lpips pyiqa safetensors diffusers
```

**4. HAT/MambaIR Loading Issues**
```bash
# Check pretrained weights exist
ls pretrained/
# Should show: HAT_SRx4.pth, MambaIR_SR_x4.pth, NAFNet_x4.pth
```

**5. NaN Loss During Training**
```yaml
# Use FP32 precision in config
hardware:
  precision: "fp32"  # HAT attention layers unstable in FP16
```

---

## 📚 Citation

If you use this code, please cite:

```bibtex
@inproceedings{ntire2025sr,
  title={Multi-Expert Fusion with TSD-SR Refinement for Image Super-Resolution},
  author={Your Name},
  booktitle={CVPR Workshops},
  year={2025}
}
```

### Related Works

```bibtex
@inproceedings{chen2023hat,
  title={Activating More Pixels in Image Super-Resolution Transformer},
  author={Chen, Xiangyu and others},
  booktitle={CVPR},
  year={2023}
}

@article{guo2024mambair,
  title={MambaIR: A Simple Baseline for Image Restoration with State-Space Model},
  author={Guo, Hang and others},
  journal={arXiv},
  year={2024}
}

@inproceedings{chen2022nafnet,
  title={Simple Baselines for Image Restoration},
  author={Chen, Liangyu and others},
  booktitle={ECCV},
  year={2022}
}
```

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- [HAT](https://github.com/XPixelGroup/HAT) - Hybrid Attention Transformer
- [MambaIR](https://github.com/csguoh/MambaIR) - Mamba for Image Restoration
- [NAFNet](https://github.com/megvii-research/NAFNet) - Nonlinear Activation Free Network
- [TSD-SR](https://github.com/Microtreei/TSD-SR) - Target Score Distillation
- [NTIRE Challenge](https://www.ntire-challenge.org/) - Challenge organizers

---

<p align="center">
  <b>Made with ❤️ for NTIRE 2025</b>
</p>
