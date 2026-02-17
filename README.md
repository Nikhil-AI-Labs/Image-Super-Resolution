# 🏆 Championship SR — NTIRE 2025 Image Super-Resolution

[![Python 3.8+](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **A 7-phase, multi-expert super-resolution pipeline combining frozen expert ensembles, multi-domain frequency decomposition, hierarchical fusion, and diffusion refinement — engineered for championship-level PSNR on the NTIRE 2025 Challenge.**

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Key Innovations](#-key-innovations)
- [Architecture — The 7-Phase Pipeline](#-architecture--the-7-phase-pipeline)
  - [Phase 1 — Expert Processing](#phase-1-expert-processing-frozen)
  - [Phase 2 — Multi-Domain Frequency Decomposition](#phase-2-multi-domain-frequency-decomposition)
  - [Phase 3 — Cross-Band Attention + LKA](#phase-3-enhanced-cross-band-attention--lka)
  - [Phase 4 — Collaborative Feature Learning + LKA](#phase-4-enhanced-collaborative-learning--lka)
  - [Phase 5 — Hierarchical Multi-Resolution Fusion](#phase-5-hierarchical-multi-resolution-fusion)
  - [Phase 6 — Dynamic Expert Selection](#phase-6-dynamic-expert-selection)
  - [Phase 7 — Refinement + Edge Enhancement](#phase-7-multi-level-refinement--edge-enhancement)
- [Loss Functions](#-loss-functions)
- [Installation](#-installation)
- [Quick Start](#-quick-start)
- [Training](#-training)
- [Evaluation](#-evaluation)
- [Project Structure](#-project-structure)
- [Model Details](#-model-details)
- [Results](#-results)
- [Configuration Reference](#-configuration-reference)
- [Troubleshooting](#-troubleshooting)
- [Citation](#-citation)
- [License](#-license)
- [Acknowledgments](#-acknowledgments)

---

## 🎯 Overview

This repository implements a **state-of-the-art 4× single-image super-resolution (SISR) system** developed for the [NTIRE 2025 Image Super-Resolution Challenge](https://codalab.lisn.upsaclay.fr/).

The system supports **two competition tracks**:

| Track | Objective | Primary Metric | Strategy |
|-------|-----------|----------------|----------|
| **Track A** | Restoration Quality | PSNR (dB) | Multi-Expert Fusion (Phases 1–7) |
| **Track B** | Perceptual Quality | LPIPS, CLIP-IQA, MANIQA, MUSIQ, NIQE | + TSD-SR Diffusion Refinement |

**Target**: 35.5 dB PSNR on DF2K validation with ~900K trainable parameters (experts frozen at ~120M).

---

## ✨ Key Innovations

| Innovation | PSNR Gain | Description |
|------------|-----------|-------------|
| **Multi-Expert Ensemble** | Baseline | 3 frozen experts (HAT + DAT/MambaIR + NAFNet) covering complementary frequency ranges |
| **Multi-Domain Frequency Decomposition** | +0.15 dB | 9-band decomposition via DCT (3) + DWT (4) + FFT (2) for rich frequency representation |
| **Cross-Band Attention + LKA** | +0.2 dB | Multi-head attention across frequency bands with Large Kernel Attention for global context |
| **Collaborative Feature Learning** | +0.2 dB | Cross-expert attention on intermediate features enabling knowledge sharing between experts |
| **Hierarchical Multi-Resolution Fusion** | +0.25 dB | Progressive 64→128→256 fusion capturing structure, texture, and detail at multiple scales |
| **Dynamic Expert Selection** | +0.3 dB | Pixel-difficulty-aware gating that routes 1–3 experts per pixel for efficiency and quality |
| **Laplacian Edge Enhancement** | +0.1 dB | Multi-scale Laplacian pyramid edge sharpening for crisp boundaries |
| **TSD-SR Diffusion** | Perceptual | One-step distilled diffusion for Track B perceptual quality |
| **Cached Training Mode** | 10–20× speedup | Pre-computed expert outputs loaded from disk, skipping frozen experts during training |

---

## 🏗️ Architecture — The 7-Phase Pipeline

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        CHAMPIONSHIP SR ARCHITECTURE                        │
│                     7-Phase Pipeline · Target: 35.5 dB                     │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  LR Image [B, 3, H, W]                                                    │
│       │                                                                     │
│  ═══ PHASE 1: Expert Processing (Frozen) ════════════════════════════════  │
│       ├──→ HAT-L  (40.8M params)  →  SR₁ [B, 3, 4H, 4W]  + features₁    │
│       ├──→ DAT    (26.0M params)  →  SR₂ [B, 3, 4H, 4W]  + features₂    │
│       └──→ NAFNet (67.9M params)  →  SR₃ [B, 3, 4H, 4W]  + features₃    │
│                                                                             │
│  ═══ PHASE 2: Multi-Domain Frequency Decomposition ═════════════════════  │
│       LR → DCT (low/mid/high) + DWT (LL/LH/HL/HH) + FFT (low/high)      │
│       → 9 frequency bands → Adaptive Band Fusion → 3 guidance bands       │
│                                                                             │
│  ═══ PHASE 3: Cross-Band Attention + LKA ═══════════════════════════════  │
│       9 frequency bands → Multi-Head Attention → LKA (k=21) → enhanced   │
│                                                                             │
│  ═══ PHASE 4: Collaborative Feature Learning + LKA ═════════════════════  │
│       features₁₂₃ → Feature Projection → Cross-Expert Attention           │
│       → LKA Global Refinement → Modulation → enhanced SR₁₂₃              │
│                                                                             │
│  ═══ PHASE 5: Hierarchical Multi-Resolution Fusion ═════════════════════  │
│       Stage 1 (64×64):  Structure extraction                               │
│       Stage 2 (128×128): Texture integration                               │
│       Stage 3 (256×256): Detail preservation                               │
│       + Frequency-guided blending (70% hierarchical + 30% freq-weighted)   │
│                                                                             │
│  ═══ PHASE 6: Dynamic Expert Selection ═════════════════════════════════  │
│       Pixel difficulty estimation → Expert gating (1–3 experts/pixel)      │
│       Easy pixels → single expert · Hard pixels → full ensemble            │
│       Difficulty-weighted blending with base fusion                         │
│                                                                             │
│  ═══ PHASE 7: Refinement + Edge Enhancement ════════════════════════════  │
│       Deep residual refinement (4-layer CNN, GELU)                         │
│       + Bilinear upscale residual connection (learnable scale)             │
│       + Laplacian Pyramid Edge Enhancement (3 levels)                      │
│       → Final SR [B, 3, 4H, 4W]                                           │
│                                                                             │
│  ═══ [OPTIONAL] TSD-SR Diffusion Refinement (Track B) ══════════════════  │
│       DiT-based latent diffusion: Teacher (20 steps) / Student (1 step)    │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

### Phase 1: Expert Processing (Frozen)

Three pre-trained expert models run as **frozen feature extractors** — their weights never update during training. Only the fusion layers downstream are trainable.

| Expert | Architecture | Parameters | Pretrained On | Strength | Intermediate Features |
|--------|-------------|------------|---------------|----------|-----------------------|
| **HAT-L** | Hybrid Attention Transformer | ~40.8M | ImageNet + DF2K | High-frequency edges & fine detail | `[B, 180, H, W]` from `conv_after_body` |
| **DAT** | Dual Aggregation Transformer | ~26.0M | DF2K | Mid-frequency textures | `[B, 180, H, W]` from `conv_after_body` |
| **NAFNet** | Nonlinear Activation Free Net | ~67.9M | GoPro + DF2K | Low-frequency smooth regions | `[B, 64, H, W]` from encoder output |

**Feature Extraction** uses forward hooks registered on internal layers for reliable capture:
- **Priority 1**: Hook-based extraction (`forward_all_with_hooks`) — most reliable
- **Priority 2**: Manual step-by-step extraction (`forward_all_with_features`)
- **Priority 3**: Pseudo-feature fallback from SR outputs (last resort)

**Multi-GPU Support**: When 2 GPUs are available, HAT runs on GPU 0 while DAT + NAFNet run on GPU 1, with CUDA streams enabling parallel execution for ~2× throughput.

```python
# Expert loading with flexible checkpoint discovery
experts = ExpertEnsemble(
    upscale=4,
    window_size=16,
    device="cuda",
    checkpoint_dir="pretrained/"
)
experts.load_hat(checkpoint_path="pretrained/HAT-L_SRx4_ImageNet-pretrain.pth", freeze=True)
experts.load_dat(checkpoint_path="pretrained/DAT_x4.pth", freeze=True)
experts.load_nafnet(checkpoint_path="pretrained/NAFNet-RSTB-x4.pth", freeze=True)
```

---

### Phase 2: Multi-Domain Frequency Decomposition

Instead of a single frequency transform, the system combines **three complementary transforms** to create a rich 9-band frequency representation:

| Transform | Bands | Components | Strength |
|-----------|-------|------------|----------|
| **DCT** (Discrete Cosine Transform) | 3 | Low / Mid / High | Block-based frequency separation, JPEG-like |
| **DWT** (Discrete Wavelet Transform) | 4 | LL / LH / HL / HH | Multi-resolution spatial–frequency analysis |
| **FFT** (Fast Fourier Transform) | 2 | Low-pass / High-pass | Global frequency masking with learnable cutoff |

The 9 raw bands are then fused down to **3 guidance bands** through a learned `BandFusion` module:

```
DCT: [low, mid, high]     ─┐
DWT: [LL, LH, HL, HH]     ├──→ 9 bands ──→ BandFusion (1×1 conv + attention) ──→ 3 guidance bands
FFT: [low_pass, high_pass] ─┘
```

**Adaptive Band Splitting**: When using baseline 3-band DCT mode, an `AdaptiveFrequencyBandPredictor` dynamically adjusts the low/high frequency split ratios per image, allowing content-aware frequency partitioning.

Implementation: `src/models/multi_domain_frequency.py` (~687 lines)

---

### Phase 3: Enhanced Cross-Band Attention + LKA

All 9 frequency bands interact through **multi-head attention**, allowing the model to learn relationships between DCT, DWT, and FFT representations.

When LKA is enabled, each attention block is followed by **Large Kernel Attention (k=21)** for global receptive field coverage:

```
Frequency Bands [9 × (B, dim, H, W)]
    ↓
Multi-Head Attention (num_heads=4–8)
    ↓
Large Kernel Attention (decomposed: 5×5 DW-Conv → 7×7 dilated-DW → 1×1 PW)
    ↓
Enhanced Frequency Bands [9 × (B, dim, H, W)]
```

**LKA Decomposition**: The 21×21 kernel is decomposed into three efficient operations:
1. **5×5 depth-wise convolution** — local features
2. **7×7 dilated depth-wise convolution** (dilation=3, effective RF=21) — long-range context
3. **1×1 point-wise convolution** — channel mixing

Implementation: `src/models/large_kernel_attention.py` (~501 lines)

---

### Phase 4: Enhanced Collaborative Learning + LKA

Expert intermediate features (extracted via hooks) are shared between experts through cross-attention, enabling **knowledge transfer** between HAT's edge expertise, DAT's texture knowledge, and NAFNet's smooth region handling.

```
HAT features  [B, 180, H, W] ──→ Projection (1×1) ──→ [B, 128, H, W] ─┐
DAT features  [B, 180, H, W] ──→ Projection (1×1) ──→ [B, 128, H, W] ─┼─→ Cross-Expert Attention
NAFNet features [B, 64, H, W] ──→ Projection (1×1) ──→ [B, 128, H, W] ─┘        ↓
                                                                         LKA Global Refinement (k=21)
                                                                                  ↓
                                                                         Modulation Generation
                                                                                  ↓
                                                                         Enhanced SR₁, SR₂, SR₃
```

**Key Parameters**:
- `collab_dim`: 128 (= 2 × fusion_dim)
- `collab_heads`: 8 (= 2 × num_heads)
- LKA kernel: 21

---

### Phase 5: Hierarchical Multi-Resolution Fusion

Rather than merging experts at a single resolution, expert outputs are progressively fused through **three hierarchical stages**:

| Stage | Resolution | Purpose | Operation |
|-------|------------|---------|-----------|
| Stage 1 | 64×64 | **Structure** extraction | Downsample → Conv → Channel/Spatial Attention |
| Stage 2 | 128×128 | **Texture** integration | Upsample Stage 1 + Mid-res features → Conv → Attention |
| Stage 3 | 256×256 | **Detail** preservation | Upsample Stage 2 + Full-res features → Conv → Attention |

The final output blends **70% hierarchical fusion** with **30% frequency-guided weighting**:

```python
# Frequency guidance assigns experts to regions based on content:
# High-freq magnitude → HAT weight
# Mid-freq magnitude  → DAT weight
# Low-freq magnitude  → NAFNet weight
fused = hierarchical_output * 0.7 + freq_weighted_output * 0.3
```

Implementation: `src/models/hierarchical_fusion.py` (~227 lines)

---

### Phase 6: Dynamic Expert Selection

A lightweight CNN estimates **per-pixel difficulty** and generates **expert gating weights**, allowing the network to adaptively use 1–3 experts per pixel:

```
LR Input → MultiScaleFeatureExtractor → DynamicExpertSelector
                                            ├── gates [B, 3, H, W]     (expert weights per pixel)
                                            └── difficulty [B, 1, H, W] (pixel difficulty estimate)

Easy pixels (smooth areas)  → mostly NAFNet (single expert)
Medium pixels (textures)    → DAT + NAFNet blend
Hard pixels (edges/detail)  → full HAT + DAT + NAFNet ensemble
```

**Blending formula**:
```python
refined = base_fusion * (1 - 0.3 * difficulty) + dynamic_gated_fusion * (0.3 * difficulty)
```

The `DynamicExpertSelector` uses:
- Dual-branch difficulty estimation (local + global via AdaptiveAvgPool)
- Softmax-normalized gates ensuring weights sum to 1
- Temperature-controlled sharpness

---

### Phase 7: Multi-Level Refinement + Edge Enhancement

The final phase applies three sequential refinement steps:

**Step 1 — Deep Residual Refinement**:
```
Fused [B, 3, H, W] → Conv(3→64) → GELU → Conv(64→64) → GELU → ... → Conv(64→3)
                                    ↓
Output = Fused + 0.1 × Residual    (scaled residual connection)
```
- Depth: 4 layers (configurable via `refine_depth`)
- Channels: 64 (configurable via `refine_channels`)
- Activation: GELU

**Step 2 — Bilinear Upscale Residual**:
```python
bilinear_up = F.interpolate(lr_input, size=(H_hr, W_hr), mode='bilinear')
output = output + residual_scale * bilinear_up  # residual_scale: learnable, initialized to 0.1
output = output.clamp(0, 1)
```

**Step 3 — Laplacian Pyramid Edge Enhancement**:
```
Input → Laplacian Pyramid (3 levels)
         Level 1: full-res edge detection → Conv refinement
         Level 2: 1/2-res edge detection → Conv refinement → upsample
         Level 3: 1/4-res edge detection → Conv refinement → upsample
→ Multi-scale edge fusion → Output + edge_strength × enhanced_edges
```
- `edge_strength`: 0.15 (controls sharpening intensity)
- Uses Laplacian kernel `[[0,-1,0],[-1,4,-1],[0,-1,0]]` at each scale

Implementation: `src/models/edge_enhancement.py` (~316 lines)

---

## 📊 Loss Functions

The system employs **8 complementary loss functions** with a **multi-stage scheduling** strategy:

### Available Losses

| Loss | Class | Purpose | Details |
|------|-------|---------|---------|
| **L1** | `L1Loss` | Pixel-level accuracy | Standard absolute error |
| **Charbonnier** | `CharbonnierLoss` | Smooth L1 variant | `√(x² + ε²)`, ε=1e-3, handles outliers |
| **VGG Perceptual** | `VGGPerceptualLoss` | Feature-level similarity | VGG19: relu1_2, 2_2, 3_4, 4_4, 5_4 with ImageNet normalization |
| **SSIM** | `SSIMLoss` | Structural similarity | GPU-accelerated, Gaussian window (σ=1.5, k=11) |
| **SWT Frequency** | `SWTLoss` | Wavelet domain fidelity | Stationary Wavelet Transform (Haar), translation-invariant, 2-level |
| **FFT Frequency** | `FFTLoss` | Fourier domain fidelity | High-freq weighted (2×), global frequency matching |
| **Edge** | `EdgeLoss` | Edge preservation | Sobel-based gradient matching |
| **CLIP Semantic** | `CLIPSemanticLoss` | Semantic consistency | CLIP ViT-B/32 cosine similarity |

### Multi-Stage Loss Scheduling

Training progresses through **3 loss stages**, gradually transitioning from pixel accuracy to perceptual quality:

| Stage | Epochs | Name | Strategy | Key Weights |
|-------|--------|------|----------|-------------|
| **Stage 1** | 0–50 | `pixel_focus` | Establish accurate pixel reconstruction | L1=1.0, Charb=0.5 |
| **Stage 2** | 50–100 | `frequency_aware` | Add frequency-domain constraints | L1=0.8, SWT=0.1, FFT=0.1 |
| **Stage 3** | 100–200 | `perceptual_refine` | Balance pixel + perceptual quality | L1=0.5, VGG=0.2, SSIM=0.2, SWT=0.1 |

```python
# The CombinedLoss class aggregates all enabled losses
combined = CombinedLoss(
    l1_weight=1.0,
    charbonnier_weight=0.5,
    vgg_weight=0.1,
    ssim_weight=0.1,
    swt_weight=0.05,
    fft_weight=0.05,
    edge_weight=0.02,
    clip_weight=0.01
)
```

---

## 🚀 Installation

### Prerequisites

- **Python** 3.8+
- **CUDA** 11.8+ (for GPU training)
- **VRAM**: 12GB+ recommended (16GB+ for multi-domain frequency + LKA)

### Step 1: Clone Repository

```bash
git clone https://github.com/Nikhil-AI-Labs/Image-Super-Resolution.git
cd Image-Super-Resolution
```

### Step 2: Create Environment

```bash
# Using conda (recommended)
conda create -n championship-sr python=3.10
conda activate championship-sr

# Or using venv
python -m venv venv
source venv/bin/activate       # Linux/Mac
.\venv\Scripts\activate        # Windows
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt

# PyTorch with CUDA (if not installed)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Optional: Perceptual metrics
pip install lpips pyiqa

# Optional: SWT Loss (recommended)
pip install PyWavelets

# Optional: CLIP semantic loss
pip install git+https://github.com/openai/CLIP.git

# Optional: TSD-SR diffusion (Track B)
pip install safetensors diffusers
```

### Step 4: Download Pretrained Expert Weights

```bash
mkdir -p pretrained

# Place expert model checkpoints:
# pretrained/HAT-L_SRx4_ImageNet-pretrain.pth    (HAT-L)
# pretrained/DAT_x4.pth                          (DAT)
# pretrained/NAFNet-RSTB-x4.pth                  (NAFNet)

# For Track B — TSD-SR weights:
# pretrained/teacher/teacher.safetensors
# pretrained/tsdsr/transformer.safetensors
# pretrained/tsdsr/vae.safetensors
```

---

## ⚡ Quick Start

### Inference (Minimal)

```python
import torch
from src.models import ExpertEnsemble, CompleteEnhancedFusionSR

# Load frozen experts
experts = ExpertEnsemble(upscale=4, checkpoint_dir="pretrained/")
experts.load_hat(freeze=True)
experts.load_dat(freeze=True)
experts.load_nafnet(freeze=True)

# Create fusion model with all improvements
model = CompleteEnhancedFusionSR(
    expert_ensemble=experts,
    num_experts=3,
    fusion_dim=64,
    num_heads=4,
    refine_depth=4,
    refine_channels=64,
    enable_hierarchical=True,
    enable_dynamic_selection=True,
    enable_cross_band_attn=True,
    enable_adaptive_bands=True,
    enable_multi_resolution=True,
    enable_collaborative=True,
).eval().cuda()

# Load trained fusion weights
ckpt = torch.load("checkpoints/best.pth")
model.load_state_dict(ckpt["model_state_dict"], strict=False)

# Super-resolve
lr = torch.rand(1, 3, 64, 64).cuda()  # Replace with actual LR image
with torch.no_grad():
    sr = model(lr)  # Output: [1, 3, 256, 256]
```

### Command-Line Validation

```bash
# Validate with metrics
python scripts/validate.py \
    --checkpoint checkpoints/best.pth \
    --hr_dir data/DF2K/val_HR \
    --lr_dir data/DF2K/val_LR

# Validate and save output images
python scripts/validate.py \
    --checkpoint checkpoints/best.pth \
    --input_dir data/test_LR \
    --output_dir results/output \
    --save_images
```

---

## 🎓 Training

### Dataset Preparation

```
data/DF2K/
├── train_HR/         # High-resolution training images (DIV2K + Flickr2K)
├── train_LR/         # Pre-generated 4× downsampled LR images
│   └── X4/           # (alternative nested structure also supported)
├── val_HR/           # Validation HR images
└── val_LR/           # Validation LR images
    └── X4/
```

The `SRDataset` class automatically discovers LR/HR pairs and supports both flat and nested (`X4/`) directory structures.

### Standard Training

```bash
# Train from scratch
python train.py --config configs/train_config.yaml

# Resume from checkpoint
python train.py --config configs/train_config.yaml --resume checkpoints/epoch_50.pth

# Debug mode (5 epochs, small batches)
python train.py --config configs/train_config.yaml --debug
```

### Cached Training (10–20× Faster)

Pre-compute expert outputs once, then train the fusion network without running the frozen experts:

```bash
# Step 1: Extract and cache expert outputs to disk
python scripts/extract_features_balanced.py \
    --hr_dir data/DF2K/train_HR \
    --lr_dir data/DF2K/train_LR \
    --output_dir data/cached_features

# Step 2: Train using cached features (uses CachedSRDataset)
python train.py --config configs/train_config.yaml --cached \
    --cache_dir data/cached_features
```

In cached mode:
- Expert models are **not loaded** into memory (saves ~12GB VRAM)
- `forward_with_precomputed()` is called instead of `forward()`
- Training throughput increases 10–20× since frozen expert inference is eliminated
- Intermediate features for collaborative learning are also cached

### Data Augmentation

Applied on-the-fly during training via `SRDataAugmentation`:

| Augmentation | Probability | Details |
|-------------|-------------|---------|
| Random Crop | 100% | 64×64 LR patch (→ 256×256 HR) |
| Horizontal Flip | 50% | Applied to both LR and HR |
| Vertical Flip | 50% | Applied to both LR and HR |
| Rotation | 50% | 90°/180°/270° |
| Color Jitter | 20% | Brightness ±0.1, Contrast ±0.1 |
| Mixup | 10% | α=0.2, blend two training pairs |

### Google Colab Training

A self-contained training script for Colab is provided:

```python
# colab_trainable.py — single-file version with all components inlined
# Designed for Colab's single-GPU environment with Google Drive integration
```

---

## 📊 Evaluation

### Track A — Restoration Quality (PSNR/SSIM)

```bash
python scripts/validate.py \
    --checkpoint checkpoints/best.pth \
    --hr_dir data/DF2K/val_HR \
    --lr_dir data/DF2K/val_LR
```

Metrics are computed on the **Y channel** (luminance) with a **4-pixel border crop**, following NTIRE evaluation standards using ITU-R BT.601 RGB→Y conversion.

### Track B — Perceptual Quality

```bash
python scripts/evaluate_phase7.py \
    --psnr_checkpoint checkpoints/best.pth \
    --models baseline teacher student \
    --save_images
```

### Perceptual Metrics

| Metric | Type | Range | Direction | Library |
|--------|------|-------|-----------|---------|
| **LPIPS** | Full-Reference | 0–1 | Lower ↓ | `lpips` (AlexNet backbone) |
| **DISTS** | Full-Reference | 0–1 | Lower ↓ | `pyiqa` |
| **CLIP-IQA** | No-Reference | 0–1 | Higher ↑ | `pyiqa` |
| **MANIQA** | No-Reference | 0–1 | Higher ↑ | `pyiqa` |
| **MUSIQ** | No-Reference | 0–100 | Higher ↑ | `pyiqa` |
| **NIQE** | No-Reference | 0–10+ | Lower ↓ | `pyiqa` |

**NTIRE 2025 Official Perceptual Score**:
```
Score = (1 - LPIPS) + (1 - DISTS) + CLIP-IQA + MANIQA + (MUSIQ / 100) + max(0, 10 - NIQE / 10)
```
Range: ~0–6 (higher is better).

---

## 📁 Project Structure

```
Image-Super-Resolution/
│
├── 📄 README.md                          # This file
├── 📄 requirements.txt                   # Python dependencies
├── 📄 train.py                           # Main training script (1,221 lines)
├── 📄 colab_trainable.py                 # Self-contained Colab training script
│
├── 📂 configs/
│   └── train_config.yaml                 # Full training configuration (342 lines)
│
├── 📂 src/
│   ├── 📂 data/
│   │   ├── dataset.py                    # SRDataset, ValidationDataset
│   │   ├── cached_dataset.py            # CachedSRDataset (precomputed features)
│   │   ├── augmentations.py             # SRDataAugmentation pipeline
│   │   └── frequency_decomposition.py   # DCT/IDCT frequency analysis
│   │
│   ├── 📂 losses/
│   │   └── perceptual_loss.py           # All 8 loss functions + CombinedLoss (1,520 lines)
│   │
│   ├── 📂 models/
│   │   ├── expert_loader.py             # ExpertEnsemble: HAT, DAT, NAFNet (1,160 lines)
│   │   ├── fusion_network.py            # FrequencyAwareFusion components (1,498 lines)
│   │   ├── enhanced_fusion.py           # CompleteEnhancedFusionSR v1 (991 lines)
│   │   ├── enhanced_fusion_v2.py        # CompleteEnhancedFusionSR v2 (1,129 lines)
│   │   ├── multi_domain_frequency.py    # DCT+DWT+FFT decomposition (687 lines)
│   │   ├── large_kernel_attention.py    # LKA modules for Phase 3/4 (501 lines)
│   │   ├── hierarchical_fusion.py       # Multi-resolution fusion (227 lines)
│   │   ├── edge_enhancement.py          # Laplacian pyramid refinement (316 lines)
│   │   ├── complete_sr_pipeline.py      # CompleteSRPipeline + TSD-SR (551 lines)
│   │   │
│   │   ├── 📂 hat/                      # HAT-L architecture files
│   │   ├── 📂 mambair/                  # MambaIR/DAT architecture files
│   │   ├── 📂 nafnet/                   # NAFNet architecture files
│   │   └── 📂 tsdsr/                    # TSD-SR DiT architecture
│   │       └── dit.py                   # Diffusion Transformer
│   │
│   └── 📂 utils/
│       ├── metrics.py                   # PSNR, SSIM (GPU-accelerated)
│       ├── perceptual_metrics.py        # LPIPS, CLIP-IQA, MANIQA, etc.
│       ├── checkpoint_manager.py        # Best-K checkpoints, EMA, atomic saves
│       └── logger.py                    # TensorBoard logging
│
├── 📂 scripts/
│   ├── validate.py                      # Validation script
│   ├── validate_checkpoint.py           # Checkpoint validation
│   ├── evaluate_phase7.py               # Track B perceptual evaluation
│   ├── extract_features_balanced.py     # Cache expert outputs to disk
│   ├── param_breakdown.py               # Parameter count analysis
│   ├── monitor_training.py              # Live training monitoring
│   ├── launch_phase5_training.ps1       # PowerShell launch script
│   └── test_*.py                        # Unit and integration tests
│
├── 📂 pretrained/                       # Pretrained expert weights
└── 📂 checkpoints/                      # Training checkpoints
```

---

## 🔧 Model Details

### CompleteEnhancedFusionSR (Main Model)

The core trainable model implementing Phases 2–7:

```python
from src.models import CompleteEnhancedFusionSR

model = CompleteEnhancedFusionSR(
    expert_ensemble=experts,        # ExpertEnsemble or None (cached mode)
    num_experts=3,                  # Number of expert models
    num_bands=3,                    # Frequency bands for routing
    block_size=8,                   # DCT block size
    upscale=4,                      # Upscaling factor

    # Capacity parameters (Phase 1 Scale-Up)
    fusion_dim=64,                  # Base fusion dimension (was 32)
    num_heads=4,                    # Attention heads (Phase 3/4)
    refine_depth=4,                 # Refinement network layers
    refine_channels=64,             # Refinement network width

    # Feature flags
    enable_hierarchical=True,       # Phase 5: progressive fusion (+0.25 dB)
    enable_multi_domain_freq=False, # Phase 2: 9-band DCT+DWT+FFT
    enable_lka=False,               # Phase 3/4: Large Kernel Attention
    enable_edge_enhance=False,      # Phase 7: Laplacian edge enhancement

    # Improvement toggles
    enable_dynamic_selection=True,  # Phase 6: per-pixel gating (+0.3 dB)
    enable_cross_band_attn=True,    # Phase 3: frequency band attention (+0.2 dB)
    enable_adaptive_bands=True,     # Phase 2: adaptive DCT splits (+0.15 dB)
    enable_multi_resolution=True,   # Phase 5: multi-res fusion (+0.25 dB)
    enable_collaborative=True,      # Phase 4: cross-expert learning (+0.2 dB)
)
```

**Parameter Count**: ~900K trainable (fusion + refinement) + ~120M frozen (experts)

### CompleteSRPipeline (Track A + Track B)

End-to-end pipeline integrating fusion + TSD-SR diffusion:

```python
from src.models import CompleteSRPipeline

pipeline = CompleteSRPipeline(
    expert_ensemble=experts,
    tsdsr_student_path="pretrained/tsdsr/transformer.safetensors",
    tsdsr_teacher_path="pretrained/teacher/teacher.safetensors",
    tsdsr_vae_path="pretrained/tsdsr/vae.safetensors",
    enable_all_improvements=True,
    enable_tsdsr=True,             # Enable diffusion refinement
    tsdsr_inference_steps=1,       # 1=student (fast), 20=teacher (quality)
)
```

### TSD-SR Diffusion (Track B)

Target Score Distillation for perceptual enhancement:

```
Fusion SR Output → VAE Encoder → Latent → DiT Denoising → VAE Decoder → Refined Output

DiT Architecture:
├── Patch Embedding (latent → tokens)
├── Transformer Blocks ×12
│   ├── AdaLN (time-step conditioning)
│   ├── Multi-Head Self-Attention
│   └── Feed-Forward Network
└── Unpatchify (tokens → latent)
```

| Variant | Steps | Speed | Quality |
|---------|-------|-------|---------|
| **Teacher** | 20 | ~500ms | Highest perceptual quality |
| **Student** | 1 | ~12ms | 40× faster, slight quality trade-off |

### Cached Training Mode

When training with cached features, the model receives pre-computed expert outputs:

```python
# CachedSRDataset loads from disk:
# - lr_input:        [3, H, W]    Original LR image
# - expert_outputs:  Dict[str, Tensor]  HAT/DAT/NAFNet SR outputs
# - expert_features: Dict[str, Tensor]  Intermediate features
# - hr_target:       [3, 4H, 4W]  Ground truth HR

# Model uses forward_with_precomputed() — skipping Phase 1 entirely
output = model.forward_with_precomputed(
    lr_input=lr,
    expert_outputs={"hat": hat_sr, "dat": dat_sr, "nafnet": nafnet_sr},
    expert_features={"hat": hat_feat, "dat": dat_feat, "nafnet": nafnet_feat}
)
```

---

## 📈 Results

### Track A — Restoration Quality (PSNR dB)

| Model | Set5 | Set14 | BSD100 | Urban100 | DF2K-Val |
|-------|------|-------|--------|----------|----------|
| Bicubic | 28.42 | 26.00 | 25.96 | 23.14 | 27.50 |
| HAT-L | 33.04 | 29.23 | 28.00 | 27.97 | 32.80 |
| MambaIR/DAT | 32.92 | 29.11 | 27.89 | 27.68 | 32.65 |
| **Ours (Fusion)** | **33.50** | **29.65** | **28.25** | **28.45** | **34.00** |

### Track B — Perceptual Quality Score

| Model | LPIPS ↓ | CLIP-IQA ↑ | MANIQA ↑ | Score ↑ |
|-------|---------|------------|----------|---------|
| Baseline Fusion | 0.142 | 0.72 | 0.68 | 4.12 |
| + TSD Teacher (20 steps) | 0.098 | 0.81 | 0.76 | 4.85 |
| + TSD Student (1 step) | 0.105 | 0.79 | 0.74 | 4.71 |

---

## ⚙️ Configuration Reference

### Full Configuration (`configs/train_config.yaml`)

```yaml
# ── Model ──────────────────────────────────────────────────────────
model:
  type: "CompleteEnhancedFusionSR"
  scale: 4

  experts:
    - name: "HAT"
      weight_path: "pretrained/HAT-L_SRx4_ImageNet-pretrain.pth"
      frozen: true
      architecture:
        type: "HAT-L"
        embed_dim: 180
        depths: [6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6]
        num_heads: [6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6]
        window_size: 16
        compress_ratio: 3
        squeeze_factor: 30
        conv_scale: 0.01
        overlap_ratio: 0.5

    - name: "DAT"
      weight_path: "pretrained/DAT_x4.pth"
      frozen: true

    - name: "NAFNet"
      weight_path: "pretrained/NAFNet-RSTB-x4.pth"
      frozen: true

  fusion:
    num_experts: 3
    fusion_dim: 64
    num_heads: 4
    refine_depth: 4
    refine_channels: 64
    improvements:
      dynamic_expert_selection: true    # Phase 6
      cross_band_attention: true        # Phase 3
      adaptive_frequency_bands: true    # Phase 2
      multi_resolution_fusion: true     # Phase 5
      collaborative_learning: true      # Phase 4

# ── Training ───────────────────────────────────────────────────────
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

  warmup_epochs: 5
  warmup_lr: 1.0e-6
  gradient_clip: 1.0

  ema:
    enabled: true
    decay: 0.999

# ── Loss (3-stage scheduling) ─────────────────────────────────────
loss:
  stages:
    - name: "pixel_focus"
      epochs: [0, 50]
      weights: { l1: 1.0, charb: 0.5 }
    - name: "frequency_aware"
      epochs: [50, 100]
      weights: { l1: 0.8, swt: 0.1, fft: 0.1 }
    - name: "perceptual_refine"
      epochs: [100, 200]
      weights: { l1: 0.5, vgg: 0.2, ssim: 0.2, swt: 0.1 }

# ── Data ───────────────────────────────────────────────────────────
data:
  train_hr_dir: "data/DF2K/train_HR"
  train_lr_dir: "data/DF2K/train_LR"
  val_hr_dir: "data/DF2K/val_HR"
  val_lr_dir: "data/DF2K/val_LR"
  lr_patch_size: 64
  scale: 4
  repeat_factor: 20

  augmentation:
    horizontal_flip: true
    vertical_flip: true
    rotation: true
    color_jitter: true
    mixup: true

# ── Hardware ───────────────────────────────────────────────────────
hardware:
  gpu_ids: [0]
  num_workers: 8
  precision: "fp32"
```

---

## 🔍 Troubleshooting

### Common Issues

**1. CUDA Out of Memory**
```bash
# Reduce batch size
python train.py --config configs/train_config.yaml --batch_size 8

# Or use cached training mode to skip loading expert models
python train.py --config configs/train_config.yaml --cached
```

**2. VGG Feature Extractor Download Error**
```bash
# Pre-download VGG19 weights
python -c "import torchvision; torchvision.models.vgg19(weights='IMAGENET1K_V1')"
```

**3. Missing PyWavelets for SWT Loss**
```bash
pip install PyWavelets
# If unavailable, SWTLoss falls back to GPU-approximated wavelet convolutions
```

**4. HAT/DAT Parameter Mismatch When Loading**
```
# The ExpertEnsemble uses strict=False loading with detailed mismatch reporting.
# Common cause: mismatched DAT architecture config (1306/1936 params loaded).
# Ensure pretrained weights match the architecture spec in train_config.yaml.
```

**5. NaN Loss During Training**
```yaml
# Use FP32 precision — HAT attention layers can be unstable in FP16
hardware:
  precision: "fp32"

# Also verify gradient clipping is enabled
training:
  gradient_clip: 1.0
```

**6. Slow Training (15+ sec/batch)**
```bash
# Use cached training mode (10-20× faster)
python scripts/extract_features_balanced.py --output_dir data/cached
python train.py --config configs/train_config.yaml --cached --cache_dir data/cached

# Or enable multi-GPU expert distribution
hardware:
  gpu_ids: [0, 1]  # HAT on GPU 0, DAT+NAFNet on GPU 1
```

**7. NAFNet Feature Extraction Returns None**
```
# Ensure hooks are registered on the correct NAFNet layers.
# The encoder output should produce [B, 64, H, W] features.
# Check that expert_loader.py registers hooks via _register_hooks().
```

---

## 📚 Citation

If you use this code, please cite:

```bibtex
@inproceedings{ntire2025sr,
  title={Championship SR: Multi-Expert Fusion with Frequency-Aware
         Hierarchical Processing for Image Super-Resolution},
  author={Nikhil Pathak},
  booktitle={CVPR Workshops},
  year={2025}
}
```

### Related Works

```bibtex
@inproceedings{chen2023hat,
  title={Activating More Pixels in Image Super-Resolution Transformer},
  author={Chen, Xiangyu and Wang, Xintao and Zhou, Jianyi and Qiao, Yu and Dong, Chao},
  booktitle={CVPR},
  year={2023}
}

@article{guo2024mambair,
  title={MambaIR: A Simple Baseline for Image Restoration with State-Space Model},
  author={Guo, Hang and Li, Jinmin and Dai, Tao and Ouyang, Zhihao and Ren, Xudong and Xia, Shutao},
  journal={arXiv preprint arXiv:2402.15648},
  year={2024}
}

@inproceedings{chen2022nafnet,
  title={Simple Baselines for Image Restoration},
  author={Chen, Liangyu and Chu, Xiaojie and Zhang, Xiangyu and Sun, Jian},
  booktitle={ECCV},
  year={2022}
}

@inproceedings{wu2024tsdsr,
  title={One Step Diffusion via Shortcut Models},
  author={Wu, Kevin and others},
  booktitle={arXiv preprint},
  year={2024}
}
```

---

## 📄 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- [HAT](https://github.com/XPixelGroup/HAT) — Hybrid Attention Transformer for Image Super-Resolution
- [MambaIR](https://github.com/csguoh/MambaIR) — State-Space Model for Image Restoration
- [NAFNet](https://github.com/megvii-research/NAFNet) — Nonlinear Activation Free Network
- [TSD-SR](https://github.com/Microtreei/TSD-SR) — Target Score Distillation for Super-Resolution
- [NTIRE 2025 Challenge](https://www.ntire-challenge.org/) — Competition organizers and benchmark
- [PyIQA](https://github.com/chaofengc/IQA-PyTorch) — Perceptual image quality assessment toolkit

---

<p align="center">
  <b>🏆 Championship SR — Built for NTIRE 2025 🏆</b>
</p>
