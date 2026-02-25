<div align="center">

# 🏆 Championship Super-Resolution

### Multi-Expert Fusion Architecture for NTIRE 2026

[![Python 3.8+](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://python.org)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0%2B-red.svg)](https://pytorch.org)
[![CUDA 11.8+](https://img.shields.io/badge/CUDA-11.8%2B-green.svg)](https://developer.nvidia.com/cuda-toolkit)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**4× Single-Image Super-Resolution** via frozen expert ensembles, frequency-domain analysis, and adaptive fusion

*~1.2M trainable parameters · ~131.7M frozen expert parameters · 7-phase pipeline*

</div>

---

## Table of Contents

- [Overview](#overview)
- [Key Innovations](#key-innovations)
- [Architecture](#architecture)
  - [Phase 1: Expert Processing](#phase-1-expert-processing-frozen)
  - [Phase 2: Multi-Domain Frequency Decomposition](#phase-2-multi-domain-frequency-decomposition)
  - [Phase 3: Cross-Band Attention + LKA](#phase-3-cross-band-attention--lka)
  - [Phase 4: Collaborative Feature Learning + LKA](#phase-4-collaborative-feature-learning--lka)
  - [Phase 5: Hierarchical Multi-Resolution Fusion](#phase-5-hierarchical-multi-resolution-fusion)
  - [Phase 6: Dynamic Expert Selection](#phase-6-dynamic-expert-selection)
  - [Phase 7: Multi-Level Refinement + Edge Enhancement](#phase-7-multi-level-refinement--edge-enhancement)
  - [Track B: TSD-SR Diffusion Refinement](#track-b-tsd-sr-diffusion-refinement)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Training](#training)
  - [Standard Training](#standard-training)
  - [Cached Training (10-20× Faster)](#cached-training-10-20-faster)
  - [Multi-Stage Loss Scheduling](#multi-stage-loss-scheduling)
  - [Training Configuration](#training-configuration)
- [Inference](#inference)
- [Loss Functions](#loss-functions)
- [Data Pipeline](#data-pipeline)
- [Evaluation Metrics](#evaluation-metrics)
- [Model Details](#model-details)
- [Configuration Reference](#configuration-reference)
- [Troubleshooting](#troubleshooting)
- [Citation](#citation)
- [Acknowledgments](#acknowledgments)

---

## Overview

Championship SR is a **7-phase super-resolution pipeline** designed for the NTIRE 2026 Image Super-Resolution Challenge. The core idea is to leverage four powerful pre-trained (frozen) expert SR models and train a lightweight fusion network (~1.2M parameters) that intelligently combines their outputs using frequency-domain analysis, attention mechanisms, and adaptive gating.

### Design Philosophy

Instead of training a single massive model from scratch, we:

1. **Freeze** four state-of-the-art SR models (HAT-L, DRCT-L, GRL-B, EDSR-L) totaling ~131.7M parameters
2. **Train** a lightweight fusion network that learns *when* and *how* to combine each expert's output
3. **Guide** the fusion using multi-domain frequency decomposition (DCT + DWT + FFT)
4. **Adapt** per-pixel expert selection based on local image difficulty

This approach delivers state-of-the-art results with minimal training cost, since only the fusion network's ~1.2M parameters require gradient computation.

---

## Key Innovations

| Innovation | Module | Expected Gain |
|---|---|---|
| **4-Expert Frozen Ensemble** | `expert_loader.py` | Baseline |
| **Multi-Domain Frequency Decomposition** (DCT+DWT+FFT → 9 bands) | `multi_domain_frequency.py` | +0.15 dB |
| **Cross-Band Attention + LKA** (k=21, 9 bands) | `large_kernel_attention.py` | +0.20 dB |
| **Collaborative Feature Learning + LKA** (cross-expert attention) | `enhanced_fusion_v2.py` | +0.20 dB |
| **Hierarchical Multi-Resolution Fusion** (64→128→256) | `hierarchical_fusion.py` | +0.25 dB |
| **Dynamic Expert Selection** (per-pixel difficulty gating) | `enhanced_fusion_v2.py` | +0.30 dB |
| **Laplacian Pyramid Edge Enhancement** | `edge_enhancement.py` | +0.10 dB |
| **3-Stage Loss Scheduling** (L1 → SWT+FFT → SSIM) | `perceptual_loss.py` | — |
| **Cached Training Pipeline** (10-20× speedup) | `cached_dataset.py` | — |

---

## Architecture

The complete pipeline is implemented in `src/models/complete_sr_pipeline.py` (`CompleteSRPipeline`) which wraps the core fusion model `CompleteEnhancedFusionSR` from `src/models/enhanced_fusion_v2.py`.

```
LR Input [B, 3, H, W]
    │
    ├─── Phase 1: Expert Processing (Frozen) ──────────────────────────────┐
    │    ├── HAT-L   → [B, 3, 4H, 4W] + features [B, 180, H, W]         │
    │    ├── DRCT-L  → [B, 3, 4H, 4W] + features [B, 180, H, W]         │
    │    ├── GRL-B   → [B, 3, 4H, 4W] + features [B, 180, H, W]         │
    │    └── EDSR-L  → [B, 3, 4H, 4W] + features [B, 256, H, W]         │
    │                                                                      │
    ├─── Phase 2: Frequency Decomposition (DCT+DWT+FFT → 9 bands) ───────┤
    │                                                                      │
    ├─── Phase 3: Cross-Band Attention + LKA (k=21) ─────────────────────┤
    │                                                                      │
    ├─── Phase 4: Collaborative Feature Learning + LKA ───────────────────┤
    │                                                                      │
    ├─── Phase 5: Hierarchical Multi-Resolution Fusion (64→128→256) ──────┤
    │    └── 70% hierarchical + 30% frequency-guided blending             │
    │                                                                      │
    ├─── Phase 6: Dynamic Expert Selection (per-pixel difficulty) ────────┤
    │    └── Difficulty-adaptive blend weight: 0.3 + 0.4 × difficulty     │
    │                                                                      │
    ├─── Phase 7: Deep CNN Refinement + Laplacian Edge Enhancement ───────┤
    │    └── Residual connection from bilinear upscale of LR input        │
    │                                                                      │
    └─── Output: SR Image [B, 3, 4H, 4W] ────────────────────────────────┘

    (Optional) Track B: TSD-SR Diffusion Refinement for perceptual quality
```

---

### Phase 1: Expert Processing (Frozen)

**File:** `src/models/expert_loader.py` — `ExpertEnsemble` class

Four pre-trained SR models are loaded with frozen weights. Intermediate features are extracted via PyTorch forward hooks for use in Phase 4 (Collaborative Learning).

| Expert | Architecture | Parameters | Embed Dim | Depths | Window |
|--------|-------------|-----------|-----------|--------|--------|
| **HAT-L** | Hybrid Attention Transformer | ~40.85M | 180 | [6]×12 | 16 |
| **DRCT-L** | Dense Residual Connected Transformer | ~27.58M | 180 | [6]×12 | 16 |
| **GRL-B** | Global-Regional-Local Attention | ~20.20M | 180 | [4,4,8,8,4,4] | 8 |
| **EDSR-L** | Enhanced Deep Residual SR | ~43.09M | 256 features | 32 blocks | — |

**Total frozen parameters: ~131.72M**

#### Feature Extraction via Hooks

The `ExpertEnsemble.forward_all_with_hooks()` method:
1. Registers forward hooks on intermediate layers of each expert
2. Runs all experts under `torch.no_grad()` for memory efficiency
3. Clones captured features outside inference mode for autograd compatibility
4. Returns both SR outputs `{name: [B,3,4H,4W]}` and intermediate features `{name: [B,C,H,W]}`

Feature channel dimensions:
- HAT: 180 channels (from transformer body)
- DRCT: 180 channels (from transformer body)
- GRL: 180 channels (from intermediate blocks)
- EDSR: 256 channels (from residual body)

#### Input Handling

The `ExpertEnsemble` handles input padding to ensure dimensions are divisible by each expert's window size. After expert processing, outputs are cropped back to the expected HR dimensions.

---

### Phase 2: Multi-Domain Frequency Decomposition

**File:** `src/models/multi_domain_frequency.py` — `MultiDomainFrequencyDecomposition` class

Decomposes the LR input into **9 frequency bands** using three complementary transforms:

#### DCT (Discrete Cosine Transform) — 3 bands
- Operates on 8×8 blocks (configurable `block_size`)
- Uses learned adaptive thresholds via `AdaptiveThresholdNet` (global average pooling → FC layers)
- Separates into **low**, **mid**, and **high** frequency bands using differentiable sigmoid masking along zigzag-ordered DCT coefficients
- Temperature parameter (τ=50) controls mask sharpness

#### DWT (Discrete Wavelet Transform) — 3 bands
- Uses Haar wavelet decomposition
- Produces **LL** (approximation), **LH+HL** (horizontal+vertical edges), **HH** (diagonal detail) subbands
- Each subband is upsampled back to input resolution

#### FFT (Fast Fourier Transform) — 3 bands
- Computes 2D FFT with spectral shift
- Applies learned radial masks (`FFTMaskNet`) to separate **low**, **mid**, **high** frequency rings
- Masks are generated from global features via adaptive networks
- Inverse FFT reconstructs spatial-domain bands

#### Band Attention
Each of the 9 bands passes through `BandAttention` (channel squeeze-and-excitation with reduction ratio 4) for adaptive recalibration.

**Output:** 9 raw bands `[B, 3, H, W]` each + 3 fused guidance bands (DCT+DWT+FFT fused per frequency tier)

---

### Phase 3: Cross-Band Attention + LKA

**Files:** `src/models/enhanced_fusion_v2.py` (CrossBandAttention), `src/models/large_kernel_attention.py` (LKA modules)

Multi-head attention across the 9 frequency bands, augmented with Large Kernel Attention for global spatial context.

#### Cross-Band Attention
- Projects each band to a `hidden_dim=32` space via 1×1 convolutions
- Applies `nn.MultiheadAttention` (4 heads) across bands at each spatial location
- Learnable `band_gates` (softmax-normalized) weight each band's contribution
- Residual connection: `output = original + gate_weight × attention_output`

#### Large Kernel Attention (LKA, k=21)

The `EnhancedCrossBandWithLKA` module augments cross-band attention with an LKA block that captures global spatial context.

**LKA Kernel Decomposition:** A 21×21 convolution is decomposed into four efficient operations:
```
5×5 Depthwise Conv → 1×21 Depthwise Dilated Conv (d=3)
                    → 21×1 Depthwise Dilated Conv (d=3)
                    → 1×1 Pointwise Conv
```
This decomposition reduces parameters from O(k²) to O(k) while maintaining the 21×21 effective receptive field.

**LKA Block architecture:**
```
Input → LayerNorm → LKA → Residual Add → LayerNorm → FFN (expand×4) → Residual Add → Output
```

The `EnhancedCrossBandWithLKA` processes all 9 bands:
1. Projects bands to `dim=64` channels
2. Stacks into `[B, 9, dim, H, W]` → applies cross-band attention
3. Applies shared LKA block for spatial refinement
4. Projects back to 3 channels per band

---

### Phase 4: Collaborative Feature Learning + LKA

**File:** `src/models/enhanced_fusion_v2.py` — `CollaborativeFeatureLearning` class, wrapped by `EnhancedCollaborativeWithLKA`

Cross-expert attention on intermediate features enables knowledge sharing between experts:

- **HAT** (180ch) shares high-frequency transformer features with DRCT/GRL/EDSR
- **DRCT** (180ch) shares dense-residual features with HAT/GRL/EDSR
- **GRL** (180ch) shares global/regional/local features with HAT/DRCT/EDSR
- **EDSR** (256ch) shares deep convolutional features with HAT/DRCT/GRL

#### Mechanism
1. **Feature Projection:** Each expert's features are projected to `common_dim=128` via 1×1 convolutions
2. **Cross-Expert Attention:** `nn.MultiheadAttention` (8 heads) attends across experts at each spatial location
3. **Consensus Refinement:** Mean of attention outputs → 3×3 conv refinement → consensus features
4. **Output Modulation:** Per-expert modulation maps (sigmoid, 0-1) scale each expert's SR output by `(1 + 0.2 × modulation)`
5. **LKA Enhancement:** Shared LKA block (k=21) adds global spatial context after attention

---

### Phase 5: Hierarchical Multi-Resolution Fusion

**File:** `src/models/hierarchical_fusion.py` — `HierarchicalMultiResolutionFusion` class

Progressive fusion at three resolutions captures structure, texture, and detail at different scales.

#### Three-Stage Progressive Fusion

| Level | Resolution | Focus | Components |
|-------|-----------|-------|------------|
| **Stage 1** | 64×64 | Coarse structure | 4-expert concat → ResBlocks → SpatialGating → 3ch output |
| **Stage 2** | 128×128 | Textures | 4-expert concat + Stage 1 upsampled → ResBlocks → SpatialGating |
| **Stage 3** | 256×256 | Fine details | 4-expert concat + Stage 2 upsampled → ResBlocks → SpatialGating |

Each stage uses:
- **Residual Blocks:** Conv3×3 → GELU → Conv3×3 + skip connection
- **Spatial Gating:** Conv3×3 → GELU → Conv3×3 → Sigmoid (learns where to refine)
- **Progressive residuals:** Each stage refines the previous stage's upsampled output

#### Frequency-Guided Routing

In `CompleteEnhancedFusionSR._run_pipeline()`, the hierarchical output is blended with a frequency-guided fusion:
- `routing_lr` (enhanced by Phase 3 cross-band attention) is upscaled to HR resolution
- A `freq_weight_conv` (1×1 → GELU → 1×1) predicts per-pixel per-expert softmax weights
- **Final blend: 70% hierarchical + 30% frequency-guided**

This creates a gradient path: `loss → freq_weight_conv → routing_lr → cross_band → freq_decomp`

---

### Phase 6: Dynamic Expert Selection

**File:** `src/models/enhanced_fusion_v2.py` — `DynamicExpertSelector` class

Per-pixel difficulty estimation adaptively routes experts:

#### Difficulty Estimation Network
```
LR Input → Conv3×3 → ReLU → Conv3×3 → ReLU → Conv3×3 → Sigmoid → difficulty [B,1,H,W]
```

#### Gate Network
```
LR Input → Conv3×3 → ReLU → Conv3×3 → ReLU → Conv1×1 → raw_gates [B,4,H,W]
```

#### Adaptive Gating Logic
```python
threshold = 0.7 - 0.5 × difficulty           # Lower threshold for harder pixels
gates = sigmoid(temperature × (raw_gates - threshold))  # Learnable temperature (init=10)
gates = gates / (gates.sum(dim=1) + 1e-8)    # Normalized, min gate_sum=0.3
```

**Expert routing strategy:**
- **Easy pixels** (sky, smooth): Most weight on EDSR (fast convolutional baseline)
- **Medium pixels** (texture): DRCT + GRL dominant
- **Hard pixels** (edges, complex detail): HAT + DRCT + GRL all contribute

#### Blending with Phase 5
```python
blend_weight = 0.3 + 0.4 × difficulty_hr  # Range: [0.3, 0.7]
fused = (1 - blend_weight) × phase5_fused + blend_weight × dynamic_fused
```
Harder regions get more dynamic selection influence; easier regions rely more on hierarchical fusion.

---

### Phase 7: Multi-Level Refinement + Edge Enhancement

#### 7a: Deep CNN Refinement

A deep residual refinement network processes the fused output:
```
Fused → Conv3×3 (3→128) → GELU → [Conv3×3 → GELU] × 4 → Conv3×3 (128→3) → × 0.1 → + Fused
```

- **Depth:** 6 convolutional layers (configurable `refine_depth`)
- **Width:** 128 channels (configurable `refine_channels`)
- **Residual scaling:** Output is multiplied by 0.1 before adding back (prevents early-training instability)

#### 7b: Laplacian Pyramid Edge Enhancement

**File:** `src/models/edge_enhancement.py` — `LaplacianPyramidRefinement` class

Multi-scale edge sharpening using a Gaussian blur pyramid:

1. **Gaussian Pyramid:** Input → 3 levels of Gaussian blur (σ=1.0, kernel=5×5)
2. **Laplacian Extraction:** `laplacian[i] = level[i] - blur(level[i])` (captures edges at each scale)
3. **Learned Refinement:** Each Laplacian level passes through a `RefinementBlock`:
   ```
   Laplacian → Conv3×3 (3→32) → ReLU → Conv3×3 (32→32) → ReLU → Conv3×3 (32→3) + Skip
   ```
4. **Multi-scale fusion:** `output = input + edge_strength × Σ(refined_laplacians)` where `edge_strength=0.15`

#### Global Residual Connection

After Phases 7a and 7b, a bilinear upscale of the original LR input is added:
```python
bilinear = F.interpolate(lr_input, size=(H_hr, W_hr), mode='bilinear')
final_sr = (fused + residual_scale × bilinear).clamp(0, 1)  # residual_scale is learnable (init=0.1)
```

---

### Track B: TSD-SR Diffusion Refinement

**Files:** `src/models/tsdsr_wrapper.py`, `src/models/complete_sr_pipeline.py`

For perceptual quality optimization (Track B), the pipeline integrates TSD-SR (Target Score Distillation Super-Resolution):

#### Architecture
- **VAE Encoder/Decoder:** Converts between pixel space and latent space (4 channels, 8× downscale)
- **Student Model:** One-step distilled transformer for fast inference
- **Teacher Model:** Multi-step diffusion for highest quality (optional)

#### Pipeline
1. Phase 1-7 produces a PSNR-optimized SR image
2. VAE encodes SR image to latent space `[B, 4, H/8, W/8]`
3. Student transformer denoises in one step (or teacher in multiple steps)
4. VAE decodes back to pixel space
5. Blended output: `α × PSNR_result + (1-α) × diffusion_result`

**Note:** TSD-SR is disabled during training (`use_during_training: false`) and used only at inference for Track B submissions.

---

## Project Structure

```
super-resolution/
├── train.py                          # Main training script
├── requirements.txt                  # Python dependencies
├── configs/
│   └── train_config.yaml             # Complete training configuration
├── src/
│   ├── models/
│   │   ├── __init__.py               # Module exports (v1 + v2 APIs)
│   │   ├── expert_loader.py          # ExpertEnsemble — 4 frozen experts + hooks
│   │   ├── enhanced_fusion_v2.py     # CompleteEnhancedFusionSR — Phases 2-7
│   │   ├── multi_domain_frequency.py # Phase 2: DCT+DWT+FFT decomposition
│   │   ├── large_kernel_attention.py # LKA modules (k=21 decomposed)
│   │   ├── hierarchical_fusion.py    # Phase 5: 64→128→256 progressive fusion
│   │   ├── edge_enhancement.py       # Phase 7b: Laplacian pyramid refinement
│   │   ├── complete_sr_pipeline.py   # CompleteSRPipeline — all 7 phases + TSD-SR
│   │   ├── fusion_network.py         # V1 fusion components (legacy)
│   │   ├── enhanced_fusion.py        # V1 enhanced fusion (legacy)
│   │   └── tsdsr_wrapper.py          # TSD-SR diffusion refinement
│   ├── data/
│   │   ├── dataset.py                # SRDataset, DF2KDataset, ValidationDataset
│   │   ├── augmentations.py          # Paired augmentations (crop, flip, rotate, color)
│   │   └── cached_dataset.py         # CachedSRDataset — pre-computed expert outputs
│   ├── losses/
│   │   └── perceptual_loss.py        # 8 loss functions + CombinedLoss scheduler
│   ├── training/
│   │   └── multi_stage_scheduler.py  # MultiStageLossScheduler
│   └── utils/
│       ├── metrics.py                # PSNR, SSIM, LPIPS calculators
│       ├── checkpoint_manager.py     # CheckpointManager + EMAModel
│       └── perceptual_metrics.py     # Additional perceptual quality metrics
├── scripts/                          # Utility scripts
│   ├── validate.py                   # Validation evaluation
│   ├── extract_features_balanced.py  # Cache expert outputs for fast training
│   ├── test_inference.py             # Single-image inference
│   └── ...                           # Additional utility scripts
└── pretrained/                       # Pre-trained expert weights (not tracked)
    ├── hat/HAT-L_SRx4_ImageNet-pretrain.pth
    ├── drct/DRCT-L_X4.pth
    ├── grl/GRL-B_SR_x4.pth
    └── edsr/EDSR_Lx4_f256b32_DIV2K_official-76ee1c8f.pth
```

---

## Installation

### Prerequisites
- Python 3.8+
- PyTorch 2.0+ with CUDA 11.8+
- 26GB+ VRAM (single GPU training)

### Setup

```bash
# Clone the repository
git clone https://github.com/Nikhil-AI-Labs/Image-Super-Resolution.git
cd Image-Super-Resolution

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or: venv\Scripts\activate  # Windows

# Install PyTorch with CUDA (adjust for your CUDA version)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Install dependencies
pip install -r requirements.txt
```

### Pre-trained Expert Weights

Download the four expert model weights and place them in the `pretrained/` directory:

| Expert | Source | Weight File |
|--------|--------|-------------|
| HAT-L | [HAT GitHub](https://github.com/XPixelGroup/HAT) | `pretrained/hat/HAT-L_SRx4_ImageNet-pretrain.pth` |
| DRCT-L | [DRCT GitHub](https://github.com/ming053l/DRCT) | `pretrained/drct/DRCT-L_X4.pth` |
| GRL-B | [GRL GitHub](https://github.com/ofsoundof/GRL-Image-Restoration) | `pretrained/grl/GRL-B_SR_x4.pth` |
| EDSR-L | [EDSR/BasicSR](https://github.com/sanghyun-son/EDSR-PyTorch) | `pretrained/edsr/EDSR_Lx4_f256b32_DIV2K_official-76ee1c8f.pth` |

### Dataset Preparation

Prepare the DF2K dataset (DIV2K + Flickr2K) with pre-generated 4× downscaled LR images:

```
dataset/DF2K/
├── train_HR/          # High-resolution training images
├── train_LR/          # Low-resolution training images (4× downscaled)
├── val_HR/            # High-resolution validation images
└── val_LR/            # Low-resolution validation images
```

---

## Quick Start

### Single Image Inference

```python
import torch
from src.models import ExpertEnsemble, create_training_pipeline
from PIL import Image
from torchvision import transforms

# Load expert ensemble
expert_ensemble = ExpertEnsemble(
    hat_weight_path="pretrained/hat/HAT-L_SRx4_ImageNet-pretrain.pth",
    drct_weight_path="pretrained/drct/DRCT-L_X4.pth",
    grl_weight_path="pretrained/grl/GRL-B_SR_x4.pth",
    edsr_weight_path="pretrained/edsr/EDSR_Lx4_f256b32_DIV2K_official-76ee1c8f.pth",
)

# Create the 7-phase pipeline
model = create_training_pipeline(expert_ensemble, device='cuda')

# Load and preprocess image
lr_image = Image.open("input_lr.png").convert("RGB")
lr_tensor = transforms.ToTensor()(lr_image).unsqueeze(0).cuda()

# Super-resolve
with torch.no_grad():
    sr_tensor = model(lr_tensor)

# Save result
sr_image = transforms.ToPILImage()(sr_tensor.squeeze(0).cpu().clamp(0, 1))
sr_image.save("output_sr.png")
```

---

## Training

### Standard Training

```bash
python train.py --config configs/train_config.yaml
```

This runs the full pipeline including live expert inference at every training step.

### Cached Training (10-20× Faster)

Cached training pre-computes all expert outputs and intermediate features, then loads them from disk during training. This skips Phase 1 entirely, reducing VRAM usage and achieving **10-20× speedup**.

#### Step 1: Extract and Cache Expert Outputs

```bash
python scripts/extract_features_balanced.py \
    --hr_dir dataset/DF2K/train_HR \
    --lr_dir dataset/DF2K/train_LR \
    --output_dir cached_features/train
```

This generates per-image `.pt` files containing:
- `lr`: LR input tensor `[3, H, W]`
- `hr`: HR target tensor `[3, 4H, 4W]`
- `expert_imgs`: `{hat: [3,4H,4W], drct: ..., grl: ..., edsr: ...}`
- `expert_feats`: `{hat: [180,H,W], drct: [180,H,W], grl: [180,H,W], edsr: [256,H,W]}`

#### Step 2: Train with Cached Features

```bash
python train.py --config configs/train_config.yaml --cached --cache_dir cached_features/train
```

The `CachedSRDataset` class (`src/data/cached_dataset.py`) loads pre-computed features and applies geometric augmentations (flip, rotation) consistently across all tensors.

### Resuming Training

```bash
python train.py --config configs/train_config.yaml --resume checkpoints/phase3_single_gpu/checkpoint_epoch_100.pth
```

Resumes training from a checkpoint, restoring model weights, optimizer state, scheduler state, and EMA parameters.

---

### Multi-Stage Loss Scheduling

Training uses a **3-stage loss strategy** that gradually introduces frequency and perceptual losses:

| Stage | Epochs | Name | Loss Weights | Purpose |
|-------|--------|------|-------------|---------|
| **1** | 0–80 | `foundation_psnr` | L1=1.0 | Build strong pixel-level reconstruction |
| **2** | 80–150 | `frequency_refinement` | L1=0.75, SWT=0.20, FFT=0.05 | Enhance frequency detail |
| **3** | 150–200 | `detail_enhancement` | L1=0.60, SWT=0.25, FFT=0.10, SSIM=0.05 | Final texture and edge refinement |

The `MultiStageLossScheduler` (`src/training/multi_stage_scheduler.py`) automatically transitions between stages based on the current epoch.

---

### Training Configuration

Key training hyperparameters from `configs/train_config.yaml`:

| Parameter | Value | Description |
|-----------|-------|-------------|
| Total Epochs | 200 | Full training duration |
| Batch Size | 32 | Per-GPU batch size |
| Optimizer | AdamW | With weight decay 1e-4 |
| Learning Rate | 1e-4 | Initial learning rate |
| LR Schedule | CosineAnnealingWarmRestarts | T₀=50, T_mult=2, η_min=5e-8 |
| Warmup | 5 epochs | From 5e-7 to 1e-4 |
| Gradient Clipping | 1.0 | Max gradient norm |
| Precision | FP16 (AMP) | Mixed precision training |
| EMA Decay | 0.9995 | Exponential moving average |
| LR Patch Size | 64×64 | Training crop size |
| Scale Factor | 4× | Upscaling factor |
| Dataset Repeat | 20× | Effective epoch length multiplier |

---

## Inference

### Validation

```bash
python scripts/validate.py \
    --config configs/train_config.yaml \
    --checkpoint checkpoints/phase3_single_gpu/best_psnr.pth \
    --output_dir results/validation
```

Metrics are computed on the Y channel (luminance) with a 4-pixel border crop, following NTIRE evaluation standards.

### Track B (Perceptual Quality)

For Track B submissions with TSD-SR diffusion refinement:

```python
from src.models import create_inference_pipeline

# Creates full 7-phase + TSD-SR pipeline
model = create_inference_pipeline(
    expert_ensemble=expert_ensemble,
    tsdsr_student_path="pretrained/tsdsr/transformer.safetensors",
    tsdsr_vae_path="pretrained/tsdsr/vae.safetensors",
    device='cuda'
)

with torch.no_grad():
    sr_perceptual = model(lr_tensor)
```

---

## Loss Functions

**File:** `src/losses/perceptual_loss.py`

The `CombinedLoss` class manages 8 loss functions with configurable weights:

| Loss | Class | Description |
|------|-------|-------------|
| **L1** | `L1Loss` | Pixel-wise absolute error (primary PSNR loss) |
| **Charbonnier** | `CharbonnierLoss` | Smooth L1 variant: `√(x² + ε²)`, ε=1e-6 |
| **SSIM** | `SSIMLoss` | Structural similarity (1 - SSIM), window=11 |
| **VGG Perceptual** | `VGGPerceptualLoss` | Feature matching at relu2_2, relu3_4, relu4_4 |
| **SWT Frequency** | `SWTFrequencyLoss` | Stationary Wavelet Transform L1 on subbands (db4, 3 levels) |
| **FFT Frequency** | `FFTFrequencyLoss` | Frequency-domain L1 on log-magnitude spectrum |
| **Edge** | `EdgeLoss` | Sobel-filtered edge map L1 |
| **CLIP Semantic** | `CLIPSemanticLoss` | CLIP feature-space cosine similarity |

The SWT loss uses GPU approximation when PyWavelets is unavailable, falling back to Haar wavelet convolutions.

---

## Data Pipeline

### Dataset Classes

| Class | File | Purpose |
|-------|------|---------|
| `SRDataset` | `dataset.py` | Base dataset for pre-generated LR-HR pairs |
| `DF2KDataset` | `dataset.py` | DF2K-specific with auto directory detection |
| `ValidationDataset` | `dataset.py` | Full-image loading (no patching) for evaluation |
| `CachedSRDataset` | `cached_dataset.py` | Loads pre-computed expert features from disk |

### Augmentations

**File:** `src/data/augmentations.py` — `SRTrainAugmentation` pipeline

All augmentations are applied identically to both LR and HR images to maintain alignment:

| Augmentation | Class | Default Config |
|-------------|-------|---------------|
| **Random Crop** | `PairedRandomCrop` | 64×64 LR → 256×256 HR |
| **Random Flip** | `PairedRandomFlip` | p=0.5 horizontal + vertical |
| **Random Rotation** | `PairedRandomRotation` | p=0.5, angles: {90°, 180°, 270°} |
| **Color Jitter** | `ColorJitter` | p=0.2, brightness/contrast/saturation=0.05 |
| **Gaussian Blur** | `GaussianBlur` | Optional, kernel={3,5} |
| **Random Crop Scale** | — | Scale range [0.9, 1.1] |

CutBlur and Mixup augmentations are available but disabled by default.

---

## Evaluation Metrics

**File:** `src/utils/metrics.py`

| Metric | Function | Description |
|--------|----------|-------------|
| **PSNR** | `calculate_psnr()` | Peak Signal-to-Noise Ratio (Y channel, 4px border crop) |
| **SSIM** | `calculate_ssim()` | Structural Similarity (GPU-accelerated, Gaussian window σ=1.5) |
| **LPIPS** | `LPIPSCalculator` | Learned Perceptual Image Patch Similarity (AlexNet backbone) |

The `MetricCalculator` class provides thread-safe running average tracking during training. RGB-to-Y conversion uses the ITU-R BT.601 standard.

---

## Model Details

### Parameter Counts

| Component | Parameters | Trainable |
|-----------|-----------|-----------|
| HAT-L Expert | ~40.85M | ❌ Frozen |
| DRCT-L Expert | ~27.58M | ❌ Frozen |
| GRL-B Expert | ~20.20M | ❌ Frozen |
| EDSR-L Expert | ~43.09M | ❌ Frozen |
| **Total Frozen** | **~131.72M** | — |
| Frequency Decomposition (Phase 2) | ~50K | ✅ |
| Cross-Band Attention + LKA (Phase 3) | ~200K | ✅ |
| Collaborative Learning + LKA (Phase 4) | ~300K | ✅ |
| Hierarchical Fusion (Phase 5) | ~150K | ✅ |
| Dynamic Expert Selection (Phase 6) | ~10K | ✅ |
| Deep Refinement (Phase 7a) | ~400K | ✅ |
| Edge Enhancement (Phase 7b) | ~30K | ✅ |
| Routing + Misc | ~60K | ✅ |
| **Total Trainable** | **~1.2M** | — |

### Key Design Decisions

1. **Frozen experts** — Leverages pre-trained SOTA models without fine-tuning, focusing all trainable capacity on intelligent fusion
2. **Frequency-domain guidance** — DCT+DWT+FFT decomposition provides richer signal representation than spatial-only approaches
3. **Decomposed LKA (k=21)** — Achieves a 21×21 receptive field with O(k) parameters instead of O(k²), enabling global context without excessive computation
4. **Per-pixel difficulty gating** — Adaptively allocates expert computation based on local image complexity
5. **Cached training** — Pre-computing frozen expert outputs enables 10-20× training speedup
6. **3-stage loss scheduling** — Curriculum-based loss progression prevents early-stage conflict between PSNR and perceptual objectives

### Checkpoint Management

**File:** `src/utils/checkpoint_manager.py` — `CheckpointManager` class

- **Atomic saves** — Writes to temp file first, then renames (prevents corruption on crash)
- **Best-K tracking** — Keeps top 5 checkpoints by PSNR
- **Last-N retention** — Keeps last 10 checkpoints regardless of quality
- **Milestone preservation** — Saves epochs 50, 100, 150, 200 permanently

### EMA (Exponential Moving Average)

**File:** `src/utils/checkpoint_manager.py` — `EMAModel` class

- Maintains shadow copies of all trainable parameters
- Updates every training step with decay=0.9995
- Applied during validation for smoother, more stable predictions
- Checkpoint saves include EMA state for consistent resumption

---

## Configuration Reference

The complete configuration is in `configs/train_config.yaml`. Key sections:

<details>
<summary><strong>Model Configuration</strong></summary>

```yaml
model:
  type: "CompleteEnhancedFusionSR"
  scale: 4
  experts:
    - name: "HAT"        # HAT-L: embed_dim=180, depths=[6]×12, window=16
    - name: "DRCT"       # DRCT-L: embed_dim=180, depths=[6]×12, window=16
    - name: "GRL"        # GRL-B: embed_dim=180, depths=[4,4,8,8,4,4], window=8
    - name: "EDSR"       # EDSR-L: num_feat=256, num_block=32
  fusion:
    num_experts: 4
    fusion_dim: 128
    refine_channels: 128
    refine_depth: 6
    base_channels: 64
    improvements:
      dynamic_expert_selection: true
      cross_band_attention: true
      adaptive_frequency_bands: true
      multi_resolution_fusion: true
      collaborative_learning: true
      edge_enhancement: true
```
</details>

<details>
<summary><strong>Loss Configuration</strong></summary>

```yaml
loss:
  stages:
    - epochs: [0, 80]      # Stage 1: L1=1.0
    - epochs: [80, 150]    # Stage 2: L1=0.75, SWT=0.20, FFT=0.05
    - epochs: [150, 200]   # Stage 3: L1=0.60, SWT=0.25, FFT=0.10, SSIM=0.05
  swt:
    levels: 3
    wavelet: "db4"
  fft:
    loss_type: "l1"
    log_scale: true
  ssim:
    window_size: 11
```
</details>

<details>
<summary><strong>Hardware & Training</strong></summary>

```yaml
training:
  total_epochs: 200
  batch_size: 32
  precision: "fp16"
  gradient_clip: 1.0
  optimizer:
    type: "AdamW"
    lr: 1.0e-4
    weight_decay: 1.0e-4
  scheduler:
    type: "CosineAnnealingWarmRestarts"
    T_0: 50
    T_mult: 2
    warmup_epochs: 5
  ema:
    decay: 0.9995

hardware:
  gpu_ids: [0]
  cudnn_benchmark: true

seed: 42
```
</details>

<details>
<summary><strong>Dataset & Augmentation</strong></summary>

```yaml
dataset:
  train:
    root: "dataset/DF2K"
    hr_subdir: "train_HR"
    lr_subdir: "train_LR"
  lr_patch_size: 64
  scale: 4
  augmentation:
    use_flip: true
    use_rotation: true
    use_color_jitter: true
    color_jitter_prob: 0.2
    use_random_crop_scale: true
    crop_scale_range: [0.9, 1.1]
  repeat_factor: 20
```
</details>

---

## Troubleshooting

### Common Issues

| Issue | Solution |
|-------|----------|
| **CUDA OOM during training** | Reduce `batch_size` to 16 or 8; enable `use_amp: true` |
| **Expert weights fail to load** | Verify weight file paths in config match actual files in `pretrained/` |
| **Slow training (>15s/batch)** | Use cached training mode (see [Cached Training](#cached-training-10-20-faster)) |
| **NaN loss values** | Reduce `lr` to 5e-5; check gradient clipping is enabled |
| **Channel mismatch in hierarchical fusion** | Ensure all 4 experts are loaded; check `num_experts=4` in config |
| **Import errors** | Run from project root directory; ensure all `__init__.py` files exist |
| **TSD-SR fails to load** | Install `diffusers>=0.21.0` and `safetensors>=0.3.0` |

### Memory Optimization Tips

1. **Use cached training** — Reduces VRAM from ~24GB to ~8GB by skipping live expert inference
2. **Enable AMP** — `precision: "fp16"` halves memory for activations
3. **Reduce patch size** — Lower `lr_patch_size` from 64 to 48
4. **Gradient accumulation** — Set `accumulation_steps: 2` to simulate larger batches

---

## Citation

```bibtex
@inproceedings{championshipsr2026,
  title={Championship Super-Resolution: Multi-Expert Fusion with Frequency-Guided Adaptive Selection},
  author={Nikhil AI Labs},
  booktitle={NTIRE 2026 Workshop, CVPR},
  year={2026}
}
```

---

## Acknowledgments

This project builds upon several outstanding works in image super-resolution:

- **[HAT](https://github.com/XPixelGroup/HAT)** — Hybrid Attention Transformer (Chen et al., 2023)
- **[DRCT](https://github.com/ming053l/DRCT)** — Dense Residual Connected Transformer
- **[GRL](https://github.com/ofsoundof/GRL-Image-Restoration)** — Global-Regional-Local Image Restoration (Li et al., CVPR 2023)
- **[EDSR](https://github.com/sanghyun-son/EDSR-PyTorch)** — Enhanced Deep Residual SR (Lim et al., 2017)
- **[TSD-SR](https://github.com/Microtreei/TSD-SR)** — Target Score Distillation for SR
- **[NTIRE Challenge](https://www.cvlibs.net/workshops/ntire2026/)** — Image Restoration Challenge

---

<div align="center">

**Built with ❤️ by [Nikhil AI Labs](https://github.com/Nikhil-AI-Labs)**

*Championship SR — Pushing the boundaries of image super-resolution*

</div>
