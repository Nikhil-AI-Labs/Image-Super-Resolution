# FreqFusionSR: Multi-Expert Frequency-Guided Fusion for Image Super-Resolution

### [NTIRE 2026 Challenge on Image Super-Resolution (×4)](https://cvlai.net/ntire/2026/) @ [CVPR 2026](https://cvpr.thecvf.com/)

**Team 29: Anant_SVNIT**

---

## 🏗️ Architecture Overview

```
LR Image [B, 3, H, W]
    │
    ├──→ DRCT-L    (180-dim, 12 RSTB, window=16)  ──→ SR₁ + features₁
    ├──→ GRL-B     (180-dim, 6-stage, anchored)    ──→ SR₂ + features₂
    ├──→ NAFNet    (width=64, SIDD denoiser→SR)    ──→ SR₃ + features₃
    └──→ MambaIR   (180-dim, 6 RSSB, Mamba SSM)   ──→ SR₄ + features₄
                                                         │
                     ┌───────────────────────────────────┘
                     ▼
    ┌─────────────────────────────────────────────────┐
    │     7-Phase Frequency-Guided Fusion Network     │
    │                                                 │
    │  Phase 1: Expert Processing (frozen, ~131M)     │
    │  Phase 2: Multi-Domain Frequency Decomposition  │
    │           (DCT + DWT + FFT, 9 sub-bands)        │
    │  Phase 3: Cross-Band Attention (LKA, k=21)      │
    │  Phase 4: Collaborative Feature Learning        │
    │           (Cross-Expert Attention, 8 heads)      │
    │  Phase 5: Hierarchical Multi-Resolution Fusion  │
    │           (64 → 128 → 256 channels)             │
    │  Phase 6: Dynamic Expert Selection              │
    │           (per-pixel difficulty gating)          │
    │  Phase 7: Deep CNN Refinement + Edge Enhancement│
    │           (Laplacian pyramid, 6-layer 128ch)    │
    └─────────────────────────────────────────────────┘
                     │
                     ▼
              SR Image [B, 3, 4H, 4W]
```

**Parameters:**
- **Frozen experts:** ~131M (DRCT-L + GRL-B + NAFNet + MambaIR)
- **Trainable fusion:** ~1.2M
- **Training:** DF2K (DIV2K + Flickr2K), AdamW, 3-stage loss curriculum (L1 → SWT+FFT → SSIM refinement)

---

## 📁 Repository Structure

```
Image-Super-Resolution/
├── models/
│   └── team29_FreqFusionSR/          # NTIRE submission interface
│       ├── __init__.py
│       └── io.py                     # main(model_dir, input_path, output_path, device)
├── model_zoo/
│   └── team29_FreqFusionSR/
│       └── team29_FreqFusionSR.txt   # Download links for pretrained weights
├── src/                              # Full training & model codebase
│   ├── models/
│   │   ├── expert_loader.py          # ExpertEnsemble (DRCT, GRL, NAFNet)
│   │   ├── enhanced_fusion_v2.py     # CompleteEnhancedFusionSR (7-phase fusion)
│   │   ├── mambair/                  # MambaIR architecture
│   │   ├── drct/                     # DRCT-L architecture
│   │   ├── grl/                      # GRL-B architecture
│   │   └── nafnet/                   # NAFNet architecture
│   ├── losses/                       # Multi-stage loss functions
│   ├── data/                         # Dataset loaders
│   ├── training/                     # Training utilities
│   └── utils/                        # Internal utilities
├── configs/
│   └── train_config.yaml             # Full training configuration
├── scripts/                          # Utility scripts
├── utils/                            # NTIRE evaluation utilities
├── factsheet/                        # Challenge factsheet (LaTeX)
├── figs/                             # Figures for README
├── test.py                           # NTIRE official test interface
├── eval.py                           # IQA evaluation script
├── train.py                          # Training script
├── requirements.txt                  # Dependencies
├── LICENSE                           # Apache 2.0
└── README.md                         # This file
```

---

## 🚀 Quick Start

### 1. Clone the repository

```bash
git clone https://github.com/Nikhil-AI-Labs/Image-Super-Resolution.git
cd Image-Super-Resolution
```

### 2. Install dependencies

```bash
# Create environment
conda create -n freqfusion python=3.10
conda activate freqfusion

# Install PyTorch (CUDA 11.8)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Install requirements
pip install -r requirements.txt

# Install Mamba SSM (requires CUDA)
pip install mamba-ssm causal-conv1d
```

### 3. Download pre-trained weights

Download all model weights and place them in `model_zoo/team29_FreqFusionSR/`:

| Weight File | Model | Size |
|---|---|---|
| `DRCT-L_X4.pth` | DRCT-L (×4 SR) | ~130 MB |
| `GRL-B_SR_x4.pth` | GRL-B (×4 SR) | ~130 MB |
| `NAFNet-SIDD-width64.pth` | NAFNet-SIDD (denoiser) | ~75 MB |
| `MambaIR_x4.pth` | MambaIR (×4 SR) | ~130 MB |
| `fusion_best.pth` | Trained fusion network | ~5 MB |

**Download link:** [Google Drive](https://drive.google.com/file/d/12JvYplsdrfqzgBLFceQdfiLQWy5dC-91/view?usp=sharing)

### 4. Run inference

```bash
# Test on DIV2K test set (LR images)
CUDA_VISIBLE_DEVICES=0 python test.py \
    --test_dir /path/to/DIV2K_test_LR_bicubic/X4 \
    --save_dir ./results \
    --model_id 29
```

You can also use `--valid_dir` for the validation set, or both:

```bash
CUDA_VISIBLE_DEVICES=0 python test.py \
    --valid_dir /path/to/DIV2K_valid_LR_bicubic/X4 \
    --test_dir /path/to/DIV2K_test_LR_bicubic/X4 \
    --save_dir ./results \
    --model_id 29
```

---

## 📊 Evaluation

### IQA Metrics

```bash
python eval.py \
    --output_folder "./results/29_FreqFusionSR/test" \
    --target_folder "/path/to/DIV2K_test_HR" \
    --metrics_save_path "./IQA_results" \
    --gpu_ids 0
```

### Weighted Score (Perception Quality Track)

$$\text{Score} = (1 - \text{LPIPS}) + (1 - \text{DISTS}) + \text{CLIPIQA} + \text{MANIQA} + \frac{\text{MUSIQ}}{100} + \max\left(0, \frac{10 - \text{NIQE}}{10}\right)$$

---

## 🔧 Training

```bash
# Single GPU training with DF2K dataset
python train.py --config configs/train_config.yaml
```

**Training details:**
- **Dataset:** DF2K (DIV2K + Flickr2K), 64×64 LR patches
- **Optimizer:** AdamW, lr=2×10⁻⁴, weight_decay=1×10⁻⁴
- **Scheduler:** CosineAnnealingWarmRestarts (T₀=50, T_mult=2)
- **Loss stages:**
  1. Epochs 0–50: Pure L1 (foundation)
  2. Epochs 50–100: L1 + SWT + FFT (frequency refinement)
  3. Epochs 100–150: L1 + SWT + FFT + SSIM (detail enhancement)
- **EMA:** decay=0.9995

---

## 🏗️ Method Description

### Multi-Expert Ensemble (Frozen)

We use four complementary super-resolution experts, each frozen (~131M total parameters):

| Expert | Architecture | Strength |
|---|---|---|
| **DRCT-L** | Dense Residual Connected Transformer | Fine texture reconstruction |
| **GRL-B** | Global Routing with anchored stripes | Long-range structural coherence |
| **NAFNet** | Non-linear Activation Free Network | Noise-robust denoising + SR |
| **MambaIR** | Mamba State-Space Model | Efficient long-range dependencies |

### 7-Phase Frequency-Guided Fusion (~1.2M trainable)

1. **Expert Processing** — Each expert runs independently (OOM-safe sequential execution)
2. **Multi-Domain Frequency Decomposition** — DCT (8×8 blocks) + DWT (db4, 3 levels) + FFT → 9 frequency sub-bands
3. **Cross-Band Attention** — Large Kernel Attention (k=21) enables frequency bands to communicate
4. **Collaborative Feature Learning** — Cross-expert multi-head attention (8 heads) shares complementary features
5. **Hierarchical Multi-Resolution Fusion** — Progressive 64→128→256 channel processing
6. **Dynamic Expert Selection** — Per-pixel difficulty estimation gates expert contributions (1–4 experts per pixel)
7. **Deep Refinement + Edge Enhancement** — 6-layer 128-channel CNN + Laplacian pyramid edge enhancement

---

## 👥 Team

**Team Name:** Anant_SVNIT  
**Team ID:** 29  
**Codabench Username:** nikhil-ai

| Name | Affiliation |
|---|---|
| Nikhil Pathak | SVNIT |
| Milan | SVNIT |
| Aagam | SVNIT |
| Vivek | SVNIT |
| Sarang | SVNIT |

---

## 📜 License

This project is licensed under the Apache License 2.0 — see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgement

This code is built upon the [NTIRE 2026 Image SR ×4 Challenge](https://github.com/zhengchen1999/NTIRE2026_ImageSR_x4) baseline. We thank the organizers for providing the evaluation framework.

We also acknowledge the authors of the expert models used in our ensemble:
- [DRCT](https://github.com/ming053l/DRCT) — Dense Residual Connected Transformer
- [GRL](https://github.com/ofsoundof/GRL-Image-Restoration) — Global Residual Learning
- [NAFNet](https://github.com/megvii-research/NAFNet) — Non-linear Activation Free Network
- [MambaIR](https://github.com/csguoh/MambaIR) — Mamba for Image Restoration

### NTIRE Image SR ×4 Challenge Series

- **NTIRE 2025:** [CODE](https://github.com/zhengchen1999/NTIRE2025_ImageSR_x4) | [PDF](https://openaccess.thecvf.com/content/CVPR2025/papers/)
- **NTIRE 2024:** [CODE](https://github.com/zhengchen1999/NTIRE2024_ImageSR_x4) | [PDF](https://openaccess.thecvf.com/content/CVPR2024/papers/)
- **NTIRE 2023:** [CODE](https://github.com/zhengchen1999/NTIRE2023_ImageSR_x4) | [PDF](https://openaccess.thecvf.com/content/CVPR2023/papers/)
