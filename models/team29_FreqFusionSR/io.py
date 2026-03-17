"""
FreqFusionSR — NTIRE 2026 Image Super-Resolution (×4) Challenge
================================================================
Team 29: Anant_SVNIT

Interface module following the official NTIRE 2026 submission format.
Wraps the multi-expert frequency-guided fusion pipeline into the
standard `main(model_dir, input_path, output_path, device)` interface.

Architecture:
  4 Frozen Experts → 7-Phase Frequency-Guided Fusion → SR Image (×4)
  - Expert 1: DRCT-L (180-dim, 12 RSTB, window=16)
  - Expert 2: GRL-B  (180-dim, 6-stage, global routing)
  - Expert 3: NAFNet-SIDD (width=64, U-Net denoiser → SR adapter)
  - Expert 4: MambaIR (180-dim, 6 RSSB, state-space model)
  - Fusion: DCT+DWT+FFT decomposition, cross-band attention,
            collaborative learning, dynamic expert selection,
            hierarchical multi-resolution fusion, edge enhancement
"""

import os
import sys
import types
import glob
import torch
import torch.nn.functional as F
import cv2
import yaml
import numpy as np
from pathlib import Path
from tqdm import tqdm

# =====================================================================
# Resolve repo root — needed so `from src.models...` works correctly
# =====================================================================
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

# =====================================================================
# Monkey-patch: prevent diffusers/peft/transformers crash
# tsdsr_wrapper.py imports diffusers which conflicts with Kaggle's
# peft/transformers versions. We don't use TSD-SR for inference.
# =====================================================================
for _mod_name in ['src.models.tsdsr_wrapper', 'src.models.complete_sr_pipeline']:
    if _mod_name not in sys.modules:
        _fake = types.ModuleType(_mod_name)
        for _attr in ['TSDSRInference', 'VAEWrapper', 'CompleteSRPipeline']:
            setattr(_fake, _attr, type('_Dummy', (), {}))
        for _attr in ['load_tsdsr_models', 'create_tsdsr_refinement_pipeline',
                     'create_complete_pipeline', 'create_training_pipeline',
                     'create_inference_pipeline']:
            setattr(_fake, _attr, lambda *a, **kw: None)
        sys.modules[_mod_name] = _fake

# ── Now safe to import from src ─────────────────────────────────────
from src.models.expert_loader import ExpertEnsemble
from src.models.enhanced_fusion_v2 import CompleteEnhancedFusionSR
from src.models.mambair.mambair_arch import MambaIR

# =====================================================================
# Constants
# =====================================================================
SCALE = 4
CONFIG_PATH = os.path.join(REPO_ROOT, 'configs', 'train_config.yaml')


# =====================================================================
# Utility functions
# =====================================================================
def _pad16(t):
    """Pad spatial dims to multiple of 16 (reflect)."""
    _, _, h, w = t.shape
    ph = (16 - h % 16) % 16
    pw = (16 - w % 16) % 16
    if ph or pw:
        t = F.pad(t, (0, pw, 0, ph), mode='reflect')
    return t, (h, w)


def _unpad(t, h, w, scale=4):
    """Crop back to original spatial size × scale."""
    return t[:, :, :h * scale, :w * scale]


def _imread_uint(path, n_channels=3):
    """Read image as uint8 HxWxC (RGB)."""
    if n_channels == 1:
        img = cv2.imread(path, 0)
        img = np.expand_dims(img, axis=2)
    else:
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def _uint2tensor4(img):
    """uint8 HxWxC → float32 [1,C,H,W] in [0,1]."""
    return torch.from_numpy(
        np.ascontiguousarray(img)
    ).permute(2, 0, 1).float().div(255.0).unsqueeze(0)


def _tensor2uint(img):
    """float32 [1,C,H,W] or [C,H,W] in [0,1] → uint8 HxWxC."""
    img = img.data.squeeze().float().clamp_(0, 1).cpu().numpy()
    if img.ndim == 3:
        img = np.transpose(img, (1, 2, 0))
    return np.uint8((img * 255.0).round())


def _imsave(img, img_path):
    """Save uint8 HxWxC (RGB) as image file."""
    img = np.squeeze(img)
    if img.ndim == 3:
        img = img[:, :, [2, 1, 0]]  # RGB→BGR
    cv2.imwrite(img_path, img)


# =====================================================================
# Model loading
# =====================================================================
def _load_all_models(model_dir, device):
    """
    Load all 4 expert models + fusion from model_dir.

    Expected files in model_dir:
      - DRCT-L_X4.pth
      - GRL-B_SR_x4.pth
      - NAFNet-SIDD-width64.pth
      - MambaIR_x4.pth
      - fusion_best.pth
    """
    # ── Load config for fusion parameters ────────────────────────────
    with open(CONFIG_PATH) as f:
        cfg = yaml.safe_load(f)
    fusion_cfg = cfg.get('model', {}).get('fusion', {})
    improvements = fusion_cfg.get('improvements', {})
    scale = cfg.get('dataset', {}).get('scale', SCALE)

    # ── Expert ensemble (DRCT, GRL, NAFNet) ──────────────────────────
    ckpts = {
        'drct':   os.path.join(model_dir, 'DRCT-L_X4.pth'),
        'grl':    os.path.join(model_dir, 'GRL-B_SR_x4.pth'),
        'nafnet': os.path.join(model_dir, 'NAFNet-SIDD-width64.pth'),
    }
    ensemble = ExpertEnsemble(device=device, upscale=scale)
    ensemble.load_all_experts(checkpoint_paths=ckpts, freeze=True)
    ensemble._register_all_hooks()
    print(f"  ✅ DRCT-L + GRL-B + NAFNet loaded")

    # ── MambaIR ──────────────────────────────────────────────────────
    mamba = MambaIR(
        upscale=scale, in_chans=3, img_size=64, window_size=16,
        compress_ratio=3, squeeze_factor=30, conv_scale=0.01,
        overlap_ratio=0.5, img_range=1.0,
        depths=(6, 6, 6, 6, 6, 6), embed_dim=180, mlp_ratio=2.0,
        drop_path_rate=0.1, upsampler='pixelshuffle',
        resi_connection='1conv',
    )
    mamba_path = os.path.join(model_dir, 'MambaIR_x4.pth')
    state = torch.load(mamba_path, map_location='cpu', weights_only=False)
    sd = state.get('params', state.get('state_dict', state.get('model', state)))
    sd = {k.replace('module.', ''): v for k, v in sd.items()}
    mamba.load_state_dict(sd, strict=False)
    mamba.eval().to(device)

    mamba_feat_cache = {}
    if hasattr(mamba, 'conv_after_body'):
        mamba.conv_after_body.register_forward_hook(
            lambda m, i, o: mamba_feat_cache.update({'feat': o.detach()})
        )
    print(f"  ✅ MambaIR loaded")

    # ── Fusion model (headless / cached mode) ────────────────────────
    fusion = CompleteEnhancedFusionSR(
        expert_ensemble=None,
        num_experts=fusion_cfg.get('num_experts', 4),
        fusion_dim=fusion_cfg.get('fusion_dim', 128),
        refine_channels=fusion_cfg.get('refine_channels', 128),
        refine_depth=fusion_cfg.get('refine_depth', 6),
        base_channels=fusion_cfg.get('base_channels', 64),
        block_size=fusion_cfg.get('block_size', 8),
        upscale=scale,
        enable_dynamic_selection=improvements.get('dynamic_expert_selection', True),
        enable_cross_band_attn=improvements.get('cross_band_attention', True),
        enable_adaptive_bands=improvements.get('adaptive_frequency_bands', True),
        enable_multi_resolution=improvements.get('multi_resolution_fusion', True),
        enable_collaborative=improvements.get('collaborative_learning', True),
        enable_edge_enhance=improvements.get('edge_enhancement', True),
    )

    fusion_path = os.path.join(model_dir, 'fusion_best.pth')
    ckpt = torch.load(fusion_path, map_location='cpu', weights_only=False)
    raw_sd = ckpt.get('model_state_dict', ckpt.get('state_dict', ckpt))
    clean_sd = {}
    for k, v in raw_sd.items():
        key = k
        for pfx in ['module.', 'model.']:
            if key.startswith(pfx):
                key = key[len(pfx):]
        clean_sd[key] = v

    model_sd = fusion.state_dict()
    loaded = sum(1 for k, v in clean_sd.items()
                 if k in model_sd and v.shape == model_sd[k].shape)
    model_sd.update({k: v for k, v in clean_sd.items()
                     if k in model_sd and v.shape == model_sd[k].shape})
    fusion.load_state_dict(model_sd, strict=False)
    fusion.eval().to(device)
    print(f"  ✅ Fusion loaded ({loaded} tensors)")

    return ensemble, mamba, mamba_feat_cache, fusion, scale


# =====================================================================
# Single-image inference
# =====================================================================
def _process_image(lr_tensor, ensemble, mamba, mamba_feat_cache, fusion, device, scale=4):
    """Run a single LR image through all 4 experts + fusion."""
    lr_padded, (oh, ow) = _pad16(lr_tensor.to(device))

    # ── DRCT ─────────────────────────────────────────────────────────
    ensemble._captured_features = {}
    ensemble._capture_features = True
    drct_sr = ensemble.forward_drct(lr_padded)
    drct_feat = ensemble._captured_features.get(
        'drct', torch.zeros(1, 180, oh, ow, device=device))
    ensemble._capture_features = False
    drct_sr = _unpad(drct_sr, oh, ow, scale).float()
    drct_feat = drct_feat[:, :, :oh, :ow].float()
    torch.cuda.empty_cache()

    # ── GRL ───────────────────────────────────────────────────────────
    ensemble._captured_features = {}
    ensemble._capture_features = True
    grl_sr = ensemble.forward_grl(lr_padded)
    grl_feat = ensemble._captured_features.get(
        'grl', torch.zeros(1, 180, oh, ow, device=device))
    ensemble._capture_features = False
    grl_sr = _unpad(grl_sr, oh, ow, scale).float()
    grl_feat = grl_feat[:, :, :oh, :ow].float()
    torch.cuda.empty_cache()

    # ── NAFNet ────────────────────────────────────────────────────────
    ensemble._captured_features = {}
    ensemble._capture_features = True
    naf_sr = ensemble.forward_nafnet(lr_padded)
    naf_feat = ensemble._captured_features.get(
        'nafnet', torch.zeros(1, 64, oh * scale, ow * scale, device=device))
    ensemble._capture_features = False
    naf_sr = _unpad(naf_sr, oh, ow, scale).float()
    naf_feat = F.interpolate(
        naf_feat, size=(oh, ow), mode='bilinear', align_corners=False
    ).float()
    torch.cuda.empty_cache()

    # ── MambaIR ───────────────────────────────────────────────────────
    mamba_feat_cache.clear()
    with torch.amp.autocast('cuda'):
        mamba_sr = mamba(lr_padded).clamp(0, 1)
    mamba_sr = _unpad(mamba_sr, oh, ow, scale).float()
    mamba_feat = mamba_feat_cache.get(
        'feat', torch.zeros(1, 180, oh, ow, device=device))
    mamba_feat = mamba_feat[:, :, :oh, :ow].float()
    torch.cuda.empty_cache()

    # ── Fusion forward ────────────────────────────────────────────────
    lr_in = lr_padded[:, :, :oh, :ow]
    expert_imgs = {
        'drct': drct_sr, 'grl': grl_sr, 'nafnet': naf_sr, 'mamba': mamba_sr,
    }
    expert_feats = {
        'drct': drct_feat, 'grl': grl_feat, 'nafnet': naf_feat, 'mamba': mamba_feat,
    }

    sr_out = fusion.forward_with_precomputed(lr_in, expert_imgs, expert_feats)

    # Clean up
    del lr_padded, lr_in
    del drct_sr, drct_feat, grl_sr, grl_feat
    del naf_sr, naf_feat, mamba_sr, mamba_feat
    del expert_imgs, expert_feats
    torch.cuda.empty_cache()

    return sr_out


# =====================================================================
# NTIRE submission interface
# =====================================================================
def main(model_dir, input_path, output_path, device=None):
    """
    NTIRE 2026 official submission interface.

    Args:
        model_dir  : Path to model weights directory
                     (e.g. model_zoo/team29_FreqFusionSR/)
        input_path : Folder containing LR PNG images
        output_path: Folder to save SR output PNG images
        device     : Computation device (default: cuda)
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f"\n{'='*60}")
    print(f"  FreqFusionSR — Team 29 (Anant_SVNIT)")
    print(f"  NTIRE 2026 Image Super-Resolution (×4)")
    print(f"{'='*60}")
    print(f"  Weights : {model_dir}")
    print(f"  Input   : {input_path}")
    print(f"  Output  : {output_path}")
    print(f"  Device  : {device}")
    print(f"{'='*60}\n")

    # Load all models
    ensemble, mamba, mamba_feat_cache, fusion, scale = _load_all_models(model_dir, device)

    # Scan input images
    input_imgs = sorted(
        glob.glob(os.path.join(input_path, '*.[jpJP][pnPN]*[gG]'))
    )
    os.makedirs(output_path, exist_ok=True)

    print(f"\n  Processing {len(input_imgs)} images ...\n")

    with torch.no_grad():
        for img_path in tqdm(input_imgs, desc='FreqFusionSR', ncols=80):
            img_name, ext = os.path.splitext(os.path.basename(img_path))

            # Load LR image
            img_lr = _imread_uint(img_path, n_channels=3)
            img_lr = _uint2tensor4(img_lr).to(device)

            # Super-resolve
            img_sr = _process_image(
                img_lr, ensemble, mamba, mamba_feat_cache, fusion, device, scale
            )

            # Save
            img_sr = _tensor2uint(img_sr)
            _imsave(img_sr, os.path.join(output_path, img_name + ext))

    print(f"\n  ✅ Done — {len(input_imgs)} images saved to {output_path}")
