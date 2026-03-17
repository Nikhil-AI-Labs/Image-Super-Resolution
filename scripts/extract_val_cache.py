"""
scripts/extract_val_cache.py
==============================
Extracts FULL-IMAGE features for the DIV2K Validation set (100 images).
All 4 experts (DRCT, GRL, NAFNet, MambaIR) in one pass.

MULTI-GPU SUPPORT: Automatically detects and uses all available GPUs.
  - On Kaggle 2×T4: GPU 0 processes images [0,2,4,...], GPU 1 processes [1,3,5,...]
  - Falls back to single GPU if only 1 is available.

OOM-SAFE: Experts are run sequentially per image, with CUDA cache cleared between each.
This prevents GPU OOM when processing full-resolution images (e.g. 510×340 → 2040×1360).

Includes HR ground truth so train.py can calculate true, full-image PSNR/SSIM.

Output format (CachedSRDataset compatible):
    {stem}_drct_part.pt  →  outputs.drct + features.drct + lr + hr
    {stem}_rest_part.pt  →  outputs.grl/nafnet + features.grl/nafnet
    {stem}_mamba_part.pt →  outputs.mamba + features.mamba

All tensors stored as FP16 to save disk (~2.5 GB total for 100 images).
Resume-safe: skips images whose _drct_part.pt already exists.

Usage:
    # Kaggle 2×T4 (auto-detect multi-GPU)
    !python scripts/extract_val_cache.py \\
        --dataset-dir /kaggle/input/sr-championship-df2k \\
        --output-dir /kaggle/working/cached_features_val \\
        --mamba-weights /kaggle/input/mambair-weights/MambaIR_x4.pth \\
        --resume

    # Local single GPU
    python scripts/extract_val_cache.py \\
        --dataset-dir dataset/DF2K \\
        --output-dir dataset/DF2K/cached_features_val

Author: NTIRE SR Team
"""

import os
import sys
import glob
import torch
import torch.nn.functional as F
import torch.multiprocessing as mp
import cv2
import yaml
import argparse
import numpy as np
import time
from pathlib import Path
from tqdm import tqdm

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


# =============================================================================
# Utility Functions
# =============================================================================

def pad16(t):
    """Pad tensor spatial dims to multiple of 16 (required by transformer experts)."""
    _, _, h, w = t.shape
    ph = (16 - h % 16) % 16
    pw = (16 - w % 16) % 16
    if ph or pw:
        t = F.pad(t, (0, pw, 0, ph), mode='reflect')
    return t, (ph, pw)


def unpad(t, ph, pw, scale=1):
    """Remove padding from tensor, accounting for scale factor."""
    if t is None:
        return None
    ph_s, pw_s = ph * scale, pw * scale
    if ph_s > 0:
        t = t[:, :, :-ph_s, :]
    if pw_s > 0:
        t = t[:, :, :, :-pw_s]
    return t


def load_img(path):
    """Load image as float32 tensor [1, 3, H, W] in [0, 1]."""
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Failed to load: {path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    return torch.from_numpy(img.transpose(2, 0, 1)).unsqueeze(0)


def discover_val_pairs(dataset_dir):
    """
    Discover HR-LR validation pairs.
    Handles: val_HR/, DIV2K_valid_HR/, val_LR/, DIV2K_valid_LR_bicubic/X4/
    """
    root = Path(dataset_dir)

    # Find HR dir
    hr_dir = None
    for pat in [root / 'val_HR', root / 'DIV2K_valid_HR']:
        if pat.exists():
            hr_dir = pat
            break

    # Find LR dir
    lr_dir = None
    for pat in [root / 'val_LR', root / 'DIV2K_valid_LR_bicubic' / 'X4']:
        if pat.exists():
            lr_dir = pat
            break

    if not hr_dir or not lr_dir:
        raise RuntimeError(
            f"Could not find val HR/LR dirs in {dataset_dir}.\n"
            f"Expected: val_HR/ + val_LR/ or DIV2K_valid_HR/ + DIV2K_valid_LR_bicubic/X4/"
        )

    hr_files = sorted(glob.glob(str(hr_dir / '*.png')))
    lr_files_raw = sorted(glob.glob(str(lr_dir / '*.png')))

    # Build LR lookup by cleaned stem
    def clean_stem(s):
        for sfx in ['x4', 'x2', 'x3', 'x8', '_LR', '_lr', 'LR', 'lr',
                     '_bicubic', '_BICUBIC']:
            s = s.replace(sfx, '')
        return s.rstrip('_')

    lr_dict = {clean_stem(Path(p).stem): p for p in lr_files_raw}

    pairs = []
    for hr_p in hr_files:
        stem = Path(hr_p).stem
        key = stem if stem in lr_dict else clean_stem(stem)
        if key in lr_dict:
            pairs.append((lr_dict[key], hr_p))

    return pairs


# =============================================================================
# Per-Expert Sequential Forward (OOM-Safe)
# =============================================================================

def run_expert_sequential(ensemble, mamba, mamba_feat_cache, lr_padded, device):
    """
    Run all 4 experts SEQUENTIALLY on one image, freeing memory between each.
    Returns (outputs_dict, features_dict) with all tensors on CPU as FP16.

    This is critical for full-resolution images that would OOM if all experts
    were loaded and run simultaneously.
    """
    outputs = {}
    features = {}

    # Ensure hooks are registered
    if not ensemble._hook_handles:
        ensemble._register_all_hooks()

    # --- DRCT ---
    ensemble._captured_features = {}
    ensemble._capture_features = True
    try:
        with torch.no_grad():
            drct_sr = ensemble.forward_drct(lr_padded)
        outputs['drct'] = drct_sr.cpu().half()
        if 'drct' in ensemble._captured_features:
            features['drct'] = ensemble._captured_features['drct'].clone().cpu().half()
    finally:
        ensemble._capture_features = False
    del drct_sr
    torch.cuda.empty_cache()

    # --- GRL ---
    ensemble._captured_features = {}
    ensemble._capture_features = True
    try:
        with torch.no_grad():
            grl_sr = ensemble.forward_grl(lr_padded)
        outputs['grl'] = grl_sr.cpu().half()
        if 'grl' in ensemble._captured_features:
            features['grl'] = ensemble._captured_features['grl'].clone().cpu().half()
    finally:
        ensemble._capture_features = False
    del grl_sr
    torch.cuda.empty_cache()

    # --- NAFNet ---
    ensemble._captured_features = {}
    ensemble._capture_features = True
    try:
        with torch.no_grad():
            nafnet_sr = ensemble.forward_nafnet(lr_padded)
        outputs['nafnet'] = nafnet_sr.cpu().half()
        if 'nafnet' in ensemble._captured_features:
            features['nafnet'] = ensemble._captured_features['nafnet'].clone().cpu().half()
    finally:
        ensemble._capture_features = False
    del nafnet_sr
    torch.cuda.empty_cache()

    # --- MambaIR ---
    mamba_feat_cache.clear()
    with torch.amp.autocast('cuda'):
        mamba_sr = mamba(lr_padded).clamp(0, 1)
    outputs['mamba'] = mamba_sr.cpu().half()
    if 'feat' in mamba_feat_cache:
        features['mamba'] = mamba_feat_cache['feat'].clone().cpu().half()
    del mamba_sr
    torch.cuda.empty_cache()

    return outputs, features


# =============================================================================
# GPU Worker (runs on each GPU)
# =============================================================================

def gpu_worker(rank, num_gpus, pairs, args, out_dir):
    """Worker function that runs on a single GPU."""
    device = torch.device(f'cuda:{rank}')

    # Shard: GPU 0 gets even indices, GPU 1 gets odd indices, etc.
    my_pairs = pairs[rank::num_gpus]

    is_primary = (rank == 0)
    if not is_primary:
        # Suppress stdout for non-primary to keep logs clean
        sys.stdout = open(os.devnull, 'w')

    if is_primary:
        print(f"\n  [GPU {rank}] Loading experts...")
        print(f"  Total GPUs: {num_gpus}")
        print(f"  Images per GPU: ~{len(my_pairs)}")

    # ── Load Config ──
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    # ── Load 3 Local Experts ──
    from src.models.expert_loader import ExpertEnsemble

    ckpts = {}
    for ec in cfg.get('model', {}).get('experts', []):
        name = ec.get('name', '').lower()
        weight = ec.get('weight_path', '')
        if name and weight and not ec.get('remote_only', False):
            ckpts[name] = weight

    ensemble = ExpertEnsemble(device=device, upscale=4)
    ensemble.load_all_experts(checkpoint_paths=ckpts, freeze=True)

    # ── Load MambaIR ──
    from src.models.mambair.mambair_arch import MambaIR

    mamba = MambaIR(
        upscale=4, in_chans=3, img_size=64, window_size=16,
        compress_ratio=3, squeeze_factor=30, conv_scale=0.01,
        overlap_ratio=0.5, img_range=1.0,
        depths=(6, 6, 6, 6, 6, 6), embed_dim=180, mlp_ratio=2.0,
        drop_path_rate=0.1, upsampler='pixelshuffle',
        resi_connection='1conv',
    )
    if os.path.exists(args.mamba_weights):
        state = torch.load(args.mamba_weights, map_location='cpu', weights_only=False)
        sd = state.get('params', state.get('state_dict', state.get('model', state)))
        sd = {k.replace('module.', ''): v for k, v in sd.items()}
        mamba.load_state_dict(sd, strict=False)
    mamba.eval().to(device)

    # Register MambaIR feature hook
    mamba_feat_cache = {}
    if hasattr(mamba, 'conv_after_body'):
        mamba.conv_after_body.register_forward_hook(
            lambda m, i, o: mamba_feat_cache.update({'feat': o})
        )

    if is_primary:
        print(f"  [GPU {rank}] Experts loaded. Starting extraction...")

    # ── Process images ──
    iterator = tqdm(my_pairs, desc=f'GPU {rank}', ncols=100) if is_primary else my_pairs
    processed, skipped, errors = 0, 0, 0

    with torch.no_grad():
        for lr_path, hr_path in iterator:
            stem = Path(hr_path).stem

            # Resume check
            drct_out = os.path.join(out_dir, f'{stem}_drct_part.pt')
            if args.resume and os.path.exists(drct_out):
                skipped += 1
                continue

            try:
                lr = load_img(lr_path).to(device)
                hr = load_img(hr_path)  # stays on CPU
                orig_h, orig_w = lr.shape[2], lr.shape[3]

                lr_padded, (ph, pw) = pad16(lr)

                # Run all 4 experts sequentially (OOM-safe)
                outs, feats = run_expert_sequential(
                    ensemble, mamba, mamba_feat_cache, lr_padded, device
                )

                # Unpad all tensors
                def u(tensor, scale):
                    if tensor is None:
                        return None
                    return unpad(tensor, ph, pw, scale)

                # Crop DRCT/GRL features to original LR resolution
                for name in ['drct', 'grl']:
                    if name in feats and feats[name] is not None:
                        feats[name] = feats[name][:, :, :orig_h, :orig_w]
                # NAFNet features at upscaled resolution → resize to LR
                if 'nafnet' in feats and feats[name] is not None:
                    feats['nafnet'] = F.interpolate(
                        feats['nafnet'].float(),
                        size=(orig_h, orig_w),
                        mode='bilinear', align_corners=False
                    ).half()
                # MambaIR features → crop to LR resolution
                if 'mamba' in feats and feats['mamba'] is not None:
                    feats['mamba'] = feats['mamba'][:, :, :orig_h, :orig_w]

                # Save 3-part cache
                torch.save({
                    'outputs':  {'drct': u(outs.get('drct'), 4)},
                    'features': {'drct': feats.get('drct')},
                    'lr': lr.cpu().squeeze(0).half(),
                    'hr': hr.squeeze(0).half(),
                    'filename': stem,
                    'original_size': (orig_h, orig_w),
                }, os.path.join(out_dir, f'{stem}_drct_part.pt'))

                torch.save({
                    'outputs': {
                        'grl':    u(outs.get('grl'), 4),
                        'nafnet': u(outs.get('nafnet'), 4),
                    },
                    'features': {
                        'grl':    feats.get('grl'),
                        'nafnet': feats.get('nafnet'),
                    },
                    'filename': stem,
                }, os.path.join(out_dir, f'{stem}_rest_part.pt'))

                torch.save({
                    'outputs':  {'mamba': u(outs.get('mamba'), 4)},
                    'features': {'mamba': feats.get('mamba')},
                    'filename': stem,
                }, os.path.join(out_dir, f'{stem}_mamba_part.pt'))

                processed += 1
                del lr, outs, feats
                torch.cuda.empty_cache()

            except Exception as e:
                errors += 1
                if is_primary:
                    tqdm.write(f"  [ERROR] {stem}: {e}")

    if is_primary:
        print(f"\n  [GPU {rank}] Done: {processed} processed, "
              f"{skipped} skipped, {errors} errors")


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Multi-GPU Full-Image Validation Cache Extractor'
    )
    parser.add_argument('--dataset-dir', required=True,
                        help='Dataset root (contains val_HR/ + val_LR/ or DIV2K_valid_*)')
    parser.add_argument('--output-dir', default=None,
                        help='Output dir (default: {dataset-dir}/cached_features_val)')
    parser.add_argument('--config', default='configs/train_config.yaml')
    parser.add_argument('--mamba-weights',
                        default='pretrained/mambair/MambaIR_x4.pth')
    parser.add_argument('--resume', action='store_true',
                        help='Skip images whose _drct_part.pt already exists')
    parser.add_argument('--num-gpus', type=int, default=None,
                        help='Number of GPUs (default: auto-detect)')
    args = parser.parse_args()

    out_dir = args.output_dir or os.path.join(args.dataset_dir, 'cached_features_val')
    os.makedirs(out_dir, exist_ok=True)

    num_gpus = args.num_gpus or torch.cuda.device_count()
    num_gpus = max(1, num_gpus)

    print("\n" + "=" * 70)
    print("  MULTI-GPU FULL-IMAGE VALIDATION CACHE EXTRACTOR")
    print("=" * 70)
    print(f"  Dataset:  {args.dataset_dir}")
    print(f"  Output:   {out_dir}")
    print(f"  GPUs:     {num_gpus}")
    print(f"  Resume:   {args.resume}")
    print(f"  Storage:  FP16 (half precision)")
    print("=" * 70 + "\n")

    # Discover val pairs
    print("Discovering validation image pairs...")
    pairs = discover_val_pairs(args.dataset_dir)
    print(f"  Found {len(pairs)} validation pairs")

    if not pairs:
        print("  ✗ No validation pairs found!")
        sys.exit(1)

    t0 = time.time()

    if num_gpus > 1:
        print(f"\n  Launching {num_gpus} GPU workers...")
        mp.spawn(
            gpu_worker,
            args=(num_gpus, pairs, args, out_dir),
            nprocs=num_gpus,
            join=True,
        )
    else:
        print("\n  Running single-GPU extraction...")
        gpu_worker(0, 1, pairs, args, out_dir)

    elapsed = time.time() - t0
    n_drct = len(glob.glob(os.path.join(out_dir, '*_drct_part.pt')))
    n_rest = len(glob.glob(os.path.join(out_dir, '*_rest_part.pt')))
    n_mamba = len(glob.glob(os.path.join(out_dir, '*_mamba_part.pt')))

    print(f"\n{'=' * 70}")
    print(f"  VALIDATION CACHE COMPLETE")
    print(f"{'=' * 70}")
    print(f"  Time:     {elapsed:.1f}s ({elapsed/60:.1f} min)")
    print(f"  GPUs:     {num_gpus}")
    print(f"  Files:    drct={n_drct}  rest={n_rest}  mamba={n_mamba}")
    print(f"  Location: {out_dir}")
    print(f"\n  Next: train.py --cached will use this for full-image validation")
    print(f"{'=' * 70}\n")


if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)
    main()
