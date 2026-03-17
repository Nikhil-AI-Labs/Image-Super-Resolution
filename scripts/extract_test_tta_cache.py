"""
scripts/extract_test_tta_cache.py
==================================
Extracts 8x geometric TTA variants for DIV2K Test set (NO ground truth HR).
All 4 experts (DRCT, GRL, NAFNet, MambaIR) per variant.

MULTI-GPU SUPPORT: Automatically detects and uses all available GPUs.
  - On Kaggle 2×T4: GPU 0 processes images [0,2,4,...], GPU 1 processes [1,3,5,...]
  - Falls back to single GPU if only 1 is available.

OOM-SAFE: Experts run sequentially per image with CUDA cache cleared between each.

8 Geometric Variants per Image:
    t0 = original           t4 = hflip
    t1 = rot90              t5 = hflip + rot90
    t2 = rot180             t6 = hflip + rot180
    t3 = rot270             t7 = hflip + rot270

Output per variant (3 files):
    {stem}_t{0-7}_drct_part.pt  (includes lr + tta_info)
    {stem}_t{0-7}_rest_part.pt
    {stem}_t{0-7}_mamba_part.pt

Total: 100 test images × 8 variants × 3 files = 2,400 .pt files
Resume-safe: skips variants whose _drct_part.pt already exists.

Usage:
    # Kaggle 2×T4 (auto-detect multi-GPU)
    !python scripts/extract_test_tta_cache.py \\
        --test-dir /kaggle/input/DIV2K_test_LR_bicubic/X4 \\
        --output-dir /kaggle/working/test_tta_cache \\
        --mamba-weights /kaggle/input/mambair-weights/MambaIR_x4.pth \\
        --resume

    # Chunked extraction (Kaggle notebook 1 of 2):
    !python scripts/extract_test_tta_cache.py \\
        --test-dir /kaggle/input/DIV2K_test_LR_bicubic/X4 \\
        --output-dir /kaggle/working/test_tta_cache \\
        --start-idx 0 --end-idx 50 --resume

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
    """Pad tensor spatial dims to multiple of 16."""
    _, _, h, w = t.shape
    ph = (16 - h % 16) % 16
    pw = (16 - w % 16) % 16
    if ph or pw:
        t = F.pad(t, (0, pw, 0, ph), mode='reflect')
    return t, (ph, pw)


def unpad(t, ph, pw, scale=1):
    """Remove padding from tensor."""
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


def apply_tta_transform(tensor, hflip, rot):
    """Apply geometric TTA transform: hflip then rotation."""
    if hflip:
        tensor = torch.flip(tensor, [3])
    if rot > 0:
        tensor = torch.rot90(tensor, rot, [2, 3])
    return tensor


# =============================================================================
# Per-Expert Sequential Forward (OOM-Safe)
# =============================================================================

def run_expert_sequential(ensemble, mamba, mamba_feat_cache, lr_padded, orig_h, orig_w, device):
    """
    Run all 4 experts SEQUENTIALLY on one image, freeing memory between each.
    Returns (outputs_dict, features_dict) with all tensors on CPU as FP16.
    Features are cropped/resized to original LR resolution.
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
            feat = ensemble._captured_features['drct'].clone()
            # Crop transformer features to original LR resolution
            features['drct'] = feat[:, :, :orig_h, :orig_w].cpu().half()
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
            feat = ensemble._captured_features['grl'].clone()
            features['grl'] = feat[:, :, :orig_h, :orig_w].cpu().half()
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
            feat = ensemble._captured_features['nafnet'].clone()
            # NAFNet features are at upscaled resolution → resize to LR
            features['nafnet'] = F.interpolate(
                feat, size=(orig_h, orig_w),
                mode='bilinear', align_corners=False
            ).cpu().half()
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
        feat = mamba_feat_cache['feat'].clone()
        features['mamba'] = feat[:, :, :orig_h, :orig_w].cpu().half()
    del mamba_sr
    torch.cuda.empty_cache()

    return outputs, features


# =============================================================================
# GPU Worker
# =============================================================================

def gpu_worker(rank, num_gpus, test_files, args):
    """Worker function that runs on a single GPU."""
    device = torch.device(f'cuda:{rank}')

    # Shard images across GPUs (interleaving)
    my_files = test_files[rank::num_gpus]

    is_primary = (rank == 0)
    if not is_primary:
        sys.stdout = open(os.devnull, 'w')

    if is_primary:
        print(f"\n  [GPU {rank}] Loading experts...")
        print(f"  Total GPUs: {num_gpus}")
        print(f"  Images per GPU: ~{len(my_files)}")
        print(f"  Variants per GPU: ~{len(my_files) * 8}")

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

    mamba_feat_cache = {}
    if hasattr(mamba, 'conv_after_body'):
        mamba.conv_after_body.register_forward_hook(
            lambda m, i, o: mamba_feat_cache.update({'feat': o})
        )

    if is_primary:
        print(f"  [GPU {rank}] Experts loaded. Starting 8x TTA extraction...")

    # ── 8 TTA Configs ──
    tta_configs = []
    for hflip in [False, True]:
        for rot in [0, 1, 2, 3]:
            tta_configs.append((hflip, rot))

    # ── Process images ──
    iterator = tqdm(my_files, desc=f'GPU {rank}', ncols=100) if is_primary else my_files
    processed, skipped, errors = 0, 0, 0

    with torch.no_grad():
        for lr_path in iterator:
            raw_stem = Path(lr_path).stem
            stem = raw_stem.replace('x4', '').rstrip('_')

            base_lr = load_img(lr_path).to(device)

            for t_idx, (hflip, rot) in enumerate(tta_configs):
                t_stem = f"{stem}_t{t_idx}"

                # Resume check
                drct_out = os.path.join(args.output_dir, f'{t_stem}_drct_part.pt')
                if args.resume and os.path.exists(drct_out):
                    skipped += 1
                    continue

                try:
                    # Apply geometric transform
                    lr_variant = apply_tta_transform(base_lr, hflip, rot)
                    orig_h, orig_w = lr_variant.shape[2], lr_variant.shape[3]

                    # Pad to multiple of 16
                    lr_padded, (ph, pw) = pad16(lr_variant)

                    # Run all 4 experts sequentially (OOM-safe)
                    outs, feats = run_expert_sequential(
                        ensemble, mamba, mamba_feat_cache,
                        lr_padded, orig_h, orig_w, device
                    )

                    # Helper to unpad SR outputs
                    def u(tensor, scale):
                        if tensor is None:
                            return None
                        return unpad(tensor, ph, pw, scale)

                    # Save drct_part (includes lr + tta_info)
                    torch.save({
                        'outputs':  {'drct': u(outs.get('drct'), 4)},
                        'features': {'drct': feats.get('drct')},
                        'lr': lr_variant.cpu().squeeze(0).half(),
                        'filename': t_stem,
                        'original_stem': stem,
                        'original_size': (orig_h, orig_w),
                        'tta_info': {'hflip': hflip, 'rot': rot, 't_idx': t_idx},
                    }, drct_out)

                    # Save rest_part
                    torch.save({
                        'outputs': {
                            'grl':    u(outs.get('grl'), 4),
                            'nafnet': u(outs.get('nafnet'), 4),
                        },
                        'features': {
                            'grl':    feats.get('grl'),
                            'nafnet': feats.get('nafnet'),
                        },
                        'filename': t_stem,
                    }, os.path.join(args.output_dir, f'{t_stem}_rest_part.pt'))

                    # Save mamba_part
                    torch.save({
                        'outputs':  {'mamba': u(outs.get('mamba'), 4)},
                        'features': {'mamba': feats.get('mamba')},
                        'filename': t_stem,
                    }, os.path.join(args.output_dir, f'{t_stem}_mamba_part.pt'))

                    processed += 1
                    del outs, feats
                    torch.cuda.empty_cache()

                except Exception as e:
                    errors += 1
                    if is_primary:
                        tqdm.write(f"  [ERROR] {t_stem}: {e}")

            # Clear base image after all 8 variants
            del base_lr
            torch.cuda.empty_cache()

    if is_primary:
        print(f"\n  [GPU {rank}] Done: {processed} variants, "
              f"{skipped} skipped, {errors} errors")


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Multi-GPU Test Set 8x TTA Feature Cache Extractor'
    )
    parser.add_argument('--test-dir', required=True,
                        help='Path to test LR images (e.g. DIV2K_test_LR_bicubic/X4)')
    parser.add_argument('--output-dir', required=True,
                        help='Output dir for TTA cache')
    parser.add_argument('--config', default='configs/train_config.yaml')
    parser.add_argument('--mamba-weights',
                        default='pretrained/mambair/MambaIR_x4.pth')
    parser.add_argument('--resume', action='store_true',
                        help='Skip variants whose _drct_part.pt already exists')
    parser.add_argument('--start-idx', type=int, default=0,
                        help='Start image index (for chunked extraction)')
    parser.add_argument('--end-idx', type=int, default=-1,
                        help='End image index (-1 = all)')
    parser.add_argument('--num-gpus', type=int, default=None,
                        help='Number of GPUs (default: auto-detect)')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Discover test images
    test_files = sorted(
        glob.glob(os.path.join(args.test_dir, '*.png')) +
        glob.glob(os.path.join(args.test_dir, '*.jpg'))
    )

    # Apply index range for chunked extraction
    end_idx = args.end_idx if args.end_idx > 0 else len(test_files)
    test_files = test_files[args.start_idx:end_idx]

    num_gpus = args.num_gpus or torch.cuda.device_count()
    num_gpus = max(1, num_gpus)

    print("\n" + "=" * 70)
    print("  MULTI-GPU TEST SET 8x TTA CACHE EXTRACTOR")
    print("=" * 70)
    print(f"  Test dir:     {args.test_dir}")
    print(f"  Output:       {args.output_dir}")
    print(f"  GPUs:         {num_gpus}")
    print(f"  Images:       {len(test_files)} (idx {args.start_idx}:{end_idx})")
    print(f"  TTA variants: 8 per image")
    print(f"  Total:        {len(test_files) * 8} variants = "
          f"{len(test_files) * 8 * 3} .pt files")
    print(f"  Resume:       {args.resume}")
    print(f"  Storage:      FP16")
    print("=" * 70 + "\n")

    if not test_files:
        print("  ✗ No test images found!")
        sys.exit(1)

    t0 = time.time()

    if num_gpus > 1:
        print(f"  Launching {num_gpus} GPU workers...")
        mp.spawn(
            gpu_worker,
            args=(num_gpus, test_files, args),
            nprocs=num_gpus,
            join=True,
        )
    else:
        print("  Running single-GPU extraction...")
        gpu_worker(0, 1, test_files, args)

    elapsed = time.time() - t0
    n_drct = len(glob.glob(os.path.join(args.output_dir, '*_drct_part.pt')))
    n_rest = len(glob.glob(os.path.join(args.output_dir, '*_rest_part.pt')))
    n_mamba = len(glob.glob(os.path.join(args.output_dir, '*_mamba_part.pt')))

    print(f"\n{'=' * 70}")
    print(f"  TEST TTA CACHE COMPLETE")
    print(f"{'=' * 70}")
    print(f"  Time:     {elapsed:.1f}s ({elapsed/60:.1f} min)")
    print(f"  GPUs:     {num_gpus}")
    print(f"  Files:    drct={n_drct}  rest={n_rest}  mamba={n_mamba}")
    print(f"  Expected: {len(test_files) * 8} each")
    print(f"  Location: {args.output_dir}")
    print(f"\n  Next: python scripts/generate_fast_submission.py \\")
    print(f"          --test-cache-dir {args.output_dir} \\")
    print(f"          --checkpoint checkpoints/best.pth")
    print(f"{'=' * 70}\n")


if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)
    main()
