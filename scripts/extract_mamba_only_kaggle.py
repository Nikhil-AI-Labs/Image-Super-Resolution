"""
MambaIR-Only 5-Crop Feature Extractor — Kaggle Script
======================================================
Extracts ONLY MambaIR outputs + features for all 5 crops per image.
Designed to run on Kaggle (T4×2) where mamba_ssm is available.

DRCT + GRL + NAFNet are extracted separately by extract_drct_rest_local.py.

Why separate?
  MambaIR needs mamba-ssm / causal-conv1d → Kaggle GPU environment only.
  DRCT + GRL + NAFNet can run locally (no special CUDA deps).

5 Deterministic Crops per Image:
    p0 = Top-Left      p1 = Top-Right
    p2 = Bottom-Left   p3 = Bottom-Right
    p4 = Center
    → 3,450 images × 5 = 17,250 training samples

3-Notebook Split (Kaggle 20GB Limit):
    Notebook 1: --start-idx 0    --end-idx 1150
    Notebook 2: --start-idx 1150 --end-idx 2300
    Notebook 3: --start-idx 2300 --end-idx 3450

Output per crop (1 file):
    {stem}_p{0-4}_mamba_part.pt = {
        'outputs':  {'mamba': [1, 3, 256, 256]},   (FP32)
        'features': {'mamba': [1, 180, 64, 64]},   (FP32)
        'filename': str
    }

Resume-safe: skips any crop whose _mamba_part.pt already exists.

Kaggle Notebook Cell:
    !pip install mamba-ssm causal-conv1d einops timm
    !python scripts/extract_mamba_only_kaggle.py \\
        --dataset-dir /kaggle/input/sr-championship-df2k \\
        --output-dir  /kaggle/working/cache \\
        --mamba-weights /kaggle/input/your-weights/MambaIR_x4.pth \\
        --split train --start-idx 0 --end-idx 1150 --num-gpus 2
    !cd /kaggle/working && zip -r cache_mamba_part1.zip cache/

Author: NTIRE SR Team
"""

import os
import sys
import argparse
import glob
import time
import cv2
import numpy as np
import torch
import torch.multiprocessing as mp
from pathlib import Path
from tqdm.auto import tqdm
from typing import List, Tuple

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


# =============================================================================
# CLI
# =============================================================================

def parse_args():
    p = argparse.ArgumentParser(
        description='MambaIR-Only 5-Crop Extractor (Kaggle)'
    )
    p.add_argument('--dataset-dir', type=str, required=True,
                   help='Root dataset dir (e.g. /kaggle/input/sr-championship-df2k)')
    p.add_argument('--output-dir', type=str, default='/kaggle/working/cache',
                   help='Output directory for _mamba_part.pt files (must be writable)')
    p.add_argument('--split', type=str, default='train',
                   choices=['train', 'val', 'both'])
    p.add_argument('--start-idx', type=int, default=0,
                   help='Start image index (inclusive) for notebook splitting')
    p.add_argument('--end-idx', type=int, default=-1,
                   help='End image index (exclusive). -1 = all remaining')
    p.add_argument('--num-gpus', type=int, default=-1,
                   help='Number of GPUs (-1 = all available)')
    p.add_argument('--lr-patch-size', type=int, default=64)
    p.add_argument('--scale', type=int, default=4)
    p.add_argument('--mamba-weights', type=str,
                   default='./pretrained/mambair/MambaIR_x4.pth',
                   help='Path to MambaIR pretrained weights')
    p.add_argument('--resume', action='store_true',
                   help='Skip crops whose _mamba_part.pt already exists')
    return p.parse_args()


# =============================================================================
# Image Discovery — handles multi-part Kaggle datasets
# =============================================================================

def discover_image_pairs(
    dataset_dir: str,
    split: str,
) -> List[Tuple[Path, Path]]:
    """
    Discover LR-HR image pairs from the Kaggle dataset.

    Handles:
        train_HR_part01/train_HR/*.png  +  train_LR_part01/train_LR/*x4.png
        train_HR_part02/train_HR/*.png  +  train_LR_part02/train_LR/*x4.png
        train_HR/*.png  /  train_LR/*.png  (standard single-dir)
        DIV2K_train_HR/*.png  /  DIV2K_train_LR_bicubic/X4/*.png
    """
    root = Path(dataset_dir)
    extensions = ['*.png', '*.jpg', '*.jpeg', '*.bmp', '*.tif', '*.tiff']
    hr_dirs, lr_dirs = [], []

    if split == 'train':
        # Multi-part structure (Kaggle)
        for part_dir in sorted(root.glob('train_HR_part*')):
            inner = part_dir / 'train_HR'
            hr_dirs.append(inner if inner.exists() else part_dir)
        for part_dir in sorted(root.glob('train_LR_part*')):
            inner = part_dir / 'train_LR'
            lr_dirs.append(inner if inner.exists() else part_dir)

        # Standard single-directory fallbacks
        if not hr_dirs:
            for pat in [root / 'train_HR',
                        root / 'DIV2K_train_HR',
                        root / 'DF2K_train_HR']:
                if pat.exists():
                    hr_dirs.append(pat)
                    break
        if not lr_dirs:
            for pat in [root / 'train_LR',
                        root / 'DIV2K_train_LR_bicubic' / 'X4',
                        root / 'DF2K_train_LR_bicubic' / 'X4']:
                if pat.exists():
                    lr_dirs.append(pat)
                    break

    elif split == 'val':
        for pat in [root / 'val_HR', root / 'DIV2K_valid_HR']:
            if pat.exists():
                hr_dirs.append(pat)
                break
        for pat in [root / 'val_LR', root / 'DIV2K_valid_LR_bicubic' / 'X4']:
            if pat.exists():
                lr_dirs.append(pat)
                break

    # Collect all image files
    hr_files = sorted(
        f for d in hr_dirs for ext in extensions for f in d.glob(ext)
    )
    lr_files = sorted(
        f for d in lr_dirs for ext in extensions for f in d.glob(ext)
    )
    print(f"  Discovered {len(hr_files)} HR  |  {len(lr_files)} LR images")
    if hr_dirs:
        print(f"    HR dirs: {[str(d) for d in hr_dirs]}")
    if lr_dirs:
        print(f"    LR dirs: {[str(d) for d in lr_dirs]}")

    # Match by cleaned stem
    def clean_stem(s: str) -> str:
        for sfx in ['x4', 'x2', 'x3', 'x8',
                     '_LR', '_lr', 'LR', 'lr',
                     '_bicubic', '_BICUBIC']:
            s = s.replace(sfx, '')
        return s.rstrip('_')

    hr_dict = {p.stem: p for p in hr_files}
    lr_dict = {clean_stem(p.stem): p for p in lr_files}

    pairs = []
    for hr_stem, hr_path in sorted(hr_dict.items()):
        key = hr_stem if hr_stem in lr_dict else clean_stem(hr_stem)
        if key in lr_dict:
            pairs.append((lr_dict[key], hr_path))

    print(f"  Matched {len(pairs)} LR-HR pairs for {split}")
    return pairs


# =============================================================================
# 5 Deterministic Crop Positions
# =============================================================================

def compute_5_crops(lr_h, lr_w, patch_size, scale):
    """
    Compute 5 crop boxes: TL, TR, BL, BR, Center.
    Returns: list of (name, lr_box, hr_box) where box = (top, left, h, w).
    Skips entirely if image is smaller than patch_size in either dimension.
    """
    ps = patch_size
    hr_ps = ps * scale
    crops = []

    if lr_h < ps or lr_w < ps:
        return crops

    # p0: Top-Left
    crops.append(('p0', (0, 0, ps, ps), (0, 0, hr_ps, hr_ps)))

    # p1: Top-Right
    lr_left = lr_w - ps
    crops.append(('p1', (0, lr_left, ps, ps),
                        (0, lr_left * scale, hr_ps, hr_ps)))

    # p2: Bottom-Left
    lr_top = lr_h - ps
    crops.append(('p2', (lr_top, 0, ps, ps),
                        (lr_top * scale, 0, hr_ps, hr_ps)))

    # p3: Bottom-Right
    lr_top = lr_h - ps
    lr_left = lr_w - ps
    crops.append(('p3', (lr_top, lr_left, ps, ps),
                        (lr_top * scale, lr_left * scale, hr_ps, hr_ps)))

    # p4: Center
    lr_top = (lr_h - ps) // 2
    lr_left = (lr_w - ps) // 2
    crops.append(('p4', (lr_top, lr_left, ps, ps),
                        (lr_top * scale, lr_left * scale, hr_ps, hr_ps)))

    return crops


def extract_crop(img, box):
    """Extract crop from numpy image [H, W, C]."""
    top, left, h, w = box
    return img[top:top+h, left:left+w].copy()


def load_image(path):
    """Load image as float32 RGB [H, W, 3] in range [0, 1]."""
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Failed to load: {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0


def to_tensor(img):
    """Convert [H, W, C] numpy to [C, H, W] torch tensor."""
    return torch.from_numpy(img.transpose(2, 0, 1)).float().clamp(0, 1)


# =============================================================================
# MambaIR Loader
# =============================================================================

def load_mambair(weights_path: str, device: torch.device):
    """Load MambaIR model with pretrained weights."""
    try:
        from src.models.mambair.mambair_arch import MambaIR
    except ImportError as e:
        print(f"  ❌ Cannot import MambaIR: {e}")
        print("    Run: pip install mamba-ssm causal-conv1d einops timm")
        return None

    model = MambaIR(
        upscale=4, in_chans=3, img_size=64, window_size=16,
        compress_ratio=3, squeeze_factor=30, conv_scale=0.01,
        overlap_ratio=0.5, img_range=1.0,
        depths=(6, 6, 6, 6, 6, 6),
        embed_dim=180, mlp_ratio=2.0, drop_path_rate=0.1,
        upsampler='pixelshuffle', resi_connection='1conv',
    )

    if not os.path.exists(weights_path):
        print(f"  ❌ MambaIR weights not found: {weights_path}")
        return None

    print(f"  Loading MambaIR weights: {weights_path}")
    state = torch.load(weights_path, map_location='cpu', weights_only=False)

    # Handle different checkpoint formats
    if 'params' in state:
        sd = state['params']
    elif 'state_dict' in state:
        sd = state['state_dict']
    elif 'model' in state:
        sd = state['model']
    else:
        sd = state

    clean = {k.replace('module.', ''): v for k, v in sd.items()}
    missing, unexpected = model.load_state_dict(clean, strict=False)
    if missing:
        print(f"  ⚠ {len(missing)} missing keys")
    if unexpected:
        print(f"  ⚠ {len(unexpected)} unexpected keys")

    model.eval().to(device)
    params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"  ✓ MambaIR loaded ({params:.2f}M params) → {device}")
    return model


# =============================================================================
# Per-GPU Worker — MambaIR only
# =============================================================================

def worker_process(rank, num_gpus, all_pairs, output_dir, args):
    """Each GPU loads its own MambaIR and processes its shard."""
    device = torch.device(f'cuda:{rank}')

    if rank != 0:
        import io
        sys.stdout = io.StringIO()

    print(f"\n{'='*60}")
    print(f"  GPU {rank}: Loading MambaIR only...")
    print(f"{'='*60}")

    # Load MambaIR
    mamba_model = load_mambair(args.mamba_weights, device)
    if mamba_model is None:
        print(f"  ❌ GPU {rank}: MambaIR failed to load. Aborting worker.")
        if rank != 0:
            sys.stdout = sys.__stdout__
        return

    # Register hook on conv_after_body to capture deep features
    hook_cache = {}

    def hook_fn(module, inp, out):
        hook_cache['feat'] = out

    if hasattr(mamba_model, 'conv_after_body'):
        mamba_model.conv_after_body.register_forward_hook(hook_fn)
        print(f"  GPU {rank}: ✓ Hook registered on conv_after_body")
    else:
        print(f"  GPU {rank}: ⚠ conv_after_body not found — features will be zeros")

    # Shard dataset across GPUs
    my_pairs = all_pairs[rank::num_gpus]
    print(f"  GPU {rank}: {len(my_pairs)} images ({len(my_pairs) * 5} crops)")

    os.makedirs(output_dir, exist_ok=True)
    processed, skipped, errors = 0, 0, 0
    lh = lw = args.lr_patch_size
    t0 = time.time()

    iterator = (
        tqdm(my_pairs, desc=f'GPU {rank} (mamba)', ncols=110)
        if rank == 0 else my_pairs
    )

    for lr_path, hr_path in iterator:
        stem = hr_path.stem  # e.g. "0001" or "DIV2K_0001"

        # Resume: skip if ALL 5 mamba parts already exist
        if args.resume:
            if all(
                os.path.exists(
                    os.path.join(output_dir, f'{stem}_p{i}_mamba_part.pt')
                )
                for i in range(5)
            ):
                skipped += 1
                continue

        try:
            # Load LR image (only need LR for MambaIR forward pass)
            lr_img = load_image(lr_path)
            lr_h, lr_w = lr_img.shape[:2]

            # Compute 5 crop positions
            crops = compute_5_crops(lr_h, lr_w, args.lr_patch_size, args.scale)
            if not crops:
                if rank == 0:
                    print(f"\n  [SKIP] {stem}: too small ({lr_h}×{lr_w})")
                skipped += 1
                continue

            # Gather crops that still need extraction
            lr_patches, crop_names = [], []
            for crop_name, lr_box, _ in crops:
                out_path = os.path.join(
                    output_dir, f'{stem}_{crop_name}_mamba_part.pt'
                )
                if args.resume and os.path.exists(out_path):
                    continue
                lr_patches.append(to_tensor(extract_crop(lr_img, lr_box)))
                crop_names.append(crop_name)

            if not lr_patches:
                skipped += 1
                continue

            # Stack all crops into batch: [N_crops, 3, 64, 64]
            lr_batch = torch.stack(lr_patches).to(device)

            # Forward pass with FP16 autocast
            hook_cache.clear()
            with torch.no_grad(), torch.amp.autocast('cuda'):
                mamba_sr = mamba_model(lr_batch).clamp(0, 1)
            mamba_feat = hook_cache.get('feat')

            # Save one _mamba_part.pt per crop
            for k, crop_name in enumerate(crop_names):
                sr_k = mamba_sr[k:k+1].cpu().float()   # [1, 3, 256, 256]

                if mamba_feat is not None:
                    feat_k = mamba_feat[k:k+1].cpu().float()  # [1, 180, 64, 64]
                else:
                    feat_k = torch.zeros(1, 180, lh, lw)

                torch.save({
                    'outputs':  {'mamba': sr_k},
                    'features': {'mamba': feat_k},
                    'filename': f'{stem}_{crop_name}',
                }, os.path.join(output_dir, f'{stem}_{crop_name}_mamba_part.pt'))

                processed += 1

            # Periodic VRAM cleanup + progress update
            if processed % 200 == 0:
                torch.cuda.empty_cache()
                if rank == 0 and isinstance(iterator, tqdm):
                    elapsed = time.time() - t0
                    rate = processed / (elapsed + 1e-8)
                    total_remaining = (len(my_pairs) * 5) - processed
                    eta = total_remaining / (rate + 1e-8) / 60
                    iterator.set_postfix(
                        crops=processed, skip=skipped,
                        err=errors, ETA=f'{eta:.0f}m'
                    )

        except Exception as e:
            errors += 1
            if rank == 0:
                print(f"\n  [ERROR GPU {rank}] {stem}: {e}")

    # Restore stdout for summary
    if rank != 0:
        sys.stdout = sys.__stdout__

    elapsed = time.time() - t0
    print(f"\n  GPU {rank} done: {processed} crops in {elapsed:.1f}s "
          f"({elapsed/60:.1f} min)")
    print(f"    Skipped: {skipped} images | Errors: {errors}")
    if processed > 0:
        print(f"    Throughput: {processed/elapsed:.1f} crops/sec")


# =============================================================================
# Main
# =============================================================================

def main():
    args = parse_args()

    print("\n" + "=" * 70)
    print("  MAMBA-ONLY 5-CROP EXTRACTOR — KAGGLE")
    print("=" * 70)
    print(f"  Extracts:  _mamba_part.pt ONLY  (FP32)")
    print(f"  Assumes:   _drct_part.pt + _rest_part.pt extracted separately")
    print(f"  Range:     [{args.start_idx}:{args.end_idx}]")
    print(f"  Output:    {args.output_dir}")
    print("=" * 70 + "\n")

    # GPU detection
    avail = torch.cuda.device_count()
    num_gpus = avail if args.num_gpus == -1 else min(args.num_gpus, avail)
    if num_gpus < 1:
        print("ERROR: No CUDA GPUs found!")
        sys.exit(1)

    for i in range(num_gpus):
        name = torch.cuda.get_device_name(i)
        mem = torch.cuda.get_device_properties(i).total_memory / 1e9
        print(f"  GPU {i}: {name} ({mem:.1f} GB)")

    # Process each split
    splits = ['train', 'val'] if args.split == 'both' else [args.split]

    for split in splits:
        print(f"\n{'='*70}")
        print(f"  Processing {split.upper()} split")
        print(f"{'='*70}")

        pairs = discover_image_pairs(args.dataset_dir, split)
        if not pairs:
            print(f"  ❌ No image pairs found for {split}!")
            continue

        # Apply index range for notebook splitting
        end_idx = args.end_idx if args.end_idx > 0 else len(pairs)
        start_idx = max(0, min(args.start_idx, len(pairs)))
        end_idx = max(start_idx, min(end_idx, len(pairs)))
        selected = pairs[start_idx:end_idx]

        print(f"  Index range: [{start_idx}:{end_idx}] = "
              f"{len(selected)} images → {len(selected) * 5} crops")

        if not selected:
            print(f"  ❌ No images in range [{start_idx}:{end_idx}]!")
            continue

        output_dir = os.path.join(args.output_dir, f'cached_features_{split}')
        os.makedirs(output_dir, exist_ok=True)
        print(f"  Output: {output_dir}")

        t0 = time.time()
        if num_gpus > 1:
            mp.spawn(
                worker_process,
                args=(num_gpus, selected, output_dir, args),
                nprocs=num_gpus,
                join=True,
            )
        else:
            worker_process(0, 1, selected, output_dir, args)

        n_mamba = len(glob.glob(os.path.join(output_dir, '*_mamba_part.pt')))
        total_time = time.time() - t0
        print(f"\n  {split.upper()} DONE in {total_time/60:.1f} min")
        print(f"  _mamba_part.pt on disk: {n_mamba} "
              f"(expected: {len(selected) * 5})")

    # Final summary
    print("\n" + "=" * 70)
    print("  🎉 MAMBA EXTRACTION COMPLETE!")
    print("=" * 70)
    print(f"\n  Next steps:")
    print(f"    1. cd {args.output_dir} && zip -r cache_mamba_partN.zip cached_features_train/")
    print(f"    2. Save notebook → output becomes Kaggle Dataset")
    print(f"    3. Repeat for other index ranges (3-notebook split)")
    print(f"    4. Download and merge with _drct_part.pt + _rest_part.pt locally")
    print(f"    5. Train: python train.py --cached")
    print("=" * 70 + "\n")


if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)
    main()
