"""
DRCT + GRL + NAFNet 5-Crop Feature Extractor — Local/Kaggle Script
===================================================================
Extracts DRCT, GRL, NAFNet outputs + features for all 5 crops per image.
Does NOT load MambaIR — that runs separately via extract_mamba_only_kaggle.py.

Can run:
  - Locally on your GPU (P1000, etc.) — no mamba_ssm dependency
  - On Kaggle (T4×2) if you want to extract everything remotely

5 Deterministic Crops per Image:
    p0 = Top-Left      p1 = Top-Right
    p2 = Bottom-Left   p3 = Bottom-Right
    p4 = Center
    → 3,450 images × 5 = 17,250 training samples

Output per crop (2 files):
    {stem}_p{0-4}_drct_part.pt = {
        'outputs':  {'drct': [1, 3, 256, 256]},
        'features': {'drct': [1, 180, 64, 64]},
        'lr':       [3, 64, 64],
        'hr':       [3, 256, 256],
        'filename': str
    }
    {stem}_p{0-4}_rest_part.pt = {
        'outputs':  {'grl': [1, 3, 256, 256], 'nafnet': [1, 3, 256, 256]},
        'features': {'grl': [1, 180, 64, 64], 'nafnet': [1, 64, 64, 64]},
        'filename': str
    }

Resume-safe: checks _drct_part.pt as sentinel.

Usage (local):
    python scripts/extract_drct_rest_local.py \\
        --dataset-dir dataset/DF2K \\
        --output-dir  dataset/DF2K \\
        --config configs/train_config.yaml \\
        --split train --resume

Usage (Kaggle with notebook split):
    !python scripts/extract_drct_rest_local.py \\
        --dataset-dir /kaggle/input/sr-championship-df2k \\
        --output-dir  /kaggle/working/cache \\
        --config configs/train_config.yaml \\
        --split train --start-idx 0 --end-idx 1150 --num-gpus 2

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
from typing import Dict, List, Tuple

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


# =============================================================================
# CLI
# =============================================================================

def parse_args():
    p = argparse.ArgumentParser(
        description='DRCT+GRL+NAFNet 5-Crop Extractor (Local/Kaggle)'
    )
    p.add_argument('--dataset-dir', type=str, required=True,
                   help='Root dataset dir (e.g. dataset/DF2K)')
    p.add_argument('--output-dir', type=str, required=True,
                   help='Output root (cached_features_{split}/ created inside)')
    p.add_argument('--split', type=str, default='train',
                   choices=['train', 'val', 'both'])
    p.add_argument('--start-idx', type=int, default=0,
                   help='Start image index (inclusive)')
    p.add_argument('--end-idx', type=int, default=-1,
                   help='End image index (exclusive). -1 = all')
    p.add_argument('--num-gpus', type=int, default=-1,
                   help='Number of GPUs (-1 = all available)')
    p.add_argument('--lr-patch-size', type=int, default=64)
    p.add_argument('--scale', type=int, default=4)
    p.add_argument('--resume', action='store_true',
                   help='Skip crops whose _drct_part.pt already exists')
    p.add_argument('--config', type=str,
                   default='configs/train_config.yaml',
                   help='Path to train_config.yaml (for expert checkpoint paths)')
    return p.parse_args()


# =============================================================================
# Image Discovery — handles multi-part Kaggle datasets + local dirs
# =============================================================================

def discover_image_pairs(
    dataset_dir: str,
    split: str,
) -> List[Tuple[Path, Path]]:
    """
    Discover LR-HR image pairs from the dataset.

    Handles:
        train_HR_part01/train_HR/*.png  (Kaggle multi-part)
        train_HR/*.png                  (local standard)
        DIV2K_train_HR/*.png            (DIV2K standard)
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
# Per-GPU Worker — DRCT + GRL + NAFNet only (NO MambaIR)
# =============================================================================

def worker_process(rank, num_gpus, all_pairs, output_dir, args):
    """Each GPU loads DRCT+GRL+NAFNet and processes its shard."""
    device = torch.device(f'cuda:{rank}')

    if rank != 0:
        import io
        sys.stdout = io.StringIO()

    print(f"\n{'='*60}")
    print(f"  GPU {rank}: Loading DRCT + GRL + NAFNet  (no MambaIR)")
    print(f"{'='*60}")

    # Load expert ensemble
    from src.models.expert_loader import ExpertEnsemble
    import yaml

    config = {}
    if os.path.exists(args.config):
        with open(args.config) as f:
            config = yaml.safe_load(f)

    checkpoint_paths = {}
    for ec in config.get('model', {}).get('experts', []):
        name = ec.get('name', '').lower()
        weight = ec.get('weight_path', '')
        if name and weight:
            checkpoint_paths[name] = weight

    ensemble = ExpertEnsemble(device=device, upscale=args.scale)
    results = ensemble.load_all_experts(
        checkpoint_paths=checkpoint_paths, freeze=True
    )
    loaded = [k for k, v in results.items() if v]
    failed = [k for k, v in results.items() if not v]
    print(f"  GPU {rank}: Loaded: {loaded}")
    if failed:
        print(f"  GPU {rank}: Failed: {failed}")

    if not loaded:
        print(f"  ❌ GPU {rank}: No experts loaded! Check checkpoint paths.")
        if rank != 0:
            sys.stdout = sys.__stdout__
        return

    # Shard dataset across GPUs
    my_pairs = all_pairs[rank::num_gpus]
    print(f"  GPU {rank}: {len(my_pairs)} images ({len(my_pairs) * 5} crops)")

    os.makedirs(output_dir, exist_ok=True)
    processed, skipped, errors = 0, 0, 0
    lh = lw = args.lr_patch_size
    t0 = time.time()

    iterator = (
        tqdm(my_pairs, desc=f'GPU {rank} (drct+rest)', ncols=110)
        if rank == 0 else my_pairs
    )

    for lr_path, hr_path in iterator:
        stem = hr_path.stem

        # Resume: skip if ALL 5 drct parts already exist
        if args.resume:
            if all(
                os.path.exists(
                    os.path.join(output_dir, f'{stem}_p{i}_drct_part.pt')
                )
                for i in range(5)
            ):
                skipped += 1
                continue

        try:
            # Load raw images
            lr_img = load_image(lr_path)   # [H_lr, W_lr, 3]
            hr_img = load_image(hr_path)   # [H_hr, W_hr, 3]
            lr_h, lr_w = lr_img.shape[:2]
            hr_h, hr_w = hr_img.shape[:2]

            # Verify scale relationship
            exp_h = lr_h * args.scale
            exp_w = lr_w * args.scale
            if hr_h != exp_h or hr_w != exp_w:
                hr_img = cv2.resize(
                    hr_img, (exp_w, exp_h),
                    interpolation=cv2.INTER_CUBIC
                )

            # Compute 5 crop positions
            crops = compute_5_crops(lr_h, lr_w, args.lr_patch_size, args.scale)
            if not crops:
                if rank == 0:
                    print(f"\n  [SKIP] {stem}: too small ({lr_h}×{lr_w})")
                skipped += 1
                continue

            # Gather crops that still need extraction
            lr_patches, hr_patches, crop_names = [], [], []
            for crop_name, lr_box, hr_box in crops:
                drct_out_path = os.path.join(
                    output_dir, f'{stem}_{crop_name}_drct_part.pt'
                )
                if args.resume and os.path.exists(drct_out_path):
                    continue
                lr_patches.append(to_tensor(extract_crop(lr_img, lr_box)))
                hr_patches.append(to_tensor(extract_crop(hr_img, hr_box)))
                crop_names.append(crop_name)

            if not lr_patches:
                skipped += 1
                continue

            # Stack into batch: [N_crops, 3, 64, 64]
            lr_batch = torch.stack(lr_patches).to(device)
            hr_batch_cpu = torch.stack(hr_patches)  # stays on CPU for saving

            # Forward pass through DRCT + GRL + NAFNet (with hooks)
            with torch.no_grad():
                local_outputs, local_features = \
                    ensemble.forward_all_with_hooks(lr_batch)

            # Pre-fetch tensors (avoids repeated dict lookups in loop)
            drct_out = local_outputs.get(
                'drct', torch.zeros(lr_batch.shape[0], 3, lh*4, lw*4)
            )
            drct_feat = local_features.get(
                'drct', torch.zeros(lr_batch.shape[0], 180, lh, lw)
            )
            grl_out = local_outputs.get(
                'grl', torch.zeros(lr_batch.shape[0], 3, lh*4, lw*4)
            )
            grl_feat = local_features.get(
                'grl', torch.zeros(lr_batch.shape[0], 180, lh, lw)
            )
            naf_out = local_outputs.get(
                'nafnet', torch.zeros(lr_batch.shape[0], 3, lh*4, lw*4)
            )
            naf_feat = local_features.get(
                'nafnet', torch.zeros(lr_batch.shape[0], 64, lh, lw)
            )

            # Save 2 .pt files per crop
            for k, crop_name in enumerate(crop_names):
                crop_stem = f'{stem}_{crop_name}'

                # --- drct_part (also carries lr + hr for CachedSRDataset) ---
                torch.save({
                    'outputs':  {'drct': drct_out[k:k+1].cpu()},
                    'features': {'drct': drct_feat[k:k+1].cpu()},
                    'lr':       lr_patches[k],        # [3, 64, 64] CPU
                    'hr':       hr_batch_cpu[k],      # [3, 256, 256] CPU
                    'filename': crop_stem,
                }, os.path.join(output_dir, f'{crop_stem}_drct_part.pt'))

                # --- rest_part (GRL + NAFNet) ---
                torch.save({
                    'outputs': {
                        'grl':    grl_out[k:k+1].cpu(),
                        'nafnet': naf_out[k:k+1].cpu(),
                    },
                    'features': {
                        'grl':    grl_feat[k:k+1].cpu(),
                        'nafnet': naf_feat[k:k+1].cpu(),
                    },
                    'filename': crop_stem,
                }, os.path.join(output_dir, f'{crop_stem}_rest_part.pt'))

                processed += 1

            # Periodic VRAM cleanup + progress
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
    print("  DRCT + GRL + NAFNet 5-CROP EXTRACTOR")
    print("=" * 70)
    print(f"  Extracts:  _drct_part.pt + _rest_part.pt  (FP32)")
    print(f"  Skips:     MambaIR — run extract_mamba_only_kaggle.py on Kaggle")
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

        # Apply index range
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

        # Count output files
        n_drct = len(glob.glob(os.path.join(output_dir, '*_drct_part.pt')))
        n_rest = len(glob.glob(os.path.join(output_dir, '*_rest_part.pt')))
        total_time = time.time() - t0

        print(f"\n  {split.upper()} DONE in {total_time/60:.1f} min")
        print(f"  _drct_part.pt: {n_drct}  |  _rest_part.pt: {n_rest}  "
              f"(expected: {len(selected) * 5} each)")
        if n_drct != n_rest:
            print(f"  ⚠ Count mismatch! Run again with --resume to fix.")

    # Final summary
    print("\n" + "=" * 70)
    print("  🎉 DRCT + REST EXTRACTION COMPLETE!")
    print("=" * 70)
    print(f"\n  Next steps:")
    print(f"    1. Run extract_mamba_only_kaggle.py on Kaggle for MambaIR")
    print(f"    2. Merge: cp /downloaded/*_mamba_part.pt {args.output_dir}/cached_features_train/")
    print(f"    3. Verify all 3 file types present per crop")
    print(f"    4. Train: python train.py --cached")
    print("=" * 70 + "\n")


if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)
    main()
