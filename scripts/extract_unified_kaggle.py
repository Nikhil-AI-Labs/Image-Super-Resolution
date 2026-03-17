"""
Unified 5-Crop Feature Extraction Script — Kaggle Master Extractor
====================================================================
Replaces extract_features_multi_gpu.py + extract_mamba_features.py.

Loads raw DF2K images, cuts 5 deterministic 64×64 crops per image,
runs all 4 experts (DRCT, GRL, NAFNet, MambaIR) in one GPU pass,
and writes 3 .pt files per crop — fully compatible with CachedSRDataset.

5 Crops per Image:
    p0 = Top-Left      p1 = Top-Right      p2 = Bottom-Left
    p3 = Bottom-Right   p4 = Center
    → 3,450 images × 5 = 17,250 training samples!

Kaggle 3-Notebook Split Strategy:
    Kaggle's /kaggle/working/ is limited to 20GB.
    Use --start-idx / --end-idx to split across 3 notebooks:
        Notebook 1: --start-idx 0    --end-idx 1150   → cache_part1.zip
        Notebook 2: --start-idx 1150 --end-idx 2300   → cache_part2.zip
        Notebook 3: --start-idx 2300 --end-idx 3450   → cache_part3.zip
    Each generates ~15GB, fitting under the 20GB limit.

Output per crop (3 files):
    {stem}_p{0-4}_drct_part.pt  →  {'outputs': {'drct': ...},
                                     'features': {'drct': ...},
                                     'lr': [3,64,64], 'hr': [3,256,256],
                                     'filename': str}

    {stem}_p{0-4}_rest_part.pt  →  {'outputs': {'grl': ..., 'nafnet': ...},
                                     'features': {'grl': ..., 'nafnet': ...},
                                     'filename': str}

    {stem}_p{0-4}_mamba_part.pt →  {'outputs': {'mamba': ...},      (FP16)
                                     'features': {'mamba': ...},     (FP16)
                                     'filename': str}

CachedSRDataset globs *_drct_part.pt — these appear as 17,250 unique samples.
Zero downstream code changes required.

Usage (single notebook cell):
    !pip install mamba-ssm causal-conv1d einops timm
    !python scripts/extract_unified_kaggle.py \\
        --dataset-dir /kaggle/input/sr-championship-df2k \\
        --output-dir /kaggle/working/cache \\
        --split train --start-idx 0 --end-idx 1150 --num-gpus 2
    !cd /kaggle/working && zip -r cache_part1.zip cache/

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
import torch.nn.functional as F
import torch.multiprocessing as mp
from pathlib import Path
from tqdm.auto import tqdm
from typing import Dict, List, Tuple, Optional

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


# =============================================================================
# CLI Arguments
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description='Unified 5-Crop Feature Extraction (Kaggle Master Script)'
    )
    parser.add_argument(
        '--dataset-dir', type=str, required=True,
        help='Root dataset dir (e.g. /kaggle/input/sr-championship-df2k)'
    )
    parser.add_argument(
        '--output-dir', type=str, default='/kaggle/working/cache',
        help='Output directory for .pt files (must be writable)'
    )
    parser.add_argument(
        '--split', type=str, default='train',
        choices=['train', 'val', 'both'],
        help='Which split to extract'
    )
    parser.add_argument(
        '--start-idx', type=int, default=0,
        help='Start image index (inclusive) for notebook splitting'
    )
    parser.add_argument(
        '--end-idx', type=int, default=-1,
        help='End image index (exclusive). -1 = all remaining'
    )
    parser.add_argument(
        '--num-gpus', type=int, default=-1,
        help='Number of GPUs to use (-1 = all available)'
    )
    parser.add_argument(
        '--lr-patch-size', type=int, default=64,
        help='LR patch size (64 recommended)'
    )
    parser.add_argument(
        '--scale', type=int, default=4,
        help='Upscale factor'
    )
    parser.add_argument(
        '--mamba-weights', type=str,
        default='./pretrained/mambair/MambaIR_x4.pth',
        help='Path to MambaIR pretrained weights'
    )
    parser.add_argument(
        '--resume', action='store_true',
        help='Skip images whose _drct_part.pt already exists'
    )
    parser.add_argument(
        '--config', type=str, default='configs/train_config.yaml',
        help='Path to train_config.yaml (for expert checkpoint paths)'
    )
    return parser.parse_args()


# =============================================================================
# Image Discovery — handles multi-part Kaggle datasets
# =============================================================================

def discover_image_pairs(
    dataset_dir: str,
    split: str,
) -> List[Tuple[Path, Path]]:
    """
    Discover LR-HR image pairs from the Kaggle dataset.

    Handles multi-part directory structures like:
        train_HR_part01/train_HR/*.png
        train_HR_part02/train_HR/*.png
        train_LR_part01/train_LR/*x4.png
        train_LR_part02/train_LR/*x4.png

    Also handles standard structures:
        train_HR/*.png  /  train_LR/*.png
        DIV2K_train_HR/*.png  /  DIV2K_train_LR_bicubic/X4/*.png
        val_HR/*.png  /  val_LR/*.png
    """
    root = Path(dataset_dir)
    extensions = ['*.png', '*.jpg', '*.jpeg', '*.bmp', '*.tif', '*.tiff']

    # Collect HR images from all possible locations
    hr_dirs = []
    lr_dirs = []

    if split == 'train':
        # Multi-part structure (Kaggle)
        for part_dir in sorted(root.glob('train_HR_part*')):
            inner = part_dir / 'train_HR'
            if inner.exists():
                hr_dirs.append(inner)
            elif any(part_dir.glob('*.png')):
                hr_dirs.append(part_dir)

        for part_dir in sorted(root.glob('train_LR_part*')):
            inner = part_dir / 'train_LR'
            if inner.exists():
                lr_dirs.append(inner)
            elif any(part_dir.glob('*.png')):
                lr_dirs.append(part_dir)

        # Standard single-directory structure
        if not hr_dirs:
            for pattern in [
                root / 'train_HR',
                root / 'DIV2K_train_HR',
                root / 'DF2K_train_HR',
            ]:
                if pattern.exists():
                    hr_dirs.append(pattern)
                    break

        if not lr_dirs:
            for pattern in [
                root / 'train_LR',
                root / 'DIV2K_train_LR_bicubic' / 'X4',
                root / 'DF2K_train_LR_bicubic' / 'X4',
            ]:
                if pattern.exists():
                    lr_dirs.append(pattern)
                    break

    elif split == 'val':
        for pattern in [
            root / 'val_HR',
            root / 'DIV2K_valid_HR',
        ]:
            if pattern.exists():
                hr_dirs.append(pattern)
                break

        for pattern in [
            root / 'val_LR',
            root / 'DIV2K_valid_LR_bicubic' / 'X4',
        ]:
            if pattern.exists():
                lr_dirs.append(pattern)
                break

    # Gather all image files
    hr_files = []
    for d in hr_dirs:
        for ext in extensions:
            hr_files.extend(d.glob(ext))
    hr_files = sorted(hr_files)

    lr_files = []
    for d in lr_dirs:
        for ext in extensions:
            lr_files.extend(d.glob(ext))
    lr_files = sorted(lr_files)

    print(f"  Discovered {len(hr_files)} HR images from {len(hr_dirs)} directories")
    print(f"  Discovered {len(lr_files)} LR images from {len(lr_dirs)} directories")

    # Build lookup: clean stem → path
    def clean_stem(stem: str) -> str:
        """Remove common suffixes like x4, _LR, etc."""
        for suffix in ['x4', 'x2', 'x3', 'x8']:
            stem = stem.replace(suffix, '')
        for suffix in ['_LR', '_lr', 'LR', 'lr', '_bicubic', '_BICUBIC']:
            stem = stem.replace(suffix, '')
        return stem.rstrip('_')

    hr_dict = {p.stem: p for p in hr_files}
    lr_dict = {}
    for p in lr_files:
        cleaned = clean_stem(p.stem)
        lr_dict[cleaned] = p

    # Match pairs
    pairs = []
    for hr_stem, hr_path in sorted(hr_dict.items()):
        # Try direct match first, then cleaned match
        if hr_stem in lr_dict:
            pairs.append((lr_dict[hr_stem], hr_path))
        else:
            cleaned_hr = clean_stem(hr_stem)
            if cleaned_hr in lr_dict:
                pairs.append((lr_dict[cleaned_hr], hr_path))

    print(f"  Matched {len(pairs)} LR-HR pairs for {split}")
    return pairs


# =============================================================================
# 5 Deterministic Crop Positions
# =============================================================================

def compute_5_crops(
    lr_h: int, lr_w: int,
    patch_size: int,
    scale: int,
) -> List[Tuple[str, Tuple[int, int, int, int], Tuple[int, int, int, int]]]:
    """
    Compute 5 deterministic crop positions for LR and HR images.

    Returns list of (crop_name, lr_crop_box, hr_crop_box)
    where crop_box = (top, left, height, width).

    Skips crops that don't fit (image too small).
    """
    ps = patch_size
    hr_ps = ps * scale
    crops = []

    if lr_h < ps or lr_w < ps:
        return crops  # Image too small for any crop

    # p0: Top-Left
    crops.append(('p0', (0, 0, ps, ps), (0, 0, hr_ps, hr_ps)))

    # p1: Top-Right
    if lr_w >= ps:
        lr_left = lr_w - ps
        hr_left = lr_left * scale
        crops.append(('p1', (0, lr_left, ps, ps), (0, hr_left, hr_ps, hr_ps)))

    # p2: Bottom-Left
    if lr_h >= ps:
        lr_top = lr_h - ps
        hr_top = lr_top * scale
        crops.append(('p2', (lr_top, 0, ps, ps), (hr_top, 0, hr_ps, hr_ps)))

    # p3: Bottom-Right
    if lr_h >= ps and lr_w >= ps:
        lr_top = lr_h - ps
        lr_left = lr_w - ps
        hr_top = lr_top * scale
        hr_left = lr_left * scale
        crops.append(('p3', (lr_top, lr_left, ps, ps), (hr_top, hr_left, hr_ps, hr_ps)))

    # p4: Center
    lr_top = (lr_h - ps) // 2
    lr_left = (lr_w - ps) // 2
    hr_top = lr_top * scale
    hr_left = lr_left * scale
    crops.append(('p4', (lr_top, lr_left, ps, ps), (hr_top, hr_left, hr_ps, hr_ps)))

    return crops


def extract_crop(img: np.ndarray, box: Tuple[int, int, int, int]) -> np.ndarray:
    """Extract a crop from numpy image [H, W, C]."""
    top, left, h, w = box
    return img[top:top+h, left:left+w].copy()


# =============================================================================
# Load image utilities
# =============================================================================

def load_image(path: Path) -> np.ndarray:
    """Load image as float32 RGB [H, W, 3] in range [0, 1]."""
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Failed to load image: {path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img.astype(np.float32) / 255.0


def numpy_to_tensor(img: np.ndarray) -> torch.Tensor:
    """Convert [H, W, C] float32 numpy to [C, H, W] torch tensor."""
    return torch.from_numpy(img.transpose(2, 0, 1)).float().clamp(0, 1)


# =============================================================================
# MambaIR Loader (separate from ExpertEnsemble)
# =============================================================================

def load_mambair(weights_path: str, device: torch.device):
    """Load MambaIR model with pretrained weights."""
    try:
        from src.models.mambair.mambair_arch import MambaIR
    except ImportError as e:
        print(f"  ⚠ Cannot import MambaIR: {e}")
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
        print(f"  ⚠ MambaIR weights not found: {weights_path}")
        return None

    state = torch.load(weights_path, map_location='cpu', weights_only=False)
    if 'params' in state:
        state_dict = state['params']
    elif 'state_dict' in state:
        state_dict = state['state_dict']
    elif 'model' in state:
        state_dict = state['model']
    else:
        state_dict = state

    clean_state = {k.replace('module.', ''): v for k, v in state_dict.items()}
    model.load_state_dict(clean_state, strict=False)
    model.eval().to(device)

    params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"  ✓ MambaIR loaded ({params:.2f}M params) → {device}")
    return model


# =============================================================================
# Per-GPU Worker
# =============================================================================

def worker_process(
    rank: int,
    num_gpus: int,
    all_pairs: List[Tuple[Path, Path]],
    output_dir: str,
    args,
):
    """
    Worker function for each GPU.
    Loads all 4 experts, processes its shard of images,
    generates 5 crops per image, saves 3 .pt files per crop.
    """
    device = torch.device(f'cuda:{rank}')

    # Suppress prints from non-rank-0 workers
    if rank != 0:
        import io
        sys.stdout = io.StringIO()

    print(f"\n{'='*70}")
    print(f"  GPU {rank}: Loading all 4 experts...")
    print(f"{'='*70}")

    # ── Load 3 local experts (DRCT + GRL + NAFNet) ──────────────────────
    from src.models.expert_loader import ExpertEnsemble
    import yaml

    config = {}
    if os.path.exists(args.config):
        with open(args.config) as f:
            config = yaml.safe_load(f)

    expert_configs = config.get('model', {}).get('experts', [])
    checkpoint_paths = {}
    for ec in expert_configs:
        name = ec.get('name', '').lower()
        weight = ec.get('weight_path', '')
        if name and weight:
            checkpoint_paths[name] = weight

    ensemble = ExpertEnsemble(device=device, upscale=args.scale)
    load_results = ensemble.load_all_experts(
        checkpoint_paths=checkpoint_paths,
        freeze=True,
    )

    loaded_local = [k for k, v in load_results.items() if v]
    print(f"  GPU {rank}: Local experts loaded: {loaded_local}")

    # ── Load MambaIR ────────────────────────────────────────────────────
    mamba_model = load_mambair(args.mamba_weights, device)
    mamba_hook_cache = {}

    if mamba_model is not None:
        def mamba_hook_fn(module, inp, out):
            mamba_hook_cache['feat'] = out
        if hasattr(mamba_model, 'conv_after_body'):
            mamba_model.conv_after_body.register_forward_hook(mamba_hook_fn)
            print(f"  GPU {rank}: ✓ MambaIR hook registered on conv_after_body")
        else:
            print(f"  GPU {rank}: ⚠ MambaIR conv_after_body not found!")

    # ── Shard the dataset across GPUs ───────────────────────────────────
    my_pairs = all_pairs[rank::num_gpus]
    print(f"\n  GPU {rank}: Processing {len(my_pairs)} images "
          f"({len(my_pairs) * 5} crops) out of {len(all_pairs)} total")

    # ── Extraction loop ─────────────────────────────────────────────────
    os.makedirs(output_dir, exist_ok=True)

    processed_crops = 0
    skipped_images = 0
    errors = 0
    t0 = time.time()

    iterator = tqdm(my_pairs, desc=f'GPU {rank}', ncols=110) if rank == 0 else my_pairs

    for lr_path, hr_path in iterator:
        stem = hr_path.stem  # e.g. "0001" or "DIV2K_0001"

        # Resume check: skip if all 5 drct parts exist
        if args.resume:
            all_exist = all(
                os.path.exists(os.path.join(output_dir, f"{stem}_p{i}_drct_part.pt"))
                for i in range(5)
            )
            if all_exist:
                skipped_images += 1
                continue

        try:
            # Load raw images
            lr_img = load_image(lr_path)  # [H_lr, W_lr, 3]
            hr_img = load_image(hr_path)  # [H_hr, W_hr, 3]
            lr_h, lr_w = lr_img.shape[:2]
            hr_h, hr_w = hr_img.shape[:2]

            # Verify scale relationship
            expected_hr_h = lr_h * args.scale
            expected_hr_w = lr_w * args.scale
            if hr_h != expected_hr_h or hr_w != expected_hr_w:
                hr_img = cv2.resize(
                    hr_img, (expected_hr_w, expected_hr_h),
                    interpolation=cv2.INTER_CUBIC
                )

            # Compute 5 deterministic crops
            crops = compute_5_crops(lr_h, lr_w, args.lr_patch_size, args.scale)
            if not crops:
                if rank == 0:
                    print(f"\n  [SKIP] {stem}: too small ({lr_h}×{lr_w})")
                continue

            # ── Stack all crops into a batch for maximum GPU efficiency ──
            lr_patches = []
            hr_patches = []
            crop_names = []

            for crop_name, lr_box, hr_box in crops:
                # Skip if this specific crop already exists (fine-grained resume)
                drct_path = os.path.join(output_dir, f"{stem}_{crop_name}_drct_part.pt")
                if args.resume and os.path.exists(drct_path):
                    continue

                lr_patch = extract_crop(lr_img, lr_box)
                hr_patch = extract_crop(hr_img, hr_box)
                lr_patches.append(numpy_to_tensor(lr_patch))
                hr_patches.append(numpy_to_tensor(hr_patch))
                crop_names.append(crop_name)

            if not lr_patches:
                skipped_images += 1
                continue

            # Stack: [N_crops, 3, 64, 64]
            lr_batch = torch.stack(lr_patches).to(device)
            # HR stays on CPU (only used for saving)
            hr_batch_cpu = torch.stack(hr_patches)

            # ── Run 3 local experts (DRCT + GRL + NAFNet) ───────────
            with torch.no_grad():
                local_outputs, local_features = ensemble.forward_all_with_hooks(lr_batch)

            # ── Run MambaIR with FP16 autocast ──────────────────────
            mamba_sr_batch = None
            mamba_feat_batch = None

            if mamba_model is not None:
                mamba_hook_cache.clear()
                with torch.no_grad(), torch.amp.autocast('cuda'):
                    mamba_sr_batch = mamba_model(lr_batch).clamp(0, 1)
                mamba_feat_batch = mamba_hook_cache.get('feat')

            # ── Save 3 .pt files per crop ───────────────────────────
            for k, crop_name in enumerate(crop_names):
                crop_stem = f"{stem}_{crop_name}"
                lh, lw = args.lr_patch_size, args.lr_patch_size

                # --- DRCT part (includes LR + HR for dataset) ---
                drct_out = local_outputs.get(
                    'drct', torch.zeros(lr_batch.shape[0], 3, lh*4, lw*4)
                )
                drct_feat = local_features.get(
                    'drct', torch.zeros(lr_batch.shape[0], 180, lh, lw)
                )

                torch.save({
                    'outputs':  {'drct': drct_out[k:k+1].cpu()},
                    'features': {'drct': drct_feat[k:k+1].cpu()},
                    'lr':       lr_patches[k],         # [3, 64, 64] CPU
                    'hr':       hr_batch_cpu[k],       # [3, 256, 256] CPU
                    'filename': crop_stem,
                }, os.path.join(output_dir, f'{crop_stem}_drct_part.pt'))

                # --- Rest part (GRL + NAFNet) ---
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

                # --- Mamba part (FP16) ---
                if mamba_sr_batch is not None:
                    mamba_sr = mamba_sr_batch[k:k+1].cpu().half()
                    mamba_feat = (
                        mamba_feat_batch[k:k+1].cpu().half()
                        if mamba_feat_batch is not None
                        else torch.zeros(1, 180, lh, lw, dtype=torch.float16)
                    )
                else:
                    mamba_sr = torch.zeros(1, 3, lh*4, lw*4, dtype=torch.float16)
                    mamba_feat = torch.zeros(1, 180, lh, lw, dtype=torch.float16)

                torch.save({
                    'outputs':  {'mamba': mamba_sr},
                    'features': {'mamba': mamba_feat},
                    'filename': crop_stem,
                }, os.path.join(output_dir, f'{crop_stem}_mamba_part.pt'))

                processed_crops += 1

            # Periodic VRAM cleanup
            if processed_crops % 200 == 0:
                torch.cuda.empty_cache()
                if rank == 0 and isinstance(iterator, tqdm):
                    elapsed = time.time() - t0
                    rate = processed_crops / (elapsed + 1e-8)
                    total_remaining = (len(my_pairs) * 5) - processed_crops
                    eta = total_remaining / (rate + 1e-8) / 60
                    iterator.set_postfix(
                        crops=processed_crops, skip=skipped_images,
                        err=errors, ETA=f'{eta:.0f}m'
                    )

        except Exception as e:
            errors += 1
            if rank == 0:
                print(f"\n  [ERROR GPU {rank}] {stem}: {e}")

    elapsed = time.time() - t0

    # Restore stdout for summary
    if rank != 0:
        sys.stdout = sys.__stdout__

    print(f"\n  GPU {rank} finished: {processed_crops} crops in "
          f"{elapsed:.1f}s ({elapsed/60:.1f} min)")
    print(f"    Skipped: {skipped_images} images | Errors: {errors}")


# =============================================================================
# Main
# =============================================================================

def main():
    args = parse_args()

    print("\n" + "=" * 70)
    print("  UNIFIED 5-CROP FEATURE EXTRACTOR — KAGGLE MASTER SCRIPT")
    print("=" * 70)
    print(f"  ⚡ 5 deterministic crops × all images = championship diversity")
    print(f"  ⚡ All 4 experts (DRCT + GRL + NAFNet + MambaIR) in one pass")
    print(f"  ⚡ Notebook split: indices [{args.start_idx}:{args.end_idx}]")
    print(f"  ⚡ Output: {args.output_dir}")
    print("=" * 70 + "\n")

    # ── GPU detection ───────────────────────────────────────────────────
    avail_gpus = torch.cuda.device_count()
    num_gpus = avail_gpus if args.num_gpus == -1 else min(args.num_gpus, avail_gpus)

    if num_gpus < 1:
        print("ERROR: No CUDA GPUs found!")
        sys.exit(1)

    for i in range(num_gpus):
        name = torch.cuda.get_device_name(i)
        mem = torch.cuda.get_device_properties(i).total_memory / 1e9
        print(f"  GPU {i}: {name} ({mem:.1f} GB)")

    # ── Discover image pairs ────────────────────────────────────────────
    splits = ['train', 'val'] if args.split == 'both' else [args.split]

    for split in splits:
        print(f"\n{'='*70}")
        print(f"  Processing {split.upper()} split")
        print(f"{'='*70}")

        all_pairs = discover_image_pairs(args.dataset_dir, split)

        if not all_pairs:
            print(f"  ❌ No image pairs found for {split}! Check --dataset-dir.")
            continue

        # Apply index range for notebook splitting
        end_idx = args.end_idx if args.end_idx > 0 else len(all_pairs)
        start_idx = max(0, min(args.start_idx, len(all_pairs)))
        end_idx = max(start_idx, min(end_idx, len(all_pairs)))

        selected_pairs = all_pairs[start_idx:end_idx]
        print(f"  Index range: [{start_idx}:{end_idx}] = {len(selected_pairs)} images")
        print(f"  Expected crops: {len(selected_pairs)} × 5 = {len(selected_pairs) * 5}")

        if not selected_pairs:
            print(f"  ❌ No images in range [{start_idx}:{end_idx}]!")
            continue

        # Determine output directory
        output_dir = os.path.join(args.output_dir, f'cached_features_{split}')
        os.makedirs(output_dir, exist_ok=True)
        print(f"  Output: {output_dir}")

        t0 = time.time()

        if num_gpus > 1:
            # Multi-GPU: spawn workers
            mp.spawn(
                worker_process,
                args=(num_gpus, selected_pairs, output_dir, args),
                nprocs=num_gpus,
                join=True,
            )
        else:
            # Single GPU: run directly
            worker_process(0, 1, selected_pairs, output_dir, args)

        total_time = time.time() - t0

        # Count output files
        n_drct = len(glob.glob(os.path.join(output_dir, '*_drct_part.pt')))
        n_rest = len(glob.glob(os.path.join(output_dir, '*_rest_part.pt')))
        n_mamba = len(glob.glob(os.path.join(output_dir, '*_mamba_part.pt')))

        print(f"\n  {split.upper()} COMPLETE in {total_time/60:.1f} min")
        print(f"  Files on disk:")
        print(f"    _drct_part.pt:  {n_drct}")
        print(f"    _rest_part.pt:  {n_rest}")
        print(f"    _mamba_part.pt: {n_mamba}")

    # ── Final Summary ───────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  🎉 UNIFIED 5-CROP EXTRACTION COMPLETE!")
    print("=" * 70)
    print(f"\n  Next steps:")
    print(f"    1. Zip the output:  cd {args.output_dir} && zip -r cache_partN.zip cached_features_train/")
    print(f"    2. Save the notebook — output becomes a Kaggle Dataset")
    print(f"    3. Repeat for other index ranges (3-notebook split)")
    print(f"    4. Attach all outputs to your training notebook")
    print(f"    5. Unzip and train:  python train.py --cached")
    print("=" * 70 + "\n")


if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)
    main()
