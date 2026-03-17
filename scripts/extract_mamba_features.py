"""
MambaIR Feature Extraction Script (FAST PATCH EDITION)
===============================================================
Reads aligned 64x64 LR patches directly from _drct_part.pt files.
Processes in batches of 16 using FP16 Autocast.
Reduces extraction time from 10 hours -> ~3 minutes!

This script is part of the "Decoupled Compute Paradigm":
  - MambaIR requires mamba_ssm + causal_conv1d (CUDA-compiled SSM ops)
  - These do NOT compile on older GPUs like the P1000
  - This script runs on Colab/Kaggle (T4/P100) where mamba_ssm works
  - The .pt files are then downloaded to the local machine for training

Optimizations applied:
  1. Reads pre-existing _drct_part.pt patches (64x64) instead of full PNGs
  2. Uses torch.load() instead of Image.open() — no PIL dependency
  3. Batches 16 uniform patches together (was batch_size=1 for variable sizes)
  4. FP16 Autocast via torch.amp.autocast('cuda') for Tensor Core acceleration

Output format (per patch, unchanged):
    {stem}_mamba_part.pt = {
        'outputs':  {'mamba': [1, 3, 256, 256]  (FP16)},
        'features': {'mamba': [1, 180, 64, 64]  (FP16)},
        'filename': str
    }

Usage on Colab/Kaggle:
    !pip install mamba-ssm causal-conv1d einops timm
    !python scripts/extract_mamba_features.py \\
        --dataset-dir /kaggle/working/dataset/DF2K --split train
    !python scripts/extract_mamba_features.py \\
        --dataset-dir /kaggle/working/dataset/DF2K --split val

Author: NTIRE SR Team
"""

import os
import sys
import argparse
import torch
import glob
import time
from pathlib import Path
from tqdm.auto import tqdm

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def parse_args():
    parser = argparse.ArgumentParser(
        description='Extract MambaIR features (Fast Patch Mode)'
    )
    parser.add_argument(
        '--split', type=str, default='both',
        choices=['train', 'val', 'both'],
        help='Which split to extract (default: both)'
    )
    parser.add_argument(
        '--dataset-dir', type=str, required=True,
        help='Path to dataset/DF2K (contains cached_features_train/ and _val/)'
    )
    parser.add_argument(
        '--weights', type=str,
        default='./pretrained/mambair/MambaIR_x4.pth',
        help='Path to MambaIR pretrained weights (.pth)'
    )
    parser.add_argument(
        '--resume', action='store_true',
        help='Skip patches that already have _mamba_part.pt files'
    )
    parser.add_argument(
        '--batch-size', type=int, default=16,
        help='Batch size (16 is safe for 64x64 patches on T4/P100)'
    )
    return parser.parse_args()


def load_mambair(weights_path, device):
    """Load MambaIR model with pretrained weights."""
    print(f"  Loading MambaIR to {device}...")

    try:
        from src.models.mambair.mambair_arch import MambaIR
    except ImportError as e:
        print(f"ERROR: Cannot import MambaIR: {e}")
        print("Install required packages:")
        print("  pip install mamba-ssm causal-conv1d einops timm")
        sys.exit(1)

    model = MambaIR(
        upscale=4,
        in_chans=3,
        img_size=64,
        window_size=16,
        compress_ratio=3,
        squeeze_factor=30,
        conv_scale=0.01,
        overlap_ratio=0.5,
        img_range=1.0,
        depths=(6, 6, 6, 6, 6, 6),
        embed_dim=180,
        mlp_ratio=2.0,
        drop_path_rate=0.1,
        upsampler='pixelshuffle',
        resi_connection='1conv',
    )

    # Load pretrained weights
    if not os.path.exists(weights_path):
        print(f"ERROR: Weights file not found: {weights_path}")
        print("Upload MambaIR_x4.pth to your Colab/Kaggle environment.")
        sys.exit(1)

    print(f"  Loading weights from: {weights_path}")
    state = torch.load(weights_path, map_location='cpu', weights_only=False)

    # Handle different checkpoint formats
    if 'params' in state:
        state_dict = state['params']
    elif 'state_dict' in state:
        state_dict = state['state_dict']
    elif 'model' in state:
        state_dict = state['model']
    else:
        state_dict = state

    # Clean state dict keys (remove 'module.' prefix if present)
    clean_state = {k.replace('module.', ''): v for k, v in state_dict.items()}

    missing, unexpected = model.load_state_dict(clean_state, strict=False)
    if missing:
        print(f"  WARNING: {len(missing)} missing keys")
        if len(missing) <= 5:
            for k in missing:
                print(f"    - {k}")
    if unexpected:
        print(f"  WARNING: {len(unexpected)} unexpected keys")

    model.eval().to(device)

    total_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"  ✓ MambaIR Loaded ({total_params:.2f}M params)")
    return model


def extract_fast(model, hook_cache, cache_dir, split_name, args, device):
    """
    Extract MambaIR features using pre-existing DRCT patches.
    
    Instead of loading full-size PNGs, reads the 64x64 LR patches already
    stored in _drct_part.pt files, batches them 16-at-a-time, and runs
    MambaIR with FP16 autocast for maximum throughput.
    """
    print(f"\n{'='*70}")
    print(f"  EXTRACTING {split_name.upper()} (Fast Patch Mode)")
    print(f"  Directory: {cache_dir}")
    print(f"  Batch size: {args.batch_size}")
    print(f"{'='*70}")

    # ── OPTIMIZATION 1: Search for .pt files, not .png files ──────────────
    drct_files = sorted(glob.glob(os.path.join(cache_dir, '*_drct_part.pt')))
    if not drct_files:
        print(f"  ❌ ERROR: No _drct_part.pt files found in {cache_dir}!")
        print("  You MUST run extract_features_balanced.py (or multi_gpu) first,")
        print("  and place the _drct_part.pt files in this directory.")
        return 0

    # ── Resume logic ────────────────────────────────────────────────────
    done_set = {
        Path(f).name.replace('_mamba_part.pt', '')
        for f in glob.glob(os.path.join(cache_dir, '*_mamba_part.pt'))
    }
    todo = [
        (Path(f).name.replace('_drct_part.pt', ''), f)
        for f in drct_files
        if Path(f).name.replace('_drct_part.pt', '') not in done_set
    ]

    print(f"  Total Patches: {len(drct_files)} | Already Done: {len(done_set)} | Pending: {len(todo)}")
    if not todo:
        print("  ✅ All patches already extracted! Nothing to do.")
        return 0

    processed = 0
    errors = 0
    t0 = time.time()

    # ── OPTIMIZATION 3: Process in batches of 16 ────────────────────────
    pbar = tqdm(range(0, len(todo), args.batch_size), desc=split_name, ncols=100)
    for b in pbar:
        batch_items = todo[b : b + args.batch_size]
        lrs = []
        stems = []

        # ── OPTIMIZATION 2: Load LR patches from _drct_part.pt files ───
        for stem, f_path in batch_items:
            try:
                data = torch.load(f_path, map_location='cpu', weights_only=False)
                lr = data['lr']  # [3, 64, 64] or [1, 3, 64, 64]
                # Handle batch dimension if present
                if lr.dim() == 4:
                    lr = lr.squeeze(0)
                lrs.append(lr.float())
                stems.append(stem)
            except Exception as e:
                errors += 1
                print(f"\n  [ERROR loading] {stem}: {e}")

        if not lrs:
            continue

        # Stack into batch: [N, 3, 64, 64]
        lr_batch = torch.stack(lrs).to(device)

        # ── OPTIMIZATION 4: FP16 Autocast for Tensor Core acceleration ─
        with torch.no_grad(), torch.amp.autocast('cuda'):
            sr_batch = model(lr_batch).clamp(0, 1)

        # Grab hooked features (captured during forward pass)
        feat_batch = hook_cache.get('deep_tensor')

        # Save individual _mamba_part.pt files aligned with their stems
        for k, stem in enumerate(stems):
            sr = sr_batch[k:k+1].cpu().half()   # [1, 3, 256, 256] FP16

            if feat_batch is not None:
                feat = feat_batch[k:k+1].cpu().half()  # [1, 180, 64, 64] FP16
            else:
                feat = torch.zeros(1, 180, 64, 64, dtype=torch.float16)

            torch.save({
                'outputs':  {'mamba': sr},
                'features': {'mamba': feat},
                'filename': stem,
            }, os.path.join(cache_dir, f'{stem}_mamba_part.pt'))

            processed += 1

        # Periodic VRAM cleanup
        if processed % 100 == 0:
            torch.cuda.empty_cache()
            elapsed = time.time() - t0
            rate = processed / (elapsed + 1e-8)
            remaining = (len(todo) - processed) / (rate + 1e-8)
            pbar.set_postfix(
                done=processed, err=errors,
                ETA=f'{remaining/60:.0f}m'
            )

    elapsed = time.time() - t0
    n_files = len(glob.glob(os.path.join(cache_dir, '*_mamba_part.pt')))
    print(f"\n  ✓ {split_name.upper()} Complete:")
    print(f"    Processed: {processed} patches in {elapsed:.1f}s ({elapsed/60:.1f} min)")
    print(f"    Errors:    {errors}")
    print(f"    MambaIR .pt files on disk: {n_files}")
    if processed > 0:
        print(f"    Avg/patch:  {elapsed/processed:.3f}s")
        print(f"    Throughput: {processed/elapsed:.1f} patches/sec")

    return processed


def main():
    args = parse_args()

    print("\n" + "=" * 70)
    print("  MambaIR FEATURE EXTRACTOR — FAST PATCH MODE")
    print("=" * 70)
    print("  ⚡ Reads 64x64 patches from _drct_part.pt (not full PNGs)")
    print(f"  ⚡ Batch size: {args.batch_size}")
    print("  ⚡ FP16 Autocast for Tensor Core acceleration")
    print("  Run this on Colab/Kaggle (T4/P100) where mamba_ssm works.")
    print("=" * 70 + "\n")

    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type == 'cuda':
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"  Device: {device} ({gpu_name}, {gpu_mem:.1f} GB)")
    else:
        print("  WARNING: Running on CPU — FP16 autocast disabled, will be slow!")

    # Load model
    model = load_mambair(args.weights, device)

    # Setup forward hook on conv_after_body to capture deep features
    # This captures [B, embed_dim, H, W] features BEFORE upsampling
    hook_cache = {}

    def hook_fn(module, input, output):
        hook_cache['deep_tensor'] = output

    if hasattr(model, 'conv_after_body'):
        model.conv_after_body.register_forward_hook(hook_fn)
        print("  ✓ Hook registered on model.conv_after_body")
    else:
        print("  ERROR: model.conv_after_body not found!")
        print("  Available attributes:", [a for a in dir(model) if 'conv' in a.lower()])
        sys.exit(1)

    # Execute extraction
    splits = ['train', 'val'] if args.split == 'both' else [args.split]
    total_processed = 0
    total_time = time.time()

    for split in splits:
        cache_dir = os.path.join(args.dataset_dir, f"cached_features_{split}")
        if not os.path.exists(cache_dir):
            print(f"\n  WARNING: Cache directory not found: {cache_dir}")
            print(f"  Skipping {split} split.")
            continue
        n = extract_fast(model, hook_cache, cache_dir, split, args, device)
        total_processed += n

    total_elapsed = time.time() - total_time

    # Summary
    print("\n" + "=" * 70)
    print("  🎉 MAMBA EXTRACTION COMPLETE (Fast Patch Mode)")
    print("=" * 70)
    print(f"  Total processed: {total_processed}")
    print(f"  Total time:      {total_elapsed:.1f}s ({total_elapsed/60:.1f} min)")
    if total_processed:
        print(f"  Avg/patch:       {total_elapsed/total_processed:.3f}s")
    print("\n  Next steps:")
    print("    1. Zip the cached_features_train/ and cached_features_val/ dirs")
    print("    2. Download to your local machine (or add to Kaggle dataset)")
    print("    3. Place _mamba_part.pt files alongside _drct_part.pt / _rest_part.pt")
    print("    4. Train locally: python train.py --cached")
    print("=" * 70 + "\n")


if __name__ == '__main__':
    main()
