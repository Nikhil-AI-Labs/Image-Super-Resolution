"""
scripts/extract_features_multi_gpu.py
=====================================
Multi-GPU Feature Extractor — Phase 3 (4 Experts: HAT, DRCT, GRL, EDSR)

Uses torch.multiprocessing.spawn to split the dataset across all available GPUs.
Each GPU loads its own copy of the 4 experts and processes its shard independently.
Output format is IDENTICAL to extract_features_balanced.py (_hat_part.pt + _rest_part.pt),
so the cached training pipeline works unchanged with a single GPU.

Usage:
  python scripts/extract_features_multi_gpu.py                     # use all GPUs
  python scripts/extract_features_multi_gpu.py --num-gpus 2        # use exactly 2
  python scripts/extract_features_multi_gpu.py --resume            # skip existing
  python scripts/extract_features_multi_gpu.py --test-mode         # small subset
"""

import os
import sys
import argparse
import yaml
import torch
import time
from pathlib import Path
from tqdm import tqdm
from typing import Dict
import torch.multiprocessing as mp

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def parse_args():
    parser = argparse.ArgumentParser(description='Cache expert features (Multi-GPU)')
    parser.add_argument('--config',      type=str, default='configs/train_config.yaml')
    parser.add_argument('--test-mode',   action='store_true', help='Run on small subset')
    parser.add_argument('--num-samples', type=int, default=5,  help='Samples in test mode')
    parser.add_argument('--resume',      action='store_true',  help='Skip already-extracted files')
    parser.add_argument('--train-only',  action='store_true')
    parser.add_argument('--val-only',    action='store_true')
    parser.add_argument('--num-gpus',    type=int, default=-1, help='Number of GPUs to use (-1 for all)')
    return parser.parse_args()


# ============================================================================
# Per-GPU worker: loads experts, processes its shard of the dataset
# ============================================================================

def extract_split_worker(
    rank:        int,
    num_gpus:    int,
    split_name:  str,
    hr_dir:      str,
    lr_dir:      str,
    scale:       int,
    save_dir:    Path,
    config:      Dict,
    args,
):
    """
    Worker function executed by each GPU independently.
    
    Dataset sharding:
      GPU 0 → indices [0, num_gpus, 2*num_gpus, ...]
      GPU 1 → indices [1, 1+num_gpus, 1+2*num_gpus, ...]
      etc.
    
    Each GPU writes to the SAME save_dir. File names are unique per image,
    so no GPU ever overwrites another GPU's files.
    """
    from src.models.expert_loader import ExpertEnsemble
    from src.data.dataset import SRDataset

    device = torch.device(f'cuda:{rank}')

    if rank == 0:
        print(f"\n{'='*70}")
        print(f"  EXTRACTING {split_name.upper()} SET  (across {num_gpus} GPUs)")
        print(f"{'='*70}")
        print(f"  HR dir:  {hr_dir}")
        print(f"  LR dir:  {lr_dir}")
        print(f"  Save to: {save_dir}")

    # Ensure output directory exists (safe from all ranks)
    save_dir.mkdir(parents=True, exist_ok=True)

    # Synchronize after mkdir
    if num_gpus > 1:
        torch.cuda.synchronize(device)

    # ── Load all 4 experts onto this GPU ────────────────────────────────
    expert_configs = config.get('model', {}).get('experts', [])
    checkpoint_paths = {}
    for expert_cfg in expert_configs:
        name = expert_cfg.get('name', '').lower()
        weight = expert_cfg.get('weight_path', '')
        if name and weight:
            checkpoint_paths[name] = weight

    if rank == 0:
        print(f"\n  Loading expert models on {num_gpus} GPU(s)...")

    ensemble = ExpertEnsemble(device=device, upscale=scale)
    load_results = ensemble.load_all_experts(
        checkpoint_paths=checkpoint_paths,
        freeze=True,
    )

    loaded = [k for k, v in load_results.items() if v]
    failed = [k for k, v in load_results.items() if not v]

    if rank == 0:
        print(f"  Loaded ({len(loaded)}/4): {loaded}")
        if failed:
            print(f"  FAILED:               {failed}")

    if not any(load_results.values()):
        if rank == 0:
            print("  ERROR: No experts loaded. Check checkpoint paths in train_config.yaml.")
        return 0, 0.0

    # ── Dataset & multi-GPU sharding ────────────────────────────────────
    dataset = SRDataset(
        hr_dir=hr_dir, lr_dir=lr_dir,
        lr_patch_size=64, scale=scale,
        augment=False, repeat_factor=1,
    )
    total = len(dataset)
    all_indices = list(range(min(args.num_samples, total))) if args.test_mode \
                  else list(range(total))

    # Shard: GPU k processes indices [k, k+num_gpus, k+2*num_gpus, ...]
    my_indices = all_indices[rank::num_gpus]

    if rank == 0:
        mode_str = 'Test mode' if args.test_mode else 'Full extraction'
        print(f"\n  {mode_str}: {len(all_indices)} images total, "
              f"~{len(my_indices)} per GPU")

    skipped = processed = errors = 0
    t0 = time.time()

    with torch.no_grad():
        # Only rank 0 shows the progress bar to keep terminal clean
        if rank == 0:
            pbar = tqdm(my_indices, desc=f'{split_name} (GPU 0/{num_gpus})', ncols=110)
        else:
            pbar = my_indices

        for idx in pbar:
            sample   = dataset[idx]
            stem     = Path(sample['filename']).stem
            hat_path  = save_dir / f"{stem}_hat_part.pt"
            rest_path = save_dir / f"{stem}_rest_part.pt"

            if args.resume and hat_path.exists() and rest_path.exists():
                skipped += 1
                continue

            lr = sample['lr'].unsqueeze(0).to(device)

            try:
                outputs, features = ensemble.forward_all_with_hooks(lr)
            except Exception as e:
                errors += 1
                if rank == 0:
                    print(f"\n  [ERROR GPU {rank}] {stem}: {e}")
                continue

            lh, lw = sample['lr'].shape[-2], sample['lr'].shape[-1]

            # ── HAT part (includes LR / HR for dataset reconstruction) ──
            torch.save({
                'outputs':  {'hat': outputs['hat'].cpu()},
                'features': {'hat': features.get(
                    'hat', torch.zeros(1, 180, lh, lw)).cpu()},
                'lr':       sample['lr'],
                'hr':       sample['hr'],
                'filename': stem,
            }, hat_path)

            # ── Rest part (DRCT + GRL + EDSR) ───────────────────────────
            torch.save({
                'outputs': {
                    'drct': outputs.get('drct', torch.zeros(1, 3, lh*4, lw*4)).cpu(),
                    'grl':  outputs.get('grl',  torch.zeros(1, 3, lh*4, lw*4)).cpu(),
                    'edsr': outputs.get('edsr', torch.zeros(1, 3, lh*4, lw*4)).cpu(),
                },
                'features': {
                    'drct': features.get('drct', torch.zeros(1, 180, lh, lw)).cpu(),
                    'grl':  features.get('grl',  torch.zeros(1, 180, lh, lw)).cpu(),
                    'edsr': features.get('edsr', torch.zeros(1, 256, lh, lw)).cpu(),
                },
                'filename': stem,
            }, rest_path)

            processed += 1

            # Periodic cache clear
            if processed % 50 == 0:
                torch.cuda.empty_cache()

            # Update progress bar (rank 0 only)
            if rank == 0 and isinstance(pbar, tqdm):
                elapsed = time.time() - t0
                rate    = processed / (elapsed + 1e-8)
                eta     = (len(my_indices) - processed - skipped) / (rate + 1e-8) / 60
                pbar.set_postfix(done=processed, skip=skipped,
                                 err=errors, ETA=f'{eta:.0f}m')

    elapsed = time.time() - t0

    if rank == 0:
        n_hat  = len(list(save_dir.glob("*_hat_part.pt")))
        n_rest = len(list(save_dir.glob("*_rest_part.pt")))
        print(f"\n  {split_name.upper()} complete in {elapsed/60:.1f} min")
        print(f"  GPU 0 stats — Processed: {processed}  |  Skipped: {skipped}  |  Errors: {errors}")
        print(f"  Files on disk — hat_parts: {n_hat}  rest_parts: {n_rest}")
        if n_hat != n_rest:
            print(f"  WARNING: hat/rest counts mismatch! Run again with --resume to fix.")

    return processed, elapsed


def worker_process(rank, num_gpus, args, config, dataset_cfg, scale):
    """Entry point for each spawned process."""
    # Only print expert loading info on rank 0 to reduce noise
    if rank != 0:
        import io
        sys.stdout = io.StringIO()  # suppress prints from non-zero ranks

    total_processed = 0
    total_time = 0.0

    if not args.val_only:
        root = dataset_cfg['train']['root']
        n, t = extract_split_worker(
            rank, num_gpus, 'train',
            os.path.join(root, dataset_cfg['train'].get('hr_subdir', 'train_HR')),
            os.path.join(root, dataset_cfg['train'].get('lr_subdir', 'train_LR')),
            scale,
            Path(root) / 'cached_features_train',
            config, args,
        )
        total_processed += n
        total_time += t

    if not args.train_only:
        root = dataset_cfg['val']['root']
        n, t = extract_split_worker(
            rank, num_gpus, 'val',
            os.path.join(root, dataset_cfg['val'].get('hr_subdir', 'val_HR')),
            os.path.join(root, dataset_cfg['val'].get('lr_subdir', 'val_LR')),
            scale,
            Path(root) / 'cached_features_val',
            config, args,
        )
        total_processed += n
        total_time += t

    # Restore stdout for non-zero ranks
    if rank != 0:
        sys.stdout = sys.__stdout__


def main():
    args = parse_args()

    # ── Auto-detect GPUs ────────────────────────────────────────────────
    avail_gpus = torch.cuda.device_count()
    num_gpus = avail_gpus if args.num_gpus == -1 else min(args.num_gpus, avail_gpus)

    if num_gpus < 1:
        print("\nERROR: No CUDA GPUs found! Use extract_features_balanced.py for CPU.")
        sys.exit(1)

    print("\n" + "=" * 70)
    print(f"  MULTI-GPU FEATURE EXTRACTOR  (Phase 3 — HAT/DRCT/GRL/EDSR)")
    print(f"  GPUs detected: {avail_gpus}  |  Using: {num_gpus}")
    print("=" * 70)
    for i in range(num_gpus):
        name = torch.cuda.get_device_name(i)
        mem  = torch.cuda.get_device_properties(i).total_memory / 1e9
        print(f"    GPU {i}: {name}  ({mem:.0f} GB VRAM)")
    print("=" * 70)
    print("  API: forward_all_with_hooks()  (Phase 2)")
    print("  Output format: _hat_part.pt + _rest_part.pt per image")
    print("  (Identical to single-GPU extractor — cached training unchanged)")
    print("=" * 70 + "\n")

    with open(args.config) as f:
        config = yaml.safe_load(f)

    dataset_cfg = config['dataset']
    scale       = dataset_cfg.get('scale', 4)

    t0 = time.time()

    if num_gpus > 1:
        # Spawn one process per GPU — each loads its own experts
        mp.spawn(
            worker_process,
            args=(num_gpus, args, config, dataset_cfg, scale),
            nprocs=num_gpus,
            join=True,
        )
    else:
        # Single GPU — run directly (no spawning overhead)
        worker_process(0, 1, args, config, dataset_cfg, scale)

    total_time = time.time() - t0

    print("\n" + "=" * 70)
    print(f"  EXTRACTION COMPLETE ACROSS {num_gpus} GPU(s)")
    print(f"  Total wall-clock time: {total_time/60:.1f} min")
    if num_gpus > 1:
        print(f"  Speedup estimate: ~{num_gpus}x vs single-GPU")
    print("=" * 70)
    print("\n  Next step:")
    print("    python train.py --cached")
    print("=" * 70 + "\n")


if __name__ == '__main__':
    # CRITICAL: 'spawn' start method is REQUIRED for CUDA multiprocessing
    mp.set_start_method('spawn', force=True)
    main()
