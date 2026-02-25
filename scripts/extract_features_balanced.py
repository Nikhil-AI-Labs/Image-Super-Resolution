"""
scripts/extract_features_balanced.py
=====================================
Single-GPU Feature Extractor — Phase 3 (4 Experts: HAT, DRCT, GRL, EDSR)

Uses forward_all_with_hooks() — the Phase 2 API — so no manual hook wiring needed.

Output format (backward-compatible with CachedSRDataset):
  {cache_dir}/
    {stem}_hat_part.pt   — HAT outputs + features + LR/HR images
    {stem}_rest_part.pt  — DRCT + GRL + EDSR outputs + features

VRAM estimate (fp32): HAT-L 163 MB + DRCT-L 111 MB + GRL-B 81 MB + EDSR-L 172 MB ≈ 527 MB
  All 4 experts fit comfortably in 26 GB.

Usage:
  python scripts/extract_features_balanced.py                     # full extraction
  python scripts/extract_features_balanced.py --test-mode         # 5 images
  python scripts/extract_features_balanced.py --resume            # skip existing
  python scripts/extract_features_balanced.py --train-only
  python scripts/extract_features_balanced.py --val-only
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

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def parse_args():
    parser = argparse.ArgumentParser(description='Cache expert features for fast training')
    parser.add_argument('--config',      type=str, default='configs/train_config.yaml')
    parser.add_argument('--test-mode',   action='store_true', help='Run on small subset')
    parser.add_argument('--num-samples', type=int, default=5,  help='Samples in test mode')
    parser.add_argument('--resume',      action='store_true',  help='Skip already-extracted files')
    parser.add_argument('--train-only',  action='store_true')
    parser.add_argument('--val-only',    action='store_true')
    return parser.parse_args()


def extract_split(
    split_name:  str,
    hr_dir:      str,
    lr_dir:      str,
    scale:       int,
    save_dir:    Path,
    config:      Dict,
    args,
) -> tuple:
    """
    Extract and cache all 4 expert outputs + intermediate features for one split.

    Saves two files per image (backward-compatible with CachedSRDataset):
      {stem}_hat_part.pt  → {'outputs': {'hat':...}, 'features': {'hat':...}, 'lr':..., 'hr':...}
      {stem}_rest_part.pt → {'outputs': {'drct':..., 'grl':..., 'edsr':...},
                              'features': {'drct':..., 'grl':..., 'edsr':...}}
    """
    from src.models.expert_loader import ExpertEnsemble
    from src.data.dataset import SRDataset

    print(f"\n{'='*70}")
    print(f"  EXTRACTING {split_name.upper()} SET")
    print(f"{'='*70}")
    print(f"  HR dir:  {hr_dir}")
    print(f"  LR dir:  {lr_dir}")
    print(f"  Save to: {save_dir}")
    save_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    if torch.cuda.is_available():
        mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"  Device:  {device}  ({torch.cuda.get_device_name(0)}, {mem:.0f} GB VRAM)")
    else:
        print(f"  Device:  cpu  (WARNING: very slow!)")

    # ── Load all 4 experts ─────────────────────────────────────────────────
    print("\n  Loading expert models...")

    # Build checkpoint path dict from config
    expert_configs = config.get('model', {}).get('experts', [])
    checkpoint_paths = {}
    for expert_cfg in expert_configs:
        name = expert_cfg.get('name', '').lower()
        weight = expert_cfg.get('weight_path', '')
        if name and weight:
            checkpoint_paths[name] = weight

    ensemble = ExpertEnsemble(device=device, upscale=scale)
    load_results = ensemble.load_all_experts(
        checkpoint_paths=checkpoint_paths,
        freeze=True,
    )

    loaded = [k for k, v in load_results.items() if v]
    failed = [k for k, v in load_results.items() if not v]
    print(f"  Loaded ({len(loaded)}/4): {loaded}")
    if failed:
        print(f"  FAILED:               {failed}")
        if len(loaded) == 0:
            raise RuntimeError(
                "No experts loaded at all. "
                "Check checkpoint paths in train_config.yaml."
            )

    # ── Dataset ────────────────────────────────────────────────────────────
    dataset = SRDataset(
        hr_dir=hr_dir, lr_dir=lr_dir,
        lr_patch_size=64, scale=scale,
        augment=False, repeat_factor=1,
    )
    total = len(dataset)
    indices = list(range(min(args.num_samples, total))) if args.test_mode \
              else list(range(total))
    print(f"\n  {'Test mode' if args.test_mode else 'Full extraction'}: {len(indices)} images")

    # ── Extraction loop ────────────────────────────────────────────────────
    skipped = processed = errors = 0
    t0 = time.time()

    with torch.no_grad():
        pbar = tqdm(indices, desc=f'{split_name}', ncols=110)
        for idx in pbar:
            sample   = dataset[idx]
            stem     = Path(sample['filename']).stem
            hat_path  = save_dir / f"{stem}_hat_part.pt"
            rest_path = save_dir / f"{stem}_rest_part.pt"

            if args.resume and hat_path.exists() and rest_path.exists():
                skipped += 1
                continue

            lr = sample['lr'].unsqueeze(0).to(device)   # [1, 3, H, W]

            try:
                # ONE call runs all 4 experts + captures intermediate features
                # Returns:
                #   outputs:  {'hat':[1,3,H*4,W*4], 'drct':..., 'grl':..., 'edsr':...}
                #   features: {'hat':[1,180,H,W], 'drct':[1,180,H,W],
                #              'grl':[1,180,H,W],  'edsr':[1,256,H,W]}
                outputs, features = ensemble.forward_all_with_hooks(lr)

            except Exception as e:
                errors += 1
                print(f"\n  [ERROR] {stem}: {e}")
                continue

            # ── HAT part (includes LR / HR for dataset reconstruction) ──────
            lh, lw = sample['lr'].shape[-2], sample['lr'].shape[-1]

            torch.save({
                'outputs':  {'hat': outputs['hat'].cpu()},
                'features': {'hat': features.get(
                    'hat', torch.zeros(1, 180, lh, lw)).cpu()},
                'lr':       sample['lr'],   # [3, H, W] CPU
                'hr':       sample['hr'],   # [3, H*4, W*4] CPU
                'filename': stem,
            }, hat_path)

            # ── Rest part (DRCT + GRL + EDSR) ────────────────────────────────
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

            if processed % 50 == 0:
                torch.cuda.empty_cache()
                elapsed = time.time() - t0
                rate    = processed / (elapsed + 1e-8)
                eta     = (len(indices) - processed - skipped) / (rate + 1e-8) / 60
                pbar.set_postfix(done=processed, skip=skipped,
                                 err=errors, ETA=f'{eta:.0f}m')

    # ── Summary ────────────────────────────────────────────────────────────
    elapsed = time.time() - t0
    n_hat  = len(list(save_dir.glob("*_hat_part.pt")))
    n_rest = len(list(save_dir.glob("*_rest_part.pt")))
    print(f"\n  {split_name.upper()} complete in {elapsed/60:.1f} min")
    print(f"  Processed: {processed}  |  Skipped: {skipped}  |  Errors: {errors}")
    print(f"  Files on disk — hat_parts: {n_hat}  rest_parts: {n_rest}")
    if n_hat != n_rest:
        print(f"  WARNING: hat/rest counts mismatch! Run again with --resume to fix.")

    return len(indices), elapsed


def main():
    args   = parse_args()
    print("\n" + "=" * 70)
    print("  SINGLE-GPU FEATURE EXTRACTOR  (Phase 3 — HAT/DRCT/GRL/EDSR)")
    print("=" * 70)
    print("  API: forward_all_with_hooks()  (Phase 2)")
    print("  Output format: _hat_part.pt + _rest_part.pt per image")
    print("=" * 70 + "\n")

    with open(args.config) as f:
        config = yaml.safe_load(f)

    dataset_cfg = config['dataset']
    scale       = dataset_cfg.get('scale', 4)

    total_images = total_time = 0

    if not args.val_only:
        root = dataset_cfg['train']['root']
        n, t = extract_split(
            'train',
            os.path.join(root, dataset_cfg['train'].get('hr_subdir', 'train_HR')),
            os.path.join(root, dataset_cfg['train'].get('lr_subdir', 'train_LR')),
            scale,
            Path(root) / 'cached_features_train',
            config, args,
        )
        total_images += n;  total_time += t

    if not args.train_only:
        root = dataset_cfg['val']['root']
        n, t = extract_split(
            'val',
            os.path.join(root, dataset_cfg['val'].get('hr_subdir', 'val_HR')),
            os.path.join(root, dataset_cfg['val'].get('lr_subdir', 'val_LR')),
            scale,
            Path(root) / 'cached_features_val',
            config, args,
        )
        total_images += n;  total_time += t

    print("\n" + "=" * 70)
    print("  EXTRACTION COMPLETE")
    print("=" * 70)
    print(f"  Total images : {total_images}")
    print(f"  Total time   : {total_time/60:.1f} min")
    if total_images:
        print(f"  Avg/image    : {total_time/total_images:.2f}s")
    print("\n  Next step:")
    print("    python train.py --cached")
    print("=" * 70 + "\n")


if __name__ == '__main__':
    main()
