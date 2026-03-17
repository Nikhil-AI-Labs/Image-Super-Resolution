"""
Standalone Checkpoint Validator — V2
======================================
Loads a checkpoint and runs a full PSNR/SSIM evaluation loop.
Supports CACHED mode via CachedSRDataset.

Usage:
    # Quick load + sanity check (no data required):
    python scripts/validate_checkpoint.py \
        --checkpoint checkpoints/best.pth \
        --config configs/train_config.yaml \
        --quick

    # Full validation with cached features:
    python scripts/validate_checkpoint.py \
        --checkpoint checkpoints/best.pth \
        --config configs/train_config.yaml \
        --cached \
        --cache_dir dataset/DF2K/cached_features_val

    # Save SR images:
    python scripts/validate_checkpoint.py \
        --checkpoint checkpoints/best.pth \
        --config configs/train_config.yaml \
        --cached --save_images
"""

import sys
import os
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
import yaml
import argparse
import numpy as np
from tqdm import tqdm
from typing import Dict, Optional


def parse_args():
    parser = argparse.ArgumentParser(description='Validate checkpoint (V2)')
    parser.add_argument('--checkpoint',  type=str, required=True)
    parser.add_argument('--config',      type=str, default='configs/train_config.yaml')
    parser.add_argument('--cached',      action='store_true',
                        help='Use CachedSRDataset (requires --cache_dir)')
    parser.add_argument('--cache_dir',   type=str, default=None,
                        help='Path to cached_features_val/')
    parser.add_argument('--quick',       action='store_true',
                        help='Load-only check: verify weights load, skip inference loop')
    parser.add_argument('--save_images', action='store_true')
    parser.add_argument('--output_dir',  type=str, default='results/checkpoint_val')
    parser.add_argument('--batch_size',  type=int, default=1)
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--gpu',         type=int, default=0)
    parser.add_argument('--crop_border', type=int, default=4)
    parser.add_argument('--test_y_channel', action='store_true', default=True)
    return parser.parse_args()


def load_checkpoint(checkpoint_path: str, model, device) -> dict:
    """Load checkpoint into model. Returns the raw checkpoint dict."""
    print(f"\nLoading checkpoint: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Prefer EMA → model_state_dict → state_dict → raw
    if 'ema_state_dict' in ckpt:
        print("  Using EMA weights (preferred for inference)")
        result = model.load_state_dict(ckpt['ema_state_dict'], strict=False)
    elif 'model_state_dict' in ckpt:
        print("  Using model_state_dict")
        result = model.load_state_dict(ckpt['model_state_dict'], strict=False)
    elif 'state_dict' in ckpt:
        print("  Using state_dict")
        result = model.load_state_dict(ckpt['state_dict'], strict=False)
    else:
        print("  Using raw checkpoint (direct state dict)")
        result = model.load_state_dict(ckpt, strict=False)

    if result.missing_keys:
        print(f"  ⚠  Missing keys:    {len(result.missing_keys)}")
        for k in result.missing_keys[:5]:
            print(f"       {k}")
    if result.unexpected_keys:
        print(f"  ⚠  Unexpected keys: {len(result.unexpected_keys)}")

    if isinstance(ckpt, dict):
        epoch = ckpt.get('epoch')
        psnr  = ckpt.get('best_psnr',
                ckpt.get('psnr',
                (ckpt.get('metrics') or {}).get('psnr')))
        stage = (ckpt.get('extra_state') or {}).get('stage')
        if epoch is not None: print(f"  Saved at epoch: {epoch}")
        if psnr  is not None: print(f"  Saved PSNR:     {psnr:.4f} dB")
        if stage is not None: print(f"  Training stage: {stage}")

    return ckpt


@torch.no_grad()
def run_validation(
    model,
    val_loader,
    device,
    cached_mode:    bool = False,
    crop_border:    int  = 4,
    test_y_channel: bool = True,
    save_images:    bool = False,
    output_dir:     Optional[Path] = None,
) -> Dict[str, float]:
    """Full inference + metric loop. Mirrors validate_epoch in train.py."""
    from src.utils import MetricCalculator, calculate_psnr, calculate_ssim

    model.eval()
    metric_calc = MetricCalculator(
        crop_border=crop_border,
        test_y_channel=test_y_channel,
    )
    if save_images and output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)

    per_image = []
    pbar = tqdm(val_loader, desc='Evaluating', ncols=100)

    for batch in pbar:
        lr       = batch['lr'].to(device)
        hr       = batch['hr'].to(device)
        filename = batch['filename']
        if isinstance(filename, (list, tuple)):
            filename = filename[0]

        # ── Forward pass ──────────────────────────────────────────────────
        if cached_mode and 'expert_imgs' in batch:
            # FP16 cached tensors → FP32 before entering fusion network
            expert_imgs = {
                k: v.to(device, dtype=torch.float32, non_blocking=True)
                for k, v in batch['expert_imgs'].items()
            }
            expert_feats = None
            if 'expert_feats' in batch:
                expert_feats = {
                    k: v.to(device, dtype=torch.float32, non_blocking=True)
                    for k, v in batch['expert_feats'].items()
                }
            sr = model.forward_with_precomputed(lr, expert_imgs, expert_feats)
        else:
            sr = model(lr)

        sr = sr.clamp(0, 1)

        psnr = calculate_psnr(sr[0], hr[0], crop_border, test_y_channel)
        ssim = calculate_ssim(sr[0], hr[0], crop_border, test_y_channel)
        per_image.append({'filename': filename, 'psnr': psnr, 'ssim': ssim})

        metric_calc.update(sr, hr)
        curr = metric_calc.get_metrics()
        pbar.set_postfix(PSNR=f"{curr['psnr']:.2f}", SSIM=f"{curr['ssim']:.4f}")

        if save_images and output_dir:
            import torchvision
            torchvision.utils.save_image(sr[0].cpu(),
                                         output_dir / f"{Path(filename).stem}_SR.png")

    final = metric_calc.get_metrics()
    sorted_imgs = sorted(per_image, key=lambda x: x['psnr'], reverse=True)

    print(f"\n{'='*70}")
    print(f"  PSNR: {final['psnr']:.4f} dB")
    print(f"  SSIM: {final['ssim']:.6f}")
    print(f"\n  Top 3 / Bottom 3:")
    for m in sorted_imgs[:3] + sorted_imgs[-3:]:
        print(f"    {m['filename']:<30}  {m['psnr']:.2f} dB")
    print(f"{'='*70}\n")

    return final


# ============================================================================
# Main
# ============================================================================

if __name__ == '__main__':
    args   = parse_args()
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # ── Load config ───────────────────────────────────────────────────────
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    fusion_cfg   = config['model']['fusion']
    improvements = fusion_cfg.get('improvements', {})

    # ── Build V2 model ────────────────────────────────────────────────────
    print("\nCreating model: CompleteEnhancedFusionSR (V2)")
    from src.models.enhanced_fusion_v2 import CompleteEnhancedFusionSR

    model = CompleteEnhancedFusionSR(
        expert_ensemble          = None,
        num_experts              = fusion_cfg.get('num_experts', 4),
        upscale                  = 4,
        fusion_dim               = fusion_cfg.get('fusion_dim', 128),
        refine_channels          = fusion_cfg.get('refine_channels', 128),
        refine_depth             = fusion_cfg.get('refine_depth', 6),
        base_channels            = fusion_cfg.get('base_channels', 64),
        block_size               = fusion_cfg.get('block_size', 8),
        enable_dynamic_selection = improvements.get('dynamic_expert_selection', True),
        enable_cross_band_attn   = improvements.get('cross_band_attention',     True),
        enable_adaptive_bands    = improvements.get('adaptive_frequency_bands', True),
        enable_multi_resolution  = improvements.get('multi_resolution_fusion',  True),
        enable_collaborative     = improvements.get('collaborative_learning',   True),
        enable_edge_enhance      = improvements.get('edge_enhancement',          True),
    ).to(device)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Trainable params: {trainable:,}")

    # ── Load weights ──────────────────────────────────────────────────────
    ckpt = load_checkpoint(args.checkpoint, model, device)
    model.eval()
    print("\n✓ Checkpoint loaded successfully")

    # ── Quick mode: stop here ─────────────────────────────────────────────
    if args.quick:
        print("\n[--quick] Skipping inference loop. Model architecture + weights verified.")
        print("  Run without --quick to get actual PSNR/SSIM numbers.")
        sys.exit(0)

    # ── Require --cached or provide live data ─────────────────────────────
    if not args.cached:
        print("\n⚠  No --cached flag. Full live-mode validation requires an expert ensemble.")
        print("   Use --cached --cache_dir <path> for cached-mode validation.")
        print("   Use --quick for a load-only check.")
        sys.exit(0)

    # ── Build cached dataloader ───────────────────────────────────────────
    from src.data import CachedSRDataset
    from torch.utils.data import DataLoader

    cache_dir = args.cache_dir
    if cache_dir is None:
        cache_dir = os.path.join(
            config['dataset']['val']['root'], 'cached_features_val'
        )
    print(f"\nLoading cached val dataset from: {cache_dir}")

    val_dataset = CachedSRDataset(
        feature_dir   = cache_dir,
        augment       = False,
        repeat_factor = 1,
        load_features = True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size  = args.batch_size,
        shuffle     = False,
        num_workers = args.num_workers,
        pin_memory  = True,
    )

    # ── Run validation ────────────────────────────────────────────────────
    output_dir = Path(args.output_dir) if args.save_images else None
    metrics = run_validation(
        model          = model,
        val_loader     = val_loader,
        device         = device,
        cached_mode    = True,
        crop_border    = args.crop_border,
        test_y_channel = args.test_y_channel,
        save_images    = args.save_images,
        output_dir     = output_dir,
    )

    print(f"✓ Final → PSNR: {metrics['psnr']:.4f} dB  SSIM: {metrics['ssim']:.6f}")
