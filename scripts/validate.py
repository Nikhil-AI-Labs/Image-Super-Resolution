"""
Validation Script — V2 (CompleteEnhancedFusionSR)
===================================================
Standalone validation for trained models. Supports:
  - CACHED mode: pre-computed expert features (CachedSRDataset)
  - LIVE mode: expert ensemble runs on GPU

Usage:
    # Cached mode (recommended — no live expert inference):
    python scripts/validate.py \
        --checkpoint checkpoints/best.pth \
        --config configs/train_config.yaml \
        --cached \
        --cache_dir dataset/DF2K/cached_features_val

    # Live mode (requires expert ensemble loaded):
    python scripts/validate.py \
        --checkpoint checkpoints/best.pth \
        --config configs/train_config.yaml

    # Save SR outputs to disk:
    python scripts/validate.py \
        --checkpoint checkpoints/best.pth \
        --config configs/train_config.yaml \
        --cached --save_images --output_dir results/val_sr

Author: NTIRE SR Team
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
from typing import Dict


# ============================================================================
# Argument Parser
# ============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description='Validate CompleteEnhancedFusionSR (V2)',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ── Required ──────────────────────────────────────────────────────────
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to .pth checkpoint file')
    parser.add_argument('--config', type=str, default='configs/train_config.yaml',
                        help='Path to train_config.yaml (required for model arch params)')

    # ── Cached mode ───────────────────────────────────────────────────────
    parser.add_argument('--cached', action='store_true',
                        help='Use CachedSRDataset (pre-computed expert features). '
                             'Much faster — no live expert inference required.')
    parser.add_argument('--cache_dir', type=str, default=None,
                        help='Path to cached val features. '
                             'Defaults to dataset.val.root/cached_features_val from config.')

    # ── Live mode ─────────────────────────────────────────────────────────
    parser.add_argument('--data_root', type=str, default=None,
                        help='Dataset root (live mode only; overrides config)')
    parser.add_argument('--hr_subdir', type=str, default='DIV2K_valid_HR',
                        help='HR subdirectory under data_root (live mode only)')
    parser.add_argument('--lr_subdir', type=str, default='DIV2K_valid_LR_bicubic/X4',
                        help='LR subdirectory under data_root (live mode only)')

    # ── Common ────────────────────────────────────────────────────────────
    parser.add_argument('--scale',          type=int,   default=4)
    parser.add_argument('--batch_size',     type=int,   default=1)
    parser.add_argument('--num_workers',    type=int,   default=2)
    parser.add_argument('--gpu',            type=int,   default=0)
    parser.add_argument('--crop_border',    type=int,   default=4,
                        help='Border pixels to exclude from PSNR/SSIM')
    parser.add_argument('--test_y_channel', action='store_true', default=True,
                        help='Evaluate on Y (luma) channel only (standard SR protocol)')
    parser.add_argument('--save_images',    action='store_true',
                        help='Save SR output images to --output_dir')
    parser.add_argument('--output_dir',     type=str, default='results/validation',
                        help='Directory to save SR images when --save_images is set')

    return parser.parse_args()


# ============================================================================
# Core Validation Loop
# ============================================================================

@torch.no_grad()
def validate(
    model:           torch.nn.Module,
    val_loader,
    device:          torch.device,
    cached_mode:     bool  = False,
    crop_border:     int   = 4,
    test_y_channel:  bool  = True,
    save_images:     bool  = False,
    output_dir:      Path  = None,
) -> Dict[str, float]:
    """
    Run validation loop.

    Handles both CACHED mode (forward_with_precomputed) and LIVE mode (model(lr)).
    FP16 expert tensors from cache are cast to FP32 before forward pass to prevent
    dtype mismatch with the FP32 fusion network.

    Returns:
        {'psnr': float, 'ssim': float}
    """
    from src.utils import MetricCalculator, calculate_psnr, calculate_ssim

    model.eval()
    metric_calc = MetricCalculator(
        crop_border=crop_border,
        test_y_channel=test_y_channel,
    )

    if save_images and output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*70}")
    print("  VALIDATION")
    print(f"{'='*70}")
    print(f"  Mode:      {'CACHED (pre-computed experts)' if cached_mode else 'LIVE (expert ensemble)'}")
    print(f"  Samples:   {len(val_loader.dataset)}")
    print(f"  Crop:      {crop_border}px  |  Y-channel: {test_y_channel}")
    if save_images:
        print(f"  Saving SR images → {output_dir}")
    print(f"{'='*70}\n")

    per_image_metrics = []

    pbar = tqdm(val_loader, desc='Validating', ncols=100)
    for batch in pbar:
        lr       = batch['lr'].to(device)
        hr       = batch['hr'].to(device)
        filename = batch['filename']
        if isinstance(filename, (list, tuple)):
            filename = filename[0]

        # ── Forward pass: cached vs live ──────────────────────────────────
        if cached_mode and 'expert_imgs' in batch:
            # Cast FP16 cached tensors → FP32 to match fusion network weights
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
            # Live mode: expert_ensemble must be loaded in the model
            sr = model(lr)

        sr = sr.clamp(0, 1)

        # ── Per-image metrics ──────────────────────────────────────────────
        psnr = calculate_psnr(sr[0], hr[0], crop_border, test_y_channel)
        ssim = calculate_ssim(sr[0], hr[0], crop_border, test_y_channel)
        per_image_metrics.append({'filename': filename, 'psnr': psnr, 'ssim': ssim})

        metric_calc.update(sr, hr)

        current = metric_calc.get_metrics()
        pbar.set_postfix(
            PSNR=f"{current['psnr']:.2f}",
            SSIM=f"{current['ssim']:.4f}",
        )

        # ── Optional: save SR image ────────────────────────────────────────
        if save_images and output_dir:
            import torchvision
            save_path = output_dir / f"{Path(filename).stem}_SR.png"
            torchvision.utils.save_image(sr[0].cpu(), save_path)

    # ── Summary ───────────────────────────────────────────────────────────
    final = metric_calc.get_metrics()
    sorted_imgs = sorted(per_image_metrics, key=lambda x: x['psnr'], reverse=True)

    print(f"\n{'='*70}")
    print("  VALIDATION RESULTS")
    print(f"{'='*70}")
    print(f"  PSNR: {final['psnr']:.4f} dB")
    print(f"  SSIM: {final['ssim']:.6f}")
    print(f"\n  Top 5 (highest PSNR):")
    for m in sorted_imgs[:5]:
        print(f"    {m['filename']:<30}  {m['psnr']:.2f} dB  SSIM {m['ssim']:.4f}")
    print(f"\n  Worst 5 (lowest PSNR):")
    for m in sorted_imgs[-5:]:
        print(f"    {m['filename']:<30}  {m['psnr']:.2f} dB  SSIM {m['ssim']:.4f}")
    print(f"{'='*70}\n")

    return final


# ============================================================================
# Main
# ============================================================================

def main():
    args   = parse_args()
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")
    if torch.cuda.is_available():
        print(f"GPU:    {torch.cuda.get_device_name(args.gpu)}")

    # ── Load config ───────────────────────────────────────────────────────
    print(f"\nLoading config: {args.config}")
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    fusion_cfg   = config['model']['fusion']
    improvements = fusion_cfg.get('improvements', {})

    # ── Build model (V2) ──────────────────────────────────────────────────
    print("Creating model: CompleteEnhancedFusionSR (V2, enhanced_fusion_v2)")
    from src.models.enhanced_fusion_v2 import CompleteEnhancedFusionSR

    model = CompleteEnhancedFusionSR(
        expert_ensemble         = None,            # no live experts for standalone validation
        num_experts             = fusion_cfg.get('num_experts', 4),
        upscale                 = 4,
        fusion_dim              = fusion_cfg.get('fusion_dim', 128),
        refine_channels         = fusion_cfg.get('refine_channels', 128),
        refine_depth            = fusion_cfg.get('refine_depth', 6),
        base_channels           = fusion_cfg.get('base_channels', 64),
        block_size              = fusion_cfg.get('block_size', 8),
        enable_dynamic_selection = improvements.get('dynamic_expert_selection', True),
        enable_cross_band_attn  = improvements.get('cross_band_attention',      True),
        enable_adaptive_bands   = improvements.get('adaptive_frequency_bands',  True),
        enable_multi_resolution = improvements.get('multi_resolution_fusion',   True),
        enable_collaborative    = improvements.get('collaborative_learning',     True),
        enable_edge_enhance     = improvements.get('edge_enhancement',           True),
    ).to(device)

    total_params     = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total:     {total_params:,}")
    print(f"  Trainable: {trainable_params:,}")

    # ── Load checkpoint ───────────────────────────────────────────────────
    print(f"\nLoading checkpoint: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)

    # Prefer EMA weights — they give consistently better inference quality
    if 'ema_state_dict' in ckpt:
        print("  Using EMA weights (preferred for inference)")
        model.load_state_dict(ckpt['ema_state_dict'], strict=False)
    elif 'model_state_dict' in ckpt:
        print("  Using model_state_dict")
        model.load_state_dict(ckpt['model_state_dict'], strict=False)
    elif 'state_dict' in ckpt:
        print("  Using state_dict")
        model.load_state_dict(ckpt['state_dict'], strict=False)
    else:
        print("  Using raw checkpoint dict")
        model.load_state_dict(ckpt, strict=False)

    if isinstance(ckpt, dict):
        epoch = ckpt.get('epoch')
        saved_psnr = ckpt.get('best_psnr',
                     ckpt.get('psnr',
                     (ckpt.get('metrics') or {}).get('psnr')))
        if epoch      is not None: print(f"  Saved at epoch: {epoch}")
        if saved_psnr is not None: print(f"  Saved PSNR:     {saved_psnr:.4f} dB")

    model.eval()

    # ── Build dataloader ──────────────────────────────────────────────────
    cached_mode = args.cached
    print(f"\nCreating validation dataloader (mode={'CACHED' if cached_mode else 'LIVE'})...")

    if cached_mode:
        from src.data import CachedSRDataset
        from torch.utils.data import DataLoader

        # Resolve cache directory
        cache_dir = args.cache_dir
        if cache_dir is None:
            cache_dir = os.path.join(
                config['dataset']['val']['root'], 'cached_features_val'
            )
        print(f"  Cache dir: {cache_dir}")

        val_dataset = CachedSRDataset(
            feature_dir  = cache_dir,
            augment      = False,   # NO augmentation for validation
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

    else:
        # Live mode: standard dataset
        from src.data import create_dataloaders
        dataset_cfg = config['dataset']
        data_root   = args.data_root or dataset_cfg['val']['root']

        _, val_loader = create_dataloaders(
            train_hr_dir  = os.path.join(data_root, args.hr_subdir),
            train_lr_dir  = os.path.join(data_root, args.lr_subdir),
            val_hr_dir    = os.path.join(data_root, args.hr_subdir),
            val_lr_dir    = os.path.join(data_root, args.lr_subdir),
            batch_size    = args.batch_size,
            num_workers   = args.num_workers,
            lr_patch_size = 64,
            scale         = args.scale,
            repeat_factor = 1,
        )

    # ── Run validation ────────────────────────────────────────────────────
    output_dir = Path(args.output_dir) if args.save_images else None
    metrics = validate(
        model          = model,
        val_loader     = val_loader,
        device         = device,
        cached_mode    = cached_mode,
        crop_border    = args.crop_border,
        test_y_channel = args.test_y_channel,
        save_images    = args.save_images,
        output_dir     = output_dir,
    )

    print(f"✓ Validation complete — PSNR: {metrics['psnr']:.4f} dB  SSIM: {metrics['ssim']:.6f}")
    return metrics


if __name__ == '__main__':
    main()
