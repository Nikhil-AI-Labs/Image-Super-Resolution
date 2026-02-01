"""
Validation Script
=================
Standalone validation script for evaluating trained models.

Usage:
    python scripts/validate.py --checkpoint checkpoints/best.pth
    python scripts/validate.py --checkpoint checkpoints/best.pth --save_images

Author: NTIRE SR Team
"""

import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
import argparse
from tqdm import tqdm
from typing import Dict
import numpy as np


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Validate SR model')
    
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--data_root', type=str, default='data/DF2K',
                       help='Dataset root directory')
    parser.add_argument('--hr_subdir', type=str, default='val_HR',
                       help='HR images subdirectory')
    parser.add_argument('--lr_subdir', type=str, default='val_LR',
                       help='LR images subdirectory')
    parser.add_argument('--scale', type=int, default=4,
                       help='Upscaling factor')
    parser.add_argument('--batch_size', type=int, default=1,
                       help='Batch size')
    parser.add_argument('--num_workers', type=int, default=2,
                       help='Number of workers')
    parser.add_argument('--save_images', action='store_true',
                       help='Save SR images')
    parser.add_argument('--output_dir', type=str, default='results/validation',
                       help='Output directory')
    parser.add_argument('--gpu', type=int, default=0,
                       help='GPU ID')
    parser.add_argument('--test_y_channel', action='store_true', default=True,
                       help='Calculate metrics on Y channel')
    parser.add_argument('--crop_border', type=int, default=4,
                       help='Border pixels to crop')
    
    return parser.parse_args()


@torch.no_grad()
def validate(
    model: torch.nn.Module,
    val_loader,
    device: torch.device,
    crop_border: int = 4,
    test_y_channel: bool = True,
    save_images: bool = False,
    output_dir: Path = None
) -> Dict[str, float]:
    """
    Run validation.
    
    Args:
        model: Model to validate
        val_loader: Validation data loader
        device: Device
        crop_border: Border to crop
        test_y_channel: Test on Y channel
        save_images: Save SR images
        output_dir: Output directory
        
    Returns:
        Dictionary of metrics
    """
    from src.utils import MetricCalculator
    
    model.eval()
    
    # Initialize metrics
    metric_calc = MetricCalculator(
        crop_border=crop_border,
        test_y_channel=test_y_channel
    )
    
    # Create output directory
    if save_images and output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'=' * 70}")
    print("VALIDATION")
    print(f"{'=' * 70}")
    print(f"  Total samples: {len(val_loader.dataset)}")
    print(f"  Crop border: {crop_border}")
    print(f"  Y channel: {test_y_channel}")
    print(f"{'=' * 70}\n")
    
    # Per-image metrics for detailed analysis
    per_image_metrics = []
    
    pbar = tqdm(val_loader, desc='Validating', ncols=100)
    for batch_idx, batch in enumerate(pbar):
        lr = batch['lr'].to(device)
        hr = batch['hr'].to(device)
        filename = batch['filename'][0] if isinstance(batch['filename'], (list, tuple)) else batch['filename']
        
        # Forward pass
        sr = model(lr)
        sr = sr.clamp(0, 1)
        
        # Calculate per-image metrics
        from src.utils import calculate_psnr, calculate_ssim
        psnr = calculate_psnr(sr[0], hr[0], crop_border, test_y_channel)
        ssim = calculate_ssim(sr[0], hr[0], crop_border, test_y_channel)
        
        per_image_metrics.append({
            'filename': filename,
            'psnr': psnr,
            'ssim': ssim
        })
        
        # Update running average
        metric_calc.update(sr, hr)
        
        # Update progress bar
        current = metric_calc.get_metrics()
        pbar.set_postfix({
            'PSNR': f"{current['psnr']:.2f}",
            'SSIM': f"{current['ssim']:.4f}"
        })
        
        # Save images
        if save_images and output_dir:
            import torchvision
            sr_img = sr[0].cpu()
            save_path = output_dir / f"{Path(filename).stem}_SR.png"
            torchvision.utils.save_image(sr_img, save_path)
    
    # Final metrics
    final_metrics = metric_calc.get_metrics()
    
    print(f"\n{'=' * 70}")
    print("VALIDATION RESULTS")
    print(f"{'=' * 70}")
    print(f"  PSNR: {final_metrics['psnr']:.2f} dB")
    print(f"  SSIM: {final_metrics['ssim']:.4f}")
    
    # Print best/worst images
    sorted_by_psnr = sorted(per_image_metrics, key=lambda x: x['psnr'], reverse=True)
    print(f"\n  Top 5 images:")
    for m in sorted_by_psnr[:5]:
        print(f"    {m['filename']}: {m['psnr']:.2f} dB")
    
    print(f"\n  Worst 5 images:")
    for m in sorted_by_psnr[-5:]:
        print(f"    {m['filename']}: {m['psnr']:.2f} dB")
    
    print(f"{'=' * 70}\n")
    
    return final_metrics


def main():
    """Main function."""
    args = parse_args()
    
    # Device
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")
    
    # Load model
    print(f"Loading checkpoint: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    
    # Create model
    from src.models.fusion_network import FrequencyAwareFusion
    model = FrequencyAwareFusion(
        num_experts=3,
        use_residual=True,
        use_multiscale=True
    ).to(device)
    
    # Load weights
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"  Loaded from epoch {checkpoint['epoch']}")
    if 'metrics' in checkpoint:
        print(f"  Checkpoint metrics: {checkpoint['metrics']}")
    
    # Create dataloader
    print("\nCreating validation dataloader...")
    from src.data import create_dataloaders
    
    _, val_loader = create_dataloaders(
        train_hr_dir=os.path.join(args.data_root, args.hr_subdir),
        train_lr_dir=os.path.join(args.data_root, args.lr_subdir),
        val_hr_dir=os.path.join(args.data_root, args.hr_subdir),
        val_lr_dir=os.path.join(args.data_root, args.lr_subdir),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        lr_patch_size=64,
        scale=args.scale,
        repeat_factor=1
    )
    
    # Run validation
    output_dir = Path(args.output_dir) if args.save_images else None
    metrics = validate(
        model=model,
        val_loader=val_loader,
        device=device,
        crop_border=args.crop_border,
        test_y_channel=args.test_y_channel,
        save_images=args.save_images,
        output_dir=output_dir
    )
    
    print("✓ Validation complete!")
    
    return metrics


if __name__ == '__main__':
    main()
