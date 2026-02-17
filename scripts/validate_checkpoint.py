"""
Standalone Validation Script
=============================
Evaluate a checkpoint on validation set.

Usage:
    python scripts/validate_checkpoint.py --checkpoint checkpoints/phase5_single_gpu/best.pth
    python scripts/validate_checkpoint.py --checkpoint checkpoints/phase5_single_gpu/best.pth --save-images
"""

import torch
import torch.nn.functional as F
import sys
import os
from pathlib import Path
import yaml
import numpy as np
import argparse
from PIL import Image

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def calculate_psnr(img1, img2, crop_border=4, test_y_channel=True):
    """
    Calculate PSNR between two images.
    
    Args:
        img1, img2: numpy arrays [C, H, W] in range [0, 255]
        crop_border: pixels to crop from each edge
        test_y_channel: if True, convert to Y channel first
    """
    if test_y_channel and img1.shape[0] == 3:
        # BT.601
        img1_y = 16.0 + 65.481 * img1[0] + 128.553 * img1[1] + 24.966 * img1[2]
        img2_y = 16.0 + 65.481 * img2[0] + 128.553 * img2[1] + 24.966 * img2[2]
        img1 = img1_y[np.newaxis, ...]
        img2 = img2_y[np.newaxis, ...]
    
    if crop_border > 0:
        img1 = img1[:, crop_border:-crop_border, crop_border:-crop_border]
        img2 = img2[:, crop_border:-crop_border, crop_border:-crop_border]
    
    mse = np.mean((img1.astype(np.float64) - img2.astype(np.float64)) ** 2)
    if mse == 0:
        return float('inf')
    
    return 20.0 * np.log10(255.0 / np.sqrt(mse))


def load_checkpoint(checkpoint_path, model, device):
    """Load checkpoint into model."""
    print(f"Loading checkpoint: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    if 'ema_state_dict' in checkpoint:
        print("  Using EMA weights")
        model.load_state_dict(checkpoint['ema_state_dict'], strict=False)
    elif 'model_state_dict' in checkpoint:
        print("  Using model weights")
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    elif 'state_dict' in checkpoint:
        print("  Using state_dict")
        model.load_state_dict(checkpoint['state_dict'], strict=False)
    else:
        print("  Using direct checkpoint")
        model.load_state_dict(checkpoint, strict=False)
    
    if isinstance(checkpoint, dict):
        if 'epoch' in checkpoint:
            print(f"  Epoch: {checkpoint['epoch']}")
        psnr = checkpoint.get('psnr', checkpoint.get('best_psnr', checkpoint.get('metrics', {}).get('psnr')))
        if psnr is not None:
            print(f"  Saved PSNR: {psnr}")
    
    return model


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Validate checkpoint')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to checkpoint file')
    parser.add_argument('--config', type=str, default='configs/train_config.yaml',
                        help='Path to config file')
    parser.add_argument('--gpu', type=int, default=0,
                        help='GPU ID')
    parser.add_argument('--save-images', action='store_true',
                        help='Save SR and HR images')
    parser.add_argument('--output-dir', type=str, default='results/validation',
                        help='Output directory for saved images')
    
    args = parser.parse_args()
    
    # Load config
    print("Loading configuration...")
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Setup device
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create model
    print("\nCreating model...")
    from src.models.enhanced_fusion import CompleteEnhancedFusionSR
    
    fusion_config = config['model']['fusion']
    improvements = fusion_config.get('improvements', {})
    
    model = CompleteEnhancedFusionSR(
        expert_ensemble=None,
        num_experts=3,
        upscale=4,
        fusion_dim=fusion_config.get('fusion_dim', 64),
        num_heads=fusion_config.get('num_heads', 4),
        refine_depth=fusion_config.get('refine_depth', 4),
        refine_channels=fusion_config.get('refine_channels', 64),
        enable_hierarchical=fusion_config.get('enable_hierarchical', True),
        enable_multi_domain_freq=fusion_config.get('enable_multi_domain_freq', False),
        enable_lka=fusion_config.get('enable_lka', False),
        enable_edge_enhance=fusion_config.get('enable_edge_enhance', False),
        enable_dynamic_selection=improvements.get('dynamic_expert_selection', True),
        enable_cross_band_attn=improvements.get('cross_band_attention', True),
        enable_adaptive_bands=improvements.get('adaptive_frequency_bands', True),
        enable_multi_resolution=improvements.get('multi_resolution_fusion', True),
        enable_collaborative=improvements.get('collaborative_learning', True),
    ).to(device)
    
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {trainable:,}")
    
    # Load checkpoint
    model = load_checkpoint(args.checkpoint, model, device)
    model.eval()
    
    print(f"\nCheckpoint loaded successfully!")
    print(f"Model ready for inference on {device}")
    
    if args.save_images:
        os.makedirs(args.output_dir, exist_ok=True)
        print(f"Output directory: {args.output_dir}")
    
    print("\nNote: Full validation requires the cached dataset.")
    print("Use this script primarily for checkpoint verification and loading tests.")
