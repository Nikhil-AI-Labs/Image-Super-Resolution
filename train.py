"""
NTIRE 2025 SR Training Script
==============================
Main training script for Frequency Mixer network.

Features:
- Multi-stage loss scheduling
- Learning rate warmup + cosine annealing
- Gradient clipping for stability
- EMA for stable inference
- TensorBoard logging
- Automatic checkpointing
- **CACHED MODE**: 10-20x faster training with pre-computed expert features

Usage:
    python train.py --config configs/train_config.yaml
    
Resume training:
    python train.py --config configs/train_config.yaml --resume checkpoints/latest.pth

Cached training (10-20x faster!):
    python train.py --config configs/train_config.yaml --cached --cache-dir dataset/DF2K/cached_features_v2

Author: NTIRE SR Team
"""

import sys
import os
from pathlib import Path
import argparse
import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import time
import math
from typing import Dict, Optional, Tuple, Any
import random
import numpy as np

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train SR Frequency Mixer')
    parser.add_argument('--config', type=str, default='configs/train_config.yaml',
                       help='Path to config file')
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint to resume from')
    parser.add_argument('--gpu', type=int, default=None,
                       help='GPU ID to use (overrides config)')
    parser.add_argument('--batch_size', type=int, default=None,
                       help='Batch size (overrides config)')
    parser.add_argument('--epochs', type=int, default=None,
                       help='Total epochs (overrides config)')
    parser.add_argument('--debug', action='store_true',
                       help='Debug mode (small dataset, verbose)')
    # Cached training mode (10-20x faster!)
    parser.add_argument('--cached', action='store_true',
                       help='Use cached expert features for 10-20x speedup')
    parser.add_argument('--cache-dir', type=str, default=None,
                       help='Path to cached features (default: dataset/DF2K/cached_features_v2)')
    return parser.parse_args()


def load_config(config_path: str) -> Dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def set_seed(seed: int, deterministic: bool = False):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.benchmark = True


def get_current_lr(optimizer: torch.optim.Optimizer) -> float:
    """Get current learning rate from optimizer."""
    return optimizer.param_groups[0]['lr']


def get_loss_stage(epoch: int, loss_config: Dict) -> Tuple[int, Dict[str, float], str]:
    """
    Get loss weights for current epoch based on stage.
    
    Args:
        epoch: Current epoch
        loss_config: Loss configuration
        
    Returns:
        (stage_num, weights_dict, stage_name)
    """
    stages = loss_config['stages']
    
    for i, stage in enumerate(stages):
        epoch_range = stage['epochs']
        if epoch_range[0] <= epoch < epoch_range[1]:
            return i + 1, stage['weights'], stage.get('stage_name', f'stage_{i+1}')
    
    # Return last stage if beyond all
    last_stage = stages[-1]
    return len(stages), last_stage['weights'], last_stage.get('stage_name', 'final')


def warmup_lr(optimizer: torch.optim.Optimizer, epoch: int, 
              warmup_epochs: int, warmup_lr: float, base_lr: float):
    """Apply learning rate warmup."""
    if epoch < warmup_epochs:
        lr = warmup_lr + (base_lr - warmup_lr) * epoch / warmup_epochs
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr


def train_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    criterion,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    config: Dict,
    logger = None,
    ema = None
) -> Dict[str, float]:
    """
    Train for one epoch.
    
    Args:
        model: Model to train
        train_loader: Training data loader
        criterion: Loss function
        optimizer: Optimizer
        device: Device
        epoch: Current epoch
        config: Configuration
        logger: TensorBoard logger
        ema: EMA model tracker
        
    Returns:
        Dictionary of average metrics
    """
    model.train()
    
    # Get training config
    gradient_clip = config['training'].get('gradient_clip', 0)
    print_freq = config['logging'].get('print_freq', 50)
    accumulation_steps = config['training'].get('accumulation_steps', 1)
    
    # Get current loss stage and weights
    stage_num, loss_weights, stage_name = get_loss_stage(epoch, config['loss'])
    
    # Configure loss weights
    criterion.set_weights(loss_weights)
    
    # Metrics tracking
    total_loss = 0.0
    loss_components = {}
    num_batches = len(train_loader)
    global_step = epoch * num_batches
    
    # Progress bar
    pbar = tqdm(train_loader, desc=f'Epoch {epoch} [{stage_name}]', ncols=120)
    
    optimizer.zero_grad()
    
    for batch_idx, batch in enumerate(pbar):
        lr_img = batch['lr'].to(device, non_blocking=True)
        hr_img = batch['hr'].to(device, non_blocking=True)
        
        # Forward pass
        sr_img = model(lr_img)
        
        # Ensure correct range
        sr_img = sr_img.clamp(0, 1)
        
        # Calculate loss with components
        loss, components = criterion(sr_img, hr_img, return_components=True)
        
        # Normalize for gradient accumulation
        loss = loss / accumulation_steps
        
        # Backward pass
        loss.backward()
        
        # Accumulate loss components
        for k, v in components.items():
            if k not in loss_components:
                loss_components[k] = 0.0
            loss_components[k] += v if isinstance(v, float) else v.item()
        
        # Update weights every accumulation_steps
        if (batch_idx + 1) % accumulation_steps == 0:
            # Gradient clipping
            if gradient_clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    gradient_clip
                )
            
            optimizer.step()
            optimizer.zero_grad()
            
            # Update EMA
            if ema is not None:
                ema.update(model)
        
        # Track total loss
        total_loss += loss.item() * accumulation_steps
        
        # Update progress bar
        if batch_idx % print_freq == 0:
            current_lr = get_current_lr(optimizer)
            pbar.set_postfix({
                'loss': f"{loss.item() * accumulation_steps:.4f}",
                'lr': f"{current_lr:.2e}"
            })
        
        # Log to TensorBoard
        if logger is not None and batch_idx % print_freq == 0:
            step = global_step + batch_idx
            logger.log_scalar('train/loss_iter', loss.item() * accumulation_steps, step)
            logger.log_learning_rate(get_current_lr(optimizer), step)
    
    # Average metrics
    avg_loss = total_loss / num_batches
    avg_components = {k: v / num_batches for k, v in loss_components.items()}
    
    return {
        'loss': avg_loss,
        'stage': stage_num,
        **avg_components
    }


def train_epoch_cached(
    model: nn.Module,
    train_loader: DataLoader,
    criterion,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    config: Dict,
    logger = None,
    ema = None
) -> Dict[str, float]:
    """
    Train for one epoch using CACHED expert features.
    
    This is 10-20x faster than standard training because expert models
    (HAT, DRCT, GRL, EDSR) are NOT run - their outputs are loaded from disk.
    
    The cached dataset provides:
    - expert_imgs: Dict with pre-computed SR outputs
    - expert_feats: Dict with pre-computed intermediate features
    
    Args:
        model: CompleteEnhancedFusionSR model (with expert_ensemble=None)
        train_loader: CachedSRDataset DataLoader
        criterion: Loss function
        optimizer: Optimizer
        device: Device
        epoch: Current epoch
        config: Configuration
        logger: TensorBoard logger
        ema: EMA model tracker
        
    Returns:
        Dictionary of average metrics
    """
    model.train()
    
    # Get training config
    gradient_clip = config['training'].get('gradient_clip', 0)
    accumulation_steps = config['training'].get('accumulation_steps', 1)
    print_freq = config['training'].get('print_freq', 10)
    
    # Get stage weights — strictly from YAML config
    stage_num, stage_weights, stage_name = get_loss_stage(epoch, config['loss'])
    criterion.set_weights(stage_weights)
    
    # Metrics tracking
    total_loss = 0.0
    loss_components = {}
    num_batches = len(train_loader)
    global_step = epoch * num_batches
    
    # Progress bar
    pbar = tqdm(train_loader, desc=f'Epoch {epoch} [CACHED {stage_name}]', ncols=120)
    
    optimizer.zero_grad()
    
    for batch_idx, batch in enumerate(pbar):
        # Get LR/HR images
        lr_img = batch['lr'].to(device, non_blocking=True)
        hr_img = batch['hr'].to(device, non_blocking=True)
        
        # Get pre-computed expert outputs (loaded from disk!)
        expert_imgs = {k: v.to(device, non_blocking=True) for k, v in batch['expert_imgs'].items()}
        
        # Get pre-computed features if available
        expert_feats = None
        if 'expert_feats' in batch:
            expert_feats = {k: v.to(device, non_blocking=True) for k, v in batch['expert_feats'].items()}
        
        # Forward pass using PRE-COMPUTED features (10-20x faster!)
        sr_img = model.forward_with_precomputed(lr_img, expert_imgs, expert_feats)
        
        # Ensure correct range
        sr_img = sr_img.clamp(0, 1)
        
        # Calculate loss with components
        loss, components = criterion(sr_img, hr_img, return_components=True)
        
        # Normalize for gradient accumulation
        loss = loss / accumulation_steps
        
        # Backward pass
        loss.backward()
        
        # Accumulate loss components
        for k, v in components.items():
            if k not in loss_components:
                loss_components[k] = 0.0
            loss_components[k] += v if isinstance(v, float) else v.item()
        
        # Update weights every accumulation_steps
        if (batch_idx + 1) % accumulation_steps == 0:
            # Gradient clipping
            if gradient_clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    gradient_clip
                )
            
            optimizer.step()
            optimizer.zero_grad()
            
            # Update EMA
            if ema is not None:
                ema.update(model)
        
        # Track total loss
        total_loss += loss.item() * accumulation_steps
        
        # Update progress bar
        if batch_idx % print_freq == 0:
            current_lr = get_current_lr(optimizer)
            pbar.set_postfix({
                'loss': f"{loss.item() * accumulation_steps:.4f}",
                'lr': f"{current_lr:.2e}"
            })
        
        # Log to TensorBoard
        if logger is not None and batch_idx % print_freq == 0:
            step = global_step + batch_idx
            logger.log_scalar('train/loss_iter', loss.item() * accumulation_steps, step)
            logger.log_learning_rate(get_current_lr(optimizer), step)
    
    # Average metrics
    avg_loss = total_loss / num_batches
    avg_components = {k: v / num_batches for k, v in loss_components.items()}
    
    return {
        'loss': avg_loss,
        'stage': stage_num,
        **avg_components
    }


def center_crop_to_common_size(tensors: list) -> list:
    """
    Center crop a list of tensors to the minimum common size.
    
    Args:
        tensors: List of tensors with shape [C, H, W]
        
    Returns:
        List of center-cropped tensors with uniform size
    """
    if not tensors:
        return tensors
    
    # Find minimum height and width
    min_h = min(t.shape[1] for t in tensors)
    min_w = min(t.shape[2] for t in tensors)
    
    cropped = []
    for t in tensors:
        _, h, w = t.shape
        top = (h - min_h) // 2
        left = (w - min_w) // 2
        cropped.append(t[:, top:top + min_h, left:left + min_w])
    
    return cropped


@torch.no_grad()
def validate_epoch(
    model: nn.Module,
    val_loader: DataLoader,
    device: torch.device,
    epoch: int,
    config: Dict,
    logger = None,
    log_images: bool = False,
    ema = None,
    cached_mode: bool = False  # NEW: Handle cached validation
) -> Dict[str, float]:
    """
    Validate for one epoch.
    
    Works in both standard mode (live experts) and cached mode (precomputed features).
    
    Args:
        model: Model to validate
        val_loader: Validation data loader (standard or CachedSRDataset)
        device: Device
        epoch: Current epoch
        config: Configuration
        logger: TensorBoard logger
        log_images: Whether to log images
        ema: EMA model (if using EMA for validation)
        cached_mode: If True, use precomputed expert features from batch
        
    Returns:
        Dictionary of metrics
    """
    from src.utils import MetricCalculator
    
    model.eval()
    
    # Apply EMA weights for validation if available
    if ema is not None:
        ema_backup = ema.save(model)
        ema.apply(model)
    
    # Initialize metric calculator
    val_config = config['validation']
    metric_calc = MetricCalculator(
        crop_border=val_config.get('crop_border', 4),
        test_y_channel=val_config.get('test_y_channel', True)
    )
    
    # For image logging
    images_to_log = {'lr': [], 'sr': [], 'hr': []}
    max_log_images = val_config.get('num_log_images', 4)
    
    pbar = tqdm(val_loader, desc='Validation', ncols=100)
    
    for batch_idx, batch in enumerate(pbar):
        lr_img = batch['lr'].to(device, non_blocking=True)
        hr_img = batch['hr'].to(device, non_blocking=True)
        
        # Forward pass - handle cached vs standard mode
        if cached_mode and 'expert_imgs' in batch:
            # CACHED MODE: Use precomputed expert features
            expert_imgs = {k: v.to(device, non_blocking=True) for k, v in batch['expert_imgs'].items()}
            expert_feats = None
            if 'expert_feats' in batch:
                expert_feats = {k: v.to(device, non_blocking=True) for k, v in batch['expert_feats'].items()}
            sr_img = model.forward_with_precomputed(lr_img, expert_imgs, expert_feats)
        else:
            # STANDARD MODE: Run live experts
            sr_img = model(lr_img)
        
        sr_img = sr_img.clamp(0, 1)
        
        # Update metrics
        metric_calc.update(sr_img, hr_img)
        
        # Collect images for logging
        if log_images and len(images_to_log['lr']) < max_log_images:
            images_to_log['lr'].append(lr_img[0].cpu())
            images_to_log['sr'].append(sr_img[0].cpu())
            images_to_log['hr'].append(hr_img[0].cpu())
        
        # Update progress bar
        metrics = metric_calc.get_metrics()
        pbar.set_postfix({
            'PSNR': f"{metrics['psnr']:.2f}",
            'SSIM': f"{metrics['ssim']:.4f}"
        })
    
    # Get final metrics
    metrics = metric_calc.get_metrics()
    
    # Restore original model weights
    if ema is not None:
        ema.restore(model, ema_backup)
    
    # Log to TensorBoard
    if logger is not None:
        logger.log_scalar('val/psnr', metrics['psnr'], epoch)
        logger.log_scalar('val/ssim', metrics['ssim'], epoch)
        
        if log_images and images_to_log['lr']:
            # Center crop images to common size (validation images may have different sizes)
            lr_cropped = center_crop_to_common_size(images_to_log['lr'])
            sr_cropped = center_crop_to_common_size(images_to_log['sr'])
            hr_cropped = center_crop_to_common_size(images_to_log['hr'])
            
            lr_batch = torch.stack(lr_cropped)
            sr_batch = torch.stack(sr_cropped)
            hr_batch = torch.stack(hr_cropped)
            logger.log_images('val/comparison', lr_batch, sr_batch, hr_batch, epoch)
    
    return metrics


def train(config: Dict, resume_path: Optional[str] = None, args = None):
    """
    Main training function.
    
    Args:
        config: Training configuration
        resume_path: Path to resume checkpoint
        args: Command line arguments
    """
    # Imports
    from src.data import create_dataloaders
    from src.models.fusion_network import FrequencyAwareFusion
    from src.models.enhanced_fusion_v2 import CompleteEnhancedFusionSR
    from src.losses import CombinedLoss, PYWT_AVAILABLE
    from src.utils import CheckpointManager, TensorBoardLogger, EMAModel
    
    print("\n" + "=" * 70)
    print("NTIRE 2025 SR TRAINING - FREQUENCY MIXER")
    print("=" * 70)
    print(f"Experiment: {config['experiment_name']}")
    print(f"Description: {config['description']}")
    print("=" * 70 + "\n")
    
    # Set seed
    set_seed(config.get('seed', 42), config.get('deterministic', False))
    
    # Device setup
    gpu_id = config['hardware']['gpu_ids'][0]
    if args and args.gpu is not None:
        gpu_id = args.gpu
    device = torch.device(f'cuda:{gpu_id}' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(gpu_id)}")
        print(f"Memory: {torch.cuda.get_device_properties(gpu_id).total_memory / 1e9:.1f} GB")
        # Apply cudnn benchmark setting from config (enables kernel auto-tuning)
        cudnn_benchmark = config['hardware'].get('cudnn_benchmark', True)
        torch.backends.cudnn.benchmark = cudnn_benchmark
        print(f"CUDNN Benchmark: {cudnn_benchmark}")
    
    # ========================================================================
    # CREATE DATALOADERS
    # ========================================================================
    print("\n" + "-" * 40)
    print("Creating Dataloaders...")
    print("-" * 40)
    
    dataset_config = config['dataset']
    batch_size = args.batch_size if args and args.batch_size else config['training']['batch_size']
    
    # Check for cached training mode (10-20x faster!)
    cached_mode = args and args.cached
    
    if cached_mode:
        # CACHED MODE: Use pre-computed expert features
        from src.data import CachedSRDataset, create_cached_dataloader
        
        # Determine cache directories
        train_cache_dir = os.path.join(dataset_config['train']['root'], 'cached_features_train')
        val_cache_dir = os.path.join(dataset_config['val']['root'], 'cached_features_val')
        
        # Override with command line if provided
        if args.cache_dir:
            train_cache_dir = args.cache_dir
            # Also check for val cache in same parent
            val_cache_dir = os.path.join(os.path.dirname(args.cache_dir), 'cached_features_val')
        
        print(f"\n  *** CACHED TRAINING MODE ENABLED ***")
        print(f"  Train cache: {train_cache_dir}")
        print(f"  Val cache:   {val_cache_dir}")
        print(f"  Expected speedup: 10-20x faster!\n")
        
        # Verify cache directories exist
        if not os.path.exists(train_cache_dir):
            raise RuntimeError(
                f"Train cache not found: {train_cache_dir}\n"
                f"Run: python scripts/extract_features_balanced.py"
            )
        if not os.path.exists(val_cache_dir):
            raise RuntimeError(
                f"Validation cache not found: {val_cache_dir}\n"
                f"Run: python scripts/extract_features_balanced.py\n"
                f"Both train AND val must be cached!"
            )
        
        # TRAIN LOADER (cached)
        train_loader = create_cached_dataloader(
            feature_dir=train_cache_dir,
            batch_size=batch_size,
            num_workers=config['training']['num_workers'],
            augment=True,
            repeat_factor=dataset_config['repeat_factor'],
            pin_memory=config['training']['pin_memory'],
            prefetch_factor=config['training'].get('prefetch_factor', 4),
            persistent_workers=config['training'].get('persistent_workers', True),
            load_features=True  # Need features for collaborative learning
        )
        
        # VALIDATION LOADER (cached) - CRITICAL FIX!
        val_loader = create_cached_dataloader(
            feature_dir=val_cache_dir,
            batch_size=1,  # Val uses batch_size=1 for consistent metrics
            num_workers=4,
            augment=False,  # No augmentation for validation
            repeat_factor=1,
            pin_memory=config['training']['pin_memory'],
            prefetch_factor=2,
            persistent_workers=True,
            load_features=True
        )
        
        print(f"  Train loader: {len(train_loader)} batches")
        print(f"  Val loader:   {len(val_loader)} batches\n")
    else:
        # Standard mode: compute expert features live
        train_loader, val_loader = create_dataloaders(
            train_hr_dir=os.path.join(dataset_config['train']['root'], 
                                       dataset_config['train'].get('hr_subdir', 'train_HR')),
            train_lr_dir=os.path.join(dataset_config['train']['root'], 
                                       dataset_config['train'].get('lr_subdir', 'train_LR')),
            val_hr_dir=os.path.join(dataset_config['val']['root'], 
                                     dataset_config['val'].get('hr_subdir', 'val_HR')),
            val_lr_dir=os.path.join(dataset_config['val']['root'], 
                                     dataset_config['val'].get('lr_subdir', 'val_LR')),
            batch_size=batch_size,
            num_workers=config['training']['num_workers'],
            lr_patch_size=dataset_config['lr_patch_size'],
            scale=dataset_config['scale'],
            pin_memory=config['training']['pin_memory'],
            repeat_factor=dataset_config['repeat_factor'],
            prefetch_factor=config['training'].get('prefetch_factor', 2),
            persistent_workers=config['training'].get('persistent_workers', True)
        )
    
    # ========================================================================
    # CREATE MODEL
    # ========================================================================
    print("\n" + "-" * 40)
    print("Creating Model...")
    print("-" * 40)
    
    model_config = config['model']
    fusion_config = model_config['fusion']
    
    # Check if we should load expert models (full EnhancedFusionSR with ALL improvements)
    # In CACHED MODE: Don't load experts - use precomputed features instead
    if cached_mode:
        print("\n  *** CACHED MODE: Skipping expert model loading ***")
        print("  Experts (HAT, DRCT, GRL, EDSR) are NOT needed in cached mode!")
        print("  This saves ~131M parameters and significant GPU memory.\n")
        
        # Get improvement settings from config
        improvements = fusion_config.get('improvements', {})
        
        # Create CompleteEnhancedFusionSR with NO experts (expert_ensemble=None)
        model = CompleteEnhancedFusionSR(
            expert_ensemble=None,  # CACHED MODE: no live experts!
            num_experts=fusion_config.get('num_experts', 4),
            fusion_dim=fusion_config.get('fusion_dim', 128),
            refine_channels=fusion_config.get('refine_channels', 128),
            refine_depth=fusion_config.get('refine_depth', 6),
            base_channels=fusion_config.get('base_channels', 64),
            block_size=fusion_config.get('block_size', 8),
            upscale=config['dataset'].get('scale', 4),
            # Improvement toggles
            enable_dynamic_selection=improvements.get('dynamic_expert_selection', True),
            enable_cross_band_attn=improvements.get('cross_band_attention', True),
            enable_adaptive_bands=improvements.get('adaptive_frequency_bands', True),
            enable_multi_resolution=improvements.get('multi_resolution_fusion', True),
            enable_collaborative=improvements.get('collaborative_learning', True),
            enable_edge_enhance=improvements.get('edge_enhancement', True),
        ).to(device)
        
        actual_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"  Model: CompleteEnhancedFusionSR (CACHED MODE)")
        print(f"  Trainable parameters: {actual_params:,}")
        print(f"  Expert parameters: 0 (precomputed)")
        
    elif model_config.get('type') == 'CompleteEnhancedFusionSR' and 'experts' in model_config:
        # Load expert models (single GPU)
        from src.models import expert_loader
        
        print("  Loading 4 expert models (frozen)...")
        expert_configs = model_config['experts']
        expert_weights = {e['name'].lower(): e.get('weight_path') for e in expert_configs if e.get('weight_path')}
        
        # Create ExpertEnsemble — single GPU
        ensemble = expert_loader.ExpertEnsemble(
            upscale=config['dataset'].get('scale', 4),
            device=device,
        )
        load_results = ensemble.load_all_experts(
            checkpoint_paths=expert_weights,
            freeze=True
        )
        
        # Print load status
        for name, success in load_results.items():
            if success:
                print(f"    ✓ {name}: loaded and frozen")
            else:
                print(f"    ⚠ {name}: using random weights (no checkpoint)")
        
        # Get improvement settings from config
        improvements = fusion_config.get('improvements', {})
        
        # Create CompleteEnhancedFusionSR with 4 experts
        model = CompleteEnhancedFusionSR(
            expert_ensemble=ensemble,
            num_experts=fusion_config.get('num_experts', 4),
            fusion_dim=fusion_config.get('fusion_dim', 128),
            refine_channels=fusion_config.get('refine_channels', 128),
            refine_depth=fusion_config.get('refine_depth', 6),
            base_channels=fusion_config.get('base_channels', 64),
            block_size=fusion_config.get('block_size', 8),
            upscale=config['dataset'].get('scale', 4),
            # Improvement toggles from config
            enable_dynamic_selection=improvements.get('dynamic_expert_selection', True),
            enable_cross_band_attn=improvements.get('cross_band_attention', True),
            enable_adaptive_bands=improvements.get('adaptive_frequency_bands', True),
            enable_multi_resolution=improvements.get('multi_resolution_fusion', True),
            enable_collaborative=improvements.get('collaborative_learning', True),
            enable_edge_enhance=improvements.get('edge_enhancement', True),
        ).to(device)
        
        loaded_count = len([k for k, v in load_results.items() if v])
        print(f"  Created CompleteEnhancedFusionSR with {loaded_count} loaded experts")
        print(f"  Improvements enabled:")
        print(f"    • Dynamic Expert Selection: {improvements.get('dynamic_expert_selection', True)}")
        print(f"    • Cross-Band Attention: {improvements.get('cross_band_attention', True)}")
        print(f"    • Adaptive Frequency Bands: {improvements.get('adaptive_frequency_bands', True)}")
        print(f"    • Multi-Resolution Fusion: {improvements.get('multi_resolution_fusion', True)}")
        print(f"    • Collaborative Learning: {improvements.get('collaborative_learning', True)}")
        print(f"  Trainable parameters: {model.get_trainable_params():,}")
        
    else:
        # Standalone fusion network (for testing only - requires manual expert outputs)
        from src.models.fusion_network import FrequencyAwareFusion
        
        print("  Creating standalone FrequencyAwareFusion...")
        print("  WARNING: Standalone mode requires external expert outputs!")
        model = FrequencyAwareFusion(
            num_experts=fusion_config['num_experts'],
            use_residual=fusion_config.get('use_residual', True),
            use_multiscale=fusion_config.get('use_multiscale', True)
        ).to(device)
    
    # Count trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Trainable parameters: {trainable_params:,}")
    print(f"  Total parameters: {total_params:,}")
    
    # Expert ensemble summary
    if hasattr(model, 'expert_ensemble') and model.expert_ensemble is not None:
        print("\n" + "=" * 60)
        print("EXPERT ENSEMBLE (Single GPU)")
        print("=" * 60)
        ensemble = model.expert_ensemble
        for name in ['hat', 'drct', 'grl', 'edsr']:
            if ensemble._experts_loaded.get(name, False):
                expert_model = getattr(ensemble, name, None)
                if expert_model is not None:
                    dev = next(expert_model.parameters()).device
                    mem = sum(p.numel() * p.element_size() for p in expert_model.parameters()) / 1e9
                    print(f"  ✓ {name.upper():6s}: {dev} ({mem:.2f} GB)")
        print("=" * 60)
    
    # ========================================================================
    # CREATE LOSS
    # ========================================================================
    print("\n" + "-" * 40)
    print("Creating Loss Function...")
    print("-" * 40)
    
    start_epoch = 0  # Default; overwritten if resuming from checkpoint
    
    loss_config = config['loss']
    # Initialize with ZERO weights — the stage system from YAML controls everything
    criterion = CombinedLoss(
        l1_weight=0.0,
        charbonnier_weight=0.0,
        l2_weight=0.0,
        vgg_weight=0.0,
        swt_weight=0.0,
        fft_weight=0.0,
        edge_weight=0.0,
        ssim_weight=0.0,
        clip_weight=0.0,
        use_swt=loss_config.get('swt', {}).get('enabled', True) and PYWT_AVAILABLE,
        use_fft=loss_config.get('fft', {}).get('enabled', True),
        use_clip=loss_config.get('clip', {}).get('enabled', False),
    ).to(device)
    
    # Apply initial stage weights strictly from YAML config
    _, initial_weights, initial_stage_name = get_loss_stage(start_epoch, config['loss'])
    criterion.set_weights(initial_weights)
    print(f"  Initial loss stage: {initial_stage_name}")
    print(f"  Initial weights: {initial_weights}")
    print(f"  SWT Loss: {'Enabled' if PYWT_AVAILABLE else 'Disabled (PyWavelets not installed)'}")
    
    # ========================================================================
    # CREATE OPTIMIZER
    # ========================================================================
    print("\n" + "-" * 40)
    print("Creating Optimizer...")
    print("-" * 40)
    
    opt_config = config['training']['optimizer']
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=opt_config['lr'],
        betas=tuple(opt_config['betas']),
        weight_decay=opt_config['weight_decay'],
        eps=opt_config.get('eps', 1e-8)
    )
    print(f"  Optimizer: AdamW, LR={opt_config['lr']:.2e}")
    
    # ========================================================================
    # CREATE SCHEDULER
    # ========================================================================
    sched_config = config['training']['scheduler']
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=sched_config['T_0'],
        T_mult=sched_config['T_mult'],
        eta_min=sched_config['eta_min']
    )
    print(f"  Scheduler: CosineAnnealingWarmRestarts, T_0={sched_config['T_0']}")
    
    # ========================================================================
    # CREATE EMA
    # ========================================================================
    ema = None
    if config['training'].get('ema', {}).get('enabled', False):
        ema_decay = config['training']['ema'].get('decay', 0.999)
        ema = EMAModel(model, decay=ema_decay, device=str(device))
        print(f"  EMA: Enabled, decay={ema_decay}")
    
    # ========================================================================
    # CREATE CHECKPOINT MANAGER
    # ========================================================================
    print("\n" + "-" * 40)
    print("Initializing Checkpoint Manager...")
    print("-" * 40)
    
    ckpt_config = config['checkpoint']
    checkpoint_manager = CheckpointManager(
        checkpoint_dir=os.path.join(ckpt_config['checkpoint_dir'], config['experiment_name']),
        keep_best_k=ckpt_config['keep_best_k'],
        save_every=ckpt_config['save_every'],
        metric_name=ckpt_config['metric'],
        mode=ckpt_config['mode']
    )
    
    # ========================================================================
    # CREATE LOGGER
    # ========================================================================
    logger = None
    if config['logging']['tensorboard']['enabled']:
        log_dir = os.path.join(
            config['logging']['tensorboard']['log_dir'],
            config['experiment_name']
        )
        logger = TensorBoardLogger(log_dir=log_dir, enabled=True)
    
    # ========================================================================
    # RESUME FROM CHECKPOINT
    # ========================================================================
    start_epoch = 0
    best_psnr = 0.0
    
    if resume_path is not None:
        print(f"\nResuming from: {resume_path}")
        checkpoint = checkpoint_manager.load_checkpoint(
            resume_path,
            model,
            optimizer,
            scheduler,
            load_optimizer=ckpt_config.get('load_optimizer', True),
            device=str(device)
        )
        start_epoch = checkpoint['epoch'] + 1
        if 'metrics' in checkpoint and 'psnr' in checkpoint['metrics']:
            best_psnr = checkpoint['metrics']['psnr']
        print(f"  Resuming from epoch {start_epoch}, best PSNR: {best_psnr:.2f} dB")
    
    # ========================================================================
    # TRAINING LOOP
    # ========================================================================
    total_epochs = args.epochs if args and args.epochs else config['training']['total_epochs']
    validate_every = config['validation']['validate_every']
    log_images_every = config['validation'].get('log_images_every', 10)
    warmup_epochs = config['training']['scheduler'].get('warmup_epochs', 0)
    base_lr = opt_config['lr']
    warmup_lr_val = config['training']['scheduler'].get('warmup_lr', 1e-6)
    
    # Multi-stage loss scheduler for stage-change announcements
    from src.training.multi_stage_scheduler import MultiStageLossScheduler
    loss_scheduler = MultiStageLossScheduler(config['loss']['stages'], current_epoch=start_epoch)
    
    print("\n" + "=" * 70)
    print("STARTING TRAINING")
    print("=" * 70)
    print(f"  Total epochs: {total_epochs}")
    print(f"  Start epoch: {start_epoch}")
    print(f"  Batch size: {batch_size} x {config['training'].get('accumulation_steps', 1)} = "
          f"{batch_size * config['training'].get('accumulation_steps', 1)} effective")
    print(f"  Batches per epoch: {len(train_loader)}")
    print(f"  Validate every: {validate_every} epochs")
    print(f"  Loss stages: {len(config['loss']['stages'])}")
    for i, s in enumerate(config['loss']['stages']):
        active = ' ** ACTIVE **' if s['epochs'][0] <= start_epoch < s['epochs'][1] else ''
        print(f"    Stage {i+1}: [{s['epochs'][0]}-{s['epochs'][1]}) {s['stage_name']}{active}")
    print("=" * 70 + "\n")
    
    # ========================================================================
    # CUDA WARMUP - Run a few forward passes to warmup GPU kernels
    # ========================================================================
    if torch.cuda.is_available():
        print("Warming up CUDA kernels...")
        
        # Print GPU memory BEFORE warmup
        mem_before = torch.cuda.memory_allocated(device) / 1e6
        print(f"  GPU memory before warmup: {mem_before:.1f} MB")
        
        # Verify expert devices (important for multi-GPU) - only in non-cached mode
        if hasattr(model, 'expert_ensemble') and model.expert_ensemble is not None:
            ensemble = model.expert_ensemble
            for name in ['hat', 'drct', 'grl', 'edsr']:
                if ensemble._experts_loaded.get(name):
                    expert = getattr(ensemble, name)
                    first_param = next(expert.parameters())
                    print(f"  Expert {name} on device: {first_param.device}")
        elif cached_mode:
            print("  CACHED MODE: No experts to verify (using precomputed features)")
        
        model.eval()
        warmup_input = torch.randn(1, 3, 64, 64, device=device)
        with torch.no_grad():
            if cached_mode:
                # In cached mode, use forward_with_precomputed with mock data
                print("  Warming up cached mode (no expert inference)...")
                mock_experts = {'hat': torch.randn(1, 3, 256, 256, device=device),
                               'drct': torch.randn(1, 3, 256, 256, device=device),
                               'grl': torch.randn(1, 3, 256, 256, device=device),
                               'edsr': torch.randn(1, 3, 256, 256, device=device)}
                mock_feats = {'hat': torch.randn(1, 180, 64, 64, device=device),
                              'drct': torch.randn(1, 180, 64, 64, device=device),
                              'grl': torch.randn(1, 180, 64, 64, device=device),
                              'edsr': torch.randn(1, 256, 64, 64, device=device)}
                for i in range(3):
                    torch.cuda.synchronize()
                    t0 = time.time()
                    _ = model.forward_with_precomputed(warmup_input, mock_experts, mock_feats)
                    torch.cuda.synchronize()
                    t1 = time.time()
                    print(f"  Warmup pass {i+1}: {t1-t0:.3f}s (cached mode)")
            else:
                for i in range(3):
                    torch.cuda.synchronize()
                    t0 = time.time()
                    _ = model(warmup_input)
                    torch.cuda.synchronize()
                    t1 = time.time()
                    print(f"  Warmup pass {i+1}: {t1-t0:.3f}s")
        
        # Print GPU memory AFTER warmup
        mem_after = torch.cuda.memory_allocated(device) / 1e6
        print(f"  GPU memory after warmup: {mem_after:.1f} MB")
        
        del warmup_input
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        print("  ✓ CUDA warmup complete\n")
    
    for epoch in range(start_epoch, total_epochs):
        epoch_start = time.time()
        
        # Apply warmup
        if epoch < warmup_epochs:
            warmup_lr(optimizer, epoch, warmup_epochs, warmup_lr_val, base_lr)
        
        # Get current stage info
        stage_num, loss_weights, stage_name = get_loss_stage(epoch, config['loss'])
        
        # Stage-change announcement (via MultiStageLossScheduler)
        stage_changed = loss_scheduler.step(epoch)
        if stage_changed or epoch == start_epoch:
            loss_scheduler.print_stage_info()
        
        print(f"\nEpoch {epoch}/{total_epochs-1} [{stage_name}]")
        print(f"  LR: {get_current_lr(optimizer):.2e}")
        print(f"  Weights: " + ", ".join(f"{k}={v:.2f}" for k, v in loss_weights.items() if v > 0))
        
        # Train (use cached or standard training function)
        if cached_mode:
            train_metrics = train_epoch_cached(
                model=model,
                train_loader=train_loader,
                criterion=criterion,
                optimizer=optimizer,
                device=device,
                epoch=epoch,
                config=config,
                logger=logger,
                ema=ema
            )
        else:
            train_metrics = train_epoch(
                model=model,
                train_loader=train_loader,
                criterion=criterion,
                optimizer=optimizer,
                device=device,
                epoch=epoch,
                config=config,
                logger=logger,
                ema=ema
            )
        
        # Step scheduler (after warmup)
        if epoch >= warmup_epochs:
            scheduler.step()
        
        # ==========================================================
        # FIXED VALIDATION & SAVING LOGIC
        # ==========================================================
        val_metrics = None
        is_best = False
        
        # 1. Fetch validate_start from config (default to 0 if missing)
        validate_start = config['validation'].get('validate_start', 0)
        
        # 2. Check if we should validate
        if (epoch >= validate_start and epoch % validate_every == 0) or epoch == total_epochs - 1:
            log_images = (epoch % log_images_every == 0)
            val_metrics = validate_epoch(
                model=model,
                val_loader=val_loader,
                device=device,
                epoch=epoch,
                config=config,
                logger=logger,
                log_images=log_images,
                ema=ema,
                cached_mode=cached_mode
            )
            
            # 3. IMMEDIATELY check if this is the best model so far
            current_psnr = val_metrics['psnr']
            is_best = checkpoint_manager.is_best(current_psnr)
            if is_best:
                best_psnr = current_psnr
                print(f"  ** NEW BEST PSNR: {best_psnr:.2f} dB! **")

        # 4. Save checkpoint: Save if it's a scheduled save OR if it's a new best!
        if checkpoint_manager.should_save(epoch) or is_best or epoch == total_epochs - 1:
            checkpoint_manager.save_checkpoint(
                epoch=epoch,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                metrics=val_metrics,
                is_best=is_best,
                extra_state={'stage': stage_num}
            )
        # ==========================================================
        
        # Log epoch summary to TensorBoard
        if logger is not None:
            logger.log_scalar('train/loss_epoch', train_metrics['loss'], epoch)
            if val_metrics is not None:
                logger.log_metrics(val_metrics, epoch, prefix='val')
        
        # Print summary with GPU memory and speed
        epoch_time = time.time() - epoch_start
        imgs_per_sec = len(train_loader) * batch_size / epoch_time if epoch_time > 0 else 0
        print(f"\nEpoch {epoch} completed in {epoch_time:.1f}s ({imgs_per_sec:.1f} imgs/sec)")
        print(f"  Train loss: {train_metrics['loss']:.4f}")
        if val_metrics is not None:
            print(f"  Val PSNR: {val_metrics['psnr']:.2f} dB | SSIM: {val_metrics['ssim']:.4f}")
            print(f"  Best PSNR: {best_psnr:.2f} dB")
        if torch.cuda.is_available():
            mem_used = torch.cuda.max_memory_allocated(device) / 1e9
            mem_total = torch.cuda.get_device_properties(device).total_mem / 1e9
            print(f"  GPU Memory: {mem_used:.2f}/{mem_total:.1f} GB ({mem_used/mem_total*100:.0f}%)")
            torch.cuda.reset_peak_memory_stats(device)
        # ETA estimate
        epochs_done = epoch - start_epoch + 1
        epochs_remaining = total_epochs - epoch - 1
        if epochs_done > 0 and epochs_remaining > 0:
            avg_epoch_time = (time.time() - epoch_start) if epochs_done == 1 else None
            eta_sec = epochs_remaining * epoch_time
            eta_h = eta_sec / 3600
            print(f"  ETA: {eta_h:.1f}h ({epochs_remaining} epochs remaining)")
        print("-" * 70)
        
        # Clear CUDA cache periodically
        if torch.cuda.is_available() and epoch % config['hardware'].get('empty_cache_every', 100) == 0:
            torch.cuda.empty_cache()
    
    # ========================================================================
    # TRAINING COMPLETE
    # ========================================================================
    print("\n" + "=" * 70)
    print("TRAINING COMPLETE!")
    print("=" * 70)
    print(f"  Best PSNR: {best_psnr:.2f} dB")
    print(f"  Best checkpoint: {checkpoint_manager.get_best_checkpoint()}")
    print("=" * 70 + "\n")
    
    if logger is not None:
        logger.close()
    
    return best_psnr


def main():
    """Main entry point."""
    args = parse_args()
    
    # Load config
    config = load_config(args.config)
    
    # Apply CLI overrides
    if args.gpu is not None:
        config['hardware']['gpu_ids'] = [args.gpu]
    if args.batch_size is not None:
        config['training']['batch_size'] = args.batch_size
    if args.epochs is not None:
        config['training']['total_epochs'] = args.epochs
    
    # Debug mode
    if args.debug:
        config['training']['total_epochs'] = 5
        config['validation']['validate_every'] = 1
        config['checkpoint']['save_every'] = 1
        config['dataset']['repeat_factor'] = 1
        print("\n⚠ DEBUG MODE ENABLED\n")
    
    # Resume path
    resume_path = args.resume or config['checkpoint'].get('resume')
    
    # Start training
    train(config, resume_path, args)


if __name__ == '__main__':
    main()