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

Usage:
    python train.py --config configs/train_config.yaml
    
Resume training:
    python train.py --config configs/train_config.yaml --resume checkpoints/latest.pth

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
        lr_img = batch['lr'].to(device)
        hr_img = batch['hr'].to(device)
        
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


@torch.no_grad()
def validate_epoch(
    model: nn.Module,
    val_loader: DataLoader,
    device: torch.device,
    epoch: int,
    config: Dict,
    logger = None,
    log_images: bool = False,
    ema = None
) -> Dict[str, float]:
    """
    Validate for one epoch.
    
    Args:
        model: Model to validate
        val_loader: Validation data loader
        device: Device
        epoch: Current epoch
        config: Configuration
        logger: TensorBoard logger
        log_images: Whether to log images
        ema: EMA model (if using EMA for validation)
        
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
        lr_img = batch['lr'].to(device)
        hr_img = batch['hr'].to(device)
        
        # Forward pass
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
            lr_batch = torch.stack(images_to_log['lr'])
            sr_batch = torch.stack(images_to_log['sr'])
            hr_batch = torch.stack(images_to_log['hr'])
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
    from src.models.fusion_network import FrequencyAwareFusion, MultiFusionSR
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
    
    # ========================================================================
    # CREATE DATALOADERS
    # ========================================================================
    print("\n" + "-" * 40)
    print("Creating Dataloaders...")
    print("-" * 40)
    
    dataset_config = config['dataset']
    batch_size = args.batch_size if args and args.batch_size else config['training']['batch_size']
    
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
        repeat_factor=dataset_config['repeat_factor']
    )
    
    # ========================================================================
    # CREATE MODEL
    # ========================================================================
    print("\n" + "-" * 40)
    print("Creating Model...")
    print("-" * 40)
    
    model_config = config['model']
    fusion_config = model_config['fusion']
    
    # Check if we should load expert models (full MultiFusionSR)
    if model_config.get('type') == 'MultiFusionSR' and 'experts' in model_config:
        # Load expert models
        from src.models import load_experts, MultiFusionSR
        
        print("  Loading expert models (frozen)...")
        expert_configs = model_config['experts']
        expert_names = [e['name'] for e in expert_configs]
        expert_weights = {e['name']: e['weight_path'] for e in expert_configs}
        
        # Load and freeze experts
        try:
            experts = load_experts(expert_names, expert_weights, verbose=True)
            
            # Move to device and freeze
            for name, expert in experts.items():
                expert.to(device)
                expert.eval()
                for param in expert.parameters():
                    param.requires_grad = False
                print(f"    ✓ {name}: frozen")
            
            # Create MultiFusionSR with experts
            model = MultiFusionSR(
                experts=experts,
                use_teacher=False
            ).to(device)
            print(f"  Created MultiFusionSR with {len(experts)} experts")
            
        except Exception as e:
            print(f"  Warning: Failed to load experts: {e}")
            print("  Falling back to standalone FrequencyAwareFusion...")
            model = FrequencyAwareFusion(
                num_experts=fusion_config['num_experts'],
                use_residual=fusion_config.get('use_residual', True),
                use_multiscale=fusion_config.get('use_multiscale', True)
            ).to(device)
    else:
        # Standalone fusion network (for testing or when experts not available)
        print("  Creating standalone FrequencyAwareFusion...")
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
    
    # ========================================================================
    # CREATE LOSS
    # ========================================================================
    print("\n" + "-" * 40)
    print("Creating Loss Function...")
    print("-" * 40)
    
    loss_config = config['loss']
    criterion = CombinedLoss(
        use_l1=loss_config.get('l1', {}).get('enabled', True),
        use_swt=loss_config.get('swt', {}).get('enabled', True) and PYWT_AVAILABLE,
        use_fft=loss_config.get('fft', {}).get('enabled', True),
        use_ssim=loss_config.get('ssim', {}).get('enabled', True),
        use_vgg=loss_config.get('vgg', {}).get('enabled', False),
        use_edge=loss_config.get('edge', {}).get('enabled', False),
        use_clip=loss_config.get('clip', {}).get('enabled', False),
    ).to(device)
    
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
    
    print("\n" + "=" * 70)
    print("STARTING TRAINING")
    print("=" * 70)
    print(f"  Total epochs: {total_epochs}")
    print(f"  Start epoch: {start_epoch}")
    print(f"  Batch size: {batch_size} x {config['training'].get('accumulation_steps', 1)} = "
          f"{batch_size * config['training'].get('accumulation_steps', 1)} effective")
    print(f"  Batches per epoch: {len(train_loader)}")
    print(f"  Validate every: {validate_every} epochs")
    print("=" * 70 + "\n")
    
    for epoch in range(start_epoch, total_epochs):
        epoch_start = time.time()
        
        # Apply warmup
        if epoch < warmup_epochs:
            warmup_lr(optimizer, epoch, warmup_epochs, warmup_lr_val, base_lr)
        
        # Get current stage info
        stage_num, loss_weights, stage_name = get_loss_stage(epoch, config['loss'])
        
        print(f"\nEpoch {epoch}/{total_epochs-1} [{stage_name}]")
        print(f"  LR: {get_current_lr(optimizer):.2e}")
        print(f"  Weights: " + ", ".join(f"{k}={v:.2f}" for k, v in loss_weights.items() if v > 0))
        
        # Train
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
        
        # Validate
        val_metrics = None
        if epoch % validate_every == 0 or epoch == total_epochs - 1:
            log_images = (epoch % log_images_every == 0)
            val_metrics = validate_epoch(
                model=model,
                val_loader=val_loader,
                device=device,
                epoch=epoch,
                config=config,
                logger=logger,
                log_images=log_images,
                ema=ema
            )
        
        # Save checkpoint
        if checkpoint_manager.should_save(epoch) or epoch == total_epochs - 1:
            is_best = False
            if val_metrics is not None:
                current_psnr = val_metrics['psnr']
                is_best = checkpoint_manager.is_best(current_psnr)
                if is_best:
                    best_psnr = current_psnr
            
            checkpoint_manager.save_checkpoint(
                epoch=epoch,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                metrics=val_metrics,
                is_best=is_best,
                extra_state={'stage': stage_num}
            )
        
        # Log epoch summary to TensorBoard
        if logger is not None:
            logger.log_scalar('train/loss_epoch', train_metrics['loss'], epoch)
            if val_metrics is not None:
                logger.log_metrics(val_metrics, epoch, prefix='val')
        
        # Print summary
        epoch_time = time.time() - epoch_start
        print(f"\nEpoch {epoch} completed in {epoch_time:.1f}s")
        print(f"  Train loss: {train_metrics['loss']:.4f}")
        if val_metrics is not None:
            print(f"  Val PSNR: {val_metrics['psnr']:.2f} dB | SSIM: {val_metrics['ssim']:.4f}")
            print(f"  Best PSNR: {best_psnr:.2f} dB")
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
