"""
TensorBoard Logger
==================
Logging utilities for training visualization.

Features:
- Scalar metrics (loss, PSNR, SSIM, learning rate)
- Image comparisons (LR vs SR vs HR)
- Gradient histograms
- Training progress tracking

Author: NTIRE SR Team
"""

import torch
import numpy as np
from pathlib import Path
from typing import Dict, Optional, Union, List
from datetime import datetime

# Try importing TensorBoard
try:
    from torch.utils.tensorboard import SummaryWriter
    import torchvision.utils as vutils
    TENSORBOARD_AVAILABLE = True
except ImportError:
    TENSORBOARD_AVAILABLE = False
    print("Warning: TensorBoard not available. Install with: pip install tensorboard")


class TensorBoardLogger:
    """
    TensorBoard logger for training visualization.
    
    Logs:
    - Scalar metrics (loss, PSNR, SSIM)
    - Learning rate schedule
    - Images (LR, SR, HR comparisons)
    - Model gradient statistics
    """
    
    def __init__(
        self,
        log_dir: str,
        experiment_name: Optional[str] = None,
        enabled: bool = True
    ):
        """
        Args:
            log_dir: Directory for TensorBoard logs
            experiment_name: Optional experiment name suffix
            enabled: Whether logging is enabled
        """
        self.enabled = enabled and TENSORBOARD_AVAILABLE
        
        if self.enabled:
            # Add timestamp to log dir for unique runs
            if experiment_name:
                log_dir = Path(log_dir) / experiment_name
            
            self.log_dir = Path(log_dir)
            self.log_dir.mkdir(parents=True, exist_ok=True)
            
            self.writer = SummaryWriter(str(self.log_dir))
            
            print(f"\n{'=' * 70}")
            print("TENSORBOARD LOGGER INITIALIZED")
            print(f"{'=' * 70}")
            print(f"  Log dir: {self.log_dir}")
            print(f"  View logs: tensorboard --logdir {self.log_dir.parent}")
            print(f"{'=' * 70}\n")
        else:
            self.writer = None
            if not TENSORBOARD_AVAILABLE:
                print("Warning: TensorBoard logging disabled (not installed)")
    
    def log_scalar(
        self,
        tag: str,
        value: float,
        step: int
    ):
        """
        Log scalar value.
        
        Args:
            tag: Tag name (e.g., 'train/loss', 'val/psnr')
            value: Scalar value
            step: Global step (epoch or iteration)
        """
        if self.enabled and self.writer is not None:
            self.writer.add_scalar(tag, value, step)
    
    def log_scalars(
        self,
        main_tag: str,
        tag_scalar_dict: Dict[str, float],
        step: int
    ):
        """
        Log multiple scalars under same main tag.
        
        Args:
            main_tag: Main tag (e.g., 'loss_components')
            tag_scalar_dict: Dictionary of tag->scalar values
            step: Global step
        """
        if self.enabled and self.writer is not None:
            self.writer.add_scalars(main_tag, tag_scalar_dict, step)
    
    def log_images(
        self,
        tag: str,
        lr_images: torch.Tensor,
        sr_images: torch.Tensor,
        hr_images: torch.Tensor,
        step: int,
        max_images: int = 4,
        normalize: bool = False
    ):
        """
        Log image comparison (LR → SR vs HR).
        
        Creates a grid showing: [LR (upsampled) | SR | HR]
        
        Args:
            tag: Tag name
            lr_images: LR images [B, C, H, W]
            sr_images: SR images [B, C, H, W]
            hr_images: HR images [B, C, H, W]
            step: Global step
            max_images: Maximum number of images to log
            normalize: Whether to normalize images
        """
        if not self.enabled or self.writer is None:
            return
        
        # Limit number of images
        batch_size = min(lr_images.size(0), max_images)
        lr_images = lr_images[:batch_size].detach().cpu()
        sr_images = sr_images[:batch_size].detach().cpu()
        hr_images = hr_images[:batch_size].detach().cpu()
        
        # Ensure values in [0, 1]
        lr_images = lr_images.clamp(0, 1)
        sr_images = sr_images.clamp(0, 1)
        hr_images = hr_images.clamp(0, 1)
        
        # Upsample LR to match SR/HR size for visual comparison
        lr_upsampled = torch.nn.functional.interpolate(
            lr_images,
            size=sr_images.shape[-2:],
            mode='nearest'
        )
        
        # Create comparison grid: [LR_up | SR | HR] for each image
        comparison = []
        for i in range(batch_size):
            comparison.extend([
                lr_upsampled[i],
                sr_images[i],
                hr_images[i]
            ])
        
        # Stack into grid (3 columns: LR, SR, HR)
        grid = vutils.make_grid(
            comparison,
            nrow=3,
            normalize=normalize,
            scale_each=False,
            padding=4,
            pad_value=1.0  # White padding
        )
        
        self.writer.add_image(tag, grid, step)
    
    def log_single_image(
        self,
        tag: str,
        image: torch.Tensor,
        step: int
    ):
        """
        Log a single image.
        
        Args:
            tag: Tag name
            image: Image tensor [C, H, W] or [B, C, H, W]
            step: Global step
        """
        if not self.enabled or self.writer is None:
            return
        
        if image.ndim == 4:
            image = image[0]
        
        image = image.detach().cpu().clamp(0, 1)
        self.writer.add_image(tag, image, step)
    
    def log_histogram(
        self,
        tag: str,
        values: torch.Tensor,
        step: int
    ):
        """
        Log histogram of values.
        
        Args:
            tag: Tag name
            values: Tensor values
            step: Global step
        """
        if self.enabled and self.writer is not None:
            self.writer.add_histogram(tag, values.detach().cpu(), step)
    
    def log_model_gradients(
        self,
        model: torch.nn.Module,
        step: int,
        prefix: str = 'gradients'
    ):
        """
        Log model gradient statistics.
        
        Args:
            model: Model to log gradients from
            step: Global step
            prefix: Tag prefix
        """
        if not self.enabled or self.writer is None:
            return
        
        for name, param in model.named_parameters():
            if param.grad is not None:
                grad = param.grad.detach()
                
                # Log gradient norm
                grad_norm = grad.norm().item()
                self.log_scalar(f'{prefix}_norm/{name}', grad_norm, step)
                
                # Optionally log histogram (expensive)
                # self.writer.add_histogram(f'{prefix}/{name}', grad, step)
    
    def log_learning_rate(
        self,
        lr: float,
        step: int
    ):
        """
        Log current learning rate.
        
        Args:
            lr: Learning rate value
            step: Global step
        """
        self.log_scalar('learning_rate', lr, step)
    
    def log_loss_components(
        self,
        loss_dict: Dict[str, float],
        step: int,
        prefix: str = 'train'
    ):
        """
        Log individual loss components.
        
        Args:
            loss_dict: Dictionary of loss component name -> value
            step: Global step
            prefix: Tag prefix ('train' or 'val')
        """
        if not self.enabled or self.writer is None:
            return
        
        for loss_name, loss_value in loss_dict.items():
            if isinstance(loss_value, torch.Tensor):
                loss_value = loss_value.item()
            self.log_scalar(f'{prefix}/loss_{loss_name}', loss_value, step)
    
    def log_metrics(
        self,
        metrics: Dict[str, float],
        step: int,
        prefix: str = 'val'
    ):
        """
        Log metrics (PSNR, SSIM, etc.).
        
        Args:
            metrics: Dictionary of metric name -> value
            step: Global step
            prefix: Tag prefix
        """
        for name, value in metrics.items():
            self.log_scalar(f'{prefix}/{name}', value, step)
    
    def log_epoch_summary(
        self,
        epoch: int,
        train_loss: float,
        val_metrics: Optional[Dict[str, float]] = None,
        lr: float = None
    ):
        """
        Log epoch summary.
        
        Args:
            epoch: Current epoch
            train_loss: Training loss
            val_metrics: Validation metrics
            lr: Current learning rate
        """
        self.log_scalar('train/loss_epoch', train_loss, epoch)
        
        if val_metrics is not None:
            for name, value in val_metrics.items():
                self.log_scalar(f'val/{name}', value, epoch)
        
        if lr is not None:
            self.log_learning_rate(lr, epoch)
    
    def log_text(
        self,
        tag: str,
        text: str,
        step: int
    ):
        """
        Log text.
        
        Args:
            tag: Tag name
            text: Text to log
            step: Global step
        """
        if self.enabled and self.writer is not None:
            self.writer.add_text(tag, text, step)
    
    def flush(self):
        """Flush pending writes to disk."""
        if self.enabled and self.writer is not None:
            self.writer.flush()
    
    def close(self):
        """Close the logger."""
        if self.enabled and self.writer is not None:
            self.writer.close()
            print("\n✓ TensorBoard logger closed")
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()


class ProgressLogger:
    """
    Simple console progress logger as fallback.
    """
    
    def __init__(self, experiment_name: str = "training"):
        self.experiment_name = experiment_name
        self.start_time = datetime.now()
    
    def log_epoch(
        self,
        epoch: int,
        total_epochs: int,
        train_loss: float,
        val_psnr: Optional[float] = None,
        val_ssim: Optional[float] = None,
        lr: Optional[float] = None
    ):
        """Log epoch progress to console."""
        elapsed = datetime.now() - self.start_time
        
        msg = f"Epoch [{epoch}/{total_epochs}] "
        msg += f"Loss: {train_loss:.4f}"
        
        if val_psnr is not None:
            msg += f" | PSNR: {val_psnr:.2f} dB"
        if val_ssim is not None:
            msg += f" | SSIM: {val_ssim:.4f}"
        if lr is not None:
            msg += f" | LR: {lr:.2e}"
        
        msg += f" | Elapsed: {elapsed}"
        print(msg)


# ============================================================================
# Testing
# ============================================================================

def test_logger():
    """Test TensorBoard logger."""
    print("\n" + "=" * 70)
    print("TENSORBOARD LOGGER - TEST")
    print("=" * 70)
    
    import tempfile
    import shutil
    
    # Create temporary directory
    temp_dir = Path(tempfile.mkdtemp())
    
    # Initialize logger
    logger = TensorBoardLogger(
        log_dir=str(temp_dir / 'logs'),
        experiment_name='test_run',
        enabled=True
    )
    
    if not logger.enabled:
        print("  TensorBoard not available, skipping tests")
        shutil.rmtree(temp_dir)
        return
    
    print("\n--- Test 1: Log Scalars ---")
    for step in range(10):
        logger.log_scalar('train/loss', 1.0 - step * 0.1, step)
        logger.log_scalar('train/psnr', 30.0 + step * 0.5, step)
        logger.log_scalar('train/ssim', 0.9 + step * 0.01, step)
    print("  Logged 10 scalar steps")
    print("  [PASSED]")
    
    print("\n--- Test 2: Log Images ---")
    lr = torch.rand(2, 3, 64, 64)
    sr = torch.rand(2, 3, 256, 256)
    hr = torch.rand(2, 3, 256, 256)
    logger.log_images('images/comparison', lr, sr, hr, step=0)
    print("  Logged image comparison")
    print("  [PASSED]")
    
    print("\n--- Test 3: Log Learning Rate ---")
    logger.log_learning_rate(1e-4, step=0)
    print("  [PASSED]")
    
    print("\n--- Test 4: Log Loss Components ---")
    loss_dict = {'l1': 0.1, 'swt': 0.05, 'total': 0.15}
    logger.log_loss_components(loss_dict, step=0)
    print("  [PASSED]")
    
    print("\n--- Test 5: Log Metrics ---")
    metrics = {'psnr': 32.5, 'ssim': 0.95}
    logger.log_metrics(metrics, step=0, prefix='val')
    print("  [PASSED]")
    
    print("\n--- Test 6: Flush and Close ---")
    logger.flush()
    logger.close()
    print("  [PASSED]")
    
    # Cleanup
    shutil.rmtree(temp_dir)
    
    print("\n" + "=" * 70)
    print("✓ ALL LOGGER TESTS PASSED!")
    print("=" * 70 + "\n")


if __name__ == '__main__':
    test_logger()
