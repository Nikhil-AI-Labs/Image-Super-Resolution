"""
Checkpoint Management
=====================
Save, load, and manage training checkpoints with best-K tracking.

Features:
- Automatic best checkpoint tracking
- Training history logging
- Resume support with optimizer/scheduler state
- Safe checkpoint saving with atomic writes

Author: NTIRE SR Team
"""

import os
import torch
import shutil
from pathlib import Path
from typing import Dict, Optional, List, Any, Union
import json
from datetime import datetime
import tempfile


class CheckpointManager:
    """
    Manages model checkpoints during training.
    
    Features:
    - Save/load checkpoints with all training state
    - Keep best K checkpoints automatically
    - Resume training from any checkpoint
    - Track training history in JSON
    """
    
    def __init__(
        self,
        checkpoint_dir: str,
        keep_best_k: int = 3,
        save_every: int = 10,
        metric_name: str = 'psnr',
        mode: str = 'max'
    ):
        """
        Args:
            checkpoint_dir: Directory to save checkpoints
            keep_best_k: Number of best checkpoints to keep
            save_every: Save checkpoint every N epochs
            metric_name: Metric to track for best checkpoint
            mode: 'max' or 'min' for metric comparison
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        self.keep_best_k = keep_best_k
        self.save_every = save_every
        self.metric_name = metric_name
        self.mode = mode
        
        # Track best checkpoints: List of (metric_value, checkpoint_path)
        self.best_checkpoints: List[tuple] = []
        
        # Training history
        self.history_file = self.checkpoint_dir / 'training_history.json'
        self.history = self._load_history()
        
        print(f"\n{'=' * 70}")
        print("CHECKPOINT MANAGER INITIALIZED")
        print(f"{'=' * 70}")
        print(f"  Checkpoint dir: {self.checkpoint_dir}")
        print(f"  Keep best: {self.keep_best_k}")
        print(f"  Save every: {self.save_every} epochs")
        print(f"  Metric: {self.metric_name} ({self.mode})")
        print(f"{'=' * 70}\n")
    
    def save_checkpoint(
        self,
        epoch: int,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[Any] = None,
        metrics: Optional[Dict[str, float]] = None,
        is_best: bool = False,
        extra_state: Optional[Dict] = None
    ) -> str:
        """
        Save training checkpoint with atomic write for safety.
        
        Args:
            epoch: Current epoch
            model: Model to save
            optimizer: Optimizer state
            scheduler: LR scheduler state (optional)
            metrics: Current metrics
            is_best: Whether this is the best checkpoint
            extra_state: Additional state to save
            
        Returns:
            Path to saved checkpoint
        """
        # Create checkpoint dictionary
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'metrics': metrics or {},
            'timestamp': datetime.now().isoformat(),
        }
        
        if scheduler is not None:
            checkpoint['scheduler_state_dict'] = scheduler.state_dict()
        
        if extra_state is not None:
            checkpoint.update(extra_state)
        
        # Determine checkpoint filename
        if is_best and metrics is not None:
            metric_val = metrics.get(self.metric_name, 0)
            checkpoint_name = f'best_epoch{epoch:04d}_{self.metric_name}{metric_val:.2f}.pth'
        else:
            checkpoint_name = f'checkpoint_epoch{epoch:04d}.pth'
        
        checkpoint_path = self.checkpoint_dir / checkpoint_name
        
        # Atomic save: write to temp file first, then rename
        temp_path = checkpoint_path.with_suffix('.tmp')
        torch.save(checkpoint, temp_path)
        temp_path.rename(checkpoint_path)
        
        print(f"  ✓ Saved checkpoint: {checkpoint_name}")
        
        # Update history
        self.history.append({
            'epoch': epoch,
            'checkpoint': checkpoint_name,
            'metrics': metrics or {},
            'timestamp': checkpoint['timestamp'],
            'is_best': is_best,
        })
        self._save_history()
        
        # Manage best checkpoints
        if is_best and metrics is not None:
            metric_value = metrics.get(self.metric_name, 0)
            self._update_best_checkpoints(metric_value, checkpoint_path)
        
        # Update latest checkpoint symlink/copy
        latest_path = self.checkpoint_dir / 'latest.pth'
        if latest_path.exists():
            latest_path.unlink()
        shutil.copy2(checkpoint_path, latest_path)
        
        return str(checkpoint_path)
    
    def _update_best_checkpoints(self, metric_value: float, checkpoint_path: Path):
        """Update list of best checkpoints and remove old ones."""
        self.best_checkpoints.append((metric_value, str(checkpoint_path)))
        
        # Sort based on mode
        if self.mode == 'max':
            self.best_checkpoints.sort(key=lambda x: x[0], reverse=True)
        else:
            self.best_checkpoints.sort(key=lambda x: x[0])
        
        # Keep only best K
        while len(self.best_checkpoints) > self.keep_best_k:
            _, worst_path = self.best_checkpoints.pop()
            worst_path = Path(worst_path)
            if worst_path.exists() and 'best_' in worst_path.name:
                worst_path.unlink()
                print(f"  ✗ Removed old best: {worst_path.name}")
    
    def load_checkpoint(
        self,
        checkpoint_path: str,
        model: torch.nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
        load_optimizer: bool = True,
        device: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Load checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint file
            model: Model to load weights into
            optimizer: Optimizer to load state into (optional)
            scheduler: Scheduler to load state into (optional)
            load_optimizer: Whether to load optimizer state
            device: Device to map checkpoint to
            
        Returns:
            Checkpoint dictionary
        """
        map_location = device if device else 'cpu'
        checkpoint = torch.load(checkpoint_path, map_location=map_location)
        
        # Load model weights
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"  ✓ Loaded model weights from epoch {checkpoint['epoch']}")
        
        # Load optimizer state
        if load_optimizer and optimizer is not None and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            print(f"  ✓ Loaded optimizer state")
        
        # Load scheduler state
        if scheduler is not None and 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            print(f"  ✓ Loaded scheduler state")
        
        # Print loaded metrics
        if 'metrics' in checkpoint and checkpoint['metrics']:
            metrics = checkpoint['metrics']
            print(f"  ✓ Checkpoint metrics: " + ", ".join(
                f"{k}={v:.4f}" for k, v in metrics.items()
            ))
        
        return checkpoint
    
    def get_best_checkpoint(self) -> Optional[str]:
        """Get path to best checkpoint."""
        if self.best_checkpoints:
            return self.best_checkpoints[0][1]
        
        # Fallback: find best checkpoint file
        best_files = list(self.checkpoint_dir.glob('best_*.pth'))
        if best_files:
            return str(sorted(best_files)[-1])
        return None
    
    def get_latest_checkpoint(self) -> Optional[str]:
        """Get path to latest checkpoint."""
        latest_path = self.checkpoint_dir / 'latest.pth'
        if latest_path.exists():
            return str(latest_path)
        
        # Fallback: find most recent checkpoint
        checkpoints = list(self.checkpoint_dir.glob('checkpoint_epoch*.pth'))
        if checkpoints:
            return str(sorted(checkpoints)[-1])
        return None
    
    def _load_history(self) -> List[Dict]:
        """Load training history from file."""
        if self.history_file.exists():
            try:
                with open(self.history_file, 'r') as f:
                    return json.load(f)
            except json.JSONDecodeError:
                return []
        return []
    
    def _save_history(self):
        """Save training history to file."""
        with open(self.history_file, 'w') as f:
            json.dump(self.history, f, indent=2)
    
    def should_save(self, epoch: int) -> bool:
        """Check if should save checkpoint at this epoch."""
        return epoch % self.save_every == 0 or epoch == 1
    
    def is_best(self, current_metric: float) -> bool:
        """Check if current metric is the best so far."""
        if not self.best_checkpoints:
            return True
        
        best_metric = self.best_checkpoints[0][0]
        
        if self.mode == 'max':
            return current_metric > best_metric
        else:
            return current_metric < best_metric
    
    def get_best_metric(self) -> Optional[float]:
        """Get best metric value so far."""
        if self.best_checkpoints:
            return self.best_checkpoints[0][0]
        return None
    
    def cleanup_old_checkpoints(self, keep_last_n: int = 5):
        """Remove old non-best checkpoints, keeping last N."""
        checkpoints = sorted(self.checkpoint_dir.glob('checkpoint_epoch*.pth'))
        
        # Get best checkpoint paths
        best_paths = set(path for _, path in self.best_checkpoints)
        
        # Remove old checkpoints
        for ckpt in checkpoints[:-keep_last_n]:
            if str(ckpt) not in best_paths:
                ckpt.unlink()
                print(f"  ✗ Removed old checkpoint: {ckpt.name}")


class EMAModel:
    """
    Exponential Moving Average of model parameters.
    
    Helps stabilize training and often improves final performance.
    """
    
    def __init__(
        self,
        model: torch.nn.Module,
        decay: float = 0.999,
        device: Optional[str] = None
    ):
        """
        Args:
            model: Model to track
            decay: EMA decay rate (0.999 typical)
            device: Device for shadow parameters
        """
        self.decay = decay
        self.device = device
        
        # Create shadow parameters
        self.shadow = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()
                if device:
                    self.shadow[name] = self.shadow[name].to(device)
    
    def update(self, model: torch.nn.Module):
        """Update shadow parameters with current model."""
        for name, param in model.named_parameters():
            if name in self.shadow:
                self.shadow[name] = (
                    self.decay * self.shadow[name] +
                    (1.0 - self.decay) * param.data
                )
    
    def apply(self, model: torch.nn.Module):
        """Apply shadow parameters to model."""
        for name, param in model.named_parameters():
            if name in self.shadow:
                param.data.copy_(self.shadow[name])
    
    def restore(self, model: torch.nn.Module, backup: Dict[str, torch.Tensor]):
        """Restore model parameters from backup."""
        for name, param in model.named_parameters():
            if name in backup:
                param.data.copy_(backup[name])
    
    def save(self, model: torch.nn.Module) -> Dict[str, torch.Tensor]:
        """Save current model parameters before applying EMA."""
        backup = {}
        for name, param in model.named_parameters():
            if name in self.shadow:
                backup[name] = param.data.clone()
        return backup


# ============================================================================
# Testing
# ============================================================================

def test_checkpoint_manager():
    """Test checkpoint manager."""
    print("\n" + "=" * 70)
    print("CHECKPOINT MANAGER - TEST")
    print("=" * 70)
    
    import tempfile
    import torch.nn as nn
    
    # Create temporary directory
    temp_dir = Path(tempfile.mkdtemp())
    
    # Create dummy model and optimizer
    model = nn.Sequential(
        nn.Linear(10, 20),
        nn.ReLU(),
        nn.Linear(20, 10)
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10)
    
    # Initialize manager
    manager = CheckpointManager(
        checkpoint_dir=str(temp_dir / 'checkpoints'),
        keep_best_k=2,
        save_every=5,
        metric_name='psnr',
        mode='max'
    )
    
    print("\n--- Test 1: Save Checkpoints ---")
    for epoch in range(1, 6):
        psnr = 30.0 + epoch * 0.5  # Increasing PSNR
        metrics = {'psnr': psnr, 'ssim': 0.9 + epoch * 0.01}
        is_best = manager.is_best(psnr)
        
        manager.save_checkpoint(
            epoch=epoch,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            metrics=metrics,
            is_best=is_best
        )
        print(f"    Epoch {epoch}: PSNR={psnr:.2f}, Best={is_best}")
    
    print("  [PASSED]")
    
    print("\n--- Test 2: Load Checkpoint ---")
    # Create new model
    new_model = nn.Sequential(
        nn.Linear(10, 20),
        nn.ReLU(),
        nn.Linear(20, 10)
    )
    new_optimizer = torch.optim.Adam(new_model.parameters())
    
    latest_path = manager.get_latest_checkpoint()
    assert latest_path is not None, "No latest checkpoint found"
    
    checkpoint = manager.load_checkpoint(
        latest_path,
        new_model,
        new_optimizer,
        load_optimizer=True
    )
    print(f"    Loaded epoch: {checkpoint['epoch']}")
    print("  [PASSED]")
    
    print("\n--- Test 3: Best Checkpoint Tracking ---")
    best_path = manager.get_best_checkpoint()
    best_metric = manager.get_best_metric()
    print(f"    Best checkpoint: {Path(best_path).name if best_path else 'None'}")
    print(f"    Best PSNR: {best_metric:.2f}")
    assert len(manager.best_checkpoints) <= 2, "Too many best checkpoints"
    print("  [PASSED]")
    
    print("\n--- Test 4: History Tracking ---")
    print(f"    Total entries in history: {len(manager.history)}")
    assert len(manager.history) == 5, "History should have 5 entries"
    print("  [PASSED]")
    
    print("\n--- Test 5: EMA Model ---")
    ema = EMAModel(model, decay=0.999)
    
    # Simulate training step
    x = torch.randn(4, 10)
    y = model(x)
    loss = y.sum()
    loss.backward()
    optimizer.step()
    
    # Update EMA
    ema.update(model)
    
    # Apply EMA and restore
    backup = ema.save(model)
    ema.apply(model)
    ema.restore(model, backup)
    print("    EMA update, apply, restore: OK")
    print("  [PASSED]")
    
    # Cleanup
    shutil.rmtree(temp_dir)
    
    print("\n" + "=" * 70)
    print("✓ ALL CHECKPOINT MANAGER TESTS PASSED!")
    print("=" * 70 + "\n")


if __name__ == '__main__':
    test_checkpoint_manager()
