"""
Multi-Stage Loss Scheduler for Championship Training
=====================================================
Automatically adjusts loss weights based on training stage.

Usage:
    from src.training.multi_stage_scheduler import MultiStageLossScheduler
    
    stages = config['loss']['stages']
    scheduler = MultiStageLossScheduler(stages)
    
    # In training loop:
    if scheduler.step(epoch):
        scheduler.print_stage_info()
    
    weights = scheduler.get_loss_weights()
    total_loss = weights['l1'] * l1_loss + weights['swt'] * swt_loss + ...
"""

from typing import Dict, List


class MultiStageLossScheduler:
    """
    Manages loss weight transitions across training stages.
    
    Args:
        stages: List of stage configurations from YAML config
        current_epoch: Starting epoch (for resume training)
    
    Example:
        stages = [
            {
                'epochs': [0, 80],
                'stage_name': 'foundation',
                'weights': {'l1': 1.0, 'swt': 0.0, 'fft': 0.0}
            },
            ...
        ]
    """
    
    def __init__(self, stages: List[Dict], current_epoch: int = 0):
        self.stages = stages
        self.current_epoch = current_epoch
        self.current_stage_idx = self._get_stage_idx(current_epoch)
        self._validate_stages()
    
    def _validate_stages(self):
        """Validate stage configuration."""
        if not self.stages:
            raise ValueError("Stages list cannot be empty")
        
        for i in range(len(self.stages) - 1):
            current_end = self.stages[i]['epochs'][1]
            next_start = self.stages[i + 1]['epochs'][0]
            if current_end != next_start:
                raise ValueError(
                    f"Stage epochs not continuous: "
                    f"Stage {i} ends at {current_end}, "
                    f"Stage {i+1} starts at {next_start}"
                )
    
    def _get_stage_idx(self, epoch: int) -> int:
        """Get stage index for given epoch."""
        for idx, stage in enumerate(self.stages):
            start, end = stage['epochs']
            if start <= epoch < end:
                return idx
        return len(self.stages) - 1
    
    def step(self, epoch: int) -> bool:
        """
        Update to new epoch.
        
        Returns:
            True if stage changed, False otherwise
        """
        self.current_epoch = epoch
        old_stage = self.current_stage_idx
        self.current_stage_idx = self._get_stage_idx(epoch)
        return old_stage != self.current_stage_idx
    
    def get_loss_weights(self) -> Dict[str, float]:
        """Get current loss weights."""
        stage = self.stages[self.current_stage_idx]
        return stage['weights'].copy()
    
    def get_stage_info(self) -> Dict:
        """Get current stage information."""
        stage = self.stages[self.current_stage_idx]
        return {
            'stage_idx': self.current_stage_idx,
            'stage_name': stage['stage_name'],
            'description': stage.get('description', ''),
            'epoch_range': stage['epochs'],
            'weights': stage['weights'].copy(),
            'progress': self._get_stage_progress()
        }
    
    def _get_stage_progress(self) -> float:
        """Calculate progress within current stage (0.0 to 1.0)."""
        stage = self.stages[self.current_stage_idx]
        start, end = stage['epochs']
        if end <= start:
            return 1.0
        progress = (self.current_epoch - start) / (end - start)
        return min(max(progress, 0.0), 1.0)
    
    def print_stage_info(self):
        """Print current stage information."""
        info = self.get_stage_info()
        print(f"\n{'=' * 70}")
        print(f"  LOSS STAGE TRANSITION: {info['stage_name'].upper()}")
        print(f"{'=' * 70}")
        print(f"  Description: {info['description']}")
        print(f"  Epoch Range: {info['epoch_range'][0]} - {info['epoch_range'][1]}")
        print(f"  Current Epoch: {self.current_epoch}")
        print(f"  Progress: {info['progress']*100:.1f}%")
        print(f"\n  Active Loss Weights:")
        for loss_name, weight in info['weights'].items():
            if weight > 0:
                print(f"    {loss_name}: {weight:.3f}")
        print(f"{'=' * 70}\n")


def test_multi_stage_scheduler():
    """Test the multi-stage loss scheduler."""
    print("=" * 70)
    print("TESTING MULTI-STAGE LOSS SCHEDULER")
    print("=" * 70 + "\n")
    
    stages = [
        {
            'epochs': [0, 80],
            'stage_name': 'foundation_psnr',
            'description': 'Build strong pixel-level reconstruction',
            'weights': {'l1': 1.0, 'swt': 0.0, 'fft': 0.0, 'ssim': 0.0}
        },
        {
            'epochs': [80, 150],
            'stage_name': 'frequency_refinement',
            'description': 'Enhance frequency with SWT + FFT',
            'weights': {'l1': 0.75, 'swt': 0.20, 'fft': 0.05, 'ssim': 0.0}
        },
        {
            'epochs': [150, 200],
            'stage_name': 'detail_enhancement',
            'description': 'Final edge/texture refinement',
            'weights': {'l1': 0.60, 'swt': 0.25, 'fft': 0.10, 'ssim': 0.05}
        }
    ]
    
    scheduler = MultiStageLossScheduler(stages)
    
    test_epochs = [0, 40, 79, 80, 100, 149, 150, 180, 199]
    
    print("Testing epoch progression:\n")
    all_ok = True
    for epoch in test_epochs:
        changed = scheduler.step(epoch)
        info = scheduler.get_stage_info()
        
        print(f"  Epoch {epoch:3d}: Stage '{info['stage_name']}'", end="")
        if changed:
            print("  ** STAGE CHANGED **")
            scheduler.print_stage_info()
        else:
            print(f" (Progress: {info['progress']*100:.0f}%)")
    
    # Verify stage transitions
    scheduler.step(0)
    assert scheduler.get_stage_info()['stage_name'] == 'foundation_psnr'
    scheduler.step(79)
    assert scheduler.get_stage_info()['stage_name'] == 'foundation_psnr'
    scheduler.step(80)
    assert scheduler.get_stage_info()['stage_name'] == 'frequency_refinement'
    scheduler.step(149)
    assert scheduler.get_stage_info()['stage_name'] == 'frequency_refinement'
    scheduler.step(150)
    assert scheduler.get_stage_info()['stage_name'] == 'detail_enhancement'
    scheduler.step(199)
    assert scheduler.get_stage_info()['stage_name'] == 'detail_enhancement'
    
    # Verify weights
    scheduler.step(0)
    w = scheduler.get_loss_weights()
    assert w['l1'] == 1.0 and w['swt'] == 0.0
    
    scheduler.step(80)
    w = scheduler.get_loss_weights()
    assert w['l1'] == 0.75 and w['swt'] == 0.20
    
    scheduler.step(150)
    w = scheduler.get_loss_weights()
    assert w['l1'] == 0.60 and w['ssim'] == 0.05
    
    print("\n" + "=" * 70)
    print("ALL TESTS PASSED!")
    print("=" * 70 + "\n")
    
    return True


if __name__ == '__main__':
    test_multi_stage_scheduler()
