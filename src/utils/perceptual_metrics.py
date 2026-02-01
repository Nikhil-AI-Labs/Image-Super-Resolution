"""
Perceptual Quality Metrics
==========================
Championship-level perceptual metrics for NTIRE Track B evaluation.

Metrics:
- LPIPS: Learned Perceptual Image Patch Similarity (lower is better)
- DISTS: Deep Image Structure and Texture Similarity (lower is better)
- CLIP-IQA: CLIP-based Image Quality Assessment (higher is better)
- NIQE: Naturalness Image Quality Evaluator (lower is better)
- MUSIQ: Multi-Scale Image Quality Transformer (higher is better)
- MANIQA: Multi-dimension Attention Network for IQA (higher is better)

Combined Score (NTIRE Track B formula):
    Score = 0.3 * CLIP-IQA + 0.25 * MANIQA + 0.25 * MUSIQ + 0.2 * (1 - LPIPS)

Author: NTIRE SR Team
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Optional, List, Union
from pathlib import Path

# Check available metrics libraries
LPIPS_AVAILABLE = False
PYIQA_AVAILABLE = False
CLIP_AVAILABLE = False

try:
    import lpips
    LPIPS_AVAILABLE = True
except ImportError:
    pass

try:
    import pyiqa
    PYIQA_AVAILABLE = True
except ImportError:
    pass

try:
    import clip
    CLIP_AVAILABLE = True
except ImportError:
    pass


class LPIPSMetric:
    """
    LPIPS: Learned Perceptual Image Patch Similarity.
    
    Lower is better. Measures perceptual distance between images.
    Used by most SR papers for perceptual quality.
    """
    
    def __init__(self, net: str = 'alex', device: str = 'cuda'):
        """
        Args:
            net: Backbone network ('alex', 'vgg', 'squeeze')
            device: Device
        """
        self.device = device
        self.available = LPIPS_AVAILABLE
        
        if self.available:
            self.model = lpips.LPIPS(net=net).to(device)
            self.model.eval()
            print(f"  ✓ LPIPS ({net}) loaded")
        else:
            print(f"  ✗ LPIPS not available (pip install lpips)")
    
    @torch.no_grad()
    def __call__(self, sr: torch.Tensor, hr: torch.Tensor) -> float:
        """
        Calculate LPIPS distance.
        
        Args:
            sr, hr: Images [B, 3, H, W], range [0, 1]
            
        Returns:
            LPIPS distance (lower is better)
        """
        if not self.available:
            return 0.0
        
        # LPIPS expects [-1, 1]
        sr = (sr * 2 - 1).to(self.device)
        hr = (hr * 2 - 1).to(self.device)
        
        return self.model(sr, hr).mean().item()


class PyIQAMetric:
    """
    PyIQA-based metrics wrapper.
    
    Supports: CLIPIQA, MANIQA, MUSIQ, NIQE, DISTS, etc.
    """
    
    def __init__(self, metric_name: str, device: str = 'cuda', as_loss: bool = False):
        """
        Args:
            metric_name: Metric name ('clipiqa', 'maniqa', 'musiq', 'niqe', 'dists')
            device: Device
            as_loss: Whether to return as loss (for DISTS)
        """
        self.metric_name = metric_name
        self.device = device
        self.available = PYIQA_AVAILABLE
        self.as_loss = as_loss
        
        if self.available:
            try:
                self.model = pyiqa.create_metric(metric_name, device=device, as_loss=as_loss)
                self.is_lower_better = self.model.lower_better if hasattr(self.model, 'lower_better') else False
                print(f"  ✓ {metric_name.upper()} loaded")
            except Exception as e:
                print(f"  ✗ {metric_name.upper()} failed: {e}")
                self.available = False
        else:
            print(f"  ✗ {metric_name.upper()} not available (pip install pyiqa)")
    
    @torch.no_grad()
    def __call__(self, sr: torch.Tensor, hr: Optional[torch.Tensor] = None) -> float:
        """
        Calculate metric.
        
        Args:
            sr: SR image [B, 3, H, W], range [0, 1]
            hr: HR image (for FR metrics), range [0, 1]
            
        Returns:
            Metric value
        """
        if not self.available:
            return 0.0
        
        sr = sr.to(self.device).clamp(0, 1)
        
        if hr is not None and self.as_loss:
            # Full-reference metric
            hr = hr.to(self.device).clamp(0, 1)
            return self.model(sr, hr).mean().item()
        else:
            # No-reference metric
            return self.model(sr).mean().item()


class PerceptualEvaluator:
    """
    Comprehensive perceptual quality evaluator.
    
    Calculates all perceptual metrics for NTIRE Track B evaluation.
    
    Metrics included:
    - LPIPS (full-reference, lower is better)
    - DISTS (full-reference, lower is better)
    - CLIP-IQA (no-reference, higher is better)
    - MANIQA (no-reference, higher is better)
    - MUSIQ (no-reference, higher is better)
    - NIQE (no-reference, lower is better)
    """
    
    def __init__(
        self,
        device: str = 'cuda',
        metrics: Optional[List[str]] = None
    ):
        """
        Args:
            device: Device
            metrics: List of metrics to use (None = all available)
        """
        self.device = device
        
        print(f"\n{'='*70}")
        print("LOADING PERCEPTUAL METRICS")
        print(f"{'='*70}")
        
        # Default metrics
        if metrics is None:
            metrics = ['lpips', 'dists', 'clipiqa', 'maniqa', 'musiq', 'niqe']
        
        self.metrics = {}
        
        # LPIPS (always try to load)
        if 'lpips' in metrics:
            self.metrics['lpips'] = LPIPSMetric(net='alex', device=device)
        
        # PyIQA metrics
        pyiqa_metrics = {
            'dists': ('dists', True),    # FR, loss mode
            'clipiqa': ('clipiqa', False),  # NR
            'maniqa': ('maniqa', False),   # NR
            'musiq': ('musiq', False),     # NR
            'niqe': ('niqe', False),       # NR
        }
        
        for name, (pyiqa_name, as_loss) in pyiqa_metrics.items():
            if name in metrics:
                self.metrics[name] = PyIQAMetric(pyiqa_name, device, as_loss)
        
        print(f"{'='*70}")
        print(f"  Loaded {len(self.metrics)} metrics")
        print(f"{'='*70}\n")
        
        # Running averages
        self.reset()
    
    def reset(self):
        """Reset accumulated metrics."""
        self.values = {name: [] for name in self.metrics}
        self.count = 0
    
    @torch.no_grad()
    def update(self, sr: torch.Tensor, hr: torch.Tensor):
        """
        Update metrics with new batch.
        
        Args:
            sr: SR images [B, 3, H, W], range [0, 1]
            hr: HR images [B, 3, H, W], range [0, 1]
        """
        sr = sr.clamp(0, 1).to(self.device)
        hr = hr.clamp(0, 1).to(self.device)
        
        # Calculate each metric
        for name, metric in self.metrics.items():
            try:
                if name in ['lpips', 'dists']:
                    # Full-reference
                    value = metric(sr, hr)
                else:
                    # No-reference
                    value = metric(sr)
                
                self.values[name].append(value)
            except Exception as e:
                # Log error once per metric, then append 0
                if not hasattr(self, f'_error_logged_{name}'):
                    print(f"  ⚠️  {name} metric failed: {e}")
                    setattr(self, f'_error_logged_{name}', True)
                self.values[name].append(0.0)
        
        self.count += 1
    
    def get_average_metrics(self) -> Dict[str, float]:
        """
        Get average metrics.
        
        Returns:
            Dictionary of metric averages
        """
        averages = {}
        
        for name, values in self.values.items():
            if values:
                averages[name] = np.mean(values)
            else:
                averages[name] = 0.0
        
        # Calculate combined perceptual score (NTIRE Track B style)
        # Higher is better
        perceptual_score = self._calculate_perceptual_score(averages)
        averages['perceptual_score'] = perceptual_score
        
        return averages
    
    def _calculate_perceptual_score(self, metrics: Dict[str, float]) -> float:
        """
        Calculate NTIRE 2025 official perceptual score for Track B.
        
        Official Formula:
        Score = (1-LPIPS) + (1-DISTS) + CLIP-IQA + MANIQA + (MUSIQ/100) + max(0, 10 - NIQE/10)
        
        Range: ~0-6 (higher is better)
        """
        score = 0.0
        
        # (1 - LPIPS): Convert perceptual distance to quality (0-1)
        if 'lpips' in metrics and metrics['lpips'] > 0:
            score += max(0.0, 1.0 - metrics['lpips'])
        
        # (1 - DISTS): Convert texture distance to quality (0-1)
        if 'dists' in metrics and metrics['dists'] > 0:
            score += max(0.0, 1.0 - metrics['dists'])
        
        # CLIP-IQA: Direct quality score (0-1)
        if 'clipiqa' in metrics and metrics['clipiqa'] > 0:
            score += metrics['clipiqa']
        
        # MANIQA: Direct quality score (0-1)
        if 'maniqa' in metrics and metrics['maniqa'] > 0:
            score += metrics['maniqa']
        
        # MUSIQ: Normalize from 0-100 to 0-1
        if 'musiq' in metrics and metrics['musiq'] > 0:
            score += metrics['musiq'] / 100.0
        
        # NIQE: Inverted (lower is better, typical range 2-10)
        if 'niqe' in metrics and metrics['niqe'] > 0:
            score += max(0.0, 10.0 - metrics['niqe'] / 10.0)
        
        return score
    
    def __call__(self, sr: torch.Tensor, hr: torch.Tensor) -> Dict[str, float]:
        """
        Calculate all metrics for a single batch.
        
        Args:
            sr: SR images [B, 3, H, W]
            hr: HR images [B, 3, H, W]
            
        Returns:
            Dictionary of metrics
        """
        self.reset()
        self.update(sr, hr)
        return self.get_average_metrics()


# ============================================================================
# Standalone functions for quick evaluation
# ============================================================================

@torch.no_grad()
def calculate_lpips(sr: torch.Tensor, hr: torch.Tensor, device: str = 'cuda') -> float:
    """Calculate LPIPS distance."""
    if not LPIPS_AVAILABLE:
        return 0.0
    
    model = lpips.LPIPS(net='alex').to(device).eval()
    sr = (sr * 2 - 1).to(device)
    hr = (hr * 2 - 1).to(device)
    return model(sr, hr).mean().item()


@torch.no_grad()  
def calculate_clipiqa(img: torch.Tensor, device: str = 'cuda') -> float:
    """Calculate CLIP-IQA quality score."""
    if not PYIQA_AVAILABLE:
        return 0.0
    
    model = pyiqa.create_metric('clipiqa', device=device)
    return model(img.to(device)).mean().item()


# ============================================================================
# Testing
# ============================================================================

def test_perceptual_metrics():
    """Test perceptual metrics."""
    print("\n" + "="*70)
    print("PERCEPTUAL METRICS TEST")
    print("="*70)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Create test images
    sr = torch.rand(1, 3, 256, 256)
    hr = sr + torch.randn_like(sr) * 0.05
    hr = hr.clamp(0, 1)
    
    print("\n--- Test 1: Individual Metrics ---")
    
    # LPIPS
    if LPIPS_AVAILABLE:
        lpips_val = calculate_lpips(sr, hr, device)
        print(f"  LPIPS: {lpips_val:.4f}")
    else:
        print("  LPIPS: Not available")
    
    # CLIP-IQA
    if PYIQA_AVAILABLE:
        clipiqa_val = calculate_clipiqa(sr, device)
        print(f"  CLIP-IQA: {clipiqa_val:.4f}")
    else:
        print("  CLIP-IQA: Not available")
    
    print("\n--- Test 2: PerceptualEvaluator ---")
    evaluator = PerceptualEvaluator(device=device)
    
    # Update with batch
    evaluator.update(sr, hr)
    metrics = evaluator.get_average_metrics()
    
    print("\n  Results:")
    for name, value in metrics.items():
        print(f"    {name}: {value:.4f}")
    
    print("\n" + "="*70)
    print("✓ PERCEPTUAL METRICS TEST COMPLETE")
    print("="*70)
    print(f"\n  Available libraries:")
    print(f"    LPIPS: {'✓' if LPIPS_AVAILABLE else '✗'}")
    print(f"    PyIQA: {'✓' if PYIQA_AVAILABLE else '✗'}")
    print(f"    CLIP: {'✓' if CLIP_AVAILABLE else '✗'}")
    print("\n  To install missing:")
    print("    pip install lpips pyiqa")
    print("    pip install git+https://github.com/openai/CLIP.git")
    print("="*70 + "\n")


if __name__ == '__main__':
    test_perceptual_metrics()
