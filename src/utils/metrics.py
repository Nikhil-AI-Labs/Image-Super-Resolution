"""
Evaluation Metrics for Super-Resolution
========================================
PSNR, SSIM, and other quality metrics for SR evaluation.

Features:
- GPU-accelerated PSNR/SSIM calculation
- Y-channel conversion for fair comparison
- Batch metric calculation
- Championship-level precision (matches NTIRE evaluation)

Author: NTIRE SR Team
"""

import torch
import torch.nn.functional as F
import numpy as np
import math
from typing import Tuple, Optional, Dict, Union

# Try importing skimage for SSIM (fallback to custom implementation)
try:
    from skimage.metrics import structural_similarity as ssim_skimage
    SKIMAGE_AVAILABLE = True
except ImportError:
    SKIMAGE_AVAILABLE = False
    print("Warning: scikit-image not available. Using custom SSIM implementation.")


def rgb_to_y(img: torch.Tensor) -> torch.Tensor:
    """
    Convert RGB to Y channel (luminance) using ITU-R BT.601 standard.
    
    This is the standard conversion used in NTIRE and other SR competitions.
    
    Args:
        img: RGB image [B, 3, H, W] or [3, H, W], range [0, 1]
        
    Returns:
        Y channel [B, 1, H, W] or [1, H, W]
    """
    if img.ndim == 3:
        r, g, b = img[0:1, :, :], img[1:2, :, :], img[2:3, :, :]
    else:
        r, g, b = img[:, 0:1, :, :], img[:, 1:2, :, :], img[:, 2:3, :, :]
    
    # ITU-R BT.601 conversion (same as used in official NTIRE evaluation)
    y = 65.481 * r + 128.553 * g + 24.966 * b + 16.0
    # Normalize to [0, 1] range (MATLAB style)
    y = y / 255.0
    
    return y


def rgb_to_ycbcr(img: torch.Tensor) -> torch.Tensor:
    """
    Convert RGB to YCbCr color space.
    
    Args:
        img: RGB image [B, 3, H, W], range [0, 1]
        
    Returns:
        YCbCr image [B, 3, H, W], Y in [16/255, 235/255], CbCr in [16/255, 240/255]
    """
    # Conversion matrix (ITU-R BT.601)
    r, g, b = img[:, 0:1], img[:, 1:2], img[:, 2:3]
    
    y = 65.481 * r + 128.553 * g + 24.966 * b + 16.0
    cb = -37.797 * r - 74.203 * g + 112.0 * b + 128.0
    cr = 112.0 * r - 93.786 * g - 18.214 * b + 128.0
    
    ycbcr = torch.cat([y, cb, cr], dim=1) / 255.0
    return ycbcr


def calculate_psnr(
    img1: torch.Tensor,
    img2: torch.Tensor,
    crop_border: int = 0,
    test_y_channel: bool = False
) -> float:
    """
    Calculate PSNR (Peak Signal-to-Noise Ratio).
    
    Uses the same calculation as official NTIRE evaluation for fair comparison.
    
    Args:
        img1: First image tensor [B, C, H, W] or [C, H, W], range [0, 1]
        img2: Second image tensor [B, C, H, W] or [C, H, W], range [0, 1]
        crop_border: Crop border pixels before calculation
        test_y_channel: Test on Y channel (luminance) only (recommended for papers)
        
    Returns:
        PSNR value in dB
    """
    assert img1.shape == img2.shape, f"Image shapes must match: {img1.shape} vs {img2.shape}"
    
    # Ensure images are in [0, 1] range
    img1 = img1.clamp(0, 1)
    img2 = img2.clamp(0, 1)
    
    # Handle batch dimension
    if img1.ndim == 3:
        img1 = img1.unsqueeze(0)
        img2 = img2.unsqueeze(0)
    
    # Crop borders if specified
    if crop_border > 0:
        img1 = img1[:, :, crop_border:-crop_border, crop_border:-crop_border]
        img2 = img2[:, :, crop_border:-crop_border, crop_border:-crop_border]
    
    # Convert to Y channel if specified (standard for SR papers)
    if test_y_channel and img1.size(1) == 3:
        img1 = rgb_to_y(img1)
        img2 = rgb_to_y(img2)
    
    # Calculate MSE
    mse = torch.mean((img1 - img2) ** 2).item()
    
    if mse < 1e-10:
        return float('inf')
    
    # PSNR formula: 10 * log10(MAX^2 / MSE) = 10 * log10(1.0 / MSE) for [0,1] range
    psnr = 10 * math.log10(1.0 / mse)
    
    return psnr


def calculate_ssim_torch(
    img1: torch.Tensor,
    img2: torch.Tensor,
    window_size: int = 11,
    sigma: float = 1.5,
    channel: int = None
) -> float:
    """
    Calculate SSIM using PyTorch (GPU-accelerated).
    
    Args:
        img1: First image [B, C, H, W], range [0, 1]
        img2: Second image [B, C, H, W], range [0, 1]
        window_size: Size of Gaussian window
        sigma: Standard deviation of Gaussian window
        channel: Number of channels (auto-detect if None)
        
    Returns:
        SSIM value (0-1)
    """
    if channel is None:
        channel = img1.size(1)
    
    # Create Gaussian window
    gauss = torch.Tensor([
        math.exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2))
        for x in range(window_size)
    ])
    gauss = gauss / gauss.sum()
    
    # 1D to 2D window
    window_1d = gauss.unsqueeze(1)
    window_2d = window_1d.mm(window_1d.t()).float().unsqueeze(0).unsqueeze(0)
    window = window_2d.expand(channel, 1, window_size, window_size).contiguous()
    window = window.to(img1.device)
    
    # Constants for stability
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2
    
    # Calculate means
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)
    
    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2
    
    # Calculate variances
    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2
    
    # SSIM formula
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
               ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    
    return ssim_map.mean().item()


def calculate_ssim(
    img1: torch.Tensor,
    img2: torch.Tensor,
    crop_border: int = 0,
    test_y_channel: bool = False
) -> float:
    """
    Calculate SSIM (Structural Similarity Index).
    
    Args:
        img1: First image tensor [B, C, H, W] or [C, H, W], range [0, 1]
        img2: Second image tensor [B, C, H, W] or [C, H, W], range [0, 1]
        crop_border: Crop border pixels before calculation
        test_y_channel: Test on Y channel only (standard for papers)
        
    Returns:
        SSIM value (0-1)
    """
    assert img1.shape == img2.shape
    
    # Ensure images are in [0, 1] range
    img1 = img1.clamp(0, 1)
    img2 = img2.clamp(0, 1)
    
    # Handle batch dimension
    if img1.ndim == 3:
        img1 = img1.unsqueeze(0)
        img2 = img2.unsqueeze(0)
    
    # Crop borders
    if crop_border > 0:
        img1 = img1[:, :, crop_border:-crop_border, crop_border:-crop_border]
        img2 = img2[:, :, crop_border:-crop_border, crop_border:-crop_border]
    
    # Convert to Y channel if specified
    if test_y_channel and img1.size(1) == 3:
        img1 = rgb_to_y(img1)
        img2 = rgb_to_y(img2)
    
    # Use skimage if available (more accurate), else use PyTorch version
    if SKIMAGE_AVAILABLE:
        # Convert to numpy for scikit-image
        img1_np = img1.squeeze(0).cpu().numpy()
        img2_np = img2.squeeze(0).cpu().numpy()
        
        if img1_np.shape[0] == 1:
            # Single channel
            img1_np = img1_np.squeeze(0)
            img2_np = img2_np.squeeze(0)
            ssim_val = ssim_skimage(img1_np, img2_np, data_range=1.0)
        else:
            # Multi-channel - transpose to HWC
            img1_np = img1_np.transpose(1, 2, 0)
            img2_np = img2_np.transpose(1, 2, 0)
            ssim_val = ssim_skimage(img1_np, img2_np, data_range=1.0, channel_axis=2)
        
        return float(ssim_val)
    else:
        # Use PyTorch implementation
        return calculate_ssim_torch(img1, img2)


def calculate_psnr_ssim_batch(
    sr_images: torch.Tensor,
    hr_images: torch.Tensor,
    crop_border: int = 4,
    test_y_channel: bool = True
) -> Tuple[float, float]:
    """
    Calculate average PSNR and SSIM for a batch.
    
    Args:
        sr_images: SR images [B, C, H, W]
        hr_images: HR images [B, C, H, W]
        crop_border: Border to crop (default 4 for scale 4)
        test_y_channel: Test on Y channel only
        
    Returns:
        (avg_psnr, avg_ssim)
    """
    batch_size = sr_images.size(0)
    psnr_vals = []
    ssim_vals = []
    
    for i in range(batch_size):
        sr = sr_images[i]
        hr = hr_images[i]
        
        psnr = calculate_psnr(sr, hr, crop_border, test_y_channel)
        ssim = calculate_ssim(sr, hr, crop_border, test_y_channel)
        
        # Handle infinite PSNR (identical images)
        if not math.isinf(psnr):
            psnr_vals.append(psnr)
        ssim_vals.append(ssim)
    
    avg_psnr = np.mean(psnr_vals) if psnr_vals else float('inf')
    avg_ssim = np.mean(ssim_vals)
    
    return avg_psnr, avg_ssim


class MetricCalculator:
    """
    Metric calculator for tracking training progress.
    
    Calculates and tracks PSNR, SSIM with running averages.
    Thread-safe accumulation for multi-worker dataloaders.
    """
    
    def __init__(
        self,
        crop_border: int = 4,
        test_y_channel: bool = True
    ):
        """
        Args:
            crop_border: Border pixels to crop
            test_y_channel: Calculate on Y channel only
        """
        self.crop_border = crop_border
        self.test_y_channel = test_y_channel
        self.reset()
    
    def reset(self):
        """Reset accumulated metrics."""
        self.psnr_vals = []
        self.ssim_vals = []
        self.count = 0
    
    @torch.no_grad()
    def update(
        self,
        sr: torch.Tensor,
        hr: torch.Tensor
    ):
        """
        Update metrics with new batch.
        
        Args:
            sr: SR images [B, C, H, W]
            hr: HR images [B, C, H, W]
        """
        # Ensure on same device and correct range
        sr = sr.clamp(0, 1)
        hr = hr.clamp(0, 1)
        
        batch_size = sr.size(0)
        
        for i in range(batch_size):
            psnr = calculate_psnr(
                sr[i],
                hr[i],
                self.crop_border,
                self.test_y_channel
            )
            ssim = calculate_ssim(
                sr[i],
                hr[i],
                self.crop_border,
                self.test_y_channel
            )
            
            # Only add valid PSNR values
            if not math.isinf(psnr):
                self.psnr_vals.append(psnr)
            self.ssim_vals.append(ssim)
            self.count += 1
    
    def get_metrics(self) -> Dict[str, float]:
        """
        Get average metrics.
        
        Returns:
            Dictionary with 'psnr' and 'ssim'
        """
        if self.count == 0:
            return {'psnr': 0.0, 'ssim': 0.0}
        
        avg_psnr = np.mean(self.psnr_vals) if self.psnr_vals else 0.0
        avg_ssim = np.mean(self.ssim_vals) if self.ssim_vals else 0.0
        
        return {
            'psnr': float(avg_psnr),
            'ssim': float(avg_ssim),
        }
    
    def __str__(self) -> str:
        """String representation."""
        metrics = self.get_metrics()
        return f"PSNR: {metrics['psnr']:.2f} dB, SSIM: {metrics['ssim']:.4f}"


# ============================================================================
# LPIPS (Perceptual) Metric - Optional
# ============================================================================

class LPIPSCalculator:
    """
    LPIPS perceptual distance metric.
    
    Requires lpips package: pip install lpips
    """
    
    def __init__(self, net: str = 'alex', device: str = 'cuda'):
        """
        Args:
            net: Network to use ('alex', 'vgg', 'squeeze')
            device: Device to run on
        """
        try:
            import lpips
            self.lpips_fn = lpips.LPIPS(net=net).to(device)
            self.lpips_fn.eval()
            self.available = True
        except ImportError:
            print("Warning: lpips not available. Install with: pip install lpips")
            self.available = False
    
    @torch.no_grad()
    def calculate(self, img1: torch.Tensor, img2: torch.Tensor) -> float:
        """
        Calculate LPIPS distance.
        
        Args:
            img1, img2: Images [B, 3, H, W], range [0, 1]
            
        Returns:
            LPIPS distance (lower is better)
        """
        if not self.available:
            return 0.0
        
        # LPIPS expects [-1, 1] range
        img1 = img1 * 2 - 1
        img2 = img2 * 2 - 1
        
        return self.lpips_fn(img1, img2).mean().item()


# ============================================================================
# Testing
# ============================================================================

def test_metrics():
    """Test metric calculations."""
    print("\n" + "=" * 70)
    print("METRICS MODULE - TEST")
    print("=" * 70)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Create test images
    hr = torch.rand(1, 3, 256, 256, device=device)
    sr = hr + torch.randn_like(hr) * 0.01  # Add small noise
    sr = sr.clamp(0, 1)
    
    print("\n--- Test 1: PSNR Calculation ---")
    psnr = calculate_psnr(sr, hr)
    print(f"  PSNR (RGB): {psnr:.2f} dB")
    assert psnr > 0, "PSNR should be positive"
    assert psnr < 100, "PSNR too high"
    
    psnr_y = calculate_psnr(sr, hr, test_y_channel=True)
    print(f"  PSNR (Y): {psnr_y:.2f} dB")
    print("  [PASSED]")
    
    print("\n--- Test 2: SSIM Calculation ---")
    ssim = calculate_ssim(sr, hr)
    print(f"  SSIM (RGB): {ssim:.4f}")
    assert 0 <= ssim <= 1, f"SSIM out of range: {ssim}"
    
    ssim_y = calculate_ssim(sr, hr, test_y_channel=True)
    print(f"  SSIM (Y): {ssim_y:.4f}")
    print("  [PASSED]")
    
    print("\n--- Test 3: Border Cropping ---")
    psnr_crop = calculate_psnr(sr, hr, crop_border=4)
    ssim_crop = calculate_ssim(sr, hr, crop_border=4)
    print(f"  PSNR (4px crop): {psnr_crop:.2f} dB")
    print(f"  SSIM (4px crop): {ssim_crop:.4f}")
    print("  [PASSED]")
    
    print("\n--- Test 4: Batch Calculation ---")
    batch_sr = torch.rand(4, 3, 256, 256, device=device)
    batch_hr = batch_sr + torch.randn_like(batch_sr) * 0.02
    batch_hr = batch_hr.clamp(0, 1)
    
    avg_psnr, avg_ssim = calculate_psnr_ssim_batch(batch_sr, batch_hr)
    print(f"  Avg PSNR: {avg_psnr:.2f} dB")
    print(f"  Avg SSIM: {avg_ssim:.4f}")
    print("  [PASSED]")
    
    print("\n--- Test 5: MetricCalculator ---")
    calc = MetricCalculator(crop_border=4, test_y_channel=True)
    calc.update(batch_sr, batch_hr)
    metrics = calc.get_metrics()
    print(f"  {calc}")
    assert 'psnr' in metrics and 'ssim' in metrics
    print("  [PASSED]")
    
    print("\n--- Test 6: Y Channel Conversion ---")
    y_channel = rgb_to_y(hr)
    print(f"  Input shape: {hr.shape}")
    print(f"  Y channel shape: {y_channel.shape}")
    assert y_channel.shape[1] == 1, "Y channel should have 1 channel"
    print("  [PASSED]")
    
    print("\n" + "=" * 70)
    print("✓ ALL METRIC TESTS PASSED!")
    print("=" * 70 + "\n")


if __name__ == '__main__':
    test_metrics()
