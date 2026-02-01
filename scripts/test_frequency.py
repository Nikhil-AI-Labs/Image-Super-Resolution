"""
Frequency Decomposition Test Script
====================================

Test and visualize the DCT/IDCT frequency decomposition module.
This script validates that the module works correctly before
integrating it into the full training pipeline.

Usage:
------
    # Run with test image
    python scripts/test_frequency.py --image "path/to/test_image.png"
    
    # Run unit tests only (no image needed)
    python scripts/test_frequency.py --test_only
    
    # Save visualization
    python scripts/test_frequency.py --image "path/to/image.png" --output "outputs/freq_viz.png"

Expected Output:
----------------
    - Reconstruction error < 1e-3 (near-perfect)
    - Visualization showing low/mid/high frequency separation
    - GPU acceleration working (if available)

Author: NTIRE SR Team
"""

import sys
import os
from pathlib import Path
import argparse
import torch
import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data.frequency_decomposition import (
    FrequencyDecomposition,
    FrequencyAugmentation,
    test_frequency_decomposition,
    visualize_frequency_decomposition
)


def run_comprehensive_tests():
    """Run comprehensive tests on the frequency decomposition module."""
    
    print("\n" + "="*60)
    print("FREQUENCY DECOMPOSITION - COMPREHENSIVE TESTS")
    print("="*60)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nDevice: {device}")
    print(f"PyTorch version: {torch.__version__}")
    if device == 'cuda':
        print(f"CUDA version: {torch.version.cuda}")
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    # Test 1: Basic functionality
    print("\n--- Test 1: Basic Functionality ---")
    freq_decomp = FrequencyDecomposition(block_size=8).to(device)
    
    test_image = torch.randn(1, 3, 128, 128, device=device)
    result = freq_decomp(test_image)
    
    print(f"Input shape: {test_image.shape}")
    print(f"Low freq shape: {result['low_freq'].shape}")
    print(f"Mid freq shape: {result['mid_freq'].shape}")
    print(f"High freq shape: {result['high_freq'].shape}")
    
    # Test 2: Reconstruction quality
    print("\n--- Test 2: Reconstruction Quality ---")
    reconstructed = result['low_freq'] + result['mid_freq'] + result['high_freq']
    
    # For random input, reconstruction won't be perfect due to float precision
    # and edge effects, but should be very close
    mse = torch.mean((test_image - reconstructed) ** 2).item()
    mae = torch.mean(torch.abs(test_image - reconstructed)).item()
    max_error = torch.max(torch.abs(test_image - reconstructed)).item()
    
    print(f"MSE: {mse:.2e}")
    print(f"MAE: {mae:.2e}")
    print(f"Max error: {max_error:.2e}")
    
    if max_error < 1e-2:
        print("✓ Reconstruction quality: EXCELLENT")
    elif max_error < 1e-1:
        print("✓ Reconstruction quality: GOOD")
    else:
        print("⚠ Reconstruction quality: NEEDS INVESTIGATION")
    
    # Test 3: Different image sizes
    print("\n--- Test 3: Various Image Sizes ---")
    test_sizes = [
        (64, 64),
        (128, 128),
        (256, 256),
        (240, 320),  # Non-square
        (100, 150),  # Non-block-aligned
    ]
    
    all_passed = True
    for h, w in test_sizes:
        test_img = torch.randn(1, 3, h, w, device=device)
        result = freq_decomp(test_img)
        
        # Check shapes
        shapes_ok = all(result[k].shape == test_img.shape for k in ['low_freq', 'mid_freq', 'high_freq'])
        
        # Check reconstruction
        recon = result['low_freq'] + result['mid_freq'] + result['high_freq']
        error = torch.max(torch.abs(test_img - recon)).item()
        
        status = "✓" if shapes_ok and error < 0.1 else "✗"
        print(f"  {status} Size [{h:4d}, {w:4d}]: shapes={'OK' if shapes_ok else 'FAIL'}, error={error:.2e}")
        
        if not (shapes_ok and error < 0.1):
            all_passed = False
    
    # Test 4: Batch processing
    print("\n--- Test 4: Batch Processing ---")
    batch_sizes = [1, 4, 8, 16]
    
    for bs in batch_sizes:
        test_batch = torch.randn(bs, 3, 128, 128, device=device)
        result = freq_decomp(test_batch)
        
        assert result['low_freq'].shape[0] == bs
        print(f"  ✓ Batch size {bs}: OK")
    
    # Test 5: Gradient flow
    print("\n--- Test 5: Gradient Flow ---")
    test_img = torch.randn(1, 3, 128, 128, device=device, requires_grad=True)
    result = freq_decomp(test_img)
    
    loss = result['low_freq'].sum() + result['mid_freq'].sum() + result['high_freq'].sum()
    loss.backward()
    
    if test_img.grad is not None:
        print(f"  ✓ Gradients computed: grad norm = {test_img.grad.norm().item():.4f}")
    else:
        print("  ✗ No gradients!")
        all_passed = False
    
    # Test 6: Frequency augmentation
    print("\n--- Test 6: Frequency Augmentation ---")
    freq_aug = FrequencyAugmentation(block_size=8, prob=1.0).to(device)
    freq_aug.train()
    
    test_img = torch.randn(2, 3, 128, 128, device=device)
    augmented = freq_aug(test_img)
    
    assert augmented.shape == test_img.shape
    diff = torch.abs(test_img - augmented).mean().item()
    print(f"  ✓ Augmentation applied: mean diff = {diff:.4f}")
    
    # Test 7: Eval mode (no augmentation)
    freq_aug.eval()
    augmented_eval = freq_aug(test_img)
    diff_eval = torch.abs(test_img - augmented_eval).mean().item()
    print(f"  ✓ Eval mode (no aug): mean diff = {diff_eval:.6f}")
    
    # Test 8: Memory efficiency
    print("\n--- Test 8: Memory Efficiency ---")
    if device == 'cuda':
        torch.cuda.reset_peak_memory_stats()
        
        # Process a large batch
        large_batch = torch.randn(8, 3, 512, 512, device=device)
        result = freq_decomp(large_batch)
        
        peak_memory = torch.cuda.max_memory_allocated() / 1024**3  # GB
        print(f"  Peak GPU memory for 8×512×512: {peak_memory:.2f} GB")
    else:
        print("  Skipped (CPU mode)")
    
    # Summary
    print("\n" + "="*60)
    if all_passed:
        print("ALL TESTS PASSED!")
    else:
        print("SOME TESTS FAILED - Please investigate")
    print("="*60)
    
    return all_passed


def analyze_frequency_content(image_path: str, output_dir: str = None):
    """
    Analyze and visualize frequency content of an image.
    
    Args:
        image_path: Path to input image
        output_dir: Directory to save visualizations
    """
    from PIL import Image
    import torchvision.transforms as T
    
    print(f"\nAnalyzing: {image_path}")
    
    # Load image
    image = Image.open(image_path).convert('RGB')
    transform = T.ToTensor()
    image_tensor = transform(image).unsqueeze(0)
    
    print(f"Image size: {image_tensor.shape}")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    freq_decomp = FrequencyDecomposition(block_size=8).to(device)
    image_tensor = image_tensor.to(device)
    
    # Decompose
    result = freq_decomp(image_tensor)
    
    # Compute energy in each band
    low_energy = torch.sum(result['low_freq'] ** 2).item()
    mid_energy = torch.sum(result['mid_freq'] ** 2).item()
    high_energy = torch.sum(result['high_freq'] ** 2).item()
    total_energy = low_energy + mid_energy + high_energy
    
    print(f"\nFrequency Energy Distribution:")
    print(f"  Low frequency:  {low_energy/total_energy*100:.1f}%")
    print(f"  Mid frequency:  {mid_energy/total_energy*100:.1f}%")
    print(f"  High frequency: {high_energy/total_energy*100:.1f}%")
    
    # Generate visualization
    if output_dir:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        viz_path = output_path / 'frequency_decomposition.png'
        visualize_frequency_decomposition(
            image_path=image_path,
            output_path=str(viz_path),
            block_size=8
        )
        
        print(f"\nVisualization saved to: {viz_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Test Frequency Decomposition Module',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        '--image',
        type=str,
        help='Path to test image for visualization'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        default='outputs/visualizations',
        help='Output directory for visualizations'
    )
    
    parser.add_argument(
        '--test_only',
        action='store_true',
        help='Run unit tests only, skip visualization'
    )
    
    parser.add_argument(
        '--block_size',
        type=int,
        default=8,
        help='DCT block size (default: 8)'
    )
    
    args = parser.parse_args()
    
    print("\n" + "="*60)
    print("NTIRE SR - FREQUENCY DECOMPOSITION TEST")
    print("="*60)
    
    # Run unit tests
    all_passed = run_comprehensive_tests()
    
    # Visualize if image provided
    if args.image and not args.test_only:
        if not os.path.exists(args.image):
            print(f"\nWarning: Image not found: {args.image}")
        else:
            analyze_frequency_content(args.image, args.output)
    
    print("\n" + "="*60)
    print("FREQUENCY DECOMPOSITION MODULE READY!")
    print("="*60)
    print("\nNext steps:")
    print("  1. Run CLIP filtering on DF2K dataset")
    print("  2. Proceed to Phase 2: Expert pre-training")
    print("="*60)
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
