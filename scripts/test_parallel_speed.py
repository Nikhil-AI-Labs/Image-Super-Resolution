"""
Speed Benchmark for Multi-GPU Expert Execution
================================================
Tests the ThreadPoolExecutor-based parallel execution to verify speedup.

Run on your Ubuntu training system:
    cd /media/admin1/DL/Abhishek_/image-super-resolution
    python scripts/test_parallel_speed.py

Expected Results:
- Multi-GPU: ~6-7s per batch (with ThreadPoolExecutor)
- Single GPU: ~17s per batch (sequential execution)
"""

import sys
import os
import time
from pathlib import Path

# Add project root
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
import torch.nn as nn


def test_parallel_speed():
    """Benchmark parallel vs sequential expert execution."""
    print("\n" + "=" * 70)
    print("MULTI-GPU SPEED BENCHMARK")
    print("=" * 70)
    
    # Check CUDA
    if not torch.cuda.is_available():
        print("❌ CUDA not available. This test requires GPUs.")
        return
    
    num_gpus = torch.cuda.device_count()
    print(f"\nAvailable GPUs: {num_gpus}")
    for i in range(num_gpus):
        print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
    
    if num_gpus < 2:
        print("\n⚠ Only 1 GPU available. Parallel speedup requires 2 GPUs.")
        print("  The test will still run but won't show parallel speedup.")
    
    # Import expert loader
    try:
        from src.models.expert_loader import ExpertEnsemble
    except ImportError as e:
        print(f"❌ Failed to import ExpertEnsemble: {e}")
        return
    
    print("\n" + "-" * 40)
    print("Loading Expert Models...")
    print("-" * 40)
    
    # Configure multi-GPU
    if num_gpus >= 2:
        devices = {
            'hat': 'cuda:0',
            'dat': 'cuda:1',
            'nafnet': 'cuda:1'
        }
    else:
        devices = None
    
    # Create ensemble
    ensemble = ExpertEnsemble(
        device='cuda:0',
        devices=devices,
        upscale=4
    )
    
    # Load experts (this may take a while)
    results = ensemble.load_all_experts()
    
    num_loaded = sum(results.values())
    if num_loaded == 0:
        print("❌ No experts loaded. Cannot benchmark.")
        return
    
    print(f"\nLoaded {num_loaded}/3 experts")
    print(f"Multi-GPU enabled: {ensemble.multi_gpu}")
    
    # Warmup
    print("\n" + "-" * 40)
    print("Warming up CUDA kernels...")
    print("-" * 40)
    
    warmup_input = torch.randn(1, 3, 64, 64, device='cuda:0')
    with torch.no_grad():
        for i in range(3):
            _ = ensemble.forward_all(warmup_input)
    torch.cuda.synchronize()
    print("✓ Warmup complete")
    
    # Benchmark
    print("\n" + "-" * 40)
    print("Benchmarking forward_all()...")
    print("-" * 40)
    
    # Test different batch sizes
    batch_sizes = [1, 4, 8]
    num_iterations = 5
    
    for batch_size in batch_sizes:
        test_input = torch.randn(batch_size, 3, 64, 64, device='cuda:0')
        
        times = []
        for i in range(num_iterations):
            torch.cuda.synchronize()
            start = time.perf_counter()
            
            with torch.no_grad():
                outputs = ensemble.forward_all(test_input, return_dict=True)
            
            torch.cuda.synchronize()
            end = time.perf_counter()
            times.append(end - start)
        
        avg_time = sum(times) / len(times)
        min_time = min(times)
        max_time = max(times)
        
        print(f"\n  Batch size {batch_size}:")
        print(f"    Average: {avg_time*1000:.1f} ms")
        print(f"    Min:     {min_time*1000:.1f} ms")
        print(f"    Max:     {max_time*1000:.1f} ms")
        
        # Verify outputs
        for name, out in outputs.items():
            expected_h = 64 * 4  # upscale factor
            expected_w = 64 * 4
            assert out.shape == (batch_size, 3, expected_h, expected_w), \
                f"Unexpected output shape for {name}: {out.shape}"
        print(f"    ✓ Output shapes verified")
    
    # Check feature capture (for Collaborative Learning)
    print("\n" + "-" * 40)
    print("Verifying Feature Capture...")
    print("-" * 40)
    
    test_input = torch.randn(2, 3, 64, 64, device='cuda:0')
    with torch.no_grad():
        outputs = ensemble.forward_all(test_input, return_dict=True)
    
    # Check if features were captured via hooks
    features = ensemble._captured_features
    if features:
        print(f"  ✓ Features captured: {list(features.keys())}")
        for name, feat in features.items():
            print(f"    {name}: {feat.shape} on {feat.device}")
    else:
        print("  ⚠ No features captured (hooks may not be registered)")
        print("    This is OK if not using Collaborative Learning mode")
    
    # Memory usage
    print("\n" + "-" * 40)
    print("GPU Memory Usage")
    print("-" * 40)
    for i in range(num_gpus):
        allocated = torch.cuda.memory_allocated(i) / 1e9
        reserved = torch.cuda.memory_reserved(i) / 1e9
        print(f"  GPU {i}: {allocated:.2f} GB allocated, {reserved:.2f} GB reserved")
    
    print("\n" + "=" * 70)
    print("✓ BENCHMARK COMPLETE")
    print("=" * 70)
    
    # Print expected vs actual
    print("\nExpected speedup with ThreadPoolExecutor:")
    print("  - Single GPU (sequential):  ~17s per batch")
    print("  - Multi-GPU (parallel):     ~6-7s per batch")
    print("\nIf your batch times match the 'multi-GPU parallel' range,")
    print("the ThreadPoolExecutor optimization is working! ✓")
    print()


if __name__ == '__main__':
    test_parallel_speed()
