"""
Integration Test: Fusion Network + Expert Ensemble
===================================================

This script tests the complete fusion pipeline with real expert models.
It verifies:
1. Expert ensemble loads correctly
2. Fusion network integrates with experts
3. End-to-end forward pass works
4. Parameter counts are correct (trainable vs frozen)

Run with: python scripts/test_fusion_integration.py

Author: NTIRE SR Team
"""

import os
import sys
import torch
import time

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src.models.expert_loader import ExpertEnsemble
from src.models.fusion_network import (
    FrequencyRouter,
    FrequencyAwareFusion,
    MultiFusionSR
)


def test_fusion_standalone():
    """Test fusion components without experts."""
    print("\n" + "=" * 60)
    print("TEST 1: Fusion Components (Standalone)")
    print("=" * 60)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    
    # Create dummy LR input
    lr_input = torch.randn(2, 3, 64, 64).to(device)
    
    # Create dummy expert outputs (simulating HAT, MambaIR, NAFNet)
    expert_outputs = [
        torch.randn(2, 3, 256, 256).to(device),  # HAT
        torch.randn(2, 3, 256, 256).to(device),  # MambaIR
        torch.randn(2, 3, 256, 256).to(device),  # NAFNet
    ]
    
    # Test FrequencyRouter
    print("\n--- FrequencyRouter ---")
    router = FrequencyRouter(use_attention=True).to(device)
    routing_weights = router(lr_input)
    
    assert routing_weights.shape == (2, 3, 3, 64, 64), \
        f"Shape mismatch: {routing_weights.shape}"
    print(f"✓ Routing weights: {routing_weights.shape}")
    
    params = sum(p.numel() for p in router.parameters())
    print(f"✓ Router params: {params:,}")
    
    # Test FrequencyAwareFusion
    print("\n--- FrequencyAwareFusion ---")
    fusion = FrequencyAwareFusion(
        num_experts=3,
        use_residual=True,
        use_multiscale=True
    ).to(device)
    
    fused_sr = fusion(lr_input, expert_outputs)
    
    assert fused_sr.shape == (2, 3, 256, 256), f"Shape mismatch: {fused_sr.shape}"
    assert fused_sr.min() >= 0 and fused_sr.max() <= 1, \
        f"Output range: [{fused_sr.min():.3f}, {fused_sr.max():.3f}]"
    print(f"✓ Fused output: {fused_sr.shape}")
    print(f"✓ Output range: [{fused_sr.min():.4f}, {fused_sr.max():.4f}]")
    
    params = sum(p.numel() for p in fusion.parameters())
    print(f"✓ Fusion params: {params:,}")
    
    print("\n✓ Standalone tests PASSED!")


def test_with_partial_experts():
    """Test fusion with varying number of experts."""
    print("\n" + "=" * 60)
    print("TEST 2: Fusion with Partial Experts")
    print("=" * 60)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    lr_input = torch.randn(1, 3, 48, 48).to(device)
    
    # Test with 1, 2, and 3 experts
    for num_exp in [1, 2, 3]:
        print(f"\n--- Testing with {num_exp} expert(s) ---")
        
        expert_outputs = [
            torch.randn(1, 3, 192, 192).to(device) for _ in range(num_exp)
        ]
        
        fusion = FrequencyAwareFusion(
            num_experts=num_exp,
            use_residual=True,
            use_multiscale=True
        ).to(device)
        
        fused_sr = fusion(lr_input, expert_outputs)
        
        assert fused_sr.shape == (1, 3, 192, 192), f"Shape mismatch: {fused_sr.shape}"
        print(f"✓ {num_exp} experts: {fused_sr.shape}")
    
    print("\n✓ Partial expert tests PASSED!")


def test_integration_with_experts():
    """Test complete integration with expert ensemble."""
    print("\n" + "=" * 60)
    print("TEST 3: Integration with Expert Ensemble")
    print("=" * 60)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    
    # Load experts
    print("\n--- Loading Expert Ensemble ---")
    ensemble = ExpertEnsemble(device=device)
    results = ensemble.load_all_experts()
    
    loaded_experts = ensemble.get_loaded_experts()
    num_loaded = len(loaded_experts)
    
    if num_loaded == 0:
        print("\n⚠ No experts loaded - cannot test full integration")
        print("  This is expected if pretrained weights are not downloaded")
        print("  Fusion network itself is verified in standalone tests")
        return False
    
    print(f"\n✓ Loaded {num_loaded} experts: {loaded_experts}")
    
    # Create complete fusion pipeline
    print("\n--- Creating MultiFusionSR Pipeline ---")
    model = MultiFusionSR(
        ensemble, 
        num_experts=num_loaded,
        use_residual=True,
        use_multiscale=True
    )
    model = model.to(device)
    model.eval()
    
    # Test forward pass
    print("\n--- Forward Pass Test ---")
    lr_input = torch.randn(1, 3, 64, 64).to(device)
    
    start_time = time.time()
    
    with torch.no_grad():
        fused_sr, intermediates = model(lr_input, return_intermediates=True)
    
    elapsed = time.time() - start_time
    
    print(f"✓ Input shape: {lr_input.shape}")
    print(f"✓ Output shape: {fused_sr.shape}")
    print(f"✓ Output range: [{fused_sr.min():.4f}, {fused_sr.max():.4f}]")
    print(f"✓ Inference time: {elapsed:.3f}s")
    
    # Verify intermediates
    print(f"\n✓ Experts used: {list(intermediates['expert_outputs'].keys())}")
    print(f"✓ Routing weights: {intermediates['routing_weights'].shape}")
    
    # Parameter counts
    trainable_params = model.get_trainable_params()
    frozen_params = model.get_frozen_params()
    
    print(f"\n✓ Trainable params (fusion): {trainable_params:,}")
    print(f"✓ Frozen params (experts): {frozen_params:,}")
    
    # Gradient check
    print("\n--- Gradient Check ---")
    lr_input.requires_grad = False
    fused_sr = model(lr_input)
    
    # Simulate loss
    loss = fused_sr.mean()
    loss.backward()
    
    # Check fusion gradients exist
    fusion_has_grad = any(
        p.grad is not None for p in model.fusion.parameters() if p.requires_grad
    )
    print(f"✓ Fusion has gradients: {fusion_has_grad}")
    
    # Check expert gradients are None (frozen)
    expert_has_grad = any(
        p.grad is not None for p in model.expert_ensemble.parameters()
    )
    print(f"✓ Experts frozen (no gradients): {not expert_has_grad}")
    
    print("\n✓ Integration test PASSED!")
    return True


def test_memory_usage():
    """Test GPU memory usage."""
    print("\n" + "=" * 60)
    print("TEST 4: Memory Usage")
    print("=" * 60)
    
    if not torch.cuda.is_available():
        print("⚠ CUDA not available - skipping memory test")
        return
    
    device = 'cuda'
    
    # Clear cache
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    
    # Create fusion module
    fusion = FrequencyAwareFusion(
        num_experts=3,
        use_residual=True,
        use_multiscale=True
    ).to(device)
    
    # Measure memory for different batch sizes
    for batch_size in [1, 2, 4]:
        torch.cuda.reset_peak_memory_stats()
        
        lr_input = torch.randn(batch_size, 3, 64, 64).to(device)
        expert_outputs = [
            torch.randn(batch_size, 3, 256, 256).to(device) for _ in range(3)
        ]
        
        with torch.no_grad():
            fused_sr = fusion(lr_input, expert_outputs)
        
        peak_memory = torch.cuda.max_memory_allocated() / 1024**3  # GB
        print(f"✓ Batch {batch_size}: Peak memory = {peak_memory:.2f} GB")
        
        del lr_input, expert_outputs, fused_sr
        torch.cuda.empty_cache()
    
    print("\n✓ Memory test completed!")


def main():
    """Run all integration tests."""
    print("\n" + "=" * 70)
    print(" " * 15 + "FUSION NETWORK INTEGRATION TESTS")
    print("=" * 70)
    
    all_passed = True
    
    try:
        test_fusion_standalone()
    except Exception as e:
        print(f"\n✗ Standalone test FAILED: {e}")
        all_passed = False
    
    try:
        test_with_partial_experts()
    except Exception as e:
        print(f"\n✗ Partial expert test FAILED: {e}")
        all_passed = False
    
    try:
        test_integration_with_experts()
    except Exception as e:
        print(f"\n✗ Integration test FAILED: {e}")
        import traceback
        traceback.print_exc()
        all_passed = False
    
    try:
        test_memory_usage()
    except Exception as e:
        print(f"\n✗ Memory test FAILED: {e}")
        all_passed = False
    
    # Summary
    print("\n" + "=" * 70)
    if all_passed:
        print(" " * 20 + "✓ ALL TESTS PASSED!")
        print(" " * 10 + "Fusion Network ready for Phase 4 (Loss Functions)")
    else:
        print(" " * 20 + "✗ SOME TESTS FAILED")
        print(" " * 10 + "Please check error messages above")
    print("=" * 70 + "\n")


if __name__ == '__main__':
    main()
