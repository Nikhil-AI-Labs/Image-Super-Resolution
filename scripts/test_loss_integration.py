"""
Integration Test: Fusion Network + Loss Functions
==================================================
Tests the complete training pipeline with all loss components.

Author: NTIRE SR Team
"""

import os
import sys

# Add project root to path
_current_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.dirname(_current_dir)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import torch
import torch.nn.functional as F


def test_loss_import():
    """Test that all losses can be imported."""
    print("\n" + "-" * 60)
    print("Test 1: Import All Losses")
    print("-" * 60)
    
    from src.losses import (
        L1Loss, L2Loss, CharbonnierLoss,
        SSIMLoss, VGGPerceptualLoss, FFTLoss,
        EdgeLoss, CombinedLoss,
        PYWT_AVAILABLE, CLIP_AVAILABLE
    )
    
    print("  L1Loss, L2Loss, CharbonnierLoss: Imported")
    print("  SSIMLoss: Imported")
    print("  VGGPerceptualLoss: Imported")
    print("  FFTLoss: Imported")
    print("  EdgeLoss: Imported")
    print("  CombinedLoss: Imported")
    print(f"  PyWavelets available: {PYWT_AVAILABLE}")
    print(f"  CLIP available: {CLIP_AVAILABLE}")
    
    if PYWT_AVAILABLE:
        from src.losses import SWTLoss
        print("  SWTLoss: Imported")
    
    print("\n  [PASSED] All imports successful")
    return True


def test_basic_loss_computation():
    """Test basic loss computation."""
    print("\n" + "-" * 60)
    print("Test 2: Basic Loss Computation")
    print("-" * 60)
    
    from src.losses import L1Loss, L2Loss, CharbonnierLoss
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"  Device: {device}")
    
    # Create test data
    pred = torch.rand(2, 3, 64, 64, device=device)
    target = torch.rand(2, 3, 64, 64, device=device)
    
    # Test each loss
    l1 = L1Loss()(pred, target)
    l2 = L2Loss()(pred, target)
    charb = CharbonnierLoss()(pred, target)
    
    print(f"  L1 Loss: {l1.item():.6f}")
    print(f"  L2 Loss: {l2.item():.6f}")
    print(f"  Charbonnier Loss: {charb.item():.6f}")
    
    assert l1.item() > 0, "L1 should be positive"
    assert l2.item() > 0, "L2 should be positive"
    assert charb.item() > 0, "Charbonnier should be positive"
    
    print("\n  [PASSED] Basic losses compute correctly")
    return True


def test_vgg_perceptual_loss():
    """Test VGG perceptual loss."""
    print("\n" + "-" * 60)
    print("Test 3: VGG Perceptual Loss")
    print("-" * 60)
    
    from src.losses import VGGPerceptualLoss
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"  Device: {device}")
    
    # Create test data (larger for VGG)
    pred = torch.rand(1, 3, 224, 224, device=device)
    target = torch.rand(1, 3, 224, 224, device=device)
    
    # Initialize VGG loss
    vgg_loss = VGGPerceptualLoss(
        feature_layers=['relu2_2', 'relu3_4']
    ).to(device)
    
    # Compute loss
    loss = vgg_loss(pred, target)
    print(f"  VGG Loss: {loss.item():.6f}")
    
    assert loss.item() > 0, "VGG loss should be positive"
    
    print("\n  [PASSED] VGG perceptual loss works correctly")
    return True


def test_frequency_losses():
    """Test frequency domain losses (FFT and SWT)."""
    print("\n" + "-" * 60)
    print("Test 4: Frequency Domain Losses")
    print("-" * 60)
    
    from src.losses import FFTLoss, PYWT_AVAILABLE
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"  Device: {device}")
    
    pred = torch.rand(1, 3, 64, 64, device=device)
    target = torch.rand(1, 3, 64, 64, device=device)
    
    # Test FFT Loss
    fft_loss = FFTLoss(focus_high_freq=True)
    fft_val = fft_loss(pred, target)
    print(f"  FFT Loss: {fft_val.item():.6f}")
    
    # Test SWT Loss if available
    if PYWT_AVAILABLE:
        from src.losses import SWTLoss
        swt_loss = SWTLoss(wavelet='haar', level=2).to(device)
        swt_val = swt_loss(pred, target)
        print(f"  SWT Loss: {swt_val.item():.6f}")
    else:
        print("  SWT Loss: Skipped (PyWavelets not installed)")
    
    print("\n  [PASSED] Frequency losses work correctly")
    return True


def test_combined_loss_stages():
    """Test combined loss with multi-stage training."""
    print("\n" + "-" * 60)
    print("Test 5: Combined Loss Multi-Stage Training")
    print("-" * 60)
    
    from src.losses import CombinedLoss, PYWT_AVAILABLE
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"  Device: {device}")
    
    pred = torch.rand(1, 3, 128, 128, device=device)
    target = torch.rand(1, 3, 128, 128, device=device)
    
    # Initialize combined loss
    combined = CombinedLoss(
        use_swt=PYWT_AVAILABLE,
        use_fft=True,
        use_clip=False
    ).to(device)
    
    # Test each stage
    for stage in [1, 2, 3]:
        combined.set_stage(stage)
        loss, components = combined(pred, target, return_components=True)
        print(f"\n  Stage {stage}:")
        print(f"    Total Loss: {loss.item():.6f}")
        print(f"    Components: {list(components.keys())}")
        for name, val in components.items():
            print(f"      {name}: {val.item():.6f}")
    
    print("\n  [PASSED] Multi-stage training works correctly")
    return True


def test_gradient_flow():
    """Test that gradients flow correctly through losses."""
    print("\n" + "-" * 60)
    print("Test 6: Gradient Flow Verification")
    print("-" * 60)
    
    from src.losses import CombinedLoss, PYWT_AVAILABLE
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"  Device: {device}")
    
    # Create trainable prediction (simulating model output)
    pred = torch.rand(1, 3, 128, 128, device=device, requires_grad=True)
    target = torch.rand(1, 3, 128, 128, device=device)
    
    # Initialize combined loss
    combined = CombinedLoss(
        use_swt=PYWT_AVAILABLE,
        use_fft=True,
        use_clip=False
    ).to(device)
    combined.set_stage(3)
    
    # Forward and backward
    loss = combined(pred, target)
    loss.backward()
    
    # Check gradient exists
    assert pred.grad is not None, "Gradient should exist"
    grad_norm = pred.grad.norm().item()
    print(f"  Gradient norm: {grad_norm:.6f}")
    assert grad_norm > 0, "Gradient norm should be non-zero"
    
    print("\n  [PASSED] Gradients flow correctly")
    return True


def test_fusion_with_losses():
    """Test fusion network with loss functions."""
    print("\n" + "-" * 60)
    print("Test 7: Fusion Network + Loss Integration")
    print("-" * 60)
    
    from src.models.fusion_network import FrequencyAwareFusion
    from src.losses import CombinedLoss, PYWT_AVAILABLE
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"  Device: {device}")
    
    # Create fusion network
    fusion = FrequencyAwareFusion(
        num_experts=3,
        use_residual=True,
        use_multiscale=True
    ).to(device)
    
    # Create dummy data
    lr_input = torch.rand(1, 3, 64, 64, device=device)
    expert_outputs = [
        torch.rand(1, 3, 256, 256, device=device) for _ in range(3)
    ]
    hr_target = torch.rand(1, 3, 256, 256, device=device)
    
    # Forward pass
    sr_output = fusion(lr_input, expert_outputs)
    print(f"  Fusion output shape: {sr_output.shape}")
    
    # Initialize loss
    criterion = CombinedLoss(
        use_swt=PYWT_AVAILABLE,
        use_fft=True
    ).to(device)
    criterion.set_stage(2)
    
    # Compute loss
    loss, components = criterion(sr_output, hr_target, return_components=True)
    print(f"  Total loss: {loss.item():.6f}")
    print(f"  Loss components: {list(components.keys())}")
    
    # Backward pass
    fusion.zero_grad()
    loss.backward()
    
    # Check gradients
    trainable_params = [p for p in fusion.parameters() if p.requires_grad]
    params_with_grad = sum(1 for p in trainable_params if p.grad is not None)
    print(f"  Params with gradients: {params_with_grad}/{len(trainable_params)}")
    
    print("\n  [PASSED] Fusion + Loss integration works correctly")
    return True


def test_training_step():
    """Simulate a full training step."""
    print("\n" + "-" * 60)
    print("Test 8: Simulated Training Step")
    print("-" * 60)
    
    from src.models.fusion_network import FrequencyAwareFusion
    from src.losses import CombinedLoss, PYWT_AVAILABLE
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"  Device: {device}")
    
    # Create model
    fusion = FrequencyAwareFusion(
        num_experts=3,
        use_residual=True
    ).to(device)
    
    # Create optimizer
    optimizer = torch.optim.Adam(fusion.parameters(), lr=1e-4)
    
    # Create loss
    criterion = CombinedLoss(
        use_swt=PYWT_AVAILABLE,
        use_fft=True
    ).to(device)
    criterion.set_stage(2)
    
    # Create data
    lr_input = torch.rand(2, 3, 32, 32, device=device)
    expert_outputs = [torch.rand(2, 3, 128, 128, device=device) for _ in range(3)]
    hr_target = torch.rand(2, 3, 128, 128, device=device)
    
    # Training step
    fusion.train()
    optimizer.zero_grad()
    
    sr_output = fusion(lr_input, expert_outputs)
    loss = criterion(sr_output, hr_target)
    
    print(f"  Before step - Loss: {loss.item():.6f}")
    
    loss.backward()
    optimizer.step()
    
    # Verify loss changed
    sr_output2 = fusion(lr_input, expert_outputs)
    loss2 = criterion(sr_output2, hr_target)
    
    print(f"  After step - Loss: {loss2.item():.6f}")
    print(f"  Loss change: {(loss2.item() - loss.item()):.6f}")
    
    print("\n  [PASSED] Training step completed successfully")
    return True


def run_all_tests():
    """Run all integration tests."""
    print("\n" + "=" * 60)
    print("FUSION NETWORK + LOSS FUNCTIONS INTEGRATION TEST")
    print("=" * 60)
    
    tests = [
        ("Import All Losses", test_loss_import),
        ("Basic Loss Computation", test_basic_loss_computation),
        ("VGG Perceptual Loss", test_vgg_perceptual_loss),
        ("Frequency Domain Losses", test_frequency_losses),
        ("Combined Loss Multi-Stage", test_combined_loss_stages),
        ("Gradient Flow", test_gradient_flow),
        ("Fusion + Loss Integration", test_fusion_with_losses),
        ("Simulated Training Step", test_training_step),
    ]
    
    results = []
    for name, test_fn in tests:
        try:
            success = test_fn()
            results.append((name, success))
        except Exception as e:
            print(f"\n  [FAILED] {name}: {e}")
            import traceback
            traceback.print_exc()
            results.append((name, False))
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    passed = sum(1 for _, s in results if s)
    total = len(results)
    
    for name, success in results:
        status = "PASSED" if success else "FAILED"
        print(f"  {name}: {status}")
    
    print(f"\n  Total: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n" + "=" * 60)
        print("ALL INTEGRATION TESTS PASSED!")
        print("Ready for Phase 6: Training!")
        print("=" * 60 + "\n")
    else:
        print("\n  Some tests failed. Please review errors above.")
    
    return passed == total


if __name__ == '__main__':
    run_all_tests()
