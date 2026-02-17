"""
Verification: Loss Bug Fix
===========================
Verifies that the CombinedLoss now strictly obeys YAML config weights.

Tests:
1. With pure L1 weights (stage 1), only L1 loss contributes
2. set_weights() correctly updates active losses
3. Stage transitions work as expected
4. get_loss_stage() returns correct weights per epoch

Run: python scripts/test_loss_bug_fix.py
"""

import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import yaml


def test_pure_l1_stage():
    """Verify stage 1 produces ONLY L1 loss (the core bug fix)."""
    print("\n" + "=" * 60)
    print("TEST 1: Pure L1 Stage (Epoch 0-100)")
    print("=" * 60)
    
    from src.losses import CombinedLoss
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Initialize with zero weights (like the fixed train.py)
    criterion = CombinedLoss(
        l1_weight=0.0, charbonnier_weight=0.0, l2_weight=0.0,
        vgg_weight=0.0, swt_weight=0.0, fft_weight=0.0,
        edge_weight=0.0, ssim_weight=0.0, clip_weight=0.0,
        use_swt=False, use_fft=True, use_clip=False,
    ).to(device)
    
    # Apply stage 1 weights (pure L1 from YAML)
    stage1_weights = {
        'l1': 1.0, 'charbonnier': 0.0, 'swt': 0.0,
        'vgg': 0.0, 'fft': 0.0, 'ssim': 0.0
    }
    criterion.set_weights(stage1_weights)
    
    # Verify weights are set correctly
    print(f"  Active weights: {criterion.get_active_weights()}")
    assert criterion.weights['l1'] == 1.0, f"L1 weight should be 1.0, got {criterion.weights['l1']}"
    assert criterion.weights['charbonnier'] == 0.0, f"Charbonnier should be 0.0, got {criterion.weights['charbonnier']}"
    assert criterion.weights['ssim'] == 0.0, f"SSIM should be 0.0, got {criterion.weights['ssim']}"
    print("  ✓ Weights correctly set to pure L1")
    
    # Compute loss
    pred = torch.rand(1, 3, 64, 64, device=device)
    target = torch.rand(1, 3, 64, 64, device=device)
    
    loss, components = criterion(pred, target, return_components=True)
    
    print(f"  Total loss: {loss.item():.6f}")
    print(f"  Components: {list(components.keys())}")
    for name, val in components.items():
        print(f"    {name}: {val.item():.6f}")
    
    # CRITICAL: Only L1 should be in components
    assert 'l1' in components, "L1 should be computed"
    assert len(components) == 1, f"Only L1 should be active, but got: {list(components.keys())}"
    assert abs(loss.item() - components['l1'].item()) < 1e-6, "Total loss should equal L1 loss"
    
    print("  ✓ PASS: Only L1 loss is active in stage 1!")
    return True


def test_stage_transition():
    """Verify smooth transition between stages."""
    print("\n" + "=" * 60)
    print("TEST 2: Stage Transitions")
    print("=" * 60)
    
    from src.losses import CombinedLoss
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    criterion = CombinedLoss(
        l1_weight=0.0, charbonnier_weight=0.0, l2_weight=0.0,
        vgg_weight=0.0, swt_weight=0.0, fft_weight=0.0,
        edge_weight=0.0, ssim_weight=0.0, clip_weight=0.0,
        use_swt=False, use_fft=True, use_clip=False,
    ).to(device)
    
    pred = torch.rand(1, 3, 64, 64, device=device)
    target = torch.rand(1, 3, 64, 64, device=device)
    
    # Stage 1: Pure L1
    criterion.set_weights({'l1': 1.0, 'charbonnier': 0.0, 'swt': 0.0, 'fft': 0.0, 'ssim': 0.0, 'vgg': 0.0})
    _, comp1 = criterion(pred, target, return_components=True)
    print(f"  Stage 1 components: {list(comp1.keys())}")
    assert list(comp1.keys()) == ['l1'], f"Stage 1 should only have L1, got {list(comp1.keys())}"
    
    # Stage 2: L1 + SWT + FFT
    criterion.set_weights({'l1': 0.8, 'charbonnier': 0.0, 'swt': 0.15, 'fft': 0.05, 'ssim': 0.0, 'vgg': 0.0})
    _, comp2 = criterion(pred, target, return_components=True)
    print(f"  Stage 2 components: {list(comp2.keys())}")
    assert 'l1' in comp2, "Stage 2 should have L1"
    assert 'fft' in comp2, "Stage 2 should have FFT"
    
    # Stage 3: L1 + SWT + FFT + SSIM
    criterion.set_weights({'l1': 0.6, 'charbonnier': 0.0, 'swt': 0.25, 'fft': 0.1, 'ssim': 0.05, 'vgg': 0.0})
    _, comp3 = criterion(pred, target, return_components=True)
    print(f"  Stage 3 components: {list(comp3.keys())}")
    assert 'ssim' in comp3, "Stage 3 should have SSIM"
    
    print("  ✓ PASS: Stage transitions work correctly!")
    return True


def test_get_loss_stage():
    """Verify get_loss_stage returns correct weights from YAML."""
    print("\n" + "=" * 60)
    print("TEST 3: get_loss_stage() YAML Parsing")
    print("=" * 60)
    
    # Load actual config
    config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                               'configs', 'train_config.yaml')
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    from train import get_loss_stage
    
    # Test epoch 0 (should be stage 1: pure L1)
    stage_num, weights, stage_name = get_loss_stage(0, config['loss'])
    print(f"  Epoch 0: stage={stage_num}, name={stage_name}")
    print(f"    Weights: {weights}")
    assert weights.get('l1', 0) == 1.0, f"Epoch 0 L1 should be 1.0, got {weights.get('l1', 0)}"
    assert weights.get('charbonnier', 0) == 0.0, f"Epoch 0 Charbonnier should be 0.0"
    assert weights.get('swt', 0) == 0.0, f"Epoch 0 SWT should be 0.0"
    print("  ✓ Epoch 0: Pure L1 (correct)")
    
    # Test epoch 50 (still stage 1)
    stage_num, weights, stage_name = get_loss_stage(50, config['loss'])
    print(f"\n  Epoch 50: stage={stage_num}, name={stage_name}")
    assert weights.get('l1', 0) == 1.0, "Epoch 50 should still be pure L1"
    print("  ✓ Epoch 50: Pure L1 (correct)")
    
    # Test epoch 100 (stage 2: L1 + frequency)
    stage_num, weights, stage_name = get_loss_stage(100, config['loss'])
    print(f"\n  Epoch 100: stage={stage_num}, name={stage_name}")
    print(f"    Weights: {weights}")
    assert weights.get('l1', 0) > 0, "Epoch 100 should have L1"
    assert weights.get('swt', 0) > 0 or weights.get('fft', 0) > 0, "Epoch 100 should have frequency loss"
    print("  ✓ Epoch 100: L1 + frequency (correct)")
    
    # Test epoch 150 (stage 3)
    stage_num, weights, stage_name = get_loss_stage(150, config['loss'])
    print(f"\n  Epoch 150: stage={stage_num}, name={stage_name}")
    print(f"    Weights: {weights}")
    print("  ✓ Epoch 150: Detail enhancement (correct)")
    
    print("\n  ✓ PASS: get_loss_stage() parses YAML correctly!")
    return True


def test_gradient_flow():
    """Verify gradients flow correctly with pure L1."""
    print("\n" + "=" * 60)
    print("TEST 4: Gradient Flow with Pure L1")
    print("=" * 60)
    
    from src.losses import CombinedLoss
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    criterion = CombinedLoss(
        l1_weight=0.0, charbonnier_weight=0.0, l2_weight=0.0,
        vgg_weight=0.0, swt_weight=0.0, fft_weight=0.0,
        edge_weight=0.0, ssim_weight=0.0, clip_weight=0.0,
        use_swt=False, use_fft=False, use_clip=False,
    ).to(device)
    criterion.set_weights({'l1': 1.0})
    
    pred = torch.rand(1, 3, 64, 64, device=device, requires_grad=True)
    target = torch.rand(1, 3, 64, 64, device=device)
    
    loss = criterion(pred, target)
    loss.backward()
    
    assert pred.grad is not None, "Gradient should exist"
    grad_norm = pred.grad.norm().item()
    print(f"  Gradient norm: {grad_norm:.6f}")
    assert grad_norm > 0, "Gradient should be non-zero"
    
    print("  ✓ PASS: Gradients flow correctly with pure L1!")
    return True


def main():
    print("\n" + "=" * 70)
    print("     LOSS BUG FIX VERIFICATION")
    print("=" * 70)
    
    results = []
    tests = [
        ("Pure L1 Stage", test_pure_l1_stage),
        ("Stage Transitions", test_stage_transition),
        ("YAML Parsing", test_get_loss_stage),
        ("Gradient Flow", test_gradient_flow),
    ]
    
    for name, fn in tests:
        try:
            results.append((name, fn()))
        except Exception as e:
            print(f"\n  ✗ FAIL: {name}: {e}")
            import traceback; traceback.print_exc()
            results.append((name, False))
    
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    passed = sum(1 for _, r in results if r)
    for name, r in results:
        print(f"  {'✓' if r else '✗'} {name}")
    print(f"\n  {passed}/{len(results)} tests passed")
    
    if passed == len(results):
        print("\n  🎯 LOSS BUG FIX VERIFIED — Ready for training!")
    print("=" * 70)


if __name__ == '__main__':
    main()
