"""
Phase 3 Dry-Run Verification Script
====================================
Tests:
1. Standalone LKA modules (shapes, gradients, param counts)
2. Full model with Phase 1+2+3 (forward pass, gradients, output validity)
3. YAML config compatibility
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import yaml


def test_standalone_lka():
    """Test all standalone LKA modules."""
    from src.models.large_kernel_attention import (
        LargeKernelAttention,
        LKABlock,
        EnhancedCrossBandWithLKA,
        EnhancedCollaborativeWithLKA,
    )
    
    results = []
    results.append("=" * 60)
    results.append("TEST 1: Standalone LKA Modules")
    results.append("=" * 60)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    results.append(f"Device: {device}")
    
    ok = True
    
    # 1a: Core LKA
    lka = LargeKernelAttention(dim=64, kernel_size=21).to(device)
    p = sum(pp.numel() for pp in lka.parameters())
    x = torch.randn(2, 64, 64, 64, device=device)
    out = lka(x)
    results.append(f"\n  LargeKernelAttention: {p:,} params")
    results.append(f"    Shape: {x.shape} -> {out.shape}")
    results.append(f"    NaN: {torch.isnan(out).any().item()}")
    if out.shape != x.shape:
        results.append("    FAIL: shape mismatch"); ok = False
    
    # 1b: LKA Block
    block = LKABlock(dim=64, kernel_size=21, ffn_ratio=2.0).to(device)
    p = sum(pp.numel() for pp in block.parameters())
    out = block(x)
    results.append(f"\n  LKABlock: {p:,} params")
    results.append(f"    Shape: {x.shape} -> {out.shape}")
    if out.shape != x.shape:
        results.append("    FAIL: shape mismatch"); ok = False
    
    # 1c: Enhanced Cross-Band (9 bands)
    ecb = EnhancedCrossBandWithLKA(dim=64, num_bands=9, num_heads=4, lka_kernel=21).to(device)
    p_ecb = sum(pp.numel() for pp in ecb.parameters())
    bands = [torch.randn(2, 3, 64, 64, device=device) for _ in range(9)]
    out_bands = ecb(bands)
    results.append(f"\n  EnhancedCrossBandWithLKA: {p_ecb:,} params")
    results.append(f"    Input: 9 bands, Output: {len(out_bands)} bands")
    for i, b in enumerate(out_bands):
        if b.shape != bands[0].shape:
            results.append(f"    FAIL: band {i} shape {b.shape}"); ok = False
        if torch.isnan(b).any():
            results.append(f"    FAIL: band {i} has NaN"); ok = False
    
    # Gradient check
    ecb.zero_grad()
    loss = sum(b.mean() for b in ecb(bands))
    loss.backward()
    g = sum(1 for pp in ecb.parameters() if pp.requires_grad and pp.grad is not None and pp.grad.abs().sum() > 0)
    t = sum(1 for pp in ecb.parameters() if pp.requires_grad)
    results.append(f"    Gradients: {g}/{t}")
    if g == 0:
        results.append("    FAIL: no gradients"); ok = False
    
    # 1d: Enhanced Collaborative (3 experts)
    ecl = EnhancedCollaborativeWithLKA(
        num_experts=3, feature_dim=128, num_heads=8, lka_kernel=21
    ).to(device)
    p_ecl = sum(pp.numel() for pp in ecl.parameters())
    feats = {
        'hat': torch.randn(2, 180, 64, 64, device=device),
        'dat': torch.randn(2, 180, 64, 64, device=device),
        'nafnet': torch.randn(2, 64, 64, 64, device=device),
    }
    outputs = [torch.randn(2, 3, 256, 256, device=device) for _ in range(3)]
    enhanced = ecl(feats, outputs)
    results.append(f"\n  EnhancedCollaborativeWithLKA: {p_ecl:,} params")
    results.append(f"    Input: 3 experts, Output: {len(enhanced)} experts")
    for i, e in enumerate(enhanced):
        if e.shape != outputs[0].shape:
            results.append(f"    FAIL: expert {i} shape {e.shape}"); ok = False
    
    # Gradient check
    ecl.zero_grad()
    loss2 = sum(e.mean() for e in ecl(feats, outputs))
    loss2.backward()
    g2 = sum(1 for pp in ecl.parameters() if pp.requires_grad and pp.grad is not None and pp.grad.abs().sum() > 0)
    t2 = sum(1 for pp in ecl.parameters() if pp.requires_grad)
    results.append(f"    Gradients: {g2}/{t2}")
    if g2 == 0:
        results.append("    FAIL: no gradients"); ok = False
    
    results.append(f"\n  LKA param total (cross-band + collab): {p_ecb + p_ecl:,}")
    results.append(f"\nTest 1 Result: {'PASS' if ok else 'FAIL'}")
    return results, ok


def test_full_model():
    """Test full model with Phase 1+2+3."""
    from src.models.enhanced_fusion import CompleteEnhancedFusionSR
    
    results = []
    results.append("\n" + "=" * 60)
    results.append("TEST 2: Full Model (Phase 1 + 2 + 3)")
    results.append("=" * 60)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    model = CompleteEnhancedFusionSR(
        expert_ensemble=None,
        num_experts=3,
        num_bands=3,
        block_size=8,
        upscale=4,
        fusion_dim=64,
        num_heads=4,
        refine_depth=4,
        refine_channels=64,
        enable_hierarchical=True,
        enable_multi_domain_freq=True,   # Phase 2
        enable_lka=True,                 # Phase 3
        enable_dynamic_selection=True,
        enable_cross_band_attn=True,
        enable_adaptive_bands=True,
        enable_multi_resolution=True,
        enable_collaborative=True,
    ).to(device)
    
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    results.append(f"Total trainable params: {total_params:,}")
    
    results.append("\nModule breakdown:")
    for name, child in model.named_children():
        p = sum(pp.numel() for pp in child.parameters() if pp.requires_grad)
        if p > 0:
            results.append(f"  {name}: {p:,}")
    child_params = sum(pp.numel() for n, c in model.named_children() for pp in c.parameters() if pp.requires_grad)
    orphan = total_params - child_params
    if orphan > 0:
        results.append(f"  (direct params): {orphan:,}")
    
    # Forward pass
    B = 2
    lr_input = torch.randn(B, 3, 64, 64, device=device)
    expert_outputs = {
        'hat': torch.randn(B, 3, 256, 256, device=device),
        'dat': torch.randn(B, 3, 256, 256, device=device),
        'nafnet': torch.randn(B, 3, 256, 256, device=device),
    }
    expert_features = {
        'hat': torch.randn(B, 180, 64, 64, device=device),
        'dat': torch.randn(B, 180, 64, 64, device=device),
        'nafnet': torch.randn(B, 64, 64, 64, device=device),
    }
    
    output = model.forward_with_precomputed(lr_input, expert_outputs, expert_features)
    
    results.append(f"\nForward pass:")
    results.append(f"  Input: {lr_input.shape}")
    results.append(f"  Output: {output.shape}")
    results.append(f"  Range: [{output.min():.4f}, {output.max():.4f}]")
    results.append(f"  NaN: {torch.isnan(output).any().item()}")
    results.append(f"  Inf: {torch.isinf(output).any().item()}")
    
    # Gradient check
    model.zero_grad()
    loss = output.mean()
    loss.backward()
    
    grads = sum(1 for p in model.parameters() if p.requires_grad and p.grad is not None and p.grad.abs().sum() > 0)
    total = sum(1 for p in model.parameters() if p.requires_grad)
    results.append(f"\nGradient flow: {grads}/{total} ({100*grads/total:.1f}%)")
    
    results.append("Per-module gradient flow:")
    for name, child in model.named_children():
        t = sum(1 for p in child.parameters() if p.requires_grad)
        g = sum(1 for p in child.parameters() if p.requires_grad and p.grad is not None and p.grad.abs().sum() > 0)
        if t > 0:
            results.append(f"  {name}: {g}/{t}")
    
    # LKA-specific gradient check
    lka_grads = sum(1 for n, p in model.named_parameters()
                    if p.requires_grad and p.grad is not None
                    and p.grad.abs().sum() > 0
                    and ('lka' in n or 'lka_block' in n or 'lka_global' in n))
    results.append(f"\nLKA-related params with gradients: {lka_grads}")
    
    # Assertions
    ok = True
    if output.shape != (B, 3, 256, 256):
        results.append(f"FAIL: Wrong output shape {output.shape}"); ok = False
    if torch.isnan(output).any():
        results.append("FAIL: NaN in output"); ok = False
    if torch.isinf(output).any():
        results.append("FAIL: Inf in output"); ok = False
    if grads == 0:
        results.append("FAIL: No gradients"); ok = False
    if lka_grads == 0:
        results.append("FAIL: LKA modules not receiving gradients"); ok = False
    
    results.append(f"\nTest 2 Result: {'PASS' if ok else 'FAIL'}")
    return results, ok


def test_yaml_config():
    """Verify YAML config has Phase 3 settings."""
    results = []
    results.append("\n" + "=" * 60)
    results.append("TEST 3: YAML Config Compatibility")
    results.append("=" * 60)
    
    config_path = os.path.join(os.path.dirname(__file__), '..', 'configs', 'train_config.yaml')
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    fusion = config.get('model', {}).get('fusion', {})
    
    ok = True
    checks = {
        'enable_hierarchical': True,
        'enable_multi_domain_freq': True,
        'enable_lka': True,
    }
    
    for key, expected in checks.items():
        val = fusion.get(key)
        status = "PASS" if val == expected else "FAIL"
        results.append(f"  {key}: {val} (expected {expected}) - {status}")
        if val != expected:
            ok = False
    
    results.append(f"\nTest 3 Result: {'PASS' if ok else 'FAIL'}")
    return results, ok


if __name__ == '__main__':
    all_results = []
    all_passed = True
    
    for test_fn in [test_standalone_lka, test_full_model, test_yaml_config]:
        try:
            results, passed = test_fn()
            all_results.extend(results)
            all_passed = all_passed and passed
        except Exception as e:
            import traceback
            all_results.append(f"\nEXCEPTION in {test_fn.__name__}:")
            all_results.append(traceback.format_exc())
            all_passed = False
    
    all_results.append("\n" + "=" * 60)
    all_results.append(f"OVERALL: {'ALL TESTS PASSED' if all_passed else 'SOME TESTS FAILED'}")
    all_results.append("=" * 60)
    
    output_path = os.path.join(os.path.dirname(__file__), 'phase3_results.txt')
    with open(output_path, 'w', encoding='ascii', errors='replace') as f:
        f.write('\n'.join(all_results))
    
    print(f"Results written to {output_path}")
    for line in all_results:
        print(line)
