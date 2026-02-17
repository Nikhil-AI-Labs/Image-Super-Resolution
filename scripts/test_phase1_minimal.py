"""Minimal Phase 1 verification with corrected dimensions."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F
import yaml

results = []

# Test 1: Model creation + param count with correct defaults
try:
    from src.models.enhanced_fusion import CompleteEnhancedFusionSR
    model = CompleteEnhancedFusionSR(
        expert_ensemble=None, num_experts=3, num_bands=3, block_size=8, upscale=4,
        fusion_dim=64, num_heads=4, refine_depth=4, refine_channels=64,
        enable_hierarchical=True,
        enable_dynamic_selection=True, enable_cross_band_attn=True,
        enable_adaptive_bands=True, enable_multi_resolution=True,
        enable_collaborative=True,
    )
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    target = 900000
    in_range = 450000 <= trainable <= 1350000
    results.append(f"PARAMS={trainable} TARGET={target} IN_RANGE={in_range}")
    
    for name, mod in model.named_children():
        p = sum(pp.numel() for pp in mod.parameters() if pp.requires_grad)
        if p > 0:
            results.append(f"  {name}={p}")
    results.append(f"CREATE=PASS")
except Exception as e:
    results.append(f"CREATE=FAIL {e}")

# Test 2: Forward pass
try:
    lr = torch.randn(2, 3, 64, 64)
    eo = {k: torch.randn(2, 3, 256, 256).clamp(0,1) for k in ['hat','dat','nafnet']}
    model.eval()
    with torch.no_grad():
        out = model.forward_with_precomputed(lr, eo)
    shape_ok = list(out.shape) == [2, 3, 256, 256]
    range_ok = out.min().item() >= 0 and out.max().item() <= 1
    nan_ok = not torch.isnan(out).any().item()
    results.append(f"SHAPE={list(out.shape)} OK={shape_ok}")
    results.append(f"RANGE=[{out.min().item():.4f},{out.max().item():.4f}] OK={range_ok}")
    results.append(f"NAN={torch.isnan(out).any().item()} OK={nan_ok}")
    results.append(f"FORWARD={'PASS' if shape_ok and range_ok and nan_ok else 'FAIL'}")
except Exception as e:
    results.append(f"FORWARD=FAIL {e}")

# Test 3: Gradients
try:
    model.train()
    model.zero_grad()
    out2 = model.forward_with_precomputed(lr, eo)
    loss = F.l1_loss(out2, torch.randn(2,3,256,256).clamp(0,1))
    loss.backward()
    total_p, with_g = 0, 0
    no_grad_names = []
    for n, p in model.named_parameters():
        if p.requires_grad:
            total_p += 1
            if p.grad is not None and p.grad.abs().sum() > 0:
                with_g += 1
            else:
                no_grad_names.append(n)
    ratio = with_g / total_p if total_p > 0 else 0
    grad_ok = ratio > 0.5
    results.append(f"GRADS={with_g}/{total_p} ({ratio:.1%}) OK={grad_ok}")
    if no_grad_names:
        results.append(f"NO_GRAD={no_grad_names[:5]}")
    results.append(f"GRADIENT={'PASS' if grad_ok else 'FAIL'}")
except Exception as e:
    results.append(f"GRADIENT=FAIL {e}")

# Test 4: YAML config
try:
    with open('configs/train_config.yaml', 'r') as f:
        cfg = yaml.safe_load(f)
    fc = cfg['model']['fusion']
    checks = {
        'fusion_dim': fc.get('fusion_dim') == 64,
        'num_heads': fc.get('num_heads') == 4,
        'refine_depth': fc.get('refine_depth') == 4,
        'refine_channels': fc.get('refine_channels') == 64,
        'enable_hierarchical': fc.get('enable_hierarchical') == True,
    }
    all_ok = all(checks.values())
    results.append(f"YAML={checks} ALL_OK={all_ok}")
    results.append(f"CONFIG={'PASS' if all_ok else 'FAIL'}")
except Exception as e:
    results.append(f"CONFIG=FAIL {e}")

# Summary
passes = sum(1 for r in results if '=PASS' in r)
fails = sum(1 for r in results if '=FAIL' in r)
results.append(f"SUMMARY: {passes} PASS, {fails} FAIL")

with open('scripts/phase1_final_results.txt', 'w', encoding='utf-8') as f:
    f.write('\n'.join(results))
print('DONE')
