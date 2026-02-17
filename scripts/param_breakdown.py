import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch

def count_params(fusion_dim, refine_channels, refine_depth, num_heads):
    from src.models.enhanced_fusion import CompleteEnhancedFusionSR
    model = CompleteEnhancedFusionSR(
        expert_ensemble=None, num_experts=3, num_bands=3, block_size=8, upscale=4,
        fusion_dim=fusion_dim, num_heads=num_heads, refine_depth=refine_depth, 
        refine_channels=refine_channels,
        enable_hierarchical=True,
        enable_dynamic_selection=True, enable_cross_band_attn=True,
        enable_adaptive_bands=True, enable_multi_resolution=True,
        enable_collaborative=True,
    )
    breakdown = {}
    for name, mod in model.named_children():
        p = sum(pp.numel() for pp in mod.parameters() if pp.requires_grad)
        if p > 0:
            breakdown[name] = p
    total = sum(p.numel() for p in model.parameters() if p.requires_grad)
    breakdown['_TOTAL'] = total
    return breakdown

# Test different configurations
configs = [
    (128, 128, 6, 8),  # Current
    (64, 64, 6, 8),    # Halved
    (64, 64, 4, 4),    # Smaller
    (48, 48, 4, 4),    # Even smaller
    (80, 80, 5, 8),    # Mid-range
]

results = []
for fd, rc, rd, nh in configs:
    b = count_params(fd, rc, rd, nh)
    line = f"dim={fd} ref_ch={rc} ref_d={rd} heads={nh} => TOTAL={b['_TOTAL']:,}"
    results.append(line)
    for k, v in sorted(b.items()):
        if k != '_TOTAL':
            results.append(f"  {k}: {v:,}")

with open('scripts/param_sweep.txt', 'w', encoding='utf-8') as f:
    f.write('\n'.join(results))
print('SWEEP_DONE')
