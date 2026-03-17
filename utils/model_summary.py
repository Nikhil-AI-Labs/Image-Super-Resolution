"""
Model summary utilities — FLOPs and parameter counting.
Based on the official NTIRE2026_ImageSR_x4 baseline.
"""

import torch
import torch.nn as nn


def get_model_flops(model, input_size, print_per_layer_stat=False, as_strings=False):
    """
    Estimate FLOPs for a model given input size.

    Args:
        model: nn.Module
        input_size: tuple (C, H, W) or (B, C, H, W)
        print_per_layer_stat: print per-layer statistics
        as_strings: return human-readable strings

    Returns:
        flops: total FLOPs (float or string)
    """
    try:
        from fvcore.nn import FlopCountAnalysis
        if len(input_size) == 3:
            input_tensor = torch.randn(1, *input_size).to(next(model.parameters()).device)
        else:
            input_tensor = torch.randn(*input_size).to(next(model.parameters()).device)

        flops = FlopCountAnalysis(model, input_tensor)
        total_flops = flops.total()

        if as_strings:
            if total_flops >= 1e12:
                return f"{total_flops / 1e12:.2f} TFLOPs"
            elif total_flops >= 1e9:
                return f"{total_flops / 1e9:.2f} GFLOPs"
            elif total_flops >= 1e6:
                return f"{total_flops / 1e6:.2f} MFLOPs"
            else:
                return f"{total_flops:.0f} FLOPs"

        return total_flops

    except ImportError:
        print("Warning: fvcore not installed. Cannot compute FLOPs.")
        return 0


def get_model_params(model, as_strings=False):
    """Count total parameters in a model."""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params = total_params - trainable_params

    if as_strings:
        return {
            'total': f"{total_params / 1e6:.2f}M",
            'trainable': f"{trainable_params / 1e6:.2f}M",
            'frozen': f"{frozen_params / 1e6:.2f}M",
        }

    return {
        'total': total_params,
        'trainable': trainable_params,
        'frozen': frozen_params,
    }
