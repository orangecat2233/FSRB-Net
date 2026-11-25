from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn as nn

from src.ablations import build_ablation
from src.models import FSRBNet, build_baseline


def fallback_macs(model, inputs):
    total = 0
    handles = []

    def conv_hook(module, input_values, output):
        nonlocal total
        batch = output.shape[0]
        out_channels = output.shape[1]
        out_h, out_w = output.shape[-2:]
        kernel_h, kernel_w = module.kernel_size
        per_output = (module.in_channels // module.groups) * kernel_h * kernel_w
        total += batch * out_channels * out_h * out_w * per_output

    def linear_hook(module, input_values, output):
        nonlocal total
        batch = output.shape[0] if output.ndim > 1 else 1
        total += batch * module.in_features * module.out_features

    for module in model.modules():
        if isinstance(module, nn.Conv2d):
            handles.append(module.register_forward_hook(conv_hook))
        elif isinstance(module, nn.Linear):
            handles.append(module.register_forward_hook(linear_hook))
    with torch.no_grad():
        model(*inputs)
    for handle in handles:
        handle.remove()
    return total


def main():
    parser = argparse.ArgumentParser(description="Measure model complexity")
    parser.add_argument("--dataset", choices=("HUST", "CWRU"), required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--checkpoint", default=None)
    args = parser.parse_args()
    classes = 9 if args.dataset == "HUST" else 10
    if args.model in {"A1", "A2", "A3", "A4", "A5", "A6"}:
        model = build_ablation(args.model, classes)
    elif args.model == "fsrb_net":
        model = FSRBNet(classes)
    else:
        model = build_baseline(args.model, classes, pretrained=False)
    model.eval()
    dummy = torch.randn(1, 3, args.image_size, args.image_size)
    try:
        from thop import profile
        macs, _ = profile(model, inputs=(dummy,), verbose=False)
        counter = "thop"
    except ImportError:
        macs = fallback_macs(model, (dummy,))
        counter = "conv_linear_fallback"
    params = sum(parameter.numel() for parameter in model.parameters())
    result = {
        "dataset": args.dataset,
        "model": args.model,
        "classes": classes,
        "input_shape": [1, 3, args.image_size, args.image_size],
        "params_m": params / 1e6,
        "macs_g": macs / 1e9,
        "flops_g": 2.0 * macs / 1e9,
        "pth_mb": Path(args.checkpoint).stat().st_size / (1024 ** 2) if args.checkpoint else None,
        "counter": counter,
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
