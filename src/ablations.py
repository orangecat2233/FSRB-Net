from __future__ import annotations

import torch.nn as nn

from .models import FSRBNet


def standard_conv(in_channels, out_channels, stride=1):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
    )


def build_ablation(name, num_classes=9, in_channels=3):
    if name not in {"A1", "A2", "A3", "A4", "A5", "A6"}:
        raise ValueError(f"Unsupported ablation: {name}")
    model = FSRBNet(num_classes=num_classes, in_channels=in_channels)
    if name == "A2":
        model.stage1[0] = standard_conv(128, 128, 1)
        model.stage1[1] = standard_conv(128, 128, 2)
        model.stage2[1] = standard_conv(128, 256, 2)
        model.stage2[2] = standard_conv(256, 256, 1)
        model.stage2[3] = standard_conv(256, 512, 2)
        model.stage2[4] = standard_conv(512, 512, 1)
    elif name == "A3":
        model.feature_enhance = nn.Identity()
    elif name == "A4":
        model.feature_enhance = standard_conv(128, 128, 1)
    elif name == "A5":
        model.stage2 = nn.Sequential(*list(model.stage2.children())[:-1])
    return model

ABLATION_PLAN = (
    ("A1", "source_only"),
    ("A2", "dann"),
    ("A3", "dann"),
    ("A4", "dann"),
    ("A5", "dann"),
    ("A6", "dann"),
)
