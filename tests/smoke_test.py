from __future__ import annotations

import torch
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.adaptation import CORALLoss, DWDANLoss, MKMMDLoss, MMDLoss
from src.ablations import build_ablation


def main():
    expected = {
        (9, "A1"): 1.6352,
        (9, "A2"): 4.9395,
        (9, "A3"): 1.6138,
        (9, "A4"): 1.7615,
        (9, "A5"): 1.6095,
        (9, "A6"): 1.6352,
        (10, "A1"): 1.6357,
        (10, "A6"): 1.6357,
    }
    for (classes, key), expected_m in expected.items():
        model = build_ablation(key, classes).eval()
        params_m = sum(p.numel() for p in model.parameters()) / 1e6
        assert abs(params_m - expected_m) < 0.0001, (classes, key, params_m, expected_m)
        x = torch.randn(2, 3, 128, 128)
        with torch.no_grad():
            features, logits = model(x)
        assert features.shape == (2, 512)
        assert logits.shape == (2, classes)

    source = torch.randn(8, 32, requires_grad=True)
    target = torch.randn(8, 32, requires_grad=True)
    for criterion in (MMDLoss(), CORALLoss(), MKMMDLoss()):
        loss = criterion(source, target)
        assert loss.ndim == 0 and torch.isfinite(loss)
    dwdan = DWDANLoss(num_classes=4)
    labels = torch.randint(0, 4, (8,))
    logits = torch.randn(8, 4, requires_grad=True)
    result = dwdan(source, labels, target, logits)
    assert "joint_loss" in result and torch.isfinite(result["joint_loss"])
    print("Smoke tests passed")


if __name__ == "__main__":
    main()
