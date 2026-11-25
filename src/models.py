from __future__ import annotations

import torch
import torch.nn as nn

from .innovations import CoordAtt, FourierConv, FourierSR


class ConvGNReLU(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=None, groups=1):
        super().__init__()
        if padding is None:
            padding = kernel_size // 2
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, groups=groups, bias=False),
            nn.GroupNorm(min(8, out_channels), out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1):
        super().__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size, stride, kernel_size // 2, groups=in_channels, bias=False)
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, bias=False)

    def forward(self, x):
        return self.pointwise(self.depthwise(x))


class FSRB_A(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.main = nn.Sequential(
            FourierConv(channels, channels, 1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
        )
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.activation(self.main(x) + x)


class FSRB_B(nn.Module):
    def __init__(self, in_channels, out_channels, stride=2):
        super().__init__()
        self.main = nn.Sequential(
            DepthwiseSeparableConv(in_channels, out_channels, 3, stride),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            DepthwiseSeparableConv(out_channels, out_channels, 3, 1),
            nn.BatchNorm2d(out_channels),
        )
        self.shortcut = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, stride, bias=False),
            nn.BatchNorm2d(out_channels),
        )
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.activation(self.main(x) + self.shortcut(x))


class FSRBNet(nn.Module):
    def __init__(self, num_classes=9, in_channels=3):
        super().__init__()
        self.stem = nn.Sequential(
            ConvGNReLU(in_channels, 64, 7, 2, 3),
            nn.MaxPool2d(4, 4),
            ConvGNReLU(64, 128, 1, 1, 0),
        )
        self.stage1 = nn.Sequential(FSRB_A(128), FSRB_B(128, 128, 2))
        self.feature_enhance = FourierSR(128, 128)
        self.stage2 = nn.Sequential(
            ConvGNReLU(128, 128, 3, 1),
            FSRB_B(128, 256, 2),
            FSRB_A(256),
            FSRB_B(256, 512, 2),
            FSRB_A(512),
            CoordAtt(512),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(512, num_classes)

    def forward_features(self, x):
        x = self.stem(x)
        x = self.stage1(x)
        x = self.feature_enhance(x)
        x = self.stage2(x)
        return torch.flatten(self.avgpool(x), 1)

    def forward(self, x):
        features = self.forward_features(x)
        return features, self.classifier(features)


SOTA_MODELS = (
    "resnet18",
    "mobilenet_v3_small",
    "efficientnet_b0",
    "fasternet_t0",
    "tinyvit_5m",
    "mobilevit_xxs",
    "repvit_m1_1",
    "swiftformer_xs",
    "fsrb_net",
)


def build_baseline(model_name, num_classes, pretrained=False):
    from torchvision import models as tvm
    tv_map = {
        "resnet18": (tvm.resnet18, "fc"),
        "mobilenet_v3_small": (tvm.mobilenet_v3_small, "classifier"),
        "efficientnet_b0": (tvm.efficientnet_b0, "classifier"),
    }
    if model_name in tv_map:
        builder, head_name = tv_map[model_name]
        try:
            model = builder(weights="DEFAULT" if pretrained else None)
        except TypeError:
            model = builder(pretrained=pretrained)
        head = getattr(model, head_name)
        if isinstance(head, nn.Linear):
            setattr(model, head_name, nn.Linear(head.in_features, num_classes))
        elif isinstance(head, nn.Sequential):
            for index in range(len(head) - 1, -1, -1):
                if isinstance(head[index], nn.Linear):
                    head[index] = nn.Linear(head[index].in_features, num_classes)
                    break
        return model
    timm_map = {
        "repvit_m1_1": "repvit_m1_1",
        "fasternet_t0": "fasternet_t0",
        "swiftformer_xs": "swiftformer_xs",
        "tinyvit_5m": "tiny_vit_5m_224",
        "mobilevit_xxs": "mobilevit_xxs",
    }
    if model_name in timm_map:
        try:
            import timm
        except ImportError as exc:
            raise ImportError("Install timm to use the selected SOTA backbone") from exc
        return timm.create_model(timm_map[model_name], pretrained=pretrained, num_classes=num_classes)
    raise ValueError(f"Unsupported model: {model_name}")


class TransferModel(nn.Module):
    def __init__(self, model_name, num_classes, pretrained=False):
        super().__init__()
        self.model_name = model_name
        if model_name == "fsrb_net":
            self.model = FSRBNet(num_classes=num_classes)
            self.native_features = True
        elif model_name in {"A1", "A2", "A3", "A4", "A5", "A6"}:
            from .ablations import build_ablation
            self.model = build_ablation(model_name, num_classes)
            self.native_features = True
        else:
            self.model = build_baseline(model_name, num_classes, pretrained)
            self.native_features = False
            self.features = None
            linear_layers = [(name, module) for name, module in self.model.named_modules() if isinstance(module, nn.Linear)]
            if not linear_layers:
                raise ValueError(f"No linear classifier found in {model_name}")
            self.classifier_name, classifier = linear_layers[-1]
            self.feature_dim = classifier.in_features
            self.hook = classifier.register_forward_pre_hook(self._capture_features)

    def _capture_features(self, module, inputs):
        self.features = inputs[0].flatten(1) if inputs[0].ndim > 2 else inputs[0]

    def forward(self, x):
        if self.native_features:
            return self.model(x)
        logits = self.model(x)
        if self.features is None:
            raise RuntimeError("Feature hook did not capture classifier input")
        return self.features, logits

    def remove_hook(self):
        if getattr(self, "hook", None) is not None:
            self.hook.remove()
            self.hook = None
