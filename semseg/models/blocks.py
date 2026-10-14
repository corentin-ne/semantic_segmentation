"""Shared building blocks and backbone wrappers."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tvm


class ConvBNAct(nn.Sequential):
    def __init__(self, in_ch: int, out_ch: int, kernel: int = 3, stride: int = 1, dilation: int = 1,
                 groups: int = 1, act: type[nn.Module] | None = nn.ReLU):
        padding = dilation * (kernel - 1) // 2
        layers: list[nn.Module] = [
            nn.Conv2d(in_ch, out_ch, kernel, stride, padding, dilation, groups, bias=False),
            nn.BatchNorm2d(out_ch),
        ]
        if act is not None:
            layers.append(act(inplace=True))
        super().__init__(*layers)


class SeparableConv(nn.Sequential):
    """Depthwise 3x3 + pointwise 1x1: ~8-9x fewer FLOPs than a dense 3x3."""

    def __init__(self, in_ch: int, out_ch: int, dilation: int = 1, act=nn.ReLU):
        super().__init__(
            ConvBNAct(in_ch, in_ch, 3, dilation=dilation, groups=in_ch, act=act),
            ConvBNAct(in_ch, out_ch, 1, act=act),
        )


def upsample_like(x: torch.Tensor, ref: torch.Tensor | tuple[int, int]) -> torch.Tensor:
    size = ref.shape[-2:] if isinstance(ref, torch.Tensor) else ref
    return F.interpolate(x, size=size, mode="bilinear", align_corners=False)


def init_decoder(module: nn.Module) -> None:
    for m in module.modules():
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
            nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)


class MobileNetV3Encoder(nn.Module):
    """MobileNetV3 features split at every downsampling: returns maps at strides 2, 4, 8, 16, 32.

    ``dilated=True`` replaces the last stride by dilation (output stride 16), as DeepLab needs.
    """

    def __init__(self, variant: str = "large", pretrained: bool = True, dilated: bool = False):
        super().__init__()
        if variant == "large":
            net = tvm.mobilenet_v3_large(weights="IMAGENET1K_V1" if pretrained else None, dilated=dilated)
        else:
            net = tvm.mobilenet_v3_small(weights="IMAGENET1K_V1" if pretrained else None, dilated=dilated)
        features = net.features
        down = [i for i, b in enumerate(features) if getattr(b, "_is_cn", False)]
        bounds = [0] + down + [len(features)]
        self.stages = nn.ModuleList(features[a:b] for a, b in zip(bounds[:-1], bounds[1:]))
        with torch.no_grad():
            feats = self._forward(torch.zeros(1, 3, 64, 64))
        self.channels = [f.shape[1] for f in feats]

    def _forward(self, x):
        out = []
        for stage in self.stages:
            x = stage(x)
            out.append(x)
        return out

    def forward(self, x):
        return self._forward(x)


class ResNetEncoder(nn.Module):
    """ResNet-50 at strides 2, 4, 8, 16, 32 (or 16 for the last stage when dilated)."""

    def __init__(self, depth: int = 50, pretrained: bool = True, dilated: bool = False):
        super().__init__()
        builder = {50: tvm.resnet50, 101: tvm.resnet101}[depth]
        net = builder(weights="IMAGENET1K_V2" if pretrained else None,
                      replace_stride_with_dilation=[False, False, dilated])
        self.stages = nn.ModuleList([
            nn.Sequential(net.conv1, net.bn1, net.relu),
            nn.Sequential(net.maxpool, net.layer1),
            net.layer2, net.layer3, net.layer4,
        ])
        self.channels = [64, 256, 512, 1024, 2048]

    def forward(self, x):
        out = []
        for stage in self.stages:
            x = stage(x)
            out.append(x)
        return out


def build_encoder(backbone: str, pretrained: bool, dilated: bool) -> nn.Module:
    if backbone in ("mobilenet_v3_large", "mobilenet"):
        return MobileNetV3Encoder("large", pretrained, dilated)
    if backbone == "mobilenet_v3_small":
        return MobileNetV3Encoder("small", pretrained, dilated)
    if backbone in ("resnet50", "resnet101"):
        return ResNetEncoder(int(backbone[6:]), pretrained, dilated)
    raise ValueError(f"Unknown backbone '{backbone}'")
