"""DeepLabV3+ (Chen et al., 2018) with separable ASPP, for MobileNetV3 or ResNet encoders."""
from __future__ import annotations

import torch
import torch.nn as nn

from .blocks import ConvBNAct, SeparableConv, build_encoder, init_decoder, upsample_like


class ASPP(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, rates=(6, 12, 18), dropout: float = 0.1):
        super().__init__()
        self.branches = nn.ModuleList(
            [ConvBNAct(in_ch, out_ch, 1)] + [SeparableConv(in_ch, out_ch, dilation=r) for r in rates]
        )
        # No BatchNorm after global pooling: a 1x1 map would break BN with batch size 1.
        self.pool = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(in_ch, out_ch, 1), nn.ReLU(inplace=True))
        self.project = nn.Sequential(
            ConvBNAct(out_ch * (len(rates) + 2), out_ch, 1), nn.Dropout2d(dropout),
        )

    def forward(self, x):
        outs = [b(x) for b in self.branches]
        outs.append(self.pool(x).expand(-1, -1, *x.shape[-2:]))
        return self.project(torch.cat(outs, dim=1))


class DeepLabV3Plus(nn.Module):
    def __init__(self, num_classes: int = 21, pretrained: bool = True, backbone: str = "mobilenet_v3_large",
                 aspp_channels: int | None = None, low_level_channels: int = 48, rates=(6, 12, 18),
                 dropout: float = 0.1):
        super().__init__()
        self.encoder = build_encoder(backbone, pretrained, dilated=True)  # output stride 16
        ch = self.encoder.channels
        light = backbone.startswith("mobilenet")
        aspp_channels = aspp_channels or (128 if light else 256)
        self.aspp = ASPP(ch[4], aspp_channels, rates, dropout)
        self.low_level = ConvBNAct(ch[1], low_level_channels, 1)
        self.decoder = nn.Sequential(
            SeparableConv(aspp_channels + low_level_channels, aspp_channels),
            SeparableConv(aspp_channels, aspp_channels),
        )
        self.classifier = nn.Conv2d(aspp_channels, num_classes, 1)
        init_decoder(nn.ModuleList([self.aspp, self.low_level, self.decoder, self.classifier]))

    def forward(self, x):
        size = x.shape[-2:]
        feats = self.encoder(x)
        low = self.low_level(feats[1])  # stride 4
        y = upsample_like(self.aspp(feats[4]), low)
        y = self.decoder(torch.cat([y, low], dim=1))
        return upsample_like(self.classifier(y), size)
