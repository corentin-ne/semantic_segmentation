"""MobileUNet — successor of the 2023 DeconvMobileUNet.

Keeps the idea (MobileNet encoder + U-Net skip connections) but fixes what limited it:
skips at every stride instead of shape-dependent concatenations, bilinear upsampling instead
of nearest (no blocky edges), depthwise-separable decoder convs, an LR-ASPP style global
context gate on the bottleneck, and any input size.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .blocks import ConvBNAct, SeparableConv, build_encoder, init_decoder, upsample_like


class GlobalContext(nn.Module):
    """Squeeze-excitation over the whole image: cheap global context for the bottleneck."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.proj = ConvBNAct(in_ch, out_ch, 1)
        self.gate = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(in_ch, out_ch, 1), nn.Sigmoid())

    def forward(self, x):
        return self.proj(x) * self.gate(x)


class DecoderBlock(nn.Module):
    def __init__(self, in_ch: int, skip_ch: int, out_ch: int):
        super().__init__()
        self.fuse = nn.Sequential(SeparableConv(in_ch + skip_ch, out_ch), SeparableConv(out_ch, out_ch))

    def forward(self, x, skip):
        return self.fuse(torch.cat([upsample_like(x, skip), skip], dim=1))


class MobileUNet(nn.Module):
    def __init__(self, num_classes: int = 21, pretrained: bool = True, backbone: str = "mobilenet_v3_large",
                 decoder_channels: tuple[int, ...] = (128, 96, 64, 32), dropout: float = 0.1):
        super().__init__()
        self.encoder = build_encoder(backbone, pretrained, dilated=False)
        ch = self.encoder.channels  # strides 2, 4, 8, 16, 32
        self.bottleneck = GlobalContext(ch[4], decoder_channels[0])
        blocks, in_ch = [], decoder_channels[0]
        for skip_ch, out_ch in zip(reversed(ch[:4]), decoder_channels):
            blocks.append(DecoderBlock(in_ch, skip_ch, out_ch))
            in_ch = out_ch
        self.decoder = nn.ModuleList(blocks)
        self.head = nn.Sequential(nn.Dropout2d(dropout), nn.Conv2d(in_ch, num_classes, 1))
        init_decoder(nn.ModuleList([self.bottleneck, self.decoder, self.head]))

    def forward(self, x):
        size = x.shape[-2:]
        feats = self.encoder(x)
        y = self.bottleneck(feats[4])
        for block, skip in zip(self.decoder, reversed(feats[:4])):
            y = block(y, skip)
        return upsample_like(self.head(y), size)
