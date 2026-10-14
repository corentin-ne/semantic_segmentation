"""DeconvMobileNet — the 2023 baseline, kept as a reference point.

Same design as the original notebook (MobileNetV3-Small encoder, FC layers recast as 1x1
"conv6/conv7" initialised from the ImageNet classifier, five x2 upsample + conv decoder
stages, no skip connections) with the dead code removed and an input-size agnostic forward.
"""
from __future__ import annotations

import torch.nn as nn
import torchvision.models as tvm

from .blocks import init_decoder, upsample_like


def _deconv_block(channels: list[int]) -> nn.Sequential:
    layers: list[nn.Module] = []
    for c_in, c_out in zip(channels[:-1], channels[1:]):
        layers += [nn.ConvTranspose2d(c_in, c_out, 3, padding=1), nn.BatchNorm2d(c_out), nn.ReLU(inplace=True)]
    return nn.Sequential(*layers)


class DeconvMobileNet(nn.Module):
    def __init__(self, num_classes: int = 21, pretrained: bool = True):
        super().__init__()
        net = tvm.mobilenet_v3_small(weights="IMAGENET1K_V1" if pretrained else None)
        self.encoder = net.features  # 576 channels, stride 32

        self.conv67 = nn.Sequential(
            nn.Conv2d(576, 1024, 1), nn.BatchNorm2d(1024), nn.ReLU(inplace=True),
            nn.Conv2d(1024, 1000, 1), nn.BatchNorm2d(1000), nn.ReLU(inplace=True),
        )
        self.deconv67 = nn.Sequential(nn.ConvTranspose2d(1000, 1024, 1), nn.BatchNorm2d(1024), nn.ReLU(inplace=True))
        self.deconv5 = _deconv_block([1024, 576, 576, 96])
        self.deconv4 = _deconv_block([96, 96, 96, 40])
        self.deconv3 = _deconv_block([40, 40, 40, 24])
        self.deconv2 = _deconv_block([24, 24, 16])
        self.deconv1 = nn.Sequential(_deconv_block([16, 16, 16]), nn.ConvTranspose2d(16, num_classes, 1))

        init_decoder(nn.ModuleList([self.conv67, self.deconv67, self.deconv5, self.deconv4,
                                    self.deconv3, self.deconv2, self.deconv1]))
        if pretrained:  # FCN trick: reuse the classifier's Linear layers as 1x1 convolutions
            fc6, fc7 = net.classifier[0], net.classifier[3]
            self.conv67[0].weight.data.copy_(fc6.weight.data[:, :, None, None])
            self.conv67[0].bias.data.copy_(fc6.bias.data)
            self.conv67[3].weight.data.copy_(fc7.weight.data[:, :, None, None])
            self.conv67[3].bias.data.copy_(fc7.bias.data)

    def forward(self, x):
        size = x.shape[-2:]
        f = self.encoder(x)
        x = self.deconv67(self.conv67(f))
        for block in (self.deconv5, self.deconv4, self.deconv3, self.deconv2):
            x = block(upsample_like(x, (x.shape[-2] * 2, x.shape[-1] * 2)))
        return upsample_like(self.deconv1(x), size)
