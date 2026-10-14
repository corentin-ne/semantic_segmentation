"""Thin wrappers around torchvision segmentation models (returns logits, not a dict)."""
from __future__ import annotations

import torch.nn as nn
import torchvision.models.segmentation as tvs

_BUILDERS = {
    "lraspp_mobilenet": (tvs.lraspp_mobilenet_v3_large, "IMAGENET1K_V1"),
    "deeplabv3_mobilenet": (tvs.deeplabv3_mobilenet_v3_large, "IMAGENET1K_V1"),
    "deeplabv3_resnet50": (tvs.deeplabv3_resnet50, "IMAGENET1K_V2"),
}


class TorchvisionSegmenter(nn.Module):
    def __init__(self, name: str, num_classes: int = 21, pretrained: str = "imagenet"):
        super().__init__()
        builder, backbone_weights = _BUILDERS[name]
        if pretrained == "coco_voc":
            # COCO images restricted to the 20 VOC categories: the strongest starting point for VOC.
            model = builder(weights="COCO_WITH_VOC_LABELS_V1")
            if num_classes != 21:
                _replace_head(model, num_classes)
        else:
            weights_backbone = backbone_weights if pretrained == "imagenet" else None
            model = builder(weights=None, weights_backbone=weights_backbone, num_classes=num_classes)
        if getattr(model, "aux_classifier", None) is not None:
            model.aux_classifier = None
        self.model = model

    def forward(self, x):
        return self.model(x)["out"]


def _replace_head(model: nn.Module, num_classes: int) -> None:
    head = model.classifier
    if isinstance(head, tvs.lraspp.LRASPPHead):
        head.low_classifier = nn.Conv2d(head.low_classifier.in_channels, num_classes, 1)
        head.high_classifier = nn.Conv2d(head.high_classifier.in_channels, num_classes, 1)
    else:
        last = head[-1]
        head[-1] = nn.Conv2d(last.in_channels, num_classes, 1)

