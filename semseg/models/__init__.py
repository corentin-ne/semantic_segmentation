"""Model registry: ``build_model(name, num_classes, pretrained)``."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch.nn as nn

from .deconv_mobilenet import DeconvMobileNet
from .deeplabv3plus import DeepLabV3Plus
from .mobile_unet import MobileUNet
from .torchvision_models import TorchvisionSegmenter


@dataclass(frozen=True)
class ModelSpec:
    builder: Callable[..., nn.Module]
    family: str
    description: str


MODELS: dict[str, ModelSpec] = {
    "deconv_mobilenet": ModelSpec(
        lambda nc, pt, **kw: DeconvMobileNet(nc, pt == "imagenet", **kw),
        "legacy", "2023 baseline: MobileNetV3-Small + deconvolution decoder, no skips"),
    "mobile_unet": ModelSpec(
        lambda nc, pt, **kw: MobileUNet(nc, pt == "imagenet", **kw),
        "unet", "MobileNetV3-Large U-Net, separable decoder, global context gate"),
    "deeplabv3plus_mobilenet": ModelSpec(
        lambda nc, pt, **kw: DeepLabV3Plus(nc, pt == "imagenet", backbone="mobilenet_v3_large", **kw),
        "deeplab", "DeepLabV3+ (OS16), MobileNetV3-Large, separable ASPP — best speed/accuracy"),
    "deeplabv3plus_resnet50": ModelSpec(
        lambda nc, pt, **kw: DeepLabV3Plus(nc, pt == "imagenet", backbone="resnet50", **kw),
        "deeplab", "DeepLabV3+ (OS16), ResNet-50 — accuracy reference"),
    "lraspp_mobilenet": ModelSpec(
        lambda nc, pt, **kw: TorchvisionSegmenter("lraspp_mobilenet", nc, pt),
        "torchvision", "LR-ASPP MobileNetV3-Large — fastest, edge/real-time"),
    "deeplabv3_mobilenet": ModelSpec(
        lambda nc, pt, **kw: TorchvisionSegmenter("deeplabv3_mobilenet", nc, pt),
        "torchvision", "torchvision DeepLabV3 MobileNetV3-Large"),
    "deeplabv3_resnet50": ModelSpec(
        lambda nc, pt, **kw: TorchvisionSegmenter("deeplabv3_resnet50", nc, pt),
        "torchvision", "torchvision DeepLabV3 ResNet-50"),
}


def build_model(name: str, num_classes: int = 21, pretrained: str = "imagenet", bn_momentum: float = 0.1,
                **kwargs) -> nn.Module:
    if name not in MODELS:
        raise KeyError(f"Unknown model '{name}'. Available: {', '.join(MODELS)}")
    if pretrained == "coco_voc" and MODELS[name].family != "torchvision":
        raise ValueError("pretrained='coco_voc' is only available for torchvision models")
    model = MODELS[name].builder(num_classes, pretrained, **kwargs)
    # torchvision's MobileNetV3 ships BatchNorm with momentum 0.01, tuned for long ImageNet runs:
    # running statistics then lag far behind during a VOC fine-tune and eval-mode predictions collapse.
    for m in model.modules():
        if isinstance(m, nn.modules.batchnorm._BatchNorm):
            m.momentum = bn_momentum
    return model


def backbone_parameters(model: nn.Module) -> tuple[list[nn.Parameter], list[nn.Parameter]]:
    """Split parameters into (pretrained encoder, freshly initialised head)."""
    inner = getattr(model, "model", model)
    encoder = getattr(inner, "encoder", None) or getattr(inner, "backbone", None)
    enc_ids = {id(p) for p in encoder.parameters()} if encoder is not None else set()
    enc = [p for p in model.parameters() if id(p) in enc_ids]
    head = [p for p in model.parameters() if id(p) not in enc_ids]
    return enc, head


__all__ = ["MODELS", "build_model", "backbone_parameters", "DeconvMobileNet", "DeepLabV3Plus", "MobileUNet"]
