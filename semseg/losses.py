"""Segmentation losses. All of them honour ``ignore_index`` (VOC borders and padding)."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class OhemCrossEntropy(nn.Module):
    """Online hard example mining: backprop only through pixels the model is unsure about."""

    def __init__(self, thresh: float = 0.7, min_kept: int = 100_000, ignore_index: int = 255):
        super().__init__()
        self.thresh = -torch.log(torch.tensor(thresh)).item()
        self.min_kept = min_kept
        self.ignore_index = ignore_index

    def forward(self, logits, target):
        loss = F.cross_entropy(logits, target, ignore_index=self.ignore_index, reduction="none").flatten()
        loss = loss[target.flatten() != self.ignore_index]
        if loss.numel() == 0:
            return logits.sum() * 0
        hard = loss[loss > self.thresh]
        if hard.numel() < self.min_kept:
            hard = loss.topk(min(self.min_kept, loss.numel())).values
        return hard.mean()


class DiceLoss(nn.Module):
    def __init__(self, ignore_index: int = 255, smooth: float = 1.0):
        super().__init__()
        self.ignore_index = ignore_index
        self.smooth = smooth

    def forward(self, logits, target):
        num_classes = logits.shape[1]
        valid = (target != self.ignore_index).unsqueeze(1)
        probs = logits.float().softmax(dim=1) * valid
        onehot = F.one_hot(target.clamp(0, num_classes - 1), num_classes).permute(0, 3, 1, 2) * valid
        dims = (0, 2, 3)
        inter = (probs * onehot).sum(dims)
        denom = probs.sum(dims) + onehot.sum(dims)
        return 1 - ((2 * inter + self.smooth) / (denom + self.smooth)).mean()


class CEDice(nn.Module):
    def __init__(self, dice_weight: float = 0.5, ignore_index: int = 255, label_smoothing: float = 0.0):
        super().__init__()
        self.ce = nn.CrossEntropyLoss(ignore_index=ignore_index, label_smoothing=label_smoothing)
        self.dice = DiceLoss(ignore_index)
        self.dice_weight = dice_weight

    def forward(self, logits, target):
        return self.ce(logits, target) + self.dice_weight * self.dice(logits, target)


def build_loss(cfg) -> nn.Module:
    if cfg.name == "ce":
        return nn.CrossEntropyLoss(ignore_index=cfg.ignore_index, label_smoothing=cfg.label_smoothing)
    if cfg.name == "ohem":
        return OhemCrossEntropy(cfg.ohem_thresh, cfg.ohem_min_kept, cfg.ignore_index)
    if cfg.name == "ce_dice":
        return CEDice(cfg.dice_weight, cfg.ignore_index, cfg.label_smoothing)
    raise ValueError(f"Unknown loss '{cfg.name}'")
