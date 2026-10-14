"""Confusion-matrix metrics, accumulated on device with a single ``bincount`` per batch."""
from __future__ import annotations

import torch


class SegmentationMetrics:
    def __init__(self, num_classes: int, ignore_index: int = 255, device: torch.device | str = "cpu"):
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.confusion = torch.zeros(num_classes, num_classes, dtype=torch.int64, device=device)

    def reset(self) -> None:
        self.confusion.zero_()

    @torch.no_grad()
    def update(self, pred: torch.Tensor, target: torch.Tensor) -> None:
        """``pred``: logits (N, C, H, W) or labels (N, H, W); ``target``: labels (N, H, W)."""
        if pred.ndim == 4:
            pred = pred.argmax(dim=1)
        valid = target != self.ignore_index
        idx = target[valid] * self.num_classes + pred[valid]
        self.confusion += torch.bincount(idx, minlength=self.num_classes ** 2).view(self.num_classes, self.num_classes)

    def compute(self) -> dict:
        cm = self.confusion.double()
        tp = cm.diag()
        gt, predicted = cm.sum(1), cm.sum(0)
        union = gt + predicted - tp
        present = gt > 0  # classes absent from the ground truth are excluded from the means
        iou = torch.where(union > 0, tp / union.clamp(min=1), torch.full_like(tp, float("nan")))
        acc = torch.where(gt > 0, tp / gt.clamp(min=1), torch.full_like(tp, float("nan")))
        dice = torch.where(gt + predicted > 0, 2 * tp / (gt + predicted).clamp(min=1), torch.full_like(tp, float("nan")))
        freq = gt / gt.sum().clamp(min=1)
        return {
            "mIoU": float(iou[present].mean()) if present.any() else 0.0,
            "pixel_acc": float(tp.sum() / cm.sum().clamp(min=1)),
            "mean_acc": float(acc[present].mean()) if present.any() else 0.0,
            "fwIoU": float((freq[present] * iou[present]).sum()),
            "mDice": float(dice[present].mean()) if present.any() else 0.0,
            "per_class_iou": [None if v != v else round(float(v), 5) for v in iou.tolist()],
        }
