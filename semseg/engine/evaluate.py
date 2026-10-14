"""Validation loop shared by the trainer and ``semseg evaluate``."""
from __future__ import annotations

import contextlib
import time

import numpy as np
import torch

from ..metrics import SegmentationMetrics
from ..visualize import sample_grid


@torch.inference_mode()
def evaluate_model(model: torch.nn.Module, loader, num_classes: int, device: torch.device,
                   criterion: torch.nn.Module | None = None, ignore_index: int = 255,
                   amp_dtype: torch.dtype | None = None, memory_format=torch.channels_last,
                   num_samples: int = 0, palette: np.ndarray | None = None) -> tuple[dict, object]:
    model.eval()
    metrics = SegmentationMetrics(num_classes, ignore_index, device)
    autocast = torch.autocast(device.type, dtype=amp_dtype) if amp_dtype else contextlib.nullcontext()
    total_loss, batches, start = 0.0, 0, time.perf_counter()
    samples: list[tuple] = []
    for images, targets in loader:
        images = images.to(device, non_blocking=True).contiguous(memory_format=memory_format)
        targets = targets.to(device, non_blocking=True)
        with autocast:
            logits = model(images)
            if criterion is not None:
                total_loss += criterion(logits, targets).item()
        preds = logits.argmax(1)
        metrics.update(preds, targets)
        batches += 1
        need = num_samples - len(samples)
        if need > 0:
            samples += list(zip(images[:need].float().cpu(), targets[:need].cpu(), preds[:need].cpu()))
    result = metrics.compute()
    if criterion is not None:
        result["val_loss"] = total_loss / max(1, batches)
    result["val_time_s"] = round(time.perf_counter() - start, 2)
    result["confusion"] = metrics.confusion.tolist()
    grid = None
    if samples and palette is not None:
        imgs, tgts, prds = (torch.stack(x) for x in zip(*samples))
        grid = sample_grid(imgs, tgts, prds, palette, ignore_index)
    return result, grid
