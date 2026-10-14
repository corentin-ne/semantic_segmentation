"""Optimizer with differential learning rates and no weight decay on norms/biases; LR schedules."""
from __future__ import annotations

import math

import torch
import torch.nn as nn

from ..models import backbone_parameters


def _split_decay(params: list[nn.Parameter]) -> tuple[list[nn.Parameter], list[nn.Parameter]]:
    decay = [p for p in params if p.requires_grad and p.ndim > 1]
    no_decay = [p for p in params if p.requires_grad and p.ndim <= 1]
    return decay, no_decay


def build_optimizer(cfg, model: nn.Module) -> torch.optim.Optimizer:
    enc, head = backbone_parameters(model)
    groups = []
    for params, mult in ((enc, cfg.backbone_lr_mult), (head, 1.0)):
        decay, no_decay = _split_decay(params)
        if decay:
            groups.append({"params": decay, "weight_decay": cfg.weight_decay, "lr_mult": mult})
        if no_decay:
            groups.append({"params": no_decay, "weight_decay": 0.0, "lr_mult": mult})
    for g in groups:
        g["lr"] = cfg.lr * g["lr_mult"]
    fused = torch.cuda.is_available() and all(p.is_cuda for g in groups for p in g["params"])
    if cfg.optimizer == "adamw":
        return torch.optim.AdamW(groups, lr=cfg.lr, fused=fused or None)
    if cfg.optimizer == "sgd":
        return torch.optim.SGD(groups, lr=cfg.lr, momentum=cfg.momentum, nesterov=True)
    raise ValueError(f"Unknown optimizer '{cfg.optimizer}'")


def build_scheduler(cfg, optimizer, total_steps: int, warmup_steps: int) -> torch.optim.lr_scheduler.LambdaLR:
    """Per-iteration schedule: linear warmup, then poly / cosine decay."""

    def factor(step: int) -> float:
        if step < warmup_steps:
            return (step + 1) / max(1, warmup_steps)
        progress = min(1.0, (step - warmup_steps) / max(1, total_steps - warmup_steps))
        if cfg.scheduler == "poly":
            return (1 - progress) ** cfg.poly_power
        if cfg.scheduler == "cosine":
            return 0.5 * (1 + math.cos(math.pi * progress))
        return 1.0

    return torch.optim.lr_scheduler.LambdaLR(optimizer, factor)
