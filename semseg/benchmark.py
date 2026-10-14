"""Model cost: parameters, FLOPs (torch.utils.flop_counter) and measured latency."""
from __future__ import annotations

import time

import torch
import torch.nn as nn
from torch.utils.flop_counter import FlopCounterMode


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


@torch.no_grad()
def count_gflops(model: nn.Module, size: int = 512) -> float:
    """Multiply-accumulates x2, for one image."""
    model.eval()
    x = torch.zeros(1, 3, size, size, device=next(model.parameters()).device)
    with FlopCounterMode(display=False) as counter:
        model(x)
    return counter.get_total_flops() / 1e9


@torch.no_grad()
def measure_latency(model: nn.Module, size: int = 512, batch: int = 1, device: str | torch.device = "cpu",
                    warmup: int = 5, iters: int = 20, channels_last: bool = True, half: bool = False) -> dict:
    device = torch.device(device)
    model = model.eval().to(device)
    x = torch.randn(batch, 3, size, size, device=device)
    if channels_last:
        model = model.to(memory_format=torch.channels_last)
        x = x.to(memory_format=torch.channels_last)
    dtype = torch.float16 if half and device.type == "cuda" else None
    sync = torch.cuda.synchronize if device.type == "cuda" else (lambda: None)
    with torch.autocast(device.type, dtype=dtype, enabled=dtype is not None):
        for _ in range(warmup):
            model(x)
        sync()
        times = []
        for _ in range(iters):
            t0 = time.perf_counter()
            model(x)
            sync()
            times.append(time.perf_counter() - t0)
    times.sort()
    median = times[len(times) // 2]
    return {"latency_ms": median * 1e3, "p90_ms": times[int(len(times) * 0.9) - 1] * 1e3,
            "fps": batch / median, "device": str(device), "batch": batch, "size": size}


def profile_model(model: nn.Module, size: int = 512, device: str = "cpu", latency: bool = True, **kwargs) -> dict:
    stats = {"params_m": count_parameters(model) / 1e6, "gflops": count_gflops(model.to(device), size)}
    if latency:
        stats.update(measure_latency(model, size, device=device, **kwargs))
    return stats
