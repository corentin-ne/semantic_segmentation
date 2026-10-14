"""Mask colouring and side-by-side panels (PIL only, no matplotlib needed)."""
from __future__ import annotations

import numpy as np
import torch
from PIL import Image, ImageDraw

from .data.transforms import IMAGENET_MEAN, IMAGENET_STD


def colorize(mask: np.ndarray | torch.Tensor, palette: np.ndarray, ignore_index: int = 255) -> np.ndarray:
    mask = mask.cpu().numpy() if isinstance(mask, torch.Tensor) else np.asarray(mask)
    out = palette[np.clip(mask, 0, len(palette) - 1)]
    out[mask == ignore_index] = (255, 255, 255)
    return out.astype(np.uint8)


def denormalize(image: torch.Tensor) -> np.ndarray:
    mean = torch.tensor(IMAGENET_MEAN, device=image.device).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=image.device).view(3, 1, 1)
    return ((image * std + mean).clamp(0, 1) * 255).byte().permute(1, 2, 0).cpu().numpy()


def overlay(image: np.ndarray, color_mask: np.ndarray, alpha: float = 0.55) -> np.ndarray:
    return (image * (1 - alpha) + color_mask * alpha).astype(np.uint8)


def sample_grid(images: torch.Tensor, targets: torch.Tensor, preds: torch.Tensor, palette: np.ndarray,
                ignore_index: int = 255, tile: int = 160) -> Image.Image:
    """Rows of [image | ground truth | prediction], cropped to the valid (unpadded) area."""
    rows = []
    for img, tgt, prd in zip(images, targets, preds):
        valid = (tgt != ignore_index).nonzero()
        if len(valid):
            h, w = int(valid[:, 0].max()) + 1, int(valid[:, 1].max()) + 1
            img, tgt, prd = img[:, :h, :w], tgt[:h, :w], prd[:h, :w]
        panels = [denormalize(img), colorize(tgt, palette, ignore_index), colorize(prd, palette, ignore_index)]
        resized = [Image.fromarray(p).resize((tile, round(tile * p.shape[0] / p.shape[1])), Image.NEAREST) for p in panels]
        rows.append(resized)
    pad, header = 4, 18
    width = 3 * tile + 4 * pad
    height = header + sum(r[0].height + pad for r in rows) + pad
    grid = Image.new("RGB", (width, height), (24, 24, 27))
    draw = ImageDraw.Draw(grid)
    for col, title in enumerate(("image", "ground truth", "prediction")):
        draw.text((pad + col * (tile + pad) + 4, 3), title, fill=(220, 220, 220))
    y = header
    for row in rows:
        for col, panel in enumerate(row):
            grid.paste(panel, (pad + col * (tile + pad), y))
        y += row[0].height + pad
    return grid
