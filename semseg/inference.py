"""Load a trained run and segment arbitrary images, with optional flip / multi-scale TTA."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from .data import dataset_meta
from .data.transforms import normalize, to_uint8_tensors
from .engine.trainer import select_device
from .models import build_model


def find_checkpoint(path: str | Path) -> Path:
    path = Path(path)
    if path.is_dir():
        for candidate in (path / "checkpoints" / "best.pt", path / "best.pt", path / "checkpoints" / "last.pt"):
            if candidate.exists():
                return candidate
        raise FileNotFoundError(f"No checkpoint found in {path}")
    return path


def load_model(checkpoint: str | Path, device: str | torch.device = "auto") -> tuple[torch.nn.Module, dict]:
    device = select_device(device) if isinstance(device, str) else device
    state = torch.load(find_checkpoint(checkpoint), map_location="cpu", weights_only=False)
    model = build_model(state["model_name"], state["num_classes"], "none", **state.get("model_kwargs", {}))
    model.load_state_dict(state["weights"])
    model.to(device, memory_format=torch.channels_last).eval()
    return model, state


class Predictor:
    def __init__(self, checkpoint: str | Path, device: str = "auto", max_size: int = 512):
        self.model, self.meta = load_model(checkpoint, device)
        self.device = next(self.model.parameters()).device
        self.class_names, self.palette = dataset_meta(self.meta["dataset"])
        self.max_size = max_size

    @torch.inference_mode()
    def logits(self, image: torch.Tensor, scales=(1.0,), flip: bool = False) -> torch.Tensor:
        """``image``: normalised (1, 3, H, W). Returns averaged softmax probabilities (1, C, H, W)."""
        h, w = image.shape[-2:]
        probs = torch.zeros(1, self.meta["num_classes"], h, w, device=self.device)
        for s in scales:
            x = image if s == 1.0 else F.interpolate(image, scale_factor=s, mode="bilinear", align_corners=False)
            x = x.contiguous(memory_format=torch.channels_last)
            views = [x, x.flip(-1)] if flip else [x]
            for k, v in enumerate(views):
                out = self.model(v)
                if k == 1:
                    out = out.flip(-1)
                probs += F.interpolate(out, size=(h, w), mode="bilinear", align_corners=False).softmax(1)
        return probs / (len(scales) * (2 if flip else 1))

    def predict(self, image: Image.Image | np.ndarray, tta: bool = False) -> tuple[np.ndarray, np.ndarray]:
        """Returns (label map HxW at the input resolution, per-pixel confidence HxW)."""
        img, _ = to_uint8_tensors(image, None)
        h, w = img.shape[-2:]
        ratio = min(1.0, self.max_size / max(h, w))
        x = normalize(img)[None].to(self.device)
        if ratio < 1.0:
            x = F.interpolate(x, scale_factor=ratio, mode="bilinear", align_corners=False)
        scales, flip = ((0.75, 1.0, 1.25), True) if tta else ((1.0,), False)
        probs = self.logits(x, scales, flip)
        if ratio < 1.0:
            probs = F.interpolate(probs, size=(h, w), mode="bilinear", align_corners=False)
        conf, labels = probs[0].max(0)
        return labels.byte().cpu().numpy(), conf.cpu().numpy()


def predict_file(checkpoint: str, image_path: str, output: str | None = None, tta: bool = False) -> Path:
    from .visualize import colorize, overlay

    predictor = Predictor(checkpoint)
    image = Image.open(image_path).convert("RGB")
    labels, _ = predictor.predict(image, tta=tta)
    color = colorize(labels, predictor.palette)
    panel = np.concatenate([np.asarray(image), overlay(np.asarray(image), color), color], axis=1)
    out = Path(output or Path(image_path).with_suffix(".seg.png"))
    Image.fromarray(panel).save(out)
    present = {predictor.class_names[c]: round(float((labels == c).mean()) * 100, 1) for c in np.unique(labels)}
    print(f"Saved {out}  |  classes (% pixels): {present}")
    return out
