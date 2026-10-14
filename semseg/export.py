"""Export a trained model to ONNX (dynamic batch / resolution) for deployment."""
from __future__ import annotations

from pathlib import Path

import torch

from .inference import find_checkpoint, load_model


def export_onnx(checkpoint: str, output: str | None = None, size: int = 512, opset: int = 17) -> Path:
    model, _ = load_model(checkpoint, "cpu")
    model = model.to(memory_format=torch.contiguous_format)
    out = Path(output or find_checkpoint(checkpoint).with_suffix(".onnx"))
    dummy = torch.randn(1, 3, size, size)
    torch.onnx.export(
        model, (dummy,), str(out), opset_version=opset, input_names=["image"], output_names=["logits"],
        dynamic_axes={"image": {0: "batch", 2: "height", 3: "width"},
                      "logits": {0: "batch", 2: "height", 3: "width"}},
        dynamo=False,
    )
    print(f"Exported {out} ({out.stat().st_size / 2**20:.1f} MB)")
    return out
