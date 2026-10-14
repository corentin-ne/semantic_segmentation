"""Experiment configuration: typed dataclasses loaded from YAML, overridable from the CLI."""
from __future__ import annotations

import ast
import dataclasses
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class DataConfig:
    dataset: str = "voc"              # voc | synthetic
    root: str = "./data"
    year: str = "2012"
    train_split: str = "train"        # train | trainaug (VOC train + SBD, 10 582 images)
    val_split: str = "val"
    download: bool = True
    crop_size: int = 512
    scale_range: tuple[float, float] = (0.5, 2.0)
    hflip: float = 0.5
    color_jitter: float = 0.2
    eval_size: int = 512              # validation images are padded (never resized) to this size
    num_workers: int = 4
    # synthetic dataset only
    synthetic_train: int = 2000
    synthetic_val: int = 300
    synthetic_size: int = 160


@dataclass
class ModelConfig:
    name: str = "deeplabv3plus_mobilenet"
    pretrained: str = "imagenet"      # none | imagenet | coco_voc (torchvision models only)
    num_classes: int = 21
    kwargs: dict[str, Any] = field(default_factory=dict)


@dataclass
class OptimConfig:
    optimizer: str = "adamw"          # adamw | sgd
    lr: float = 6e-4
    backbone_lr_mult: float = 0.1     # pretrained encoder learns slower than the fresh decoder
    weight_decay: float = 1e-4
    momentum: float = 0.9
    scheduler: str = "poly"           # poly | cosine | constant
    poly_power: float = 0.9
    warmup_epochs: float = 1.0
    grad_clip: float = 5.0


@dataclass
class LossConfig:
    name: str = "ce"                  # ce | ohem | ce_dice
    label_smoothing: float = 0.0
    ohem_thresh: float = 0.7
    ohem_min_kept: int = 100_000
    dice_weight: float = 0.5
    ignore_index: int = 255


@dataclass
class TrainConfig:
    epochs: int = 60
    batch_size: int = 16
    eval_batch_size: int = 8
    accumulate_steps: int = 1
    amp: bool = True                  # mixed precision on CUDA (bf16 when supported, else fp16)
    channels_last: bool = True
    compile: bool = False             # torch.compile (PyTorch >= 2)
    ema: bool = True
    ema_decay: float = 0.9995
    eval_every: int = 1
    patience: int = 15                # early stopping on val mIoU, 0 disables it
    seed: int = 42
    device: str = "auto"
    log_every: int = 20
    num_samples: int = 6              # validation images rendered each evaluation
    resume: str = ""


@dataclass
class LoggingConfig:
    runs_dir: str = "runs"
    run_name: str = ""
    wandb: bool = False
    wandb_project: str = "semantic-segmentation"


@dataclass
class Config:
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    optim: OptimConfig = field(default_factory=OptimConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "Config":
        cfg = cls()
        for section, values in (raw or {}).items():
            if not hasattr(cfg, section):
                raise KeyError(f"Unknown config section '{section}'")
            target = getattr(cfg, section)
            for key, value in (values or {}).items():
                _set_field(target, section, key, value)
        return cfg


def _set_field(target: Any, section: str, key: str, value: Any) -> None:
    if not hasattr(target, key):
        raise KeyError(f"Unknown config key '{section}.{key}'")
    current = getattr(target, key)
    if isinstance(current, tuple) and isinstance(value, (list, tuple)):
        value = tuple(type(c)(v) for c, v in zip(current, value))
    elif isinstance(current, bool) and isinstance(value, str):
        value = value.lower() in {"1", "true", "yes", "on"}
    elif isinstance(current, (int, float)) and not isinstance(current, bool) and isinstance(value, str):
        value = type(current)(float(value)) if isinstance(current, int) else float(value)
    setattr(target, key, value)


def _parse_value(text: str) -> Any:
    try:
        return ast.literal_eval(text)
    except (ValueError, SyntaxError):
        return text


def load_config(path: str | Path | None = None, overrides: list[str] | None = None) -> Config:
    """Load a YAML config then apply ``section.key=value`` overrides."""
    raw: dict[str, Any] = {}
    if path:
        with open(path, encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}
    cfg = Config.from_dict(raw)
    for item in overrides or []:
        if "=" not in item or "." not in item.split("=", 1)[0]:
            raise ValueError(f"Override must look like 'section.key=value', got '{item}'")
        dotted, value = item.split("=", 1)
        section, key = dotted.split(".", 1)
        if not hasattr(cfg, section):
            raise KeyError(f"Unknown config section '{section}'")
        target = getattr(cfg, section)
        if key.startswith("kwargs."):
            target.kwargs[key.split(".", 1)[1]] = _parse_value(value)
        else:
            _set_field(target, section, key, _parse_value(value))
    return cfg
