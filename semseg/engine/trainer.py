"""Training loop: AMP, channels-last, optional torch.compile, EMA, grad accumulation,
per-iteration LR schedule, early stopping on validation mIoU, resumable checkpoints."""
from __future__ import annotations

import contextlib
import math
import random
import time
from pathlib import Path

import numpy as np
import torch

from ..benchmark import count_gflops, count_parameters
from ..config import Config
from ..data import build_dataloaders, dataset_meta
from ..losses import build_loss
from ..models import build_model
from .ema import ModelEMA
from .evaluate import evaluate_model
from .optim import build_optimizer, build_scheduler
from .run import RunLogger


def select_device(name: str = "auto") -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class Trainer:
    def __init__(self, cfg: Config, loaders=None):
        self.cfg = cfg
        t = cfg.train
        seed_everything(t.seed)
        self.device = select_device(t.device)
        if self.device.type == "cuda":
            torch.backends.cudnn.benchmark = True
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

        self.class_names, self.palette = dataset_meta(cfg.data.dataset)
        cfg.model.num_classes = len(self.class_names)
        self.num_classes = cfg.model.num_classes
        self.ignore_index = cfg.loss.ignore_index

        self.train_loader, self.val_loader = loaders or build_dataloaders(cfg)
        self.model = build_model(cfg.model.name, self.num_classes, cfg.model.pretrained, **cfg.model.kwargs)
        self.memory_format = torch.channels_last if t.channels_last else torch.contiguous_format
        self.model.to(self.device, memory_format=self.memory_format)
        self.ema = ModelEMA(self.model, t.ema_decay) if t.ema else None

        self.steps_per_epoch = max(1, len(self.train_loader) // t.accumulate_steps)
        self.optimizer = build_optimizer(cfg.optim, self.model)
        self.scheduler = build_scheduler(cfg.optim, self.optimizer, t.epochs * self.steps_per_epoch,
                                         int(cfg.optim.warmup_epochs * self.steps_per_epoch))
        self.criterion = build_loss(cfg.loss)

        use_amp = t.amp and self.device.type == "cuda"
        self.amp_dtype = (torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16) if use_amp else None
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.amp_dtype == torch.float16)
        self.forward_model = torch.compile(self.model) if t.compile else self.model

        self.start_epoch, self.global_step = 0, 0
        self.best_miou, self.best_epoch, self.bad_epochs = -1.0, 0, 0
        resume_dir = None
        if t.resume:
            resume_dir = self._load_checkpoint(t.resume)

        run_name = cfg.logging.run_name or f"{cfg.model.name}-{cfg.data.dataset}"
        self.run = RunLogger(cfg.logging.runs_dir, run_name, cfg.to_dict(), cfg.logging.wandb,
                             cfg.logging.wandb_project, existing_dir=resume_dir)
        self.model.eval()
        gflops = count_gflops(self.model, cfg.data.eval_size)
        self.run.update_summary(
            model=cfg.model.name, dataset=cfg.data.dataset, class_names=list(self.class_names),
            params_m=round(count_parameters(self.model) / 1e6, 3), gflops=round(gflops, 2),
            gflops_size=cfg.data.eval_size, device=str(self.device), epochs=t.epochs,
            amp=str(self.amp_dtype).replace("torch.", "") if self.amp_dtype else "fp32",
            train_images=len(self.train_loader.dataset), val_images=len(self.val_loader.dataset),
        )

    # ------------------------------------------------------------------ helpers
    def _autocast(self):
        if self.amp_dtype is None:
            return contextlib.nullcontext()
        return torch.autocast(self.device.type, dtype=self.amp_dtype)

    def _to_device(self, images, targets):
        images = images.to(self.device, non_blocking=True).contiguous(memory_format=self.memory_format)
        return images, targets.to(self.device, non_blocking=True)

    @property
    def eval_model(self) -> torch.nn.Module:
        return self.ema.module if self.ema else self.model

    # ------------------------------------------------------------------ train / eval
    def train_one_epoch(self, epoch: int) -> dict:
        t = self.cfg.train
        self.model.train()
        start = time.perf_counter()
        running = torch.zeros((), device=self.device)
        seen, batches = 0, 0
        self.optimizer.zero_grad(set_to_none=True)
        for i, (images, targets) in enumerate(self.train_loader):
            images, targets = self._to_device(images, targets)
            with self._autocast():
                loss = self.criterion(self.forward_model(images), targets)
            self.scaler.scale(loss / t.accumulate_steps).backward()
            running += loss.detach()
            seen += images.shape[0]
            batches += 1
            if (i + 1) % t.accumulate_steps:
                continue
            if self.cfg.optim.grad_clip > 0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.optim.grad_clip)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad(set_to_none=True)
            self.scheduler.step()
            if self.ema:
                self.ema.update(self.model)
            self.global_step += 1
            if self.global_step % t.log_every == 0:
                value = loss.item()
                if not math.isfinite(value):
                    raise FloatingPointError(f"Loss became {value} at step {self.global_step}")
                self.run.log_step({"step": self.global_step, "epoch": epoch + (i + 1) / len(self.train_loader),
                                   "loss": round(value, 5), "lr": self.optimizer.param_groups[-1]["lr"]})
        elapsed = time.perf_counter() - start
        return {"train_loss": (running / max(1, batches)).item(), "train_time_s": round(elapsed, 2),
                "train_img_per_s": round(seen / elapsed, 2)}

    def evaluate(self, model: torch.nn.Module | None = None) -> tuple[dict, object]:
        return evaluate_model(model or self.eval_model, self.val_loader, self.num_classes, self.device,
                              self.criterion, self.ignore_index, self.amp_dtype, self.memory_format,
                              self.cfg.train.num_samples, self.palette)

    # ------------------------------------------------------------------ checkpoints
    def _save_checkpoint(self, name: str, epoch: int, full: bool) -> Path:
        path = self.run.dir / "checkpoints" / f"{name}.pt"
        state = {
            "weights": self.eval_model.state_dict(),
            "model_name": self.cfg.model.name,
            "num_classes": self.num_classes,
            "model_kwargs": self.cfg.model.kwargs,
            "dataset": self.cfg.data.dataset,
            "class_names": list(self.class_names),
            "epoch": epoch,
            "mIoU": self.best_miou,
        }
        if full:
            state.update(model=self.model.state_dict(), optimizer=self.optimizer.state_dict(),
                         scheduler=self.scheduler.state_dict(), scaler=self.scaler.state_dict(),
                         ema=self.ema.state_dict() if self.ema else None, global_step=self.global_step,
                         best_epoch=self.best_epoch, bad_epochs=self.bad_epochs, config=self.cfg.to_dict())
        tmp = path.with_suffix(".tmp")
        torch.save(state, tmp)
        tmp.replace(path)
        return path

    def _load_checkpoint(self, path: str) -> Path:
        state = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(state["model"])
        self.optimizer.load_state_dict(state["optimizer"])
        self.scheduler.load_state_dict(state["scheduler"])
        self.scaler.load_state_dict(state["scaler"])
        if self.ema and state.get("ema"):
            self.ema.load_state_dict(state["ema"])
        self.start_epoch = state["epoch"]
        self.global_step = state["global_step"]
        self.best_miou, self.best_epoch = state["mIoU"], state["best_epoch"]
        self.bad_epochs = state["bad_epochs"]
        print(f"Resumed from {path} (epoch {self.start_epoch}, best mIoU {self.best_miou:.4f})")
        return Path(path).parent.parent

    # ------------------------------------------------------------------ main loop
    def fit(self) -> dict:
        t = self.cfg.train
        status = "running"
        self.run.update_summary(status=status)
        print(f"Run: {self.run.dir}  |  device: {self.device}  |  "
              f"{self.run.summary['params_m']}M params, {self.run.summary['gflops']} GFLOPs")
        try:
            for epoch in range(self.start_epoch, t.epochs):
                record = {"epoch": epoch + 1, **self.train_one_epoch(epoch),
                          "lr": self.optimizer.param_groups[-1]["lr"]}
                last = epoch + 1 == t.epochs
                if (epoch + 1) % t.eval_every and not last:
                    continue
                result, grid = self.evaluate()
                confusion = result.pop("confusion")
                record.update(result)
                if self.device.type == "cuda":
                    record["peak_mem_mb"] = round(torch.cuda.max_memory_allocated() / 2**20, 1)
                self.run.log_epoch(record)
                if grid is not None:
                    self.run.save_sample(epoch + 1, grid)

                if result["mIoU"] > self.best_miou:
                    self.best_miou, self.best_epoch, self.bad_epochs = result["mIoU"], epoch + 1, 0
                    self._save_checkpoint("best", epoch + 1, full=False)
                    self.run.save_confusion(confusion)
                    self.run.update_summary(best={k: v for k, v in record.items()})
                else:
                    self.bad_epochs += t.eval_every
                self._save_checkpoint("last", epoch + 1, full=True)
                self.run.update_summary(epoch=epoch + 1, best_mIoU=self.best_miou, best_epoch=self.best_epoch)
                print(f"epoch {epoch + 1:3d}/{t.epochs}  loss {record['train_loss']:.4f}  "
                      f"val {result['val_loss']:.4f}  mIoU {result['mIoU']:.4f}  "
                      f"acc {result['pixel_acc']:.4f}  ({record['train_img_per_s']} img/s)")
                if t.patience and self.bad_epochs >= t.patience:
                    print(f"Early stopping: no mIoU gain for {self.bad_epochs} epochs")
                    status = "early_stopped"
                    break
            else:
                status = "finished"
        except KeyboardInterrupt:
            status = "interrupted"
        except Exception:
            self.run.update_summary(status="failed")
            self.run.close()
            raise
        self.run.update_summary(status=status, finished_at=time.time())
        self.run.close()
        return {"run_dir": str(self.run.dir), "best_mIoU": self.best_miou, "best_epoch": self.best_epoch}
