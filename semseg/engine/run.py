"""Run directory: everything the dashboard needs, written as plain JSON / PNG files.

runs/<name>/
    config.json      full experiment config
    summary.json     model info, status, best metrics (rewritten every evaluation)
    metrics.jsonl    one line per evaluated epoch
    steps.jsonl      training loss / LR every ``log_every`` iterations
    confusion.json   confusion matrix of the best checkpoint
    samples/epoch_XXX.png
    checkpoints/{best,last}.pt
"""
from __future__ import annotations

import json
import os
import time
from datetime import datetime
from pathlib import Path


class RunLogger:
    def __init__(self, runs_dir: str, name: str, config: dict, use_wandb: bool = False, wandb_project: str = "",
                 existing_dir: str | Path | None = None):
        if existing_dir:  # resuming: keep appending to the same run
            self.dir = Path(existing_dir)
            self.summary: dict = json.loads((self.dir / "summary.json").read_text())
        else:
            stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            self.dir = Path(runs_dir) / f"{name}-{stamp}"
            self.summary = {"name": self.dir.name, "status": "starting", "started_at": time.time()}
        (self.dir / "samples").mkdir(parents=True, exist_ok=True)
        (self.dir / "checkpoints").mkdir(exist_ok=True)
        self._write_json("config.json", config)
        self.wandb = None
        if use_wandb:
            try:
                import wandb
                self.wandb = wandb.init(project=wandb_project, name=self.dir.name, config=config)
            except Exception as exc:  # wandb is optional: never fail a training because of it
                print(f"[wandb] disabled: {exc}")

    def _write_json(self, filename: str, data) -> None:
        tmp = self.dir / f".{filename}.tmp"
        tmp.write_text(json.dumps(data, indent=2))
        os.replace(tmp, self.dir / filename)  # atomic: the dashboard never reads half a file

    def _append(self, filename: str, record: dict) -> None:
        with open(self.dir / filename, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")

    def log_step(self, record: dict) -> None:
        self._append("steps.jsonl", record)
        if self.wandb:
            self.wandb.log({f"step/{k}": v for k, v in record.items()})

    def log_epoch(self, record: dict) -> None:
        self._append("metrics.jsonl", record)
        if self.wandb:
            flat = {k: v for k, v in record.items() if not isinstance(v, list)}
            self.wandb.log(flat)

    def update_summary(self, **fields) -> None:
        self.summary.update(fields, updated_at=time.time())
        self._write_json("summary.json", self.summary)

    def save_confusion(self, matrix: list[list[int]]) -> None:
        self._write_json("confusion.json", matrix)

    def save_sample(self, epoch: int, image) -> Path:
        path = self.dir / "samples" / f"epoch_{epoch:03d}.png"
        image.save(path)
        return path

    def close(self) -> None:
        if self.wandb:
            self.wandb.finish()
