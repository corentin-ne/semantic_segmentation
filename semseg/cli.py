"""Command line: ``python -m semseg {train,evaluate,benchmark,predict,export,models}``."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def cmd_train(args) -> None:
    from .config import load_config
    from .engine.trainer import Trainer

    cfg = load_config(args.config, args.override)
    result = Trainer(cfg).fit()
    print(json.dumps(result, indent=2))


def cmd_evaluate(args) -> None:
    from .config import Config
    from .data import build_datasets
    from .engine.evaluate import evaluate_model
    from .inference import find_checkpoint, load_model
    from .losses import build_loss
    from torch.utils.data import DataLoader

    run_dir = Path(args.run)
    cfg = Config.from_dict(json.loads((run_dir / "config.json").read_text()))
    model, meta = load_model(find_checkpoint(run_dir), args.device)
    _, val_set = build_datasets(cfg)
    loader = DataLoader(val_set, batch_size=cfg.train.eval_batch_size, num_workers=cfg.data.num_workers)
    result, _ = evaluate_model(model, loader, meta["num_classes"], next(model.parameters()).device,
                               build_loss(cfg.loss), cfg.loss.ignore_index)
    result.pop("confusion")
    for name, iou in zip(meta["class_names"], result.pop("per_class_iou")):
        print(f"  {name:>14s}  {'n/a' if iou is None else f'{iou * 100:5.1f}'}")
    print(json.dumps(result, indent=2))


def cmd_benchmark(args) -> None:
    import torch

    from .benchmark import profile_model
    from .models import MODELS, build_model

    torch.set_num_threads(args.threads) if args.threads else None
    names = args.models.split(",") if args.models else list(MODELS)
    rows = []
    for name in names:
        model = build_model(name, args.num_classes, "none")
        stats = profile_model(model, args.size, args.device, iters=args.iters)
        rows.append({"model": name, "family": MODELS[name].family, **{k: round(v, 3) if isinstance(v, float) else v
                                                                      for k, v in stats.items()}})
        print(f"{name:26s} {stats['params_m']:7.2f}M  {stats['gflops']:7.2f} GFLOPs  "
              f"{stats['latency_ms']:8.1f} ms  {stats['fps']:6.1f} img/s")
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"device": args.device, "size": args.size, "torch": torch.__version__,
                               "threads": torch.get_num_threads(), "results": rows}, indent=2))
    print(f"Saved {out}")


def cmd_predict(args) -> None:
    from .inference import predict_file

    predict_file(args.run, args.image, args.output, args.tta)


def cmd_export(args) -> None:
    from .export import export_onnx

    export_onnx(args.run, args.output, args.size)


def cmd_models(_args) -> None:
    from .models import MODELS

    for name, spec in MODELS.items():
        print(f"{name:26s} [{spec.family:11s}] {spec.description}")


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(prog="semseg", description="Semantic segmentation on Pascal VOC")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("train", help="train a model from a YAML config")
    p.add_argument("--config", "-c", required=True)
    p.add_argument("--override", "-o", nargs="*", default=[], help="e.g. optim.lr=1e-3 train.epochs=10")
    p.set_defaults(func=cmd_train)

    p = sub.add_parser("evaluate", help="evaluate the best checkpoint of a run on its validation set")
    p.add_argument("--run", required=True)
    p.add_argument("--device", default="auto")
    p.set_defaults(func=cmd_evaluate)

    p = sub.add_parser("benchmark", help="params / GFLOPs / latency of every model")
    p.add_argument("--models", default="")
    p.add_argument("--size", type=int, default=512)
    p.add_argument("--device", default="cpu")
    p.add_argument("--iters", type=int, default=20)
    p.add_argument("--threads", type=int, default=0)
    p.add_argument("--num-classes", type=int, default=21)
    p.add_argument("--output", default="runs/benchmark.json")
    p.set_defaults(func=cmd_benchmark)

    p = sub.add_parser("predict", help="segment an image with a trained run")
    p.add_argument("--run", required=True, help="run directory or checkpoint")
    p.add_argument("--image", required=True)
    p.add_argument("--output")
    p.add_argument("--tta", action="store_true", help="flip + multi-scale test-time augmentation")
    p.set_defaults(func=cmd_predict)

    p = sub.add_parser("export", help="export a trained run to ONNX")
    p.add_argument("--run", required=True)
    p.add_argument("--output")
    p.add_argument("--size", type=int, default=512)
    p.set_defaults(func=cmd_export)

    sub.add_parser("models", help="list available models").set_defaults(func=cmd_models)

    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
