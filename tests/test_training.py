import json

from semseg.config import load_config
from semseg.engine import Trainer
from semseg.inference import Predictor


def test_end_to_end(tmp_path):
    cfg = load_config("configs/synthetic_demo.yaml", [
        "model.name=lraspp_mobilenet", "train.epochs=2", "train.batch_size=4", "train.eval_batch_size=4",
        "data.synthetic_train=8", "data.synthetic_val=4", "data.synthetic_size=64", "data.crop_size=64",
        "data.eval_size=64", "data.num_workers=0", "train.log_every=1", f"logging.runs_dir={tmp_path}",
    ])
    result = Trainer(cfg).fit()
    run = tmp_path / next(p.name for p in tmp_path.iterdir())
    summary = json.loads((run / "summary.json").read_text())
    assert summary["status"] == "finished" and summary["epoch"] == 2
    assert len((run / "metrics.jsonl").read_text().splitlines()) == 2
    assert (run / "checkpoints" / "best.pt").exists() and (run / "samples" / "epoch_002.png").exists()
    assert 0.0 <= result["best_mIoU"] <= 1.0

    # resume one more epoch into the same run directory
    cfg.train.epochs, cfg.train.resume = 3, str(run / "checkpoints" / "last.pt")
    Trainer(cfg).fit()
    assert len((run / "metrics.jsonl").read_text().splitlines()) == 3

    labels, conf = Predictor(run, device="cpu").predict(__import__("PIL.Image").Image.new("RGB", (50, 70)))
    assert labels.shape == (70, 50) and conf.shape == (70, 50)
