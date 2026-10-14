import numpy as np
import torch
from PIL import Image

from semseg.config import load_config
from semseg.data import EvalTransform, ShapesDataset, TrainTransform


def _pair(h=90, w=120):
    image = Image.fromarray(np.random.randint(0, 255, (h, w, 3), dtype=np.uint8))
    mask = np.zeros((h, w), dtype=np.uint8)
    mask[10:40, 10:40] = 3
    mask[0, :] = 255  # VOC border
    return image, Image.fromarray(mask)


def test_train_transform_shapes_and_pad():
    img, msk = TrainTransform(crop_size=128, scale_range=(0.5, 0.5))(*_pair())
    assert img.shape == (3, 128, 128) and img.dtype == torch.float32
    assert msk.shape == (128, 128) and msk.dtype == torch.long
    assert (msk == 255).any()  # padding is ignored, not background
    assert set(msk.unique().tolist()) <= {0, 3, 255}


def test_eval_transform_keeps_resolution():
    img, msk = EvalTransform(size=160)(*_pair(90, 120))
    assert img.shape == (3, 160, 160)
    assert (msk[90:, :] == 255).all() and (msk[:, 120:] == 255).all()
    assert (msk[10:40, 10:40] == 3).all()


def test_shapes_dataset_deterministic():
    ds = ShapesDataset(4, size=64)
    a, b = np.asarray(ds[1][1]), np.asarray(ds[1][1])
    assert (a == b).all() and a.max() > 0
    val = ShapesDataset(4, size=64, split="val")
    assert not (np.asarray(val[1][0]) == np.asarray(ds[1][0])).all()


def test_config_overrides(tmp_path):
    path = tmp_path / "c.yaml"
    path.write_text("model:\n  name: mobile_unet\ntrain:\n  epochs: 3\n")
    cfg = load_config(path, ["optim.lr=0.01", "train.amp=false", "data.scale_range=[0.5,1.5]", "model.kwargs.dropout=0.2"])
    assert cfg.model.name == "mobile_unet" and cfg.train.epochs == 3
    assert cfg.optim.lr == 0.01 and cfg.train.amp is False
    assert cfg.data.scale_range == (0.5, 1.5) and cfg.model.kwargs == {"dropout": 0.2}
