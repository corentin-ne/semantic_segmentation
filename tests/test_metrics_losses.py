import math

import torch

from semseg.losses import CEDice, DiceLoss, OhemCrossEntropy
from semseg.metrics import SegmentationMetrics


def test_perfect_prediction():
    target = torch.tensor([[[0, 1], [2, 255]]])
    m = SegmentationMetrics(3)
    m.update(target.clamp(max=2), target)
    r = m.compute()
    assert r["mIoU"] == 1.0 and r["pixel_acc"] == 1.0


def test_known_confusion():
    target = torch.tensor([[0, 0, 1, 1]])
    pred = torch.tensor([[0, 1, 1, 1]])
    m = SegmentationMetrics(2)
    m.update(pred, target)
    r = m.compute()
    # class 0: tp 1, fp 0, fn 1 -> 1/2 ; class 1: tp 2, fp 1, fn 0 -> 2/3
    assert math.isclose(r["mIoU"], (0.5 + 2 / 3) / 2, rel_tol=1e-6)
    assert math.isclose(r["pixel_acc"], 0.75)
    assert m.confusion.tolist() == [[1, 1], [0, 2]]


def test_ignore_index_and_absent_class():
    target = torch.tensor([[0, 255, 255]])
    pred = torch.tensor([[0, 2, 1]])
    m = SegmentationMetrics(3)
    m.update(pred, target)
    r = m.compute()
    assert r["mIoU"] == 1.0  # classes 1 and 2 are absent from the ground truth
    assert r["per_class_iou"][1] is None  # predictions on ignored pixels do not count


def test_logits_accepted():
    logits = torch.zeros(1, 3, 2, 2)
    logits[:, 2] = 1
    m = SegmentationMetrics(3)
    m.update(logits, torch.full((1, 2, 2), 2))
    assert m.compute()["mIoU"] == 1.0


def test_losses_ignore_and_finite():
    logits = torch.randn(2, 4, 8, 8, requires_grad=True)
    target = torch.randint(0, 4, (2, 8, 8))
    target[:, :2] = 255
    for loss_fn in (OhemCrossEntropy(min_kept=10), DiceLoss(), CEDice()):
        loss = loss_fn(logits, target)
        loss.backward()
        assert torch.isfinite(loss)


def test_ohem_all_ignored():
    logits = torch.randn(1, 3, 4, 4, requires_grad=True)
    loss = OhemCrossEntropy()(logits, torch.full((1, 4, 4), 255))
    assert loss.item() == 0.0
