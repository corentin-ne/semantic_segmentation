import pytest
import torch

from semseg.models import MODELS, backbone_parameters, build_model


@pytest.mark.parametrize("name", list(MODELS))
@pytest.mark.parametrize("size", [(64, 64), (72, 100)])
def test_output_matches_input_size(name, size):
    model = build_model(name, num_classes=7, pretrained="none").eval()
    with torch.no_grad():
        out = model(torch.randn(2, 3, *size))
    assert out.shape == (2, 7, *size)


@pytest.mark.parametrize("name", ["deeplabv3plus_mobilenet", "mobile_unet", "lraspp_mobilenet"])
def test_backward_and_param_split(name):
    model = build_model(name, num_classes=5, pretrained="none").train()
    out = model(torch.randn(2, 3, 64, 64))
    out.mean().backward()
    enc, head = backbone_parameters(model)
    assert enc and head
    assert len(enc) + len(head) == len(list(model.parameters()))


def test_unknown_model():
    with pytest.raises(KeyError):
        build_model("nope")


def test_coco_weights_restricted_to_torchvision():
    with pytest.raises(ValueError):
        build_model("mobile_unet", pretrained="coco_voc")
