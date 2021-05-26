import torch
from torch import Tensor

import deepali.core.image as U


def test_fill_border():
    image = torch.zeros((2, 1, 7, 5), dtype=torch.float)

    result = U.fill_border(image, margin=1, value=1)
    assert isinstance(result, Tensor)
    assert result.shape == image.shape
    assert result[:, :, :1, :].eq(1).all()
    assert result[:, :, :, :1].eq(1).all()
    assert result[:, :, 6:, :].eq(1).all()
    assert result[:, :, :, 4:].eq(1).all()
    assert result[:, :, 1:6, 1:4].eq(0).all()
    assert result.sum() == 40

    result = U.fill_border(image, margin=(1, 2), value=1)
    assert isinstance(result, Tensor)
    assert result.shape == image.shape
    assert result[:, :, :2, :].eq(1).all()
    assert result[:, :, :, :1].eq(1).all()
    assert result[:, :, 5:, :].eq(1).all()
    assert result[:, :, :, 4:].eq(1).all()
    assert result[:, :, 2:5, 1:4].eq(0).all()
    assert result.sum() == 52

    image = torch.zeros((2, 1, 7, 5, 11), dtype=torch.float)
    result = U.fill_border(image, margin=(3, 1, 2), value=1)
    assert isinstance(result, Tensor)
    assert result.shape == image.shape
    assert result[:, :, :2, :, :].eq(1).all()
    assert result[:, :, :, :1, :].eq(1).all()
    assert result[:, :, :, :, :3].eq(1).all()
    assert result[:, :, 5:, :, :].eq(1).all()
    assert result[:, :, :, 4:, :].eq(1).all()
    assert result[:, :, :, :, 8:].eq(1).all()
    assert result[:, :, 2:5, 1:4, 3:8].eq(0).all()
    assert result.sum() == 680
