import torch
from torch import Tensor

from deepali.core import Grid
from deepali.core import functional as U


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


def test_sample_image() -> None:
    shape = torch.Size((5, 2, 32, 64, 63))
    image: Tensor = torch.arange(shape.numel())
    image = image.reshape(shape)
    grid = Grid(shape=shape[2:])

    indices = torch.arange(0, grid.numel(), 10)
    voxels = U.unravel_coords(indices.unsqueeze(0), grid.size())
    coords = grid.index_to_cube(voxels)
    assert coords.dtype == grid.dtype
    assert coords.dtype.is_floating_point
    assert coords.shape == (1, len(indices), 3)
    assert coords.min().ge(-1)
    assert coords.max().le(1)

    result = U.sample_image(image, coords, mode="nearest")
    expected = image.flatten(2).index_select(2, indices)
    assert result.eq(expected).all()
