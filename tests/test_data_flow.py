from typing import Tuple

import pytest
import torch
from torch import Tensor

from deepali.core import Axes, Grid
from deepali.data import FlowField, FlowFields, Image, ImageBatch


def image_size(sdim: int) -> Tuple[int, ...]:
    if sdim == 2:
        return (64, 57)
    if sdim == 3:
        return (64, 57, 31)
    raise ValueError("image_size() 'sdim' must be 2 or 3")


def image_shape(sdim: int) -> Tuple[int, ...]:
    return tuple(reversed(image_size(sdim)))


@pytest.fixture(scope="function")
def grid(request) -> Tensor:
    size = image_size(request.param)
    spacing = (0.25, 0.2, 0.5)[: len(size)]
    return Grid(size=size, spacing=spacing)


@pytest.fixture(scope="function")
def data(request) -> Tensor:
    shape = image_shape(request.param)
    return torch.randn((len(shape),) + shape).mul_(100)


@pytest.fixture(scope="function")
def zeros(request) -> Tensor:
    shape = image_shape(request.param)
    return torch.zeros((len(shape),) + shape)


@pytest.mark.parametrize("zeros,grid", [(d, d) for d in (3,)], indirect=True)
def test_flowfield_torch_function(zeros: Tensor, grid: Grid) -> None:
    data = zeros

    image = FlowField(data, grid)
    assert type(image) is FlowField
    assert hasattr(image, "_grid")
    assert image.grid() is grid

    result = image.type(image.dtype)
    assert result is image

    result = image.type(torch.int16)
    assert result is not image
    assert type(result) is FlowField
    assert hasattr(result, "_grid")
    assert result.grid() is image.grid()
    assert result.dtype is torch.int16
    assert result.data_ptr() != image.data_ptr()

    result = image.eq(0)
    assert type(result) is FlowField
    assert hasattr(result, "_grid")
    assert result.grid() is image.grid()
    assert result.dtype is torch.bool
    assert result.data_ptr() != image.data_ptr()

    result = result.all()
    assert type(result) is Tensor
    assert not hasattr(result, "_grid")
    assert result.ndim == 0
    assert result.item() is True

    result = torch.add(image, 2)
    assert type(result) is FlowField
    assert hasattr(result, "_grid")
    assert result.grid() is image.grid()
    assert result.eq(2).all()

    result = torch.add(4, image)
    assert type(result) is FlowField
    assert hasattr(result, "_grid")
    assert result.grid() is image.grid()
    assert result.eq(4).all()

    result = image.add(1)
    assert type(result) is FlowField
    assert hasattr(result, "_grid")
    assert result.grid() is image.grid()
    assert result.eq(1).all()

    result = image.clone()
    assert type(result) is FlowField
    assert hasattr(result, "_grid")
    assert result.grid() is not image.grid()
    assert result.grid() == image.grid()
    assert result.data_ptr() != image.data_ptr()

    result = image.to("cpu", torch.int16)
    assert type(result) is FlowField
    assert hasattr(result, "_grid")
    assert result.grid() is image.grid()
    assert result.device == torch.device("cpu")
    assert result.dtype == torch.int16

    if torch.cuda.is_available():
        result = image.cuda()
        assert type(result) is FlowField
        assert hasattr(result, "_grid")
        assert result.grid() is image.grid()
        assert result.is_cuda

    result = torch.cat([image, image], dim=0)
    assert isinstance(result, Image)
    assert type(result) == Image
    assert hasattr(result, "_grid")
    assert result.shape[0] == image.shape[0] * 2
    assert result.shape[1:] == image.shape[1:]
    assert result.grid() is image.grid()


@pytest.mark.parametrize("zeros,grid", [(d, d) for d in (3,)], indirect=True)
def test_flowfields_torch_function(zeros: Tensor, grid: Grid) -> None:
    data = zeros

    batch = FlowFields(data.unsqueeze(0), grid, Axes.WORLD)

    result = torch.cat([batch, batch], dim=0)
    assert isinstance(result, FlowFields)
    assert result.shape[0] == batch.shape[0] + batch.shape[0]
    assert result.shape[1:] == batch.shape[1:]

    result = torch.cat([batch, batch], dim=1)
    assert type(result) == ImageBatch
    assert result.shape[0] == batch.shape[0]
    assert result.shape[1] == batch.shape[1] * 2
    assert result.shape[2:] == batch.shape[2:]

    with pytest.raises(ValueError):
        # Cannot batch together flow fields with mismatching Axes
        torch.cat([batch, batch.axes(Axes.CUBE_CORNERS)], dim=0)
