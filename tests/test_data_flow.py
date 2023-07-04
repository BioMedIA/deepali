from copy import deepcopy
from typing import Tuple

import numpy as np
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
    axes = Axes.WORLD  # different from default to check if attribute is preserved

    image = FlowField(data, grid, axes)
    assert type(image) is FlowField
    assert hasattr(image, "_axes")
    assert image.axes() is axes
    assert hasattr(image, "_grid")
    assert image.grid() is grid

    result = image.type(image.dtype)
    assert result is image

    result = image.type(torch.int16)
    assert result is not image
    assert type(result) is FlowField
    assert hasattr(result, "_axes")
    assert result.axes() is axes
    assert hasattr(result, "_grid")
    assert result.grid() is image.grid()
    assert result.dtype is torch.int16
    assert result.data_ptr() != image.data_ptr()

    result = image.eq(0)
    assert type(result) is FlowField
    assert hasattr(result, "_axes")
    assert result.axes() is axes
    assert hasattr(result, "_grid")
    assert result.grid() is image.grid()
    assert result.dtype is torch.bool
    assert result.data_ptr() != image.data_ptr()

    result = result.all()
    assert type(result) is Tensor
    assert not hasattr(result, "_axes")
    assert not hasattr(result, "_grid")
    assert result.ndim == 0
    assert result.item() is True

    result = torch.add(image, 2)
    assert type(result) is FlowField
    assert hasattr(result, "_axes")
    assert result.axes() is image.axes()
    assert hasattr(result, "_grid")
    assert result.grid() is image.grid()
    assert result.eq(2).all()

    result = torch.add(4, image)
    assert type(result) is FlowField
    assert hasattr(result, "_axes")
    assert result.axes() is image.axes()
    assert hasattr(result, "_grid")
    assert result.grid() is image.grid()
    assert result.eq(4).all()

    result = image.add(1)
    assert type(result) is FlowField
    assert hasattr(result, "_axes")
    assert result.axes() is image.axes()
    assert hasattr(result, "_grid")
    assert result.grid() is image.grid()
    assert result.eq(1).all()

    result = image.clone()
    assert type(result) is FlowField
    assert hasattr(result, "_axes")
    assert result.axes() is image.axes()
    assert hasattr(result, "_grid")
    assert result.grid() is not image.grid()
    assert result.grid() == image.grid()
    assert result.data_ptr() != image.data_ptr()

    result = image.to("cpu", torch.int16)
    assert type(result) is FlowField
    assert hasattr(result, "_axes")
    assert result.axes() is image.axes()
    assert hasattr(result, "_grid")
    assert result.grid() is image.grid()
    assert result.device == torch.device("cpu")
    assert result.dtype == torch.int16

    if torch.cuda.is_available():
        result = image.cuda()
        assert type(result) is FlowField
        assert hasattr(result, "_axes")
        assert result.axes() is image.axes()
        assert hasattr(result, "_grid")
        assert result.grid() is image.grid()
        assert result.is_cuda

    result = torch.cat([image, image], dim=0)
    assert isinstance(result, Image)
    assert type(result) == Image
    assert not hasattr(result, "_axes")
    assert hasattr(result, "_grid")
    assert result.shape[0] == image.shape[0] * 2
    assert result.shape[1:] == image.shape[1:]
    assert result.grid() is image.grid()


@pytest.mark.parametrize("zeros,grid", [(d, d) for d in (3,)], indirect=True)
def test_flowfields_torch_function(zeros: Tensor, grid: Grid) -> None:
    data = zeros

    batch = FlowFields(data.unsqueeze(0), grid, Axes.WORLD)

    result = batch.detach()
    assert isinstance(result, FlowFields)
    assert result.shape == batch.shape
    assert result.axes() == batch.axes()
    assert result.grids() == batch.grids()

    result = torch.cat([batch, batch], dim=0)
    assert isinstance(result, FlowFields)
    assert result.shape[0] == batch.shape[0] + batch.shape[0]
    assert result.shape[1:] == batch.shape[1:]
    assert result.axes() == batch.axes()
    assert result.grids() == batch.grids() * 2

    result = torch.cat([batch, batch], dim=1)
    assert type(result) == ImageBatch
    assert result.shape[0] == batch.shape[0]
    assert result.shape[1] == batch.shape[1] * 2
    assert result.shape[2:] == batch.shape[2:]
    assert result.grids() == batch.grids()

    with pytest.raises(ValueError):
        # Cannot batch together flow fields with mismatching Axes
        torch.cat([batch, batch.axes(Axes.CUBE_CORNERS)], dim=0)


@pytest.mark.parametrize("zeros,grid", [(d, d) for d in (3,)], indirect=True)
def test_flowfields_getitem(zeros: Tensor, grid: Grid) -> None:
    grids = [deepcopy(grid) for _ in range(5)]  # make grids distinguishable
    batch = FlowFields(zeros.unsqueeze(0).expand((5,) + zeros.shape), grids, axes=Axes.GRID)
    for i in range(len(batch)):
        batch.tensor()[i] = i

    for i in range(len(batch)):
        item = batch[i]
        assert type(item) is FlowField
        assert item.axes() is batch.axes()
        assert item.grid() is batch.grid(i)
        assert torch.allclose(item, batch.tensor()[i])

    item = batch[1:-1:2]
    assert type(item) is FlowFields
    assert item.axes() is batch.axes()
    assert torch.allclose(item.tensor(), batch.tensor()[1:-1:2])

    item = batch[...]
    assert type(item) is FlowFields
    assert item.axes() is batch.axes()
    assert torch.allclose(item.tensor(), batch.tensor()[...])

    item = batch[:, ...]
    assert type(item) is FlowFields
    assert item.axes() is batch.axes()
    assert torch.allclose(item.tensor(), batch.tensor()[:, ...])

    item = batch[[4]]
    assert type(item) is FlowFields
    assert len(item) == 1
    assert item.axes() is batch.axes()
    assert item.grid(0) is batch.grid(4)
    assert torch.allclose(item.tensor(), batch.tensor()[[0]])

    index = [3, 2]
    item = batch[index]
    assert type(item) is FlowFields
    assert len(item) == 2
    assert item.axes() is batch.axes()
    assert item.grid(0) is batch.grid(3)
    assert item.grid(1) is batch.grid(2)
    assert torch.allclose(item.tensor(), batch.tensor()[index])

    index = (0, 2)
    item = batch[index]
    assert type(item) is Tensor
    assert torch.allclose(item, batch.tensor()[index])

    index = np.array([1, 3])
    item = batch[index]
    assert type(item) is FlowFields
    assert len(item) == 2
    assert item.axes() is batch.axes()
    assert item.grid(0) is batch.grid(1)
    assert item.grid(1) is batch.grid(3)
    assert torch.allclose(item.tensor(), batch.tensor()[index])

    index = torch.tensor([0, 2])
    item = batch[index]
    assert type(item) is FlowFields
    assert len(item) == 2
    assert item.axes() is batch.axes()
    assert item.grid(0) is batch.grid(0)
    assert item.grid(1) is batch.grid(2)
    assert torch.allclose(item.tensor(), batch.tensor()[index])

    item = batch[:, 0]
    assert type(item) is Tensor
    assert torch.allclose(item, batch.tensor()[:, 0])

    item = batch[:, 1:2]
    assert type(item) is ImageBatch
    assert all(a is b for a, b in zip(item.grids(), batch.grids()))
    assert torch.allclose(item, batch.tensor()[:, 1:2])

    if batch.ndim == 5:
        item = batch[:, :2, 0]
        assert type(item) is Tensor
        assert item.shape[1] == item.ndim - 2
        assert torch.allclose(item, batch.tensor()[:, :2, 0])

    item = batch[3, 1:2]
    assert type(item) is Image
    assert item.grid() is batch.grid(3)
    assert torch.allclose(item, batch.tensor()[3, 1:2])
