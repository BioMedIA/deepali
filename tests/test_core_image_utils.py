import os

import pytest
import torch
from torch import Tensor
import torch.nn.functional as F

from deepali.core import functional as U
from deepali.core import Grid


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


def test_finite_differences() -> None:
    image = torch.randn((3, 2, 8, 16, 24))

    # Forward difference scheme
    image.requires_grad = True
    deriv = U.finite_differences(image, "x", mode="forward")
    assert isinstance(deriv, Tensor)
    assert deriv.requires_grad is True
    deriv = deriv.detach()
    assert deriv.requires_grad is False
    assert deriv.dtype.is_floating_point
    assert deriv.shape == image.shape

    expected = F.pad(image, (0, 1, 0, 0, 0, 0), mode="replicate")
    expected = expected[..., 1:].sub(expected[..., :-1])
    assert torch.allclose(deriv, expected)

    # Backward difference scheme
    image.requires_grad = False
    deriv = U.finite_differences(image, "y", mode="backward")
    assert isinstance(deriv, Tensor)
    assert deriv.requires_grad is False
    assert deriv.dtype.is_floating_point
    assert deriv.shape == image.shape

    expected = F.pad(image, (0, 0, 1, 0, 0, 0), mode="replicate")
    expected = expected[..., 1:, :].sub(expected[..., :-1, :])
    assert torch.allclose(deriv, expected)

    # Central difference scheme
    image.requires_grad = False
    deriv = U.finite_differences(image, "z", mode="central")
    assert isinstance(deriv, Tensor)
    assert deriv.requires_grad is False
    assert deriv.dtype.is_floating_point
    assert deriv.shape == image.shape

    expected = F.pad(image, (0, 0, 0, 0, 1, 1), mode="replicate")
    expected = expected[:, :, 2:, :, :].sub(expected[:, :, :-2, :, :]).div(2)
    assert torch.allclose(deriv, expected)

    # Forward-central-backward difference scheme
    deriv = U.finite_differences(image, "z", mode="forward_central_backward")
    assert isinstance(deriv, Tensor)
    assert deriv.requires_grad is False
    assert deriv.dtype.is_floating_point
    assert deriv.shape == image.shape

    expected = torch.cat(
        [
            image[:, :, 1:2, :, :].sub(image[:, :, :1, :, :]),
            image[:, :, 2:, :, :].sub(image[:, :, :-2, :, :]).div(2),
            image[:, :, -1:, :, :].sub(image[:, :, -2:-1, :, :]),
        ],
        dim=2,
    )
    assert torch.allclose(deriv, expected)

    # Zero-th order
    assert U.finite_differences(image, "x", mode="central", order=0) is image


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_rand_sample_cuda() -> None:
    if "CUDA_VISIBLE_DEVICES" not in os.environ:
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    device = torch.device("cuda:0")
    generator = torch.Generator(device=device).manual_seed(123456789)

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    shape = torch.Size((5, 1, 32, 64, 128))
    data = torch.arange(shape[2:].numel()).reshape((1, 1) + shape[2:]).expand(shape).to(device)
    num_samples = 10

    # Draw unweighted samples with and without replacement
    t_elapsed_1 = 0
    for i in range(5):
        start.record()
        values = U.rand_sample(data, num_samples, mask=None, replacement=False, generator=generator)
        end.record()
        torch.cuda.synchronize()
        if i > 0:
            t_elapsed_1 += start.elapsed_time(end)
    t_elapsed_1 /= 4

    assert not torch.allclose(values[0], values[1])

    t_elapsed_2 = 0
    for i in range(5):
        start.record()
        values = U.rand_sample(data, num_samples, mask=None, replacement=True, generator=generator)
        end.record()
        torch.cuda.synchronize()
        if i > 0:
            t_elapsed_2 += start.elapsed_time(end)
    t_elapsed_2 /= 4

    assert not torch.allclose(values[0], values[1])

    # Compare with using multinomial with an all-one mask
    t_elapsed_3 = 0
    for i in range(5):
        start.record()
        mask = torch.ones((1, 1) + data.shape[2:], device=device)
        values = U.rand_sample(data, num_samples, mask=mask, replacement=False, generator=generator)
        end.record()
        torch.cuda.synchronize()
        if i > 0:
            t_elapsed_3 += start.elapsed_time(end)
    t_elapsed_3 /= 4

    assert not torch.allclose(values[0], values[1])

    t_elapsed_4 = 0
    for i in range(5):
        start.record()
        mask = torch.ones((1, 1) + data.shape[2:], device=device)
        values = U.rand_sample(data, num_samples, mask=mask, replacement=True, generator=generator)
        end.record()
        torch.cuda.synchronize()
        if i > 0:
            t_elapsed_4 += start.elapsed_time(end)
    t_elapsed_4 /= 4

    assert not torch.allclose(values[0], values[1])


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
