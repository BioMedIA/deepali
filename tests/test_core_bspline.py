import pytest

import torch
from torch import Tensor

from deepali.core import bspline as B


def test_cubic_bspline_interpolation_weights() -> None:
    kernel = B.cubic_bspline_interpolation_weights(5)
    assert isinstance(kernel, Tensor)
    assert kernel.shape == (5, 4)

    assert torch.allclose(kernel, B.cubic_bspline_interpolation_weights(5, derivative=0))

    kernels = B.cubic_bspline_interpolation_weights(5, derivative=[0, 1])
    assert isinstance(kernels, tuple)
    assert len(kernels) == 2
    assert torch.allclose(kernels[0], B.cubic_bspline_interpolation_weights(5, derivative=0))
    assert torch.allclose(kernels[1], B.cubic_bspline_interpolation_weights(5, derivative=1))

    with pytest.raises(ValueError):
        B.cubic_bspline_interpolation_weights([5], derivative=[0, 1])
    with pytest.raises(ValueError):
        B.cubic_bspline_interpolation_weights([5, 5], derivative=[0])


def test_cubic_bspline_interpolation_at_control_points() -> None:
    r"""Test evaluation of cubic B-spline at control point grid locations only."""

    kernel_all = B.cubic_bspline_interpolation_weights(5)
    kernel_cps = B.cubic_bspline_interpolation_weights(1)
    assert kernel_all.shape == (5, 4)
    assert kernel_cps.shape == (1, 4)
    assert torch.allclose(kernel_cps, kernel_all[0])

    # Single non-zero control point coefficient
    data = torch.zeros((1, 1, 4))
    data[0, 0, 1] = 1

    values_all = B.evaluate_cubic_bspline(data, stride=5, kernel=kernel_all)
    values_cps = B.evaluate_cubic_bspline(data, stride=1, kernel=kernel_cps)
    assert torch.allclose(B.evaluate_cubic_bspline(data, stride=1), values_cps)

    assert values_all.shape == (1, 1, 5)
    assert values_cps.shape == (1, 1, 1)

    assert torch.allclose(values_cps, values_all[:, :, ::5])
    assert torch.allclose(values_cps[0, 0], kernel_cps[0, 1])

    # Random uniformly distributed control point coefficients
    data = torch.rand((1, 1, 35))

    values_all = B.evaluate_cubic_bspline(data, stride=5, kernel=kernel_all)
    values_cps = B.evaluate_cubic_bspline(data, stride=1, kernel=kernel_cps)
    assert torch.allclose(B.evaluate_cubic_bspline(data, stride=1), values_cps)

    assert values_all.shape == (1, 1, 32 * 5)
    assert values_cps.shape == (1, 1, 32)
    assert torch.allclose(values_cps, values_all[:, :, ::5])
