import pytest

import torch
from torch import Tensor

from deepali.core import bspline as B


def test_cubic_bspline_interpolation_weights() -> None:
    kernel = B.cubic_bspline_interpolation_weights(5)
    assert isinstance(kernel, Tensor)

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
