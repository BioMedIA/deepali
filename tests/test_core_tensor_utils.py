import numpy as np

import torch
from torch import Tensor

from deepali.core import tensor as U


def test_unravel_coords():
    indices = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8], dtype=torch.int64)
    size = (3, 3)
    shape = tuple(reversed(size))
    coords = U.unravel_coords(indices, size)
    assert isinstance(coords, Tensor)
    assert coords.dtype == torch.int64

    expected = np.array(np.unravel_index(indices.numpy(), shape)).T
    expected = np.flip(expected, axis=-1)
    assert np.all(coords.numpy() == expected)

    indices = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=torch.int32)
    size = (2, 5)
    shape = tuple(reversed(size))
    coords = U.unravel_coords(indices, size)
    assert isinstance(coords, Tensor)
    assert coords.dtype == torch.int32

    expected = np.array(np.unravel_index(indices.numpy(), shape)).T
    expected = np.flip(expected, axis=-1)
    assert np.all(coords.numpy() == expected)


def test_unravel_index():
    indices = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8], dtype=torch.int64)
    shape = (3, 3)
    result = U.unravel_index(indices, shape)
    assert isinstance(result, Tensor)
    assert result.dtype == torch.int64

    expected = np.array(np.unravel_index(indices.numpy(), shape)).T
    assert np.all(result.numpy() == expected)

    indices = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=torch.int32)
    shape = (2, 5)
    result = U.unravel_index(indices, shape)
    assert isinstance(result, Tensor)
    assert result.dtype == torch.int32

    expected = np.array(np.unravel_index(indices.numpy(), shape)).T
    assert np.all(result.numpy() == expected)
