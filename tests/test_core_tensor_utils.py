import numpy as np

import torch
from torch import Tensor

from deepali.core import tensor as U


def test_unravel_coords():
    indices = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8], dtype=torch.int64)
    shape = (3, 3)
    coords = U.unravel_coords(indices, shape)
    assert isinstance(coords, Tensor)
    assert coords.dtype == torch.int64

    expected = np.array(np.unravel_index(indices.numpy(), shape)).T
    assert np.all(coords.numpy() == expected)

    indices = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=torch.int32)
    shape = (2, 5)
    coords = U.unravel_coords(indices, shape)
    assert isinstance(coords, Tensor)
    assert coords.dtype == torch.int32

    expected = np.array(np.unravel_index(indices.numpy(), shape)).T
    assert np.all(coords.numpy() == expected)
