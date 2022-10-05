import pytest
import torch
from torch import Tensor

from deepali.core.kernels import gaussian1d
from deepali.core.random import _multinomial


@pytest.fixture
def generator() -> torch.Generator:
    return torch.Generator("cpu").manual_seed(123456789)


def test_multinomial_with_replacement(generator: torch.Generator) -> None:
    r"""Test weighted sampling with replacement using inverse transform sampling."""

    input = torch.arange(1000)

    # Input checks
    with pytest.raises(ValueError):
        _multinomial(torch.tensor(0), 1, replacement=True)
    with pytest.raises(ValueError):
        _multinomial(torch.ones((1, 2, 3)), 1, replacement=True)

    index = _multinomial(torch.ones(10), 11, replacement=True)  # no exception
    assert isinstance(index, Tensor)
    assert index.dtype == torch.int64
    assert index.shape == (11,)

    # Output type and shape
    index = _multinomial(input, 10, replacement=True)
    assert isinstance(index, Tensor)
    assert index.dtype == torch.int64
    assert index.shape == (10,)

    index = _multinomial(input.unsqueeze(0), 10, replacement=True)
    assert isinstance(index, Tensor)
    assert index.dtype == torch.int64
    assert index.shape == (1, 10)

    index = _multinomial(input.unsqueeze(0).repeat(3, 1), 10, replacement=True)
    assert isinstance(index, Tensor)
    assert index.dtype == torch.int64
    assert index.shape == (3, 10)

    # Only samples with non-zero weight
    subset = input.clone()
    subset[subset.lt(100)] = 0
    subset[subset.gt(200)] = 0

    for _ in range(10):
        index = _multinomial(subset, 10, replacement=True, generator=generator)
        assert isinstance(index, Tensor)
        assert index.dtype == torch.int64
        assert index.shape == (10,)
        assert subset[index].ge(100).le(200).all()

    # Sample from a spatial Gaussian distribution
    num_samples = 1_000_000
    input = gaussian1d(5)
    index = _multinomial(input, num_samples, replacement=True, generator=generator)
    assert isinstance(index, Tensor)
    assert index.dtype == torch.int64
    assert index.shape == (num_samples,)
    freq = index.bincount().div(num_samples)
    assert freq.allclose(input, rtol=0.1)


def test_multinomial_without_replacement(generator: torch.Generator) -> None:
    r"""Test weighted sampling without replacement using Gumbel-max trick."""

    input = torch.arange(1000)

    # Input checks
    with pytest.raises(ValueError):
        _multinomial(torch.tensor(0), 1, replacement=False)
    with pytest.raises(ValueError):
        _multinomial(torch.ones((1, 2, 3)), 1, replacement=False)

    index = _multinomial(torch.ones(10), 10, replacement=False)
    assert isinstance(index, Tensor)
    assert index.dtype == torch.int64
    assert index.shape == (10,)
    assert index.sort().values.eq(torch.arange(10)).all()

    with pytest.raises(ValueError):
        _multinomial(torch.ones(10), 11, replacement=False)

    # Output type and shape
    index = _multinomial(input, 10, replacement=False)
    assert isinstance(index, Tensor)
    assert index.dtype == torch.int64
    assert index.shape == (10,)

    index = _multinomial(input.unsqueeze(0), 10, replacement=False)
    assert isinstance(index, Tensor)
    assert index.dtype == torch.int64
    assert index.shape == (1, 10)

    index = _multinomial(input.unsqueeze(0).repeat(3, 1), 10, replacement=False)
    assert isinstance(index, Tensor)
    assert index.dtype == torch.int64
    assert index.shape == (3, 10)

    # No duplicates
    subset = input.clone()
    subset[subset.lt(100)] = 0
    subset[subset.gt(200)] = 0
    subset[subset.gt(0)] = 1

    num_samples = 10
    num_repeat = 100

    for _ in range(num_repeat):
        index = _multinomial(subset, num_samples, replacement=False, generator=generator)
        assert isinstance(index, Tensor)
        assert index.dtype == torch.int64
        assert index.shape == (num_samples,)
        assert index.unique().shape == (num_samples,)
        assert subset[index].eq(1).all()
