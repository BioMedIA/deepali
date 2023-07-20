r"""Basic math operations with tensors."""

from typing import Optional, Union

import torch
from torch import Tensor

from .typing import Scalar


def abspow(x: Tensor, exponent: Union[int, float]) -> Tensor:
    r"""Compute ``abs(x)**exponent``."""
    if exponent == 1:
        return x.abs()
    return x.abs().pow(exponent)


def atanh(x: Tensor) -> Tensor:
    r"""Inverse of tanh function.

    Args:
        x: Function argument.

    Returns:
        Inverse of tanh function, i.e., ``y`` where ``x = tanh(y)``.

    See also:
        https://github.com/pytorch/pytorch/issues/10324

    """
    return torch.log1p(2 * x / (1 - x)) / 2


def max_difference(source: Tensor, target: Tensor) -> Tensor:
    r"""Maximum possible intensity difference.

    Note that the two input images need not be sampled on the same grid.

    Args:
        source: Source image.
        target: Reference image.

    Returns:
        Maximum possible intensity difference.

    """
    smin, smax = source.min(), source.max()
    if target is source:
        tmin, tmax = smin, smax
    else:
        tmin, tmax = target.min(), target.max()
    return torch.max(torch.abs(smax - tmin), torch.abs(tmax - smin))


def round_decimals(tensor: Tensor, decimals: int = 0, out: Optional[Tensor] = None) -> Tensor:
    r"""Round tensor values to specified number of decimals."""
    if not decimals:
        result = torch.round(tensor, out=out)
    else:
        scale = 10**decimals
        if out is tensor:
            tensor *= scale
        else:
            tensor = tensor * scale
        result = torch.round(tensor, out=out)
        result /= scale
    return result


def threshold(data: Tensor, min: Optional[Scalar], max: Optional[Scalar] = None) -> Tensor:
    r"""Get mask for given lower and upper thresholds.

    Args:
        data: Input data tensor.
        min: Lower threshold. If ``None``, use ``data.min()``.
        max: Upper threshold. If ``None``, use ``data.max()``.

    Returns:
        Boolean tensor with same shape as ``data``, where only elements with a value
        greater than or equal ``min`` and less than or equal ``max`` are ``True``.

    """
    if min is None and max is None:
        return torch.ones_like(data, dtype=torch.bool)
    if min is None:
        return data <= max
    if max is None:
        return data >= min
    return (min <= data) & (data <= max)
