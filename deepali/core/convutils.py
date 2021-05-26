from typing import Tuple

import torch
from torch import Tensor

from .types import ScalarOrTuple


def same_padding(
    kernel_size: ScalarOrTuple[int], dilation: ScalarOrTuple[int] = 1
) -> ScalarOrTuple[int]:
    r"""Padding value needed to ensure convolution preserves input tensor shape.

    Return the padding value needed to ensure a convolution using the given kernel size produces an output of the same
    shape as the input for a stride of 1, otherwise ensure a shape of the input divided by the stride rounded down.

    Raises:
        NotImplementedError: When ``(kernel_size - 1) * dilation`` is an odd number.

    """
    # Adapted from Project MONAI
    # https://github.com/Project-MONAI/MONAI/blob/db8f7877da06a9b3710071c626c0488676716be1/monai/networks/layers/convutils.py
    k: Tensor = torch.atleast_1d(torch.tensor(kernel_size, dtype=torch.int, device="cpu"))
    d: Tensor = torch.atleast_1d(torch.tensor(dilation, dtype=torch.int, device="cpu"))
    assert k.ndim == 1, "same_padding() 'kernel_size' must be scalar or sequence"
    assert d.ndim == 1, "same_padding() 'dilation' must be scalar or sequence"
    if k.sub(1).mul(d).fmod(2).eq(1).any():
        raise NotImplementedError(
            f"Same padding not available for kernel_size={tuple(k.tolist())} and dilation={tuple(d.tolist())}."
        )
    p = k.sub(1).div(2).mul(d).type(torch.int)
    return p.item() if len(p) == 1 else tuple(p.tolist())


def stride_minus_kernel_padding(
    kernel_size: ScalarOrTuple[int], stride: ScalarOrTuple[int]
) -> ScalarOrTuple[int]:
    # Adapted from Project MONAI
    # https://github.com/Project-MONAI/MONAI/blob/db8f7877da06a9b3710071c626c0488676716be1/monai/networks/layers/convutils.py
    k: Tensor = torch.atleast_1d(torch.tensor(kernel_size, dtype=torch.int, device="cpu"))
    s: Tensor = torch.atleast_1d(torch.tensor(stride, dtype=torch.int, device="cpu"))
    assert k.ndim == 1, "stride_minus_kernel_padding() 'kernel_size' must be scalar or sequence"
    assert s.ndim == 1, "stride_minus_kernel_padding() 'stride' must be scalar or sequence"
    p = s.sub(k).type(torch.int)
    return p.item() if len(p) == 1 else tuple(p.tolist())


def upsample_padding(
    kernel_size: ScalarOrTuple[int], scale_factor: ScalarOrTuple[int]
) -> Tuple[int, ...]:
    r"""Padding on both sides for transposed convolution."""
    k: Tensor = torch.atleast_1d(torch.tensor(kernel_size, dtype=torch.int, device="cpu"))
    s: Tensor = torch.atleast_1d(torch.tensor(scale_factor, dtype=torch.int, device="cpu"))
    assert k.ndim == 1, "upsample_padding() 'kernel_size' must be scalar or sequence"
    assert s.ndim == 1, "upsample_padding() 'scale_factor' must be scalar or sequence"
    p = k.sub(s).add(1).div(2).type(torch.int)
    if p.lt(0).any():
        raise ValueError(
            "upsample_padding() 'kernel_size' must be greater than or equal to 'scale_factor'"
        )
    return tuple(p.tolist())


def upsample_output_padding(
    kernel_size: ScalarOrTuple[int], scale_factor: ScalarOrTuple[int], padding: ScalarOrTuple[int]
) -> Tuple[int, ...]:
    r"""Output padding on one side for transposed convolution."""
    k: Tensor = torch.atleast_1d(torch.tensor(kernel_size, dtype=torch.int, device="cpu"))
    s: Tensor = torch.atleast_1d(torch.tensor(scale_factor, dtype=torch.int, device="cpu"))
    p: Tensor = torch.atleast_1d(torch.tensor(padding, dtype=torch.int, device="cpu"))
    assert k.ndim == 1, "upsample_output_padding() 'kernel_size' must be scalar or sequence"
    assert s.ndim == 1, "upsample_output_padding() 'scale_factor' must be scalar or sequence"
    assert p.ndim == 1, "upsample_output_padding() 'padding' must be scalar or sequence"
    op = p.mul(2).sub(k).add(s).type(torch.int)
    if op.lt(0).any():
        raise ValueError(
            "upsample_output_padding() 'output_padding' must be greater than or equal to zero"
        )
    return tuple(op.tolist())
