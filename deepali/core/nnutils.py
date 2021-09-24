from typing import Optional, Sequence, Tuple, overload

import torch
from torch import Size, Tensor

from .types import ScalarOrTuple


def conv_output_size(
    in_size: ScalarOrTuple[int],
    kernel_size: ScalarOrTuple[int],
    stride: ScalarOrTuple[int] = 1,
    padding: ScalarOrTuple[int] = 0,
    dilation: ScalarOrTuple[int] = 1,
) -> ScalarOrTuple[int]:
    r"""Calculate spatial size of output tensor after convolution."""
    device = torch.device("cpu")
    m: Tensor = torch.atleast_1d(torch.tensor(in_size, dtype=torch.int, device=device))
    k: Tensor = torch.atleast_1d(torch.tensor(kernel_size, dtype=torch.int, device=device))
    s: Tensor = torch.atleast_1d(torch.tensor(stride, dtype=torch.int, device=device))
    d: Tensor = torch.atleast_1d(torch.tensor(dilation, dtype=torch.int, device=device))
    if m.ndim != 1:
        raise ValueError("conv_output_size() 'in_size' must be scalar or sequence")
    ndim = m.shape[0]
    if ndim == 1 and k.shape[0] > 1:
        ndim = k.shape[0]
    for arg, name in zip([k, s, d], ["kernel_size", "stride", "dilation"]):
        if arg.ndim != 1 or arg.shape[0] not in (1, ndim):
            raise ValueError(
                f"conv_output_size() {name!r} must be scalar or sequence of length {ndim}"
            )
    k = k.expand(ndim)
    s = s.expand(ndim)
    d = d.expand(ndim)
    if padding == "valid":
        padding = 0
    elif padding == "same":
        if not s.eq(1).all():
            raise ValueError("conv_output_size() padding='same' requires stride=1")
        padding = same_padding(kernel_size=kernel_size, dilation=dilation)
    elif isinstance(padding, str):
        raise ValueError("conv_output_size() 'padding' string must be 'valid' or 'same'")
    p: Tensor = torch.atleast_1d(torch.tensor(padding, dtype=torch.int, device=device))
    if p.ndim != 1 or p.shape[0] not in (1, ndim):
        raise ValueError(
            f"conv_output_size() 'padding' must be scalar or sequence of length {ndim}"
        )
    p = p.expand(ndim)
    n = p.mul(2).add_(m).sub_(k.sub(1).mul_(d)).sub_(1).float().div_(s).add_(1).floor_().long()
    if isinstance(in_size, int):
        return n[0].item()
    return Size(n.tolist())


def conv_transposed_output_size(
    in_size: ScalarOrTuple[int],
    kernel_size: ScalarOrTuple[int],
    stride: ScalarOrTuple[int] = 1,
    padding: ScalarOrTuple[int] = 0,
    output_padding: ScalarOrTuple[int] = 0,
    dilation: ScalarOrTuple[int] = 1,
) -> ScalarOrTuple[int]:
    r"""Calculate spatial size of output tensor after transposed convolution."""
    device = torch.device("cpu")
    m: Tensor = torch.atleast_1d(torch.tensor(in_size, dtype=torch.int, device=device))
    k: Tensor = torch.atleast_1d(torch.tensor(kernel_size, dtype=torch.int, device=device))
    s: Tensor = torch.atleast_1d(torch.tensor(stride, dtype=torch.int, device=device))
    p: Tensor = torch.atleast_1d(torch.tensor(padding, dtype=torch.int, device=device))
    o: Tensor = torch.atleast_1d(torch.tensor(output_padding, dtype=torch.int, device=device))
    d: Tensor = torch.atleast_1d(torch.tensor(dilation, dtype=torch.int, device=device))
    if m.ndim != 1:
        raise ValueError("conv_transposed_output_size() 'in_size' must be scalar or sequence")
    ndim = m.shape[0]
    if ndim == 1 and k.shape[0] > 1:
        ndim = k.shape[0]
    for arg, name in zip(
        [k, s, p, o, d], ["kernel_size", "stride", "padding", "output_padding", "dilation"]
    ):
        if arg.ndim != 1 or arg.shape[0] not in (1, ndim):
            raise ValueError(
                f"conv_transposed_output_size() {name!r} must be scalar or sequence of length {ndim}"
            )
    k = k.expand(ndim)
    s = s.expand(ndim)
    p = p.expand(ndim)
    o = o.expand(ndim)
    d = d.expand(ndim)
    n = m.sub(1).mul_(s).sub_(p.mul(2)).add_(k.sub(1).mul_(d)).add_(o).add_(1)
    if isinstance(in_size, int):
        return n.item()
    return Size(n.tolist())


def pad_output_size(
    in_size: ScalarOrTuple[int],
    padding: ScalarOrTuple[int] = 0,
) -> ScalarOrTuple[int]:
    r"""Calculate spatial size of output tensor after padding."""
    device = torch.device("cpu")
    m: Tensor = torch.atleast_1d(torch.tensor(in_size, dtype=torch.int, device=device))
    p: Tensor = torch.atleast_1d(torch.tensor(padding, dtype=torch.int, device=device))
    if m.ndim != 1:
        raise ValueError("pad_output_size() 'in_size' must be scalar or sequence")
    ndim = m.shape[0]
    if ndim == 1 and p.shape[0] > 1 and p.shape[0] % 2:
        ndim = p.shape[0] // 2
    if p.ndim != 1 or p.shape[0] not in (1, 2 * ndim):
        raise ValueError(
            f"pad_output_size() 'padding' must be scalar or sequence of length {2 * ndim}"
        )
    p = p.expand(2 * ndim)
    n = p.reshape(ndim, 2).sum(dim=1).add(m)
    if isinstance(in_size, int):
        return n[0].item()
    return Size(n.tolist())


def pool_output_size(
    in_size: ScalarOrTuple[int],
    kernel_size: ScalarOrTuple[int],
    stride: ScalarOrTuple[int] = 1,
    padding: ScalarOrTuple[int] = 0,
    dilation: ScalarOrTuple[int] = 1,
    ceil_mode: bool = False,
) -> ScalarOrTuple[int]:
    r"""Calculate spatial size of output tensor after pooling."""
    device = torch.device("cpu")
    m: Tensor = torch.atleast_1d(torch.tensor(in_size, dtype=torch.int, device=device))
    k: Tensor = torch.atleast_1d(torch.tensor(kernel_size, dtype=torch.int, device=device))
    s: Tensor = torch.atleast_1d(torch.tensor(stride, dtype=torch.int, device=device))
    p: Tensor = torch.atleast_1d(torch.tensor(padding, dtype=torch.int, device=device))
    d: Tensor = torch.atleast_1d(torch.tensor(dilation, dtype=torch.int, device=device))
    if m.ndim != 1:
        raise ValueError("pool_output_size() 'in_size' must be scalar or sequence")
    ndim = m.shape[0]
    if ndim == 1 and k.shape[0] > 1:
        ndim = k.shape[0]
    for arg, name in zip([k, s, p, d], ["kernel_size", "stride", "padding", "dilation"]):
        if arg.ndim != 1 or arg.shape[0] not in (1, ndim):
            raise ValueError(
                f"pool_output_size() {name!r} must be scalar or sequence of length {ndim}"
            )
    k = k.expand(ndim)
    s = s.expand(ndim)
    p = p.expand(ndim)
    d = d.expand(ndim)
    n = p.mul(2).add_(m).sub_(k.sub(1).mul_(d)).sub_(1).float().div_(s).add_(1)
    n = n.ceil() if ceil_mode else n.floor()
    n = n.long()
    if isinstance(in_size, int):
        return n[0].item()
    return Size(n.tolist())


def unpool_output_size(
    in_size: ScalarOrTuple[int],
    kernel_size: ScalarOrTuple[int],
    stride: ScalarOrTuple[int] = 1,
    padding: ScalarOrTuple[int] = 0,
) -> ScalarOrTuple[int]:
    r"""Calculate spatial size of output tensor after unpooling."""
    device = torch.device("cpu")
    m: Tensor = torch.atleast_1d(torch.tensor(in_size, dtype=torch.int, device=device))
    k: Tensor = torch.atleast_1d(torch.tensor(kernel_size, dtype=torch.int, device=device))
    s: Tensor = torch.atleast_1d(torch.tensor(stride, dtype=torch.int, device=device))
    p: Tensor = torch.atleast_1d(torch.tensor(padding, dtype=torch.int, device=device))
    if m.ndim != 1:
        raise ValueError("unpool_output_size() 'in_size' must be scalar or sequence")
    ndim = m.shape[0]
    if ndim == 1 and k.shape[0] > 1:
        ndim = k.shape[0]
    for arg, name in zip([k, s, p], ["kernel_size", "stride", "padding"]):
        if arg.ndim != 1 or arg.shape[0] not in (1, ndim):
            raise ValueError(
                f"unpool_output_size() {name!r} must be scalar or sequence of length {ndim}"
            )
    k = k.expand(ndim)
    s = s.expand(ndim)
    p = p.expand(ndim)
    n = m.sub(1).mul_(s).sub_(p.mul(2)).add(k)
    if isinstance(in_size, int):
        return n[0].item()
    return Size(n.tolist())


@overload
def same_padding(kernel_size: int, dilation: int = 1) -> int:
    ...


@overload
def same_padding(kernel_size: Sequence[int], dilation: int = 1) -> Tuple[int, ...]:
    ...


@overload
def same_padding(kernel_size: int, dilation: Sequence[int] = 1) -> Tuple[int, ...]:
    ...


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
    device = torch.device("cpu")
    k: Tensor = torch.atleast_1d(torch.tensor(kernel_size, dtype=torch.int, device=device))
    d: Tensor = torch.atleast_1d(torch.tensor(dilation, dtype=torch.int, device=device))
    if k.ndim != 1:
        raise ValueError("same_padding() 'kernel_size' must be scalar or sequence")
    ndim = k.shape[0]
    if ndim == 1 and d.shape[0] > 1:
        ndim = d.shape[0]
    for arg, name in zip([k, d], ["kernel_size", "dilation"]):
        if arg.ndim != 1 or arg.shape[0] not in (1, ndim):
            raise ValueError(f"same_padding() {name!r} must be scalar or sequence of length {ndim}")
    if k.sub(1).mul(d).fmod(2).eq(1).any():
        raise NotImplementedError(
            f"Same padding not available for kernel_size={tuple(k.tolist())} and dilation={tuple(d.tolist())}."
        )
    p = k.sub(1).div(2).mul(d).type(torch.int)
    if isinstance(kernel_size, int) and isinstance(dilation, int):
        return p[0].item()
    return tuple(p.tolist())


@overload
def stride_minus_kernel_padding(kernel_size: int, stride: int) -> int:
    ...


@overload
def stride_minus_kernel_padding(kernel_size: Sequence[int], stride: int) -> Tuple[int, ...]:
    ...


@overload
def stride_minus_kernel_padding(kernel_size: int, stride: Sequence[int]) -> Tuple[int, ...]:
    ...


def stride_minus_kernel_padding(
    kernel_size: ScalarOrTuple[int], stride: ScalarOrTuple[int]
) -> ScalarOrTuple[int]:
    # Adapted from Project MONAI
    # https://github.com/Project-MONAI/MONAI/blob/db8f7877da06a9b3710071c626c0488676716be1/monai/networks/layers/convutils.py
    device = torch.device("cpu")
    k: Tensor = torch.atleast_1d(torch.tensor(kernel_size, dtype=torch.int, device=device))
    s: Tensor = torch.atleast_1d(torch.tensor(stride, dtype=torch.int, device=device))
    if k.ndim != 1:
        raise ValueError("stride_minus_kernel_padding() 'kernel_size' must be scalar or sequence")
    ndim = k.shape[0]
    if ndim == 1 and s.shape[0] > 1:
        ndim = s.shape[0]
    for arg, name in zip([k, s], ["kernel_size", "stride"]):
        if arg.ndim != 1 or arg.shape[0] not in (1, ndim):
            raise ValueError(f"stride_minus_kernel_padding() {name!r} must be scalar or sequence of length {ndim}")
    assert k.ndim == 1, "stride_minus_kernel_padding() 'kernel_size' must be scalar or sequence"
    assert s.ndim == 1, "stride_minus_kernel_padding() 'stride' must be scalar or sequence"
    p = s.sub(k).type(torch.int)
    if isinstance(kernel_size, int) and isinstance(stride, int):
        return p[0].item()
    return tuple(p.tolist())


def upsample_padding(
    kernel_size: ScalarOrTuple[int], scale_factor: ScalarOrTuple[int]
) -> Tuple[int, ...]:
    r"""Padding on both sides for transposed convolution."""
    device = torch.device("cpu")
    k: Tensor = torch.atleast_1d(torch.tensor(kernel_size, dtype=torch.int, device=device))
    s: Tensor = torch.atleast_1d(torch.tensor(scale_factor, dtype=torch.int, device=device))
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
    device = torch.device("cpu")
    k: Tensor = torch.atleast_1d(torch.tensor(kernel_size, dtype=torch.int, device=device))
    s: Tensor = torch.atleast_1d(torch.tensor(scale_factor, dtype=torch.int, device=device))
    p: Tensor = torch.atleast_1d(torch.tensor(padding, dtype=torch.int, device=device))
    assert k.ndim == 1, "upsample_output_padding() 'kernel_size' must be scalar or sequence"
    assert s.ndim == 1, "upsample_output_padding() 'scale_factor' must be scalar or sequence"
    assert p.ndim == 1, "upsample_output_padding() 'padding' must be scalar or sequence"
    op = p.mul(2).sub(k).add(s).type(torch.int)
    if op.lt(0).any():
        raise ValueError(
            "upsample_output_padding() 'output_padding' must be greater than or equal to zero"
        )
    return tuple(op.tolist())


def upsample_output_size(
    in_size: ScalarOrTuple[int],
    size: Optional[ScalarOrTuple[int]] = None,
    scale_factor: Optional[ScalarOrTuple[float]] = None,
) -> ScalarOrTuple[int]:
    r"""Calculate spatial size of output tensor after unpooling."""
    if size is not None and scale_factor is not None:
        raise ValueError("upsample_output_size() 'size' and 'scale_factor' are mutually exclusive")
    device = torch.device("cpu")
    m: Tensor = torch.atleast_1d(torch.tensor(in_size, dtype=torch.int, device=device))
    if m.ndim != 1:
        raise ValueError("upsample_output_size() 'in_size' must be scalar or sequence")
    ndim = m.shape[0]
    if size is not None:
        s: Tensor = torch.atleast_1d(torch.tensor(size, dtype=torch.int, device=device))
        if s.ndim != 1 or s.shape[0] not in (1, ndim):
            raise ValueError(
                f"upsample_output_size() 'size' must be scalar or sequence of length {ndim}"
            )
        n = s.expand(ndim)
    elif scale_factor is not None:
        s: Tensor = torch.atleast_1d(torch.tensor(scale_factor, dtype=torch.int, device=device))
        if s.ndim != 1 or s.shape[0] not in (1, ndim):
            raise ValueError(
                f"upsample_output_size() 'scale_factor' must be scalar or sequence of length {ndim}"
            )
        n = m.float().mul(s).floor().long()
    else:
        n = m
    if isinstance(in_size, int):
        return n.item()
    return Size(n.tolist())
