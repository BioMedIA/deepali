r"""Low-level tensor utility functions."""

from logging import Logger
from typing import Callable, Optional, Tuple, Union

import torch
from torch import Tensor

from .typing import Array, Device, DType, Scalar


def as_tensor(
    arg: Union[Scalar, Array], dtype: Optional[DType] = None, device: Optional[Device] = None
) -> Tensor:
    r"""Create tensor from array if argument is not of type torch.Tensor.

    Unlike ``torch.as_tensor()``, this function preserves the tensor device if ``device=None``.

    """
    if device is None and isinstance(arg, Tensor):
        device = arg.device
    return torch.as_tensor(arg, dtype=dtype, device=device)  # type: ignore


def as_float_tensor(arr: Array) -> Tensor:
    r"""Create tensor with floating point type from argument if it is not yet."""
    arr_ = as_tensor(arr)
    if not torch.is_floating_point(arr_):
        return arr_.type(torch.float)
    return arr_


def as_one_hot_tensor(
    tensor: Tensor,
    num_classes: int,
    ignore_index: Optional[int] = None,
    dtype: Optional[DType] = None,
) -> Tensor:
    r"""Converts label image to one-hot encoding of multi-class segmentation.

    Adapted from: https://github.com/wolny/pytorch-3dunet

    Args:
        tensor: Input tensor of shape ``(N, 1, ..., X)`` or ``(N, C, ..., X)``.
            When a tensor with ``C == num_classes`` is given, it is converted to the specified
            ``dtype`` but not modified otherwise. Otherwise, the input tensor must contain
            class labels in a single channel.
        num_classes: Number of channels/labels.
        ignore_index: Ignore index to be kept during the expansion. The locations of the index
            value in the GT image is stored in the corresponding locations across all channels so
            that this location can be ignored across all channels later e.g. in Dice computation.
            This argument must be ``None`` if ``tensor`` has ``C == num_channels``.
        dtype: Data type of output tensor. Default is ``torch.float``.

    Returns:
        Output tensor of shape ``(N, C, ..., X)``.

    """
    if dtype is None:
        dtype = torch.float
    if not isinstance(tensor, Tensor):
        raise TypeError("as_one_hot_tensor() 'tensor' must be torch.Tensor")
    if tensor.dim() < 3:
        raise ValueError("as_one_hot_tensor() 'tensor' must have shape (N, C, ..., X)")
    if tensor.shape[1] == num_classes:
        return tensor.to(dtype=dtype)
    elif tensor.shape[1] != 1:
        raise ValueError(
            f"as_one_hot_tensor() 'tensor' must have shape (N, 1|{num_classes}, ..., X)"
        )
    # create result tensor shape (NxCxDxHxW)
    shape = list(tensor.shape)
    shape[1] = num_classes
    # scatter to get the one-hot tensor
    if ignore_index is None:
        return torch.zeros(shape, dtype=dtype).to(tensor.device).scatter_(1, tensor, 1)
    # create ignore_index mask for the result
    mask = tensor.expand(shape) == ignore_index
    # clone the src tensor and zero out ignore_index in the inputs
    inputs = tensor.clone()
    inputs[inputs == ignore_index] = 0
    # scatter to get the one-hot tensor
    result = torch.zeros(shape, dtype=dtype).to(inputs.device).scatter_(1, inputs, 1)
    # bring back the ignore_index in the result
    result[mask] = ignore_index
    return result


def atleast_1d(
    arr: Array, dtype: Optional[DType] = None, device: Optional[Device] = None
) -> Tensor:
    r"""Convert array-like argument to 1- or more-dimensional PyTorch tensor."""
    arr_ = as_tensor(arr, dtype=dtype, device=device)
    return arr_.unsqueeze(0) if arr_.ndim == 0 else arr_


def cat_scalars(
    arg: Union[Scalar, Array],
    *args: Scalar,
    num: int = 0,
    dtype: Optional[DType] = None,
    device: Optional[Device] = None,
) -> Tensor:
    r"""Join arguments into single 1-dimensional tensor.

    This auxiliary function is used by ``Grid``, ``Image``, and ``ImageBatch`` to support
    method arguments for different spatial dimensions as either scalar constant, list
    of scalar ``*args``, or single ``Array`` argument. If a single argument of type ``Array``
    is given, it must be a sequence of scalar values.

    Args:
        arg: Either a single scalar or sequence of scalars. If the argument is a ``Tensor``,
            it is cloned and detached in order to avoid unintended side effects.
        args: Additional scalars. If ``arg`` is a sequence, ``args`` must be empty.
        num: Number of expected scalar values. If a single scalar ``arg`` is given,
            it is repeated ``num`` times to create a 1-dimensional array. If ``num=0``,
            the length of the returned array corresponds to the number of given scalars.
        dtype: Data type of output tensor.
        device: Device on which to store tensor.

    Returns:
        Scalar arguments joined into a 1-dimensional tensor.

    """
    if args:
        if isinstance(arg, (tuple, list)) or isinstance(arg, Tensor):
            raise ValueError("arg and args must either be all scalars, or args empty")
        arg = torch.tensor((arg,) + args, dtype=dtype, device=device)
    else:
        arg = as_tensor(arg, dtype=dtype, device=device)
    if arg.ndim == 0:
        arg = arg.unsqueeze(0)
    if arg.ndim != 1:
        if num > 0:
            raise ValueError(f"Expected one scalar, a sequence of length {num}, or {num} args")
        raise ValueError("Expected one scalar, a sequence of scalars, or multiple scalars")
    if num > 0:
        if len(arg) == 1:
            arg = arg.repeat(num)
        elif len(arg) != num:
            raise ValueError(f"Expected one scalar, a sequence of length {num}, or {num} args")
    return arg


def batched_index_select(input: Tensor, dim: int, index: Tensor) -> Tensor:
    r"""Batched version of torch.index_select().

    See https://discuss.pytorch.org/t/batched-index-select/9115/9.

    """
    for i in range(1, len(input.shape)):
        if i != dim:
            index = index.unsqueeze(i)
    shape = list(input.shape)
    shape[0] = -1
    shape[dim] = -1
    index = index.expand(shape)
    return torch.gather(input, dim, index)


def move_dim(tensor: Tensor, dim: int, pos: int) -> Tensor:
    r"""Move the specified tensor dimension to another position."""
    if dim < 0:
        dim = tensor.ndim + dim
    if pos < 0:
        pos = tensor.ndim + pos
    if pos == dim:
        return tensor
    if dim < pos:
        pos += 1
    tensor = tensor.unsqueeze(pos)
    if pos <= dim:
        dim += 1
    tensor = tensor.transpose(dim, pos).squeeze(dim)
    return tensor


def unravel_coords(indices: Tensor, size: Tuple[int, ...]) -> Tensor:
    r"""Converts flat indices into unraveled grid coordinates.

    Args:
        indices: A tensor of flat indices with shape ``(..., N)``.
        size: Sampling grid size with order ``(X, ...)``.

    Returns:
        Grid coordinates of corresponding grid points.

    """
    size = torch.Size(size)
    numel = size.numel()
    if indices.ge(numel).any():
        raise ValueError(f"unravel_coords() indices must be smaller than {numel}")
    coords = torch.zeros(indices.size() + (len(size),), dtype=indices.dtype, device=indices.device)
    for i, n in enumerate(size):
        coords[..., i] = indices % n
        indices = indices // n
    return coords


def unravel_index(indices: Tensor, shape: Tuple[int, ...]) -> Tensor:
    r"""Converts flat indices into unraveled coordinates in a target shape.

    This is a `torch` implementation of `numpy.unravel_index`, but returning a
    tensor of shape (..., N, D) rather than a D-dimensional tuple. See also
    https://github.com/pytorch/pytorch/issues/35674#issuecomment-739492875.

    Args:
        indices: A tensor of indices with shape (..., N).
        shape: The targeted tensor shape of length D.

    Returns:
        Unraveled coordinates as tensor of shape (..., N, D) with coordinates
        in the same order as the input ``shape`` dimensions.

    """
    shape = torch.Size(shape)
    numel = shape.numel()
    if indices.ge(numel).any():
        raise ValueError(f"unravel_coords() indices must be smaller than {numel}")
    coords = torch.zeros(indices.size() + (len(shape),), dtype=indices.dtype, device=indices.device)
    for i, n in enumerate(reversed(shape)):
        coords[..., i] = indices % n
        indices = indices // n
    return coords.flip(-1)


def log_grad_hook(name: str, logger: Optional[Logger] = None) -> Callable[[Tensor], None]:
    r"""Backward hook to print tensor gradient information for debugging."""

    def printer(grad: Tensor) -> None:
        if grad.nelement() == 1:
            msg = f"{name}.grad: value={grad}"
        else:
            msg = (
                f"{name}.grad: shape={tuple(grad.shape)}"
                f", max={grad.max()}, min={grad.min()}"
                f", mean={grad.mean()}"
            )
        if logger is None:
            print(msg)
        else:
            logger.debug(msg)

    return printer


def register_backward_hook(
    tensor: Tensor, hook: Callable[[Tensor], None], retain_grad: bool = False
) -> Tensor:
    r"""Register backward hook and optionally enable retaining gradient."""
    if tensor.requires_grad:
        if retain_grad:
            tensor.retain_grad()
        tensor.register_hook(hook)
    return tensor
