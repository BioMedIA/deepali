r"""Functions for B-spline interpolation."""

from typing import Callable, Optional, Sequence, Tuple, Union, overload

import torch
from torch import Size, Tensor
from torch.nn import functional as F

from .enum import PaddingMode, SpatialDim, SpatialDimArg
from .image import conv, conv1d
from .kernels import cubic_bspline1d
from .tensor import move_dim
from .types import ScalarOrTuple


@overload
def cubic_bspline_control_point_grid_size(size: int, stride: int) -> int:
    ...


@overload
def cubic_bspline_control_point_grid_size(size: Sequence[int], stride: int) -> Tuple[int, ...]:
    ...


@overload
def cubic_bspline_control_point_grid_size(size: int, stride: Sequence[int]) -> Tuple[int, ...]:
    ...


@overload
def cubic_bspline_control_point_grid_size(
    size: Sequence[int], stride: Sequence[int]
) -> Tuple[int, ...]:
    ...


def cubic_bspline_control_point_grid_size(
    size: ScalarOrTuple[int], stride: ScalarOrTuple[int]
) -> ScalarOrTuple[int]:
    r"""Calculate required number of cubic B-spline coefficients for given output size."""
    device = torch.device("cpu")
    m: Tensor = torch.atleast_1d(torch.tensor(size, dtype=torch.int, device=device))
    s: Tensor = torch.atleast_1d(torch.tensor(stride, dtype=torch.int, device=device))
    if m.ndim != 1:
        raise ValueError(
            "cubic_bspline_control_point_grid_size() 'size' must be scalar or sequence"
        )
    if m.le(0).any():
        raise ValueError("cubic_bspline_control_point_grid_size() 'size' must be positive")
    if s.le(0).any():
        raise ValueError("cubic_bspline_control_point_grid_size() 'stride' must be positive")
    ndim = m.shape[0]
    if ndim == 1 and s.shape[0] > 1:
        ndim = s.shape[0]
    for arg, name in zip([m, s], ["size", "stride"]):
        if arg.ndim != 1 or arg.shape[0] not in (1, ndim):
            raise ValueError(
                f"cubic_bspline_control_point_grid_size() {name!r} must be scalar or sequence of length {ndim}"
            )
    m = m.expand(ndim)
    s = s.expand(ndim)
    n = m.div(s, rounding_mode="floor").add_(3)
    n = n.where(m % s == 0, n.add(1))
    if isinstance(size, int) and isinstance(stride, int):
        return n[0].item()
    return Size(n.tolist())


@overload
def bspline_interpolation_weights(
    degree: int,
    stride: int,
    dtype: Optional[torch.dtype] = None,
    device: Optional[torch.device] = None,
) -> Tensor:
    ...


@overload
def bspline_interpolation_weights(
    degree: int,
    stride: Sequence[int],
    dtype: Optional[torch.dtype] = None,
    device: Optional[torch.device] = None,
) -> Tuple[Tensor, ...]:
    ...


def bspline_interpolation_weights(
    degree: int,
    stride: ScalarOrTuple[int],
    dtype: Optional[torch.dtype] = None,
    device: Optional[torch.device] = None,
) -> Union[Tensor, Tuple[Tensor, ...]]:
    r"""Compute B-spline interpolation weights."""
    # Adapted from MIRTK ComputeBSplineIndicesAndWeights() at
    # https://github.com/BioMedIA/MIRTK/blob/877917b1b3ec9e602f7395dbbb1270e4334fc311/Modules/Numerics/include/mirtk/BSpline.h#L670-L789
    kernels = {}
    return_single_tensor = False
    if isinstance(stride, int):
        stride = [stride]
        return_single_tensor = True
    for s in stride:
        if s in kernels:
            continue
        kernel = torch.empty((s, degree + 1), dtype=dtype, device=device)
        offset = torch.arange(0, 1, 1 / s, dtype=kernel.dtype, device=kernel.device)
        offset = offset.sub(offset.floor() if degree & 1 else offset.round())
        if degree == 2:
            kernel[:, 1] = 0.75 - offset.square()
            kernel[:, 2] = offset.sub(kernel[:, 1]).add_(1).mul_(0.5)
            kernel[:, 0] = -kernel[:, [1, 2]].sum(1).sub(1)
        elif degree == 3:
            kernel[:, 3] = offset.pow(3).mul_(1 / 6)
            kernel[:, 0] = offset.mul(offset.sub(1)).mul_(0.5).add_(1 / 6).sub_(kernel[:, 3])
            kernel[:, 2] = offset.add(kernel[:, 0]).sub_(kernel[:, 3].mul(2))
            kernel[:, 1] = -kernel[:, [0, 2, 3]].sum(1).sub(1)
        elif degree == 4:
            # MIRTK code variable names: w=offset, w2=a
            a = offset.square()
            t = a.mul(1 / 6)
            t0 = t.sub(11 / 24).mul(offset)
            t1 = t.sub(0.25).mul_(-a).add_(19 / 96)
            kernel[:, 0] = torch.tensor(0.5, dtype=dtype, device=device).sub(offset).square()
            kernel[:, 0] = kernel[:, 0].mul(kernel[:, 0].mul(1 / 24))
            kernel[:, 1] = t1.add(t0)
            kernel[:, 3] = t1.sub(t0)
            kernel[:, 4] = offset.mul(0.5).add_(kernel[:, 0]).add_(t0)
            kernel[:, 2] = -kernel[:, [0, 1, 3, 4]].sum(1).sub(1)
        elif degree == 5:
            # MIRTK code variable names: w=offset, w2=a, w4=b
            a = offset.square()
            kernel[:, 5] = offset.mul(a.square()).mul_(1 / 120)
            a = a.sub_(offset)
            b = a.square()
            offset = offset.sub_(0.5)
            t = a.sub(3).mul_(a)
            kernel[:, 0] = a.add(b).add_(1 / 5).mul_(1 / 24).sub_(kernel[:, 5])
            t0 = a.sub(5).mul_(a).add_(46 / 5).mul_(1 / 24)
            t1 = t.add(4).mul_(offset).mul_(-1 / 12)
            kernel[:, 2] = t0.add(t1)
            kernel[:, 3] = t0.sub(t1)
            t0 = t.sub(9 / 5).mul_(1.0 / 16.0)
            t1 = b.sub(a).sub_(5).mul_(offset).mul_(1.0 / 24.0)
            kernel[:, 1] = t0.add(t1)
            kernel[:, 4] = t0.sub(t1)
        else:
            raise NotImplementedError(f"B-spline interpolation for degree={degree}")
        kernels[s] = kernel
    kernels = tuple(kernels[s] for s in stride)
    if return_single_tensor:
        assert len(kernels) == 1
        return kernels[0]
    return kernels


def evaluate_cubic_bspline(
    data: Tensor,
    stride: ScalarOrTuple[int],
    size: Optional[Size] = None,
    shape: Optional[Size] = None,
    kernel: Optional[Union[Tensor, Sequence[Tensor]]] = None,
    transpose: bool = False,
) -> Tensor:
    r"""Evaluate cubic B-spline function.

    Args:
        data: Cubic B-spline interpolation coefficients as tensor of shape ``(N, C, ..., X)``.
        stride: Number of output grid points between control points plus one. This is the stride of the
            transposed convolution used to upsample the control point displacements to the output size.
            If a sequence of values is given, these must be the strides for the different spatial
            dimensions in the order ``(sx, ...)``.
        size: Spatial size of output tensor in the order ``(nx, ...).
        shape: Spatial size of output tensor in the order ``(..., nx)``.
        kernel: Precomputed cubic B-spline interpolation kernel. When multiple 1D kernels are given,
            these must be in the order ``(kx, ...)``.
        transpose: Whether to use separable transposed convolution as implemented in AIRLab.
            When ``False``, a more efficient implementation using multi-channel convolution followed
            by a reshuffling of the output is performed. This more efficient and also more accurate
            implementation is adapted from the C++ code of MIRTK (``mirtk::BSplineInterpolateImageFunction``).

    Returns:
        Cubic B-spline function values as tensor of shape ``(N, C, ..., X')``, where ``X' = sx * X``
        when neither output ``size`` nor ``shape`` is specified. Otherwise, the output tensor is cropped
        to the requested spatial output size.

    """
    if not isinstance(data, Tensor):
        raise TypeError("evaluate_cubic_bspline() 'data' must be torch.Tensor")
    if not torch.is_floating_point(data):
        raise TypeError("evaluate_cubic_bspline() 'data' must have floating point dtype")
    if data.ndim < 3:
        raise ValueError("evaluate_cubic_bspline() 'data' must have shape (N, C, ..., X)")
    if size is not None:
        if shape is not None:
            raise ValueError("evaluate_cubic_bspline() 'size' and 'shape' are mutually exclusive")
        shape = Size(reversed(size))
    D = data.ndim - 2
    N = data.shape[0]
    C = data.shape[1]
    if isinstance(stride, int):
        stride = [stride] * D
    # Implementation inspired by AIRLab
    if transpose:
        if kernel is None:
            kernels = {}
            for s in stride:
                if s not in kernels:
                    kernels[s] = cubic_bspline1d(s)
            kernel = [kernels[s] for s in stride]
        stride = tuple(reversed(stride))
        if isinstance(kernel, Sequence):
            kernel = tuple(reversed(kernel))
        output = conv(data, kernel=kernel, stride=stride, padding=PaddingMode.ZEROS, transpose=True)
        if shape is not None:
            output = output[
                (slice(0, N), slice(0, C)) + tuple(slice(s, s + n) for s, n in zip(stride, shape))
            ]
    # Implementation adapted from MIRTK
    else:
        if kernel is None:
            kernel = bspline_interpolation_weights(
                degree=3, stride=stride, dtype=data.dtype, device=data.device
            )
        output = data
        dims = tuple(SpatialDim(dim).tensor_dim(data.ndim) for dim in range(D))
        conv_fn: Callable[..., Tensor] = [F.conv1d, F.conv2d, F.conv3d][D - 1]
        for dim, w in zip(dims, kernel):
            weight = w.reshape((w.shape[0], 1, w.shape[1]) + (1,) * (D - 1))
            weight = weight.tile((C,) + (1,) * (weight.ndim - 1))
            output = move_dim(output, dim, 2)
            output = conv_fn(output, weight, groups=C)
            output = output.reshape((N, C, w.shape[0]) + (output.shape[2:]))
            output = output.transpose(2, 3).flatten(2, 3)
            output = move_dim(output, 2, dim)
        if shape is not None:
            output = output[(slice(0, N), slice(0, C)) + tuple(slice(0, n) for n in shape)]
    return output


def subdivide_cubic_bspline(
    data: Tensor, dims: Optional[Union[SpatialDimArg, Sequence[SpatialDimArg]]] = None
) -> Tensor:
    r"""Compute cubic B-spline coefficients for subdivided control point grid.

    Args:
        data: Input control point coefficients as tensor of shape ``(N, C, ..., X)``.
        dims: Spatial dimensions along which to subdivide.

    Returns:
        Coefficients of subdivided cubic B-spline function.

    """
    if not isinstance(data, Tensor):
        raise TypeError("subdivide_cubic_bspline() 'data' must be torch.Tensor")
    if not torch.is_floating_point(data):
        raise TypeError("subdivide_cubic_bspline() 'data' must have floating point dtype")
    if data.ndim < 4:
        raise ValueError("subdivide_cubic_bspline() 'data' must have shape (N, C, ..., X)")
    if dims is None:
        dims = tuple(range(data.ndim - 2))
    elif isinstance(dims, (int, str)):
        dims = [dims]
    elif not isinstance(dims, Sequence):
        raise TypeError("subdivide_cubic_bspline() 'dims' must be int, str, or Sequence thereof")
    dims = sorted(SpatialDim.from_arg(dim).tensor_dim(data.ndim) for dim in dims)
    output = data
    kernel_1 = torch.tensor([0.125, 0.75, 0.125], dtype=data.dtype, device=data.device)
    kernel_2 = torch.tensor([0.5, 0.5], dtype=data.dtype, device=data.device)
    for dim in dims:
        # Allocate tensor for subdivided control point coefficients
        shape = output.shape[:dim] + (2 * output.shape[dim] - 1,) + output.shape[dim + 1 :]
        temp = torch.empty(shape, dtype=data.dtype, device=data.device)
        # Evaluate coefficients at original control point positions
        indices = [slice(0, n) for n in shape]
        indices[dim] = slice(0, shape[dim], 2)
        temp[indices] = conv1d(output, kernel_1, dim=dim, padding=1)
        # Evaluate coefficients at subdivided intermediate positions
        indices = [slice(0, n) for n in shape]
        indices[dim] = slice(1, shape[dim], 2)
        temp[indices] = conv1d(output, kernel_2, dim=dim, padding=0)
        output = temp
    return output
