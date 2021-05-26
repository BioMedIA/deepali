from typing import Optional, Sequence, Union

import torch
from torch import Tensor

from .image import conv1d


def subdivide_cubic_bspline(
    data: Tensor, dims: Optional[Union[int, Sequence[int]]] = None
) -> Tensor:
    r"""Compute cubic B-spline coefficients for subdivided control point grid.

    Args:
        data: Input control point coefficients as tensor of shape ``(N, C, ..., X)``.
        dims: Tensor dimensions along which to subdivide.

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
        dims = tuple(range(2, data.ndim))
    output = data
    kernel_1 = torch.tensor([0.125, 0.75, 0.125], dtype=data.dtype, device=data.device)
    kernel_2 = torch.tensor([0.5, 0.5], dtype=data.dtype, device=data.device)
    for dim in dims:
        if dim < 2:
            raise ValueError("subdivide_cubic_bspline() 'dims' must be greater than 1")
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
