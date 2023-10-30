r"""Predefined convolutional kernels."""

import math
from typing import Optional, Sequence, Union

import torch
from torch import Tensor
import torch.nn.functional as F

from .linalg import tensordot
from .tensor import as_tensor, cat_scalars
from .typing import Array, Device, DType, Scalar


def bspline1d(stride: int, order: int = 4) -> Tensor:
    r"""B-spline kernel of given order for specified control point spacing.

    Implementation adopted from AirLab:
    https://github.com/airlab-unibas/airlab/blob/80c9d487c012892c395d63c6d937a67303c321d1/airlab/utils/kernelFunction.py#L218

    This function computes the kernel recursively by convolving with a box filter (cf. Cox-de Boor's recursion formula).
    The resulting kernel differs from the analytic B-spline function. This may be due to the box filter having extend
    to the borders of the pixels, where it should drop to zero at pixel centers rather.

    The exact B-spline kernel of order 4 is computed by ``cubic_bspline1d()``.

    Args:
        stride: Spacing between control points with respect to original (upsampled) image grid.
        order: Order of B-spline kernel, where the degree of the spline polynomials is order minus 1.

    Returns:
        B-spline convolution kernel.

    """
    kernel = kernel_ones = torch.ones(1, 1, stride, dtype=torch.float)
    for _ in range(1, order + 1):
        kernel = F.conv1d(kernel, kernel_ones, padding=stride - 1) / stride
    return kernel.reshape(-1)


def cubic_bspline_value(x: float, derivative: int = 0) -> float:
    r"""Evaluate 1-dimensional cubic B-spline."""
    t = abs(x)
    # outside local support region
    if t >= 2:
        return 0
    # 0-th order derivative
    if derivative == 0:
        if t < 1:
            return 2 / 3 + (0.5 * t - 1) * t**2
        return -((t - 2) ** 3) / 6
    # 1st order derivative
    if derivative == 1:
        if t < 1:
            return (1.5 * t - 2.0) * x
        if x < 0:
            return 0.5 * (t - 2) ** 2
        return -0.5 * (t - 2) ** 2
    # 2nd oder derivative
    if derivative == 2:
        if t < 1:
            return 3 * t - 2
        return -t + 2


def cubic_bspline(
    stride: Union[int, Sequence[int]],
    *args: int,
    derivative: int = 0,
    dtype: Optional[DType] = None,
    device: Optional[Device] = None,
):
    r"""Get n-dimensional cubic B-spline kernel.

    Args:
        stride: Spacing between control points with respect to original (upsampled) image grid.
        derivative: Order of cubic B-spline derivative.

    Returns:
        Cubic B-spline convolution kernel.

    """
    stride_ = cat_scalars(
        stride,
        *args,
        derivative=derivative,
        dtype=torch.int32,
        device=torch.device("cpu"),
    ).tolist()
    D = len(stride_)
    if D == 1:
        return cubic_bspline1d(stride_, derivative=derivative, dtype=dtype, device=device)
    if D == 2:
        return cubic_bspline2d(stride_, derivative=derivative, dtype=dtype, device=device)
    if D == 3:
        return cubic_bspline3d(stride_, derivative=derivative, dtype=dtype, device=device)
    raise NotImplementedError(f"cubic_bspline() {D}-dimensional kernel")


def cubic_bspline1d(
    stride: Union[int, Sequence[int]],
    derivative: int = 0,
    dtype: Optional[DType] = None,
    device: Optional[Device] = None,
) -> Tensor:
    r"""Cubic B-spline kernel for specified control point spacing.

    Args:
        stride: Spacing between control points with respect to original (upsampled) image grid.
        derivative: Order of cubic B-spline derivative.

    Returns:
        Cubic B-spline convolution kernel.

    """
    if dtype is None:
        dtype = torch.float
    if not isinstance(stride, int):
        (stride,) = stride
    kernel = torch.ones(4 * stride - 1, dtype=torch.float)
    radius = kernel.shape[0] // 2
    for i in range(kernel.shape[0]):
        kernel[i] = cubic_bspline_value((i - radius) / stride, derivative=derivative)
    if device is None:
        device = kernel.device
    return kernel.to(device)


def cubic_bspline2d(
    stride: Union[int, Sequence[int]],
    *args: int,
    derivative: int = 0,
    dtype: Optional[DType] = None,
    device: Optional[Device] = None,
) -> Tensor:
    r"""Cubic B-spline kernel for specified control point spacing.

    Args:
        stride: Spacing between control points with respect to original (upsampled) image grid.
        derivative: Order of cubic B-spline derivative.

    Returns:
        Cubic B-spline convolution kernel.

    """
    if dtype is None:
        dtype = torch.float
    stride_ = cat_scalars(stride, *args, num=2, dtype=torch.int32, device=torch.device("cpu"))
    kernel = torch.ones((4 * stride_ - 1).tolist(), dtype=dtype)
    radius = [n // 2 for n in kernel.shape]
    for j in range(kernel.shape[1]):
        w_j = cubic_bspline_value((j - radius[1]) / stride[1], derivative=derivative)
        for i in range(kernel.shape[0]):
            w_i = cubic_bspline_value((i - radius[0]) / stride[0], derivative=derivative)
            kernel[j, i] = w_i * w_j
    if device is None:
        device = kernel.device
    return kernel.to(device)


def cubic_bspline3d(
    stride: Union[int, Sequence[int]],
    *args: int,
    derivative: int = 0,
    dtype: Optional[DType] = None,
    device: Optional[Device] = None,
) -> Tensor:
    r"""Cubic B-spline kernel for specified control point spacing.

    Args:
        stride: Spacing between control points with respect to original (upsampled) image grid.
        derivative: Order of cubic B-spline derivative.

    Returns:
        Cubic B-spline convolution kernel.

    """
    if dtype is None:
        dtype = torch.float
    stride_ = cat_scalars(stride, *args, num=3, dtype=torch.int32, device=torch.device("cpu"))
    kernel = torch.ones((4 * stride_ - 1).tolist(), dtype=torch.float)
    radius = [n // 2 for n in kernel.shape]
    for k in range(kernel.shape[2]):
        w_k = cubic_bspline_value((k - radius[2]) / stride[2], derivative=derivative)
        for j in range(kernel.shape[1]):
            w_j = cubic_bspline_value((j - radius[1]) / stride[1], derivative=derivative)
            for i in range(kernel.shape[0]):
                w_i = cubic_bspline_value((i - radius[0]) / stride[0], derivative=derivative)
                kernel[k, j, i] = w_i * w_j * w_k
    if device is None:
        device = kernel.device
    return kernel.to(device)


def gaussian_kernel_radius(sigma: Union[Scalar, Array], factor: Scalar = 3) -> Tensor:
    r"""Radius of truncated Gaussian kernel.

    Args:
        sigma: Standard deviation in grid units.
        factor: Number of standard deviations at which to truncate.

    Returns:
        Radius of truncated Gaussian kernel in grid units.

    """
    sigma = as_tensor(sigma, dtype=torch.float32, device="cpu")
    is_scalar = sigma.ndim == 0
    if is_scalar:
        sigma = sigma.unsqueeze(0)
    if sigma.ndim != 1:
        raise ValueError("gaussian() 'sigma' must be scalar or sequence")
    if sigma.shape[0] == 0:
        raise ValueError("gaussian() 'sigma' must be scalar or non-empty sequence")
    if sigma.lt(0).any():
        raise ValueError("Gaussian standard deviation must be non-negative")
    factor = as_tensor(factor, dtype=sigma.dtype, device=sigma.device)
    radius = sigma.mul(factor).floor().type(torch.int64)
    if is_scalar:
        radius = radius
    return radius


def gaussian(
    sigma: Union[Scalar, Array],
    *args: Scalar,
    normalize: bool = True,
    dtype: Optional[DType] = None,
    device: Optional[Device] = None,
):
    r"""Get n-dimensional Gaussian kernel."""
    sigma_ = cat_scalars(sigma, *args, dtype=dtype, device=torch.device("cpu"))
    if not torch.is_floating_point(sigma_):
        if dtype is not None:
            raise TypeError("Gaussian kernel dtype must be floating point type")
        sigma_ = sigma_.type(torch.float)
    if sigma_.ndim == 0:
        sigma_ = sigma_.unsqueeze(0)
    if sigma_.ndim != 1:
        raise ValueError("gaussian() 'sigma' must be scalar or sequence")
    if sigma_.shape[0] == 0:
        raise ValueError("gaussian() 'sigma' must be scalar or non-empty sequence")
    kernel = gaussian1d(sigma_[0], normalize=False, dtype=torch.float64)
    for std in sigma_[1:]:
        other = gaussian1d(std, normalize=False, dtype=torch.float64)
        kernel = tensordot(kernel, other, dims=0)
    if normalize:
        kernel /= kernel.sum()
    return kernel.to(dtype=sigma_.dtype, device=device)


def gaussian1d(
    sigma: Scalar,
    radius: Optional[Union[int, Tensor]] = None,
    scale: Optional[Scalar] = None,
    normalize: bool = True,
    dtype: Optional[DType] = None,
    device: Optional[Device] = None,
) -> Tensor:
    r"""Get 1-dimensional Gaussian kernel."""
    sigma = as_tensor(sigma, device="cpu")
    if sigma.ndim != 0:
        raise ValueError("gaussian1d() 'sigma' must be scalar")
    if sigma.lt(0):
        raise ValueError("Gaussian standard deviation must be non-negative")
    if dtype is not None and not dtype.is_floating_point:
        raise TypeError("Gaussian kernel dtype must be floating point type")
    if radius is None:
        radius = gaussian_kernel_radius(sigma)
    radius = int(radius)
    if radius > 0:
        size = 2 * radius + 1
        x = torch.linspace(-radius, radius, steps=size, dtype=dtype, device=device)
        sigma = sigma.to(dtype=dtype, device=device)
        kernel = torch.exp(-0.5 * ((x / sigma) ** 2))
        if scale is None:
            scale = 1 / sigma.mul(math.sqrt(2 * math.pi))
        else:
            scale = as_tensor(scale, dtype=dtype, device=device)
        kernel *= scale
        if normalize:
            kernel /= kernel.sum()
    else:
        if scale is None:
            scale = 1
        kernel = as_tensor([scale], dtype=dtype, device=device)
    return kernel


def gaussian1d_I(
    sigma: Scalar,
    normalize: bool = True,
    dtype: Optional[DType] = None,
    device: Optional[Device] = None,
) -> Tensor:
    r"""Get 1st order derivative of 1-dimensional Gaussian kernel."""
    if torch.is_tensor(sigma):
        sigma_ = as_tensor(sigma)  # satisfy static type checker
        if sigma_.ndim != 0:
            raise ValueError("gaussian1d() 'sigma' must be scalar")
        sigma = sigma_.item()
    if sigma < 0:
        raise ValueError("Gaussian standard deviation must be non-negative")
    if dtype is not None and not dtype.is_floating_point:
        raise TypeError("Gaussian kernel dtype must be floating point type")
    radius = int(gaussian_kernel_radius(sigma).item())
    if radius > 0:
        size = 2 * radius + 1
        x = torch.linspace(-radius, radius, steps=size, dtype=dtype, device=device)
        norm = torch.tensor(1 / (sigma * math.sqrt(2 * math.pi)), dtype=dtype, device=device)
        var = sigma**2
        # Note that conv1d() computes correlation, i.e., the kernel is mirrored
        kernel = norm * torch.exp(-0.5 * x**2 / var) * (x / var)
        if normalize:
            kernel /= (norm * torch.exp(-0.5 * x**2 / var)).sum()
    else:
        kernel = torch.tensor([1], dtype=dtype, device=device)
    return kernel


def gaussian2d(sigma: Union[Scalar, Array], *args: Scalar, **kwargs) -> Tensor:
    r"""Get 2-dimensional Gaussian kernel."""
    sigma = cat_scalars(sigma, *args, num=2, device=torch.device("cpu"))
    return gaussian(sigma, **kwargs)


def gaussian3d(sigma: Union[Scalar, Array], *args: Scalar, **kwargs) -> Tensor:
    r"""Get 3-dimensional Gaussian kernel."""
    sigma = cat_scalars(sigma, *args, num=3, device=torch.device("cpu"))
    return gaussian(sigma, **kwargs)
