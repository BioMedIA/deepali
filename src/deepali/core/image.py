r"""Functions relating to tensors representing images."""

import itertools
import math
from typing import Dict, Optional, Sequence, Union

import torch
import torch.nn.functional as F
from torch import Generator, Tensor

from .enum import PaddingMode, Sampling
from .enum import SpatialDim, SpatialDimArg, SpatialDerivativeKeys
from .grid import ALIGN_CORNERS, Axes, Grid, grid_transform_points
from .kernels import gaussian1d, gaussian1d_I
from .nnutils import same_padding, stride_minus_kernel_padding
from .random import multinomial
from .tensor import as_tensor, cat_scalars, move_dim
from .typing import Array, Device, DType, Scalar, ScalarOrTuple, Size, Shape, is_float_dtype


def avg_pool(
    data: Tensor,
    kernel_size: ScalarOrTuple[int],
    stride: Optional[ScalarOrTuple[int]] = None,
    padding: Optional[ScalarOrTuple[int]] = 0,
    ceil_mode: bool = False,
    count_include_pad: bool = True,
    divisor_override: Optional[Scalar] = None,
) -> Tensor:
    r"""Average pooling of image data."""
    if not isinstance(data, Tensor):
        raise TypeError("avg_pool() 'data' must be torch.Tensor")
    if data.ndim < 4:
        raise ValueError("avg_pool() 'data' must have shape (N, C, ..., X)")
    D = data.ndim - 2
    if D == 1:
        avg_pool_fn = F.avg_pool1d
    elif D == 2:
        avg_pool_fn = F.avg_pool2d
    elif D == 3:
        avg_pool_fn = F.avg_pool3d
    else:
        raise ValueError("avg_pool() number of spatial 'data' dimensions must be 1, 2, or 3")
    if isinstance(kernel_size, int):
        kernel_size = (kernel_size,) * D
    elif len(kernel_size) != D:
        raise ValueError(f"avg_pool() 'kernel_size' must be scalar or {D}-tuple")
    if stride is None:
        stride = kernel_size
    if padding is None:
        padding = tuple(n // 2 for n in kernel_size)
    return avg_pool_fn(
        data,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        ceil_mode=ceil_mode,
        count_include_pad=count_include_pad,
        divisor_override=divisor_override,  # type: ignore
    )


def max_pool(
    data: Tensor,
    kernel_size: ScalarOrTuple[int],
    stride: Optional[ScalarOrTuple[int]] = None,
    padding: Optional[ScalarOrTuple[int]] = 0,
    dilation: ScalarOrTuple[int] = 1,
    ceil_mode: bool = False,
) -> Tensor:
    r"""Max pooling of image data."""
    if not isinstance(data, Tensor):
        raise TypeError("max_pool() 'data' must be torch.Tensor")
    if data.ndim < 4:
        raise ValueError("max_pool() 'data' must have shape (N, C, ..., X)")
    D = data.ndim - 2
    if D == 1:
        max_pool_fn = F.max_pool1d
    elif D == 2:
        max_pool_fn = F.max_pool2d
    elif D == 3:
        max_pool_fn = F.max_pool3d
    else:
        raise ValueError("max_pool() number of spatial 'data' dimensions must be 1, 2, or 3")
    if isinstance(kernel_size, int):
        kernel_size = (kernel_size,) * D
    elif len(kernel_size) != D:
        raise ValueError(f"max_pool() 'kernel_size' must be scalar or {D}-tuple")
    if stride is None:
        stride = kernel_size
    if padding is None:
        padding = tuple(n // 2 for n in kernel_size)
    return max_pool_fn(
        data,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        dilation=dilation,
        ceil_mode=ceil_mode,
    )


def min_pool(
    data: Tensor,
    kernel_size: ScalarOrTuple[int],
    stride: Optional[ScalarOrTuple[int]] = None,
    padding: Optional[ScalarOrTuple[int]] = 0,
    dilation: ScalarOrTuple[int] = 1,
    ceil_mode: bool = False,
) -> Tensor:
    r"""Min pooling of image data, i.e., negate max_pool() result of negated input data."""
    return -max_pool(
        -data,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        dilation=dilation,
        ceil_mode=ceil_mode,
    )


def conv(
    data: Tensor,
    kernel: Union[Tensor, Sequence[Optional[Tensor]]],
    stride: ScalarOrTuple[int] = 1,
    dilation: ScalarOrTuple[int] = 1,
    padding: Union[PaddingMode, str, ScalarOrTuple[int]] = None,
    output_padding: Optional[ScalarOrTuple[int]] = None,
    transpose: bool = False,
) -> Tensor:
    r"""Convolve images in batch with a given (separable) kernel.

    Args:
        data: Image batch tensor of shape ``(N, C, ..., X)``.
        kernel: Tensor of shape ``(..., X)`` with weights of kernel used to filter the images
            in this batch by. If the input ``data`` tensor is of non-floating point type, the
            dtype of the kernel defines the intermediate data type used for convolutions.
            If a 1-dimensional kernel is given, it is used as separable convolution kernel in
            all spatial image dimensions. Otherwise, the kernel is applied to the last spatial
            image dimensions. For example, a 2D kernel applied to a batch of 3D image volumes
            is applied slice-by-slice by convolving along the y and x image axes.
            In order to anisotropically convolve the input data with 1-dimensional kernels of
            different sizes, a sequence of at most ``D`` 1-dimensional kernel tensors can be given,
            where ``D`` is the number of spatial dimensions. If the sequence contains ``None``,
            no convolution is performed along the corresponding spatial dimension. The first kernel
            in the sequence is applied to the last spatial grid dimension, which corresponds to
            the ``data`` tensor dimension ``X``, e.g., ``(kz, ky, kx)``.
        stride: Stride by which convolution kernel is advanced.
        dilation: Spacing between kernel elements.
        padding: Image padding mode. If ``int``, pad with zeros the specified margin at each
            side. Otherwise, use ``same_padding()`` calculated from kernel size and dilation such
            that output size is equal to input size, unless ``PaddingMode.NONE`` is given. If ``None``,
            use default mode ``PaddingMode.ZEROS`` with "same" padding.
        output_padding: Output padding for transposed convolution.
        transpose: Whether to compute transposed convolution.

    Returns:
        Result of filtering operation with data type set to the image data type before convolution.
        If dtype is not a floating point data type, the filtered data is being rounded and clamped.

    """
    if not isinstance(data, Tensor):
        raise TypeError("conv() 'data' must be torch.Tensor")
    if data.ndim < 3:
        raise ValueError("conv() 'data' must have shape (N, C, ..., X)")
    D = data.ndim - 2
    if isinstance(kernel, Tensor) and kernel.ndim == 1:
        kernel = [kernel] * D
    dtype = data.dtype
    if isinstance(kernel, Tensor):
        # One n-dimensional kernel (n > 1)
        if isinstance(padding, int):
            margin = (padding,) * kernel.ndim
            padding = PaddingMode.ZEROS
        elif isinstance(padding, Sequence):
            margin = padding
            padding = PaddingMode.ZEROS
        else:
            margin = same_padding(kernel.shape, dilation)
            padding = PaddingMode.from_arg(padding)
        kernel_dtype = kernel.dtype
    else:
        # Sequence of 1-dimensional kernels
        kernel = list(kernel)  # in case argument is a generator
        kernel_dtype = None
        kernel_size = []
        for k in kernel:
            if k is None:
                kernel_size.append(1)
            else:
                if kernel_dtype is None:
                    kernel_dtype = k.dtype
                if k.ndim != 1:
                    raise ValueError(
                        "conv() 'kernel' must be n-dimensional tensor or sequence of 1-dimensional tensors"
                    )
                kernel_size.append(len(k))
        if kernel_dtype is None:
            # No kernels in sequence
            return data
        if isinstance(padding, int):
            margin = (padding,) * len(kernel)
            padding = PaddingMode.ZEROS
        elif isinstance(padding, Sequence):
            margin = padding
            padding = PaddingMode.ZEROS
        else:
            margin = same_padding(kernel_size, dilation)
            padding = PaddingMode.from_arg(padding)
    if sum(margin) != 0 and padding not in (PaddingMode.NONE, PaddingMode.ZEROS):
        if transpose:
            raise NotImplementedError(f"conv() 'transpose=True' with padding {padding.value}")
        margin = tuple(reversed(margin))
        tensor = pad(data, margin=margin, mode=padding)
        return conv(tensor, kernel, stride=stride, dilation=dilation, padding=PaddingMode.NONE)
    if not is_float_dtype(dtype):
        dtype = kernel_dtype if is_float_dtype(kernel_dtype) else torch.float
    tensor = data.type(dtype)
    device = tensor.device
    # Convolve last n dimension by n-dimensional kernel
    if isinstance(kernel, Tensor):
        K = kernel.ndim
        if K > D:
            raise ValueError("conv() 'kernel' has too many dimensions")
        if K == 2:
            conv_fn = F.conv_transpose2d if transpose else F.conv2d
        elif K == 3:
            conv_fn = F.conv_transpose3d if transpose else F.conv3d
        else:
            raise ValueError("conv() 'kernel' must have 1, 2, or 3 spatial dimensions")
        shape_ = tensor.shape
        kernel = kernel.to(dtype=dtype, device=device)
        kernel = kernel.reshape(1, 1, *kernel.shape)
        if tensor.ndim > kernel.ndim:
            groups = tensor.shape[1:-K].numel()
            tensor = tensor.reshape(tensor.shape[0], groups, *tensor.shape[-K:])
        else:
            groups = tensor.shape[1]
        weight = kernel.expand(groups, 1, *kernel.shape[-K:])
        kwargs = dict(
            tensor,
            weight,
            stride=stride,
            dilation=dilation,
            padding=0 if padding == PaddingMode.NONE else margin,
            groups=groups,
        )
        if transpose:
            if output_padding is None:
                output_padding = stride_minus_kernel_padding(1, stride)
            kwargs["output_padding"] = output_padding
        result = conv_fn(tensor, weight, **kwargs)
        result = result.reshape(shape_[:-K] + result.shape[-K:])
    # Convolve last n dimensions by sequence of n 1-dimensional kernels
    else:
        result = tensor
        kernels = list(kernel)
        for i, k in enumerate(kernels):
            if k is not None:
                kernels[i] = k.to(dtype=dtype, device=device)
        K = len(kernels)
        if isinstance(stride, int):
            stride = (stride,) * K
        elif len(stride) != K:
            raise ValueError(f"conv() 'stride' must be int or sequence of {K} ints")
        if isinstance(dilation, int):
            dilation = (dilation,) * K
        elif len(dilation) != K:
            raise ValueError(f"conv() 'dilation' must be int or sequence of {K} ints")
        if output_padding is None:
            output_padding = stride_minus_kernel_padding(1, stride)
        elif isinstance(output_padding, int):
            output_padding = (output_padding,) * K
        elif len(output_padding) != K:
            raise ValueError(f"conv() 'output_padding' must be None, int, or sequence of {K} ints")
        args = zip(kernels, stride, dilation, margin, output_padding)
        for i, (k, s, d, p, op) in enumerate(args):
            if k is None:
                continue
            result = conv1d(
                result,
                k,
                dim=result.ndim - K + i,
                stride=s,
                dilation=d,
                padding=0 if padding == PaddingMode.NONE else p,
                output_padding=op,
                transpose=transpose,
            )
    if not torch.is_floating_point(data):
        result = result.round_()
        result = result.clamp_(min=float(data.min()), max=float(data.max()))
    result = result.type_as(data)
    return result


def conv1d(
    data: Tensor,
    kernel: Tensor,
    dim: int = -1,
    stride: int = 1,
    dilation: int = 1,
    padding: Union[PaddingMode, str, int] = None,
    output_padding: Optional[int] = None,
    transpose: bool = False,
    dtype: Optional[DType] = None,
) -> Tensor:
    r"""Convolve data with 1-dimensional kernel along specified dimension."""
    if not isinstance(data, Tensor):
        raise TypeError("conv1d() 'data' must be torch.Tensor")
    if data.ndim < 3:
        raise ValueError("conv1d() 'data' must have shape (N, C, ..., X)")
    if not isinstance(kernel, Tensor):
        raise TypeError("conv1d() 'kernel' must be of type Tensor")
    if kernel.ndim != 1:
        raise ValueError("conv1d() 'kernel' must be 1-dimensional")
    if dtype is None:
        dtype = data.dtype
    if is_float_dtype(dtype):
        kernel = kernel.type(dtype)
    elif not is_float_dtype(kernel.dtype):
        kernel = kernel.type(torch.float)
    if isinstance(padding, int):
        margin = padding
        padding = PaddingMode.ZEROS
    else:
        padding = PaddingMode.from_arg(padding)
        if padding is PaddingMode.NONE:
            margin = 0
        else:
            margin = same_padding(kernel.shape, dilation)
    result = data.type(kernel.dtype)
    result = move_dim(result, dim, -1)
    shape_ = result.shape
    groups = shape_[1:-1].numel()
    weight = kernel.expand(groups, 1, kernel.shape[-1])
    result = result.reshape(shape_[0], groups, shape_[-1])
    if margin and padding is not PaddingMode.ZEROS:
        result = F.pad(result, (margin, margin), mode=padding.pad_mode(1))
        margin = 0
    conv_fn = F.conv_transpose1d if transpose else F.conv1d
    kwargs = dict(
        stride=stride,
        dilation=dilation,
        padding=margin,
        groups=groups,
    )
    if transpose:
        if output_padding is None:
            output_padding = stride_minus_kernel_padding(1, stride)
        kwargs["output_padding"] = output_padding
    result = conv_fn(result, weight, **kwargs)
    result = result.reshape(shape_[0:-1] + result.shape[-1:])
    result = move_dim(result, -1, dim)
    if not is_float_dtype(dtype):
        result = result.round_()
        result = result.clamp_(min=float(data.min()), max=float(data.max()))
    result = result.type(dtype)
    return result


def dot_batch(a: Tensor, b: Tensor, weight: Optional[Tensor] = None) -> Tensor:
    r"""Weighted dot product between batches of image batch tensors.

    Args:
        a: Image data tensor of shape ``(N, C, ..., X)``.
        b: Image data tensor of shape ``(N, C, ..., X)``.

    Returns:
        Tensor of shape ``(N,)`` containing batchwise dot products.

    """
    return dot_channels(a, b, weight=weight).sum(dim=1)


def dot_channels(a: Tensor, b: Tensor, weight: Optional[Tensor] = None) -> Tensor:
    r"""Weighted dot product between channels of image batch tensors.

    Args:
        a: Image data tensor of shape ``(N, C, ..., X)``.
        b: Image data tensor of shape ``(N, C, ..., X)``.

    Returns:
        Tensor of shape ``(N, C)`` containing channelwise dot products.

    """
    if not isinstance(a, Tensor):
        raise TypeError("dot_channels() 'a' and 'b' must be tensors")
    if a.ndim < 4:
        raise ValueError("dot_channels() 'a' and 'b' must have shape (N, C, ..., X)")
    if a.shape != b.shape:
        raise ValueError("dot_channels() 'a' and 'b' must have identical shape")
    c = a * b
    if weight is not None:
        c *= weight
    return c.view(c.shape[0], c.shape[1], -1).sum(dim=2)


def downsample(
    data: Tensor,
    levels: int = 1,
    dims: Optional[Sequence[SpatialDimArg]] = None,
    sigma: Optional[Union[Scalar, Array]] = None,
    mode: Optional[Union[Sampling, str]] = None,
    min_size: int = 0,
    align_corners: bool = ALIGN_CORNERS,
) -> Tensor:
    r"""Downsample images after optional convolution with truncated Gaussian kernel.

    Args:
        data: Image batch tensor of shape ``(N, C, ..., X)``.
        levels: Number of times the image size is halved. If zero, a reference to the
            unmodified input ``data`` tensor is returned. If negative, the images are
            upsampled instead.
        dims: Spatial dimensions along which to downsample. If not specified, consider all spatial dimensions.
        sigma: Standard deviation of Gaussian used for each downsampling step.
            If a scalar or 1-element sequence is given, an isotropic Gaussian
            kernel is used. Otherwise, the first value is the standard deviation
            of the 1-dimensional Gaussian applied to the first grid dimension, which
            is the last ``data`` tensor dimension, e.g., ``(sx, sy)``. If ``sigma=0``,
            no low-pass filter is applied. If ``None``, a default value is used.
        min_size: Required minimum grid size.
        align_corners: Whether to preserve corner points (True) or grid extent (False).

    Returns:
        Downsampled image data.

    """
    if not isinstance(data, Tensor):
        raise TypeError("downsample() 'data' must be torch.Tensor")
    if data.ndim < 4:
        raise ValueError("downsample() 'data' must have shape (N, C, ..., X)")
    if not torch.is_floating_point(data):
        raise TypeError("downsample() 'data' must have floating point dtype")
    try:
        levels = int(levels)
    except TypeError:
        raise TypeError("downsample() 'levels' must be scalar of type int")
    if levels == 0:
        return data
    if levels < 0:
        return upsample(
            data,
            levels=-levels,
            dims=dims,
            sigma=sigma,
            mode=mode,
            align_corners=align_corners,
        )
    grid = Grid(shape=data.shape[2:])
    if not dims:
        dims = tuple(dim for dim in range(grid.ndim))
    dims = tuple(SpatialDim.from_arg(dim) for dim in dims)
    grid = grid.downsample(levels, dims=dims, min_size=min_size, align_corners=align_corners)
    if sigma is None:
        # This standard deviation is used by NiftyReg and also MIRTK to create the image resolution pyramid for registration
        # https://github.com/BioMedIA/MIRTK/blob/c8e35554f1c23ef14a1c1c51b042e43f82fb44a7/Modules/Registration/src/GenericRegistrationFilter.cc#L526
        sigma = 0.7355
    sigma: Tensor = torch.atleast_1d(as_tensor(sigma, dtype=torch.float))
    if sigma.ndim > 1:
        raise ValueError("downsample() 'sigma' must be scalar or 1-dimensional tensor")
    if sigma.ne(0).any():
        if levels > 1:
            var = sigma.new_zeros(sigma.shape)
            for level in range(levels):
                var += sigma.mul(2**level).pow(2)  # type: ignore
            sigma = var.sqrt()
        if sigma.shape[0] == 1:
            sigma = sigma.repeat(grid.ndim)
            for i in range(grid.ndim):
                if i in dims:
                    continue
                sigma[i] = 0
        kernels = []
        kernels_ = {}
        for i in range(grid.ndim):
            std = float(sigma[i] if i < len(sigma) else 0)
            if std > 0 and grid.size()[i] != data.shape[grid.ndim - i + 1]:
                kernel = kernels_.get(std)
                if kernel is None:
                    kernel = gaussian1d(std, dtype=torch.float, device=data.device)
                    kernels_[std] = kernel
                kernels.append(kernel)
            else:
                kernels.append(None)
        data = conv(data, reversed(kernels))
    mode = Sampling.from_arg(mode).interpolate_mode(grid.ndim)
    return F.interpolate(data, size=grid.shape, mode=mode, align_corners=align_corners)


def upsample(
    data: Tensor,
    levels: int = 1,
    dims: Optional[Sequence[SpatialDimArg]] = None,
    sigma: Optional[Union[Scalar, Array]] = None,
    mode: Optional[Union[Sampling, str]] = None,
    align_corners: bool = ALIGN_CORNERS,
) -> Tensor:
    r"""Upsample images and opitonally deconvolve with truncated Gaussian kernel.

    Args:
        data: Image batch tensor of shape ``(N, C, ..., X)``.
        levels: Number of times the image size is doubled. If zero, a reference to the
            unmodified input ``data`` tensor is returned. If negative, the images are
            downsampled instead.
        dims: Spatial dimensions along which to upsample. If not specified, consider all spatial dimensions.
        sigma: Standard deviation of Gaussian used for each upsampling step.
            If a scalar or 1-element sequence is given, an isotropic Gaussian
            kernel is used. Otherwise, the first value is the standard deviation
            of the 1-dimensional Gaussian applied to the first grid dimension, which
            is the last ``data`` tensor dimension, e.g., ``(sx, sy)``. If ``sigma=0``
            or ``None``, no transposed convolution is applied.
        align_corners: Whether to preserve corner points (True) or grid extent (False).

    Returns:
        Upsampled image data.

    """
    if not isinstance(data, Tensor):
        raise TypeError("upsample() 'data' must be torch.Tensor")
    if data.ndim < 4:
        raise ValueError("upsample() 'data' must have shape (N, C, ..., X)")
    if not torch.is_floating_point(data):
        raise TypeError("upsample() 'data' must have floating point dtype")
    try:
        levels = int(levels)
    except TypeError:
        raise TypeError("upsample() 'levels' must be scalar of type int")
    if levels == 0:
        return data
    if levels < 0:
        return downsample(
            data,
            levels=-levels,
            dims=dims,
            sigma=sigma,
            mode=mode,
            align_corners=align_corners,
        )
    grid = Grid(shape=data.shape[2:], align_corners=align_corners)
    if not dims:
        dims = tuple(dim for dim in range(grid.ndim))
    dims = tuple(SpatialDim.from_arg(dim) for dim in dims)
    grid = grid.upsample(levels, dims=dims)
    mode = Sampling.from_arg(mode).interpolate_mode(grid.ndim)
    result: Tensor = F.interpolate(data, size=grid.shape, mode=mode, align_corners=align_corners)
    if sigma is not None:
        sigma: Tensor = torch.atleast_1d(as_tensor(sigma, dtype=torch.float))
        if sigma.ndim > 1:
            raise ValueError("upsample() 'sigma' must be scalar or 1-dimensional tensor")
        if levels > 1:
            var = sigma.new_zeros(sigma.shape)
            for level in range(levels):
                var += sigma.mul(2**level).pow(2)  # type: ignore
            sigma = var.sqrt()
        if sigma.shape[0] == 1:
            sigma = sigma.repeat(grid.ndim)
            for i in range(grid.ndim):
                if i in dims:
                    continue
                sigma[i] = 0
        kernels = []
        kernels_ = {}
        for i in range(grid.ndim):
            std = float(sigma[i] if i < len(sigma) else 0)
            if std > 0:
                kernel = kernels_.get(std)
                if kernel is None:
                    kernel = gaussian1d(std, dtype=torch.float, device=result.device)
                    kernels_[std] = kernel
                kernels.append(kernel)
            else:
                kernels.append(None)
        result = conv(result, reversed(kernels), transpose=True)
    return result


def gaussian_pyramid(
    data: Tensor,
    levels: int,
    start: int = 0,
    dims: Optional[Sequence[SpatialDimArg]] = None,
    sigma: Optional[Union[Scalar, Array]] = None,
    mode: Optional[Union[Sampling, str]] = None,
    min_size: int = 0,
    align_corners: bool = ALIGN_CORNERS,
) -> Dict[int, Tensor]:
    r"""Create Gaussian image resolution pyramid.

    Args:
        data: Image data tensor of shape ``(N, C, ..., X)``.
        levels: Coarsest resolution level.
        start: Finest resolution level, where 0 corresponds to the original resolution.
        dims: Spatial dimensions along which to downsample. If not specified, consider all spatial dimensions.
        sigma: Standard deviation of Gaussian filter applied at each downsampling level.
        mode: Interpolation mode for resampling image data on downsampled grid.
        min_size: Minimum grid size.
        align_corners: Whether to preserve corner points (True) or grid extent (False).

    Returns:
        Dictionary of downsampled image tensors with keys corresponding to level indices.

    """
    if not isinstance(data, Tensor):
        raise TypeError("gaussian_pyramid() 'data' must be torch.Tensor")
    if data.ndim < 4:
        raise ValueError("gaussian_pyramid() 'data' must have shape (N, C, ..., X)")
    if not isinstance(levels, int):
        raise TypeError("gaussian_pyramid() 'levels' must be of type int")
    if levels < 0:
        raise ValueError("gaussian_pyramid() 'levels' must be positive")
    if not isinstance(start, int):
        raise TypeError("gaussian_pyramid() 'start' must be of type int")
    if start < 0:
        raise ValueError("gaussian_pyramid() 'start' must not be negative")
    pyramid = {0: data}
    if start == 0:
        start = 1
        levels -= 1
    for i, level in enumerate(range(start, start + levels)):
        data = downsample(
            data,
            levels=level if i == 0 else 1,
            dims=dims,
            sigma=sigma,
            mode=mode,
            min_size=min_size,
            align_corners=align_corners,
        )
        pyramid[level] = data
    return pyramid


def crop(
    data: Tensor,
    margin: Optional[Union[int, Array]] = None,
    num: Optional[Union[int, Array]] = None,
    mode: Union[PaddingMode, str] = PaddingMode.CONSTANT,
    value: Scalar = 0,
) -> Tensor:
    r"""Crop or pad images at border.

    Args:
        data: Image batch tensor of shape ``(N, C, ..., X)``.
        margin: Number of spatial grid points to remove (positive) or add (negative) at each border.
            Use instead of ``num`` in order to symmetrically crop the input ``data`` tensor, e.g.,
            ``(nx, ny, nz)`` is equivalent to ``num=(nx, nx, ny, ny, nz, nz)``.
        num: Number of spatial gird points to remove (positive) or add (negative) at each border,
            where margin of the last dimension of the ``data`` tensor must be given first, e.g.,
            ``(nx, nx, ny, ny)``. If a scalar is given, the input is cropped equally at all borders.
            Otherwise, the given sequence must have an even length.
        mode: Image extrapolation mode.
        value: Constant value used for extrapolation if ``mode=PaddingMode.CONSTANT``.

    Returns:
        Cropped or padded image batch data of shape ``(N, C, ..., X)``.

    """
    if not isinstance(data, Tensor):
        raise TypeError("crop() 'data' must be torch.Tensor")
    if data.ndim < 4:
        raise ValueError("crop() 'data' must have shape (N, C, ..., X)")
    D = data.ndim - 2
    if num is not None and margin is not None:
        raise AssertionError("crop() 'margin' and 'num' are mutually exclusive")
    if isinstance(num, int):
        pad_ = (-num,) * 2 * D
    elif num is not None:
        pad_ = tuple(-int(n) for n in num)
        if len(pad_) % 2 == 1:
            raise ValueError("crop() 'num' must be int or have even length")
    elif isinstance(margin, int):
        pad_ = (-margin,) * 2 * D
    elif margin is not None:
        pad_ = tuple(-int(n) for nn in ((n, n) for n in margin) for n in nn)
    else:
        raise AssertionError("crop() either 'margin' or 'num' is required")
    if all(n == 0 for n in pad_):
        return data
    mode = PaddingMode.from_arg(mode)
    if mode == PaddingMode.ZEROS:
        mode = PaddingMode.CONSTANT
        value = 0
    else:
        value = float(value)
    mode = mode.pad_mode(D)
    return F.pad(data, pad_, mode=mode, value=value)


def pad(
    data: Tensor,
    margin: Optional[Union[int, Array]] = None,
    num: Optional[Union[int, Array]] = None,
    mode: Union[PaddingMode, str] = PaddingMode.CONSTANT,
    value: Scalar = 0,
) -> Tensor:
    r"""Pad or crop images at border.

    Args:
        data: Image batch tensor of shape ``(N, C, ..., X)``.
        margin: Number of spatial grid points to add (positive) or remove (negative) at each border,
            Use instead of ``num`` in order to symmetrically pad the input ``data`` tensor.
        num: Number of spatial gird points to add (positive) or remove (negative) at each border,
            where margin of the last dimension of the ``data`` tensor must be given first, e.g.,
            ``(nx, ny, nz)``. If a scalar is given, the input is padded equally at all borders.
            Otherwise, the given sequence must have an even length.
        mode: Image extrapolation mode.
        value: Constant value used for extrapolation if ``mode=PaddingMode.CONSTANT``.

    Returns:
        Padded or cropped image batch data of shape ``(N, C, ..., X)``.

    """
    if not isinstance(data, Tensor):
        raise TypeError("pad() 'data' must be torch.Tensor")
    if data.ndim < 4:
        raise ValueError("pad() 'data' must have shape (N, C, ..., X)")
    D = data.ndim - 2
    if num is not None and margin is not None:
        raise AssertionError("pad() 'margin' and 'num' are mutually exclusive")
    if isinstance(num, int):
        pad_ = (num,) * 2 * D
    elif num is not None:
        pad_ = tuple(int(n) for n in num)
        if len(pad_) % 2 == 1:
            raise ValueError("pad() 'num' must be int or have even length")
    elif isinstance(margin, int):
        pad_ = (margin,) * 2 * D
    elif margin is not None:
        pad_ = tuple(int(n) for nn in ((n, n) for n in margin) for n in nn)
    else:
        raise AssertionError("pad() either 'pad' or 'margin' is required")
    if all(n == 0 for n in pad_):
        return data
    mode = PaddingMode.from_arg(mode)
    if mode == PaddingMode.ZEROS:
        mode = PaddingMode.CONSTANT
        value = 0
    else:
        value = float(value)
    mode = mode.pad_mode(D)
    return F.pad(data, pad_, mode=mode, value=value)


def center_crop(data: Tensor, size: Union[int, Sequence[int]]) -> Tensor:
    r"""Crop image tensor to specified maximum size.

    Args:
        data: Input tensor of shape ``(N, C, ..., X)``.
        size: Maximum output size, where the size of the last tensor
            dimension must be given first, i.e., ``(X, ...)``.
            If an ``int`` is given, all spatial output dimensions
            are cropped to this maximum size. If the length of size
            is less than the spatial dimensions of the ``data`` tensor,
            then only the last ``len(size)`` dimensions are modified.

    Returns:
        Output tensor of shape ``(N, C, ..., X)``.

    """
    if not isinstance(data, Tensor):
        raise TypeError("center_crop() 'data' must be torch.Tensor")
    if data.dim() < 4:
        raise ValueError("center_crop() 'data' must be tensor of shape (N, C, ..., X)")
    sdim = data.dim() - 2
    if isinstance(size, int):
        shape = (size,) * sdim
    elif len(size) == 0:
        return data
    elif len(size) > sdim:
        raise ValueError("center_crop() 'data' has fewer spatial dimensions than output 'size'")
    else:
        shape = data.shape[2:][: sdim - len(size)] + tuple(reversed(size))
    crop = [max(0, m - n) for m, n in zip(data.shape[2:], shape)]
    if sum(crop) == 0:
        return data
    crop = (n // 2 for n in crop)
    crop = (slice(0, data.shape[0]), slice(0, data.shape[1])) + tuple(
        slice(i, i + n) for i, n in zip(crop, shape)
    )
    return data[crop]


def center_pad(
    data: Tensor,
    size: Union[int, Sequence[int]],
    mode: Union[PaddingMode, str] = PaddingMode.CONSTANT,
    value: Scalar = 0,
) -> Tensor:
    r"""Pad image tensor to specified minimum size.

    Args:
        data: Input tensor of shape ``(N, C, ..., X)``.
        size: Minimum output size, where the size of the last tensor
            dimension must be given first, i.e., ``(X, ...)``.
            If an ``int`` is given, all spatial output dimensions
            are cropped or padded to this size. If the length of size
            is less than the spatial dimensions of the ``data`` tensor,
            then only the last ``len(size)`` dimensions are modified.
        mode: PaddingMode mode (cf. ``torch.nn.functional.pad()``).
        value: Value for padding mode "constant".

    Returns:
        Output tensor of shape ``(N, C, ..., X)``.

    """
    if not isinstance(data, Tensor):
        raise TypeError("center_pad() 'data' must be torch.Tensor")
    if data.dim() < 4:
        raise ValueError("center_pad() 'data' must be tensor of shape (N, C, ..., X)")
    sdim = data.dim() - 2
    if isinstance(size, int):
        shape = (size,) * sdim
    elif len(size) == 0:
        return data
    elif len(size) > sdim:
        raise ValueError("center_pad() 'data' has fewer spatial dimensions than output 'size'")
    else:
        shape = data.shape[2:][: sdim - len(size)] + tuple(reversed(size))
    pad = [max(0, n - m) for m, n in zip(data.shape[2:], shape)]
    pad = [(n // 2, (n + 1) // 2) for n in reversed(pad)]
    pad = [n for x in pad for n in x]
    if sum(pad) == 0:
        return data
    mode = PaddingMode.from_arg(mode)
    if mode == PaddingMode.ZEROS:
        mode = PaddingMode.CONSTANT
        value = 0
    mode = mode.pad_mode(sdim)
    return F.pad(data, pad, mode=mode, value=value)


def region_of_interest(
    data: Tensor,
    start: ScalarOrTuple[int],
    size: ScalarOrTuple[int],
    padding: Union[PaddingMode, str, float] = PaddingMode.CONSTANT,
    value: float = 0,
) -> Tensor:
    r"""Extract region of interest from image tensor.

    Args:
        data: Input tensor of shape ``(N, C, ..., X)``.
        start: Indices of lower left corner of region of interest, e.g., ``(x, y, z)``.
        size: Size of region of interest, e.g., ``(nx, ny, nz)``.
        padding: Padding mode to use when extrapolating input image or constant fill value.
        value: Fill value to use when ``padding=Padding.CONSTANT``.

    Returns:
        Tensor of shape ``(N, C, ..., X)`` with spatial size equal to ``size``.

    """
    if not isinstance(data, Tensor):
        raise TypeError("region_of_interest() 'data' must be torch.Tensor")
    if data.dim() < 4:
        raise ValueError("region_of_interest() 'data' must be tensor of shape (N, C, ..., X)")
    sdim = data.dim() - 2
    num = []
    # Parse start index
    if isinstance(start, int):
        start = (start,) * sdim
    elif not isinstance(start, Sequence) or not all(isinstance(n, int) for n in start):
        raise TypeError("region_of_interest() 'start' must be int or sequence of ints")
    elif len(start) != 3:
        raise ValueError(f"region_of_interest() 'start' must be int or sequence of length {sdim}")
    # Parse ROI size
    if isinstance(size, int):
        size = (size,) * sdim
    elif not isinstance(size, Sequence) or not all(isinstance(n, int) for n in size):
        raise TypeError("region_of_interest() 'size' must be int or sequence of ints")
    elif len(size) != 3:
        raise ValueError(f"region_of_interest() 'size' must be int or sequence of length {sdim}")
    # Padding mode and fill value
    if isinstance(padding, (PaddingMode, str)):
        mode = PaddingMode.from_arg(padding)
        value = value
    elif isinstance(padding, (int, float)):
        mode = PaddingMode.CONSTANT
        value = padding
    else:
        raise TypeError("region_of_interest() 'padding' must be str, Padding, or fill value")
    # Compute size of margins to crop/pad
    num = [[start[i], data.shape[data.ndim - 1 - i] - (start[i] + size[i])] for i in range(sdim)]
    num = [n for nn in num for n in nn]
    return crop(data, num=num, mode=mode, value=value)


def fill_border(
    data: Tensor, margin: ScalarOrTuple[int], value: float = 0, inplace: bool = False
) -> Tensor:
    r"""Fill image border with specified value."""
    if not isinstance(data, Tensor):
        raise TypeError("fill_border() 'data' must be torch.Tensor")
    if data.ndim < 4:
        raise ValueError("fill_border() 'data' must have shape (N, C, ..., X)")
    if isinstance(margin, int):
        margin = (margin,) * (data.ndim - 2)
    if len(margin) > data.ndim - 2:
        raise ValueError(f"fill_border() 'margin' must be at most {data.ndim - 2}-dimensional")
    if not inplace:
        data = data.clone()
    for i, m in enumerate(margin):
        dim = data.ndim - i - 1
        idx = slice(0, m)
        idx = tuple(idx if j == dim else slice(None) for j in range(data.ndim))
        data[idx] = value
        idx = slice(data.shape[dim] - m, data.shape[dim])
        idx = tuple(idx if j == dim else slice(None) for j in range(data.ndim))
        data[idx] = value
    return data


def flatten_channels(data: Tensor) -> Tensor:
    r"""Flatten image tensor channels.

    Adopted from: https://github.com/wolny/pytorch-3dunet

    Args:
        data: Input tensor of shape ``(N, C, ..., X)``.

    Returns:
        Tensor of shape ``(C, N * ... * X)``.

    """
    if not isinstance(data, Tensor):
        raise TypeError("flatten_channels() 'data' must be torch.Tensor")
    if data.ndim < 4:
        raise ValueError("flatten_channels() 'data' must have shape (N, C, ..., X)")
    C = data.size(1)
    # Transpose: (N, C, D, H, W) -> (C, N, D, H, W)
    axis_order = (1, 0) + tuple(range(2, data.ndim))
    transposed = data.permute(axis_order)
    # Flatten: (C, N, D, H, W) -> (C, N * D * H * W)
    return transposed.view(C, -1)


def grid_resample(
    data: Tensor,
    in_spacing: Union[float, Array],
    out_spacing: Union[float, Array],
    *args: float,
    mode: Union[Sampling, str] = None,
    padding: Union[PaddingMode, str, Scalar] = None,
) -> Tensor:
    r"""Interpolate image on minimum bounding grid with specified spacing.

    Args:
        data: Image batch tensor with shape ``(N, C, ..., X)``.
        in_spacing: Current grid spacing.
        out_spacing: Spacing of grid on which to sample images, where the spacing
            of the first grid dimension, which is the last ``data`` dimension
            must be given first, e.g., ``(sx, sy, sz)``. If a scalar value is
            given, the images are resampled to this isotropic spacing.
        mode: Image data interpolation mode.
        padding: Image data extrapolation mode.

    Returns:
        This image batch with given spacing and interpolated image tensor.

    """
    if not isinstance(data, Tensor):
        raise TypeError("grid_resample() 'data' must be torch.Tensor")
    if data.ndim < 4:
        raise ValueError("grid_resample() 'data' must have shape (N, C, ..., X)")
    D = data.ndim - 2
    in_spacing = cat_scalars(in_spacing, *args, num=D, device=data.device)
    out_spacing = cat_scalars(out_spacing, *args, num=D, device=data.device)
    mode = Sampling.from_arg(mode).interpolate_mode(D)
    input_grid = Grid(shape=data.shape[2:], spacing=in_spacing)
    output_grid = input_grid.resample(out_spacing)
    if output_grid.shape == input_grid.shape:
        return data
    align_corners = input_grid.align_corners()
    axes = Axes.from_align_corners(align_corners)
    coords = output_grid.coords(align_corners=align_corners, device=data.device)
    coords = grid_transform_points(coords, output_grid, axes, input_grid, axes)
    return grid_sample(data, coords, mode=mode, padding=padding, align_corners=align_corners)


def grid_reshape(
    data: Tensor,
    shape: Union[int, Array, Shape],
    *args: int,
    mode: Union[Sampling, str] = Sampling.LINEAR,
    align_corners: bool = ALIGN_CORNERS,
) -> Tensor:
    r"""Interpolate image with specified spatial image tensor shape.

    Args:
        data: Image batch tensor with shape ``(N, C, ..., X)``.
        shape: Size of spatial image dimensions, where the size of the first grid
            dimension, which is the last ``data`` dimension, must be given last,
            e.g., ``(nz, ny, nx)``.
        mode: Image data interpolation mode.
        align_corners: Whether to preserve corner points (True) or grid extent (False).

    Returns:
        Interpolated image data of specified ``size`` of spatial dimensions.

    """
    if not isinstance(data, Tensor):
        raise TypeError("grid_resample() 'data' must be torch.Tensor")
    if data.ndim < 4:
        raise ValueError("grid_reshape() 'data' must have shape (N, C, ..., X)")
    D = data.ndim - 2
    shape_ = cat_scalars(shape, *args, num=D)
    return grid_resize(data, shape_.flip(0), mode=mode, align_corners=align_corners)


def grid_resize(
    data: Tensor,
    size: Union[int, Array, Shape],
    *args: int,
    mode: Union[Sampling, str] = Sampling.LINEAR,
    align_corners: bool = ALIGN_CORNERS,
) -> Tensor:
    r"""Interpolate image with specified spatial image tensor shape.

    Args:
        data: Image batch tensor with shape ``(N, C, ..., X)``.
        size: Size of spatial image dimensions, where size of first grid dimension, which
            is the last ``data`` dimension, must be given first, e.g., ``(nx, ny, nz)``.
        mode: Image data interpolation mode.
        align_corners: Whether to preserve corner points (True) or grid extent (False).

    Returns:
        Interpolated image data of specified ``size`` of spatial dimensions.

    """
    if not isinstance(data, Tensor):
        raise TypeError("grid_resample() 'data' must be torch.Tensor")
    if data.ndim < 4:
        raise ValueError("grid_resize() 'data' must have shape (N, C, ..., X)")
    D = data.ndim - 2
    size_ = cat_scalars(size, *args, num=D)
    mode_ = Sampling.from_arg(mode).interpolate_mode(D)
    grid = Grid(shape=data.shape[2:], align_corners=align_corners)
    grid = grid.resize(size_)
    if grid.shape == data.shape[2:]:
        return data
    if mode_ in ("area", "nearest", "nearest-exact"):
        align_corners = None
    return F.interpolate(data, size=grid.shape, mode=mode_, align_corners=align_corners)


def check_sample_grid(func: str, data: Tensor, grid: Tensor) -> Tensor:
    r"""Normalize sample grid tensor."""
    if not isinstance(data, Tensor):
        raise TypeError(f"{func}() 'data' must be torch.Tensor")
    if data.ndim < 4:
        raise ValueError(f"{func}() 'data' must have shape (N, C, ..., X)")
    if not isinstance(grid, Tensor):
        raise TypeError(f"{func}() 'grid' must be torch.Tensor")
    N = data.shape[0]
    D = data.ndim - 2
    if not torch.is_floating_point(grid):
        raise TypeError("{func}() 'grid' must have floating point dtype")
    if grid.shape[-1] != D:
        raise ValueError(
            f"Last {func}() 'grid' dimension size must match number of spatial image dimensions"
        )
    if grid.ndim == data.ndim - 1:
        grid = grid.unsqueeze(0)
    elif grid.ndim != data.ndim:
        raise ValueError(
            f"{func}() expected 'grid' tensor with {data.ndim - 1} or {data.ndim} dimensions"
        )
    if N == 1 and grid.shape[0] > 1:
        N = grid.shape[0]
    elif N > 1 and grid.shape[0] == 1:
        grid = grid.expand(N, *grid.shape[1:])
    if grid.shape[0] != N:
        msg = f"{func}() expected tensor 'grid' of shape (..., X, {D})"
        msg += f" or (1, ..., X, {D})" if N == 1 else f" or (1|{N}, ..., X, {D})"
        raise ValueError(msg)
    return grid


def grid_sample(
    data: Tensor,
    grid: Tensor,
    mode: Optional[Union[Sampling, str]] = None,
    padding: Optional[Union[PaddingMode, str, Scalar]] = None,
    align_corners: bool = ALIGN_CORNERS,
) -> Tensor:
    r"""Sample data at grid points.

    Args:
        data: Image batch tensor of shape ``(1, C, ..., X)`` or ``(N, C, ..., X)``.
        grid: Grid points tensor of shape  ``(..., X, D)``, ``(1, ..., X, D)``, or ``(N, ..., X, D)``.
            Coordinates of points at which to sample ``data`` must be with respect to ``Axes.CUBE``.
        mode: Image interpolate mode.
        padding: Image extrapolation mode or constant by which to pad input ``data``.
        align_corners: Whether ``grid`` extrema ``(-1, 1)`` refer to the grid boundary
            edges (``align_corners=False``) or corner points (``align_corners=True``).

    Returns:
        Image batch tensor of sampled data with spatial shape determined by ``grid``, and batch
        size ``N`` based on ``data.shape[0]`` or ``grid.shape[0]``, respectively. The data type
        of the returned tensor is ``data.dtype`` if it is a floating point type or ``mode="nearest"``.
        Otherwise, the output data type matches ``grid.dtype``, which must be a floating point type.

    """
    grid = check_sample_grid("grid_sample", data, grid)
    if data.device != grid.device:
        raise ValueError("grid_sample() 'data' and 'grid' tensors must be on same device")
    N = grid.shape[0]
    D = grid.shape[-1]
    if data.shape[0] != N:
        data = data.expand(N, *data.shape[1:])
    mode = Sampling.from_arg(mode).grid_sample_mode(D)
    if isinstance(padding, (PaddingMode, str)):
        padding_mode = PaddingMode.from_arg(padding).grid_sample_mode(D)
        padding_value = 0
    else:
        padding_mode = "zeros"
        padding_value = float(padding or 0)
    out = data.type_as(grid)
    if padding_value != 0:
        if out.data_ptr() == data.data_ptr():
            out = out.sub(padding_value)
        else:
            out = out.sub_(padding_value)
    out = F.grid_sample(
        out, grid, mode=mode, padding_mode=padding_mode, align_corners=align_corners
    )
    if padding_mode == "zeros" and padding_value != 0:
        out = out.add_(padding_value)
    if mode == "nearest" or data.is_floating_point():
        out = out.type_as(data)
    return out


def grid_sample_mask(
    data: Tensor,
    grid: Tensor,
    threshold: float = 0,
    align_corners: bool = ALIGN_CORNERS,
) -> Tensor:
    r"""Sample binary mask at grid points.

    Args:
        data: Image batch tensor of shape ``(N, 1, ..., X)``.
        grid: Grid points tensor of shape  ``(..., X, D)``, ``(1, ..., X, D)``, or``(N, ..., X, D)``.
            Coordinates of points at which to sample ``data`` must be with respect to ``Axes.CUBE``.
        threshold: Scalar value used to binarize input mask. Values above this threshold are assigned
            value 1, and values below this threshold are assigned value 0.
        align_corners: Whether ``grid`` extrema ``(-1, 1)`` refer to the grid boundary edges (``False``)
            or corner points (``True``), respectively.

    Returns:
        Batch tensor of sampled mask with spatial shape determined by ``grid``, and floating point values
        in the closed interval ``[0, 1]`` as obtained by linear interpolation of the binarized input mask.

    """
    grid = check_sample_grid("grid_sample_mask", data, grid)
    if data.shape[1] != 1:
        raise ValueError("grid_sample_mask() 'data' must have single channel")
    mask = data if data.dtype == torch.bool else data > threshold
    mask = mask.to(dtype=torch.float32, device=data.device)
    return grid_sample(
        mask,
        grid,
        mode=Sampling.LINEAR,
        padding=PaddingMode.ZEROS,
        align_corners=align_corners,
    )


def rand_sample(
    data: Union[Tensor, Sequence[Tensor]],
    num_samples: int,
    mask: Optional[Tensor] = None,
    replacement: bool = False,
    generator: Optional[Generator] = None,
) -> Union[Tensor, Sequence[Tensor]]:
    r"""Random sampling of voxels within an image.

    Args:
        data: One or more image batch tensors with shape ``(N, C, ... X)`` to sample
            from at the same random image grid positions, i.e., voxel indices. Note that
            all input tensors must have the same number and size of spatial dimensions.
        num_samples: Number of spatial samples to draw.
        mask: Optional mask to use for spatially weighted sampling.
        replacement: Whether to sample with or without replacement.
        generator: Random number generator to use.

    Returns:
        Tensor of shape ``(N, C, num_samples)`` with input ``data`` values at randomly
        sampled spatial grid points. When ``data`` is a sequence of tensors, a list
        of tensors with order matching the input data is returned.

    """
    input: Sequence[Tensor]
    if isinstance(data, Tensor):
        input = [data]
    elif isinstance(data, Sequence) and all(isinstance(x, Tensor) for x in data):
        input = data
    else:
        raise TypeError("rand_sample() 'data' must be Tensor or Sequence[Tensor]")
    if not input:
        return []
    if any(x.ndim < 3 for x in input):
        raise ValueError("rand_sample() 'data' must be one or more tensors of shape (N, C, ..., X)")
    shape = input[0].shape
    device = input[0].device
    if any(x.shape[2:] != shape[2:] for x in input):
        raise ValueError("rand_sample() 'data' tensors must have identical spatial shape")
    numel = shape[2:].numel()
    if not replacement and num_samples > numel:
        raise ValueError("rand_sample() 'num_samples' is greater than number of spatial points")
    input = [x.flatten(2) for x in input]
    if mask is None:
        if replacement:
            index = torch.randint(
                0,
                numel,
                (shape[0], num_samples),
                generator=generator,
                device=device,
                dtype=torch.int64,
            )
        else:
            perm = torch.empty(numel, dtype=torch.int64, device=device)
            index = torch.empty((shape[0], num_samples), dtype=torch.int64, device=device)
            for row in index:
                torch.randperm(numel, generator=generator, out=perm, device=device)
                row.copy_(perm.narrow(0, 0, num_samples), non_blocking=True)
    else:
        if mask.ndim < 3:
            raise ValueError("rand_sample() 'mask' must be tensor of shape (N, C, ..., X)")
        if mask.shape[2:] != shape[2:]:
            raise ValueError("rand_sample() 'mask' has different spatial shape than 'data'")
        if mask.shape[1] != 1:
            raise ValueError("rand_sample() 'mask' must be scalar image tensor")
        mask = mask.flatten(2).squeeze(1).expand(shape[0], numel)
        index = multinomial(mask, num_samples, replacement=replacement, generator=generator)
    index = index.unsqueeze(1).repeat(1, shape[1], 1)
    out = [x.gather(2, index) for x in input]
    if len(out) == 1 and isinstance(data, Tensor):
        return out[0]
    return out


def image_slice(data: Tensor, offset: Optional[int] = None) -> Tensor:
    r"""Get slice from image tensor.

    Args:
        data: Image data tensor of shape ``(N, C, ..., Y, X)``.
        offset: Slice offset. If ``None``, use ``Z // 2``. This argument is
            ignored when the input image is 2-dimensional.

    Returns:
        View of image tensor slice with shape ``(N, C, Y, X)``.

    """
    if data.ndim == 4:
        return data
    if data.ndim < 4 or data.ndim > 5:
        raise ValueError("image_slice() 'data' must be 4- or 5-dimensional")
    if offset is None:
        offset = data.shape[2] // 2
    return data.narrow(2, offset, 1).squeeze(2)


def normalize_image(
    data: Tensor,
    mode: str = "unit",
    min: Optional[float] = None,
    max: Optional[float] = None,
    inplace: bool = False,
) -> Tensor:
    r"""Normalize image intensities in [min, max].

    Args:
        data: Input image data.
        mode: How to normalize image values:
            - ``center``: Linearly rescale to [-0.5, 0.5]
            - ``unit``: Linearly rescale to [0, 1].
            - ``zscore``: Linearly  rescale to zero mean and unit variance.
        min: Minimum intensity at which to clamp input.
        max: Maximum intensity at which to clamp input.
        inplace: Whether to modify ``data`` in place.

    Returns:
        Normalized image data.

    Raises:
        TypeError: When ``inplace=True`` and ``data.dtype`` is not a floating point data type.

    """
    if inplace:
        if not data.is_floating_point():
            raise AssertionError("normalize_image() 'data.dtype' must be float when inplace=True")

        def add_fn(data: Tensor, a: float) -> Tensor:
            return data.add_(a)

        def sub_fn(data: Tensor, a: float) -> Tensor:
            return data.sub_(a)

        def mul_fn(data: Tensor, a: float) -> Tensor:
            return data.mul_(a)

        def clamp_fn(data: Tensor, a: float, b: float) -> Tensor:
            return data.clamp_(a, b)

    else:
        if not data.is_floating_point():
            data = data.float()

        def add_fn(data: Tensor, a: float) -> Tensor:
            return data.add(a)

        def sub_fn(data: Tensor, a: float) -> Tensor:
            return data.sub(a)

        def mul_fn(data: Tensor, a: float) -> Tensor:
            return data.mul(a)

        def clamp_fn(data: Tensor, a: float, b: float) -> Tensor:
            return data.clamp(a, b)

    if mode in ("zscore", "z-score"):
        data = clamp_fn(data, min, max)
        stdev, mean = torch.std_mean(data)
        data = sub_fn(data, mean)
        if stdev > 1e-15:
            data = mul_fn(data, 1 / stdev)
    elif mode in ("unit", "center"):
        if min is None:
            min = float(data.min())
        if max is None:
            max = float(data.max())
        dif = max - min
        mul = 1 if abs(dif) < 1e-9 else 1 / dif
        add = -mul * min
        if mode == "center":
            add -= 0.5
        if mul != 1:
            data = mul_fn(data, mul)
        if add != 0:
            data = add_fn(data, add)
        if mode == "center":
            data = clamp_fn(data, -0.5, 0.5)
        else:
            data = clamp_fn(data, 0, 1)
    return data


def rescale(
    data: Tensor,
    min: Optional[Scalar] = None,
    max: Optional[Scalar] = None,
    data_min: Optional[Scalar] = None,
    data_max: Optional[Scalar] = None,
    dtype: Optional[DType] = None,
) -> Tensor:
    r"""Linearly rescale values to specified output interval.

    Args:
        data: Input tensor.
        min: Minimum value of output tensor. Use ``data_min`` if ``None``.
        max: Maximum value of output tensor. Use ``data_max`` if ``None``.
        data_min: Minimum value of input ``data``. Use ``data.min()`` if ``None``.
        data_max: Maximum value of input ``data``. Use ``data.max()`` if ``None``.
        dtype: Cast rescaled values to specified output data type. If ``None``,
            use ``data.dtype`` if it is a floating point type, otherwise ``torch.float``.

    Returns:
        Tensor of same shape as ``data`` with specified ``dtype`` and values in closed interval ``[min, max]``.

    """
    if not isinstance(data, Tensor):
        raise TypeError("rescale() 'data' must be torch.Tensor")
    if dtype is None:
        dtype = data.dtype
        if not is_float_dtype(dtype):
            dtype = torch.float
    if dtype.is_floating_point:
        interim_dtype = dtype
    elif data.dtype.is_floating_point:
        interim_dtype = data.dtype
    else:
        interim_dtype = torch.float
    data = data.type(interim_dtype)
    if data_min is None:
        data_min = data.min()
    data_min = float(data_min)
    if data_max is None:
        data_max = data.max()
    data_max = float(data_max)
    min = data_min if min is None else float(min)
    max = data_max if max is None else float(max)
    norm = data_max - data_min
    if norm < 1e-15:
        result = data.new_empty(data.shape).fill_(min)
    else:
        scale = (max - min) / norm
        result = min + scale * (data - data_min)
    if not dtype.is_floating_point:
        result = result.round_()
    result = result.clamp_(min=min, max=max)
    result = result.type(dtype)
    return result


def sample_image(
    data: Tensor,
    coords: Tensor,
    mode: Optional[Union[Sampling, str]] = None,
    padding: Optional[Union[PaddingMode, str, Scalar]] = None,
    align_corners: bool = ALIGN_CORNERS,
) -> Tensor:
    r"""Sample images at given points.

    This function samples a batch of images at spatial points. The ``coords`` tensor can be of any shape,
    including ``(N, M, D)``, i.e., a batch of N point sets with cardianality M, and ``(N, ..., X, D)`` ,
    i.e., a (deformed) regular sampling grid (cf. ``grid_sample()``).

    Args:
        data: Batch of images as tensor of shape ``(N, C, ..., X)``. If batch size is one,
            but the batch size of ``coords`` is greater than one, this single image is sampled
            at the different sets of points.
        coords: Normalized coordinates of points given as tensor of shape ``(N, ..., D)``
            or ``(1, ..., D)``. If batch size is one, all images are sampled at the same points.
        align_corners: Whether point coordinates are with respect to ``Axes.CUBE`` (False)
            or ``Axes.CUBE_CORNERS`` (True). This option is in particular passed on to the
            ``grid_sample()`` function used to sample the images at the given points.

    Returns:
        Sampled image data as tensor of shape ``(N, C, *coords.shape[1:-1])``.

    """
    if not isinstance(data, Tensor):
        raise TypeError("sample_image() 'data' must be of type torch.Tensor")
    data = data.as_subclass(Tensor)
    if data.ndim < 4:
        raise ValueError("sample_image() 'data' must be at least 4-dimensional tensor")
    if not isinstance(coords, Tensor):
        raise TypeError("sample_image() 'coords' must be of type torch.Tensor")
    if coords.ndim < 2:
        raise ValueError("sample_image() 'coords' must be at least 2-dimensional tensor")
    G = data.shape[0]
    N = coords.shape[0] if G == 1 else G
    D = data.ndim - 2
    if coords.shape[0] not in (1, N):
        raise ValueError(f"sample_image() 'coords' must be batch of length 1 or {N}")
    if coords.shape[-1] != D:
        raise ValueError(f"sample_image() 'coords' must be tensor of {D}-dimensional points")
    x = coords.expand((N,) + coords.shape[1:])
    data = data.expand((N,) + data.shape[1:])
    grid = x.reshape((N,) + (1,) * (data.ndim - 3) + (-1, D))  # pseudo grid
    data = grid_sample(data, grid, mode=mode, padding=padding, align_corners=align_corners)
    return data.reshape(data.shape[:2] + coords.shape[1:-1])


def spatial_derivatives(
    data: Tensor,
    which: Optional[Union[str, Sequence[str]]] = None,
    order: Optional[int] = None,
    mode: Optional[str] = None,
    sigma: Optional[float] = None,
    spacing: Optional[Union[Scalar, Array]] = None,
    stride: Optional[ScalarOrTuple[int]] = None,
) -> Dict[str, Tensor]:
    r"""Calculate spatial image derivatives.

    Args:
        data: Image data tensor of shape ``(N, C, ..., X)``.
        which: String codes of spatial deriviatives to compute. See ``SpatialDerivativeKeys``.
        order: Order of spatial derivative. If zero, the input ``data`` is returned.
        mode: Method to use for approximating spatial image derivative.
            If ``forward``, ``backward``, ``central``, or ``forward_central_backward``,
            the respective finite difference scheme is used to approximate the image derivative,
            optionally after smoothing the input image with a Gaussian kernel if ``sigma`` is a
            positive number. The ``forward_central_backward`` finite difference scheme consists
            of forward differences at the lower boundary, a backward differences at the upper
            boundary, and central differences in between. If ``gaussian``, the image derivative
            is computed by convolving the image with a derivative of Gaussian kernel.
            If ``bspline``, the input ``data`` is used as coefficients of a cubic B-spline
            function, with the number of coefficients along each dimension equal the size of
            the output tensor along this dimension plus one coefficient at the lower bound, and
            plus two additional coefficients at the upper bound. In other words, the output tensor
            will have a spatial size minus three with respect to the input ``data`` tensor.
            If ``None``, ``forward_central_backward`` is used as default mode.
        sigma: Standard deviation of Gaussian kernel in grid units. If ``None`` or zero,
            no Gaussian smoothing is used for calculation of finite differences, and a
            default standard deviation of 0.7355 is used when ``mode="gaussian"``. With a smaller
            standard deviation, the magnitude of the derivative values starts to deviate between
            ``mode="gaussian"`` and finite differences of a Gaussian smoothed input. This is likely
            due to a too small discretized Gaussian filter and its derivative.
        spacing: Physical spacing between image grid points, e.g., ``(sx, sy, sz)``.
            When a scalar is given, the same spacing is used for each image and spatial dimension.
            If a sequence is given, it must be of length equal to the number of spatial dimensions ``D``,
            and specify a separate spacing for each dimension in the order ``(x, ...)``. In order to
            specify a different spacing for each image in the input ``data`` batch, a 2-dimensional
            tensor must be given, where the size of the first dimension is equal to ``N``. The second
            dimension can have either size 1 for an isotropic spacing, or ``D`` in case of an
            anisotropic grid spacing.
        stride: Number of output grid points between control points plus one for ``mode='bspline'``.
            If a sequence of values is given, these must be the strides for the different spatial
            dimensions in the order ``(sx, ...)``. A stride of 1 is equivalent to evaluating partial
            derivatives only at the usually coarser resolution of the control point grid. It should
            be noted that the stride need not match the stride used to densely sample the spline at
            a given fixed target image resolution.

    Returns:
        Mapping from spatial derivative keys to corresponding tensors of the respective spatial
        image derivatives of shape ``(N, C, ..., X)``. The keys are sequences of letters identifying
        the spatial dimensions along which a derivative was taken (cf. ``SpatialDerivativeKeys``).
        If ``mode='bspline'``, the output size is ``X - 3`` for each spatial dimension.

    """
    if not isinstance(data, Tensor):
        raise TypeError("spatial_derivatives() 'data' must be torch.Tensor")
    if data.ndim < 4:
        raise ValueError("spatial_derivatives() 'data' must be at least 4-dimensional")

    N = data.shape[0]
    D = data.ndim - 2

    if spacing is None:
        spacing = torch.ones((N, D), dtype=torch.float, device=torch.device("cpu"))
    else:
        spacing = as_tensor(spacing, dtype=torch.float, device=torch.device("cpu"))
        spacing = torch.atleast_1d(spacing)
        if spacing.ndim == 1:
            spacing = spacing.unsqueeze(0)
        if spacing.ndim != 2 or spacing.shape[0] not in (1, N) or spacing.shape[1] not in (1, D):
            raise ValueError(
                f"spatial_derivatives() 'spacing' must be scalar, {D}-dimensional vector,"
                f" or 2-dimensional array of shape (1, {D}), ({N}, 1), or ({N}, {D})"
            )
        spacing = spacing.expand(N, D)

    if isinstance(which, str):
        which = (which,)
    if which is None:
        if order is None:
            order = 1
        which = SpatialDerivativeKeys.all(spatial_dims=D, order=order)
    elif order is not None:
        which = [arg for arg in which if len(arg) == order]

    derivs = {}
    if not which:
        return derivs

    unique_keys = SpatialDerivativeKeys.unique(which)
    max_order = SpatialDerivativeKeys.max_order(which)

    if not data.is_floating_point():
        data = data.float()

    if mode is None:
        mode = "forward_central_backward"

    if mode in ("forward", "backward", "central", "forward_central_backward", "prewitt", "sobel"):
        if sigma and sigma > 0:
            blur = gaussian1d(sigma, dtype=torch.float, device=data.device)
            data = conv(data, blur, padding=PaddingMode.REPLICATE)
        if mode in ("prewitt", "sobel"):
            avg_kernel = torch.tensor([1, 1 if mode == "prewitt" else 2, 1], dtype=data.dtype)
            avg_kernel /= avg_kernel.sum()
            avg_kernel = avg_kernel.to(data.device)
            fd_mode = "forward_central_backward"
        else:
            avg_kernel = None
            fd_mode = mode
        for i, code in itertools.product(range(max_order), unique_keys):
            key = code[: i + 1]
            if i < len(code) and key not in derivs:
                sdim = SpatialDim.from_arg(code[i])
                result = data if i == 0 else derivs[code[:i]]
                if avg_kernel is not None:
                    for d in (d for d in range(D) if d != sdim):
                        dim = SpatialDim(d).tensor_dim(result.ndim)
                        result = conv1d(
                            result,
                            avg_kernel,
                            dim=dim,
                            padding=len(avg_kernel) // 2,
                        )
                fd_spacing = spacing[:, sdim]
                result = finite_differences(result, sdim, mode=fd_mode, spacing=fd_spacing)
                derivs[key] = result
        derivs = {key: derivs[SpatialDerivativeKeys.sorted(key)] for key in which}

    elif mode == "bspline":
        from .bspline import cubic_bspline_interpolation_weights, evaluate_cubic_bspline

        if sigma and sigma > 0:
            blur = gaussian1d(sigma, dtype=torch.float, device=data.device)
            data = conv(data, blur, padding=PaddingMode.REPLICATE)

        if stride is None:
            stride = 1
        if isinstance(stride, int):
            stride = (stride,) * D

        kernels = {}

        def bspline1d(s: int, d: int) -> Tensor:
            key = (s, d)
            kernel = kernels.get(key, None)
            if kernel is None:
                kernel = cubic_bspline_interpolation_weights(
                    stride=s,
                    derivative=d,
                    dtype=data.dtype,
                    device=data.device,
                )
                kernels[key] = kernel
            return kernel

        for code in unique_keys:
            order = [0] * D
            for spatial_dim in SpatialDerivativeKeys.split(code):
                order[spatial_dim] += 1
            kernel = [bspline1d(s, d) for s, d in zip(stride, order)]
            deriv = evaluate_cubic_bspline(data, kernel=kernel)
            if sum(order) > 0:
                denom = torch.ones(N, dtype=spacing.dtype, device=spacing.device)
                for delta, d in zip(spacing.transpose(0, 1), order):
                    if d > 0:
                        denom.mul_(delta.pow(d))
                denom = denom.reshape((N,) + (1,) * (deriv.ndim - 1))
                deriv = deriv.div_(denom.to(deriv))
            derivs[code] = deriv

    elif mode == "gaussian":

        def pad_spatial_dim(data: Tensor, sdim: int, padding: int) -> Tensor:
            pad = [(padding, padding) if d == sdim else (0, 0) for d in range(data.ndim - 2)]
            pad = [n for v in pad for n in v]
            return F.pad(data, pad, mode="replicate")

        if not sigma:
            sigma = 0.7355  # same default value as used in downsample()
        kernel_0 = gaussian1d(sigma, normalize=False, dtype=torch.float, device=data.device)
        kernel_1 = gaussian1d_I(sigma, normalize=False, dtype=torch.float, device=data.device)
        for i in range(max_order):
            for code in unique_keys:
                key = code[: i + 1]
                if i < len(code) and key not in derivs:
                    sdim = SpatialDim.from_arg(code[i])
                    deriv = data if i == 0 else derivs[code[:i]]
                    for d in range(D):
                        dim = SpatialDim(d).tensor_dim(deriv.ndim)
                        kernel = kernel_1 if sdim == d else kernel_0
                        deriv = pad_spatial_dim(deriv, d, len(kernel) // 2)
                        deriv = conv1d(deriv, kernel, dim=dim, padding=0)
                    denom = spacing.narrow(1, sdim, 1).reshape((N,) + (1,) * (deriv.ndim - 1))
                    deriv = deriv.div_(denom.to(deriv))
                    derivs[key] = deriv
        derivs = {key: derivs[SpatialDerivativeKeys.sorted(key)] for key in which}

    else:
        raise ValueError(
            "spatial_derivatives() 'mode' must be 'forward', 'backward', 'central', or 'gaussian'"
        )

    return derivs


def finite_differences(
    data: Tensor,
    sdim: SpatialDimArg,
    mode: str = "forward_central_backward",
    order: int = 1,
    dilation: int = 1,
    spacing: Union[float, Sequence[float]] = 1,
) -> Tensor:
    r"""Calculate spatial image derivative using finite differences.

    Args:
        data: Image data tensor of shape ``(N, C, ..., X)``.
        sdim: Spatial dimension along which to compute spatial derivative.
        mode: Finite differences to use for approximating spatial derivative.
        order: Order of spatial derivative. If zero, the input ``data`` is returned.
        dilation: Step size for finite differences.
        spacing: Physical spacing between image grid points along dimension ``sdim``.
            When a scalar is given, the same spacing is used for all images in the
            input ``data`` batch. Otherwise, a separate spacing must be specified for
            each image as sequence of float values.

    Returns:
        Tensor of spatial derivative with respect to specified spatial dimension.

    """
    if not isinstance(data, Tensor):
        raise TypeError("finite_differences() 'data' must be torch.Tensor")
    if data.ndim < 4:
        raise ValueError("finite_differences() 'data' must have shape (N, C, ..., X)")
    if not isinstance(order, int):
        raise TypeError("finite_differences() 'order' must be int")
    if order < 0:
        raise ValueError("finite_differences() 'order' must be non-negative")
    if not isinstance(dilation, int):
        raise TypeError("finite_differences() 'dilation' must be int")
    if dilation < 1:
        raise ValueError("finite_differences() 'dilation' must be positive")
    if mode not in ("forward", "backward", "central", "forward_central_backward"):
        raise ValueError(
            "finite_differences() 'mode' must be"
            " 'forward', 'backward', 'central', or 'forward_central_backward'"
        )

    if order == 0:
        return data

    N = data.shape[0]
    spatial_dim = SpatialDim.from_arg(sdim)
    dim = spatial_dim.tensor_dim(data.ndim)

    step_size: Tensor = torch.atleast_1d(as_tensor(spacing, dtype=data.dtype, device=data.device))
    if step_size.ndim > 1 or step_size.shape[0] not in (1, N):
        raise ValueError(f"finite_differences() 'spacing' must be scalar or sequence of length {N}")

    if not data.is_floating_point():
        data = data.float()

    def pad_spatial_dim(data: Tensor, left: int, right: int) -> Tensor:
        pad = [(left, right) if d == spatial_dim else (0, 0) for d in range(data.ndim - 2)]
        pad = [n for v in pad for n in v]
        return F.pad(data, pad, mode="replicate")

    def finite_difference(data: Tensor, i: slice, j: slice) -> Tensor:
        h = step_size.mul(j.start - i.start)
        h = h.reshape((h.shape[0],) + (1,) * (data.ndim - 1))
        a = tuple((i if d == dim else slice(0, n, 1) for d, n in enumerate(data.shape)))
        b = tuple((j if d == dim else slice(0, n, 1) for d, n in enumerate(data.shape)))
        return data[b].sub(data[a]).div_(h)

    if order == 1:
        if mode == "forward":
            data = pad_spatial_dim(data, 0, dilation)
            i = slice(0, data.shape[dim] - dilation, 1)
            j = slice(dilation, data.shape[dim], 1)
            deriv = finite_difference(data, i, j)
        elif mode == "backward":
            data = pad_spatial_dim(data, dilation, 0)
            i = slice(0, data.shape[dim] - dilation, 1)
            j = slice(dilation, data.shape[dim], 1)
            deriv = finite_difference(data, i, j)
        elif mode == "central":
            data = pad_spatial_dim(data, dilation, dilation)
            i = slice(0, data.shape[dim] - 2 * dilation, 1)
            j = slice(2 * dilation, data.shape[dim], 1)
            deriv = finite_difference(data, i, j)
        else:
            if data.shape[dim] < 2 * dilation:
                raise ValueError(
                    "finite_differences() mode 'forward_central_backward' requires spatial"
                    " dimension to have size of at least twice the 'dilation' step size"
                    " used to calculate finite differences"
                )
            # forward differences at lower boundary
            i = slice(0, dilation, 1)
            j = slice(dilation, 2 * dilation, 1)
            lower = finite_difference(data, i, j)
            # central differences within boundaries
            i = slice(0, data.shape[dim] - 2 * dilation, 1)
            j = slice(2 * dilation, data.shape[dim], 1)
            deriv = finite_difference(data, i, j)
            # backward differences at upper boundary
            i = slice(data.shape[dim] - 2 * dilation, data.shape[dim] - dilation, 1)
            j = slice(data.shape[dim] - dilation, data.shape[dim], 1)
            upper = finite_difference(data, i, j)
            # concatenate computed derivatives
            deriv = torch.cat([lower, deriv, upper], dim=dim)
    else:
        raise NotImplementedError(f"finite_differences(..., order={order})")

    return deriv


def _image_size(
    fn_name: str,
    size: Optional[Union[int, Size, Grid]] = None,
    shape: Optional[Shape] = None,
    ndim: Optional[int] = None,
) -> torch.Size:
    r"""Parse 'size' and/or 'shape' argument of image creation function."""
    if size is None and shape is None:
        raise AssertionError(f"{fn_name}() 'size' or 'shape' required")
    if isinstance(size, Grid):
        size = size.size()
    if size is not None and shape is not None and size != tuple(reversed(shape)):
        raise AssertionError(f"{fn_name}() mismatch between 'size' and 'shape'")
    if size is None:
        if ndim and len(shape) != ndim:
            raise ValueError(f"{fn_name}() 'shape' must be tuple of length {ndim}")
        size = tuple(reversed(shape))
    elif isinstance(size, int):
        size = (size, size)
    elif ndim and len(size) != 2:
        raise ValueError(f"{fn_name}() 'size' must be tuple of length {ndim}")
    return torch.Size(size)


def circle_image(
    size: Optional[Union[int, Size, Grid]] = None,
    shape: Optional[Shape] = None,
    num: Optional[int] = None,
    center: Optional[Sequence[int]] = None,
    radius: Optional[float] = None,
    sigma: float = 0,
    x_max: Optional[Union[float, Sequence[float]]] = None,
    dtype: Optional[DType] = None,
    device: Optional[Device] = None,
) -> Tensor:
    r"""Synthetic image of a circle.

    Args:
        size: Spatial size in the order ``(X, Y)``.
        shape: Spatial size in the order ``(Y, X)``.
        num: Number ``N`` of images in batch.
            If zero, return a single unbatched image tensor.
        center: Coordinates of center pixel in the order ``(x, y)``.
        radius: Radius of circle in pixel units.
        sigma: Standard deviation of isotropic Gaussian blurring kernel in pixel units.
        x_max: Maximum ``x`` pixel index at which to clamp image to zero.
            This can be used to create partial circles such as a half circle.
        dtype: Data type of output tensor. Use ``torch.uint8`` if ``None``.
            When the output data type is a floating point type, the output tensor
            values are in the interval ``[0, 1]``. Otherwise, the output values are
            in the interval ``[0, 255]``.
        device: Device on which to create image tensor.

    Returns:
        Image tensor of shape ``(N, 1, Y, X)`` or ``(1, Y, X)`` (``num=0``).

    """
    size = _image_size("circle_image", size, shape, ndim=2)
    if center is None:
        center = tuple(((n - 1) / 2 for n in size))
    center = tuple(float(x) for x in center)
    grid = Grid(size=size)
    if radius is None:
        radius = max(0, (min(center) - 1) - math.ceil(2 * sigma))
    _dtype = torch.float32
    _device = torch.device("cpu")
    c = as_tensor(center, dtype=_dtype, device=_device)
    x = grid.coords(normalize=False, dtype=_dtype, device=_device)
    x = x.reshape(1, 1, *grid.shape, grid.ndim).sub_(c)
    data = torch.norm(x, dim=-1) <= radius
    if x_max is not None:
        x_threshold = as_tensor(x_max, dtype=_dtype, device=_device)
        if x_threshold.ndim == 0:
            data &= x[..., 0] <= x_threshold
        else:
            data &= (x <= x_threshold).all(dim=-1)
    data = data.type(_dtype)
    if sigma > 0:
        kernel = gaussian1d(sigma, dtype=data.dtype, device=_device)
        data = conv(data, kernel / kernel.sum())
    if dtype is None:
        dtype = torch.uint8
    if not dtype.is_floating_point:
        data = data.mul_(255).div_(data.max())
    data = data.to(dtype=dtype, device=device)
    if num is not None:
        if num == 0:
            data = data.squeeze_(0)
        elif num > 1:
            data = data.expand(num, *data.shape[1:])
    return data


def cshape_image(
    size: Optional[Union[int, Size, Grid]] = None,
    shape: Optional[Shape] = None,
    num: Optional[int] = None,
    center: Optional[Sequence[float]] = None,
    radius: Optional[float] = None,
    width: Optional[float] = None,
    sigma: float = 0,
    x_max: Optional[Union[float, Sequence[float]]] = None,
    dtype: Optional[DType] = None,
    device: Optional[Device] = None,
) -> Tensor:
    r"""Synthetic C-shaped image.

    Args:
        size: Spatial size in the order ``(X, Y)``.
        shape: Spatial size in the order ``(Y, X)``.
        num: Number ``N`` of images in batch.
            If zero, return a single unbatched image tensor.
        center: Coordinates of center pixel in the order ``(y, x)``.
        radius: Radius of circle in pixel units.
        width: Difference between outer and inner circle radius.
            A default width of ``radius // 2`` is used if ``None``.
        sigma: Standard deviation of isotropic Gaussian blurring kernel in pixel units.
        x_max: Maximum ``x`` pixel index at which to clamp image to zero. This is used to create
            partial circles with an opening for the "C" shape to the right. If ``None``, a default
            value equal to 75% of the inner circle radius is chosen.
        dtype: Data type of output tensor. Use ``torch.uint8`` if ``None``.
            When the output data type is a floating point type, the output tensor
            values are in the interval ``[0, 1]``. Otherwise, the output values are
            in the interval ``[0, 255]``.
        device: Device on which to create image tensor.

    Returns:
        Image tensor of shape ``(N, 1, Y, X)`` or ``(1, Y, X)`` (``num=0``).

    """
    size = _image_size("cshape_image", size, shape, ndim=2)
    if dtype is None:
        dtype = torch.uint8
    if radius is None:
        center = tuple(float(x) for x in center)
        radius = max(0, (min(center) - 1) - math.ceil(2 * sigma))
    if width is None:
        width = radius // 2
    if x_max is None:
        x_max = 0.75 * (radius - width)
    outer = circle_image(size, num=1, x_max=x_max, sigma=0, center=center, radius=radius)
    inner = circle_image(size, num=1, x_max=x_max, sigma=0, center=center, radius=radius - width)
    image = outer.type(torch.float32).sub(inner)
    if sigma > 0:
        kernel = gaussian1d(sigma, dtype=torch.float32, device=image.device)
        image = conv(image, kernel)
        if dtype.is_floating_point:
            image /= image.max()
            image.clamp_(min=0, max=1)
        else:
            image *= 255 / image.max()
            image.clamp_(min=0, max=255)
    image = image.to(dtype=dtype, device=device)
    if num is not None:
        if num == 0:
            image = image.squeeze_(0)
        elif num > 1:
            image = image.expand((1,) + image.shape[1:])
    return image


def empty_image(
    size: Optional[Union[int, Size, Grid]] = None,
    shape: Optional[Shape] = None,
    num: Optional[int] = None,
    channels: Optional[int] = None,
    dtype: Optional[DType] = None,
    device: Optional[Device] = None,
) -> Tensor:
    """Create new batch of uninitalized image data.

    Args:
        size: Spatial size in the order ``(X, ...)``.
        shape: Spatial size in the order ``(..., X)``.
        num: Number of images in batch.
            If zero, return a single unbatched image tensor.
        channels: Number of channels per image.
        dtype: Data type of image tensor.
        device: Device on which to store image data.

    Returns:
        Image tensor of shape ``(N, 1, ..., X)`` or ``(1, ..., X)`` (``num=0``).

    """
    size = _image_size("empty_image", size, shape)
    shape = (num or 1, channels or 1) + tuple(reversed(size))
    data = torch.empty(shape, dtype=dtype, device=device)
    if num == 0:
        data = data.squeeze_(0)
    return data


def grid_image(
    size: Optional[Union[int, Size, Grid]] = None,
    shape: Optional[Shape] = None,
    num: Optional[int] = None,
    stride: Optional[Union[int, Sequence[int]]] = None,
    inverted: bool = False,
    dtype: Optional[DType] = None,
    device: Optional[Device] = None,
) -> Tensor:
    """Create batch of regularly spaced grid images.

    Args:
        size: Spatial size in the order ``(X, ...)``.
        shape: Spatial size in the order ``(..., X)``.
        num: Number of images in batch. When ``shape`` is not a ``Grid``, must
            match the size of the first dimension in ``shape`` if not ``None``.
            If zero, return a single unbatched image tensor.
        stride: Spacing between grid lines. To draw in-plane grid lines on a
            D-dimensional image where ``D>2``, specify a sequence of two stride
            values, where the first stride applies to the last tensor dimension,
            which corresponds to the first spatial grid dimension.
        inverted: Whether to draw grid lines in black (0) over white (1) background.
        dtype: Data type of image tensor.
        device: Device on which to store image data.

    Returns:
        Image tensor of shape ``(N, 1, ..., X)`` or ``(1, ..., X)`` (``num=0``).

    """
    size = _image_size("grid_image", size, shape)
    data = empty_image(size, num=1, channels=1, dtype=dtype, device=device)
    data.fill_(1 if inverted else 0)
    if stride is None:
        stride = 4
    if isinstance(stride, int):
        stride = (stride,) * (data.ndim - 2)
    if len(stride) > data.ndim - 2:
        raise ValueError(
            "grid_image() 'stride' length must not be greater than number of spatial dimensions"
        )
    start = data.ndim - len(stride)
    for dim, step in zip(range(start, data.ndim), reversed(stride)):
        n = data.shape[dim]
        index = torch.arange((n % step) // 2, n, step, dtype=torch.int64, device=data.device)
        data.index_fill_(dim, index, 0 if inverted else 1)
    if num is not None:
        if num == 0:
            data = data.squeeze_(0)
        elif num > 1:
            data = data.expand((1,) + data.shape[1:])
    return data


def ones_image(
    size: Optional[Union[int, Size, Grid]] = None,
    shape: Optional[Shape] = None,
    num: Optional[int] = None,
    channels: Optional[int] = None,
    dtype: Optional[DType] = None,
    device: Optional[Device] = None,
) -> Tensor:
    """Create new batch of image data filled with ones.

    Args:
        size: Spatial size in the order ``(X, ...)``.
        shape: Spatial size in the order ``(..., X)``.
        num: Number of images in batch.
        channels: Number of channels per image.
        dtype: Data type of image tensor.
        device: Device on which to store image data.

    Returns:
        Image tensor of shape ``(N, 1, ..., X)`` or ``(1, ..., X)`` (``num=0``) filled with ones.

    """
    size = _image_size("ones_image", size, shape)
    data = empty_image(size, num=num, channels=channels, dtype=dtype, device=device)
    return data.fill_(1)


def zeros_image(
    size: Optional[Union[int, Size, Grid]] = None,
    shape: Optional[Shape] = None,
    num: Optional[int] = None,
    channels: Optional[int] = None,
    dtype: Optional[DType] = None,
    device: Optional[Device] = None,
) -> Tensor:
    """Create new batch of image data filled with zeros.

    Args:
        size: Spatial size in the order ``(X, ...)``.
        shape: Spatial size in the order ``(..., X)``.
        num: Number of images in batch.
        channels: Number of channels per image.
        dtype: Data type of image tensor.
        device: Device on which to store image data.

    Returns:
        Image tensor of shape ``(N, 1, ..., X)`` or ``(1, ..., X)`` (``num=0``) filled with zeros.

    """
    size = _image_size("zeros_image", size, shape)
    data = empty_image(size, num=num, channels=channels, dtype=dtype, device=device)
    return data.fill_(0)
