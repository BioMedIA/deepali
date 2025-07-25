r"""Convoluational modules with fixed parameters."""

import math
from typing import Optional, Union

from packaging.version import Version

import torch
from torch import Tensor
from torch.nn import Module, functional as F

from deepali.core import functional as U
from deepali.core.enum import PaddingMode
from deepali.core.kernels import gaussian1d
from deepali.core.typing import ScalarOrTuple


class FilterImage(Module):
    r"""Convoles an image with a predefined filter kernel."""

    def __init__(
        self,
        kernel: Optional[Tensor],
        padding: Optional[Union[PaddingMode, str]] = None,
    ):
        r"""Initialize parameters.

        Args:
            kernel: Predefined convolution kernel.
            padding: Image extrapolation mode.

        """
        super().__init__()
        self.padding = PaddingMode.CONSTANT if padding is None else PaddingMode.from_arg(padding)
        self.register_buffer("kernel", kernel)

    def forward(self, x: Tensor) -> Tensor:
        r"""Convolve input image with predefined filter kernel."""
        kernel: Optional[Tensor] = self.kernel
        if kernel is None or kernel.numel() < 2:
            return x
        return U.conv(x, kernel, padding=self.padding)

    def extra_repr(self) -> str:
        return f"padding={repr(self.padding.value)}"


class BlurImage(FilterImage):
    r"""Blurs an image by a predefined Gaussian low-pass filter."""

    def __init__(self, sigma: float, padding: Optional[Union[PaddingMode, str]] = None):
        r"""Initialize parameters.

        Args:
            sigma: Standard deviation of isotropic Gaussian kernel in grid units (pixel, voxel).
            padding: Image extrapolation mode.

        """
        sigma = float(sigma)
        kernel = gaussian1d(sigma) if sigma > 0 else None
        super().__init__(kernel=kernel, padding=padding)
        self.sigma = sigma

    def extra_repr(self) -> str:
        return f"sigma={repr(self.sigma)}, " + super().extra_repr()


class GaussianConv(Module):
    r"""Blurs an image by a predefined Gaussian low-pass filter."""

    def __init__(
        self,
        channels: int,
        kernel_size: ScalarOrTuple[int] | Tensor,
        sigma: ScalarOrTuple[float] | Tensor,
        dim: int = 3,
    ) -> None:
        r"""Initialize Gaussian convolution kernel.

        Args:
            channels (int, sequence): Number of channels of the input and output tensors.
            kernel_size (int, sequence): Size of the gaussian kernel.
            sigma (float, sequence): Standard deviation of the gaussian kernel.
            dim (int, optional): The number of dimensions of the data.

        """
        if dim < 2 or dim > 3:
            raise ValueError(f"Only 2 and 3 dimensions are supported, got: {dim}")
        super().__init__()
        # Convert kernel size argument
        if isinstance(kernel_size, Tensor):
            if kernel_size.ndim == 0 or (kernel_size.ndim == 1 and len(kernel_size) == 1):
                kernel_size = (int(kernel_size.item()),) * dim
            else:
                kernel_size = tuple(int(x) for x in kernel_size)
        elif isinstance(kernel_size, int):
            kernel_size = (kernel_size,) * dim
        else:
            kernel_size = tuple(int(x) for x in kernel_size)
        assert isinstance(kernel_size, tuple)
        assert all(isinstance(x, int) for x in kernel_size)
        # Convert kernel standard deviation argument
        if isinstance(sigma, Tensor):
            if sigma.ndim == 0 or (sigma.ndim == 1 and len(sigma) == 1):
                sigma = (float(sigma.item()),) * dim
            else:
                sigma = tuple(float(x) for x in sigma)
        elif isinstance(sigma, float):
            sigma = (sigma,) * dim
        else:
            sigma = tuple(float(x) for x in sigma)  # type: ignore
        assert isinstance(sigma, tuple)
        assert all(isinstance(x, float) for x in sigma)
        # The gaussian kernel is the product of the gaussian function of each dimension.
        kernel = torch.tensor(1, dtype=torch.float32, device="cpu")
        mgrids = [torch.arange(n, dtype=torch.float32) for n in kernel_size]
        if Version(torch.__version__) < Version("1.10"):
            mgrids = torch.meshgrid(mgrids)
        else:
            mgrids = torch.meshgrid(mgrids, indexing="ij")
        norm = math.sqrt(2 * math.pi)
        for size, std, mgrid in zip(kernel_size, sigma, mgrids):
            mean = (size - 1) / 2
            kernel *= 1 / (std * norm) * torch.exp(-(((mgrid - mean) / std) ** 2) / 2)
        # Make sure sum of values in gaussian kernel equals 1.
        kernel = kernel.div_(kernel.sum())
        # Reshape to depthwise convolutional weight
        kernel = kernel.view(1, 1, *kernel.size())
        kernel = kernel.repeat(channels, *((1,) * (kernel.dim() - 1)))
        self.register_buffer("kernel", kernel, persistent=True)
        self.groups = channels
        # Padding for output to equal input dimensions
        self.pad = (kernel_size[0] // 2,) * (2 * dim)
        self.conv = F.conv2d if dim == 2 else F.conv3d

    def forward(self, x: Tensor) -> Tensor:
        r"""Convolve input with Gaussian kernel."""
        # Use mode='replicate' to ensure image boundaries are not degraded in output for background channel
        kernel: Tensor = self.kernel
        data = F.pad(x, self.pad, mode="replicate")
        data = self.conv(data, kernel, groups=self.groups)
        return data
