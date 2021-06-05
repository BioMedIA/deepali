r"""Decorator for tensors of image data."""

from __future__ import annotations

from copy import copy as shallow_copy
from typing import Dict, Optional, Sequence, Tuple, Type, TypeVar, Union, overload

import torch
from torch import Tensor

try:
    import SimpleITK as sitk
    from ..utils.sitk.torch import image_from_tensor, tensor_from_image
    from ..utils.sitk.imageio import read_image
except ImportError:
    sitk = None

from ..core.enum import PaddingMode, Sampling
from ..core.grid import ALIGN_CORNERS, Axes, Grid
from ..core import image as U
from ..core.names import image_batch_tensor_names, image_tensor_names
from ..core.path import unlink_or_mkdir
from ..core.tensor import as_tensor, cat_scalars
from ..core.types import Array, Device, PathStr, Scalar, ScalarOrTuple, Size

from .tensor import TensorDecorator


TImage = TypeVar("TImage", bound="Image")
TImageBatch = TypeVar("TImageBatch", bound="ImageBatch")


__all__ = ("Image", "ImageBatch")


class ImageBatch(TensorDecorator):
    r"""Batch of images sampled on regular oriented grids."""

    __slots__ = ("_grid",)

    def __init__(
        self: TImageBatch,
        data: Array,
        grid: Optional[Union[Grid, Sequence[Grid]]] = None,
        device: Optional[Device] = None,
    ) -> None:
        r"""Initialize image batch.

        Args:
            data: Image data tensor of shape (N, C, ...X).
            grid: Sampling grid of image data oriented in world space. Can be either a single shared
                sampling grid, or a separate grid for each image in the batch. Note that operations
                which would result in differently sized images (e.g., resampling to a certain voxel
                size, when images have different resolutions) will raise an exception. All images in
                a batch must have the same number of channels and spatial size. If ``None``, a default
                grid whose world space coordinate axes are aligned with the image axes, unit spacing,
                and origin at the image centers is created. By default, image grid attributes are always
                stored in CPU memory, regardless of the ``device`` on which the image data is located.
            device: Device on which to store image data.

        """
        data_ = as_tensor(data, device=device)
        if data_.ndim < 4:
            raise ValueError("Image batch tensor must have at least four dimensions")
        names = image_batch_tensor_names(data_.ndim)
        if None in data_.names:
            data_ = data.refine_names(*names)  # type: ignore
        data_: Tensor = data_.align_to(*names)  # type: ignore
        data_ = data_.rename(None)  # named tensor support still experimental
        super().__init__(data_)
        self.grid_(grid)

    @overload
    def tensor(self: TImageBatch, named: bool = False) -> Tensor:
        r"""Get image batch tensor."""
        ...

    @overload
    def tensor(self: TImageBatch, data: Array, **kwargs) -> TImageBatch:
        r"""Get instance with specified image batch tensor."""
        ...

    def tensor(
        self: TImageBatch,
        data: Optional[Array] = None,
        grid: Optional[Union[Grid, Sequence[Grid]]] = None,
        named: bool = False,
        device: Optional[Device] = None,
    ) -> Union[Tensor, TImageBatch]:
        r"""Get image batch tensor or new batch with specified tensor, respectively."""
        if data is None:
            tensor = self._tensor
            if named:
                names = image_batch_tensor_names(tensor.ndim)
                tensor = tensor.refine_names(*names)
            return tensor
        other = shallow_copy(self)
        other.tensor_(data, grid=grid, device=device)
        return other

    def tensor_(
        self: TImageBatch,
        data: Array,
        grid: Optional[Union[Grid, Sequence[Grid]]] = None,
        device: Optional[Device] = None,
    ) -> TImageBatch:
        r"""Change data tensor of this image batch."""
        if device is None:
            device = self.device
        if isinstance(data, Tensor):
            data_ = data.to(device)
        else:
            data_ = torch.tensor(data, device=device)
        if data_.ndim < 4:
            raise ValueError("Image batch tensor must have at least the four dimensions")
        if not grid:
            grid_: Tuple[Grid, ...] = self._grid
        elif isinstance(grid, Grid):
            grid_ = (grid,) * data_.shape[0]
        else:
            grid_ = tuple(grid)
        if any(g.shape != data_.shape[2:] for g in grid_):
            raise ValueError("Image grid sizes must match spatial dimensions of image batch tensor")
        names = image_batch_tensor_names(data_.ndim)
        if None in data_.names:
            data_ = data_.refine_names(*names)  # type: ignore
        data_: Tensor = data_.align_to(*names)  # type: ignore
        data_ = data_.rename(None)  # named tensor support still experimental
        self._tensor = data_
        self._grid = grid_
        return self

    @overload
    def grid(self: TImageBatch, n: int = 0) -> Grid:
        r"""Get sampling grid of n-th image in batch."""
        ...

    @overload
    def grid(self: TImageBatch, arg: Union[Grid, Sequence[Grid]]) -> TImageBatch:
        r"""Get new image batch with specified sampling grid(s)."""
        ...

    def grid(
        self: TImageBatch, arg: Optional[Union[int, Grid, Sequence[Grid]]] = None
    ) -> Union[Grid, TImageBatch]:
        r"""Get sampling grid of images in batch or new batch with specified grid, respectively."""
        if arg is None:
            arg = 0
        if isinstance(arg, int):
            return self._grid[arg]
        copy = shallow_copy(self)
        return copy.grid_(arg)

    def grid_(self: TImageBatch, arg: Union[Grid, Sequence[Grid], None]) -> TImageBatch:
        r"""Change image sampling grid of this image batch."""
        shape = self._tensor.shape
        if arg is None:
            arg = (Grid(shape=shape[2:]),) * shape[0]
        elif isinstance(arg, Grid):
            grid = arg
            if grid.shape != shape[2:]:
                raise ValueError(
                    "Image grid size does not match spatial dimensions of image batch tensor"
                )
            arg = (grid,) * shape[0]
        else:
            arg = tuple(arg)
            if any(grid.shape != shape[2:] for grid in arg):
                raise ValueError(
                    "Image grid sizes must match spatial dimensions of image batch tensor"
                )
        self._grid = arg
        return self

    def grids(self: TImageBatch) -> Tuple[Grid, ...]:
        r"""Get sampling grids of images in batch."""
        return self._grid

    def align_corners(self: TImageBatch) -> bool:
        r"""Whether image resizing operations by default preserve corner points or grid extent."""
        return self._grid[0].align_corners()

    def spacing(self: TImageBatch) -> Tensor:
        r"""Image spacing as tensor of shape (N, D)."""
        return torch.cat([grid.spacing().unsqueeze(0) for grid in self.grids()], dim=0)

    def __len__(self: TImageBatch) -> int:
        r"""Number of images in batch."""
        return self._tensor.shape[0]

    def __getitem__(self: TImageBatch, index: int) -> Image:
        r"""Get image at specified batch index."""
        if index < 0 or index >= len(self):
            raise IndexError(f"Image batch index ({index}) out of range [0, {len(self)}]")
        return Image(data=self._tensor[index], grid=self._grid[index])

    @property
    def ndim(self: TImageBatch) -> int:
        r"""Number of spatial dimensions."""
        return self._tensor.ndim - 2

    @property
    def nchannels(self: TImageBatch) -> int:
        r"""Number of image channels."""
        return self._tensor.shape[1]

    def normalize(self: TImageBatch, *args, **kwargs) -> TImageBatch:
        r"""Normalize image intensities in [min, max]."""
        copy = shallow_copy(self)
        return copy.normalize_(*args, **kwargs)

    def normalize_(
        self: TImageBatch,
        mode: str = "unit",
        min: Optional[float] = None,
        max: Optional[float] = None,
    ) -> TImageBatch:
        r"""Normalize image intensities in [min, max]."""
        data = self.tensor()
        data = data.float()
        data = U.normalize_image(data, mode=mode, min=min, max=max, inplace=True)
        return self.tensor_(data)

    def resize(self: TImageBatch, *args, **kwargs) -> TImageBatch:
        r"""Interpolate images on grid with specified size."""
        copy = shallow_copy(self)
        return copy.resize_(*args, **kwargs)

    def resize_(
        self: TImageBatch,
        size: Union[int, Array, Size],
        *args: int,
        mode: Union[Sampling, str] = Sampling.LINEAR,
        align_corners: Optional[bool] = None,
    ) -> TImageBatch:
        r"""Interpolate images on grid with specified size.

        Args:
            size: Size of spatial image dimensions, where the size of the last tensor dimension,
                which corresponds to the first grid dimension, must be given first, e.g., ``(nx, ny, nz)``.
            mode: Image data interpolation mode.
            align_corners: Whether to preserve grid extent (False) or corner points (True).
                If ``None``, the default of the image sampling grid is used.

        Returns:
            This image batch with specified size of spatial dimensions.

        """
        if align_corners is None:
            align_corners = self.align_corners()
        size = self._cat_scalars(size, *args)
        data = U.grid_resize(self._tensor, size, mode=mode, align_corners=align_corners)
        grid = [grid.resize(size, align_corners=align_corners) for grid in self._grid]
        return self.tensor_(data, grid=grid)

    def resample(self: TImageBatch, *args, **kwargs) -> TImageBatch:
        r"""Interpolate images on grid with specified spacing."""
        copy = shallow_copy(self)
        return copy.resample_(*args, **kwargs)

    def resample_(
        self: TImageBatch,
        spacing: Union[float, Array, str],
        *args: float,
        mode: Union[Sampling, str] = Sampling.LINEAR,
    ) -> TImageBatch:
        r"""Interpolate images on grid with specified spacing.

        Args:
            spacing: Spacing of grid on which to resample image data, where the spacing corresponding
                to first grid dimension, which corresponds to the last tensor dimension, must be given
                first, e.g., ``(sx, sy, sz)``. Alternatively, can be string 'min' or 'max' to resample
                to the minimum or maximum voxel size, respectively.
            mode: Image data interpolation mode.

        Returns:
            This image batch with given grid spacing.

        """
        in_spacing = self._grid[0].spacing()
        if not all(torch.allclose(grid.spacing(), in_spacing) for grid in self._grid):
            raise AssertionError(
                f"{type(self).__name__}.resample_() requires all images in batch to have the same grid spacing"
            )
        if spacing == "min":
            assert not args
            out_spacing = in_spacing.min()
        elif spacing == "max":
            assert not args
            out_spacing = in_spacing.max()
        else:
            out_spacing = spacing
        out_spacing = self._cat_scalars(out_spacing, *args)
        data = U.grid_resample(
            self._tensor, in_spacing=in_spacing, out_spacing=out_spacing, mode=mode
        )
        grid = [grid.resample(out_spacing) for grid in self._grid]
        return self.tensor_(data, grid=grid)

    def avg_pool(self: TImageBatch, *args, **kwargs) -> TImageBatch:
        r"""Average pooling of image data."""
        copy = shallow_copy(self)
        return copy.avg_pool_(*args, **kwargs)

    def avg_pool_(
        self: TImageBatch,
        kernel_size: ScalarOrTuple[int],
        stride: Optional[ScalarOrTuple[int]] = None,
        padding: ScalarOrTuple[int] = 0,
        ceil_mode: bool = False,
        count_include_pad: bool = True,
    ) -> TImageBatch:
        r"""Average pooling of image data."""
        data = U.avg_pool(
            self._tensor,
            kernel_size,
            stride=stride,
            padding=padding,
            ceil_mode=ceil_mode,
            count_include_pad=count_include_pad,
        )
        grid = [
            grid.avg_pool(
                kernel_size,
                stride=stride,
                padding=padding,
                ceil_mode=ceil_mode,
            )
            for grid in self._grid
        ]
        return self.tensor_(data, grid=grid)

    def downsample(self: TImageBatch, levels: int = 1, **kwargs) -> TImageBatch:
        r"""Downsample images in batch by halving their size the specified number of times."""
        copy = shallow_copy(self)
        return copy.downsample_(levels, **kwargs)

    def downsample_(
        self: TImageBatch,
        levels: int = 1,
        sigma: Optional[Union[Scalar, Array]] = None,
        mode: Optional[Union[Sampling, str]] = None,
        min_size: int = 0,
        align_corners: Optional[bool] = None,
    ) -> TImageBatch:
        r"""Downsample images in batch by halving their size the specified number of times.

        Args:
            levels: Number of times the image size is halved (>0) or doubled (<0).
            sigma: Standard deviation of Gaussian filter applied at each downsampling level.
            mode: Image interpolation mode.
            align_corners: Whether to preserve grid extent (False) or corner points (True).
                If ``None``, the default of the image sampling grid is used.

        Returns:
            Reference to this image batch.

        """
        if not isinstance(levels, int):
            raise TypeError("ImageBatch.downsample() 'levels' must be of type int")
        if align_corners is None:
            align_corners = self.align_corners()
        data = U.downsample(
            self._tensor,
            levels,
            sigma=sigma,
            mode=mode,
            min_size=min_size,
            align_corners=align_corners,
        )
        grid = [
            grid.downsample(levels, min_size=min_size, align_corners=align_corners)
            for grid in self._grid
        ]
        return self.tensor_(data, grid=grid)

    def upsample(self: TImageBatch, levels: int = 1, **kwargs) -> TImageBatch:
        r"""Upsample image in batch by doubling their size the specified number of times."""
        copy = shallow_copy(self)
        return copy.upsample_(levels, **kwargs)

    def upsample_(
        self: TImageBatch,
        levels: int = 1,
        sigma: Optional[Union[Scalar, Array]] = None,
        mode: Optional[Union[Sampling, str]] = None,
        align_corners: Optional[bool] = None,
    ) -> TImageBatch:
        r"""Upsample image in batch by doubling their size the specified number of times.

        Args:
            levels: Number of times the image size is doubled (>0) or halved (<0).
            sigma: Standard deviation of Gaussian filter applied at each downsampling level.
            mode: Image interpolation mode.
            align_corners: Whether to preserve grid extent (False) or corner points (True).
                If ``None``, the default of the image sampling grid is used.

        Returns:
            Reference to this image batch.

        """
        if not isinstance(levels, int):
            raise TypeError("ImageBatch.upsample() 'levels' must be of type int")
        if align_corners is None:
            align_corners = self.align_corners()
        data = U.upsample(self._tensor, levels, sigma=sigma, mode=mode, align_corners=align_corners)
        grid = [grid.upsample(levels, align_corners=align_corners) for grid in self._grid]
        return self.tensor_(data, grid=grid)

    def pyramid(
        self: TImageBatch,
        levels: int,
        start: int = 0,
        end: int = -1,
        sigma: Optional[Union[Scalar, Array]] = None,
        mode: Optional[Union[Sampling, str]] = None,
        spacing: Optional[float] = None,
        min_size: int = 0,
        align_corners: Optional[bool] = None,
    ) -> Dict[int, TImageBatch]:
        r"""Create Gaussian resolution pyramid.

        Args:
            levels: Number of resolution levels.
            start: Highest resolution level to return, where 0 corresponds to the finest resolution.
            end: Lowest resolution level to return (inclusive).
            sigma: Standard deviation of Gaussian filter applied at each downsampling level.
            mode: Interpolation mode for resampling image data on downsampled grid.
            spacing: Grid spacing at finest resolution level. Note that this option may increase the
                cube extent of the multi-resolution pyramid sampling grids.
            min_size: Minimum grid size.
            align_corners: Whether to preserve grid extent (False) or corner points (True).
                If ``None``, the default of the image sampling grid is used.

        Returns:
            Dictionary of downsampled image batches with keys corresponding to level indices.

        """
        if not isinstance(levels, int):
            raise TypeError("ImageBatch.pyramid() 'levels' must be int")
        if not isinstance(start, int):
            raise TypeError("ImageBatch.pyramid() 'start' must be int")
        if not isinstance(end, int):
            raise TypeError("ImageBatch.pyramid() 'end' must be int")
        if start < 0:
            start = levels + start
        if start < 0 or start > levels - 1:
            raise ValueError(f"ImageBatch.pyramid() 'start' must be in [{-levels}, {levels - 1}]")
        if end < 0:
            end = levels + end
        if end < 0 or end > levels - 1:
            raise ValueError(f"ImageBatch.pyramid() 'end' must be in [{-levels}, {levels - 1}]")
        if start > end:
            return {}
        # Current image grids
        if align_corners is None:
            align_corners = self.align_corners()
        grids = tuple(grid.align_corners(align_corners) for grid in self._grid)
        # Finest level grids of multi-level resolution pyramid
        if spacing is not None:
            spacing0 = grids[0].spacing()
            if not all(torch.allclose(grid.spacing(), spacing0) for grid in grids):
                raise AssertionError(
                    f"{type(self).__name__}.pyramid() requires all images to have the same grid"
                    " spacing when output 'spacing' at finest level is specified"
                )
            grids = tuple(grid.resample(spacing) for grid in grids)
        grids = tuple(grid.pyramid(levels, min_size=min_size)[0] for grid in grids)
        assert all(grid.size() == grids[0].size() for grid in grids)
        # Resize image to match finest resolution grid
        data = self.tensor()
        if torch.allclose(grids[0].cube_extent(), self._grid[0].cube_extent()):
            size = grids[0].size()
            data = U.grid_resize(data, size, mode=mode, align_corners=align_corners)
        else:
            points = grids[0].coords(device=self.device)
            data = U.grid_sample(data, points, mode=mode, align_corners=align_corners)
        # Construct image pyramid by repeated downsampling
        pyramid = {}
        image = self.tensor(data, grid=grids)
        if start == 0:
            pyramid[0] = image
        for level in range(1, end + 1):
            image = image.downsample(sigma=sigma, mode=mode, min_size=min_size)
            if level >= start:
                pyramid[level] = image
        return pyramid

    def crop(self: TImageBatch, *args, **kwargs) -> TImageBatch:
        r"""Crop images at boundary."""
        copy = shallow_copy(self)
        return copy.crop_(*args, **kwargs)

    def crop_(
        self: TImageBatch,
        margin: Optional[Union[int, Array]] = None,
        num: Optional[Union[int, Array]] = None,
        mode: Union[PaddingMode, str] = PaddingMode.CONSTANT,
        value: Scalar = 0,
    ) -> TImageBatch:
        r"""Crop images at boundary.

        Args:
            margin: Number of spatial grid points to remove (positive) or add (negative) at each border.
                Use instead of ``num`` in order to symmetrically crop the input ``data`` tensor, e.g.,
                ``(nx, ny, nz)`` is equivalent to ``num=(nx, nx, ny, ny, nz, nz)``.
            num: Number of spatial gird points to remove (positive) or add (negative) at each border,
                where margin of the last dimension of the ``data`` tensor must be given first, e.g.,
                ``(nx, nx, ny, ny)``. If a scalar is given, the input is cropped equally at all borders.
                Otherwise, the given sequence must have an even length.
            mode: Image extrapolation mode in case of negative crop value.
            value: Constant value used for extrapolation if ``mode=PaddingMode.CONSTANT``.

        Returns:
            This image with modified size, but unchanged spacing.

        """
        grids: Sequence[Grid] = self._grid
        data = U.crop(self._tensor, margin=margin, num=num, mode=mode, value=value)
        grid = [grid.crop(margin=margin, num=num) for grid in grids]
        return self.tensor_(data, grid=grid)

    def center_crop(self: TImageBatch, *args, **kwargs) -> TImageBatch:
        r"""Crop image to specified maximum size."""
        copy = shallow_copy(self)
        return copy.center_crop_(*args, **kwargs)

    def center_crop_(self: TImageBatch, size: Union[int, Array], *args: int) -> TImageBatch:
        r"""Crop image to specified maximum size.

        Args:
            size: Maximum output size, where the size of the last tensor
                dimension must be given first, i.e., ``(X, ...)``.
                If an ``int`` is given, all spatial output dimensions
                are cropped to this maximum size. If the length of size
                is less than the spatial dimensions of the ``data`` tensor,
                then only the last ``len(size)`` dimensions are modified.

        """
        size_ = self._cat_scalars(size, *args)
        data = U.center_crop(self._tensor, size_)
        grid = [grid.center_crop(size_) for grid in self._grid]
        return self.tensor_(data, grid=grid)

    def pad(self: TImageBatch, *args, **kwargs) -> TImageBatch:
        r"""Pad images at boundary."""
        copy = shallow_copy(self)
        return copy.pad_(*args, **kwargs)

    def pad_(
        self: TImageBatch,
        margin: Optional[Union[int, Array]] = None,
        num: Optional[Union[int, Array]] = None,
        mode: Union[PaddingMode, str] = PaddingMode.CONSTANT,
        value: Scalar = 0,
    ) -> TImageBatch:
        r"""Pad images at boundary.

        Args:
            margin: Number of spatial grid points to add (positive) or remove (negative) at each border,
                Use instead of ``num`` in order to symmetrically pad the input ``data`` tensor.
            num: Number of spatial gird points to add (positive) or remove (negative) at each border,
                where margin of the last dimension of the ``data`` tensor must be given first, e.g.,
                ``(nx, ny, nz)``. If a scalar is given, the input is padded equally at all borders.
                Otherwise, the given sequence must have an even length.
            mode: Image extrapolation mode in case of positive pad value.
            value: Constant value used for extrapolation if ``mode=PaddingMode.CONSTANT``.

        Returns:
            This image with modified size, but unchanged spacing.

        """
        grids: Sequence[Grid] = self._grid
        data = U.pad(self._tensor, margin=margin, num=num, mode=mode, value=value)
        grid = [grid.pad(margin=margin, num=num) for grid in grids]
        return self.tensor_(data, grid=grid)

    def center_pad(self: TImageBatch, *args, **kwargs) -> TImageBatch:
        r"""Pad image to specified minimum size."""
        copy = shallow_copy(self)
        return copy.center_pad_(*args, **kwargs)

    def center_pad_(
        self: TImageBatch,
        size: Union[int, Array],
        *args: int,
        mode: Union[PaddingMode, str] = PaddingMode.CONSTANT,
        value: Scalar = 0,
    ) -> TImageBatch:
        r"""Pad image to specified minimum size.

        Args:
            size: Minimum output size, where the size of the last tensor
                dimension must be given first, i.e., ``(X, ...)``.
                If an ``int`` is given, all spatial output dimensions
                are cropped to this maximum size. If the length of size
                is less than the spatial dimensions of the ``data`` tensor,
                then only the last ``len(size)`` dimensions are modified.
            mode: Padding mode (cf. ``torch.nn.functional.pad()``).
            value: Value for padding mode "constant".

        """
        size_ = self._cat_scalars(size, *args)
        data = U.center_pad(self._tensor, size_, mode=mode, value=value)
        grid = [grid.center_pad(size_) for grid in self._grid]
        return self.tensor_(data, grid=grid)

    def conv(self: TImageBatch, *args, **kwargs) -> TImageBatch:
        r"""Filter images in batch with a given (separable) kernel."""
        copy = shallow_copy(self)
        return copy.conv_(*args, **kwargs)

    def conv_(
        self: TImageBatch, kernel: Tensor, padding: Union[PaddingMode, str, int] = None
    ) -> TImageBatch:
        r"""Filter images in batch with a given (separable) kernel.

        Args:
            kernel: Weights of kernel used to filter the images in this batch by.
                The dtype of the kernel defines the intermediate data type used for convolutions.
                If a 1-dimensional kernel is given, it is used as seperable convolution kernel in
                all spatial image dimensions. Otherwise, the kernel is applied to the last spatial
                image dimensions. For example, a 2D kernel applied to a batch of 3D image volumes
                is applied slice-by-slice by convoling along the X and Y image axes.
            padding: Image padding mode or margin size. If ``None``, use default mode ``PaddingMode.ZEROS``.

        Returns:
            This image batch with data tensor replaced by result of filtering operation with data
            type set to the image data type before convolution. If this data type is not a floating
            point data type, the filtered data is rounded and clamped before cast to this dtype.

        """
        data = U.conv(self._tensor, kernel, padding=padding)
        crop = tuple((m - n) // 2 for m, n in zip(self._tensor.shape[2:], data.shape[2:]))
        crop = tuple(reversed(crop))
        grid = [grid.crop(crop) for grid in self._grid]
        return self.tensor_(data, grid=grid)

    def rescale(self: TImageBatch, *args, **kwargs) -> TImageBatch:
        r"""Clamp and linearly rescale image values."""
        copy = shallow_copy(self)
        return copy.rescale_(*args, **kwargs)

    def rescale_(
        self: TImageBatch,
        min: Optional[Scalar] = None,
        max: Optional[Scalar] = None,
        data_min: Optional[Scalar] = None,
        data_max: Optional[Scalar] = None,
    ) -> TImageBatch:
        r"""Clamp and linearly rescale image values."""
        data = self.tensor()
        data = U.rescale(data, min=min, max=max, data_min=data_min, data_max=data_max)
        return self.tensor_(data)

    @overload
    def sample(
        self: TImageBatch,
        grid: Grid,
        mode: Optional[Union[Sampling, str]] = None,
        padding: Optional[Union[PaddingMode, str, Scalar]] = None,
    ) -> TImageBatch:
        r"""Sample images at optionally deformed unit grid points.

        Args:
            grid: Spatial grid points at which to sample image values.
            mode: Image interpolation mode.
            padding: Image extrapolation mode or scalar padding value.

        Returns:
            A new ``ImageBatch`` with the resampled data as tensor attribute and
            the given sampling grid instance set as grid attribute.

        """
        ...

    @overload
    def sample(
        self: TImageBatch,
        grid: Tensor,
        mode: Optional[Union[Sampling, str]] = None,
        padding: Optional[Union[PaddingMode, str, Scalar]] = None,
    ) -> Tensor:
        r"""Sample images at optionally deformed unit grid points.

        Args:
            grid: Spatial grid points at which to sample image values.
            mode: Image interpolation mode.
            padding: Image extrapolation mode or scalar padding value.

        Returns:
            A tensor of shape (N, C, ..., X) of sampled image values as ``grid`` points.

        """
        ...

    def sample(
        self: TImageBatch,
        grid: Union[Grid, Tensor],
        mode: Optional[Union[Sampling, str]] = None,
        padding: Optional[Union[PaddingMode, str, Scalar]] = None,
    ) -> Union[TImageBatch, Tensor]:
        r"""Sample images at optionally deformed unit grid points.

        Args:
            grid: Spatial grid points at which to sample image values.
            mode: Image interpolation mode.
            padding: Image extrapolation mode or scalar padding value.

        Returns:
            If ``grid`` is of type ``Grid``, an ``ImageBatch`` with the resampled data as
            tensor attribute and the given sampling grid instance set as grid attribute.
            Otherwise, a ``Tensor`` of grid sample points with respect to the unit cube is
            expected as ``grid``, and the returned value is a ``Tensor`` of the sampled values.

        """
        if isinstance(grid, Grid):
            copy = shallow_copy(self)
            return copy.sample_(grid, mode=mode, padding=padding)
        else:
            align_corners = self.align_corners()
            return U.grid_sample(
                self._tensor, grid, mode=mode, padding=padding, align_corners=align_corners
            )

    def sample_(
        self: TImageBatch,
        grid: Grid,
        mode: Optional[Union[Sampling, str]] = None,
        padding: Optional[Union[PaddingMode, str, Scalar]] = None,
    ) -> TImageBatch:
        r"""Sample images at optionally deformed unit grid points.

        Args:
            grid: Spatial grid points at which to sample image values.
            mode: Image interpolation mode.
            padding: Image extrapolation mode or scalar padding value.

        Returns:
            A reference to this modified image batch instance.

        """
        if not isinstance(grid, Grid):
            raise TypeError(
                "ImageBatch.sample_() 'grid' must by Grid; maybe try ImageBatch.sample() instead"
            )
        if all(grid == g for g in self._grid):
            return self
        align_corners = grid.align_corners()
        axes = Axes.from_align_corners(align_corners)
        coords = grid.coords(align_corners=align_corners, device=self.device).unsqueeze(0)
        points = [grid.transform_points(coords, axes, to_grid=g) for g in self._grid]
        points = torch.cat(points, dim=0)
        data = U.grid_sample(
            self._tensor,
            points,
            mode=mode,
            padding=padding,
            align_corners=align_corners,
        )
        return self.tensor_(data, grid=grid)

    def _cat_scalars(self: TImageBatch, arg, *args) -> Tensor:
        r"""Concatenate or repeat scalar function arguments."""
        return cat_scalars(arg, *args, num=self.ndim, device=self.device)


class Image(TensorDecorator):
    r"""Image sampled on oriented grid."""

    __slots__ = ("_grid",)

    def __init__(
        self: TImage, data: Array, grid: Optional[Grid] = None, device: Optional[Device] = None
    ) -> None:
        r"""Initialize image decorator.

        Args:
            data: Image data tensor of shape (C, ...X). To create an ``Image`` instance from an
                image of a mini-batch without creating a copy of the data, simply provide the
                respective slice of the mini-batch corresponding to this image, e.g., ``batch[i]``.
            grid: Sampling grid of image ``data`` oriented in world space.
                If ``None``, a default grid whose world space coordinate axes are aligned with the
                image axes, unit spacing, and origin at the image center is created on CPU.
            device: Device on which to store image data. A copy of the data is only made when
                the data has to be copied to a different device.

        """
        data_ = as_tensor(data, device=device)
        if data_.ndim < 3:
            raise ValueError("Image tensor must have at least three dimensions (C, H, W)")
        names = image_tensor_names(data_.ndim)
        if None in data_.names:
            data_ = data_.refine_names(*names)  # type: ignore
        data_: Tensor = data_.align_to(*names)  # type: ignore
        data_ = data_.rename(None)  # named tensor support still experimental
        super().__init__(data_)
        self.grid_(grid or Grid(shape=data_.shape[1:]))

    def batch(self: TImage) -> ImageBatch:
        r"""Image batch consisting of this image only.

        Because batched operations are generally more efficient, especially when executed on the GPU,
        most ``Image`` operations are implemented by ``ImageBatch``. The single-image batch instance
        property of this ``Image`` instance is used to execute single-image operations of ``self``.
        The ``ImageBatch`` uses a view on the tensor data of this ``Image``, as well as the ``Grid``
        object reference. No copies are made.

        """
        data: Tensor = self._tensor.rename(None)
        data = data.unsqueeze(0)
        return ImageBatch(data=data, grid=self._grid)

    @overload
    def tensor(self: TImage, named: bool = False) -> Tensor:
        r"""Get image data tensor."""
        ...

    @overload
    def tensor(self: TImage, data: Array, **kwargs) -> Image:
        r"""Get new image with given data tensor."""
        ...

    def tensor(
        self: TImage,
        data: Optional[Array] = None,
        grid: Optional[Grid] = None,
        named: bool = False,
        device: Optional[Device] = None,
    ) -> Union[Tensor, TImage]:
        r"""Get data tensor or new image with given tensor, respectively."""
        if data is None:
            tensor = self._tensor
            if named:
                names = image_tensor_names(tensor.ndim)
                tensor = tensor.refine_names(*names)
            return tensor
        copy = shallow_copy(self)
        return copy.tensor_(data, grid=grid, device=device)

    def tensor_(
        self: TImage,
        data: Array,
        grid: Optional[Grid] = None,
        device: Optional[Device] = None,
    ) -> TImage:
        r"""Change data tensor of this image."""
        if device is None:
            device = self.device
        if torch.is_tensor(data):
            data_ = as_tensor(data).to(device)
        else:
            data_ = torch.tensor(data, device=device)
        if data_.ndim < 3:
            raise ValueError("Image tensor must have at least three dimensions")
        if grid is None:
            grid = self._grid
        assert data_.shape[1:] == grid.shape
        names = image_tensor_names(data_.ndim)
        if None in data_.names:
            data_ = data_.refine_names(*names)  # type: ignore
        data_: Tensor = data_.align_to(*names)  # type: ignore
        data_ = data_.rename(None)  # named tensor support still experimental
        self._tensor = data_
        self._grid = grid
        return self

    @overload
    def grid(self: TImage) -> Grid:
        r"""Get sampling grid."""
        ...

    @overload
    def grid(self: TImage, grid: Grid) -> Image:
        r"""Get new image with given sampling grid."""
        ...

    def grid(self: TImage, grid: Optional[Grid] = None) -> Union[Grid, TImage]:
        r"""Get sampling grid or image with given grid, respectively."""
        if grid is None:
            return self._grid
        copy = shallow_copy(self)
        return copy.grid_(grid)

    def grid_(self: TImage, grid: Grid) -> TImage:
        r"""Change image sampling grid of this image."""
        if grid.shape != self._tensor.shape[1:]:
            raise ValueError("Image grid size does not match spatial dimensions of image tensor")
        self._grid = grid
        return self

    def align_corners(self: TImage) -> bool:
        r"""Whether image resizing operations by default preserve corner points or grid extent."""
        return self._grid.align_corners()

    @property
    def ndim(self: TImage) -> int:
        r"""Number of spatial dimensions."""
        return self._grid.ndim

    @property
    def nchannels(self: TImage) -> int:
        r"""Number of image channels."""
        return self._tensor.shape[0]

    @classmethod
    def from_sitk(
        cls: Type[TImage],
        image: "sitk.Image",
        align_corners: bool = ALIGN_CORNERS,
        dtype: Optional[torch.dtype] = None,
        device: Optional[Device] = None,
    ) -> TImage:
        r"""Create image from ``SimpleITK.Image`` instance."""
        if sitk is None:
            raise RuntimeError(f"{cls.__name__}.from_sitk() requires SimpleITK")
        data = tensor_from_image(image, dtype=dtype, device=device)
        grid = Grid.from_sitk(image, align_corners=align_corners)
        return cls(data=data, grid=grid)

    def sitk(self: TImage) -> "sitk.Image":
        r"""Create ``SimpleITK.Image`` from this image."""
        if sitk is None:
            raise RuntimeError(f"{type(self).__name__}.sitk() requires SimpleITK")
        grid = self._grid
        origin = grid.origin().tolist()
        spacing = grid.spacing().tolist()
        direction = grid.direction().flatten().tolist()
        return image_from_tensor(self._tensor, origin=origin, spacing=spacing, direction=direction)

    @classmethod
    def read(
        cls: Type[TImage],
        path: PathStr,
        align_corners: bool = ALIGN_CORNERS,
        dtype: Optional[torch.dtype] = None,
        device: Optional[Device] = None,
    ) -> TImage:
        r"""Read image data from file."""
        image = cls._read_sitk(path)
        return cls.from_sitk(image, align_corners=align_corners, dtype=dtype, device=device)

    @classmethod
    def _read_sitk(cls, path: PathStr) -> "sitk.Image":
        r"""Read SimpleITK.Image from file path."""
        if sitk is None:
            raise RuntimeError(f"{cls.__name__}.read() requires SimpleITK")
        return read_image(path)

    def write(self: TImage, path: PathStr, compress: bool = True) -> None:
        r"""Write image data to file."""
        if sitk is None:
            raise RuntimeError(f"{type(self).__name__}.write() requires SimpleITK")
        image = self.sitk()
        path = unlink_or_mkdir(path)
        sitk.WriteImage(image, str(path), compress)

    def normalize(self: TImage, *args, **kwargs) -> TImage:
        r"""Normalize image intensities in [min, max]."""
        batch = self.batch()
        return batch.normalize(*args, **kwargs)[0]

    def normalize_(self: TImage, *args, **kwargs) -> TImage:
        r"""Normalize image intensities in [min, max]."""
        batch = self.batch()
        other = batch.normalize_(*args, **kwargs)[0]
        return self.tensor_(other._tensor, grid=other._grid)

    def resize(self: TImage, *args, **kwargs) -> TImage:
        r"""Interpolate image with specified spatial image grid size."""
        batch = self.batch()
        return batch.resize(*args, **kwargs)[0]

    def resize_(self: TImage, *args, **kwargs) -> TImage:
        r"""Interpolate image with specified spatial image grid size."""
        batch = self.batch()
        other = batch.resize_(*args, **kwargs)[0]
        return self.tensor_(other._tensor, grid=other._grid)

    def resample(self: TImage, *args, **kwargs) -> TImage:
        r"""Interpolate image with specified spacing."""
        batch = self.batch()
        return batch.resample(*args, **kwargs)[0]

    def resample_(self: TImage, *args, **kwargs) -> TImage:
        r"""Interpolate image with specified spacing."""
        batch = self.batch()
        other = batch.resample_(*args, **kwargs)[0]
        return self.tensor_(other._tensor, grid=other._grid)

    def avg_pool(self: TImage, *args, **kwargs) -> TImage:
        r"""Average pooling of image data."""
        batch = self.batch()
        return batch.avg_pool(*args, **kwargs)[0]

    def avg_pool_(self: TImage, *args, **kwargs) -> TImage:
        r"""Average pooling of image data."""
        batch = self.batch()
        other = batch.avg_pool_(*args, **kwargs)[0]
        return self.tensor_(other._tensor, grid=other._grid)

    def downsample(self: TImage, *args, **kwargs) -> TImage:
        r"""Downsample image a given number of times."""
        batch = self.batch()
        return batch.downsample(*args, **kwargs)[0]

    def downsample_(self: TImage, *args, **kwargs) -> TImage:
        r"""Downsample image a given number of times."""
        batch = self.batch()
        other = batch.downsample_(*args, **kwargs)[0]
        return self.tensor_(other._tensor, grid=other._grid)

    def upsample(self: TImage, *args, **kwargs) -> TImage:
        r"""Upsample image a given number of times."""
        batch = self.batch()
        return batch.upsample(*args, **kwargs)[0]

    def upsample_(self: TImage, *args, **kwargs) -> TImage:
        r"""Upsample image a given number of times."""
        batch = self.batch()
        other = batch.upsample(*args, **kwargs)[0]
        return self.tensor_(other._tensor, grid=other._grid)

    def pyramid(self: TImage, levels: int, start: int = 0, **kwargs) -> Dict[int, TImage]:
        r"""Create Gaussian resolution pyramid."""
        batch = self.batch()
        batches = batch.pyramid(levels, start=start, **kwargs)
        return {level: batch[0] for level, batch in batches.items()}

    def crop(self: TImage, *args, **kwargs) -> TImage:
        r"""Crop image at boundary."""
        batch = self.batch()
        return batch.crop(*args, **kwargs)[0]

    def crop_(self: TImage, *args, **kwargs) -> TImage:
        r"""Crop image at boundary."""
        batch = self.batch()
        other = batch.crop_(*args, **kwargs)[0]
        return self.tensor_(other._tensor, grid=other._grid)

    def center_crop(self: TImage, *args, **kwargs) -> TImage:
        r"""Crop image to specified maximum size."""
        batch = self.batch()
        return batch.center_crop(*args, **kwargs)[0]

    def center_crop_(self: TImage, *args, **kwargs) -> TImage:
        r"""Crop image to specified maximum size."""
        batch = self.batch()
        other = batch.center_crop_(*args, **kwargs)[0]
        return self.tensor_(other._tensor, grid=other._grid)

    def pad(self: TImage, *args, **kwargs) -> TImage:
        r"""Pad image at boundary."""
        batch = self.batch()
        return batch.pad(*args, **kwargs)[0]

    def pad_(self: TImage, *args, **kwargs) -> TImage:
        r"""Pad image at boundary."""
        batch = self.batch()
        other = batch.pad_(*args, **kwargs)[0]
        return self.tensor_(other._tensor, grid=other._grid)

    def center_pad(self: TImage, *args, **kwargs) -> TImage:
        r"""Pad image to specified minimum size."""
        batch = self.batch()
        return batch.center_pad(*args, **kwargs)[0]

    def center_pad_(self: TImage, *args, **kwargs) -> TImage:
        r"""Pad image to specified minimum size."""
        batch = self.batch()
        other = batch.center_pad_(*args, **kwargs)[0]
        return self.tensor_(other._tensor, grid=other._grid)

    def conv(self: TImage, *args, **kwargs) -> TImage:
        r"""Filter image with a given (separable) kernel."""
        batch = self.batch()
        return batch.conv(*args, **kwargs)[0]

    def conv_(self: TImage, *args, **kwargs) -> TImage:
        r"""Filter image with a given (separable) kernel."""
        batch = self.batch()
        other = batch.conv_(*args, **kwargs)[0]
        return self.tensor_(other._tensor, grid=other._grid)

    def rescale(self: TImage, *args, **kwargs) -> TImage:
        r"""Clamp and linearly rescale image values."""
        batch = self.batch()
        return batch.rescale(*args, **kwargs)[0]

    def rescale_(self: TImage, *args, **kwargs) -> TImage:
        r"""Clamp and linearly rescale image values."""
        batch = self.batch()
        other = batch.rescale_(*args, **kwargs)[0]
        return self.tensor_(other._tensor, grid=other._grid)

    @overload
    def sample(self: TImage, grid: Tensor, *args, **kwargs) -> Tensor:
        r"""Sample image at optionally deformed unit grid points."""
        ...

    @overload
    def sample(self: TImage, grid: Grid, *args, **kwargs) -> TImage:
        r"""Sample image at optionally deformed unit grid points."""
        ...

    def sample(self: TImage, grid: Union[Grid, Tensor], *args, **kwargs) -> Union[Tensor, TImage]:
        r"""Sample image at optionally deformed unit grid points."""
        batch = self.batch()
        if isinstance(grid, Tensor):
            if grid.ndim == self._tensor.ndim:
                grid = grid.unsqueeze(0)
                data = batch.sample(grid, *args, **kwargs)
                assert isinstance(data, Tensor)
                assert data.shape[0] == 1
                data = data.squeeze(0)
            elif grid.ndim == self._tensor.ndim + 1:
                data = batch.sample(grid, *args, **kwargs)
                assert isinstance(data, Tensor)
            else:
                raise ValueError(
                    f"Image.sample() 'grid' tensor must be {self._tensor.ndim}-"
                    f" or {self._tensor.ndim + 1}-dimensional"
                )
            return data
        return batch.sample(grid, *args, **kwargs)[0]

    def sample_(self: TImage, grid: Grid, *args, **kwargs) -> TImage:
        r"""Sample image at optionally deformed unit grid points."""
        if not isinstance(grid, Grid):
            raise TypeError(
                "Image.sample_() 'grid' must be a Grid; maybe try Image.sample() instead"
            )
        batch = self.batch()
        other = batch.sample(grid, *args, **kwargs)[0]
        return self.tensor_(other._tensor, grid=other._grid)
