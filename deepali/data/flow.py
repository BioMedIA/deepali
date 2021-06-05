r"""Flow vector fields."""

from __future__ import annotations

from typing import Optional, Sequence, Type, TypeVar, Union, overload

import torch
from torch import Tensor

from ..core.enum import PaddingMode, Sampling
from ..core import flow as U
from ..core.grid import Axes, Grid
from ..core.tensor import move_dim
from ..core.types import Array, Device, DType, PathStr

from .image import Image, ImageBatch


TFlowField = TypeVar("TFlowField", bound="FlowField")
TFlowFields = TypeVar("TFlowFields", bound="FlowFields")


__all__ = ("FlowField", "FlowFields")


class FlowFields(ImageBatch):
    r"""Batch of flow fields."""

    def __init__(
        self: TFlowFields,
        data: Union[Array, ImageBatch],
        grid: Optional[Union[Grid, Sequence[Grid]]] = None,
        axes: Optional[Axes] = None,
        dtype: Optional[DType] = None,
        device: Optional[Device] = None,
        requires_grad: Optional[bool] = None,
        pin_memory: bool = False,
    ) -> None:
        r"""Initialize flow fields.

        Args:
            data: Batch data tensor of shape (N, D, ...X), where N is the batch size, and D
                must be equal the number of spatial dimensions. The order of the image channels
                must be such that vector components are in the order ``(x, ...)``.
            grid: Flow field sampling grids. If not otherwise specified, this attribute
                defines the fixed target image domain on which to resample a moving source image.
            axes: Axes with respect to which vectors are defined. By default, it is assumed that
                vectors are with respect to the unit ``grid`` cube in ``[-1, 1]^D``, where D are the
                number of spatial dimensions. If ``grid.align_corners() == False``, the extrema
                ``(-1, 1)`` refer to the boundary of the vector field ``grid``. Otherwise, the
                extrema coincide with the corner points of the sampling grid.
            dtype: Data type of the image data. A copy of the data is only made when the desired ``dtype``
                is not ``None`` and not the same as ``data.dtype``.
            device: Device on which to store image data. A copy of the data is only made when the data
                has to be copied to a different device.
            requires_grad: If autograd should record operations on the returned image tensor.
            pin_memory: If set, returned image tensor would be allocated in the pinned memory.
                Works only for CPU tensors.

        """
        # DataTensor.__new__() creates the tensor subclass given arguments:
        # data, dtype, device, requires_grad, pin_memory
        if grid is None and isinstance(data, ImageBatch):
            grid = data.grid()
            data = data.tensor()
        super().__init__(
            data,
            grid,
            dtype=dtype,
            device=device,
            requires_grad=requires_grad,
            pin_memory=pin_memory,
        )
        if self.shape[1] != self.sdim:
            raise ValueError(
                f"{type(self).__name__}() 'data' nchannels={self.shape[1]} must be equal spatial ndim={self.sdim}"
            )
        if axes is None:
            axes = Axes.from_grid(self._grid[0])
        else:
            axes = Axes.from_arg(axes)
        self._axes = axes

    def _make_instance(
        self: TFlowFields,
        data: Tensor,
        grid: Optional[Sequence[Grid]] = None,
        axes: Optional[Axes] = None,
        **kwargs,
    ) -> TFlowFields:
        r"""Create a new instance while preserving subclass meta-data."""
        kwargs["axes"] = axes or self._axes
        return super()._make_instance(data, grid, **kwargs)

    def __getitem__(self: TFlowFields, index: int) -> FlowField:
        r"""Get flow field at specified batch index."""
        # Attention: Tensor.__getitem__ leads to endless recursion!
        data = self.tensor().narrow(0, index, 1).squeeze(0)
        return FlowField(data, self._grid[index], self._axes)

    @overload
    def axes(self: TFlowFields) -> Axes:
        r"""Get axes with respect to which flow vectors are defined."""
        ...

    @overload
    def axes(self: TFlowFields, axes: Axes) -> TFlowFields:
        r"""Get new batch of flow fields with flow vectors defined with respect to specified axes."""
        ...

    def axes(self: TFlowFields, axes: Optional[Axes] = None) -> Union[Axes, TFlowFields]:
        r"""Rescale and reorient vectors."""
        if axes is None:
            return self._axes
        data = self.tensor()
        data = move_dim(data, 1, -1)
        data = tuple(
            grid.transform_vectors(data[i : i + 1], axes=self._axes, to_axes=axes)
            for i, grid in enumerate(self._grid)
        )
        data = torch.cat(data, dim=0)
        data = move_dim(data, -1, 1)
        return self._make_instance(data, self._grid, axes)

    def curl(self: TFlowFields, mode: str = "central") -> ImageBatch:
        if self.ndim not in (2, 3):
            raise RuntimeError("Cannot compute curl of {self.ndim}-dimensional flow field")
        spacing = self.spacing()
        data = self.tensor()
        data = U.curl(data, spacing=spacing, mode=mode)
        return ImageBatch(data, self._grid)

    def exp(
        self: TFlowFields,
        scale: Optional[float] = None,
        steps: Optional[int] = None,
        sampling: Union[Sampling, str] = Sampling.LINEAR,
        padding: Union[PaddingMode, str] = PaddingMode.BORDER,
    ) -> TFlowFields:
        r"""Group exponential maps of flow fields computed using scaling and squaring."""
        axes = self._axes
        align_corners = axes is Axes.CUBE_CORNERS
        flow = self.axes(Axes.CUBE_CORNERS if align_corners else Axes.CUBE)
        data = self.tensor()
        data = U.expv(
            data,
            scale=scale,
            steps=steps,
            sampling=sampling,
            padding=padding,
            align_corners=align_corners,
        )
        flow = self._make_instance(data, flow._grid, flow._axes)
        flow = flow.axes(axes)  # restore original axes
        return flow

    def warp_image(
        self: TFlowFields,
        image: Union[Image, ImageBatch],
        sampling: Optional[Union[Sampling, str]] = None,
        padding: Optional[Union[PaddingMode, str]] = None,
    ) -> ImageBatch:
        r"""Deform given image (batch) using this batch of vector fields.

        Args:
            image: Single image or image batch. If a single ``Image`` is given, it is deformed by
                all the displacement fields in this batch. If an ``ImageBatch`` is given, the number
                of images in the batch must match the number of displacement fields in this batch.
            sampling: Interpolation mode for sampling values from ``image`` at deformed grid points.
            padding: Extrapolation mode for sampling values outside ``image`` domain.

        Returns:
            Batch of input images deformed by the vector fields of this batch.

        """
        if isinstance(image, Image):
            image = image.batch()
        align_corners = self._axes is Axes.CUBE_CORNERS
        grid = (g.coords(align_corners=align_corners, device=self.device) for g in self._grid)
        grid = torch.cat(tuple(g.unsqueeze(0) for g in grid), dim=0)
        flow = self.axes(Axes.from_align_corners(align_corners))
        flow = flow.tensor()
        flow = move_dim(flow, 1, -1)
        data = image.tensor()
        data = U.warp_image(
            data,
            grid,
            flow=flow,
            mode=sampling,
            padding=padding,
            align_corners=align_corners,
        )
        return image._make_instance(data, self._grid)


class FlowField(Image):
    r"""Flow field image.

    A (dense) flow field is a vector image where the number of channels equals the number of spatial dimensions.
    The starting points of the vectors are defined on a regular oriented sampling grid positioned in world space.
    Orientation and scale of the vectors are defined with respect to a specified regular grid domain, which either
    coincides with the sampling grid, the world coordinate system, or the unit cube with side length 2 centered at
    the center of the sampling grid with axes parallel to the sampling grid. This unit cube domain is used by the
    ``torch.nn.functional.grid_sample()`` and ``torch.nn.functional.interpolate()`` functions.

    When a flow field is convert to a ``SimpleITK.Image``, the vectors are by default reoriented and rescaled such
    that these are with respect to the world coordinate system, a format common to ITK functions and other toolkits.

    """

    def __init__(
        self: TFlowField,
        data: Array,
        grid: Optional[Grid] = None,
        axes: Optional[Axes] = None,
        dtype: Optional[DType] = None,
        device: Optional[Device] = None,
        requires_grad: Optional[bool] = None,
        pin_memory: bool = False,
    ) -> None:
        r"""Initialize flow field.

        Args:
            data: Flow field data tensor of shape (C, ...X), where C must be equal the number of spatial dimensions.
                The order of the image channels must be such that vector components are in the order X, Y,...
            grid: Flow field sampling grid. If not otherwise specified, this attribute often also defines the fixed
                target image domain on which to resample a moving source image.
            axes: Axes with respect to which vectors are defined. By default, it is assumed that vectors are with
                respect to the unit ``grid`` cube in ``[-1, 1]^D``, where D are the number of spatial dimensions.
                If ``None`` and ``grid.align_corners() == False``, the extrema ``(-1, 1)`` refer to the boundary of
                the vector field ``grid``, and coincide with the grid corner points otherwise.
            dtype: Data type of the image data. A copy of the data is only made when the desired ``dtype``
                is not ``None`` and not the same as ``data.dtype``.
            device: Device on which to store image data. A copy of the data is only made when the data
                has to be copied to a different device.
            requires_grad: If autograd should record operations on the returned image tensor.
            pin_memory: If set, returned image tensor would be allocated in the pinned memory.
                Works only for CPU tensors.

        """
        # DataTensor.__new__() creates the tensor subclass given arguments:
        # data, dtype, device, requires_grad, pin_memory
        super().__init__(
            data,
            grid,
            dtype=dtype,
            device=device,
            requires_grad=requires_grad,
            pin_memory=pin_memory,
        )
        if self.nchannels != self._grid.ndim:
            raise ValueError(
                f"{type(self).__name__} nchannels={self.nchannels} must be equal grid.ndim={self._grid.ndim}"
            )
        if axes is None:
            axes = Axes.from_grid(self._grid)
        else:
            axes = Axes.from_arg(axes)
        self._axes = axes

    def _make_instance(
        self: TFlowField,
        data: Tensor,
        grid: Optional[Grid] = None,
        axes: Optional[Axes] = None,
        **kwargs,
    ) -> TFlowField:
        r"""Create a new instance while preserving subclass meta-data."""
        kwargs["axes"] = axes or self._axes
        return super()._make_instance(data, grid, **kwargs)

    @classmethod
    def from_image(cls: Type[TFlowField], image: Image, axes: Optional[Axes] = None) -> TFlowField:
        r"""Create flow field from image instance."""
        return cls(image, image._grid, axes)

    def batch(self: TFlowField) -> FlowFields:
        r"""Batch of flow fields containing only this flow field."""
        data = self.unsqueeze(0)
        return FlowFields(data, self._grid, self._axes)

    @overload
    def axes(self: TFlowField) -> Axes:
        r"""Get axes with respect to which flow vectors are defined."""
        ...

    @overload
    def axes(self: TFlowField, axes: Axes) -> TFlowField:
        r"""Get new flow field with flow vectors defined with respect to specified axes."""
        ...

    def axes(self: TFlowField, axes: Optional[Axes] = None) -> Union[Axes, TFlowField]:
        r"""Rescale and reorient vectors with respect to specified axes."""
        if axes is None:
            return self._axes
        batch = self.batch()
        batch = batch.axes(axes)
        data = batch.tensor().squeeze(0)
        axes = batch.axes()
        return self._make_instance(data, self._grid, axes)

    @classmethod
    def from_sitk(
        cls: Type[TFlowField], image: "sitk.Image", axes: Optional[Axes] = None
    ) -> TFlowField:
        r"""Create vector field from ``SimpleITK.Image``."""
        image = super().from_sitk(image)
        return cls.from_image(image, axes=axes or Axes.WORLD)

    def sitk(self: TFlowField, axes: Optional[Axes] = None) -> "sitk.Image":
        r"""Create ``SimpleITK.Image`` from this vector field."""
        disp = self.detach()
        assert isinstance(disp, type(self))
        disp = disp.axes(axes or Axes.WORLD)
        return Image.sitk(disp)

    @classmethod
    def read(cls: Type[TFlowField], path: PathStr, axes: Optional[Axes] = None) -> TFlowField:
        r"""Read image data from file."""
        image = cls._read_sitk(path)
        return cls.from_sitk(image, axes)

    def exp(self: TFlowField, **kwargs) -> TFlowField:
        r"""Group exponential map of vector field computed using scaling and squaring."""
        batch = self.batch()
        flow = batch.exp(**kwargs)[0]
        return type(self).from_image(flow)

    def curl(self: TFlowField, **kwargs) -> Image:
        r"""Compute curl of vector field."""
        batch = self.batch()
        rotvec = batch.curl(**kwargs)
        return rotvec[0]

    @overload
    def warp_image(self: TFlowField, image: Image, **kwargs) -> Image:
        r"""Deform given image using this displacement field."""
        ...

    @overload
    def warp_image(self: TFlowField, image: ImageBatch, **kwargs) -> ImageBatch:
        r"""Deform images in batch using this displacement field."""
        ...

    def warp_image(
        self: TFlowField, image: Union[Image, ImageBatch], **kwargs
    ) -> Union[Image, ImageBatch]:
        r"""Deform given image (batch) using this displacement field.

        Args:
            image: Single image or batch of images.
            kwargs: Keyword arguments to pass on to ``ImageBatch.warp()``.

        Returns:
            If ``image`` is an ``ImageBatch``, each image in the batch is deformed by this flow field
            and a batch of deformed images is returned. Otherwise, a single deformed image is returned.

        """
        batch = self.batch()
        result = batch.warp_image(image, **kwargs)
        if isinstance(image, Image) and len(result) == 1:
            return result[0]
        return result
