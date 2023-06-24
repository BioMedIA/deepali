r"""Modules which apply a spatial transformation to a given input data tensor.

A spatial transformer applies a :class:`.SpatialTransform`, which takes as input point coordinates
and maps these to new spatial locations, to a given input data tensor.

"""

from __future__ import annotations

from copy import copy as shallow_copy
from typing import Dict, Optional, Tuple, Union, cast, overload

from torch import Tensor
from torch.nn import Module

from ..core.enum import PaddingMode, Sampling
from ..core.grid import Axes, Grid
from ..core.types import Scalar
from ..modules import SampleImage

from .base import SpatialTransform


class ImageTransformer(Module):
    r"""Spatially transform an image.

    The :class:`.ImageTransformer` applies a :class:`.SpatialTransform` to the sampling grid
    points of the target domain, optionally followed by linear transformation from target to
    source domain, and samples the input image of shape ``(N, C, ..., X)`` at these deformed
    source grid points. If the spatial transformation is non-rigid, this is also commonly
    referred to as warping the input image.

    """

    def __init__(
        self,
        transform: SpatialTransform,
        target: Optional[Grid] = None,
        source: Optional[Grid] = None,
        sampling: Union[Sampling, str] = Sampling.LINEAR,
        padding: Union[PaddingMode, str, Scalar] = PaddingMode.BORDER,
        align_centers: bool = False,
        flip_coords: bool = False,
    ) -> None:
        r"""Initialize spatial image transformer.

        Args:
            transform: Spatial coordinate transformation which is applied to ``target`` grid points.
            target: Sampling grid of output images. If ``None``, use ``transform.axes()``.
            source: Sampling grid of input images. If ``None``, use ``target``.
            sampling: Image interpolation mode.
            padding: Image extrapolation mode or scalar out-of-domain value.
            align_centers: Whether to implicitly align the ``target`` and ``source`` centers.
                If ``True``, only the affine component of the target to source transformation
                is applied after the spatial grid ``transform``. If ``False``, also the
                translation of grid center points is considered.
            flip_coords: Whether spatial transformation applies to flipped grid point coordinates
                in the order (z, y, x). The default is grid point coordinates in the order (x, y, z).

        """
        if not isinstance(transform, SpatialTransform):
            raise TypeError(
                f"{type(self).__name__}() requires 'transform' of type SpatialTransform"
            )
        if target is None:
            target = transform.grid()
        if source is None:
            source = target
        if not isinstance(target, Grid):
            raise TypeError(f"{type(self).__name__}() 'target' must be of type Grid")
        if not isinstance(source, Grid):
            raise TypeError(f"{type(self).__name__}() 'source' must be of type Grid")
        if not transform.grid().same_domain_as(target):
            raise ValueError(
                f"{type(self).__name__}() 'target' and 'transform' grid must define the same domain"
            )
        super().__init__()
        self._transform = transform
        self._sample = SampleImage(
            target=target,
            source=source,
            sampling=sampling,
            padding=padding,
            align_centers=align_centers,
        )
        grid_coords = target.coords(flip=flip_coords).unsqueeze(0)
        self.register_buffer("grid_coords", grid_coords, persistent=False)
        self.flip_coords = bool(flip_coords)

    @overload
    def condition(self) -> Tuple[tuple, dict]:
        r"""Get arguments on which transformation is conditioned.

        Returns:
            args: Positional arguments.
            kwargs: Keyword arguments.

        """
        ...

    @overload
    def condition(self, *args, **kwargs) -> ImageTransformer:
        r"""Get new transformation which is conditioned on the specified arguments."""
        ...

    def condition(self, *args, **kwargs) -> Union[ImageTransformer, Tuple[tuple, dict]]:
        r"""Get or set data tensors and parameters on which transformation is conditioned."""
        if args:
            return shallow_copy(self).condition_(*args)
        return self._transform.condition()

    def condition_(self, *args, **kwargs) -> ImageTransformer:
        r"""Set data tensors and parameters on which this transformation is conditioned."""
        self._transform.condition_(*args, **kwargs)
        return self

    @property
    def transform(self) -> SpatialTransform:
        r"""Spatial grid transformation."""
        return self._transform

    @property
    def sample(self) -> SampleImage:
        r"""Source image sampler."""
        return self._sample

    def target_grid(self) -> Grid:
        r"""Sampling grid of output images."""
        return self._sample.target_grid()

    def source_grid(self) -> Grid:
        r"""Sampling grid of input images."""
        return self._sample.source_grid()

    def align_centers(self) -> bool:
        r"""Whether grid center points are implicitly aligned."""
        return self._sample.align_centers()

    @overload
    def forward(self, data: Tensor) -> Tensor:
        r"""Sample batch of images at spatially transformed target grid points."""
        ...

    @overload
    def forward(self, data: Tensor, mask: Tensor) -> Tuple[Tensor, Tensor]:
        r"""Sample batch of masked images at spatially transformed target grid points."""
        ...

    @overload
    def forward(self, data: Dict[str, Union[Tensor, Grid]]) -> Dict[str, Union[Tensor, Grid]]:
        r"""Sample batch of optionally masked images at spatially transformed target grid points."""
        ...

    def forward(
        self,
        data: Union[Tensor, Dict[str, Union[Tensor, Grid]]],
        mask: Optional[Tensor] = None,
    ) -> Union[Tensor, Tuple[Tensor, Tensor], Dict[str, Union[Tensor, Grid]]]:
        r"""Sample batch of images at spatially transformed target grid points."""
        grid: Tensor = cast(Tensor, self.grid_coords)
        grid = self._transform(grid, grid=True)
        if self.flip_coords:
            grid = grid.flip((-1,))
        return self._sample(grid, data, mask)


class PointSetTransformer(Module):
    r"""Spatially transform a set of points.

    The :class:`.PointSetTransformer` applies a :class:`.SpatialTransform` to a set of input points
    with coordinates defined with respect to a specified target domain. This coordinate map may
    further be followed by a linear transformation from the grid domain of the spatial transform
    to a given source domain. When no spatial transform is given, use :func:`.grid_transform_points`.

    """

    def __init__(
        self,
        transform: SpatialTransform,
        grid: Optional[Grid] = None,
        axes: Optional[Union[Axes, str]] = None,
        to_grid: Optional[Grid] = None,
        to_axes: Optional[Union[Axes, str]] = None,
    ) -> None:
        r"""Initialize point set transformer.

        This point set transformer calls :meth:`.SpatialTransform.points` with the specified module attributes.

        Args:
            transform: Spatial coordinate transformation which is applied to input points.
            grid: Grid with respect to which input points are defined. Uses ``transform.grid()`` if ``None``.
            axes: Coordinate axes with respect to which input points are defined. Uses ``transform.axes()`` if ``None``.
            to_grid: Grid with respect to which output points are defined. Same as ``grid`` if ``None``.
            to_axes: Coordinate axes to which input points should be mapped to. Same as ``axes`` if ``None``.

        """
        if not isinstance(transform, SpatialTransform):
            raise TypeError(f"{type(self).__name__}() 'transform' must be a SpatialTransform")
        if grid is None:
            grid = transform.grid()
        if axes is None:
            axes = transform.axes()
        else:
            axes = Axes.from_arg(axes)
        if to_grid is None:
            to_grid = grid
        if to_axes is None:
            to_axes = axes
        else:
            to_axes = Axes.from_arg(to_axes)
        super().__init__()
        self._transform = transform
        self._grid = grid
        self._axes = axes
        self._to_grid = to_grid
        self._to_axes = to_axes

    @overload
    def condition(self) -> Tuple[tuple, dict]:
        r"""Get arguments on which transformation is conditioned.

        Returns:
            args: Positional arguments.
            kwargs: Keyword arguments.

        """
        ...

    @overload
    def condition(self, *args, **kwargs) -> PointSetTransformer:
        r"""Get new transformation which is conditioned on the specified arguments."""
        ...

    def condition(self, *args, **kwargs) -> Union[PointSetTransformer, Tuple[tuple, dict]]:
        r"""Get or set data tensors and parameters on which transformation is conditioned."""
        if args:
            return shallow_copy(self).condition_(*args)
        return self._transform.condition()

    def condition_(self, *args, **kwargs) -> PointSetTransformer:
        r"""Set data tensors and parameters on which this transformation is conditioned."""
        self._transform.condition_(*args, **kwargs)
        return self

    @property
    def transform(self) -> SpatialTransform:
        r"""Spatial transformation."""
        return self._transform

    def target_axes(self) -> Axes:
        r"""Coordinate axes with respect to which input points are defined."""
        return self._axes

    def target_grid(self) -> Grid:
        r"""Sampling grid with respect to which input points are defined."""
        return self._grid

    def source_axes(self) -> Axes:
        r"""Coordinate axes with respect to which output points are defined."""
        return self._to_axes

    def source_grid(self) -> Grid:
        r"""Sampling grid with respect to which output points are defined."""
        return self._to_grid

    def forward(self, points: Tensor) -> Tensor:
        r"""Spatially transform a set of points."""
        return self._transform.points(
            points,
            grid=self._grid,
            axes=self._axes,
            to_grid=self._to_grid,
            to_axes=self._to_axes,
        )
