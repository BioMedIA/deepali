r"""Bounding box oriented in world space which defines a normalized domain."""

from __future__ import annotations

from copy import copy as shallow_copy
from typing import Any, Optional, Sequence, Union, overload

import numpy as np

import torch
from torch import Tensor

from .grid import ALIGN_CORNERS, Axes, Grid
from .linalg import hmm, homogeneous_matrix, homogeneous_transform
from .tensor import as_tensor, cat_scalars
from .typing import Array, Device, Shape, Size


class Cube(object):
    r"""Bounding box oriented in world space which defines a normalized domain.

    Coordinates of points within this domain can be either with respect to the world coordinate
    system or the cube defined by the bounding box where coordinate axes are parallel to the
    cube edges and have a uniform side length of 2. The latter are the normalized coordinates
    used by ``torch.nn.functional.grid_sample()``, in particular. In terms of the coordinate
    transformations, a :class:`.Cube` is thus equivalent to a :class:`.Grid` with three points
    along each dimension and ``align_corners=True``.

    A regular sampling :class:`.Grid`, on the other hand, subsamples the world space within the bounds
    defined by the cube into a number of equally sized cells or equally spaced points, respectivey.
    How the grid points relate to the faces of the cube depends on :meth:`.Grid.align_corners`.

    """

    __slots__ = ("_center", "_direction", "_extent")

    def __init__(
        self,
        extent: Optional[Union[Array, float]],
        center: Optional[Union[Array, float]] = None,
        origin: Optional[Union[Array, float]] = None,
        direction: Optional[Array] = None,
        device: Optional[Device] = None,
    ):
        r"""Initialize cube attributes.

        Args:
            extent: Extent ``(extent_x, ...)`` of the cube in world units.
            center: Cube center point ``(x, ...)`` in world space.
            origin: World coordinates ``(x, ...)`` of lower left corner.
            direction: Direction cosines defining orientation of cube in world space.
                The direction cosines are vectors that point along the cube edges.
                Each column of the matrix indicates the direction cosines of the unit vector
                that is parallel to the cube edge corresponding to that dimension.
            device: Device on which to store attributes. Uses ``"cpu"`` if ``None``.

        """
        extent = as_tensor(extent, device=device or "cpu")
        if not extent.is_floating_point():
            extent = extent.float()
        self._extent = extent
        # Set other properties AFTER _extent, which implicitly defines 'device' and 'ndim'.
        # Use in-place setters to take care of conversion and value assertions.
        if direction is None:
            direction = torch.eye(self.ndim, dtype=self.dtype, device=self.device)
        self.direction_(direction)
        # Set center to default, specified center point, or derived from origin
        if origin is None:
            self.center_(0 if center is None else center)
        elif center is None:
            # ATTENTION: This must be done AFTER extent and direction are set!
            self.origin_(origin)
        else:
            self.center_(center)
            if not torch.allclose(origin, self.origin()):
                raise ValueError("Cube() 'center' and 'origin' are inconsistent")

    def numpy(self) -> np.ndarray:
        r"""Get cube attributes as 1-dimensional NumPy array."""
        return np.concatenate(
            [
                self._extent.numpy(),
                self._center.numpy(),
                self._direction.flatten().numpy(),
            ],
            axis=0,
        )

    @classmethod
    def from_numpy(cls, attrs: Union[Sequence[float], np.ndarray], origin: bool = False) -> Cube:
        r"""Create Cube from 1-dimensional NumPy array."""
        if isinstance(attrs, np.ndarray):
            seq = attrs.astype(float).tolist()
        else:
            seq = attrs
        return cls.from_seq(seq, origin=origin)

    @classmethod
    def from_seq(cls, attrs: Sequence[float], origin: bool = False) -> Cube:
        r"""Create Cube from sequence of attribute values.

        Args:
            attrs: Array of length (D + 2) * D, where ``D=2`` or ``D=3`` is the number
                of spatial cube dimensions and array items are given as
                ``(sx, ..., cx, ..., d11, ..., d21, ....)``, where ``(sx, ...)`` is the
                cube extent, ``(cx, ...)`` the cube center coordinates, and ``(d11, ...)``
                are the cube direction cosines. The argument can be a Python list or tuple,
                NumPy array, or PyTorch tensor.
            origin: Whether ``(cx, ...)`` specifies Cube origin rather than center.

        Returns:
            Cube instance.

        """
        if len(attrs) == 8:
            d = 2
        elif len(attrs) == 15:
            d = 3
        else:
            raise ValueError(
                f"{cls.__name__}.from_seq() expected array of length 8 (D=2) or 15 (D=3)"
            )
        kwargs = dict(
            extent=attrs[0:d],
            direction=attrs[2 * d :],
        )
        if origin:
            kwargs["origin"] = attrs[d : 2 * d]
        else:
            kwargs["center"] = attrs[d : 2 * d]
        return Cube(**kwargs)

    @classmethod
    def from_grid(cls, grid: Grid, align_corners: Optional[bool] = None) -> Cube:
        r"""Get cube with respect to which normalized grid coordinates are defined."""
        if align_corners is not None:
            grid = grid.align_corners(align_corners)
        return cls(
            extent=grid.cube_extent(),
            center=grid.center(),
            direction=grid.direction(),
            device=grid.device,
        )

    def grid(
        self,
        size: Optional[Union[int, Size, Array]] = None,
        shape: Optional[Union[int, Shape, Array]] = None,
        spacing: Optional[Union[Array, float]] = None,
        align_corners: bool = ALIGN_CORNERS,
    ) -> Grid:
        r"""Create regular sampling grid which covers the world space bounded by the cube."""
        if size is None and shape is None:
            if spacing is None:
                raise ValueError(
                    "Cube.grid() requires either the desired grid 'size'/'shape' or point 'spacing'"
                )
            size = self.extent().div(spacing).round()
            size = torch.Size(size.type(torch.int).tolist())
            if align_corners:
                size = torch.Size(n + 1 for n in size)
            spacing = None
        else:
            if isinstance(size, int):
                size = (size,) * self.ndim
            if isinstance(shape, int):
                shape = (shape,) * self.ndim
            size = Grid(size=size, shape=shape).size()
        ncells = torch.tensor(size)
        if align_corners:
            ncells = ncells.sub_(1)
        ncells = ncells.to(dtype=self.dtype, device=self.device)
        grid = Grid(
            size=size,
            spacing=self.extent().div(ncells),
            center=self.center(),
            direction=self.direction(),
            align_corners=align_corners,
            device=self.device,
        )
        with torch.no_grad():
            if not torch.allclose(grid.cube_extent(), self.extent()):
                raise ValueError(
                    "Cube.grid() 'size'/'shape' times 'spacing' does not match cube extent"
                )
        return grid

    def dim(self) -> int:
        r"""Number of cube dimensions."""
        return len(self._extent)

    @property
    def ndim(self) -> int:
        r"""Number of cube dimensions."""
        return len(self._extent)

    @property
    def dtype(self) -> torch.dtype:
        r"""Get data type of cube attribute tensors."""
        return self._extent.dtype

    @property
    def device(self) -> Device:
        r"""Get device on which cube attribute tensors are stored."""
        return self._extent.device

    def clone(self) -> Cube:
        r"""Make deep copy of this instance."""
        cube = shallow_copy(self)
        for name in self.__slots__:
            value = getattr(self, name)
            if isinstance(value, Tensor):
                setattr(cube, name, value.clone())
        return cube

    def __deepcopy__(self, memo) -> Cube:
        r"""Support copy.deepcopy to clone this cube."""
        if id(self) in memo:
            return memo[id(self)]
        copy = self.clone()
        memo[id(self)] = copy
        return copy

    @overload
    def center(self) -> Tensor:
        r"""Get center point in world space."""
        ...

    @overload
    def center(self, arg: Union[float, Array], *args: float) -> Cube:
        r"""Get new cube with same orientation and extent, but specified center point."""
        ...

    def center(self, *args) -> Union[Tensor, Cube]:
        r"""Get center point in world space or new cube with specified center point."""
        if args:
            return shallow_copy(self).center_(*args)
        return self._center

    def center_(self, arg: Union[Array, float], *args: float) -> Cube:
        r"""Set center point in world space of this cube."""
        self._center = cat_scalars(arg, *args, num=self.ndim, dtype=self.dtype, device=self.device)
        return self

    @overload
    def origin(self) -> Tensor:
        r"""Get world coordinates of lower left corner."""
        ...

    @overload
    def origin(self, arg: Union[Array, float], *args: float) -> Cube:
        r"""Get new cube with specified world coordinates of lower left corner."""
        ...

    def origin(self, *args) -> Union[Tensor, Cube]:
        r"""Get origin in world space or new cube with specified origin."""
        if args:
            return shallow_copy(self).origin_(*args)
        offset = torch.matmul(self.direction(), self.spacing())
        origin = self._center.sub(offset)
        return origin

    def origin_(self, arg: Union[Array, float], *args: float) -> Cube:
        r"""Set world coordinates of lower left corner."""
        center = cat_scalars(arg, *args, num=self.ndim, dtype=self.dtype, device=self.device)
        offset = torch.matmul(self.direction(), self.spacing())
        self._center = center.add(offset)
        return self

    def spacing(self) -> Tensor:
        r"""Cube unit spacing in world space."""
        return self._extent.div(2)

    @overload
    def direction(self) -> Tensor:
        r"""Get edge direction cosines matrix."""
        ...

    @overload
    def direction(self, arg: Union[Array, float], *args: float) -> Cube:
        r"""Get new cube with specified edge direction cosines."""
        ...

    def direction(self, *args) -> Union[Tensor, Cube]:
        r"""Get edge direction cosines matrix or new cube with specified orientation."""
        if args:
            return shallow_copy(self).direction_(*args)
        return self._direction

    def direction_(self, arg: Union[Array, float], *args: float) -> Cube:
        r"""Set edge direction cosines matrix of this cube."""
        D = self.ndim
        if args:
            direction = torch.tensor((arg,) + args)
        else:
            direction = as_tensor(arg)
        direction = direction.to(dtype=self.dtype, device=self.device)
        if direction.ndim == 1:
            if direction.shape[0] != D * D:
                raise ValueError(
                    f"Cube direction must be array or square matrix with numel={D * D}"
                )
            direction = direction.reshape(D, D)
        else:
            if (
                direction.ndim != 2
                or direction.shape[0] != direction.shape[1]
                or direction.shape[0] != D
            ):
                raise ValueError(
                    f"Cube direction must be array or square matrix with numel={D * D}"
                )
        with torch.no_grad():
            if abs(direction.det().abs().item() - 1) > 1e-4:
                raise ValueError("Cube direction cosines matrix must be valid rotation matrix")
        self._direction = direction
        return self

    @overload
    def extent(self) -> Tensor:
        r"""Extent of cube in world space."""
        ...

    @overload
    def extent(self, arg: Union[float, Array], *args, float) -> Cube:
        r"""Get cube with same center and orientation but different extent."""
        ...

    def extent(self, *args) -> Union[Tensor, Cube]:
        r"""Get extent of this cube or a new cube with same center and orientation but specified extent."""
        if args:
            return shallow_copy(self).extent_(*args)
        return self._extent

    def extent_(self, arg: Union[Array, float], *args) -> Cube:
        r"""Set the extent of this cube, keeping center and orientation the same."""
        self._extent = cat_scalars(arg, *args, num=self.ndim, device=self.device, dtype=self.dtype)
        return self

    def affine(self) -> Tensor:
        r"""Affine transformation from cube to world space, excluding translation."""
        return torch.mm(self.direction(), torch.diag(self.spacing()))

    def inverse_affine(self) -> Tensor:
        r"""Affine transformation from world to cube space, excluding translation."""
        one = torch.tensor(1, dtype=self.dtype, device=self.device)
        return torch.mm(torch.diag(one / self.spacing()), self.direction().t())

    def transform(
        self,
        axes: Optional[Union[Axes, str]] = None,
        to_axes: Optional[Union[Axes, str]] = None,
        to_cube: Optional[Cube] = None,
        vectors: bool = False,
    ) -> Tensor:
        r"""Transformation of coordinates from this cube to another cube.

        Args:
            axes: Axes with respect to which input coordinates are defined.
                If ``None`` and also ``to_axes`` and ``to_cube`` is ``None``,
                returns the transform which maps from cube to world space.
            to_axes: Axes of cube to which coordinates are mapped. Use ``axes`` if ``None``.
            to_cube: Other cube. Use ``self`` if ``None``.
            vectors: Whether transformation is used to rescale and reorient vectors.

        Returns:
            If ``vectors=False``, a homogeneous coordinate transformation of shape ``(D, D + 1)``.
            Otherwise, a square transformation matrix of shape ``(D, D)`` is returned.

        """
        if axes is None and to_axes is None and to_cube is None:
            return self.transform(Axes.CUBE, Axes.WORLD, vectors=vectors)
        if axes is None:
            raise ValueError(
                "Cube.transform() 'axes' required when 'to_axes' or 'to_cube' specified"
            )
        axes = Axes(axes)
        to_axes = axes if to_axes is None else Axes(to_axes)
        if axes is Axes.GRID or to_axes is Axes.GRID:
            raise ValueError("Cube.transform() Axes.GRID is only valid for a Grid")
        if axes == to_axes and axes is Axes.CUBE_CORNERS:
            axes = to_axes = Axes.CUBE
        elif axes is Axes.CUBE_CORNERS and to_axes is Axes.WORLD:
            axes = Axes.CUBE
        elif axes is Axes.WORLD and to_axes is Axes.CUBE_CORNERS:
            to_axes = Axes.CUBE
        if axes is Axes.CUBE_CORNERS or to_axes is Axes.CUBE_CORNERS:
            raise ValueError(
                "Cube.transform() cannot map between Axes.CUBE and Axes.CUBE_CORNERS."
                " Use Cube.grid().transform() instead."
            )
        if axes == to_axes and (axes is Axes.WORLD or to_cube is None or to_cube == self):
            return torch.eye(self.ndim, dtype=self.dtype, device=self.device)
        if axes == to_axes:
            assert axes is Axes.CUBE
            cube_to_world = self.transform(Axes.CUBE, Axes.WORLD, vectors=vectors)
            world_to_cube = to_cube.transform(Axes.WORLD, Axes.CUBE, vectors=vectors)
            if vectors:
                return torch.mm(world_to_cube, cube_to_world)
            return hmm(world_to_cube, cube_to_world)
        if axes is Axes.CUBE:
            assert to_axes is Axes.WORLD
            if vectors:
                return self.affine()
            return homogeneous_matrix(self.affine(), self.center())
        assert axes is Axes.WORLD
        assert to_axes is Axes.CUBE
        if vectors:
            return self.inverse_affine()
        return hmm(self.inverse_affine(), -self.center())

    def inverse_transform(self, vectors: bool = False) -> Tensor:
        r"""Transform which maps from world to cube space."""
        return self.transform(Axes.WORLD, Axes.CUBE, vectors=vectors)

    def apply_transform(
        self,
        arg: Array,
        axes: Union[Axes, str],
        to_axes: Optional[Union[Axes, str]] = None,
        to_cube: Optional[Cube] = None,
        vectors: bool = False,
    ) -> Tensor:
        r"""Map point coordinates or displacement vectors from one cube to another.

        Args:
            arg: Coordinates of points or displacement vectors as tensor of shape ``(..., D)``.
            axes: Axes of this cube with respect to which input coordinates are defined.
            to_axes: Axes of cube to which coordinates are mapped. Use ``axes`` if ``None``.
            to_cube: Other cube. Use ``self`` if ``None``.
            vectors: Whether ``arg`` contains displacements (``True``) or point coordinates (``False``).

        Returns:
            Points or displacements with respect to ``to_cube`` and ``to_axes``.
            If ``to_cube == self`` and ``to_axes == axes`` or both ``axes`` and ``to_axes`` are
            ``Axes.WORLD`` and ``arg`` is a ``torch.Tensor``, a reference to the unmodified input
            tensor is returned.

        """
        axes = Axes(axes)
        to_axes = axes if to_axes is None else Axes(to_axes)
        if to_cube is None:
            to_cube = self
        tensor = as_tensor(arg)
        if not tensor.is_floating_point():
            tensor = tensor.type(self.dtype)
        if axes is to_axes and axes is Axes.WORLD:
            return tensor
        if (to_cube is not None and to_cube != self) or axes is not to_axes:
            matrix = self.transform(axes, to_axes, to_cube=to_cube, vectors=vectors)
            matrix = matrix.unsqueeze(0).to(device=tensor.device)
            tensor = homogeneous_transform(matrix, tensor)
        return tensor

    def transform_points(
        self,
        points: Array,
        axes: Union[Axes, str],
        to_axes: Optional[Union[Axes, str]] = None,
        to_cube: Optional[Cube] = None,
    ) -> Tensor:
        r"""Map point coordinates from one cube to another.

        Args:
            points: Coordinates of points to transform as tensor of shape ``(..., D)``.
            axes: Axes of this cube with respect to which input coordinates are defined.
            to_axes: Axes of cube to which coordinates are mapped. Use ``axes`` if ``None``.
            to_cube: Other cube. Use ``self`` if ``None``.

        Returns:
            Point coordinates with respect to ``to_cube`` and ``to_axes``. If ``to_cube == self``
            and ``to_axes == axes`` or both ``axes`` and ``to_axes`` are ``Axes.WORLD`` and ``arg``
            is a ``torch.Tensor``, a reference to the unmodified input tensor is returned.

        """
        return self.apply_transform(points, axes, to_axes, to_cube=to_cube, vectors=False)

    def transform_vectors(
        self,
        vectors: Array,
        axes: Union[Axes, str],
        to_axes: Optional[Union[Axes, str]] = None,
        to_cube: Optional[Cube] = None,
    ) -> Tensor:
        r"""Rescale and reorient flow vectors.

        Args:
            vectors: Displacement vectors as tensor of shape ``(..., D)``.
            axes: Axes of this cube with respect to which input coordinates are defined.
            to_axes: Axes of cube to which coordinates are mapped. Use ``axes`` if ``None``.
            to_cube: Other cube. Use ``self`` if ``None``.

        Returns:
            Rescaled and reoriented displacement vectors. If ``to_cube == self`` and
            ``to_axes == axes`` or both ``axes`` and ``to_axes`` are ``Axes.WORLD`` and ``arg``
            is a ``torch.Tensor``, a reference to the unmodified input tensor is returned.

        """
        return self.apply_transform(vectors, axes, to_axes, to_cube=to_cube, vectors=True)

    def cube_to_world(self, coords: Array) -> Tensor:
        r"""Map point coordinates from cube to world space.

        Args:
            coords: Normalized coordinates with respect to this cube as tensor of shape ``(..., D)``.

        Returns:
            Coordinates of points in world space.

        """
        return self.apply_transform(coords, Axes.CUBE, Axes.WORLD, vectors=False)

    def world_to_cube(self, points: Array) -> Tensor:
        r"""Map point coordinates from world to cube space.

        Args:
            points: Coordinates of points in world space as tensor of shape ``(..., D)``.

        Returns:
            Normalized coordinates of points with respect to this cube.

        """
        return self.apply_transform(points, Axes.WORLD, Axes.CUBE, vectors=False)

    def __eq__(self, other: Any) -> bool:
        r"""Compare this cube to another."""
        if other is self:
            return True
        if not isinstance(other, self.__class__):
            return False
        for name in self.__slots__:
            value = getattr(self, name)
            other_value = getattr(other, name)
            if type(value) != type(other_value):
                return False
            if isinstance(value, Tensor):
                assert isinstance(other_value, Tensor)
                if value.shape != other_value.shape:
                    return False
                other_value = other_value.to(device=value.device)
                if not torch.allclose(value, other_value, rtol=1e-5, atol=1e-8):
                    return False
            elif value != other_value:
                return False
        return True

    def __repr__(self) -> str:
        """String representation."""
        origin = ", ".join([f"{v:.5f}" for v in self.origin()])
        center = ", ".join([f"{v:.5f}" for v in self._center])
        direction = ", ".join([f"{v:.5f}" for v in self._direction.flatten()])
        extent = ", ".join([f"{v:.5f}" for v in self._extent])
        return (
            f"{type(self).__name__}("
            + f"origin=({origin})"
            + f", center=({center})"
            + f", extent=({extent})"
            + f", direction=({direction})"
            + f", device={repr(str(self.device))}"
            + ")"
        )


def cube_points_transform(cube: Cube, axes: Axes, to_cube: Cube, to_axes: Optional[Axes] = None):
    r"""Get linear transformation of points from one cube to another.

    Args:
        cube: Sampling grid with respect to which input points are defined.
        axes: Grid axes with respect to which input points are defined.
        to_cube: Sampling grid with respect to which output points are defined.
        to_axes: Grid axes with respect to which output points are defined.

    Returns:
        Homogeneous coordinate transformation matrix as tensor of shape ``(D, D + 1)``.

    """
    return cube.transform(axes=axes, to_axes=to_axes, to_cube=to_cube, vectors=False)


def cube_vectors_transform(cube: Cube, axes: Axes, to_cube: Cube, to_axes: Optional[Axes] = None):
    r"""Get affine transformation which maps vectors with respect to one cube to another.

    Args:
        cube: Cube with respect to which (normalized) input vectors are defined.
        axes: Cube axes with respect to which input vectors are defined.
        to_cube: Cube with respect to which (normalized) output vectors are defined.
        to_axes: Cube axes with respect to which output vectors are defined.

    Returns:
        Affine transformation matrix as tensor of shape ``(D, D)``.

    """
    return cube.transform(axes=axes, to_axes=to_axes, to_cube=to_cube, vectors=True)


def cube_transform_points(
    points: Tensor,
    cube: Cube,
    axes: Axes,
    to_cube: Cube,
    to_axes: Optional[Axes] = None,
):
    return cube.transform_points(points, axes=axes, to_axes=to_axes, to_cube=to_cube)


def cube_transform_vectors(
    vectors: Tensor,
    cube: Cube,
    axes: Axes,
    to_cube: Cube,
    to_axes: Optional[Axes] = None,
):
    return cube.transform_vectors(vectors, axes=axes, to_axes=to_axes, to_cube=to_cube)
