r"""Utility functions for point sets."""

from typing import Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torch import Tensor

from . import affine as A
from .flow import warp_grid, warp_points
from .grid import ALIGN_CORNERS
from .tensor import move_dim


def bounding_box(points: Tensor) -> Tuple[Tensor, Tensor]:
    r"""Compute corners of minimum axes-aligned bounding box of given points."""
    return points.amin(0), points.amax(0)


def distance_matrix(x: Tensor, y: Tensor) -> Tensor:
    r"""Compute squared Euclidean distances between all pairs of points.

    Args:
        x: Point set of shape ``(N, X, D)``.
        y: Point set of shape ``(N, Y, D)``.

    Returns:
        Tensor of distance matrices of shape ``(N, X, Y)``.

    See also:
        https://discuss.pytorch.org/t/efficient-distance-matrix-computation/9065

    """
    if not isinstance(x, Tensor) or not isinstance(y, Tensor):
        raise TypeError("distance_matrix() 'x' and 'y' must be torch.Tensor")
    if x.ndim != 3 or y.ndim != 3:
        raise ValueError("distance_matrix() 'x' and 'y' must have shape (N, X, D)")
    N, _, D = x.shape
    if y.shape[0] != N:
        raise ValueError("distance_matrix() 'x' and 'y' must have same batch size N")
    if y.shape[2] != D:
        raise ValueError("distance_matrix() 'x' and 'y' must have same point dimension D")
    out_dtype = x.dtype
    if not out_dtype.is_floating_point:
        out_dtype = torch.float32
    x = x.type(torch.float64)
    y = y.type(torch.float64)
    x_norm = x.pow(2).sum(2).view(N, -1, 1)
    y_norm = y.pow(2).sum(2).view(N, 1, -1)
    dist = x_norm + y_norm - 2.0 * torch.bmm(x, torch.transpose(y, 1, 2))
    return dist.type(out_dtype)


def closest_point_distances(x: Tensor, y: Tensor, split_size: int = 10000) -> Tensor:
    r"""Compute minimum Euclidean distance from each point in ``x`` to point set ``y``.

    Args:
        x: Point set of shape ``(N, X, D)``.
        y: Point set of shape ``(N, Y, D)``.
        split_size: Maximum number of points in ``x`` to consider each time when computing
            the full distance matrix between these points in ``x`` and every point in ``y``.
            This is required to limit the size of the distance matrix.

    Returns:
        Tensor of shape ``(N, X)`` with minimum distances from points in ``x`` to points in ``y``.

    """
    if not isinstance(x, Tensor) or not isinstance(y, Tensor):
        raise TypeError("closest_point_distances() 'x' and 'y' must be torch.Tensor")
    if x.ndim != 3 or y.ndim != 3:
        raise ValueError("closest_point_distances() 'x' and 'y' must have shape (N, X, D)")
    N, _, D = x.shape
    if y.shape[0] != N:
        raise ValueError("closest_point_distances() 'x' and 'y' must have same batch size N")
    if y.shape[2] != D:
        raise ValueError("closest_point_distances() 'x' and 'y' must have same point dimension D")
    x = x.float()
    y = y.type(x.dtype)
    min_dists = torch.empty(x.shape[0:2], dtype=x.dtype, device=x.device)
    for i, points in enumerate(x.split(split_size, dim=1)):
        dists = distance_matrix(points, y)
        j = slice(i * split_size, i * split_size + points.shape[1])
        min_dists[:, j] = torch.min(dists, dim=2).values
    return min_dists


def closest_point_indices(x: Tensor, y: Tensor, split_size: int = 10000) -> Tensor:
    r"""Determine indices of points in ``y`` with minimum Euclidean distance from each point in ``x``.

    Args:
        x: Point set of shape ``(N, X, D)``.
        y: Point set of shape ``(N, Y, D)``.
        split_size: Maximum number of points in ``x`` to consider each time when computing
            the full distance matrix between these points in ``x`` and every point in ``y``.
            This is required to limit the size of the distance matrix.

    Returns:
        Tensor of shape ``(N, X)`` with indices of closest points in ``y``.

    """
    if not isinstance(x, Tensor) or not isinstance(y, Tensor):
        raise TypeError("closest_point_indices() 'x' and 'y' must be torch.Tensor")
    if x.ndim != 3 or y.ndim != 3:
        raise ValueError("closest_point_indices() 'x' and 'y' must have shape (N, X, D)")
    N, _, D = x.shape
    if y.shape[0] != N:
        raise ValueError("closest_point_indices() 'x' and 'y' must have same batch size N")
    if y.shape[2] != D:
        raise ValueError("closest_point_indices() 'x' and 'y' must have same point dimension D")
    x = x.float()
    y = y.type(x.dtype)
    indices = torch.empty(x.shape[0:2], dtype=torch.int64, device=x.device)
    for i, points in enumerate(x.split(split_size, dim=1)):
        dists = distance_matrix(points, y)
        j = slice(i * split_size, i * split_size + points.shape[1])
        indices[:, j] = torch.min(dists, dim=2).indices
    return indices


def normalize_grid(
    grid: Tensor,
    size: Optional[Union[Tensor, torch.Size]] = None,
    side_length: float = 2,
    align_corners: bool = ALIGN_CORNERS,
    channels_last: bool = True,
) -> Tensor:
    r"""Map unnormalized grid coordinates to normalized grid coordinates."""
    if not isinstance(grid, Tensor):
        raise TypeError("normalize_grid() 'grid' must be tensors")
    if not grid.is_floating_point():
        grid = grid.float()
    if size is None:
        if channels_last:
            if grid.ndim < 4 or grid.shape[-1] != grid.ndim - 2:
                raise ValueError(
                    "normalize_grid() 'grid' must have shape (N, ..., X, D) when 'size' not given"
                )
            size = torch.Size(reversed(grid.shape[1:-1]))  # X,...
        else:
            if grid.ndim < 4 or grid.shape[1] != grid.ndim - 2:
                raise ValueError(
                    "normalize_grid() 'grid' must have shape (N, D, ..., X) when 'size' not given"
                )
            size = torch.Size(reversed(grid.shape[2:]))  # X,...
    zero = torch.tensor(0, dtype=grid.dtype, device=grid.device)
    size = torch.as_tensor(size, dtype=grid.dtype, device=grid.device)
    size_ = size.sub(1) if align_corners else size
    if not channels_last:
        grid = move_dim(grid, 1, -1)
    if side_length != 1:
        grid = grid.mul(side_length)
    grid = torch.where(size > 1, grid.div(size_).sub(1), zero)
    if not channels_last:
        grid = move_dim(grid, -1, 1)
    return grid


def denormalize_grid(
    grid: Tensor,
    size: Optional[Union[Tensor, torch.Size]] = None,
    side_length: float = 2,
    align_corners: bool = ALIGN_CORNERS,
    channels_last: bool = True,
) -> Tensor:
    r"""Map normalized grid coordinates to unnormalized grid coordinates."""
    if not isinstance(grid, Tensor):
        raise TypeError("denormalize_grid() 'grid' must be tensors")
    if size is None:
        if grid.ndim < 4 or grid.shape[-1] != grid.ndim - 2:
            raise ValueError(
                "normalize_grid() 'grid' must have shape (N, ..., X, D) when 'size' not given"
            )
        size = torch.Size(reversed(grid.shape[1:-1]))  # X,...
    zero = torch.tensor(0, dtype=grid.dtype, device=grid.device)
    size = torch.as_tensor(size, dtype=grid.dtype, device=grid.device)
    size_ = size.sub(1) if align_corners else size
    if not channels_last:
        grid = move_dim(grid, 1, -1)
    grid = torch.where(size > 1, grid.add(1).mul(size_), zero)
    if side_length != 1:
        grid = grid.div(side_length)
    if not channels_last:
        grid = move_dim(grid, -1, 1)
    return grid


def polyline_directions(
    points: Tensor, normalize: bool = False, repeat_last: bool = True
) -> Tensor:
    r"""Compute proximal to distal facing tangent vectors."""
    if not isinstance(points, Tensor):
        raise TypeError("polyline_directions() 'points' must be Tensor")
    if points.ndim < 2:
        raise ValueError("polyline_directions() 'points' must have shape (..., N, 3)")
    dim = points.ndim - 2
    n = points.shape[dim]
    d = points.narrow(dim, 1, n - 1).sub(points.narrow(dim, 0, n - 1))
    if normalize:
        d = F.normalize(d, p=2, dim=dim)
    if repeat_last:
        d = torch.cat([d, d.narrow(dim, n - 2, 1)], dim=dim)
    return d


def polyline_tangents(points: Tensor, normalize: bool = False, repeat_first: bool = True) -> Tensor:
    r"""Compute distal to proximal facing tangent vectors."""
    if not isinstance(points, Tensor):
        raise TypeError("polyline_tangents() 'points' must be Tensor")
    if points.ndim < 2:
        raise ValueError("polyline_tangents() 'points' must have shape (..., N, 3)")
    dim = points.ndim - 2
    n = points.shape[dim]
    d = points.narrow(dim, 0, n - 1).sub(points.narrow(dim, 1, n - 1))
    if normalize:
        d = F.normalize(d, p=2, dim=dim)
    if repeat_first:
        d = torch.cat([d.narrow(dim, 0, 1), d], dim=dim)
    return d


def transform_grid(transform: Tensor, grid: Tensor, align_corners: bool = ALIGN_CORNERS) -> Tensor:
    r"""Transform undeformed grid by a spatial transformation.

    This function applies a spatial transformation to map a tensor of undeformed grid points to a
    tensor of deformed grid points with the same shape as the input tensor. The input points must be
    the positions of undeformed spatial grid points, because in case of a non-rigid transformation,
    this function uses interpolation to resize the vector fields to the size of the input ``grid``.
    This assumes that input points ``x`` are the coordinates of points located on a regularly spaced
    undeformed grid which is aligned with the borders of the grid domain on which the vector fields
    of the non-rigid transformations are sampled, i.e., ``y = x + u``.

    In case of a linear transformation ``y = Ax + t``.

    If in doubt whether the input points will be sampled regularly at grid points of the domain of
    the spatial transformation, use ``transform_points()`` instead.

    Args:
        transform: Tensor representation of spatial transformation, where the shape of the tensor
            determines the type of transformation. A translation-only transformation must be given
            as tensor of shape ``(N, D, 1)``. An affine-only transformation without translation can
            be given as tensor of shape ``(N, D, D)``, and an affine transformation with translation
            as tensor of shape ``(N, D, D + 1)``. Flow fields of non-rigid transformations, on the
            other hand, are tensors of shape ``(N, D, ..., X)``, i.e., linear transformations are
            represented by 3-dimensional tensors, and non-rigid transformations by tensors of at least
            4 dimensions. If batch size is one, but the batch size of ``points`` is greater than one,
            all point sets are transformed by the same non-rigid transformation.
        grid: Coordinates of undeformed grid points as tensor of shape ``(N, ..., D)`` or ``(1, ..., D)``.
            If batch size is one, but multiple flow fields are given, this single point set is
            transformed by each non-rigid transformation to produce ``N`` output point sets.
        align_corners: Whether flow vectors in case of a non-rigid transformation are with respect to
            ``Axes.CUBE`` (False) or ``Axes.CUBE_CORNERS`` (True). The input ``grid`` points must be
            with respect to the same spatial grid domain as the input flow fields. This option is in
            particular passed on to the ``grid_reshape()`` function used to resize the flow fields to
            the shape of the input grid.

    Returns:
        Tensor of shape ``(N, ..., D)`` with coordinates of spatially transformed points.

    """
    if not isinstance(transform, Tensor):
        raise TypeError("transform_grid() 'transform' must be Tensor")
    if transform.ndim < 3:
        raise ValueError("transform_grid() 'transform' must be at least 3-dimensional tensor")
    if transform.ndim == 3:
        return A.transform_points(transform, grid)
    return warp_grid(transform, grid, align_corners=align_corners)


def transform_points(
    transform: Tensor, points: Tensor, align_corners: bool = ALIGN_CORNERS
) -> Tensor:
    r"""Transform set of points by a tensor of non-rigid flow fields.

    This function applies a spatial transformation to map a tensor of points to a tensor of transformed
    points of the same shape as the input tensor. Unlike ``transform_grid()``, it can be used to spatially
    transform any set of points which are defined with respect to the grid domain of the spatial transformation,
    including a tensor of shape ``(N, M, D)``, i.e., a batch of N point sets with cardianality M. It can also
    be applied to a tensor of grid points of shape ``(N, ..., X, D)`` regardless if the grid points are located
    at the undeformed grid positions or an already deformed grid. Therefore, in case of a non-rigid transformation,
    the given flow fields are sampled at the input points ``x`` using linear interpolation. The flow vectors ``u(x)``
    are then added to the input points, i.e., ``y = x + u(x)``.

    In case of a linear transformation ``y = Ax + t``.

    Args:
        transform: Tensor representation of spatial transformation, where the shape of the tensor
            determines the type of transformation. A translation-only transformation must be given
            as tensor of shape ``(N, D, 1)``. An affine-only transformation without translation can
            be given as tensor of shape ``(N, D, D)``, and an affine transformation with translation
            as tensor of shape ``(N, D, D + 1)``. Flow fields of non-rigid transformations, on the
            other hand, are tensors of shape ``(N, D, ..., X)``, i.e., linear transformations are
            represented by 3-dimensional tensors, and non-rigid transformations by tensors of at least
            4 dimensions. If batch size is one, but the batch size of ``points`` is greater than one,
            all point sets are transformed by the same non-rigid transformation.
        points: Coordinates of points given as tensor of shape ``(N, ..., D)`` or ``(1, ..., D)``.
            If batch size is one, but multiple flow fields are given, this single point set is
            transformed by each non-rigid transformation to produce ``N`` output point sets.
        align_corners: Whether flow vectors in case of a non-rigid transformation are with respect to
            ``Axes.CUBE`` (False) or ``Axes.CUBE_CORNERS`` (True). The input ``points`` must be
            with respect to the same spatial grid domain as the input flow fields. This option is in
            particular passed on to the ``grid_sample()`` function used to sample the flow vectors at
            the input points.

    Returns:
        Tensor of shape ``(N, ..., D)`` with coordinates of spatially transformed points.

    """
    if not isinstance(transform, Tensor):
        raise TypeError("transform_points() 'transform' must be Tensor")
    if transform.ndim < 3:
        raise ValueError("transform_points() 'transform' must be at least 3-dimensional tensor")
    if transform.ndim == 3:
        return A.transform_points(transform, points)
    return warp_points(transform, points, align_corners=align_corners)
