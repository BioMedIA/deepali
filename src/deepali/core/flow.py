r"""Functions relating to tensors representing vector fields."""

from typing import Optional, Union

import torch
from torch import Tensor
from torch.nn import functional as F

from . import affine as A
from .enum import PaddingMode, Sampling
from .grid import ALIGN_CORNERS, Grid
from .image import check_sample_grid, grid_reshape, grid_sample
from .image import spatial_derivatives
from .image import _image_size, zeros_image
from .tensor import move_dim
from .typing import Array, Device, DType, Scalar, Shape, Size


def affine_flow(matrix: Tensor, grid: Union[Grid, Tensor], channels_last: bool = False) -> Tensor:
    r"""Compute dense flow field from homogeneous transformation.

    Args:
        matrix: Homogeneous coordinate transformation matrices of shape ``(N, D, 1)`` (translation),
            ``(N, D, D)`` (affine), or ``(N, D, D + 1)`` (homogeneous), respectively.
        grid: Image sampling ``Grid`` or tensor of shape ``(N, ..., X, D)`` of points at
            which to sample flow fields. If an object of type ``Grid`` is given, the value
            of ``grid.align_corners()`` determines if output flow vectors are with respect to
            ``Axes.CUBE`` (False) or ``Axes.CUBE_CORNERS`` (True), respectively.
        channels_last: If ``True``, flow vector components are stored in the last dimension
            of the output tensor, and first dimension otherwise.

    Returns:
        Tensor of shape ``(N, C, ..., X)`` if ``channels_last=False`` and ``(N, ..., X, C)``, otherwise.

    """
    if matrix.ndim != 3:
        raise ValueError(
            f"affine_flow() 'matrix' must be tensor of shape (N, D, 1|D|D+1), not {matrix.shape}"
        )
    device = matrix.device
    if isinstance(grid, Grid):
        grid = grid.coords(device=device)
        grid = grid.unsqueeze(0)
    elif grid.ndim < 3:
        raise ValueError(
            f"affine_flow() 'grid' must be tensor of shape (N, ...X, D), not {grid.shape}"
        )
    assert grid.device == device
    flow = A.transform_points(matrix, grid) - grid
    if not channels_last:
        flow = move_dim(flow, -1, 1)
    assert flow.device == device
    return flow


def compose_flows(a: Tensor, b: Tensor, align_corners: bool = True) -> Tensor:
    r"""Compute composite flow field ``c = b o a``."""
    a = move_dim(b, 1, -1)
    c = F.grid_sample(b, a, mode="bilinear", padding_mode="border", align_corners=align_corners)
    return c


def curl(
    flow: Tensor, spacing: Optional[Union[Scalar, Array]] = None, mode: str = "central"
) -> Tensor:
    r"""Calculate curl of vector field.

    TODO: Implement curl for 2D vector field.

    Args:
        flow: Vector field as tensor of shape ``(N, 3, Z, Y, X)``.
        spacing: Physical size of image voxels used to compute ``spatial_derivatives()``.
        mode: Mode of ``spatial_derivatives()`` approximation.

    Returns:
        In case of a 3D input vector field, output is another 3D vector field of rotation vectors,
        where axis of rotation corresponds to the unit vector and rotation angle to the magnitude
        of the rotation vector, as tensor of shape ``(N, 3, Z, Y, X)``.

    """
    if flow.ndim == 4:
        if flow.shape[1] != 2:
            raise ValueError("curl() 'flow' must have shape (N, 2, Y, X)")
        raise NotImplementedError("curl() of 2-dimensional vector field")
    if flow.ndim == 5:
        if flow.shape[1] != 3:
            raise ValueError("curl() 'flow' must have shape (N, 3, Z, Y, X)")
        dx = spatial_derivatives(flow.narrow(1, 0, 1), mode=mode, which=("y", "z"), spacing=spacing)
        dy = spatial_derivatives(flow.narrow(1, 1, 1), mode=mode, which=("x", "z"), spacing=spacing)
        dz = spatial_derivatives(flow.narrow(1, 2, 1), mode=mode, which=("x", "y"), spacing=spacing)
        rotvec = torch.cat(
            [
                dz["y"] - dy["z"],
                dx["z"] - dz["x"],
                dy["x"] - dx["y"],
            ],
            dim=1,
        )
        return rotvec
    raise ValueError("curl() 'flow' must be 2- or 3-dimensional vector field")


def expv(
    flow: Tensor,
    scale: Optional[float] = None,
    steps: Optional[int] = None,
    sampling: Union[Sampling, str] = Sampling.LINEAR,
    padding: Union[PaddingMode, str] = PaddingMode.BORDER,
    align_corners: bool = ALIGN_CORNERS,
) -> Tensor:
    r"""Group exponential maps of flow fields computed using scaling and squaring.

    Args:
        flow: Batch of flow fields as tensor of shape ``(N, D, ..., X)``.
        scale: Constant flow field scaling factor.
        steps: Number of scaling and squaring steps.
        sampling: Flow field interpolation mode.
        padding: Flow field extrapolation mode.
        align_corners: Whether ``flow`` vectors are defined with respect to
            ``Axes.CUBE`` (False) or ``Axes.CUBE_CORNERS`` (True).

    Returns:
        Exponential map of input flow field. If ``steps=0``, a reference to ``flow`` is returned.

    """
    if scale is None:
        scale = 1
    if steps is None:
        steps = 5
    if not isinstance(steps, int):
        raise TypeError("expv() 'steps' must be of type int")
    if steps < 0:
        raise ValueError("expv() 'steps' must be positive value")
    if steps == 0:
        return flow
    device = flow.device
    grid = Grid(shape=flow.shape[2:], align_corners=align_corners)
    grid = grid.coords(dtype=flow.dtype, device=device)
    assert grid.device == device
    disp = flow * (scale / 2**steps)
    assert disp.device == device
    for _ in range(steps):
        disp = disp + warp_image(
            disp,
            grid,
            flow=move_dim(disp, 1, -1),  # channels last
            mode=sampling,
            padding=padding,
            align_corners=align_corners,
        )
        assert disp.device == device
    return disp


def jacobian_det(u: torch.Tensor, mode: str = "central", channels_last: bool = False) -> Tensor:
    r"""Evaluate Jacobian determinant of given flow field using finite difference approximations.

    Note that for differentiable parametric spatial transformation models, an accurate Jacobian could
    be calculated instead from the given transformation parameters. See for example ``cubic_bspline_jacobian_det()``
    which is specialized for a free-form deformation (FFD) determined by a continuous cubic B-spline function.

    Args:
        u: Input vector field as tensor of shape ``(N, D, ..., X)`` when ``channels_last=False`` and
            shape ``(N, ..., X, D)`` when ``channels_last=True``.
        mode: Mode of ``spatial_derivatives()`` to use for approximating spatial partial derivatives.
        channels_last: Whether input vector field has vector (channels) dimension at second or last index.

    Returns:
        Scalar field of approximate Jacobian determinant values as tensor of shape ``(N, 1, ..., X)`` when
        ``channels_last=False`` and ``(N, ..., X, 1)`` when ``channels_last=True``.

    """
    if u.ndim < 4:
        shape_str = "(N, ..., X, D)" if channels_last else "(N, D, ..., X)"
        raise ValueError(f"jacobian_det() 'u' must be dense vector field of shape {shape_str}")
    shape = u.shape[1:-1] if channels_last else u.shape[2:]
    mat = torch.empty((u.shape[0],) + shape + (3, 3), dtype=u.dtype, device=u.device)
    for i, which in enumerate(("x", "y", "z")):
        deriv = spatial_derivatives(u, mode=mode, which=which)[which]
        if not channels_last:
            deriv = move_dim(deriv, 1, -1)
        mat[..., i] = deriv
    for i in range(mat.shape[-1]):
        mat[..., i, i].add_(1)
    jac = mat.det().unsqueeze_(-1 if channels_last else 1)
    return jac


def normalize_flow(
    data: Tensor,
    size: Optional[Union[Tensor, torch.Size]] = None,
    side_length: float = 2,
    align_corners: bool = ALIGN_CORNERS,
    channels_last: bool = False,
) -> Tensor:
    r"""Map vectors with respect to unnormalized grid to vectors with respect to normalized grid."""
    if not isinstance(data, Tensor):
        raise TypeError("normalize_flow() 'data' must be tensor")
    if not data.is_floating_point():
        data = data.float()
    if size is None:
        if data.ndim < 4 or data.shape[1] != data.ndim - 2:
            raise ValueError(
                "normalize_flow() 'data' must have shape (N, D, ..., X) when 'size' not given"
            )
        size = torch.Size(reversed(data.shape[2:]))  # X,...
    zero = torch.tensor(0, dtype=data.dtype, device=data.device)
    size = torch.as_tensor(size, dtype=data.dtype, device=data.device)
    size_ = size.sub(1) if align_corners else size
    if not channels_last:
        data = move_dim(data, 1, -1)
    if side_length != 1:
        data = data.mul(side_length)
    data = torch.where(size > 1, data.div(size_), zero)
    if not channels_last:
        data = move_dim(data, -1, 1)
    return data


def denormalize_flow(
    data: Tensor,
    size: Optional[Union[Tensor, torch.Size]] = None,
    side_length: float = 2,
    align_corners: bool = ALIGN_CORNERS,
    channels_last: bool = False,
) -> Tensor:
    r"""Map vectors with respect to normalized grid to vectors with respect to unnormalized grid."""
    if not isinstance(data, Tensor):
        raise TypeError("denormalize_flow() 'data' must be tensors")
    if size is None:
        if data.ndim < 4 or data.shape[1] != data.ndim - 2:
            raise ValueError(
                "denormalize_flow() 'data' must have shape (N, D, ..., X) when 'size' not given"
            )
        size = torch.Size(reversed(data.shape[2:]))  # X,...
    zero = torch.tensor(0, dtype=data.dtype, device=data.device)
    size = torch.as_tensor(size, dtype=data.dtype, device=data.device)
    size_ = size.sub(1) if align_corners else size
    if not channels_last:
        data = move_dim(data, 1, -1)
    data = torch.where(size > 1, data.mul(size_), zero)
    if side_length != 1:
        data = data.div(side_length)
    if not channels_last:
        data = move_dim(data, -1, 1)
    return data


def sample_flow(flow: Tensor, coords: Tensor, align_corners: bool = ALIGN_CORNERS) -> Tensor:
    r"""Sample non-rigid flow fields at given points.

    This function samples a vector field at spatial points. The ``coords`` tensor can be of any shape,
    including ``(N, M, D)``, i.e., a batch of N point sets with cardianality M. It can also be applied to
    a tensor of grid points of shape ``(N, ..., X, D)`` regardless if the grid points are located at the
    undeformed grid positions or an already deformed grid. The given non-rigid flow field is interpolated
    at the input points ``x`` using linear interpolation. These flow vectors ``u(x)`` are returned.

    Args:
        flow: Flow fields of non-rigid transformations given as tensor of shape ``(N, D, ..., X)``
            or ``(1, D, ..., X)``. If batch size is one, but the batch size of ``coords`` is greater
            than one, this single flow fields is sampled at the different sets of points.
        coords: Normalized coordinates of points given as tensor of shape ``(N, ..., D)``
            or ``(1, ..., D)``. If batch size is one, all flow fields are sampled at the same points.
        align_corners: Whether point coordinates are with respect to ``Axes.CUBE`` (False)
            or ``Axes.CUBE_CORNERS`` (True). This option is in particular passed on to the
            ``grid_sample()`` function used to sample the flow vectors at the input points.

    Returns:
        Tensor of shape ``(N, ..., D)``.

    """
    if not isinstance(flow, Tensor):
        raise TypeError("sample_flow() 'flow' must be of type torch.Tensor")
    if flow.ndim < 4:
        raise ValueError("sample_flow() 'flow' must be at least 4-dimensional tensor")
    if not isinstance(coords, Tensor):
        raise TypeError("sample_flow() 'coords' must be of type torch.Tensor")
    if coords.ndim < 2:
        raise ValueError("sample_flow() 'coords' must be at least 2-dimensional tensor")
    G = flow.shape[0]
    N = coords.shape[0] if G == 1 else G
    D = flow.shape[1]
    if coords.shape[0] not in (1, N):
        raise ValueError(f"sample_flow() 'coords' must be batch of length 1 or {N}")
    if coords.shape[-1] != D:
        raise ValueError(f"sample_flow() 'coords' must be tensor of {D}-dimensional points")
    x = coords.expand((N,) + coords.shape[1:])
    t = flow.expand((N,) + flow.shape[1:])
    g = x.reshape((N,) + (1,) * (t.ndim - 3) + (-1, D))
    u = grid_sample(t, g, padding=PaddingMode.BORDER, align_corners=align_corners)
    u = move_dim(u, 1, -1)
    u = u.reshape(x.shape)
    return u


def warp_grid(flow: Tensor, grid: Tensor, align_corners: bool = ALIGN_CORNERS) -> Tensor:
    r"""Transform undeformed grid by a tensor of non-rigid flow fields.

    This function applies a non-rigid transformation to map a tensor of undeformed grid points to a
    tensor of deformed grid points with the same shape as the input tensor. The input points must be
    the positions of undeformed spatial grid points, because this function uses interpolation to
    resize the vector fields to the size of the input ``grid``. This assumes that input points ``x``
    are the coordinates of points located on a regularly spaced undeformed grid which is aligned with
    the borders of the grid domain on which the vector fields of the non-rigid transformations are
    sampled, i.e., ``y = x + u``.

    If in doubt whether the input points will be sampled regularly at grid points of the domain of
    the spatial transformation, use ``warp_points()`` instead.

    Args:
        flow: Flow fields of non-rigid transformations given as tensor of shape ``(N, D, ..., X)``
            or ``(1, D, ..., X)``. If batch size is one, but the batch size of ``points`` is greater
            than one, all point sets are transformed by the same non-rigid transformation.
        grid: Coordinates of points given as tensor of shape ``(N, ..., D)`` or ``(1, ..., D)``.
            If batch size is one, but multiple flow fields are given, this single point set is
            transformed by each non-rigid transformation to produce ``N`` output point sets.
        align_corners: Whether grid points and flow vectors are with respect to ``Axes.CUBE``
            (False) or ``Axes.CUBE_CORNERS`` (True). This option is in particular passed on to
            the ``grid_reshape()`` function used to resize the flow fields to the ``grid`` shape.

    Returns:
        Tensor of shape ``(N, ..., D)`` with coordinates of spatially transformed points.

    """
    if not isinstance(flow, Tensor):
        raise TypeError("warp_grid() 'flow' must be of type torch.Tensor")
    if flow.ndim < 4:
        raise ValueError("warp_grid() 'flow' must be at least 4-dimensional tensor")
    if not isinstance(grid, Tensor):
        raise TypeError("warp_grid() 'grid' must be of type torch.Tensor")
    if grid.ndim < 4:
        raise ValueError("warp_grid() 'grid' must be at least 4-dimensional tensor")
    G = flow.shape[0]
    N = grid.shape[0] if G == 1 else G
    D = flow.shape[1]
    if grid.shape[0] not in (1, N):
        raise ValueError(f"warp_grid() 'grid' must be batch of length 1 or {N}")
    if grid.shape[-1] != D:
        raise ValueError(f"warp_grid() 'grid' must be tensor of {D}-dimensional points")
    x = grid.expand((N,) + grid.shape[1:])
    t = flow.expand((N,) + flow.shape[1:])
    u = grid_reshape(t, grid.shape[1:-1], align_corners=align_corners)
    u = move_dim(u, 1, -1)
    y = x + u
    return y


def warp_points(flow: Tensor, coords: Tensor, align_corners: bool = ALIGN_CORNERS) -> Tensor:
    r"""Transform set of points by a tensor of non-rigid flow fields.

    This function applies a non-rigid transformation to map a tensor of spatial points to another tensor
    of spatial points of the same shape as the input tensor. Unlike ``warp_grid()``, it can be used
    to spatially transform any set of points which are defined with respect to the grid domain of the
    non-rigid transformation, including a tensor of shape ``(N, M, D)``, i.e., a batch of N point sets with
    cardianality M. It can also be applied to a tensor of grid points of shape ``(N, ..., X, D)`` regardless
    if the grid points are located at the undeformed grid positions or an already deformed grid. The given
    non-rigid flow field is interpolated at the input points ``x`` using linear interpolation. These flow
    vectors ``u(x)`` are then added to the input points, i.e., ``y = x + u(x)``.

    Args:
        flow: Flow fields of non-rigid transformations given as tensor of shape ``(N, D, ..., X)``
            or ``(1, D, ..., X)``. If batch size is one, but the batch size of ``points`` is greater
            than one, all point sets are transformed by the same non-rigid transformation.
        coords: Normalized coordinates of points given as tensor of shape ``(N, ..., D)``
            or ``(1, ..., D)``. If batch size is one, this single point set is deformed by each
            flow field to produce ``N`` output point sets.
        align_corners: Whether points and flow vectors are with respect to ``Axes.CUBE`` (False)
            or ``Axes.CUBE_CORNERS`` (True). This option is in particular passed on to the
            ``grid_sample()`` function used to sample the flow vectors at the input points.

    Returns:
        Tensor of shape ``(N, ..., D)`` with coordinates of spatially transformed points.

    """
    x = coords
    u = sample_flow(flow, coords, align_corners=align_corners)
    y = x + u
    return y


def warp_image(
    data: Tensor,
    grid: Tensor,
    flow: Optional[Tensor] = None,
    mode: Optional[Union[Sampling, str]] = None,
    padding: Optional[Union[PaddingMode, str, Scalar]] = None,
    align_corners: bool = ALIGN_CORNERS,
) -> Tensor:
    r"""Sample data at optionally displaced grid points.

    Args:
        data: Image batch tensor of shape ``(1, C, ..., X)`` or ``(N, C, ..., X)``.
        grid: Grid points tensor of shape  ``(..., X, D)``, ``(1, ..., X, D)``, or``(N, ..., X, D)``.
            Coordinates of points at which to sample ``data`` must be with respect to ``Axes.CUBE``.
        flow: Batch of flow fields of shape  ``(..., X, D)``, ``(1, ..., X, D)``, or``(N, ..., X, D)``.
            If specified, the flow field(s) are added to ``grid`` in order to displace the grid points.
        mode: Image interpolate mode.
        padding: Image extrapolation mode or constant by which to pad input ``data``.
        align_corners: Whether ``grid`` extrema ``(-1, 1)`` refer to the grid boundary
            edges (``align_corners=False``) or corner points (``align_corners=True``).

    Returns:
        Image batch tensor of sampled data with shape determined by ``grid``.

    """
    if data.ndim < 4:
        raise ValueError("warp_image() expected tensor 'data' of shape (N, C, ..., X)")
    grid = check_sample_grid("warp", data, grid)
    N = grid.shape[0]
    D = grid.shape[-1]
    if flow is not None:
        if flow.ndim == data.ndim - 1:
            flow = flow.unsqueeze(0)
        elif flow.ndim != data.ndim:
            raise ValueError(
                f"warp_image() expected 'flow' tensor with {data.ndim - 1} or {data.ndim} dimensions"
            )
        if flow.shape[0] != N:
            flow = flow.expand(N, *flow.shape[1:])
        if flow.shape[0] != N or flow.shape[-1] != D:
            msg = f"warp_image() expected tensor 'flow' of shape (..., X, {D})"
            msg += f" or (1, ..., X, {D})" if N == 1 else f" or (1|{N}, ..., X, {D})"
            raise ValueError(msg)
        grid = grid + flow
    assert data.device == grid.device
    return grid_sample(data, grid, mode=mode, padding=padding, align_corners=align_corners)


def zeros_flow(
    size: Optional[Union[int, Size, Grid]] = None,
    shape: Optional[Shape] = None,
    num: Optional[int] = None,
    dtype: Optional[DType] = None,
    device: Optional[Device] = None,
) -> Tensor:
    r"""Create batch of flow fields filled with zeros for given image batch size or grid."""
    size = _image_size("zeros_flow", size, shape)
    return zeros_image(size, num=num, channels=len(size), dtype=dtype, device=device)
