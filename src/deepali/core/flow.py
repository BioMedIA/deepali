r"""Functions relating to tensors representing vector fields."""

from itertools import permutations, product
from typing import Dict, Optional, Sequence, Tuple, Union

import torch
from torch import Tensor
from torch.nn import functional as F

from . import affine as A
from .enum import FlowDerivativeKeys, SpatialDerivativeKeys, PaddingMode, Sampling
from .grid import ALIGN_CORNERS, Grid
from .image import check_sample_grid, grid_reshape, grid_sample
from .image import spatial_derivatives
from .image import _image_size, zeros_image
from .itertools import is_even_permutation
from .tensor import move_dim
from .typing import Array, Device, DType, Scalar, ScalarOrTuple, Shape, Size


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


def compose_flows(u: Tensor, v: Tensor, align_corners: bool = True) -> Tensor:
    r"""Compute composite flow field ``w = v o u = u(x) + v(x + u(x))``."""
    grid = Grid(shape=u.shape[2:], align_corners=align_corners)
    x = grid.coords(channels_last=False, dtype=u.dtype, device=u.device)
    x = move_dim(x.unsqueeze(0).add_(u), 1, -1)
    v = F.grid_sample(v, x, mode="bilinear", padding_mode="border", align_corners=align_corners)
    return u.add(v)


def compose_svfs(
    u: Tensor,
    v: Tensor,
    mode: Optional[str] = None,
    sigma: Optional[float] = None,
    spacing: Optional[Union[Scalar, Array]] = None,
    stride: Optional[ScalarOrTuple[int]] = None,
    bch_terms: int = 3,
) -> Tensor:
    r"""Approximate stationary velocity field (SVF) of composite deformation.

    The output velocity field is ``w = log(exp(v) o exp(u))``, where ``exp`` is the exponential map
    of a stationary velocity field, and ``log`` its inverse. The velocity field ``w`` is given by the
    `Baker-Campbell-Hausdorff (BCH) formula <https://en.wikipedia.org/wiki/Baker%E2%80%93Campbell%E2%80%93Hausdorff_formula>`_.

    The BCH formula with 5 Lie bracket terms (cf. ``bch_terms`` parameter) is

    .. math::

        w = v + u + \frac{1}{2} [v, u]
            + \frac{1}{12} ([v, [v, u]] - [u, [v, u]])
            + \frac{1}{48} ([[v, [v, u]], u] - [v, [u, [v, u]]])

    where

    .. math::

        [[v, [v, u]], u] - [v, [u, [v, u]]] = -2 [u, [v, [v, u]]]

    References:
    - Bossa & Olmos, 2008. A new algorithm for the computation of the group logarithm of diffeomorphisms.
        https://inria.hal.science/inria-00629873
    - Vercauteren et al., 2008. Symmetric log-domain diffeomorphic registration: A Demons-based approach.
        https://doi.org/10.1007/978-3-540-85988-8_90

    Args:
        u: First applied stationary velocity field as tensor of shape ``(N, D, ..., X)``.
        v: Second applied stationary velocity field as tensor of shape ``(N, D, ..., X)``.
        bch_terms: Number of Lie bracket terms of the BCH formula to consider.
            When 0, the returned velocity field is the sum of ``u`` and ``v``.
            This approximation is accurate if the input velocity fields commute, i.e.,
            the Lie bracket [v, u] = 0. When ``bch_terms=1``, the approximation is given by
            ``w = v + u + 1/2 [v, u]`` (note ``exp(u)`` is applied before ``exp(v)``). Formula
            ``w = v + u + \frac{1}{2} [v, u] + \frac{1}{12} ([v, [v, u]] - [u, [v, u]])`` is
            used by default, i.e., ``bch_terms=3``.
        mode: Mode of :func:`flow_derivatives()` approximation.
        sigma: Standard deviation of Gaussian used for computing spatial derivatives.
        spacing: Physical size of image voxels used to compute spatial derivatives.
        stride: Number of output grid points between control points plus one for ``mode='bspline'``.

    Returns:
        Approximation of BCH formula as tensor of shape ``(N, D, ..., X)``.

    """

    def lb(a: Tensor, b: Tensor) -> Tensor:
        return lie_bracket(a, b, mode=mode, sigma=sigma, spacing=spacing, stride=stride)

    for name, flow in [("u", u), ("v", v)]:
        if flow.ndim < 4:
            raise ValueError(
                f"compose_svfs() '{name}' must be vector field of shape (N, D, ..., X)"
            )
        if flow.shape[1] != flow.ndim - 2:
            raise ValueError(f"compose_svfs() '{name}' must have shape (N, D, ..., X)")
    if u.shape != v.shape:
        raise ValueError("compose_svfs() 'u' and 'v' must have the same shape")
    if bch_terms < 0:
        raise ValueError("compose_svfs() 'bch_terms' must not be negative")
    elif bch_terms > 5:
        raise NotImplementedError("compose_svfs() 'bch_terms' of more than 6 not implemented")

    # w = v + u
    w = v.add(u)
    if bch_terms >= 1:
        # + 1/2 [v, u]
        vu = lb(v, u)
        w = w.add(vu.mul(0.5))
    if bch_terms >= 2:
        # + 1/12 [v, [v, u]]
        vvu = lb(v, vu)
        w = w.add(vvu.mul(1 / 12))
    if bch_terms >= 3:
        # - 1/12 [u, [v, u]]
        uvu = lb(u, vu)
        w = w.sub(uvu.mul(1 / 12))
    if bch_terms >= 4:
        # + 1/48 [[v, [v, u]], u] = - 1/48 [u, [v, [v, u]]]
        # - 1/48 [v, [u, [v, u]]] = - 1/48 [u, [v, [v, u]]]
        uvvu = lb(u, vvu)
        w = w.sub(uvvu.mul((1 if bch_terms == 4 else 2) / 48))

    return w


def curl(
    flow: Tensor,
    mode: Optional[str] = None,
    sigma: Optional[float] = None,
    spacing: Optional[Union[Scalar, Array]] = None,
    stride: Optional[ScalarOrTuple[int]] = None,
) -> Tensor:
    r"""Calculate curl of vector field.

    Args:
        flow: Vector field as tensor of shape ``(N, D, ..., X)``, where ``D`` must be 3.
        mode: Mode of :func:`flow_derivatives` approximation.
        sigma: Standard deviation of Gaussian used for computing spatial derivatives.
        spacing: Physical size of image voxels used to compute spatial derivatives.
        stride: Number of output grid points between control points plus one for ``mode='bspline'``.

    Returns:
        In case of a 3D input vector field, output is another 3D vector field of rotation vectors,
        where axis of rotation corresponds to the unit vector and rotation angle to the magnitude
        of the rotation vector, as tensor of shape ``(N, 3, Z, Y, X)``.

    """
    if not isinstance(flow, Tensor):
        raise TypeError("curl() 'flow' must be of type torch.Tensor")
    if not mode:
        mode = "forward_central_backward"
    kwargs = dict(mode=mode, sigma=sigma, spacing=spacing, stride=stride)
    if flow.ndim == 4:
        if flow.shape[1] != 2:
            raise ValueError("curl() 'flow' must have shape (N, 2, Y, X)")
        which = ["du/dy", "dv/dx"]
        deriv = flow_derivatives(flow, which=which, **kwargs)
        curlv = deriv["dv/dx"].sub(deriv["du/dy"])
    elif flow.ndim == 5:
        if flow.shape[1] != 3:
            raise ValueError("curl() 'flow' must have shape (N, 3, Z, Y, X)")
        which = ["du/dy", "du/dz", "dv/dx", "dv/dz", "dw/dx", "dw/dy"]
        deriv = flow_derivatives(flow, which=which, **kwargs)
        curlv = torch.cat(
            [
                deriv["dw/dy"].sub(deriv["dv/dz"]),
                deriv["du/dz"].sub(deriv["dw/dx"]),
                deriv["dv/dx"].sub(deriv["du/dy"]),
            ],
            dim=1,
        )
    else:
        raise ValueError("curl() 'flow' must be 2- or 3-dimensional vector field")
    return curlv


def divergence(
    flow: Tensor,
    mode: Optional[str] = None,
    sigma: Optional[float] = None,
    spacing: Optional[Union[Scalar, Array]] = None,
    stride: Optional[ScalarOrTuple[int]] = None,
) -> Tensor:
    r"""Calculate divergence of vector field.

    Args:
        flow: Vector field as tensor of shape ``(N, D, ..., X)``.
        mode: Mode of :func:`flow_derivatives` approximation.
        sigma: Standard deviation of Gaussian used for computing spatial derivatives.
        spacing: Physical size of image voxels used to compute spatial derivatives.
        stride: Number of output grid points between control points plus one for ``mode='bspline'``.

    Returns:
        Scalar divergence field.

    """
    if not isinstance(flow, Tensor):
        raise TypeError("divergence() 'flow' must be of type torch.Tensor")
    if flow.ndim < 4:
        raise ValueError("divergence() 'flow' must be at least 4-dimensional tensor")
    D = flow.shape[1]
    if flow.ndim != D + 2:
        raise ValueError(
            f"divergence() 'flow' must be tensor of shape (N, {flow.ndim - 2}, ..., X)"
        )
    kwargs = dict(mode=mode, sigma=sigma, spacing=spacing, stride=stride)
    which = FlowDerivativeKeys.divergence(spatial_dims=D)
    deriv = flow_derivatives(flow, which=which, **kwargs)
    div: Optional[Tensor] = None
    for value in deriv.values():
        div = value if div is None else div.add_(value)
    assert div is not None
    return div


def divergence_free_flow(
    data: Tensor,
    mode: Optional[str] = None,
    sigma: Optional[float] = None,
    spacing: Optional[Union[Scalar, Array]] = None,
    stride: Optional[ScalarOrTuple[int]] = None,
) -> Tensor:
    r"""Construct divergence-free vector field from D-1 scalar fields or one 3-dimensional vector field, respectively.

    Experimental: This function may change in the future. Constructing a divergence-free field in 3D using curl() works best.
        The construction of a divergence free field from one or two scalar fields, respectively, may need to be revised.

    The input fields must be sufficiently smooth for the output vector field to have zero divergence. To produce a
    3-dimensional vector field, a better result may be obtained using the :func:`curl()` of another 3-dimensional
    vector field instead of two scalar fields concatenated along the channel dimension. Gaussian blurring with
    a positive ``sigma`` value or ``mode='bspline'`` may also be used to create a vector field from smoothed inputs.

    References:
        Barbarosie, Representation of divergence-free vector fields, Quart. Appl. Math. 69 (2011), 309-316
            http://dx.doi.org/10.1090/S0033-569X-2011-01215-2

    Args:
        data: A scalar field as tensor of shape ``(N, 1, Y, X)`` to generate a divergence-free 2-dimensional vector field.
            When the input is a tensor of shape ``(N, 2, Z, Y, X)``, a 3-dimensional vector field is generated using
            the cross product of the gradients of the two scalar fields. Otherwise, the input tensor must be of shape
            ``(N, 3, Z, Y, X)`` and the output is the curl of the vector field.
        mode: Mode of :func:`flow_derivatives()` approximation.
        sigma: Standard deviation of Gaussian used to smooth input field.
        spacing: Physical size of image voxels used to compute finite differences.
        stride: Number of output grid points between control points plus one for ``mode='bspline'``.

    Returns:
        Divergence-free vector field as tensor of shape ``(N, D, ..., X)``.

    """
    if not isinstance(data, Tensor):
        raise TypeError("divergence_free_flow() 'data' must be of type torch.Tensor")
    if data.ndim < 4:
        raise ValueError("divergence_free_flow() 'data' must be at least 4-dimensional tensor")
    C = data.shape[1]
    D = data.ndim - 2
    if spacing is None:
        spacing = tuple(reversed([2 / (n - 1) for n in data.shape[2:]]))
    kwargs = dict(mode=mode, sigma=sigma, spacing=spacing, stride=stride)
    if D == 2 and C == 1:
        # 90 deg rotation of gradient field
        deriv = spatial_derivatives(data, order=1, **kwargs)
        flow = torch.cat([deriv["y"].neg_(), deriv["x"]], dim=1)
    elif D == 3 and C == 2:
        deriv = spatial_derivatives(data, order=1, **kwargs)
        # Following notation at https://en.wikipedia.org/wiki/Cross_product#Coordinate_notation
        a1 = deriv["x"].narrow(1, 0, 1)
        a2 = deriv["y"].narrow(1, 0, 1)
        a3 = deriv["z"].narrow(1, 0, 1)
        b1 = deriv["x"].narrow(1, 1, 1)
        b2 = deriv["y"].narrow(1, 1, 1)
        b3 = deriv["z"].narrow(1, 1, 1)
        s1 = a2.mul(b3).sub_(a3.mul(b2))
        s2 = a3.mul(b1).sub_(a1.mul(b3))
        s3 = a1.mul(b2).sub_(a2.mul(b1))
        flow = torch.cat([s1, s2, s3], dim=1)
    elif D == 3 and C == 3:
        flow = curl(data, **kwargs)
    else:
        raise ValueError(
            "divergence_free_flow() 'data' must be tensor of shape"
            " (N, 1, Y, X), (N, 2, Z, Y, X), or (N, 3, Z, Y, X)"
        )
    return flow


def expv(
    flow: Tensor,
    scale: Optional[float] = None,
    steps: Optional[int] = None,
    sampling: Union[Sampling, str] = Sampling.LINEAR,
    padding: Union[PaddingMode, str] = PaddingMode.BORDER,
    align_corners: bool = ALIGN_CORNERS,
    inverse: bool = False,
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
        inverse: Whether to negate scaled velocity field. Setting this to ``True``
            is equivalent to negating the ``scale`` (e.g., ``scale=-1``).

    Returns:
        Exponential map of input flow field. If ``steps=0``, a reference to ``flow`` is returned.

    """
    if scale is None:
        scale = 1
    if inverse:
        scale = -scale
    if steps is None:
        steps = 5
    if not isinstance(steps, int):
        raise TypeError("expv() 'steps' must be of type int")
    if steps < 0:
        raise ValueError("expv() 'steps' must be positive value")
    if steps == 0:
        if abs(scale - 1) > 1e-15:
            flow = flow.mul(scale)
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


def flow_derivatives(
    flow: Tensor,
    which: Optional[Union[str, Sequence[str]]] = None,
    order: Optional[int] = None,
    mode: Optional[str] = None,
    sigma: Optional[float] = None,
    spacing: Optional[Union[Scalar, Array]] = None,
    stride: Optional[ScalarOrTuple[int]] = None,
) -> Dict[str, Tensor]:
    r"""Calculate spatial derivatives of flow field.

    Args:
        flow: Flow field as tensor of shape ``(N, D, ..., X)``.
        which: String codes of spatial deriviatives to compute (cf. :class:`FlowDerivativeKeys`).
            When only a sequence of spatial dimension keys is given (cf. :class:`SpatialDerivateKeys`),
            the respective spatial derivative is computed for all vector field components, i.e.,
            "x" is shorthand for "du/dx", "dv/dx", and "dw/dx" in case of a 3-dimensional flow field.
        order: Order of spatial derivatives. When both ``which`` and ``order`` are specified,
            only the derivatives listed in ``which`` that are of the given order are returned.
        mode: Method to use for approximating of :func:`spatial_derivatives()`.
        sigma: Standard deviation of Gaussian kernel in grid units. If ``None`` or zero,
            no Gaussian smoothing is used for calculation of finite differences, and a
            default standard deviation of 0.4 is used when ``mode="gaussian"``.
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
            be noted that the stride need not match the stride used to densely sample the vector
            field at a given fixed target image resolution.

    Returns:
        Mapping from partial derivative keys to spatial derivative tensors of shape ``(N, 1, ..., X)``.
        When ``mode='bspline'``, the output tensor size is reduced by three along each spatial dimension,
        and multiplied by ``stride``.

    """
    if not isinstance(flow, Tensor):
        raise TypeError("flow_derivatives() 'flow' must be torch.Tensor")
    if flow.ndim < 4:
        raise ValueError("flow_derivatives() 'flow' must be at least 4-dimensional")
    D = flow.shape[1]
    if D != flow.ndim - 2:
        raise ValueError("flow_derivatives() 'flow' must have shape (N, D, ..., X)")
    which = FlowDerivativeKeys.from_arg(spatial_dims=D, which=which, order=order)
    if spacing is None:
        spacing = tuple(reversed([2 / (n - 1) for n in flow.shape[2:]]))
    grouped_by_component = [[] for _ in range(D)]
    for channel, spatial_key in FlowDerivativeKeys.split(which):
        i = channel.index()
        if i >= D:
            raise ValueError("flow_derivatives() 'which' contains invalid spatial derivative key")
        grouped_by_component[i].append(spatial_key)
    partial_derivatives = {}
    for i, spatial_keys in enumerate(grouped_by_component):
        unique_keys = sorted(SpatialDerivativeKeys.unique(spatial_keys))
        component_derivatives = spatial_derivatives(
            flow.narrow(1, i, 1),
            which=unique_keys,
            mode=mode,
            sigma=sigma,
            spacing=spacing,
            stride=stride,
        )
        for spatial_key in spatial_keys:
            key = FlowDerivativeKeys.symbol(i, spatial_key)
            unique_key = SpatialDerivativeKeys.sorted(spatial_key)
            partial_derivatives[key] = component_derivatives[unique_key]
    return {key: partial_derivatives[key] for key in which}


def jacobian_det(
    flow: torch.Tensor,
    mode: Optional[str] = None,
    sigma: Optional[float] = None,
    spacing: Optional[Union[Scalar, Array]] = None,
    stride: Optional[ScalarOrTuple[int]] = None,
    add_identity: bool = True,
) -> Tensor:
    r"""Evaluate Jacobian determinant of spatial deformation.

    Args:
        flow: Input vector field as tensor of shape ``(N, D, ..., X)``.
        mode: Mode of :func:`flow_derivatives()` approximation.
        sigma: Standard deviation of Gaussian used for computing spatial derivatives.
        spacing: Physical size of image voxels used to compute spatial derivatives.
        stride: Number of output grid points between control points plus one for ``mode='bspline'``.
        add_identity: Whether to calculate derivatives of :math:`u(x)` (False) or the spatial
            deformation given by :math:`x + u(x)` (True), where :math:`u` is the flow field,
            by adding the identity matrix to the Jacobian of :math:`u`.

    Returns:
        Scalar field of Jacobian determinant values as tensor of shape ``(N, 1, ..., X)``.

    """
    if flow.ndim < 4:
        raise ValueError("jacobian_det() 'flow' must be dense vector field of shape (N, D, ..., X)")
    if flow.shape[1] != flow.ndim - 2:
        raise ValueError("jacobian_det() 'flow' must have shape (N, D, ..., X)")
    D = flow.shape[1]
    kwargs = dict(mode=mode, sigma=sigma, spacing=spacing, stride=stride)
    which = FlowDerivativeKeys.jacobian(spatial_dims=D)
    deriv = flow_derivatives(flow, which=which, **kwargs)
    # Add 1 to diagonal elements of Jacobian matrix, because T(x) = x + u(x)
    if add_identity:
        for i in range(D):
            deriv[FlowDerivativeKeys.symbol(i, i)].add_(1)
    if D == 2:
        a = deriv["du/dx"]
        b = deriv["du/dy"]
        c = deriv["dv/dx"]
        d = deriv["dv/dy"]
        jac = a.mul(d).sub_(b.mul(c))
    elif D == 3:
        a = deriv["du/dx"]
        b = deriv["du/dy"]
        c = deriv["du/dz"]
        d = deriv["dv/dx"]
        e = deriv["dv/dy"]
        f = deriv["dv/dz"]
        g = deriv["dw/dx"]
        h = deriv["dw/dy"]
        i = deriv["dw/dz"]
        term_1 = a.mul(e.mul(i).sub_(f.mul(h)))
        term_2 = b.mul(d.mul(i).sub_(g.mul(f)))
        term_3 = c.mul(d.mul(h).sub_(e.mul(g)))
        jac = term_1.sub(term_2).add(term_3)
    else:
        jac: Optional[Tensor] = None
        for perm in permutations(range(D)):
            term: Optional[Tensor] = None
            for i, j in zip(range(D), perm):
                dij = deriv[FlowDerivativeKeys.symbol(i, j)]
                term = dij if term is None else term.mul_(dij)
            assert term is not None
            if jac is None:
                jac = term
            elif is_even_permutation(perm):
                jac = jac.add_(term)
            else:
                jac = jac.sub_(term)
        assert jac is not None
    return jac


def jacobian_dict(
    flow: torch.Tensor,
    mode: Optional[str] = None,
    sigma: Optional[float] = None,
    spacing: Optional[Union[Scalar, Array]] = None,
    stride: Optional[ScalarOrTuple[int]] = None,
    add_identity: bool = False,
) -> Dict[Tuple[int, int], Tensor]:
    r"""Evaluate Jacobian of spatial deformation.

    Args:
        flow: Input vector field as tensor of shape ``(N, D, ..., X)``.
        mode: Mode of :func:`flow_derivatives()` approximation.
        sigma: Standard deviation of Gaussian used for computing spatial derivatives.
        spacing: Physical size of image voxels used to compute spatial derivatives.
        stride: Number of output grid points between control points plus one for ``mode='bspline'``.
        add_identity: Whether to calculate derivatives of :math:`u(x)` (False) or the spatial
            deformation given by :math:`x + u(x)` (True), where :math:`u` is the flow field,
            by adding the identity matrix to the Jacobian of :math:`u`.

    Returns:
        Dictionary of spatial derivatives with keys corresponding to (row, col) indices.

    """
    if flow.ndim < 4:
        raise ValueError("jacobian_det() 'flow' must be dense vector field of shape (N, D, ..., X)")
    if flow.shape[1] != flow.ndim - 2:
        raise ValueError("jacobian_det() 'flow' must have shape (N, D, ..., X)")
    D = flow.shape[1]
    kwargs = dict(mode=mode, sigma=sigma, spacing=spacing, stride=stride)
    which = FlowDerivativeKeys.jacobian(spatial_dims=D)
    deriv = flow_derivatives(flow, which=which, **kwargs)
    # Optionally, add 1 to diagonal elements of Jacobian matrix, because T(x) = x + u(x).
    if add_identity:
        for i in range(D):
            deriv[FlowDerivativeKeys.symbol(i, i)].add_(1)
    jac = {}
    for i, j in product(range(D), repeat=2):
        jac[(i, j)] = deriv[FlowDerivativeKeys.symbol(i, j)]
    return jac


def jacobian_matrix(
    flow: torch.Tensor,
    mode: Optional[str] = None,
    sigma: Optional[float] = None,
    spacing: Optional[Union[Scalar, Array]] = None,
    stride: Optional[ScalarOrTuple[int]] = None,
    add_identity: bool = False,
) -> Tensor:
    r"""Evaluate Jacobian of spatial deformation.

    Args:
        flow: Input vector field as tensor of shape ``(N, D, ..., X)``.
        mode: Mode of :func:`flow_derivatives()` approximation.
        sigma: Standard deviation of Gaussian used for computing spatial derivatives.
        spacing: Physical size of image voxels used to compute spatial derivatives.
        stride: Number of output grid points between control points plus one for ``mode='bspline'``.
        add_identity: Whether to calculate derivatives of :math:`u(x)` (False) or the spatial
            deformation given by :math:`x + u(x)` (True), where :math:`u` is the flow field,
            by adding the identity matrix to the Jacobian of :math:`u`.

    Returns:
        Full Jacobian matrices as tensor of shape ``(N, ..., X, D, D)``.

    """
    D = flow.ndim - 2
    deriv = jacobian_dict(
        flow,
        mode=mode,
        sigma=sigma,
        spacing=spacing,
        stride=stride,
        add_identity=add_identity,
    )
    jac = torch.cat(list(deriv.values()), dim=1)
    jac = move_dim(jac, 1, -1)
    jac = jac.reshape(jac.shape[:-1] + (D, D))
    return jac.contiguous()


def lie_bracket(
    v: Tensor,
    u: Tensor,
    mode: Optional[str] = None,
    sigma: Optional[float] = None,
    spacing: Optional[Union[Scalar, Array]] = None,
    stride: Optional[ScalarOrTuple[int]] = None,
) -> Tensor:
    r"""Lie bracket of two vector fields.

    Evaluate Lie bracket given by ``[v, u] = Jac(v) * u - Jac(u) * v`` as defined in Eq (6)
    of Vercauteren et al. (2008).

        Most authors define the Lie bracket as the opposite of (6). Numerical simulations,
        and personal communication with M. Bossa, showed the relevance of this definition.
        Future research will aim at fully understanding the reason of this discrepancy.

    References:
    - Vercauteren, 2008. Symmetric Log-Domain Diffeomorphic Registration: A Demons-based Approach.
        doi:10.1007/978-3-540-85988-8_90

    Args:
        u: Left vector field as tensor of shape ``(N, D, ..., X)``.
        v: Right vector field as tensor of shape ``(N, D, ..., X)``.
        mode: Mode of :func:`flow_derivatives()` approximation.
        sigma: Standard deviation of Gaussian used for computing spatial derivatives.
        spacing: Physical size of image voxels used to compute spatial derivatives.
        stride: Number of output grid points between control points plus one for ``mode='bspline'``.

    Returns:
        Lie bracket of vector fields as tensor of shape ``(N, D, ..., X)``.

    """
    for name, flow in [("u", u), ("v", v)]:
        if flow.ndim < 4:
            raise ValueError(f"lie_bracket() '{name}' must be vector field of shape (N, D, ..., X)")
        if flow.shape[1] != flow.ndim - 2:
            raise ValueError(f"lie_bracket() '{name}' must have shape (N, D, ..., X)")
    if u.shape != v.shape:
        raise ValueError("lie_bracket() 'u' and 'v' must have the same shape")
    jac_u = jacobian_dict(
        u,
        mode=mode,
        sigma=sigma,
        spacing=spacing,
        stride=stride,
    )
    jac_v = jacobian_dict(
        v,
        mode=mode,
        sigma=sigma,
        spacing=spacing,
        stride=stride,
    )
    D = flow.ndim - 2
    w = torch.zeros_like(u)
    for i in range(D):
        w_i = w.narrow(1, i, 1)
        for j in range(D):
            w_i = w_i.add_(jac_v[(i, j)].mul(u.narrow(1, j, 1)))
        for j in range(D):
            w_i = w_i.sub_(jac_u[(i, j)].mul(v.narrow(1, j, 1)))
    return w


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


def logv(
    flow: Tensor,
    num_iters: int = 5,
    bch_terms: int = 1,
    sigma: Optional[float] = 1.0,
    spacing: Optional[Union[Scalar, Array]] = None,
    exp_steps: Optional[int] = None,
    sampling: Union[Sampling, str] = Sampling.LINEAR,
    padding: Union[PaddingMode, str] = PaddingMode.BORDER,
    align_corners: bool = ALIGN_CORNERS,
) -> Tensor:
    r"""Group logarithmic maps of flow fields computed using algorithm by Bossa & Olsom (2008).

    References:
    - Bossa & Olmos, 2008. A new algorithm for the computation of the group logarithm of diffeomorphisms.
        https://inria.hal.science/inria-00629873

    Args:
        num_iters: Number of iterations.
        bch_terms: Number of Lie bracket terms of the Baker-Campbell-Hausdorff (BCH) formula to use
            when computing the composite of current velocity field with the correction field.
        sigma: Standard deviation of Gaussian kernel used as low-pass filter when computing spatial
            derivatives required for evaluation of Lie brackets during application of BCH formula.
        spacing: Physical size of image voxels used to compute spatial derivatives.
        exp_steps: Number of exponentiation steps to evaluate current inverse displacement field.
        sampling: Flow field interpolation mode when computing inverse displacement field.
        padding: Flow field extrapolation mode when computing inverse displacement field.
        align_corners: Whether ``flow`` vectors are defined with respect to
            ``Axes.CUBE`` (False) or ``Axes.CUBE_CORNERS`` (True).

    Returns:
        Approximate stationary velocity field which when exponentiated (cf. :func:`expv`) results
        in the given input ``flow`` field.

    """
    v = flow
    for _ in range(num_iters):
        u = expv(
            v,
            steps=exp_steps,
            sampling=sampling,
            padding=padding,
            align_corners=align_corners,
            inverse=True,
        )
        u = compose_flows(flow, u)
        v = compose_svfs(u, v, bch_terms=bch_terms, sigma=sigma, spacing=spacing)
    return v


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


def sample_flow(
    flow: Tensor,
    coords: Tensor,
    padding: Optional[Union[PaddingMode, str, Scalar]] = None,
    align_corners: bool = ALIGN_CORNERS,
) -> Tensor:
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
    if padding is None:
        padding = PaddingMode.BORDER
    x = coords.expand((N,) + coords.shape[1:])
    t = flow.expand((N,) + flow.shape[1:])
    g = x if x.ndim == t.ndim else x.reshape((N,) + (1,) * (t.ndim - 3) + (-1, D))
    u = grid_sample(t, g, padding=padding, align_corners=align_corners)
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
