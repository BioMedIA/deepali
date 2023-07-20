r"""Basic linear algebra functions, e.g., to work with homogeneous coordinate transformations."""

from enum import Enum
from functools import reduce
from operator import mul
from typing import Optional, Sequence, Tuple, Union

import torch
import torch.nn.functional as F
from torch import Tensor

from .tensor import as_tensor
from .typing import Device, DType

from ._kornia import (
    angle_axis_to_rotation_matrix,
    angle_axis_to_quaternion,
    rotation_matrix_to_angle_axis,
    rotation_matrix_to_quaternion,
    quaternion_to_angle_axis,
    quaternion_to_rotation_matrix,
    quaternion_log_to_exp,
    quaternion_exp_to_log,
    normalize_quaternion,
)


__all__ = (
    "as_homogeneous_matrix",
    "as_homogeneous_tensor",
    "hmm",
    "homogeneous_matmul",
    "homogeneous_matrix",
    "homogeneous_transform",
    "tensordot",
    "vectordot",
    "vector_rotation",
    # adapted from kornia.geometry
    "angle_axis_to_rotation_matrix",
    "angle_axis_to_quaternion",
    "rotation_matrix_to_angle_axis",
    "rotation_matrix_to_quaternion",
    "quaternion_to_angle_axis",
    "quaternion_to_rotation_matrix",
    "quaternion_log_to_exp",
    "quaternion_exp_to_log",
    "normalize_quaternion",
)


class HomogeneousTensorType(Enum):
    r"""Type of homogeneous transformation tensor."""

    AFFINE = "affine"  # Square affine transformation matrix.
    HOMOGENEOUS = "homogeneous"  # Full homogeneous transformation matrix.
    TRANSLATION = "translation"  # Translation vector.


def as_homogeneous_tensor(
    tensor: Tensor, dtype: Optional[DType] = None, device: Optional[Device] = None
) -> Tuple[Tensor, HomogeneousTensorType]:
    r"""Convert tensor to homogeneous coordinate transformation."""
    tensor_ = as_tensor(tensor, dtype=dtype, device=device)
    if tensor_.ndim == 0:
        raise ValueError("Expected at least 1-dimensional 'tensor'")
    if tensor_.ndim == 1:
        tensor_ = tensor_.unsqueeze(1)
    if tensor_.shape[-1] == 1:
        type_ = HomogeneousTensorType.TRANSLATION
    elif tensor_.shape[-1] == tensor_.shape[-2]:
        type_ = HomogeneousTensorType.AFFINE
    elif tensor_.shape[-1] == tensor_.shape[-2] + 1:
        type_ = HomogeneousTensorType.HOMOGENEOUS
    else:
        raise ValueError(f"Invalid homogeneous 'tensor' shape {tensor_.shape}")
    return tensor_, type_


def as_homogeneous_matrix(
    tensor: Tensor, dtype: Optional[DType] = None, device: Optional[Device] = None
) -> Tensor:
    r"""Convert tensor to homogeneous coordinate transformation matrix.

    Args:
        tensor: Tensor of translations of shape ``(D,)`` or ``(..., D, 1)``, tensor of square affine
            matrices of shape ``(..., D, D)``, or tensor of homogeneous transformation matrices of
            shape ``(..., D, D + 1)``.
        dtype: Data type of output matrix. If ``None``, use ``tensor.dtype`` or default.
        device: Device on which to create matrix. If ``None``, use ``tensor.device`` or default.

    Returns:
        Homogeneous coordinate transformation matrices of shape ``(..., D, D + 1)``. If ``tensor`` has already
        shape ``(..., D, D + 1)``, a reference to this tensor is returned without making a copy, unless requested
        ``dtype`` and ``device`` differ from ``tensor`` (cf. ``as_tensor()``). Use ``homogeneous_matrix()``
        if a copy of the input ``tensor`` should always be made.


    """
    tensor_, type_ = as_homogeneous_tensor(tensor, dtype=dtype, device=device)
    if type_ == HomogeneousTensorType.TRANSLATION:
        A = torch.eye(tensor_.shape[-2], dtype=tensor_.dtype, device=tensor_.device)
        tensor_ = torch.cat([A, tensor_], dim=-1)
    elif type_ == HomogeneousTensorType.AFFINE:
        t = torch.tensor(0, dtype=tensor_.dtype, device=tensor_.device).expand(
            *tensor_.shape[:-1], 1
        )
        tensor_ = torch.cat([tensor_, t], dim=-1)
    elif type_ != HomogeneousTensorType.HOMOGENEOUS:
        raise ValueError(
            "Expected 'tensor' to have shape (D,), (..., D, 1), (..., D, D) or (..., D, D + 1)"
        )
    return tensor_


def homogeneous_transform(transform: Tensor, points: Tensor, vectors: bool = False) -> Tensor:
    r"""Transform points or vectors by given homogeneous transformations.

    The data type used for matrix-vector products, as well as the data type of
    the resulting tensor, is by default set to ``points.dtype``. If ``points.dtype``
    is not a floating point data type, ``transforms.dtype`` is used instead.

    Args:
        transform: Tensor of translations of shape ``(D,)``, ``(D, 1)`` or ``(N, D, 1)``, tensor of
            affine transformation matrices of shape ``(D, D)`` or ``(N, D, D)``, or tensor of homogeneous
            matrices of shape ``(D, D + 1)`` or ``(N, D, D + 1)``, respectively. When 3-dimensional
            batch of transformation matrices is given, the size of leading dimension N must be 1
            for applying the same transformation to all points, or be equal to the leading dimension
            of ``points``, otherwise. All points within a given batch dimension are transformed by
            the matrix of matching leading index. If size of ``points`` batch dimension is one,
            the size of the leading output batch dimension is equal to the number of transforms,
            each applied to the same set of input points.
        points: Either 1-dimensional tensor of single point coordinates, or multi-dimensional tensor
            of shape ``(N, ..., D)``, where last dimension contains the spatial coordinates in the
            order ``(x, y)`` (2D) or ``(x, y, z)`` (3D), respectively.
        vectors: Whether ``points`` is tensor of vectors. If ``True``, only the affine
            component of the given ``transforms`` is applied without any translation offset.
            If ``transforms`` is a 2-dimensional tensor of translation offsets, a tensor sharing
            the data memory of the input ``points`` is returned.

    Returns:
        Tensor of transformed points/vectors with the same shape as the input ``points``, except
        for the size of the leading batch dimension if the size of the input ``points`` batch dimension
        is one, but the ``transform`` batch contains multiple transformations.

    """
    if transform.ndim == 0:
        raise TypeError("homogeneous_transform() 'transform' must be non-scalar tensor")
    if transform.ndim == 1:
        transform = transform.unsqueeze(1)
    if transform.ndim == 2:
        transform = transform.unsqueeze(0)
    N = transform.shape[0]
    D = transform.shape[1]
    if N < 1:
        raise ValueError(
            "homogeneous_transform() 'transform' size of leading dimension must not be zero"
        )
    if transform.ndim != 3 or (
        transform.shape[2] != 1 and (1 < transform.shape[2] < D or transform.shape[2] > D + 1)
    ):
        raise ValueError(
            "homogeneous_transform() 'transform' must be tensor of shape"
            + " (D,), (D, 1), (D, D), (D, D + 1) or (N, D, 1), (N, D, D), (N, D, D + 1)"
        )
    if points.ndim == 0:
        raise TypeError("'points' must be non-scalar tensor")
    if (points.ndim == 1 and len(points) != D) or (points.ndim > 1 and points.shape[-1] != D):
        raise ValueError(
            "homogeneous_transform() 'points' number of spatial dimensions does not match 'transform'"
        )
    if points.ndim == 1:
        output_shape = (N,) + points.shape if N > 1 else points.shape
        points = points.expand((N,) + points.shape)
    elif N == 1:
        output_shape = points.shape
    elif points.shape[0] == 1 or points.shape[0] == N:
        output_shape = (N,) + points.shape[1:]
        points = points.expand((N,) + points.shape[1:])
    else:
        raise ValueError(
            "homogeneous_transform() expected size of leading dimension"
            " of 'transform' and 'points' to be either 1 or equal"
        )
    points = points.reshape(N, -1, D)
    if torch.is_floating_point(points):
        transform = transform.type(points.dtype)
    else:
        points = points.type(transform.dtype)
    if transform.shape[2] == 1:
        if not vectors:
            points = points + transform[..., 0].unsqueeze(1)
    else:
        points = torch.bmm(points, transform[:, :D, :D].transpose(1, 2))
        if not vectors and transform.shape[2] == D + 1:
            points += transform[..., D].unsqueeze(1)
    return points.reshape(output_shape)


def hmm(a: Tensor, b: Tensor) -> Tensor:
    r"""Compose two homogeneous coordinate transformations.

    Args:
        a: Tensor of second homogeneous transformation.
        b: Tensor of first homogeneous transformation.

    Returns:
        Composite homogeneous transformation given by a tensor of shape ``(..., D, D + 1)``.

    See also:
        ``homogeneous_matmul()``

    """
    c = homogeneous_matmul(a, b)
    return as_homogeneous_matrix(c)


def homogeneous_matmul(*args: Tensor) -> Tensor:
    r"""Compose homogeneous coordinate transformations.

    This function performs the equivalent of a matrix-matrix product for homogeneous coordinate transformations
    given as either a translation vector (tensor of shape ``(D,)`` or ``(..., D, 1)``), a tensor of square affine
    matrices of shape ``(..., D, D)``, or a tensor of homogeneous coordinate transformation matrices of shape
    ``(..., D, D + 1)``. The size of leading dimensions must either match, or be all 1 for one of the input tensors.
    In the latter case, the same homogeneous transformation is composed with each individual trannsformation of the
    tensor with leading dimension size greater than 1.

    For example, if the shape of tensor ``a`` is either ``(D,)``, ``(D, 1)``, or ``(1, D, 1)``, and the shape of tensor
    ``b`` is ``(N, D, D)``, the translation given by ``a`` is applied after each affine transformation given by each
    matrix in grouped batch tensor ``b``, and the shape of the composite transformation tensor is ``(N, D, D + 1)``.

    Args:
        args: Tensors of homogeneous coordinate transformations, where the transformation corresponding to the first
            argument is applied last, and the transformation corresponding to the last argument is applied first.

    Returns:
        Composite homogeneous transformation given by tensor of shape ``(..., D, 1)``, ``(..., D, D)``, or  ``(..., D, D + 1)``,
        respectively, where the shape of leading dimensions is determined by input tensors.

    """
    if not args:
        raise ValueError("homogeneous_matmul() at least one argument is required")
    # Convert first input to homogeneous transformation tensor
    a = args[0]
    dtype = a.dtype
    device = a.device
    if not dtype.is_floating_point:  # type: ignore
        for b in args[1:]:
            if b.is_floating_point():
                dtype = b.dtype
                break
        if not dtype.is_floating_point:  # type: ignore
            dtype = torch.float
    a, a_type = as_homogeneous_tensor(a, dtype=dtype)
    # Successively compose transformation matrices
    D = a.shape[-2]
    for b in args[1:]:
        # Convert input to homogeneous transformation tensor
        b, b_type = as_homogeneous_tensor(b, dtype=dtype)
        if b.device != device:
            raise RuntimeError("homogeneous_matmul() tensors must be on the same 'device'")
        if b.shape[-2] != D:
            raise ValueError(
                "homogeneous_matmul() tensors have mismatching number of spatial dimensions"
                + f" ({a.shape[-2]} != {b.shape[-2]})"
            )
        # Unify shape of leading dimensions
        leading_shape = None
        a_numel = a.shape[:-2].numel()
        b_numel = b.shape[:-2].numel()
        if a_numel > 1:
            if b_numel > 1 and a.shape[:-2] != b.shape[:-2]:
                raise ValueError(
                    "Expected homogeneous tensors to have matching leading dimensions:"
                    + f" {a.shape[:-2]} != {b.shape[:-2]}"
                )
            if b.ndim > a.ndim:
                raise ValueError("Homogeneous tensors have different number of leading dimensions")
            leading_shape = a.shape[:-2]
            b = b.expand(leading_shape + b.shape[-2:])
        elif b_numel > 1:
            if a.ndim > b.ndim:
                raise ValueError("Homogeneous tensors have different number of leading dimensions")
            leading_shape = b.shape[:-2]
            a = a.expand(leading_shape + a.shape[-2:])
        elif a.ndim > b.ndim:
            leading_shape = a.shape[:-2]
            b = b.expand(a.shape[:-2] + b.shape[-2:])
        else:
            leading_shape = b.shape[:-2]
            a = a.expand(b.shape[:-2] + a.shape[-2:])
        assert leading_shape is not None
        # Compose homogeneous transformations
        a = a.reshape(-1, *a.shape[-2:])
        b = b.reshape(-1, *b.shape[-2:])
        c, c_type = None, None
        if a_type == HomogeneousTensorType.TRANSLATION:
            if b_type == HomogeneousTensorType.TRANSLATION:
                c = a + b
                c_type = HomogeneousTensorType.TRANSLATION
            elif b_type == HomogeneousTensorType.AFFINE:
                c = torch.cat([b, a], dim=-1)
                c_type = HomogeneousTensorType.HOMOGENEOUS
            elif b_type == HomogeneousTensorType.HOMOGENEOUS:
                c = b.clone()
                c[..., D] += a[..., :, 0]
                c_type = HomogeneousTensorType.HOMOGENEOUS
        elif a_type == HomogeneousTensorType.AFFINE:
            if b_type == HomogeneousTensorType.TRANSLATION:
                t = torch.bmm(a, b)
                c = torch.cat([a, t], dim=-1)
                c_type = HomogeneousTensorType.HOMOGENEOUS
            elif b_type == HomogeneousTensorType.AFFINE:
                c = torch.bmm(a, b)
                c_type = HomogeneousTensorType.AFFINE
            elif b_type == HomogeneousTensorType.HOMOGENEOUS:
                A = torch.bmm(a, b[..., :D])
                t = torch.bmm(a[..., :D], b[..., D:])
                c = torch.cat([A, t], dim=-1)
                c_type = HomogeneousTensorType.HOMOGENEOUS
        elif a_type == HomogeneousTensorType.HOMOGENEOUS:
            if b_type == HomogeneousTensorType.TRANSLATION:
                t = torch.bmm(a[..., :D], b)
                c = a.clone()
                c[..., D] += t[..., :, 0]
            elif b_type == HomogeneousTensorType.AFFINE:
                A = torch.bmm(a[..., :D], b)
                t = a[..., D:]
                c = torch.cat([A, t], dim=-1)
            elif b_type == HomogeneousTensorType.HOMOGENEOUS:
                A = torch.bmm(a[..., :D], b[..., :D])
                t = a[..., D:] + torch.bmm(a[..., :D], b[..., D:])
                c = torch.cat([A, t], dim=-1)
            c_type = HomogeneousTensorType.HOMOGENEOUS
        assert c is not None, "as_homogeneous_tensor() returned invalid 'type' enumeration value"
        assert c_type is not None
        c = c.reshape(leading_shape + c.shape[-2:])
        assert c.device == device
        a, a_type = c, c_type
    return a


def homogeneous_matrix(
    tensor: Tensor,
    offset: Optional[Tensor] = None,
    dtype: Optional[DType] = None,
    device: Optional[Device] = None,
) -> Tensor:
    r"""Convert square matrix or vector to homogeneous coordinate transformation matrix.

    Args:
        tensor: Tensor of translations of shape ``(D,)`` or ``(..., D, 1)``, tensor of square affine
            matrices of shape ``(..., D, D)``, or tensor of homogeneous transformation matrices of
            shape ``(..., D, D + 1)``.
        offset: Translation offset to add to homogeneous transformations of shape ``(..., D)``.
            If a scalar is given, this offset is used as translation along each spatial dimension.
        dtype: Data type of output matrix. If ``None``, use ``offset.dtype``.
        device: Device on which to create matrix. If ``None``, use ``offset.device``.

    Returns:
        Homogeneous coordinate transformation matrices of shape (..., D, D + 1). Always makes a copy
        of ``tensor`` even if it has already the shape of homogeneous coordinate transformation matrices.

    """
    matrix = as_homogeneous_matrix(tensor, dtype=dtype, device=device)
    if matrix is tensor:
        matrix = tensor.clone()
    if offset is not None:
        D = matrix.shape[-2]
        if offset.ndim == 0:
            offset = offset.repeat(D)
        if offset.shape[-1] != D:
            raise ValueError(
                f"Expected homogeneous_matrix() 'offset' to be scalar or have last dimension of size {D}"
            )
        matrix[..., D] += offset
    return matrix


def tensordot(
    a: Tensor,
    b: Tensor,
    dims: Union[int, Sequence[int], Tuple[Sequence[int], Sequence[int]]] = 2,
) -> Tensor:
    r"""Implements ``numpy.tensordot()`` for ``Tensor``.

    Based on https://gist.github.com/deanmark/9aec75b7dc9fa71c93c4bc85c5438777.

    """
    if isinstance(dims, int):
        axes_a = list(range(-dims, 0))
        axes_b = list(range(0, dims))
    else:
        axes_a, axes_b = dims

    if isinstance(axes_a, int):
        axes_a = [axes_a]
        na = 1
    else:
        na = len(axes_a)
        axes_a = list(axes_a)

    if isinstance(axes_b, int):
        axes_b = [axes_b]
        nb = 1
    else:
        nb = len(axes_b)
        axes_b = list(axes_b)

    a = as_tensor(a)
    b = as_tensor(b)
    as_ = a.shape
    nda = a.ndim
    bs = b.shape
    ndb = b.ndim
    equal = True
    if na != nb:
        equal = False
    else:
        for k in range(na):
            if as_[axes_a[k]] != bs[axes_b[k]]:
                equal = False
                break
            if axes_a[k] < 0:
                axes_a[k] += nda
            if axes_b[k] < 0:
                axes_b[k] += ndb
    if not equal:
        raise ValueError("shape-mismatch for sum")

    # Move the axes to sum over to the end of "a" and to the front of "b"
    notin = [k for k in range(nda) if k not in axes_a]
    newaxes_a = notin + axes_a
    N2 = 1
    for axis in axes_a:
        N2 *= as_[axis]
    newshape_a = (int(reduce(mul, [as_[ax] for ax in notin])), N2)
    olda = [as_[axis] for axis in notin]

    notin = [k for k in range(ndb) if k not in axes_b]
    newaxes_b = axes_b + notin
    N2 = 1
    for axis in axes_b:
        N2 *= bs[axis]
    newshape_b = (N2, int(reduce(mul, [bs[ax] for ax in notin])))
    oldb = [bs[axis] for axis in notin]

    at = a.permute(newaxes_a).reshape(newshape_a)
    bt = b.permute(newaxes_b).reshape(newshape_b)

    res = at.matmul(bt)
    return res.reshape(olda + oldb)


def vectordot(a: Tensor, b: Tensor, w: Optional[Tensor] = None, dim: int = -1) -> Tensor:
    r"""Inner product of vectors over specified input tensor dimension."""
    c = a.mul(b)
    if w is not None:
        c.mul(w)
    return c.sum(dim)


def vector_rotation(a: Tensor, b: Tensor) -> Tensor:
    r"""Calculate rotation matrix which aligns two 3D vectors."""
    if not isinstance(a, Tensor) or not isinstance(b, Tensor):
        raise TypeError("vector_rotation() 'a' and 'b' must be of type Tensor")
    if a.shape != b.shape:
        raise ValueError("vector_rotation() 'a' and 'b' must have identical shape")
    a = F.normalize(a, p=2, dim=-1)
    b = F.normalize(b, p=2, dim=-1)
    axis = a.cross(b, dim=-1)
    norm: Tensor = axis.norm(p=2, dim=-1, keepdim=True)
    angle_axis = axis.div(norm).mul(norm.asin())
    rotation_matrix = angle_axis_to_rotation_matrix(angle_axis)
    return rotation_matrix
