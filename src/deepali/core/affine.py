r"""Linear homogeneous coordinate transformations."""

import re
from typing import Optional, Union

import torch
from torch import Tensor

from .linalg import homogeneous_transform
from .tensor import as_float_tensor, atleast_1d, cat_scalars
from .typing import Array, Device, DType, Scalar, Shape


__all__ = (
    "affine_rotation_matrix",
    "apply_transform",
    "euler_rotation_matrix",
    "euler_rotation_angles",
    "euler_rotation_order",
    "identity_transform",
    "rotation_matrix",
    "scaling_transform",
    "shear_matrix",
    "translation",
    "transform_points",
    "transform_vectors",
)


def apply_transform(transform: Tensor, points: Tensor, vectors: bool = False) -> Tensor:
    r"""Alias for :func:`.homogeneous_transform`."""
    return homogeneous_transform(transform, points, vectors=vectors)


def affine_rotation_matrix(matrix: Tensor) -> Tensor:
    r"""Get orthonormal rotation matrix from (homogeneous) affine transformation.

    This function assumes the following order of elementary transformations:
    1) Scaling, 2) Shearing, 3) Rotation, and 4) Translation.

    See also FullAffineTransform, AffineTransform, RigidTransform, etc.

    Args:
        matrix: Affine transformation as tensor of shape (..., 3, 3) or (..., 3, 4).

    Returns:
        Orthonormal rotation matrices with determinant 1 as tensor of shape (..., 3, 3).

    """
    # Translated from the C++ MIRTK code which is based on the book Graphics Gems:
    # https://github.com/BioMedIA/MIRTK/blob/6461e43d0ad0e0dcea2a65aeea213a78b420eae5/Modules/Numerics/src/Matrix.cc#L1446-L1548
    if not isinstance(matrix, Tensor):
        raise TypeError("affine_rotation_matrix() 'matrix' must be Tensor")
    if matrix.ndim < 2 or matrix.shape[-2] != 3 or matrix.shape[-1] not in (3, 4):
        raise ValueError("affine_rotation_matrix() 'matrix' must have shape (..., 3, 3|4)")
    matrix = matrix[..., :3].clone()
    # Compute X scale factor and normalize 1st column.
    sx: Tensor = torch.linalg.norm(matrix[..., 0], ord=2, dim=-1)
    matrix[..., 0] = matrix[..., 0].div(sx.unsqueeze(-1))
    # Compute XY shear factor and make 2nd column orthogonal to 1st.
    tansxy = matrix[..., 0].mul(matrix[..., 1]).sum(dim=-1)
    matrix[..., 1] = matrix[..., 1].sub(matrix[..., 0].mul(tansxy.unsqueeze(-1)))
    # Actually, tansxy and 2nd column are still to large by a factor of sy.
    # Now, compute Y scale and normalize 2nd column and rescale tansxy.
    sy: Tensor = torch.linalg.norm(matrix[..., 1], ord=2, dim=-1)
    matrix[..., 1] = matrix[..., 1].div(sy.unsqueeze(-1))
    tansxy = tansxy.div(sy)
    # Compute XZ and YZ shears, orthogonalize 3rd column.
    tansxz = matrix[..., 0].mul(matrix[..., 2]).sum(dim=-1)
    matrix[..., 2] = matrix[..., 2].sub(matrix[..., 0].mul(tansxz.unsqueeze(-1)))
    tansyz = matrix[..., 1].mul(matrix[..., 2]).sum(dim=-1)
    matrix[..., 2] = matrix[..., 2].sub(matrix[..., 1].mul(tansyz.unsqueeze(-1)))
    # Actually, tansxz, tansyz and 2nd column are still too large by a factor of sz.
    # Next, get Z scale, normalize 3rd column and scale tansxz and tansyz.
    sz: Tensor = torch.linalg.norm(matrix[..., 2], ord=2, dim=-1)
    matrix[..., 2] = matrix[..., 2].div(sz.unsqueeze(-1))
    tansxz = tansxz.div(sz)
    tansyz = tansyz.div(sz)
    # At this point, the columns are orthonormal. Check for a coordinate system flip.
    # If the determinant is -1, then negate the matrix (and the scaling factors).
    mask = matrix[..., 0].mul(matrix[..., 1].cross(matrix[..., 2], dim=-1)).sum(dim=-1).ge(0)
    mask = mask.unsqueeze(-1).unsqueeze(-1).expand_as(matrix)
    matrix = matrix.where(mask, -matrix)
    return matrix


def identity_transform(
    shape: Union[int, Shape],
    *args,
    homogeneous: bool = False,
    dtype: Optional[DType] = None,
    device: Optional[Device] = None,
) -> Tensor:
    r"""Create homogeneous coordinate transformation matrix of identity mapping.

    Args:
        shape: Shape of non-homogeneous point coordinates tensor ``(..., D)``, where the size of the
            last dimension corresponds to the number of spatial dimensions.
        homogeneous: Whether to return homogeneous transformation matrices.
        dtype: Data type of output matrix. If ``None``, use default dtype.
        device: Device on which to create matrix. If ``None``, use default device.

    Returns:
        If ``homogeneous=Falae``, a tensor of affine matrices of shape ``(..., D, D)`` is returned, and
        a tensor of homogeneous coordinate transformation matrices of shape ``(..., D, D + 1)``, otherwise.

    """
    shape_ = [int(n) for n in cat_scalars(shape, *args, device=device)]
    D = shape_[-1]
    J = tuple(range(D))
    matrix = torch.zeros(*shape_, D + 1 if homogeneous else D, dtype=dtype, device=device)
    matrix[..., J, J] = 1
    return matrix


def rotation_matrix(*args, **kwargs) -> Tensor:
    r"""Alias for :func:`.euler_rotation_matrix`."""
    return euler_rotation_matrix(*args, **kwargs)


def euler_rotation_matrix(
    angles: Union[Scalar, Array],
    order: Optional[str] = None,
    homogeneous: bool = False,
    dtype: Optional[DType] = None,
    device: Optional[Device] = None,
) -> Tensor:
    r"""Euler rotation matrices.

    Args:
        angles: Scalar rotation angle for a 2D rotation, or the three angles (alpha, beta, gamma)
            for an extrinsic rotation in the specified `order`. The argument can be a tensor of
            shape ``(..., D)``, with angles in the last dimension. All angles must be given in
            radians, and the first angle corresponds to the left-most rotation which is applied last.
        order: Order in which to compose elemental rotations. For example in 3D, "zxz" (or "ZXZ")
            means that the first rotation occurs about z by angle gamma, the second about x by
            angle beta, and the third rotation about z again by angle alpha. In 2D, this argument
            is ignored and a single rotation about z (plane normal) is applied. Alternatively,
            strings of the form "Rz o Rx o Rz" can be given as argument. When the first and last
            extrinsic rotation is about the same axis, the rotation is called proper, and the
            angles are referred to as proper Euler angles. When each elemental rotation in X, Y,
            and Z occurs exactly once, the angles are referred to as Tait-Bryan angles.
        homogeneous: Whether to return homogeneous transformation matrices.
        dtype: Data type of rotation matrix. If ``None``, use ``angles.dtype`` if it is
            a floating point type, and ``torch.float`` otherwise.
        device: Device on which to create rotation matrix. If ``None``, use ``angles.device``.

    Returns:
        Tensor of square rotation matrices of shape ``(..., D, D)`` if ``homogeneous=False``, or tensor
        of homogeneous coordinate transformation matrices of shape ``(..., D, D + 1)`` otherwise.
        If ``angles`` is a scalar or vector, a single rotation matrix is returned.

    See also:
        - https://mathworld.wolfram.com/EulerAngles.html
        - https://en.wikipedia.org/wiki/Euler_angles#Definition_by_extrinsic_rotations

    """
    if not isinstance(order, (str, type(None))):
        raise TypeError("euler_rotation_matrix() 'order' must be None or str")
    angles_ = atleast_1d(angles, dtype=dtype, device=device)
    angles_ = as_float_tensor(angles_)
    c = torch.cos(angles_)
    s = torch.sin(angles_)
    N = angles_.shape[-1]
    D = 2 if N == 1 else N
    order = euler_rotation_order(order, ndim=D)
    matrix = angles_.new_empty(angles_.shape[:-1] + (D, D + 1 if homogeneous else D))
    if homogeneous:
        matrix[..., D] = 0
    if D == 2:
        matrix[..., 0, 0] = c[..., 0]
        matrix[..., 0, 1] = -s[..., 0]
        matrix[..., 1, 0] = s[..., 0]
        matrix[..., 1, 1] = c[..., 0]
    elif D == 3:
        # See also the following Wikipedia page. Attention: Order of angles is reversed!
        # https://en.wikipedia.org/wiki/Euler_angles#Rotation_matrix
        if order == "XYZ":
            matrix[..., 0, 0] = c[..., 1] * c[..., 2]
            matrix[..., 0, 1] = -c[..., 1] * s[..., 2]
            matrix[..., 0, 2] = s[..., 1]
            matrix[..., 1, 0] = c[..., 0] * s[..., 2] + c[..., 2] * s[..., 0] * s[..., 1]
            matrix[..., 1, 1] = c[..., 0] * c[..., 2] - s[..., 0] * s[..., 1] * s[..., 2]
            matrix[..., 1, 2] = -c[..., 1] * s[..., 0]
            matrix[..., 2, 0] = s[..., 0] * s[..., 2] - c[..., 0] * c[..., 2] * s[..., 1]
            matrix[..., 2, 1] = c[..., 2] * s[..., 0] + c[..., 0] * s[..., 1] * s[..., 2]
            matrix[..., 2, 2] = c[..., 0] * c[..., 1]
        elif order == "ZYX":
            matrix[..., 0, 0] = c[..., 0] * c[..., 1]
            matrix[..., 0, 1] = c[..., 0] * s[..., 1] * s[..., 2] - c[..., 2] * s[..., 0]
            matrix[..., 0, 2] = s[..., 0] * s[..., 2] + c[..., 0] * c[..., 2] * s[..., 1]
            matrix[..., 1, 0] = c[..., 1] * s[..., 0]
            matrix[..., 1, 1] = c[..., 0] * c[..., 2] + s[..., 0] * s[..., 1] * s[..., 2]
            matrix[..., 1, 2] = c[..., 2] * s[..., 0] * s[..., 1] - c[..., 0] * s[..., 2]
            matrix[..., 2, 0] = -s[..., 1]
            matrix[..., 2, 1] = c[..., 1] * s[..., 2]
            matrix[..., 2, 2] = c[..., 1] * c[..., 2]
        elif order == "ZXY":
            matrix[..., 0, 0] = c[..., 0] * c[..., 2] - s[..., 0] * s[..., 1] * s[..., 2]
            matrix[..., 0, 1] = -c[..., 1] * s[..., 0]
            matrix[..., 0, 2] = c[..., 0] * s[..., 2] + c[..., 2] * s[..., 0] * s[..., 1]
            matrix[..., 1, 0] = c[..., 2] * s[..., 0] + c[..., 0] * s[..., 1] * s[..., 2]
            matrix[..., 1, 1] = c[..., 0] * c[..., 1]
            matrix[..., 1, 2] = s[..., 0] * s[..., 2] - c[..., 0] * c[..., 2] * s[..., 1]
            matrix[..., 2, 0] = -c[..., 1] * s[..., 2]
            matrix[..., 2, 1] = s[..., 1]
            matrix[..., 2, 2] = c[..., 1] * c[..., 2]
        elif order == "XZX":
            matrix[..., 0, 0] = c[..., 1]
            matrix[..., 0, 1] = -s[..., 1] * c[..., 2]
            matrix[..., 0, 2] = s[..., 1] * s[..., 2]
            matrix[..., 1, 0] = c[..., 0] * s[..., 1]
            matrix[..., 1, 1] = -s[..., 0] * s[..., 2] + c[..., 0] * c[..., 1] * c[..., 2]
            matrix[..., 1, 2] = -s[..., 0] * c[..., 2] - c[..., 0] * c[..., 1] * s[..., 2]
            matrix[..., 2, 0] = s[..., 0] * s[..., 1]
            matrix[..., 2, 1] = c[..., 0] * s[..., 2] + s[..., 0] * c[..., 1] * c[..., 2]
            matrix[..., 2, 2] = c[..., 0] * c[..., 2] - s[..., 0] * c[..., 1] * s[..., 2]
        elif order == "ZXZ":
            matrix[..., 0, 0] = c[..., 0] * c[..., 2] - s[..., 0] * c[..., 1] * s[..., 2]
            matrix[..., 0, 1] = -c[..., 0] * s[..., 2] - s[..., 0] * c[..., 1] * c[..., 2]
            matrix[..., 0, 2] = s[..., 0] * s[..., 1]
            matrix[..., 1, 0] = s[..., 0] * c[..., 2] + c[..., 0] * c[..., 1] * s[..., 2]
            matrix[..., 1, 1] = -s[..., 0] * s[..., 2] + c[..., 0] * c[..., 1] * c[..., 2]
            matrix[..., 1, 2] = -c[..., 0] * s[..., 1]
            matrix[..., 2, 0] = s[..., 1] * s[..., 2]
            matrix[..., 2, 1] = s[..., 1] * c[..., 2]
            matrix[..., 2, 2] = c[..., 1]
        else:
            matrix[..., 0, 0] = 1
            matrix[..., 1, 1] = 1
            matrix[..., 2, 2] = 1
            for i, char in enumerate(order):
                rot = matrix.new_empty(matrix.shape)
                if char == "X":
                    rot[..., 0, 0] = 1
                    rot[..., 0, 1] = 0
                    rot[..., 0, 2] = 0
                    rot[..., 1, 0] = 0
                    rot[..., 1, 1] = c[..., i]
                    rot[..., 1, 2] = -s[..., i]
                    rot[..., 2, 0] = 0
                    rot[..., 2, 1] = s[..., i]
                    rot[..., 2, 2] = c[..., i]
                elif char == "Y":
                    rot[..., 0, 0] = c[..., i]
                    rot[..., 0, 1] = 0
                    rot[..., 0, 2] = s[..., i]
                    rot[..., 1, 0] = 0
                    rot[..., 1, 1] = 1
                    rot[..., 1, 2] = 0
                    rot[..., 2, 0] = -s[..., i]
                    rot[..., 2, 1] = 0
                    rot[..., 2, 2] = c[..., i]
                elif char == "Z":
                    rot[..., 0, 0] = c[..., i]
                    rot[..., 0, 1] = -s[..., i]
                    rot[..., 0, 2] = 0
                    rot[..., 1, 0] = s[..., i]
                    rot[..., 1, 1] = c[..., i]
                    rot[..., 1, 2] = 0
                    rot[..., 2, 0] = 0
                    rot[..., 2, 1] = 0
                    rot[..., 2, 2] = 1
                matrix = rot if i == 0 else torch.bmm(matrix, rot)
    else:
        raise ValueError(
            f"Expected 'angles' to be scalar or tensor with last dimension size 3, got {N}"
        )
    return matrix


def euler_rotation_angles(matrix: Tensor, order: Optional[str] = None) -> Tensor:
    r"""Compute Euler angles from rotation matrix.

    TODO: Write test for this function and check for wich quadrant a rotation is in.
          See also https://github.com/BioMedIA/MIRTK/blob/77d3f391b49b0cee9e80da774fb074995fdf415f/Modules/Numerics/src/Matrix3x3.cc#L1217.

    Args:
        order: Order in which elemental rotations are composed. For example in 3D, "zxz" means
            that the first rotation occurs about z, the second about x, and the third rotation
            about z again. In 2D, this argument is ignored. Alternatively, strings of the form
            "Rz o Rx o Rz" can be given.

    Returns:
        Rotation angles in radians, where the first angle corresponds to the
            leftmost elemental rotation which is applied last.

    """
    if matrix.ndim < 2:
        raise ValueError("euler_rotation_angles() 'matrix' must be at least 2-dimensional")
    D = matrix.shape[-2]
    if D not in (2, 3) or matrix.shape[-1] not in (D, D + 1):
        raise ValueError(
            "euler_rotation_angles() 'matrix' must have shape (N, D, D) or (N, D, D + 1) where D=2 or D=3"
        )
    order = euler_rotation_order(order, ndim=D)
    det = matrix[..., :D, :D].detach().reshape(-1, D, D).det()
    if not det.abs().allclose(torch.tensor(1.0).to(det)):
        raise ValueError(
            "euler_rotation_angles() 'matrix' must be rotation matrix, i.e., matrix.det().abs() = 1"
        )
    if D == 2:
        angles = torch.acos(matrix[..., 0, 0])
    else:
        # https://en.wikipedia.org/wiki/Euler_angles#Rotation_matrix
        angles = matrix.new_empty(matrix.shape[:-2] + (D,))
        if order == "XZX":
            angles[..., 0] = torch.atan2(matrix[..., 0, 2], -matrix[..., 0, 1])
            angles[..., 1] = torch.acos(matrix[..., 0, 0])
            angles[..., 2] = torch.atan2(matrix[..., 2, 0], matrix[..., 1, 0])
        elif order == "ZXZ":
            angles[..., 0] = torch.atan2(matrix[..., 2, 0], matrix[..., 2, 1])
            angles[..., 1] = torch.acos(matrix[..., 2, 2])
            angles[..., 2] = torch.atan2(matrix[..., 0, 2], -matrix[..., 1, 2])
        else:
            raise NotImplementedError(f"euler_rotation_angles() order={order!r}")
    return angles


def euler_rotation_order(arg: Optional[str] = None, ndim: int = 3) -> str:
    r"""Standardize rotation order argument."""
    if arg is not None and not isinstance(arg, str):
        raise TypeError("euler_rotation_order() 'arg' must be str or None")
    if not isinstance(ndim, int):
        raise TypeError("euler_rotation_order() 'ndim' must be int")
    if ndim == 2:
        return "Z"
    if ndim != 3:
        raise NotImplementedError(f"euler_rotation_order() ndim={ndim}")
    order = "ZXZ" if arg is None else arg
    if re.match(r"^(R[xyz]|[XYZ])( o (R[xyz]|[XYZ]))*$", order):
        order = re.subn(r"R([xyz])", "\\1", order).replace(" o ", "")
    order = order.upper()
    if not re.match("^[XYZ][XYZ][XYZ]$", order):
        raise ValueError(f"euler_rotation_order() invalid argument '{arg}'")
    return order


def scaling_transform(
    scales: Union[Scalar, Array],
    homogeneous: bool = False,
    dtype: Optional[DType] = None,
    device: Optional[Device] = None,
) -> Tensor:
    r"""Scaling matrices.

    Args:
        scales: Tensor of anisotropic scaling factors of shape ``(..., D)``.
        homogeneous: Whether to return homogeneous transformation matrices.
        dtype: Data type of output matrix. If ``None``, use ``scales.dtype`` if it is
            a floating point type, and ``torch.float`` otherwise.
        device: Device on which to create matrix. If ``None``, use ``scales.device``.

    Returns:
        Tensor of square scaling matrices of shape ``(..., D, D)`` if ``homogeneous=False``, or tensor of
        homogeneous coordinate transformation matrices of shape ``(..., D, D + 1)`` otherwise.

    """
    scales_ = atleast_1d(scales, dtype=dtype, device=device)
    scales_ = as_float_tensor(scales_)
    D = scales_.shape[-1]
    J = tuple(range(D))
    matrix = scales_.new_zeros(scales_.shape[:-1] + (D, D + 1 if homogeneous else D))
    matrix[..., J, J] = scales_
    return matrix


def shear_matrix(
    angles: Union[Scalar, Array],
    homogeneous: bool = False,
    dtype: Optional[DType] = None,
    device: Optional[Device] = None,
) -> Tensor:
    r"""Shear matrices.

    Args:
        angles: Tensor of anisotropic shear angles in radians of shape ``(..., N)``,
            where ``N=(D * (D - 1)) / 2`` with ``D`` denoting the number of spatial dimensions.
        homogeneous: Whether to return homogeneous transformation matrices.
        dtype: Data type of output matrix. If ``None``, use ``angles.dtype`` if it is
            a floating point type, and ``torch.float`` otherwise.
        device: Device on which to create matrix. If ``None``, use ``angles.device``.

    Returns:
        Tensor of square scaling matrices of shape ``(..., D, D)`` if ``homogeneous=False``, or tensor of
        homogeneous coordinate transformation matrices of shape ``(..., D, D + 1)`` otherwise.

    """
    angles_ = atleast_1d(angles, dtype=dtype, device=device)
    angles_ = as_float_tensor(angles_)
    N = angles_.shape[-1]
    if N == 1:
        D = 2
    elif N == 3:
        D = 3
    elif N == 6:
        D = 4
    else:
        raise ValueError(
            "shear_matrix() 'angles' must have last dimension size 1 (2D), 3 (3D), or 6 (4D)"
        )
    J = tuple(range(D))
    K = torch.triu_indices(D, D, offset=1)
    matrix = angles_.new_zeros(angles_.shape[:-1] + (D, D + 1 if homogeneous else D))
    matrix[..., J, J] = 1
    matrix[..., K[0], K[1]] = torch.tan(angles_)
    return matrix


def translation(
    offset: Union[Scalar, Array],
    homogeneous: bool = False,
    dtype: Optional[DType] = None,
    device: Optional[Device] = None,
) -> Tensor:
    r"""Translation offsets / matrices.

    Args:
        offset: Translation vectors of shape ``(..., D)`` or ``(..., D, 1)``.
        homogeneous: Whether to return homogeneous transformation matrices.
        dtype: Data type of output matrix. If ``None``, use ``offset.dtype`` if it is
            a floating point type, and ``torch.float`` otherwise.
        device: Device on which to create matrix. If ``None``, use ``offset.device``.

    Returns:
        Homogeneous coordinate transformation matrices of shape ``(..., D, 1)`` if ``homogeneous=False``,
        or shape ``(..., D, D + 1)`` otherwise.

    """
    offset_ = atleast_1d(offset, dtype=dtype, device=device)
    offset_ = as_float_tensor(offset_)
    if homogeneous:
        if offset_.ndim > 1:
            offset_ = offset_.squeeze(-1)
        D = offset_.shape[-1]
        J = tuple(range(D))
        matrix = offset_.new_zeros(offset_.shape[:-1] + (D, D + 1))
        matrix[..., J, J] = 1
        matrix[..., D] = offset_
    elif offset_.ndim < 2 or offset_.shape[-1] != 1:
        matrix = offset_.unsqueeze(-1)
    else:
        matrix = offset_
    return matrix


def transform_points(transforms: Tensor, points: Tensor) -> Tensor:
    r"""Transform points by given homogeneous transformation."""
    return apply_transform(transforms, points, vectors=False)


def transform_vectors(transforms: Tensor, vectors: Tensor) -> Tensor:
    r"""Transform vectors by given homogeneous transformation."""
    return apply_transform(transforms, vectors, vectors=True)
