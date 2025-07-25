r"""Conversion functions between different representations of 3D rotations."""

# Copyright (C) 2017-2019, Arraiy, Inc., all rights reserved.
# Copyright (C) 2019-    , Kornia authors, all rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# The conversion functions of this module have been copied and adapted from kornia
# https://github.com/kornia/kornia/blob/8b016621105fcbf1fb1ae13be8361471887bab16/kornia/geometry/conversions.py.
#
# Temporary support for QuaternionCoeffOrder.XYZW, which will be deprecated in kornia >0.6,
# has been removed from the functions here (cf. https://github.com/kornia/kornia/issues/903).


import torch
from torch import Tensor
from torch.nn import functional as F


__all__ = (
    "angle_axis_to_rotation_matrix",
    "angle_axis_to_quaternion",
    "axis_angle_to_rotation_matrix",
    "axis_angle_to_quaternion",
    "normalize_quaternion",
    "rotation_matrix_to_angle_axis",
    "rotation_matrix_to_axis_angle",
    "rotation_matrix_to_quaternion",
    "quaternion_to_angle_axis",
    "quaternion_to_axis_angle",
    "quaternion_to_rotation_matrix",
    "quaternion_log_to_exp",
    "quaternion_exp_to_log",
)


def angle_axis_to_rotation_matrix(axis_angle: Tensor) -> Tensor:
    r"""Alias for :func:`axis_angle_to_rotation_matrix`."""
    return axis_angle_to_rotation_matrix(axis_angle)


def axis_angle_to_rotation_matrix(axis_angle: Tensor) -> Tensor:
    r"""Convert 3d vector of axis-angle rotation to 3x3 rotation matrix.

    Args:
        axis_angle: tensor of 3d vector of axis-angle rotations in radians with shape :math:`(N, 3)`.

    Returns:
        tensor of rotation matrices of shape :math:`(N, 3, 3)`.

    Example:
        >>> input = tensor([[0., 0., 0.]])
        >>> axis_angle_to_rotation_matrix(input)
        tensor([[[1., 0., 0.],
                 [0., 1., 0.],
                 [0., 0., 1.]]])

        >>> input = tensor([[1.5708, 0., 0.]])
        >>> axis_angle_to_rotation_matrix(input)
        tensor([[[ 1.0000e+00,  0.0000e+00,  0.0000e+00],
                 [ 0.0000e+00, -3.6200e-06, -1.0000e+00],
                 [ 0.0000e+00,  1.0000e+00, -3.6200e-06]]])
    """
    if not isinstance(axis_angle, Tensor):
        raise TypeError(f"Input type is not a Tensor. Got {type(axis_angle)}")

    if not axis_angle.shape[-1] == 3:
        raise ValueError(f"Input size must be a (*, 3) tensor. Got {axis_angle.shape}")

    def _compute_rotation_matrix(axis_angle: Tensor, theta2: Tensor, eps: float = 1e-6) -> Tensor:
        # We want to be careful to only evaluate the square root if the
        # norm of the axis_angle vector is greater than zero. Otherwise
        # we get a division by zero.
        k_one = 1.0
        theta = torch.sqrt(theta2)
        wxyz = axis_angle / (theta + eps)
        wx, wy, wz = torch.chunk(wxyz, 3, dim=1)
        cos_theta = torch.cos(theta)
        sin_theta = torch.sin(theta)

        r00 = cos_theta + wx * wx * (k_one - cos_theta)
        r10 = wz * sin_theta + wx * wy * (k_one - cos_theta)
        r20 = -wy * sin_theta + wx * wz * (k_one - cos_theta)
        r01 = wx * wy * (k_one - cos_theta) - wz * sin_theta
        r11 = cos_theta + wy * wy * (k_one - cos_theta)
        r21 = wx * sin_theta + wy * wz * (k_one - cos_theta)
        r02 = wy * sin_theta + wx * wz * (k_one - cos_theta)
        r12 = -wx * sin_theta + wy * wz * (k_one - cos_theta)
        r22 = cos_theta + wz * wz * (k_one - cos_theta)
        rotation_matrix = torch.cat([r00, r01, r02, r10, r11, r12, r20, r21, r22], dim=1)
        return rotation_matrix.view(-1, 3, 3)

    def _compute_rotation_matrix_taylor(axis_angle: Tensor) -> Tensor:
        rx, ry, rz = torch.chunk(axis_angle, 3, dim=1)
        k_one = torch.ones_like(rx)
        rotation_matrix = torch.cat([k_one, -rz, ry, rz, k_one, -rx, -ry, rx, k_one], dim=1)
        return rotation_matrix.view(-1, 3, 3)

    # stolen from ceres/rotation.h

    _axis_angle = torch.unsqueeze(axis_angle, dim=1)
    theta2 = torch.matmul(_axis_angle, _axis_angle.transpose(1, 2))
    theta2 = torch.squeeze(theta2, dim=1)

    # compute rotation matrices
    rotation_matrix_normal = _compute_rotation_matrix(axis_angle, theta2)
    rotation_matrix_taylor = _compute_rotation_matrix_taylor(axis_angle)

    # create mask to handle both cases
    eps = 1e-6
    mask = (theta2 > eps).view(-1, 1, 1).to(theta2.device)
    mask_pos = (mask).type_as(theta2)
    mask_neg = (~mask).type_as(theta2)

    # create output pose matrix with masked values
    rotation_matrix = mask_pos * rotation_matrix_normal + mask_neg * rotation_matrix_taylor
    return rotation_matrix  # Nx3x3


def rotation_matrix_to_angle_axis(rotation_matrix: Tensor) -> Tensor:
    r"""Alias for :func:`rotation_matrix_to_axis_angle`."""
    return rotation_matrix_to_axis_angle


def rotation_matrix_to_axis_angle(rotation_matrix: Tensor) -> Tensor:
    r"""Convert 3x3 rotation matrix to Rodrigues vector.

    Args:
        rotation_matrix (Tensor): rotation matrix.

    Returns:
        Tensor: Rodrigues vector transformation.

    Shape:
        - Input: :math:`(N, 3, 3)`
        - Output: :math:`(N, 3)`

    Example:
        >>> input = torch.rand(2, 3, 3)  # Nx3x3
        >>> output = rotation_matrix_to_angle_axis(input)  # Nx3
    """
    if not isinstance(rotation_matrix, Tensor):
        raise TypeError(f"Input type is not a Tensor. Got {type(rotation_matrix)}")

    if not rotation_matrix.shape[-2:] == (3, 3):
        raise ValueError(f"Input size must be a (*, 3, 3) tensor. Got {rotation_matrix.shape}")
    quaternion: Tensor = rotation_matrix_to_quaternion(rotation_matrix)
    return quaternion_to_angle_axis(quaternion)


def rotation_matrix_to_quaternion(rotation_matrix: Tensor, eps: float = 1.0e-8) -> Tensor:
    r"""Convert 3x3 rotation matrix to 4d quaternion vector.

    The quaternion vector has components in (w, x, y, z) or (x, y, z, w) format.

    .. note::
        The (x, y, z, w) order is going to be deprecated in favor of efficiency.

    Args:
        rotation_matrix (Tensor): the rotation matrix to convert.
        eps (float): small value to avoid zero division. Default: 1e-8.
        order (QuaternionCoeffOrder): quaternion coefficient order. Default: 'xyzw'.
          Note: 'xyzw' will be deprecated in favor of 'wxyz'.

    Return:
        Tensor: the rotation in quaternion.

    Shape:
        - Input: :math:`(*, 3, 3)`
        - Output: :math:`(*, 4)`

    Example:
        >>> input = torch.rand(4, 3, 3)  # Nx3x3
        >>> output = rotation_matrix_to_quaternion(input, eps=torch.finfo(input.dtype).eps,
        ...                                        order=QuaternionCoeffOrder.WXYZ)  # Nx4
    """
    if not isinstance(rotation_matrix, Tensor):
        raise TypeError(f"Input type is not a Tensor. Got {type(rotation_matrix)}")

    if not rotation_matrix.shape[-2:] == (3, 3):
        raise ValueError(f"Input size must be a (*, 3, 3) tensor. Got {rotation_matrix.shape}")

    def safe_zero_division(numerator: Tensor, denominator: Tensor) -> Tensor:
        eps: float = torch.finfo(numerator.dtype).tiny  # type: ignore
        return numerator / torch.clamp(denominator, min=eps)

    rotation_matrix_vec: Tensor = rotation_matrix.view(*rotation_matrix.shape[:-2], 9)

    m00, m01, m02, m10, m11, m12, m20, m21, m22 = torch.chunk(rotation_matrix_vec, chunks=9, dim=-1)

    trace: Tensor = m00 + m11 + m22

    def trace_positive_cond():
        sq = torch.sqrt(trace + 1.0) * 2.0  # sq = 4 * qw.
        qw = 0.25 * sq
        qx = safe_zero_division(m21 - m12, sq)
        qy = safe_zero_division(m02 - m20, sq)
        qz = safe_zero_division(m10 - m01, sq)
        return torch.cat((qw, qx, qy, qz), dim=-1)

    def cond_1():
        sq = torch.sqrt(1.0 + m00 - m11 - m22 + eps) * 2.0  # sq = 4 * qx.
        qw = safe_zero_division(m21 - m12, sq)
        qx = 0.25 * sq
        qy = safe_zero_division(m01 + m10, sq)
        qz = safe_zero_division(m02 + m20, sq)
        return torch.cat((qw, qx, qy, qz), dim=-1)

    def cond_2():
        sq = torch.sqrt(1.0 + m11 - m00 - m22 + eps) * 2.0  # sq = 4 * qy.
        qw = safe_zero_division(m02 - m20, sq)
        qx = safe_zero_division(m01 + m10, sq)
        qy = 0.25 * sq
        qz = safe_zero_division(m12 + m21, sq)
        return torch.cat((qw, qx, qy, qz), dim=-1)

    def cond_3():
        sq = torch.sqrt(1.0 + m22 - m00 - m11 + eps) * 2.0  # sq = 4 * qz.
        qw = safe_zero_division(m10 - m01, sq)
        qx = safe_zero_division(m02 + m20, sq)
        qy = safe_zero_division(m12 + m21, sq)
        qz = 0.25 * sq
        return torch.cat((qw, qx, qy, qz), dim=-1)

    where_2 = torch.where(m11 > m22, cond_2(), cond_3())
    where_1 = torch.where((m00 > m11) & (m00 > m22), cond_1(), where_2)

    quaternion: Tensor = torch.where(trace > 0.0, trace_positive_cond(), where_1)
    return quaternion


def normalize_quaternion(quaternion: Tensor, eps: float = 1.0e-12) -> Tensor:
    r"""Normalizes a quaternion.

    The quaternion should be in (x, y, z, w) format.

    Args:
        quaternion (Tensor): a tensor containing a quaternion to be
          normalized. The tensor can be of shape :math:`(*, 4)`.
        eps (Optional[bool]): small value to avoid division by zero.
          Default: 1e-12.

    Return:
        Tensor: the normalized quaternion of shape :math:`(*, 4)`.

    Example:
        >>> quaternion = Tensor((1., 0., 1., 0.))
        >>> normalize_quaternion(quaternion)
        tensor([0.7071, 0.0000, 0.7071, 0.0000])
    """
    if not isinstance(quaternion, Tensor):
        raise TypeError("Input type is not a Tensor. Got {}".format(type(quaternion)))

    if not quaternion.shape[-1] == 4:
        raise ValueError("Input must be a tensor of shape (*, 4). Got {}".format(quaternion.shape))
    return F.normalize(quaternion, p=2.0, dim=-1, eps=eps)


# based on:
# https://github.com/matthew-brett/transforms3d/blob/8965c48401d9e8e66b6a8c37c65f2fc200a076fa/transforms3d/quaternions.py#L101
# https://github.com/tensorflow/graphics/blob/master/tensorflow_graphics/geometry/transformation/rotation_matrix_3d.py#L247


def quaternion_to_rotation_matrix(quaternion: Tensor) -> Tensor:
    r"""Converts a quaternion to a rotation matrix.

    Args:
        quaternion (Tensor): a tensor containing quaternion coefficients 'wxyz' to be
          converted. The tensor can be of shape :math:`(*, 4)`.

    Return:
        Tensor: the rotation matrix of shape :math:`(*, 3, 3)`.

    Example:
        >>> quaternion = Tensor((0., 0., 0., 1.))
        >>> quaternion_to_rotation_matrix(quaternion)
        tensor([[-1.,  0.,  0.],
                [ 0., -1.,  0.],
                [ 0.,  0.,  1.]])
    """
    if not isinstance(quaternion, Tensor):
        raise TypeError(f"Input type is not a Tensor. Got {type(quaternion)}")

    if not quaternion.shape[-1] == 4:
        raise ValueError(f"Input must be a tensor of shape (*, 4). Got {quaternion.shape}")

    # normalize the input quaternion
    quaternion_norm: Tensor = normalize_quaternion(quaternion)

    # unpack the normalized quaternion components
    w, x, y, z = torch.chunk(quaternion_norm, chunks=4, dim=-1)

    # compute the actual conversion
    tx: Tensor = 2.0 * x
    ty: Tensor = 2.0 * y
    tz: Tensor = 2.0 * z
    twx: Tensor = tx * w
    twy: Tensor = ty * w
    twz: Tensor = tz * w
    txx: Tensor = tx * x
    txy: Tensor = ty * x
    txz: Tensor = tz * x
    tyy: Tensor = ty * y
    tyz: Tensor = tz * y
    tzz: Tensor = tz * z
    one: Tensor = torch.tensor(1.0)

    matrix: Tensor = torch.stack(
        (
            one - (tyy + tzz),
            txy - twz,
            txz + twy,
            txy + twz,
            one - (txx + tzz),
            tyz - twx,
            txz - twy,
            tyz + twx,
            one - (txx + tyy),
        ),
        dim=-1,
    ).view(-1, 3, 3)

    if len(quaternion.shape) == 1:
        matrix = torch.squeeze(matrix, dim=0)
    return matrix


def quaternion_to_angle_axis(quaternion: Tensor) -> Tensor:
    r"""Alias for :func:`quaternion_to_axis_angle`."""
    return quaternion_to_axis_angle(quaternion)


def quaternion_to_axis_angle(quaternion: Tensor) -> Tensor:
    """Convert quaternion vector to angle axis of rotation.

    Adapted from ceres C++ library: ceres-solver/include/ceres/rotation.h

    Args:
        quaternion (Tensor): tensor with quaternion coefficients 'wxyz'.

    Return:
        Tensor: tensor with angle axis of rotation.

    Shape:
        - Input: :math:`(*, 4)` where `*` means, any number of dimensions
        - Output: :math:`(*, 3)`

    Example:
        >>> quaternion = torch.rand(2, 4)  # Nx4
        >>> angle_axis = quaternion_to_angle_axis(quaternion)  # Nx3
    """
    if not torch.is_tensor(quaternion):
        raise TypeError(f"Input type is not a Tensor. Got {type(quaternion)}")

    if not quaternion.shape[-1] == 4:
        raise ValueError(f"Input must be a tensor of shape Nx4 or 4. Got {quaternion.shape}")

    # unpack input and compute conversion
    q1: Tensor = torch.tensor([])
    q2: Tensor = torch.tensor([])
    q3: Tensor = torch.tensor([])
    cos_theta: Tensor = torch.tensor([])

    cos_theta = quaternion[..., 0]
    q1 = quaternion[..., 1]
    q2 = quaternion[..., 2]
    q3 = quaternion[..., 3]

    sin_squared_theta: Tensor = q1 * q1 + q2 * q2 + q3 * q3

    sin_theta: Tensor = torch.sqrt(sin_squared_theta)
    two_theta: Tensor = 2.0 * torch.where(
        cos_theta < 0.0,
        torch.atan2(-sin_theta, -cos_theta),
        torch.atan2(sin_theta, cos_theta),
    )

    k_pos: Tensor = two_theta / sin_theta
    k_neg: Tensor = 2.0 * torch.ones_like(sin_theta)
    k: Tensor = torch.where(sin_squared_theta > 0.0, k_pos, k_neg)

    axis_angle: Tensor = torch.zeros_like(quaternion)[..., :3]
    axis_angle[..., 0] += q1 * k
    axis_angle[..., 1] += q2 * k
    axis_angle[..., 2] += q3 * k
    return axis_angle


def quaternion_log_to_exp(quaternion: Tensor, eps: float = 1.0e-8) -> Tensor:
    r"""Applies exponential map to log quaternion.

    Args:
        quaternion (Tensor): a tensor containing a quaternion to be
          converted. The tensor can be of shape :math:`(*, 3)`.

    Return:
        Tensor: the quaternion exponential map of shape :math:`(*, 4)`.

    Example:
        >>> quaternion = Tensor((0., 0., 0.))
        >>> quaternion_log_to_exp(quaternion, eps=torch.finfo(quaternion.dtype).eps)
        tensor([1., 0., 0., 0.])
    """
    if not isinstance(quaternion, Tensor):
        raise TypeError(f"Input type is not a Tensor. Got {type(quaternion)}")

    if not quaternion.shape[-1] == 3:
        raise ValueError(f"Input must be a tensor of shape (*, 3). Got {quaternion.shape}")

    # compute quaternion norm
    norm_q: Tensor = torch.norm(quaternion, p=2, dim=-1, keepdim=True).clamp(min=eps)

    # compute scalar and vector
    quaternion_vector: Tensor = quaternion * torch.sin(norm_q) / norm_q
    quaternion_scalar: Tensor = torch.cos(norm_q)

    # compose quaternion and return
    quaternion_exp: Tensor = torch.tensor([])
    quaternion_exp = torch.cat((quaternion_scalar, quaternion_vector), dim=-1)

    return quaternion_exp


def quaternion_exp_to_log(quaternion: Tensor, eps: float = 1.0e-8) -> Tensor:
    r"""Applies the log map to a quaternion.

    Args:
        quaternion (Tensor): a tensor containing quaternion coefficients 'wxyz' to be
          converted. The tensor can be of shape :math:`(*, 4)`.
        eps (float): A small number for clamping.

    Return:
        Tensor: the quaternion log map of shape :math:`(*, 3)`.

    Example:
        >>> quaternion = Tensor((1., 0., 0., 0.))
        >>> quaternion_exp_to_log(quaternion, eps=torch.finfo(quaternion.dtype).eps)
        tensor([0., 0., 0.])
    """
    if not isinstance(quaternion, Tensor):
        raise TypeError(f"Input type is not a Tensor. Got {type(quaternion)}")

    if not quaternion.shape[-1] == 4:
        raise ValueError(f"Input must be a tensor of shape (*, 4). Got {quaternion.shape}")

    # unpack quaternion vector and scalar
    quaternion_vector: Tensor = torch.tensor([])
    quaternion_scalar: Tensor = torch.tensor([])

    quaternion_scalar = quaternion[..., 0:1]
    quaternion_vector = quaternion[..., 1:4]

    # compute quaternion norm
    norm_q: Tensor = torch.norm(quaternion_vector, p=2, dim=-1, keepdim=True).clamp(min=eps)

    # apply log map
    quaternion_log: Tensor = (
        quaternion_vector * torch.acos(torch.clamp(quaternion_scalar, min=-1.0, max=1.0)) / norm_q
    )

    return quaternion_log


# based on:
# https://github.com/facebookresearch/QuaterNet/blob/master/common/quaternion.py#L138


def angle_axis_to_quaternion(angle_axis: Tensor) -> Tensor:
    r"""Alias for :func:`axis_angle_to_quaternion`."""
    return axis_angle_to_quaternion(angle_axis)


def axis_angle_to_quaternion(axis_angle: Tensor) -> Tensor:
    r"""Convert an angle axis to a quaternion.

    Adapted from ceres C++ library: ceres-solver/include/ceres/rotation.h

    Args:
        axis_angle (Tensor): tensor with angle axis.

    Return:
        Tensor: tensor with quaternion coefficients 'wxyz'.

    Shape:
        - Input: :math:`(*, 3)` where `*` means, any number of dimensions
        - Output: :math:`(*, 4)`

    Example:
        >>> axis_angle = torch.rand(2, 3)  # Nx3
        >>> quaternion = axis_angle_to_quaternion(axis_angle)  # Nx4
    """
    if not torch.is_tensor(axis_angle):
        raise TypeError(f"Input type is not a Tensor. Got {type(axis_angle)}")

    if not axis_angle.shape[-1] == 3:
        raise ValueError(f"Input must be a tensor of shape Nx3 or 3. Got {axis_angle.shape}")

    # unpack input and compute conversion
    a0: Tensor = axis_angle[..., 0:1]
    a1: Tensor = axis_angle[..., 1:2]
    a2: Tensor = axis_angle[..., 2:3]
    theta_squared: Tensor = a0 * a0 + a1 * a1 + a2 * a2

    theta: Tensor = torch.sqrt(theta_squared)
    half_theta: Tensor = theta * 0.5

    mask: Tensor = theta_squared > 0.0
    ones: Tensor = torch.ones_like(half_theta)

    k_neg: Tensor = 0.5 * ones
    k_pos: Tensor = torch.sin(half_theta) / theta
    k: Tensor = torch.where(mask, k_pos, k_neg)
    w: Tensor = torch.where(mask, torch.cos(half_theta), ones)

    quaternion: Tensor = torch.zeros(
        size=(*axis_angle.shape[:-1], 4),
        dtype=axis_angle.dtype,
        device=axis_angle.device,
    )
    quaternion[..., 0:1] = w
    quaternion[..., 1:2] = a0 * k
    quaternion[..., 2:3] = a1 * k
    quaternion[..., 3:4] = a2 * k
    return quaternion
