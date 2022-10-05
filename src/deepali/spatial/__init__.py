r"""Spatial coordinate and image transformations.

The spatial transforms defined by this Python package can be used to implement both
traditional optimization based and machine learning based co-registration approaches.

"""

import sys
from typing import Any, Optional

from ..core.grid import Grid

# Base classes for type comparison and annotation
from .base import LinearTransform  # noqa
from .base import NonRigidTransform  # noqa
from .base import ReadOnlyParameters  # noqa
from .base import SpatialTransform

# Parametric transformation mix-in
from .parametric import ParametricTransform  # noqa

# Composite coordinate transformations
from .composite import CompositeTransform  # noqa
from .composite import MultiLevelTransform  # noqa
from .composite import SequentialTransform  # noqa

# Configurable transformation
from .configurable import TransformConfig  # noqa
from .configurable import ConfigurableTransform  # noqa
from .configurable import affine_first  # noqa
from .configurable import has_affine_component  # noqa
from .configurable import has_nonrigid_component  # noqa
from .configurable import nonrigid_components  # noqa
from .configurable import transform_components  # noqa

# Elemental linear transformations
from .linear import EulerRotation
from .linear import HomogeneousTransform
from .linear import QuaternionRotation
from .linear import AnisotropicScaling
from .linear import IsotropicScaling  # noqa
from .linear import Shearing
from .linear import Translation  # noqa

# Composite linear transformations
from .linear import RigidTransform
from .linear import RigidQuaternionTransform
from .linear import SimilarityTransform
from .linear import AffineTransform
from .linear import FullAffineTransform

# Non-rigid deformations
from .nonrigid import DenseVectorFieldTransform  # noqa
from .nonrigid import DisplacementFieldTransform
from .nonrigid import StationaryVelocityFieldTransform

# Free-form deformations
from .bspline import BSplineTransform  # noqa
from .bspline import FreeFormDeformation
from .bspline import StationaryVelocityFreeFormDeformation

# Spatial transformers based on a coordinate transformation
from .image import ImageTransform  # noqa


# Aliases
Affine = AffineTransform
AffineWithShearing = FullAffineTransform
Disp = DisplacementFieldTransform
DispField = DisplacementFieldTransform
DDF = DisplacementFieldTransform
DVF = DisplacementFieldTransform
FFD = FreeFormDeformation
FullAffine = FullAffineTransform
MatrixTransform = HomogeneousTransform
Quaternion = QuaternionRotation
Rigid = RigidTransform
RigidQuaternion = RigidQuaternionTransform
Rotation = EulerRotation
Scaling = AnisotropicScaling
ShearTransform = Shearing
Similarity = SimilarityTransform
SVF = StationaryVelocityFieldTransform
SVField = StationaryVelocityFieldTransform
SVFFD = StationaryVelocityFreeFormDeformation


LINEAR_TRANSFORMS = (
    "Affine",
    "AffineTransform",
    "AffineWithShearing",
    "AnisotropicScaling",
    "BSplineTransform",
    "EulerRotation",
    "IsotropicScaling",
    "FullAffine",
    "FullAffineTransform",
    "HomogeneousTransform",
    "MatrixTransform",
    "Quaternion",
    "QuaternionRotation",
    "Rigid",
    "RigidTransform",
    "RigidQuaternion",
    "RigidQuaternionTransform",
    "Rotation",
    "Scaling",
    "Shearing",
    "ShearTransform",
    "Similarity",
    "SimilarityTransform",
    "Translation",
)

NONRIGID_TRANSFORMS = (
    "Disp",
    "DispField",
    "DisplacementFieldTransform",
    "DDF",
    "DVF",
    "FFD",
    "FreeFormDeformation",
    "StationaryVelocityFieldTransform",
    "StationaryVelocityFreeFormDeformation",
    "SVF",
    "SVField",
    "SVFFD",
)

COMPOSITE_TRANSFORMS = (
    "MultiLevelTransform",
    "SequentialTransform",
)

__all__ = (
    (
        "CompositeTransform",
        "ConfigurableTransform",
        "DenseVectorFieldTransform",
        "ImageTransform",
        "LinearTransform",
        "NonRigidTransform",
        "ParametricTransform",
        "ReadOnlyParameters",
        "SpatialTransform",
        "TransformConfig",
        "affine_first",
        "has_affine_component",
        "has_nonrigid_component",
        "is_linear_transform",
        "is_nonrigid_transform",
        "is_spatial_transform",
        "new_spatial_transform",
        "nonrigid_components",
        "transform_components",
    )
    + COMPOSITE_TRANSFORMS
    + LINEAR_TRANSFORMS
    + NONRIGID_TRANSFORMS
)


def is_spatial_transform(arg: Any) -> bool:
    r"""Whether given object or named transformation is a transformation type.

    Args:
        arg: Name of type or object.

    Returns:
        Whether type of ``arg`` object or name of type is a transformation model.

    """
    if isinstance(arg, str):
        return arg in LINEAR_TRANSFORMS or arg in NONRIGID_TRANSFORMS
    return isinstance(arg, SpatialTransform)


def is_linear_transform(arg: Any) -> bool:
    r"""Whether given object is a linear transformation type.

    Args:
        arg: Name of type or object.

    Returns:
        Whether type of ``arg`` object or name of type is a linear transformation.

    """
    if isinstance(arg, str):
        return arg in LINEAR_TRANSFORMS
    if isinstance(arg, SpatialTransform):
        return arg.linear
    return False


def is_nonrigid_transform(arg: Any) -> bool:
    r"""Whether given object is a non-rigid transformation type.

    Args:
        arg: Name of type or object.

    Returns:
        Whether type of ``arg`` object or name of type is a non-rigid transformation.

    """
    if isinstance(arg, str):
        return arg in NONRIGID_TRANSFORMS
    if isinstance(arg, SpatialTransform):
        return arg.nonrigid
    return False


def new_spatial_transform(
    name: str, grid: Grid, groups: Optional[int] = None, **kwargs
) -> SpatialTransform:
    r"""Initialize new transformation model of named type.

    Args:
        name: Name of transformation model.
        grid: Grid of transformation domain.
        groups: Number of transformations.
        kwargs: Optional keyword arguments of transformation model.

    Returns:
        New transformation module with optimizable parameters.

    """
    cls = getattr(sys.modules[__name__], name, None)
    if cls is not None and (name in LINEAR_TRANSFORMS or name in NONRIGID_TRANSFORMS):
        return cls(grid, groups=groups, **kwargs)
    raise ValueError(f"new_spatial_transform() 'name={name}' is not a valid transformation type")
