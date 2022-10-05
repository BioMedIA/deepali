r"""Constants for named tensor dimensions."""

from typing import List

from .types import Name


N = "N"  # batch size
C = "C"  # number of channels
T = "T"  # number of timepoints
D = "D"  # depth of volume (z dimension)
H = "H"  # height of image/volume (y dimension)
W = "W"  # width of image/volume (x dimension)

Z = "D"  # alias for depth dimension name
Y = "H"  # alias for height dimension name
X = "W"  # alias for width dimension name


def image_tensor_names(ndim: int) -> List[Name]:
    r"""Tuple of image data tensor names for creation of named tensors."""
    if ndim == 3:
        return [C, H, W]
    if ndim == 4:
        return [C, D, H, W]
    if ndim == 5:
        return [C, T, D, H, W]
    raise ValueError(
        "Image tensors must be 3-dimensional (2D image), 4-dimensional (3D volume), or 5-dimensional (3D+t sequence)"
    )


def image_batch_tensor_names(ndim: int) -> List[Name]:
    r"""Tuple of image batch data tensor names for creation of named tensors."""
    if ndim == 4:
        return [N, C, H, W]
    if ndim == 5:
        return [N, C, D, H, W]
    if ndim == 6:
        return [N, C, T, D, H, W]
    raise ValueError(
        "Image tensors must be 4-dimensional (2D images), 5-dimensional (3D volumes), or 6-dimensional (3D+t sequences)"
    )
