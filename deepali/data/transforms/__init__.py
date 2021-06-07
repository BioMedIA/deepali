r"""Data transforms.

Note that data transforms are included in the ``data`` package to avoid cyclical imports
between modules defining specialized tensor types (e.g., ``data.image``) and datasets
defined in ``data.dataset`` which also use these transforms to read and preprocess the
loaded data (c.f., ``ImageDataset``). The data transforms can also be imported from the
top-level ``transforms`` package instead of from ``data.transforms``.

"""

from typing import Callable

from .image import AvgPoolImage
from .image import CastImage
from .image import CenterCropImage
from .image import CenterPadImage
from .image import ClampImage
from .image import ImageToTensor
from .image import NarrowImage
from .image import NormalizeImage
from .image import ReadImage
from .image import ResampleImage
from .image import RescaleImage

from .item import ItemTransform
from .item import ItemwiseTransform


Transform = Callable


__all__ = (
    # Type annotation
    "Transform",
    # Generic transforms
    "ItemTransform",
    "ItemwiseTransform",
    # Image transforms
    "AvgPoolImage",
    "CastImage",
    "CenterCropImage",
    "CenterPadImage",
    "ClampImage",
    "ImageToTensor",
    "NarrowImage",
    "NormalizeImage",
    "ReadImage",
    "ResampleImage",
    "RescaleImage",
)
