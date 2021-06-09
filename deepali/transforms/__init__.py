r"""Data transforms.

The transforms in this Python package generally built on the ``core`` package.
The classes defined by these modules can be used, for example, in a data input pipeline
which is attached to a data loader. The spatial transforms defined in ``transforms.spatial``,
on the other hand, can be used to implement either a traditional or machine learning based
image or point set registration approach.

Note that some data transforms are imported from "data.transforms". This is to avoid cyclical
imports between modules defining specialized tensor types (e.g., ``data.image``) and datasets
defined in ``data.dataset`` which also use these transforms to read and preprocess the loaded
data (c.f., ``data.dataset.ImageDataset``).

Following torchvision's lead, data transform classes which operate on tensors and do not require
lambda functions are derived from ``torch.nn.Module``. Use ``torch.nn.Sequential`` to compose
transforms instead of ``torchvision.transforms.Compose``. This is to support ``torch.jit.script``.

See also: https://github.com/pytorch/vision/blob/3852b41975702cb683a92c8e37f1ef74fd6a79b1/torchvision/transforms/transforms.py#L49.

"""

from ..data.transforms import Transform

from .item import ItemTransform
from .item import ItemwiseTransform

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
