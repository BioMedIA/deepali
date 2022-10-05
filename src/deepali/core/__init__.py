r"""Common types and functions that operate on tensors representing different types of data.

Besides defining common types and auxiliary functions extending the standard library,
this core library in particular defines functions which operate on objects of type ``torch.Tensor``.
This functional API defines a set of reusable state-less functions similar to ``torch.nn.functional``.
Object-oriented APIs use this functional API to realize their functionality. In particular, the
``forward()`` method of ``torch.nn.Module`` subclasses, such as for example data transformations
(cf. ``transforms``) and neural network components (cf. ``modules`` and ``networks``) are implemented
using these functional building blocks.

The following import statement can be used to access the functional API:

.. code::

    import deepali.core.functional as U

"""

from .config import DataclassConfig
from .config import join_kwargs_in_sequence

from .path import abspath
from .path import abspath_template
from .path import delete
from .path import make_parent_dir
from .path import make_temp_file
from .path import temp_dir
from .path import temp_file
from .path import unlink_or_mkdir

from .enum import PaddingMode
from .enum import Sampling

from .grid import ALIGN_CORNERS
from .grid import Axes
from .cube import Cube
from .grid import Grid
from .grid import grid_points_transform
from .grid import grid_vectors_transform
from .grid import grid_transform_points
from .grid import grid_transform_vectors

from .random import multinomial

from .types import Array
from .types import Batch
from .types import Dataclass
from .types import Device
from .types import DType
from .types import Name
from .types import PathStr
from .types import Sample
from .types import Scalar
from .types import ScalarOrTuple
from .types import ScalarOrTuple1d
from .types import ScalarOrTuple2d
from .types import ScalarOrTuple3d
from .types import Size
from .types import Shape
from .types import TensorCollection

from .types import RE_OUTPUT_KEY_INDEX
from .types import get_tensor
from .types import tensor_collection_entry

from .types import is_bool_dtype
from .types import is_float_dtype
from .types import is_int_dtype
from .types import is_uint_dtype
from .types import is_namedtuple
from .types import is_path_str


__all__ = (
    "ALIGN_CORNERS",
    "RE_OUTPUT_KEY_INDEX",
    "Array",
    "Axes",
    "Batch",
    "Cube",
    "Dataclass",
    "DataclassConfig",
    "Device",
    "DType",
    "Grid",
    "Name",
    "PaddingMode",
    "PathStr",
    "Sample",
    "Sampling",
    "Scalar",
    "ScalarOrTuple",
    "ScalarOrTuple1d",
    "ScalarOrTuple2d",
    "ScalarOrTuple3d",
    "Size",
    "Shape",
    "TensorCollection",
    "abspath",
    "abspath_template",
    "delete",
    "get_tensor",
    "grid_points_transform",
    "grid_vectors_transform",
    "grid_transform_points",
    "grid_transform_vectors",
    "join_kwargs_in_sequence",
    "make_parent_dir",
    "make_temp_file",
    "multinomial",
    "is_bool_dtype",
    "is_float_dtype",
    "is_int_dtype",
    "is_uint_dtype",
    "is_namedtuple",
    "is_path_str",
    "temp_dir",
    "temp_file",
    "tensor_collection_entry",
    "unlink_or_mkdir",
)
