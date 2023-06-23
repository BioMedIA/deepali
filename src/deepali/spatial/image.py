from deprecation import deprecated

from .transformer import ImageTransformer
from ..version import __version__


@deprecated(
    deprecated_in="0.3",
    removed_in="1.0",
    current_version=__version__,
    details="Use deepali.spatial.ImageTransformer instead",
)
class ImageTransform(ImageTransformer):
    ...
