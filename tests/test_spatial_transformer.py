import torch

from deepali.core import functional as U
from deepali.data import ImageBatch
from deepali.spatial import Translation
from deepali.spatial import ImageTransformer


def test_spatial_image_transformer() -> None:
    # Generate sample image
    image = ImageBatch(U.cshape_image(size=(65, 33), center=(32, 16), sigma=1, dtype=torch.float32))
    assert image.shape == (1, 1, 33, 65)

    # Apply identity transformation
    offset = torch.tensor([0, 0], dtype=torch.float32)
    translation = Translation(image.grid(), params=offset.unsqueeze(0))
    transformer = ImageTransformer(translation)
    warped = transformer.forward(image)
    assert warped.eq(image).all()

    # Shift image in world (and image) space to the left
    offset = torch.tensor([0.5, 0], dtype=torch.float32)
    translation = Translation(image.grid(), params=offset.unsqueeze(0))
    transformer = ImageTransformer(translation)
    warped = transformer.forward(image)
    expected = U.cshape_image(size=(65, 33), center=(32 - 16, 16), sigma=1, dtype=image.dtype)
    assert torch.allclose(warped, expected)

    # Shift image in world (and image) space to the right
    offset = torch.tensor([-0.5, 0], dtype=torch.float32)
    translation = Translation(image.grid(), params=offset.unsqueeze(0))
    transformer = ImageTransformer(translation)
    warped = transformer.forward(image)
    expected = U.cshape_image(size=(65, 33), center=(32 + 16, 16), sigma=1, dtype=image.dtype)
    assert torch.allclose(warped, expected)

    # Shift target sampling grid in world space and sample input with identity transform.
    # This results in a shift of the image in the image space though world positions are unchanged.
    target = image.grid().center(32 - 16, 0)
    assert not target.same_domain_as(image.grid())

    assert torch.allclose(
        image.grid().index_to_world([32, 16]),
        target.index_to_world([16, 16]),
    )

    offset = torch.tensor([0, 0], dtype=torch.float32)
    translation = Translation(image.grid(), params=offset.unsqueeze(0))
    transformer = ImageTransformer(translation, target=target, source=image.grid())
    warped = transformer.forward(image)
    expected = U.cshape_image(size=(65, 33), center=(32 - 16, 16), sigma=1, dtype=image.dtype)
    assert torch.allclose(warped, expected)
