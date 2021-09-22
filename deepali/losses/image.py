r"""Image dissimilarity measures."""

from typing import Optional

from torch import Tensor

from ..core import functional as U
from ..core.types import ScalarOrTuple

from .base import NormalizedPairwiseImageLoss
from .base import PairwiseImageLoss
from . import functional as L


class LCC(PairwiseImageLoss):
    r"""Local normalized cross correlation."""

    def __init__(self, kernel_size: ScalarOrTuple[int] = 7, epsilon: float = 1e-15) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.epsilon = epsilon

    def forward(self, source: Tensor, target: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        r"""Evaluate image dissimilarity loss."""
        return L.lcc_loss(
            source, target, mask=mask, kernel_size=self.kernel_size, epsilon=self.epsilon
        )

    def extra_repr(self) -> str:
        return f"kernel_size={self.kernel_size}, epsilon={self.epsilon:.2e}"


LNCC = LCC


class SSD(NormalizedPairwiseImageLoss):
    r"""Sum of squared intensity differences."""

    def forward(self, source: Tensor, target: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        r"""Evaluate image dissimilarity loss."""
        return L.ssd_loss(source, target, mask=mask, norm=self.norm)


class MSE(NormalizedPairwiseImageLoss):
    r"""Average squared intensity differences."""

    def forward(self, source: Tensor, target: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        r"""Evaluate image dissimilarity loss."""
        return L.mse_loss(source, target, mask=mask, norm=self.norm)


class PatchwiseImageLoss(PairwiseImageLoss):
    r"""Pairwise similarity of 2D image patches defined within a 3D volume."""

    def __init__(self, patches: Tensor, loss_fn: PairwiseImageLoss = SSD()):
        r"""Initialize loss term.

        Args:
            patches: Patch sampling points as tensor of shape ``(N, Z, Y, X, 3)``.
            loss_fn: Pairwise image similarity loss term used to evaluate similarity of patches.

        """
        super().__init__()
        if not isinstance(patches, Tensor):
            raise TypeError("PatchwiseImageLoss() 'patches' must be Tensor")
        if not patches.ndim == 5 or patches.shape[-1] != 3:
            raise ValueError("PatchwiseImageLoss() 'patches' must have shape (N, Z, Y, X, 3)")
        self.patches = patches
        self.loss_fn = loss_fn

    def forward(self, source: Tensor, target: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        r"""Evaluate patch dissimilarity loss."""
        if target.ndim != 5:
            raise ValueError(
                "PatchwiseImageLoss.forward() 'target' must have shape (N, C, Z, Y, X)"
            )
        if source.shape != target.shape:
            raise ValueError(
                "PatchwiseImageLoss.forward() 'source' must have same shape as 'target'"
            )
        if mask is not None:
            if mask.shape != target.shape:
                raise ValueError(
                    "PatchwiseImageLoss.forward() 'mask' must have same shape as 'target'"
                )
            mask = self._reshape(U.grid_sample_mask(mask, self.patches))
        source = self._reshape(U.grid_sample(source, self.patches))
        target = self._reshape(U.grid_sample(target, self.patches))
        return self.loss_fn(source, target, mask=mask)

    @staticmethod
    def _reshape(x: Tensor) -> Tensor:
        r"""Reshape tensor to (N * Z, C, 1, Y, X) such that each patch is a separate image in the batch."""
        N, C, Z, Y, X = x.shape
        x = x.permute(0, 2, 1, 3, 4)  # N, Z, C, Y, X
        x = x.reshape(N * Z, C, 1, Y, X)
        return x


PatchLoss = PatchwiseImageLoss
