r"""Free-form deformation (FFD) regularization terms."""

from torch import Tensor

from .base import BSplineLoss
from . import functional as L


class BSplineBending(BSplineLoss):
    r"""Bending energy of cubic B-spline free form deformation."""

    def forward(self, params: Tensor) -> Tensor:
        r"""Evaluate loss term for given free form deformation parameters."""
        return L.bspline_bending_loss(params, stride=self.stride, reduction=self.reduction)


BSplineBendingEnergy = BSplineBending
