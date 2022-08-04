r"""Free-form deformation (FFD) regularization terms."""

from torch import Tensor

from ..core import bspline as B
from ..core.types import ScalarOrTuple

from .base import BSplineLoss


class BSplineBending(BSplineLoss):
    r"""Bending energy of cubic B-spline free form deformation."""

    def forward(self, params: Tensor, stride: ScalarOrTuple[int]) -> Tensor:
        r"""Evaluate loss term for given free form deformation parameters."""
        return B.cubic_bspline_bending_energy(params, stride)


BSplineBendingEnergy = BSplineBending
