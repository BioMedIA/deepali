r"""Flow field regularization terms."""

from __future__ import annotations

from typing import Optional, Union

import torch
from torch import Tensor

from ..core.types import Shape

from .base import DisplacementLoss
from . import functional as L


class _SpatialDerivativesLoss(DisplacementLoss):
    r"""Base class of regularization terms based on spatial derivatives of dense displacements."""

    def __init__(
        self,
        mode: str = "central",
        sigma: Optional[float] = None,
        reduction: str = "mean",
    ):
        r"""Initialize regularization term.

        Args:
            mode: Method used to approximate spatial derivatives. See ``spatial_derivatives()``.
            sigma: Standard deviation of Gaussian in grid units. See ``spatial_derivatives()``.
            reduction: Operation to use for reducing spatially distributed loss values.

        """
        super().__init__()
        self.mode = mode
        self.sigma = float(0 if sigma is None else sigma)
        self.reduction = reduction

    def _spacing(self, u_shape: Shape) -> Optional[Tensor]:
        ndim = len(u_shape)
        if ndim < 3:
            raise ValueError(f"{type(self).__name__}.forward() 'u' must be at least 3-dimensional")
        if ndim == 3:
            return None
        size = torch.tensor(u_shape[-1:1:-1], dtype=torch.float, device=torch.device("cpu"))
        return 2 / (size - 1)

    def extra_repr(self) -> str:
        return f"mode={self.mode!r}, sigma={self.sigma!r}, reduction={self.reduction!r}"


class GradLoss(_SpatialDerivativesLoss):
    r"""Displacement field gradient loss."""

    def __init__(
        self,
        p: Union[int, float] = 2,
        q: Optional[Union[int, float]] = 1,
        mode: str = "central",
        sigma: Optional[float] = None,
        reduction: str = "mean",
    ):
        r"""Initialize regularization term.

        Args:
            mode: Method used to approximate spatial derivatives. See ``spatial_derivatives()``.
            sigma: Standard deviation of Gaussian in grid units. See ``spatial_derivatives()``.
            reduction: Operation to use for reducing spatially distributed loss values.

        """
        super().__init__(mode=mode, sigma=sigma, reduction=reduction)
        self.p = p
        self.q = q

    def forward(self, u: Tensor) -> Tensor:
        r"""Evaluate regularization loss for given transformation."""
        spacing = self._spacing(u.shape)
        return L.grad_loss(
            u,
            p=self.p,
            q=self.q,
            spacing=spacing,
            mode=self.mode,
            sigma=self.sigma,
            reduction=self.reduction,
        )

    def extra_repr(self) -> str:
        return f"p={self.p}, q={self.q}, " + super().extra_repr()


class Bending(_SpatialDerivativesLoss):
    r"""Bending energy of displacement field."""

    def forward(self, u: Tensor) -> Tensor:
        r"""Evaluate regularization loss for given transformation."""
        spacing = self._spacing(u.shape)
        return L.bending_loss(
            u,
            spacing=spacing,
            mode=self.mode,
            sigma=self.sigma,
            reduction=self.reduction,
        )


BendingEnergy = Bending
BE = Bending


class Curvature(_SpatialDerivativesLoss):
    r"""Curvature of displacement field."""

    def forward(self, u: Tensor) -> Tensor:
        r"""Evaluate regularization loss for given transformation."""
        spacing = self._spacing(u.shape)
        return L.curvature_loss(
            u,
            spacing=spacing,
            mode=self.mode,
            sigma=self.sigma,
            reduction=self.reduction,
        )


class Diffusion(_SpatialDerivativesLoss):
    r"""Diffusion of displacement field."""

    def forward(self, u: Tensor) -> Tensor:
        r"""Evaluate regularization loss for given transformation."""
        spacing = self._spacing(u.shape)
        return L.diffusion_loss(
            u,
            spacing=spacing,
            mode=self.mode,
            sigma=self.sigma,
            reduction=self.reduction,
        )


class Elasticity(_SpatialDerivativesLoss):
    r"""Linear elasticity of displacement field."""

    def forward(self, u: Tensor) -> Tensor:
        r"""Evaluate regularization loss for given transformation."""
        spacing = self._spacing(u.shape)
        return L.elasticity_loss(
            u,
            spacing=spacing,
            mode=self.mode,
            sigma=self.sigma,
            reduction=self.reduction,
        )


class TotalVariation(_SpatialDerivativesLoss):
    r"""Total variation of displacement field."""

    def forward(self, u: Tensor) -> Tensor:
        r"""Evaluate regularization loss for given transformation."""
        spacing = self._spacing(u.shape)
        return L.total_variation_loss(
            u,
            spacing=spacing,
            mode=self.mode,
            sigma=self.sigma,
            reduction=self.reduction,
        )


TV = TotalVariation
