r"""Abstract base classes of different loss terms."""

from abc import ABCMeta, abstractmethod
from collections import OrderedDict
from typing import Any, Dict, Mapping, Optional, Sequence, Union

from torch import Tensor
from torch.nn import Module, ModuleDict, ModuleList

from deepali.core.math import max_difference
from deepali.core.typing import ScalarOrTuple


RegistrationResult = Dict[str, Any]
RegistrationLosses = Union[Module, ModuleDict, ModuleList, Mapping[str, Module], Sequence[Module]]


class RegistrationLoss(Module, metaclass=ABCMeta):
    r"""Base class of registration loss functions.

    A registration loss function, also referred to as energy function, is an objective function
    to be minimized by an optimization routine. In particular, these energy functions are used inside
    the main loop which performs individual gradient steps using an instance of ``torch.optim.Optimizer``.

    Registration loss consists of one or more loss terms, which may be either one of:
    - A pairwise data term measuring the alignment of a single input pair (e.g., images, point sets, surfaces).
    - A groupwise data term measuring the alignment of two or more inputs.
    - A regularization term penalizing certain dense spatial deformations.
    - Other regularization terms based on spatial transformation parameters.

    """

    @staticmethod
    def as_module_dict(arg: Optional[RegistrationLosses], start: int = 0) -> ModuleDict:
        r"""Convert argument to ``ModuleDict``."""
        if arg is None:
            return ModuleDict()
        if isinstance(arg, ModuleDict):
            return arg
        if isinstance(arg, Module):
            arg = [arg]
        if isinstance(arg, dict):
            modules = arg
        else:
            modules = OrderedDict(
                [(f"loss_{i + start}", m) for i, m in enumerate(arg) if isinstance(m, Module)]
            )
        return ModuleDict(modules)

    @abstractmethod
    def eval(self) -> RegistrationResult:
        r"""Evaluate registration loss.

        Returns:
            Dictionary of current registration result. The entries in the dictionary depend on the
            respective registration loss function used, but must include at a minimum the total
            scalar "loss" value.

        """
        raise NotImplementedError(f"{type(self).__name__}.eval()")

    def forward(self) -> Tensor:
        r"""Evaluate registration loss."""
        result = self.eval()
        if not isinstance(result, dict):
            raise TypeError(f"{type(self).__name__}.eval() must return a dictionary")
        if "loss" not in result:
            raise ValueError(f"{type(self).__name__}.eval() result must contain key 'loss'")
        loss = result["loss"]
        if not isinstance(loss, Tensor):
            raise TypeError(f"{type(self).__name__}.eval() result 'loss' must be tensor")
        return loss


class PairwiseImageLoss(Module, metaclass=ABCMeta):
    r"""Base class of pairwise image dissimilarity criteria."""

    @abstractmethod
    def forward(self, source: Tensor, target: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        r"""Evaluate image dissimilarity loss."""
        raise NotImplementedError(f"{type(self).__name__}.forward()")


class NormalizedPairwiseImageLoss(PairwiseImageLoss):
    r"""Base class of pairwise image dissimilarity criteria with implicit input normalization."""

    def __init__(
        self,
        source: Optional[Tensor] = None,
        target: Optional[Tensor] = None,
        norm: Optional[Union[bool, Tensor]] = None,
    ):
        r"""Initialize similarity metric.

        Args:
            source: Source image from which to compute ``norm``. If ``None``, only use ``target`` if specified.
            target: Target image from which to compute ``norm``. If ``None``, only use ``source`` if specified.
            norm: Positive factor by which to divide loss. If ``None`` or ``True``, use ``source`` and/or ``target``.
                If ``False`` or both ``source`` and ``target`` are ``None``, a normalization factor of one is used.

        """
        super().__init__()
        if norm is True:
            norm = None
        if norm is None:
            if target is None:
                target = source
            elif source is None:
                source = target
            if source is not None and target is not None:
                norm = max_difference(source, target).square()
        elif norm is False:
            norm = None
        assert norm is None or isinstance(norm, (float, int, Tensor))
        self.norm = norm

    def extra_repr(self) -> str:
        s = ""
        norm = self.norm
        if isinstance(norm, Tensor) and norm.nelement() != 1:
            s += f"norm={self.norm!r}"
        elif norm is not None:
            s += f"norm={float(norm):.5f}"
        return s


class DisplacementLoss(Module, metaclass=ABCMeta):
    r"""Base class of regularization terms based on dense displacements."""

    @abstractmethod
    def forward(self, u: Tensor) -> Tensor:
        r"""Evaluate regularization loss for given transformation."""
        raise NotImplementedError(f"{type(self).__name__}.forward()")


class BSplineLoss(Module, metaclass=ABCMeta):
    r"""Base class of loss terms based on cubic B-spline deformation coefficients."""

    def __init__(self, stride: ScalarOrTuple[int] = 1, reduction: str = "mean"):
        r"""Initialize regularization term.

        Args:
            stride: Number of points between control points at which to evaluate bending energy, plus one.
                If a sequence of values is given, these must be the strides for the different spatial
                dimensions in the order ``(sx, ...)``. A stride of 1 is equivalent to evaluating bending
                energy only at the usually coarser resolution of the control point grid. It should be noted
                that the stride need not match the stride used to densely sample the spline deformation field
                at a given fixed target image resolution.
            reduction: Specifies the reduction to apply to the output: 'none' | 'mean' | 'sum'.

        """
        super().__init__()
        self.stride = stride
        self.reduction = reduction

    @abstractmethod
    def forward(self, params: Tensor, stride: ScalarOrTuple[int] = 1) -> Tensor:
        r"""Evaluate loss term for given free-form deformation parameters."""
        raise NotImplementedError(f"{type(self).__name__}.forward()")

    def extra_repr(self) -> str:
        return f"stride={self.stride!r}, reduction={self.reduction!r}"


class PointSetDistance(Module, metaclass=ABCMeta):
    r"""Base class of point set distance terms."""

    @abstractmethod
    def forward(self, x: Tensor, *ys: Tensor) -> Tensor:
        r"""Evaluate point set distance.

        Note that some point set distance measures require a 1-to-1 correspondence
        between the two input point sets, and thus ``M == N``. Other distance losses
        may compute correspondences themselves, e.g., based on closest points.

        Args:
            x: Tensor of shape ``(M, X, D)`` with points of (transformed) target point set.
            ys: Tensors of shape ``(N, Y, D)`` with points of other point sets.

        Returns:
            Point set distance.

        """
        raise NotImplementedError(f"{type(self).__name__}.forward()")


class ParamsLoss(Module, metaclass=ABCMeta):
    r"""Regularization loss based on model parameters."""

    @abstractmethod
    def forward(self, params: Tensor) -> Tensor:
        r"""Evaluate loss term for given model parameters."""
        raise NotImplementedError(f"{type(self).__name__}.forward()")
