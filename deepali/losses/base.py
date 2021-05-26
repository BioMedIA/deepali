r"""Abstract base classes of different loss terms."""

from abc import ABCMeta, abstractmethod
from collections import OrderedDict
from typing import Any, Dict, Mapping, Optional, Sequence, Union

from torch import Tensor
from torch.nn import Module, ModuleDict, ModuleList


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


class DisplacementLoss(Module, metaclass=ABCMeta):
    r"""Base class of regularization terms based on dense displacements."""

    @abstractmethod
    def forward(self, u: Tensor) -> Tensor:
        r"""Evaluate regularization loss for given transformation."""
        raise NotImplementedError(f"{type(self).__name__}.forward()")


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
