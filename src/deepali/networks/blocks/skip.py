r"""Sequential network layers with skip connections."""

from typing import Any, Callable, Mapping, Union, overload

from torch import Tensor
from torch.nn import Identity, Module, ModuleDict, Sequential

from deepali.modules.mixins import ReprWithCrossReferences

from ..layers.join import JoinFunc, join_func


# fmt: off
__all__ = (
    "DenseBlock",
    "Shortcut",
    "SkipConnection",
    "SkipFunc",
)
# fmt: on


SkipFunc = Callable[[Tensor], Tensor]


class DenseBlock(ReprWithCrossReferences, Module):
    r"""Subnetwork with dense skip connections."""

    @overload
    def __init__(self, *args: Module, join: Union[str, JoinFunc], dim: int) -> None:
        ...

    @overload
    def __init__(self, arg: Mapping[str, Module], join: Union[str, JoinFunc], dim: int) -> None:
        ...

    def __init__(self, *args: Any, join: Union[str, JoinFunc] = "cat", dim: int = 1) -> None:
        super().__init__()
        if len(args) == 1 and isinstance(args[0], Mapping):
            layers = args[0]
        else:
            layers = {str(i): m for i, m in enumerate(args)}
        self.layers = ModuleDict(layers)
        self.join = join_func(join, dim=dim)
        self.is_associative = join in ("add", "cat", "concat")

    def forward(self, x: Tensor) -> Tensor:
        y, ys = x, [x]
        join = self.join
        is_associative = self.is_associative
        for module in self.layers.values():
            x = join(ys)
            y = module(x)
            ys = [x, y] if is_associative else [*ys, y]
        return y


class SkipConnection(ReprWithCrossReferences, Module):
    r"""Combine input with subnetwork output along a single skip connection."""

    @overload
    def __init__(
        self,
        *args: Module,
        name: str = "func",
        skip: Union[str, SkipFunc] = "identity",
        join: Union[str, JoinFunc] = "cat",
        dim: int = 1,
    ) -> None:
        ...

    @overload
    def __init__(
        self,
        arg: Mapping[str, Module],
        name: str = "func",
        skip: Union[str, SkipFunc] = "identity",
        join: Union[str, JoinFunc] = "cat",
        dim: int = 1,
    ) -> None:
        ...

    def __init__(
        self,
        *args: Any,
        name: str = "func",
        skip: Union[str, SkipFunc] = "identity",
        join: Union[str, JoinFunc] = "cat",
        dim: int = 1,
    ) -> None:
        super().__init__()
        if len(args) == 1 and isinstance(args[0], Module):
            func = args[0]
        else:
            func = Sequential(*args)
        self.name = name
        if skip in (None, "identity"):
            skip = Identity()
        elif not callable(skip):
            raise ValueError("SkipConnection() 'skip' must be 'identity', callable, or None")
        self.skip = skip
        self.join = join_func(join, dim=dim)
        self._modules[self.name] = func

    @property
    def func(self) -> Module:
        return self._modules[self.name]

    @property
    def shortcut(self) -> SkipFunc:
        return self.skip

    def forward(self, x: Tensor) -> Tensor:
        a = self.skip(x)
        b = self.func(x)
        if not isinstance(a, Tensor):
            raise TypeError("SkipConnection() 'skip' function must return a Tensor")
        if not isinstance(b, Tensor):
            raise TypeError("SkipConnection() module must return a Tensor")
        c = self.join([b, a])
        if not isinstance(c, Tensor):
            raise TypeError("SkipConnection() 'join' function must return a Tensor")
        return c


Shortcut = SkipConnection
