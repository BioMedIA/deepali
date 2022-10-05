r"""Join features from separate network paths."""

from typing import Callable, Sequence, Union

import torch
from torch import Tensor

from .lambd import LambdaLayer


JoinFunc = Callable[[Sequence[Tensor]], Tensor]


def join_func(arg: Union[str, JoinFunc], dim: int = 1) -> JoinFunc:
    r"""Tensor operation which combines features of input tensors, e.g., along skip connection.

    Args:
        arg: Name of operation: 'add': Elementwise addition, 'cat' or 'concat': Concatenate along feature dimension.
        dim: Dimension of input tensors containing features.

    """
    if callable(arg):
        return arg

    if not isinstance(arg, str):
        raise TypeError("join_func() 'arg' must be str or callable")

    name = arg.lower()
    if name == "add":

        def add(args: Sequence[Tensor]) -> Tensor:
            assert args, "join_func('add') requires at least one input tensor"
            out = args[0]
            for i in range(1, len(args)):
                out = out + args[i]
            return out

        return add

    elif name in ("cat", "concat"):

        def cat(args: Sequence[Tensor]) -> Tensor:
            assert args, "join_func('cat') requires at least one input tensor"
            return torch.cat(args, dim=dim)

        return cat

    elif name == "mul":

        def mul(args: Sequence[Tensor]) -> Tensor:
            assert args, "join_func('mul') requires at least one input tensor"
            out = args[0]
            for i in range(1, len(args)):
                out = out * args[i]
            return out

        return mul

    raise ValueError("join_func() unknown merge function {name!r}")


class JoinLayer(LambdaLayer):
    r"""Merge network branches."""

    def __init__(self, arg: Union[str, JoinFunc], dim: int = 1) -> None:
        func = join_func(arg, dim=dim)
        super().__init__(func)

    def forward(self, xs: Sequence[Tensor]) -> Tensor:
        return self.func(xs)

    def extra_repr(self) -> str:
        return repr(self.func.__name__)
