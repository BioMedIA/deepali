r"""Wrap any function in a torch.nn.Module."""

from typing import Callable

from torch import Tensor
from torch.nn import Module


LambdaFunc = Callable[[Tensor], Tensor]


class LambdaLayer(Module):
    r"""Wrap any tensor operation in a network module."""

    def __init__(self, func: LambdaFunc) -> None:
        r"""Set callable tensor operation.

        Args:
            func: Callable tensor operation. Must be instance of ``torch.nn.Module``
                if it contains learnable parameters. In this case, however, the
                ``LambdaLayer`` wrapper becomes redundant. Main use is to wrap
                non-learnable Python functions.

        """
        super().__init__()
        self.func = func

    def forward(self, x: Tensor) -> Tensor:
        return self.func(x)
