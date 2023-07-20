r"""Basic tensor operations."""

from typing import Any, Mapping, Optional, Sequence, Union

from torch import Tensor
from torch.nn import Module

from deepali.core import functional as U
from deepali.core.enum import PaddingMode
from deepali.core.typing import ScalarOrTuple


class GetItem(Module):
    r"""Get item at specified input tensor sequence index or with given dictionary key."""

    def __init__(self, key: Any) -> None:
        r"""Set item index or key.

        Args:
            key: Index of item in sequence of input tensors or key into input map.

        """
        super().__init__()
        self.key = key

    def forward(self, input: Union[Sequence[Tensor], Mapping[Any, Tensor]]) -> Tensor:
        return input[self.key]

    def extra_repr(self) -> str:
        return repr(self.key)


class Narrow(Module):
    r"""Narrowed version of input tensor."""

    def __init__(self, dim: int, start: int, length: int) -> None:
        super().__init__()
        self.dim = dim
        self.start = start
        self.length = length

    def forward(self, x: Tensor) -> Tensor:
        return x.narrow(self.dim, self.start, self.length)

    def extra_repr(self) -> str:
        return f"dim={self.dim}, start={self.start}, length={self.length}"


class Pad(Module):
    r"""Pad tensor."""

    def __init__(
        self,
        margin: Optional[ScalarOrTuple[int]] = None,
        padding: Optional[ScalarOrTuple[int]] = None,
        mode: Union[PaddingMode, str] = PaddingMode.ZEROS,
        value: float = 0,
    ) -> None:
        if margin is None and padding is None:
            raise AssertionError("Pad() either 'margin' or 'padding' is required")
        if margin is not None and padding is not None:
            raise AssertionError("Pad() 'margin' and 'padding' are mutually exclusive")
        super().__init__()
        self.margin = margin
        self.padding = padding
        self.mode = PaddingMode(mode)
        self.value = value

    def forward(self, x: Tensor) -> Tensor:
        return U.pad(x, margin=self.margin, num=self.padding, mode=self.mode, value=self.value)

    def extra_repr(self) -> str:
        if self.margin is None:
            s = f"padding={self.padding}"
        else:
            s = f"margin={self.margin}"
        s += f", mode={self.mode.value!r}"
        if self.mode is PaddingMode.CONSTANT:
            s += f", value={self.value}"
        return s


class Reshape(Module):
    r"""Reshape input tensor.

    This module provides a view of the input tensor without making a copy if possible.
    Otherwise, a copy is made of the input data. See ``torch.reshape()`` for details.

    """

    def __init__(self, shape: Sequence[int]) -> None:
        r"""Set output tensor shape.

        Args:
            shape: Output tensor shape, optionally excluding first batch dimension.

        """
        super().__init__()
        self.shape = shape

    def forward(self, x: Tensor) -> Tensor:
        shape = self.shape
        if len(shape) == x.ndim - 1:
            shape = (-1,) + shape
        return x.reshape(shape)

    def extra_repr(self) -> str:
        return repr(self.shape)


class View(Module):
    r"""View input tensor with specified shape.

    See https://pytorch.org/docs/stable/tensor_view.html for when it is possible to create
    a view of a tensor without reshaping it. For a more general tensor reshape operation,
    see module ``Reshape``.

    """

    def __init__(self, shape: Sequence[int]) -> None:
        r"""Set output tensor shape.

        Args:
            shape: Output tensor shape, optionally excluding first batch dimension.

        """
        super().__init__()
        self.shape = shape

    def forward(self, x: Tensor) -> Tensor:
        shape = self.shape
        if len(shape) == x.ndim - 1:
            shape = (-1,) + shape
        return x.view(*shape)

    def extra_repr(self) -> str:
        return repr(self.shape)
