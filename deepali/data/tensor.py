r"""Tensor-like image data wrapper."""

# TODO: Since PyTorch 1.7.0, we can make Tensor API available in decorator using __torch_function__
# (cf. https://pytorch.org/docs/stable/notes/extending.html#extending-torch-with-a-tensor-wrapper-type)

from __future__ import annotations

from copy import copy as shallow_copy
from typing import Optional, TypeVar, Union, cast, overload

from numpy import ndarray
import torch
from torch import Tensor

from ..core.tensor import as_tensor, named_dims
from ..core.types import Array, Scalar


T = TypeVar("T", bound="TensorDecorator")


__all__ = ("TensorDecorator",)


class TensorDecorator(object):
    """Tensor decorator base class."""

    __slots__ = ("_tensor",)

    def __init__(self: T, data: Tensor) -> None:
        """Initialize wrapped tensor."""
        if not torch.is_tensor(data):
            raise TypeError("Decorated Tensor 'data' must be a torch.Tensor")
        self._tensor = data

    @overload
    def tensor(self: T) -> Tensor:
        """Get wrapped tensor."""
        ...

    @overload
    def tensor(self: T, data: Tensor, **kwargs) -> T:
        """Get shallow copy which wraps given tensor."""
        ...

    def tensor(
        self: T,
        data: Optional[Tensor] = None,
        **kwargs,
    ) -> Union[Tensor, T]:
        """Get wrapped tensor or shallow copy which wraps given tensor."""
        if data is None:
            return self._tensor
        return shallow_copy(self).tensor_(data, **kwargs)

    def tensor_(self: T, data: Tensor) -> T:
        """Replace wrapped tensor."""
        if not torch.is_tensor(data):
            raise TypeError("Decorated Tensor 'data' must be a torch.Tensor")
        self._tensor = data
        return self

    def numpy(self) -> ndarray:
        r"""Get tensor data as NumPy array."""
        return self._tensor.numpy()

    @property
    def dtype(self: T) -> torch.dtype:
        """Type of image data."""
        return self._tensor.dtype

    @property
    def device(self: T) -> torch.device:
        """Device on which image data is stored."""
        return self._tensor.device

    @property
    def shape(self: T) -> torch.Size:
        """Shape of image data tensor."""
        return self._tensor.shape

    @property
    def data(self: T) -> Tensor:
        """Detached image data tensor, i.e., ``self.tensor().data``."""
        return self._tensor.data

    def detach(self: T) -> T:
        """Detach image data from computation graph.

        Returns:
            Shallow copy of ``self`` with detached ``_tensor``.

        """
        other = shallow_copy(self)
        other._tensor = self._tensor.detach()
        return other

    def to(self: T, *args, **kwargs) -> T:
        """Change dtype and/or device of data tensor.

        Returns:
            Shallow copy of ``self`` with possibly new ``_tensor``.

        """
        other = shallow_copy(self)
        if len(args) == 1 and not kwargs and isinstance(args[0], TensorDecorator):
            args = [cast(TensorDecorator, args[0])._tensor]
        other._tensor = self._tensor.to(*args, **kwargs)
        return other

    def type(self: T, dtype: torch.dtype) -> T:
        """Change dtype of data tensor.

        Returns:
            Shallow copy of ``self`` with possibly new ``_tensor``.

        """
        other = shallow_copy(self)
        other._tensor = self._tensor.type(dtype)
        return other

    def pin_memory(self) -> T:
        """Put wrapped tensor data in pinned memory."""
        self._tensor = self._tensor.pin_memory()
        return self

    def is_pinned(self) -> bool:
        """Whether wrapped tensor is in pinned memory."""
        return self._tensor.is_pinned()

    def permute(self: T, *names: str) -> Tensor:
        """Image data tensor with permuted named dimensions."""
        dims = named_dims(self._tensor, *names)
        return self._tensor.permute(dims)

    def clamp(self: T, min: Optional[Scalar] = None, max: Optional[Scalar] = None) -> T:
        """Element-wise clamp values to specified min/max values."""
        return shallow_copy(self).clamp_(min=min, max=max)

    def clamp_(self: T, min: Optional[Scalar] = None, max: Optional[Scalar] = None) -> T:
        """Element-wise clamp values to specified min/max values."""
        if min is None and max is not None:
            data = self._tensor.clamp(max=float(max))
        elif min is not None and max is None:
            data = self._tensor.clamp(min=float(min))
        elif min is not None and max is not None:
            data = self._tensor.clamp(min=float(min), max=float(max))
        else:
            return self
        return self.tensor_(data)

    def add(self: T, other: Union[Scalar, Array, TensorDecorator]) -> T:
        """Element-wise add values of this and other tensor or constant."""
        if isinstance(other, TensorDecorator):
            other = other._tensor
        other = as_tensor(other, dtype=self.dtype, device=self.device)
        result = self._tensor + other
        return self.tensor(result)

    def add_(self: T, other: Union[Scalar, Array, TensorDecorator]) -> T:
        """Element-wise add values of other tensor or constant to this tensor."""
        if isinstance(other, TensorDecorator):
            other = other._tensor
        other = as_tensor(other, dtype=self.dtype, device=self.device)
        self._tensor += other
        return self

    def sub(self: T, other: Union[Scalar, Array, TensorDecorator]) -> T:
        """Element-wise subtract values of this and other tensor or constant."""
        if isinstance(other, TensorDecorator):
            other = other._tensor
        other = as_tensor(other, dtype=self.dtype, device=self.device)
        result = self._tensor - other
        return self.tensor(result)

    def sub_(self: T, other: Union[Scalar, Array, TensorDecorator]) -> T:
        """Element-wise subtract values of other tensor or constant to this tensor."""
        if isinstance(other, TensorDecorator):
            other = other._tensor
        other = as_tensor(other, dtype=self.dtype, device=self.device)
        self._tensor -= other
        return self

    def mul(self: T, other: Union[Scalar, Array, TensorDecorator]) -> T:
        """Element-wise multiply values of this and other tensor or constant."""
        if isinstance(other, TensorDecorator):
            other = other._tensor
        other = as_tensor(other, dtype=self.dtype, device=self.device)
        result = self._tensor * other
        return self.tensor(result)

    def mul_(self: T, other: Union[Scalar, Array, TensorDecorator]) -> T:
        """Element-wise multiply values of this tensor by values of other tensor or constant."""
        if isinstance(other, TensorDecorator):
            other = other._tensor
        other = as_tensor(other, dtype=self.dtype, device=self.device)
        self._tensor *= other
        return self

    def div(self: T, other: Union[Scalar, Array, TensorDecorator]) -> T:
        """Element-wise divide values of this by values of other tensor or constant."""
        if isinstance(other, TensorDecorator):
            other = other._tensor
        other = as_tensor(other, dtype=self.dtype, device=self.device)
        result = self._tensor / other
        return self.tensor(result)

    def div_(self: T, other: Union[Scalar, Array, TensorDecorator]) -> T:
        """Element-wise divide values of this tensor by values of other tensor or constant."""
        if isinstance(other, TensorDecorator):
            other = other._tensor
        other = as_tensor(other, dtype=self.dtype, device=self.device)
        self._tensor /= other
        return self
