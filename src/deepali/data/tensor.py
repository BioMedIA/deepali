r"""Base class of tensor subclasses with additional attributes and methods."""

from __future__ import annotations

from collections import OrderedDict
from typing import Optional, Type, TypeVar

import torch
import torch.utils.hooks
from torch import Tensor

from deepali.core.tensor import as_tensor
from deepali.core.typing import Array, Device, DType


T = TypeVar("T", bound="DataTensor")


__all__ = ("DataTensor",)


class DataTensor(Tensor):
    r"""Data tensor base class."""

    def __new__(
        cls: Type[T],
        data: Array,
        *args,
        dtype: Optional[DType] = None,
        device: Optional[Device] = None,
        requires_grad: Optional[bool] = None,
        pin_memory: bool = False,
        **kwargs,
    ) -> T:
        data = as_tensor(data, dtype=dtype, device=device)
        if requires_grad is None:
            requires_grad = data.requires_grad
        if pin_memory:
            data = data.pin_memory()
        return Tensor._make_subclass(cls, data, requires_grad)

    def __init__(
        self: T,
        data: Array,
        dtype: Optional[DType] = None,
        device: Optional[Device] = None,
        requires_grad: Optional[bool] = None,
        pin_memory: bool = False,
    ) -> None:
        r"""Initialize data tensor.

        Args:
            data: Tensor data.
            dtype: Data type. A copy of the data is only made when the desired
                ``dtype`` is not ``None`` and not the same as ``data.dtype``.
            device: Device on which to store the data. A copy of the data is only made when
                the data has to be copied to a different device.
            requires_grad: If autograd should record operations on the returned data tensor.
            pin_memory: If set, returned data tensor would be allocated in the pinned memory.
                Works only for CPU tensors.

        """
        ...

    def _make_instance(self: T, data: Optional[Tensor] = None, **kwargs) -> T:
        r"""Create a new instance while preserving subclass (meta-)data."""
        if data is None:
            data = self
        if type(data) is not Tensor:
            data = data.as_subclass(Tensor)
        return type(self)(data, **kwargs)

    def __copy__(self: T) -> T:
        return self._make_instance()

    def __deepcopy__(self: T, memo) -> T:
        if id(self) in memo:
            return memo[id(self)]
        result = self._make_instance(
            self.data.clone(memory_format=torch.preserve_format),
            requires_grad=self.requires_grad,
            pin_memory=self.is_pinned(),
        )
        memo[id(self)] = result
        return result

    def __reduce_ex__(self, proto):
        # See also: https://github.com/pytorch/pytorch/issues/47051#issuecomment-877788874
        torch.utils.hooks.warn_if_has_hooks(self)
        args = (
            self.storage(),
            self.storage_offset(),
            tuple(self.size()),
            self.stride(),
        )
        if self.is_quantized:
            args = args + (self.q_scale(), self.q_zero_point())
        args = args + (self.requires_grad, OrderedDict())
        f = torch._utils._rebuild_qtensor if self.is_quantized else torch._utils._rebuild_tensor_v2
        return (_rebuild_from_type, (f, type(self), args, self.__dict__))

    def tensor(self: T) -> Tensor:
        r"""Convert to plain torch.Tensor."""
        return self.as_subclass(Tensor)


def _rebuild_from_type(func, type, args, dict):
    r"""Function used by DataTensor.__reduce_ex__ to support unpickling of subclass type."""
    # from https://github.com/pytorch/pytorch/blob/13c975684a220ec096216ec6468ccd0dc90ff50a/torch/_tensor.py#L34
    ret: Tensor = func(*args)
    if type is not Tensor:
        ret = ret.as_subclass(type)
        ret.__dict__ = dict
    return ret
