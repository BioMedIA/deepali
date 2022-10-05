r"""Pooling layers."""

from functools import partial
from typing import Any, Callable, Mapping, Optional, Sequence, Type, Union

from torch import Tensor, nn
from torch.nn import Module

from .lambd import LambdaLayer


PoolFunc = Callable[[Tensor], Tensor]
PoolArg = Union[PoolFunc, str, Mapping, Sequence, None]


POOLING_TYPES = {
    "avg": (nn.AvgPool1d, nn.AvgPool2d, nn.AvgPool3d),
    "avgpool": (nn.AvgPool1d, nn.AvgPool2d, nn.AvgPool3d),
    "adaptiveavg": (nn.AdaptiveAvgPool1d, nn.AdaptiveAvgPool2d, nn.AdaptiveAvgPool3d),
    "adaptiveavgpool": (
        nn.AdaptiveAvgPool1d,
        nn.AdaptiveAvgPool2d,
        nn.AdaptiveAvgPool3d,
    ),
    "adaptivemax": (nn.AdaptiveMaxPool1d, nn.AdaptiveMaxPool2d, nn.AdaptiveMaxPool3d),
    "adaptivemaxpool": (
        nn.AdaptiveMaxPool1d,
        nn.AdaptiveMaxPool2d,
        nn.AdaptiveMaxPool3d,
    ),
    "max": (nn.MaxPool1d, nn.MaxPool2d, nn.MaxPool3d),
    "maxpool": (nn.MaxPool1d, nn.MaxPool2d, nn.MaxPool3d),
    "maxunpool": (nn.MaxUnpool1d, nn.MaxUnpool2d, nn.MaxUnpool3d),
    "identity": nn.Identity,
}


def pooling(
    arg: PoolArg,
    *args: Any,
    spatial_dims: Optional[int] = None,
    **kwargs,
) -> Module:
    r"""Get pooling layer.

    Args:
        arg: Custom pooling function or module, or name of pooling layer with optional keyword arguments.
            When ``arg`` is a callable but not of type ``torch.nn.Module``, it is wrapped in a ``PoolLayer``.
            If ``None`` or 'identity', an instance of ``torch.nn.Identity`` is returned.
        spatial_dims: Number of spatial dimensions of input tensors.
        args: Arguments to pass to init function of pooling layer. If ``arg`` is a callable, the given arguments
            are passed to the function each time it is called as arguments.
        kwargs: Additional keyword arguments for initialization of pooling layer. Overrides keyword arguments given as
            second tuple item when ``arg`` is a ``(name, kwargs)`` tuple instead of a string. When ``arg`` is a callable,
            the keyword arguments are passed each time the pooling function is called.

    Returns:
        Pooling layer instance.

    """
    if isinstance(arg, Module) and not args and not kwargs:
        return arg
    if callable(arg):
        return PoolLayer(arg, *args, **kwargs)
    pool_name = "identity"
    pool_args = {}
    if isinstance(arg, str):
        pool_name = arg
    elif isinstance(arg, Mapping):
        pool_name = arg.get("name")
        if not pool_name:
            raise ValueError("pooling() 'arg' map must contain 'name'")
        if not isinstance(pool_name, str):
            raise TypeError("pooling() 'name' must be str")
        pool_args = {key: value for key, value in arg.items() if key != "name"}
    elif isinstance(arg, Sequence):
        if len(arg) != 2:
            raise ValueError("pooling() 'arg' sequence must have length two")
        pool_name, pool_args = arg
        if not isinstance(pool_name, str):
            raise TypeError("pooling() first 'arg' sequence argument must be str")
        if not isinstance(pool_args, dict):
            raise TypeError("pooling() second 'arg' sequence argument must be dict")
        pool_args = pool_args.copy()
    elif arg is not None:
        raise TypeError("pooling() 'arg' must be str, mapping, 2-tuple, callable, or None")
    pool_type: Union[Type[Module], Sequence[Type[Module]]] = POOLING_TYPES.get(pool_name.lower())
    if pool_type is None:
        raise ValueError(f"pooling() unknown pooling layer {pool_name!r}")
    if isinstance(pool_type, Sequence):
        if spatial_dims is None:
            raise ValueError(f"pooling() 'spatial_dims' required for pooling layer {pool_name!r}")
        try:
            pool_type = pool_type[spatial_dims - 1]
        except IndexError:
            pool_type = None
        if pool_type is None:
            raise ValueError(f"pooling({pool_name!r}) does not support spatial_dims={spatial_dims}")
    pool_args.update(kwargs)
    module = pool_type(*args, **pool_args)
    return module


def pool_layer(*args, **kwargs) -> Module:
    return pooling(*args, **kwargs)


class PoolLayer(LambdaLayer):
    r"""Pooling layer."""

    def __init__(
        self,
        arg: PoolArg,
        *args: Any,
        spatial_dims: Optional[int] = None,
        **kwargs,
    ) -> None:
        if callable(arg):
            pool = partial(arg, *args, **kwargs) if args or kwargs else arg
        else:
            pool = pooling(arg, *args, spatial_dims=spatial_dims, **kwargs)
        super().__init__(pool)
