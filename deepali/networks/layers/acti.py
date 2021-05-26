r"""Non-linear activation functions."""

from functools import partial
from typing import Any, Callable, Mapping, Optional, Sequence, Type, Union

from torch import Tensor, nn
from torch.nn import Module

from .lambd import LambdaLayer


ActivationFunc = Callable[[Tensor], Tensor]
ActivationArg = Union[ActivationFunc, str, Mapping, Sequence, None]


ACTIVATION_TYPES = {
    "celu": nn.CELU,
    "elu": nn.ELU,
    "hardtanh": nn.Hardtanh,
    "identity": nn.Identity,
    "none": nn.Identity,
    "relu": nn.ReLU,
    "relu6": nn.ReLU6,
    "lrelu": nn.LeakyReLU,
    "leakyrelu": nn.LeakyReLU,
    "leaky_relu": nn.LeakyReLU,
    "rrelu": nn.RReLU,
    "selu": nn.SELU,
    "gelu": nn.GELU,
    "hardshrink": nn.Hardshrink,
    "hardsigmoid": nn.Hardsigmoid,
    "hardswish": nn.Hardswish,
    "logsigmoid": nn.LogSigmoid,
    "logsoftmax": nn.LogSoftmax,
    "prelu": nn.PReLU,
    "sigmoid": nn.Sigmoid,
    "softmax": nn.Softmax,
    "softmin": nn.Softmin,
    "softplus": nn.Softplus,
    "softshrink": nn.Softshrink,
    "softsign": nn.Softsign,
    "tanh": nn.Tanh,
    "tanhshrink": nn.Tanhshrink,
}

INPLACE_ACTIVATIONS = {
    "elu",
    "hardtanh",
    "lrelu",
    "leakyrelu",
    "relu",
    "relu6",
    "rrelu",
    "selu",
    "celu",
}

SOFTMINMAX_ACTIVATIONS = {"softmin", "softmax", "logsoftmax"}


def activation(
    arg: ActivationArg,
    *args: Any,
    dim: Optional[int] = None,
    inplace: Optional[bool] = None,
    **kwargs,
) -> Module:
    r"""Get activation function.

    Args:
        arg: Custom activation function or module, or name of activation function with optional keyword arguments.
        args: Arguments to pass to activation init function.
        dim: Dimension along which to compute softmax activations (cf. ``ACT_SOFTMINMAX``). Unused by other activations.
        inplace: Whether to compute activation output in place. Unused if unsupported by specified activation function.
        kwargs: Additional keyword arguments for activation function. Overrides keyword arguments given as second
            tuple item when ``arg`` is a ``(name, kwargs)`` tuple instead of a string.

    Returns:
        Given activation function when ``arg`` is a ``torch.nn.Module``, or a new activation module otherwise.

    """
    if isinstance(arg, Module) and not args and not kwargs:
        return arg
    if callable(arg):
        return Activation(arg, *args, **kwargs)
    acti_name = "identity"
    acti_args = {}
    if isinstance(arg, str):
        acti_name = arg
    elif isinstance(arg, Mapping):
        acti_name = arg.get("name")
        if not acti_name:
            raise ValueError("activation() 'arg' map must contain 'name'")
        if not isinstance(acti_name, str):
            raise TypeError("activation() 'name' must be str")
        acti_args = {key: value for key, value in arg.items() if key != "name"}
    elif isinstance(arg, Sequence):
        if len(arg) != 2:
            raise ValueError("activation() 'arg' sequence must have length two")
        acti_name, acti_args = arg
        if not isinstance(acti_name, str):
            raise TypeError("activation() first 'arg' sequence argument must be str")
        if not isinstance(acti_args, dict):
            raise TypeError("activation() second 'arg' sequence argument must be dict")
        acti_args = acti_args.copy()
    elif arg is not None:
        raise TypeError("activation() 'arg' must be str, mapping, 2-tuple, callable, or None")
    acti_name = acti_name.lower()
    acti_type: Type[Module] = ACTIVATION_TYPES.get(acti_name)
    if acti_type is None:
        raise ValueError(
            f"activation() 'arg' name {acti_name!r} is unknown."
            " Pass a callable activation function or module instead."
        )
    acti_args.update(kwargs)
    if inplace is not None and acti_name in INPLACE_ACTIVATIONS:
        acti_args["inplace"] = bool(inplace)
    if acti_name in SOFTMINMAX_ACTIVATIONS:
        if dim is None and len(args) == 1 and isinstance(args, int):
            dim = args[0]
        elif args or acti_args:
            raise ValueError("activation() named {act_name!r} has no parameters")
        if dim is None:
            dim = 1
        acti = acti_type(dim)
    else:
        acti = acti_type(*args, **acti_args)
    return acti


class Activation(LambdaLayer):
    r"""Non-linear activation function."""

    def __init__(
        self,
        arg: ActivationArg,
        *args: Any,
        dim: int = 1,
        inplace: Optional[bool] = None,
        **kwargs,
    ) -> None:
        if callable(arg):
            acti = partial(arg, *args, **kwargs) if args or kwargs else arg
        else:
            acti = activation(arg, *args, dim=dim, inplace=inplace, **kwargs)
        super().__init__(acti)


def is_activation(arg: Any) -> bool:
    r"""Whether given object is an non-linear activation function module."""
    if isinstance(arg, Activation):
        return True
    types = tuple(ACTIVATION_TYPES.values())
    return isinstance(arg, types)
