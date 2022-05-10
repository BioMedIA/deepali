r"""Normalization layers."""

from numbers import Integral
from functools import partial
from typing import Any, Callable, Mapping, Optional, Sequence, Union

from torch import Tensor, nn
from torch.nn import Module

from .lambd import LambdaLayer


NormFunc = Callable[[Tensor], Tensor]
NormArg = Union[NormFunc, str, Mapping[str, Any], Sequence, None]


def normalization(
    arg: NormArg,
    *args,
    spatial_dims: Optional[int] = None,
    num_features: Optional[int] = None,
    **kwargs,
) -> Module:
    r"""Create normalization layer.

    Args:
        arg: Custom normalization function or module, or name of normalization layer with optional keyword arguments.
        args: Positional arguments passed to normalization layer.
        num_features: Number of input features.
        spatial_dims: Number of spatial dimensions of input tensors.
        kwargs: Additional keyword arguments for normalization layer. Overrides keyword arguments given as second
            tuple item when ``arg`` is a ``(name, kwargs)`` tuple instead of a string.

    Returns:
        Given normalization function when ``arg`` is a ``torch.nn.Module``, or a new normalization layer otherwise.

    """
    if isinstance(arg, Module) and not args and not kwargs:
        return arg
    if callable(arg):
        return NormLayer(arg, *args, **kwargs)
    norm_name = "identity"
    norm_args = {}
    if isinstance(arg, str):
        norm_name = arg
    elif isinstance(arg, Mapping):
        norm_name = arg.get("name")
        if not norm_name:
            raise ValueError("normalization() 'arg' map must contain 'name'")
        if not isinstance(norm_name, str):
            raise TypeError("normalization() 'name' must be str")
        norm_args = {key: value for key, value in arg.items() if key != "name"}
    elif isinstance(arg, Sequence):
        if len(arg) != 2:
            raise ValueError("normalization() 'arg' sequence must have length two")
        norm_name, norm_args = arg
        if not isinstance(norm_name, str):
            raise TypeError("normalization() first 'arg' sequence argument must be str")
        if not isinstance(norm_args, dict):
            if norm_name == "group" and isinstance(norm_args, Integral):
                norm_args = dict(num_groups=norm_args)
            else:
                raise TypeError("normalization() second 'arg' sequence argument must be dict")
        norm_args = norm_args.copy()
    elif arg is not None:
        raise TypeError("normalization() 'arg' must be str, mapping, 2-tuple, callable, or None")
    norm_name = norm_name.lower()
    norm_args.update(kwargs)
    if norm_name in ("none", "identity"):
        norm = nn.Identity()
    elif norm_name in ("batch", "batchnorm"):
        if spatial_dims is None:
            raise ValueError("normalization() 'spatial_dims' required for 'batch' norm")
        if spatial_dims < 0 or spatial_dims > 3:
            raise ValueError("normalization() 'spatial_dims' must be 1, 2, or 3")
        if num_features is None:
            raise ValueError("normalization() 'num_features' required for 'batch' norm")
        norm_type = (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)[spatial_dims - 1]
        norm = norm_type(num_features, *args, **norm_args)
    elif norm_name in ("group", "groupnorm"):
        num_groups = norm_args.pop("num_groups", 1)
        if num_features is None:
            if "num_channels" not in norm_args:
                raise ValueError("normalization() 'num_features' required for 'group' norm")
            num_features = norm_args.pop("num_channels")
        norm = nn.GroupNorm(num_groups, num_features, *args, **norm_args)
    elif norm_name in ("layer", "layernorm"):
        if num_features is None:
            raise ValueError("normalization() 'num_features' required for 'layer' norm")
        # This is equivalent to ("group", channels), not torch.nn.LayerNorm
        # (see also https://arxiv.org/abs/1803.08494, Figure 2).
        norm = nn.GroupNorm(num_features, num_features, *args, **norm_args)
    elif norm_name in ("instance", "instancenorm"):
        if spatial_dims is None:
            raise ValueError("normalization() 'spatial_dims' required for 'instance' norm")
        if spatial_dims < 0 or spatial_dims > 3:
            raise ValueError("normalization() 'spatial_dims' must be 1, 2, or 3")
        if num_features is None:
            raise ValueError("normalization() 'num_features' required for 'instance' norm")
        norm_type = (nn.InstanceNorm1d, nn.InstanceNorm2d, nn.InstanceNorm3d)[spatial_dims - 1]
        norm = norm_type(num_features, *args, **norm_args)
    else:
        raise ValueError("normalization() unknown layer type {norm_name!r}")
    return norm


def norm_layer(*args, **kwargs) -> Module:
    r"""Create normalization layer, see ``normalization``."""
    return normalization(*args, **kwargs)


def is_norm_layer(arg: Any) -> bool:
    r"""Whether given module is a normalization layer."""
    if isinstance(arg, NormLayer):
        return True
    return is_batch_norm(arg) or is_group_norm(arg) or is_instance_norm(arg)


def is_batch_norm(arg: Any) -> bool:
    r"""Whether given module is a batch normalization layer."""
    return isinstance(arg, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d))


def is_group_norm(arg: Any) -> bool:
    r"""Whether given module is a group normalization layer."""
    return isinstance(arg, nn.GroupNorm)


def is_instance_norm(arg: Any) -> bool:
    r"""Whether given module is an instance normalization layer."""
    return isinstance(arg, (nn.InstanceNorm1d, nn.InstanceNorm2d, nn.InstanceNorm3d))


class NormLayer(LambdaLayer):
    r"""Normalization layer."""

    def __init__(
        self,
        arg: NormArg,
        *args,
        spatial_dims: Optional[int] = None,
        num_features: Optional[int] = None,
        **kwargs: Mapping[str, Any],
    ) -> None:
        if callable(arg):
            norm = partial(arg, *args, **kwargs) if args or kwargs else arg
        else:
            kwargs.update(dict(spatial_dims=spatial_dims, num_features=num_features))
            norm = normalization(arg, *args, **kwargs)
        super().__init__(norm)
