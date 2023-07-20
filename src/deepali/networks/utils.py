import torch
from torch import Tensor, nn
from torch.nn import Module

from deepali.core.nnutils import conv_output_size, conv_transposed_output_size
from deepali.core.nnutils import pad_output_size, upsample_output_size
from deepali.core.nnutils import pool_output_size, unpool_output_size
from deepali.core.typing import ScalarOrTuple

from .layers import Pad
from .blocks import SkipConnection
from .layers import is_activation
from .layers import is_norm_layer


def module_output_size(module: Module, in_size: ScalarOrTuple[int]) -> ScalarOrTuple[int]:
    r"""Calculate spatial size of output tensor after the given module is applied."""
    if not isinstance(module, Module) or type(module) is Module:
        raise TypeError("module_output_size() 'module' must be torch.nn.Module subclass")
    # Modules defining an output_size attribute or function
    output_size = getattr(module, "output_size", None)
    if callable(output_size):
        return output_size(in_size)
    if output_size is not None:
        device = torch.device("cpu")
        m: Tensor = torch.atleast_1d(torch.tensor(in_size, dtype=torch.int, device=device))
        if m.ndim != 1:
            raise ValueError("module_output_size() 'in_size' must be scalar or sequence")
        ndim = m.shape[0]
        s: Tensor = torch.atleast_1d(torch.tensor(output_size, dtype=torch.int, device=device))
        if s.ndim != 1 or s.shape[0] not in (1, ndim):
            raise ValueError(
                f"module_output_size() 'module.output_size' must be scalar or sequence of length {ndim}"
            )
        n = s.expand(ndim)
        if isinstance(in_size, int):
            return n.item()
        return tuple(n.tolist())
    # Network blocks
    if isinstance(module, nn.Sequential):
        size = in_size
        for m in module:
            size = module_output_size(m, size)
        return size
    if isinstance(module, SkipConnection):
        return module_output_size(module.func, in_size)
    # Convolutional layers
    if isinstance(module, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
        return conv_output_size(
            in_size,
            kernel_size=module.kernel_size,
            stride=module.stride,
            padding=module.padding,
            dilation=module.dilation,
        )
    if isinstance(module, (nn.ConvTranspose1d, nn.ConvTranspose2d, nn.ConvTranspose3d)):
        return conv_transposed_output_size(
            in_size,
            kernel_size=module.kernel_size,
            stride=module.stride,
            padding=module.padding,
            output_padding=module.output_padding,
            dilation=module.dilation,
        )
    # Pooling layers
    if isinstance(
        module,
        (
            nn.AvgPool1d,
            nn.AvgPool2d,
            nn.AvgPool3d,
            nn.MaxPool1d,
            nn.MaxPool2d,
            nn.MaxPool3d,
        ),
    ):
        return pool_output_size(
            in_size,
            kernel_size=module.kernel_size,
            stride=module.stride,
            padding=module.padding,
            dilation=module.dilation,
            ceil_mode=module.ceil_mode,
        )
    if isinstance(
        module,
        (
            nn.AdaptiveAvgPool1d,
            nn.AdaptiveAvgPool2d,
            nn.AdaptiveAvgPool3d,
            nn.AdaptiveMaxPool1d,
            nn.AdaptiveMaxPool2d,
            nn.AdaptiveMaxPool3d,
        ),
    ):
        return module.output_size
    if isinstance(module, (nn.MaxUnpool1d, nn.MaxUnpool2d, nn.MaxUnpool3d)):
        return unpool_output_size(
            in_size,
            kernel_size=module.kernel_size,
            stride=module.stride,
            padding=module.padding,
        )
    # Padding
    if isinstance(module, Pad):
        raise NotImplementedError()
    if isinstance(
        module,
        (
            nn.ReflectionPad1d,
            nn.ReflectionPad2d,
            nn.ReplicationPad1d,
            nn.ReplicationPad2d,
            nn.ReplicationPad3d,
            nn.ZeroPad2d,
            nn.ConstantPad1d,
            nn.ConstantPad2d,
            nn.ConstantPad3d,
        ),
    ):
        return pad_output_size(in_size, module.padding)
    # Upsampling
    if isinstance(module, nn.Upsample):
        return upsample_output_size(in_size, size=module.size, scale_factor=module.scale_factor)
    # Activation functions
    if is_activation(module) or isinstance(
        module,
        (
            nn.ELU,
            nn.Hardshrink,
            nn.Hardsigmoid,
            nn.Hardtanh,
            nn.Hardswish,
            nn.LeakyReLU,
            nn.LogSigmoid,
            nn.LogSoftmax,
            nn.PReLU,
            nn.ReLU,
            nn.ReLU6,
            nn.RReLU,
            nn.SELU,
            nn.CELU,
            nn.GELU,
            nn.Sigmoid,
            nn.Softmax,
            nn.Softmax2d,
            nn.Softmin,
            nn.Softplus,
            nn.Softshrink,
            nn.Softsign,
            nn.Tanh,
            nn.Tanhshrink,
            nn.Threshold,
        ),
    ):
        return in_size
    # Normalization layers
    if is_norm_layer(module) or isinstance(
        module,
        (
            nn.BatchNorm1d,
            nn.BatchNorm2d,
            nn.BatchNorm3d,
            nn.SyncBatchNorm,
            nn.GroupNorm,
            nn.InstanceNorm1d,
            nn.InstanceNorm2d,
            nn.InstanceNorm3d,
            nn.LayerNorm,
            nn.LocalResponseNorm,
        ),
    ):
        return in_size
    # Dropout layers
    if isinstance(module, (nn.AlphaDropout, nn.Dropout, nn.Dropout2d, nn.Dropout3d)):
        return in_size
    # Not implemented or invalid type
    if isinstance(module, (nn.ModuleDict, nn.ModuleList)):
        raise TypeError(
            "module_output_size() order of modules in ModuleDict or ModuleList is undetermined"
        )
    if isinstance(module, (nn.ParameterDict, nn.ParameterList)):
        raise TypeError(
            "module_output_size() 'module' cannot be torch.nn.ParameterDict or torch.nn.ParameterList"
        )
    raise NotImplementedError(
        f"module_output_size() not implemented for 'module' of type {type(module)}"
    )
