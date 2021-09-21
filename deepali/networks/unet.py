r"""U-net model architectures."""

from __future__ import annotations

from copy import deepcopy
from collections import OrderedDict
from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Type, Union

from torch import Tensor
from torch.nn import Identity, Module, ModuleDict, Sequential

from ..core.config import DataclassConfig
from ..core.enum import PaddingMode
from ..core.itertools import repeat_last
from ..core.types import ScalarOrTuple
from ..modules import GetItem, ReprWithCrossReferences

from .blocks import ResidualUnit
from .layers import ActivationArg, ConvLayer, JoinLayer, NormArg, PoolLayer
from .layers import Upsample, UpsampleMode
from .utils import module_output_size


__all__ = (
    "SequentialUNet",
    "UNet",
    "UNetConfig",
    "UNetDecoder",
    "UNetDecoderConfig",
    "UNetDownsampleConfig",
    "UNetEncoder",
    "UNetEncoderConfig",
    "UNetLayerConfig",
    "UNetOutputConfig",
    "UNetUpsampleConfig",
    "unet_conv_block",
)


ModuleFactory = Union[Callable[..., Module], Type[Module]]

NumChannels = Sequence[Union[int, Sequence[int]]]
NumBlocks = Union[int, Sequence[int]]
NumLayers = Optional[Union[int, Sequence[int]]]


def reversed_num_channels(num_channels: NumChannels) -> NumChannels:
    r"""Reverse order of per-block/-stage number of feature channels."""
    rev_channels = tuple(
        tuple(reversed(c)) if isinstance(c, Sequence) else c for c in reversed(num_channels)
    )
    return rev_channels


def first_num_channels(num_channels: NumChannels) -> int:
    r"""Get number of feature channels of first block."""
    nc = num_channels[0]
    if isinstance(nc, Sequence):
        nc = nc[0]
    return nc


def last_num_channels(num_channels: NumChannels) -> int:
    r"""Get number of feature channels of last block."""
    nc = num_channels[-1]
    if isinstance(nc, Sequence):
        nc = nc[-1]
    return nc


@dataclass
class UNetLayerConfig(DataclassConfig):

    kernel_size: ScalarOrTuple[int] = 3
    dilation: ScalarOrTuple[int] = 1
    padding: Optional[ScalarOrTuple[int]] = None
    padding_mode: Union[PaddingMode, str] = "zeros"
    init: str = "default"
    bias: Union[str, bool, None] = None
    norm: NormArg = "instance"
    acti: ActivationArg = "lrelu"
    order: str = "cna"

    def __post_init__(self):
        self._join_kwargs_in_sequence("acti")
        self._join_kwargs_in_sequence("norm")


@dataclass
class UNetDownsampleConfig(DataclassConfig):

    mode: str = "conv"
    factor: Union[int, Sequence[int]] = 2
    kernel_size: Optional[ScalarOrTuple[int]] = None
    padding: Optional[ScalarOrTuple[int]] = None


@dataclass
class UNetUpsampleConfig(DataclassConfig):

    mode: Union[str, UpsampleMode] = "deconv"
    factor: Union[int, Sequence[int]] = 2
    kernel_size: Optional[ScalarOrTuple[int]] = None
    dilation: Optional[ScalarOrTuple[int]] = None
    padding: Optional[ScalarOrTuple[int]] = None


@dataclass
class UNetOutputConfig(DataclassConfig):

    channels: int = 1
    kernel_size: int = 1
    dilation: int = 1
    padding: Optional[int] = None
    padding_mode: Union[PaddingMode, str] = "zeros"
    init: str = "default"
    bias: Union[str, bool, None] = False
    norm: NormArg = None
    acti: ActivationArg = None
    order: str = "cna"

    def __post_init__(self):
        self._join_kwargs_in_sequence("acti")
        self._join_kwargs_in_sequence("norm")


@dataclass
class UNetEncoderConfig(DataclassConfig):

    num_channels: NumChannels = (8, 16, 32, 64)
    num_blocks: NumBlocks = 2
    num_layers: NumLayers = None
    conv_layer: UNetLayerConfig = UNetLayerConfig()
    downsample: Union[str, UNetDownsampleConfig] = UNetDownsampleConfig()
    residual: bool = False

    # When torch.backends.cudnn.deterministic == True, then a dilated convolution at layer_1_2
    # causes a "CUDA error: CUBLAS_STATUS_ALLOC_FAILED when calling `cublasCreate(handle)`".
    # This can be resolved by either setting torch.backends.cudnn.deterministic = False, or
    # by not using a dilated convolution for layer_1_2. See also related GitHub issue reported
    # at https://github.com/pytorch/pytorch/issues/32035.

    # dilation different from conv_layer.dilation for first block in each stage
    block_1_dilation: Optional[int] = None
    # dilation different from conv_layer.dilation for first stage
    stage_1_dilation: Optional[int] = None

    @property
    def num_levels(self) -> int:
        r"""Number of spatial encoder levels."""
        return len(self.num_channels)

    @property
    def out_channels(self) -> int:
        return last_num_channels(self.num_channels)

    def __post_init__(self):
        if isinstance(self.downsample, str):
            self.downsample = UNetDownsampleConfig(self.downsample)


@dataclass
class UNetDecoderConfig(DataclassConfig):

    num_channels: NumChannels = (64, 32, 16, 8)
    num_blocks: NumBlocks = 2
    num_layers: NumLayers = None
    conv_layer: UNetLayerConfig = UNetLayerConfig()
    upsample: Union[str, UNetUpsampleConfig] = UNetUpsampleConfig()
    join_mode: str = "cat"
    residual: bool = False

    @property
    def num_levels(self) -> int:
        r"""Number of spatial decoder levels, including bottleneck input."""
        return len(self.num_channels)

    @property
    def in_channels(self) -> int:
        return first_num_channels(self.num_channels)

    @property
    def out_channels(self) -> int:
        return last_num_channels(self.num_channels)

    def __post_init__(self):
        if isinstance(self.upsample, (str, UpsampleMode)):
            self.upsample = UNetUpsampleConfig(self.upsample)

    @classmethod
    def from_encoder(
        cls,
        encoder: Union[UNetEncoder, UNetEncoderConfig],
        residual: Optional[bool] = None,
        **kwargs,
    ) -> UNetDecoderConfig:
        r"""Derive decoder configuration from U-net encoder configuration."""
        if isinstance(encoder, UNetEncoder):
            encoder = encoder.config
        if not isinstance(encoder, UNetEncoderConfig):
            raise TypeError(
                f"{cls.__name__}.from_encoder() argument must be UNetEncoder or UNetEncoderConfig"
            )
        if encoder.num_levels < 2:
            raise ValueError(f"{cls.__name__}.from_encoder() encoder must have at least two levels")
        if "upsample_mode" in kwargs:
            if "upsample" in kwargs:
                raise ValueError(
                    f"{cls.__name__}.from_encoder() 'upsample' and 'upsample_mode' are mutually exclusive"
                )
            kwargs["upsample"] = UNetUpsampleConfig(kwargs.pop("upsample_mode"))
        residual = encoder.residual if residual is None else residual
        num_channels = reversed_num_channels(encoder.num_channels)
        num_blocks = encoder.num_blocks
        if isinstance(num_blocks, Sequence):
            num_blocks = tuple(reversed(repeat_last(num_blocks, encoder.num_levels))[1:])
        num_layers = encoder.num_layers
        if isinstance(num_layers, Sequence):
            num_layers = tuple(reversed(repeat_last(num_layers, encoder.num_levels))[1:])
        return cls(
            num_channels=num_channels,
            num_blocks=num_blocks,
            num_layers=num_layers,
            conv_layer=encoder.conv_layer,
            residual=residual,
            **kwargs,
        )


@dataclass
class UNetConfig(DataclassConfig):

    encoder: UNetEncoderConfig = UNetEncoderConfig()
    decoder: Optional[UNetDecoderConfig] = None
    output: Optional[UNetOutputConfig] = None

    def __post_init__(self):
        if self.decoder is None:
            self.decoder = UNetDecoderConfig.from_encoder(self.encoder)


def unet_conv_block(
    dimensions: int,
    in_channels: int,
    out_channels: int,
    kernel_size: ScalarOrTuple[int] = 3,
    stride: ScalarOrTuple[int] = 1,
    padding: Optional[ScalarOrTuple[int]] = None,
    padding_mode: Union[PaddingMode, str] = "zeros",
    dilation: ScalarOrTuple[int] = 1,
    groups: int = 1,
    init: str = "default",
    bias: Optional[Union[bool, str]] = None,
    norm: NormArg = None,
    acti: ActivationArg = None,
    order: str = "CNA",
    num_layers: Optional[int] = None,
) -> Module:
    r"""Create U-net block of convolutional layers."""

    if num_layers is None:
        num_layers = 1
    elif num_layers < 1:
        raise ValueError("unet_conv_block() 'num_layers' must be positive")

    def conv_layer(m: int, n: int, s: int, d: int) -> ConvLayer:
        return ConvLayer(
            dimensions=dimensions,
            in_channels=m,
            out_channels=n,
            kernel_size=kernel_size,
            padding=padding,
            padding_mode=padding_mode,
            stride=s,
            dilation=d,
            groups=groups,
            init=init,
            bias=bias,
            norm=norm,
            acti=acti,
            order=order,
        )

    block = Sequential()
    for i in range(num_layers):
        m = in_channels if i == 0 else out_channels
        n = out_channels
        s = stride if i == 0 else 1
        d = dilation if s == 1 else 1
        conv = conv_layer(m, n, s, d)
        block.add_module(f"layer_{i + 1}", conv)
    return block


class UNetEncoder(ReprWithCrossReferences, Module):
    r"""Downsampling path of U-net model."""

    def __init__(
        self,
        dimensions: int,
        in_channels: int = 1,
        config: Optional[UNetEncoderConfig] = None,
        conv_block: Optional[ModuleFactory] = None,
    ):
        super().__init__()

        if config is None:
            config = UNetEncoderConfig()
        elif not isinstance(config, UNetEncoderConfig):
            raise TypeError(f"{type(self).__name__}() 'config' must be UNetEncoderConfig")
        if config.num_levels < 2:
            raise ValueError(
                f"{type(self).__name__} U-net must have at least two spatial resolution levels"
            )

        if conv_block is None:
            conv_block = ResidualUnit if config.residual else unet_conv_block
        elif not isinstance(conv_block, Module) and not callable(conv_block):
            raise TypeError(f"{type(self).__name__}() 'conv_block' must be Module or callable")

        num_channels = config.num_channels
        num_blocks = repeat_last(config.num_blocks, len(num_channels))
        num_layers = repeat_last(config.num_layers, len(num_channels))

        if config.downsample.mode == "none":
            down_stride = (1,) * len(num_channels)
        if isinstance(config.downsample.factor, int):
            down_stride = (1,) + (config.downsample.factor,) * (len(num_channels) - 1)
        else:
            down_stride = repeat_last(config.downsample.factor, len(num_channels))

        stages = ModuleDict()
        channels = in_channels
        for i, (s, b, l, nc) in enumerate(zip(down_stride, num_blocks, num_layers, num_channels)):
            if isinstance(nc, int):
                nc = (nc,) * b
            elif isinstance(nc, Sequence):
                b = len(nc)
            else:
                nc = None
            if not nc:
                raise ValueError(f"{type(self).__name__}() invalid 'num_channels' specification")
            stage = ModuleDict()
            if s > 1 and config.downsample.mode != "conv":
                pool_size = config.downsample.kernel_size or s
                pool_args = dict(kernel_size=pool_size, stride=s)
                if pool_size % 2 == 0:
                    pool_args["padding"] = pool_size // 2 - 1
                else:
                    pool_args["padding"] = pool_size // 2
                if config.downsample.mode == "avg":
                    pool_args["count_include_pad"] = False
                stage["pool"] = PoolLayer(
                    config.downsample.mode, dimensions=dimensions, **pool_args
                )
                s = 1
            blocks = Sequential()
            for j, c in enumerate(nc):
                k = config.conv_layer.kernel_size
                p = config.conv_layer.padding
                if s > 1:
                    k = config.downsample.kernel_size or k
                    if config.downsample.padding is not None:
                        p = config.downsample.padding
                d = 0
                if j == 0:
                    d = config.block_1_dilation
                if i == 0:
                    d = d or config.stage_1_dilation
                d = d or config.conv_layer.dilation
                block = conv_block(
                    dimensions=dimensions,
                    in_channels=channels,
                    out_channels=c,
                    kernel_size=k,
                    stride=s,
                    dilation=d,
                    padding=p,
                    padding_mode=config.conv_layer.padding_mode,
                    init=config.conv_layer.init,
                    bias=config.conv_layer.bias,
                    norm=config.conv_layer.norm,
                    acti=config.conv_layer.acti,
                    order=config.conv_layer.order,
                    num_layers=l,
                )
                blocks.add_module(f"block_{j + 1}", block)
                channels = c
                s = 1
            # mirror full modules names of UNetDecoder, e.g.,
            # encoder.stages.stage_1.blocks.block_1.layer_1.conv
            stage["blocks"] = blocks
            stages[f"stage_{i + 1}"] = stage

        self.config = deepcopy(config)
        self.dimensions = dimensions
        self.in_channels = in_channels
        self.stages = stages

    @property
    def num_channels(self) -> NumChannels:
        return self.config.num_channels

    @property
    def out_channels(self) -> int:
        return last_num_channels(self.config.num_channels)

    def output_size(self, in_size: ScalarOrTuple[int]) -> ScalarOrTuple[int]:
        r"""Calculate output size of last feature map for a tensor of given spatial input size."""
        return self.output_sizes(in_size)[-1]

    def output_sizes(self, in_size: ScalarOrTuple[int]) -> List[ScalarOrTuple[int]]:
        r"""Calculate output sizes of feature maps for a tensor of given spatial input size."""
        size = in_size
        fm_sizes = []
        for stage in self.stages.values():
            assert isinstance(stage, ModuleDict)
            if "pool" in stage:
                size = module_output_size(stage["pool"], size)
            size = module_output_size(stage["blocks"], size)
            fm_sizes.append(size)
        return fm_sizes

    def forward(self, x: Tensor) -> List[Tensor]:
        features = []
        for stage in self.stages.values():
            assert isinstance(stage, ModuleDict)
            if "pool" in stage:
                pool = stage["pool"]
                x = pool(x)
            blocks = stage["blocks"]
            x = blocks(x)
            features.append(x)
        return features


class UNetDecoder(ReprWithCrossReferences, Module):
    r"""Upsampling path of U-net model."""

    def __init__(
        self,
        dimensions: int,
        in_channels: Optional[int] = None,
        config: Optional[UNetDecoderConfig] = None,
        conv_block: Optional[ModuleFactory] = None,
        input_layer: Optional[ModuleFactory] = None,
    ) -> None:
        super().__init__()

        if config is None:
            config = UNetDecoderConfig()
        elif not isinstance(config, UNetDecoderConfig):
            raise TypeError(f"{type(self).__name__}() 'config' must be UNetDecoderConfig")
        if config.num_levels < 2:
            raise ValueError(
                f"{type(self).__name__} U-net must have at least two spatial resolution levels"
            )

        # TODO: What to do when config.num_channels[0] is a sequence of length greater than 1?
        channels = first_num_channels(config.num_channels)
        num_channels = config.num_channels[1:]
        num_blocks = repeat_last(config.num_blocks, len(num_channels))
        num_layers = repeat_last(config.num_layers, len(num_channels))
        scale_factor = repeat_last(config.upsample.factor, len(num_channels))
        upsample_mode = UpsampleMode(config.upsample.mode)
        join_mode = config.join_mode

        if conv_block is None:
            conv_block = ResidualUnit if config.residual else unet_conv_block
        elif not isinstance(conv_block, Module) and not callable(conv_block):
            raise TypeError(f"{type(self).__name__}() 'conv_block' must be Module or callable")

        if input_layer is None:
            input_layer = ConvLayer if in_channels and in_channels != channels else Identity
        elif not isinstance(input_layer, Module) and not callable(input_layer):
            raise TypeError(f"{type(self).__name__}() 'input_layer' must be Module or callable")

        if in_channels is None:
            in_channels = channels
        self.input = input_layer(
            dimensions=dimensions,
            in_channels=in_channels,
            out_channels=channels,
            kernel_size=config.conv_layer.kernel_size,
            dilation=config.conv_layer.dilation,
            padding=config.conv_layer.padding,
            padding_mode=config.conv_layer.padding_mode,
            init=config.conv_layer.init,
            bias=config.conv_layer.bias,
            norm=config.conv_layer.norm,
            acti=config.conv_layer.acti,
            order=config.conv_layer.order,
        )

        stages = ModuleDict()
        for i, (s, b, l, nc) in enumerate(zip(scale_factor, num_blocks, num_layers, num_channels)):
            if isinstance(nc, int):
                nc = (nc,) * b
            elif isinstance(nc, Sequence):
                b = len(nc)
            else:
                nc = None
            if not nc:
                raise ValueError(f"{type(self).__name__}() invalid 'num_channels' specification")
            stage = ModuleDict()
            if upsample_mode is UpsampleMode.INTERPOLATE and config.upsample.kernel_size != 0:
                p = config.upsample.padding
                if p is None:
                    p = config.conv_layer.padding
                pre_conv = ConvLayer(
                    dimensions=dimensions,
                    in_channels=channels,
                    out_channels=nc[0],
                    kernel_size=config.upsample.kernel_size or config.conv_layer.kernel_size,
                    dilation=config.upsample.dilation or config.conv_layer.dilation,
                    padding=p,
                    padding_mode=config.conv_layer.padding_mode,
                    init=config.conv_layer.init,
                    bias=config.conv_layer.bias,
                    norm=config.conv_layer.norm,
                    acti=config.conv_layer.acti,
                    order=config.conv_layer.order,
                )
            else:
                pre_conv = "default"
            if s > 1:
                upsample = Upsample(
                    dimensions=dimensions,
                    in_channels=channels,
                    out_channels=nc[0],
                    scale_factor=s,
                    mode=upsample_mode,
                    align_corners=False,
                    pre_conv=pre_conv,
                    kernel_size=config.upsample.kernel_size,
                    padding_mode=config.conv_layer.padding_mode,
                    bias=True if config.conv_layer.bias is None else config.conv_layer.bias,
                )
                stage["upsample"] = upsample
            stage["join"] = JoinLayer(join_mode, dim=1)
            channels = (2 if join_mode == "cat" else 1) * nc[0]
            blocks = Sequential()
            for j, c in enumerate(nc[1:]):
                block = conv_block(
                    dimensions=dimensions,
                    in_channels=channels,
                    out_channels=c,
                    kernel_size=config.conv_layer.kernel_size,
                    dilation=config.conv_layer.dilation,
                    padding=config.conv_layer.padding,
                    padding_mode=config.conv_layer.padding_mode,
                    init=config.conv_layer.init,
                    bias=config.conv_layer.bias,
                    norm=config.conv_layer.norm,
                    acti=config.conv_layer.acti,
                    order=config.conv_layer.order,
                    num_layers=l,
                )
                blocks.add_module(f"block_{j + 1}", block)
                channels = c
            stage["blocks"] = blocks
            stages[f"stage_{i + 1}"] = stage

        self.config = deepcopy(config)
        self.dimensions = dimensions
        self.in_channels = in_channels
        self.stages = stages

    @classmethod
    def from_encoder(
        cls,
        encoder: Union[UNetEncoder, UNetEncoderConfig],
        residual: Optional[bool] = None,
        **kwargs,
    ) -> UNetDecoder:
        r"""Create U-net decoder given U-net encoder configuration."""
        config = UNetDecoderConfig.from_encoder(encoder, residual=residual, **kwargs)
        return cls(dimensions=encoder.dimensions, config=config)

    @property
    def num_channels(self) -> NumChannels:
        return self.config.num_channels

    @property
    def out_channels(self) -> int:
        return last_num_channels(self.config.num_channels)

    def output_size(self, in_size: ScalarOrTuple[int]) -> ScalarOrTuple[int]:
        r"""Calculate output size for an initial feature map of given spatial input size."""
        return self.output_sizes(in_size)[-1]

    def output_sizes(self, in_size: ScalarOrTuple[int]) -> List[ScalarOrTuple[int]]:
        r"""Calculate output sizes for an initial feature map of given spatial input size."""
        size = module_output_size(self.input, in_size)
        out_sizes = [size]
        for stage in self.stages.values():
            if not isinstance(stage, ModuleDict):
                raise AssertionError(
                    f"{type(self).__name__}.out_sizes() expected stage ModuleDict, got {type(stage)}"
                )
            size = module_output_size(stage["upsample"], size)
            size = module_output_size(stage["blocks"], size)
            out_sizes.append(size)
        return out_sizes

    def forward(self, features: Sequence[Tensor]) -> List[Tensor]:
        if not isinstance(features, Sequence):
            raise TypeError(f"{type(self).__name__}() 'features' must be Sequence")
        features = list(features)
        if len(features) != len(self.stages) + 1:
            raise ValueError(
                f"{type(self).__name__}() 'features' must contain {len(self.stages) + 1} tensors"
            )
        x: Tensor = features.pop()
        x = self.input(x)
        output = [x]
        for stage in self.stages.values():
            if not isinstance(stage, ModuleDict):
                raise AssertionError(
                    f"{type(self).__name__}.forward() expected stage ModuleDict, got {type(stage)}"
                )
            skip = features.pop()
            upsample = stage["upsample"]
            join = stage["join"]
            blocks = stage["blocks"]
            x = upsample(x)
            x = join([x, skip])
            x = blocks(x)
            output.append(x)
        return output


class SequentialUNet(ReprWithCrossReferences, Sequential):
    r"""Sequential U-net architecture.

    The final module of this sequential module either outputs a tuple of feature maps at the
    different resolution levels (``out_channels=None``), the final decoded feature map at the
    highest resolution level (``out_channels == config.decoder.out_channels`` and ``output_layers=None``),
    or a tensor with specified number of ``out_channels`` as produced by a final output layer otherwise.
    Note that additional layers (e.g., a custom output layer or post-output layers) can be added to the
    initialized sequential U-net using ``add_module()``.

    """

    def __init__(
        self,
        dimensions: int,
        in_channels: int,
        out_channels: Optional[int] = None,
        config: Optional[UNetConfig] = None,
        conv_block: Optional[ModuleFactory] = None,
        output_layer: Optional[ModuleFactory] = None,
        bridge_layer: Optional[ModuleFactory] = None,
    ) -> None:
        super().__init__()

        # Network configuration
        if config is None:
            config = UNetConfig()
        elif not isinstance(config, UNetConfig):
            raise TypeError(f"{type(self).__name__}() 'config' must be UNetConfig")
        config = deepcopy(config)
        self.config = config

        # Downsampling path
        self.encoder = UNetEncoder(
            dimensions=dimensions,
            in_channels=in_channels,
            config=config.encoder,
            conv_block=conv_block,
        )

        # Upsamling path with skip connections
        self.decoder = UNetDecoder(
            dimensions=dimensions,
            in_channels=self.encoder.out_channels,
            config=config.decoder,
            conv_block=conv_block,
            input_layer=bridge_layer,
        )

        # Optional output layer
        channels = self.decoder.out_channels
        if not out_channels and config.output is not None:
            out_channels = config.output.channels
        if output_layer is None:
            if out_channels == channels and config.output is None:
                self.add_module("output", GetItem(-1))
            elif out_channels:
                output_layer = ConvLayer
            else:
                out_channels = self.decoder.num_channels
        if output_layer is not None:
            out_channels = out_channels or in_channels
            if config.output is None:
                config.output = UNetOutputConfig()
            output = output_layer(
                dimensions=dimensions,
                in_channels=channels,
                out_channels=out_channels,
                kernel_size=config.output.kernel_size,
                padding=config.output.padding,
                padding_mode=config.output.padding_mode,
                dilation=config.output.dilation,
                init=config.output.init,
                bias=config.output.bias,
                norm=config.output.norm,
                acti=config.output.acti,
                order=config.output.order,
            )
            output = [("input", GetItem(-1)), ("layer", output)]
            output = Sequential(OrderedDict(output))
            self.add_module("output", output)
        self.out_channels: Union[int, Sequence[int]] = out_channels

    @property
    def dimensions(self) -> int:
        return self.encoder.dimensions

    @property
    def in_channels(self) -> int:
        return self.encoder.in_channels

    @property
    def num_channels(self) -> NumChannels:
        return self.decoder.num_channels

    @property
    def num_levels(self) -> int:
        return len(self.num_channels)

    def output_size(self, in_size: ScalarOrTuple[int]) -> ScalarOrTuple[int]:
        r"""Calculate spatial output size given an input tensor with specified spatial size."""
        # ATTENTION: module_output_size(self, in_size) would cause an infinite recursion!
        size = in_size
        for module in self:
            size = module_output_size(module, size)
        return size


class UNet(ReprWithCrossReferences, Module):
    r"""U-net with optionally multiple output layers."""

    def __init__(
        self,
        dimensions: int,
        in_channels: int,
        out_channels: Optional[int] = None,
        output_modules: Optional[Mapping[str, Module]] = None,
        output_indices: Optional[Union[Mapping[str, int], int]] = None,
        config: Optional[UNetConfig] = None,
        conv_block: Optional[ModuleFactory] = None,
        bridge_layer: Optional[ModuleFactory] = None,
        output_layer: Optional[ModuleFactory] = None,
        output_name: str = "output",
    ) -> None:
        super().__init__()

        if output_modules is None:
            output_modules = {}
        if not isinstance(output_modules, Mapping):
            raise TypeError(f"{type(self).__name__}() 'output_modules' must be Mapping")

        # Network configuration
        if config is None:
            config = UNetConfig()
        elif not isinstance(config, UNetConfig):
            raise TypeError(f"{type(self).__name__}() 'config' must be UNetConfig")
        config = deepcopy(config)
        self.config = config

        # Downsampling path
        self.encoder = UNetEncoder(
            dimensions=dimensions,
            in_channels=in_channels,
            config=config.encoder,
            conv_block=conv_block,
        )

        # Upsamling path with skip connections
        self.decoder = UNetDecoder(
            dimensions=dimensions,
            in_channels=self.encoder.out_channels,
            config=config.decoder,
            conv_block=conv_block,
            input_layer=bridge_layer,
        )

        # Optional output layer
        channels = self.decoder.out_channels
        self.output_modules = ModuleDict()
        if not out_channels and config.output is not None:
            out_channels = config.output.channels
        if output_layer is None:
            if out_channels == channels and config.output is None:
                self.output_modules[output_name] = GetItem(-1)
            elif out_channels:
                output_layer = ConvLayer
            elif not output_modules:
                out_channels = self.decoder.num_channels
        if output_layer is not None:
            out_channels = out_channels or in_channels
            if config.output is None:
                config.output = UNetOutputConfig()
            output = output_layer(
                dimensions=dimensions,
                in_channels=channels,
                out_channels=out_channels,
                kernel_size=config.output.kernel_size,
                padding=config.output.padding,
                padding_mode=config.output.padding_mode,
                dilation=config.output.dilation,
                init=config.output.init,
                bias=config.output.bias,
                norm=config.output.norm,
                acti=config.output.acti,
                order=config.output.order,
            )
            output = [("input", GetItem(-1)), ("layer", output)]
            output = Sequential(OrderedDict(output))
            self.output_modules[output_name] = output
        self.out_channels: Union[int, Sequence[int], None] = out_channels

        # Additional output layers
        if output_indices is None:
            output_indices = {}
        elif isinstance(output_indices, int):
            output_indices = {name: output_indices for name in output_modules}
        for name, output in output_modules.items():
            output_index = output_indices.get(name)
            if output_index is not None:
                if not isinstance(output_index, int):
                    raise TypeError(f"{type(self).__name__}() 'output_indices' must be int")
                output = [("input", GetItem(output_index)), ("layer", output)]
                output = Sequential(OrderedDict(output))
            self.output_modules[name] = output

    @property
    def dimensions(self) -> int:
        return self.encoder.dimensions

    @property
    def in_channels(self) -> int:
        return self.encoder.in_channels

    @property
    def num_channels(self) -> NumChannels:
        return self.decoder.num_channels

    @property
    def num_levels(self) -> int:
        return len(self.num_channels)

    @property
    def num_output_layers(self) -> int:
        return len(self.output_modules)

    def output_names(self) -> Iterable[str]:
        return self.output_modules.keys()

    def output_is_dict(self) -> bool:
        r"""Whether model output is dictionary of output tensors."""
        return not (self.output_is_tensor() or self.output_is_tuple())

    def output_is_tensor(self) -> bool:
        r"""Whether model output is a single output tensor."""
        return len(self.output_modules) == 1 and bool(self.out_channels)

    def output_is_tuple(self) -> bool:
        r"""Whether model output is tuple of decoded feature maps."""
        return not self.output_modules

    def output_size(self, in_size: ScalarOrTuple[int]) -> ScalarOrTuple[int]:
        out_sizes = self.output_sizes(in_size)
        if self.output_is_tensor():
            assert len(out_sizes) == 1
            return out_sizes[0]
        if self.output_is_tuple():
            return out_sizes[-1]
        assert isinstance(out_sizes, dict) and len(out_sizes) > 0
        out_size = None
        for size in out_sizes.values():
            if out_size is None:
                out_size = size
            elif out_size != size:
                raise RuntimeError(
                    f"{type(self).__name__}.output_size() is ambiguous, use output_sizes() instead"
                )
        assert out_size is not None
        return out_size

    def output_sizes(
        self, in_size: ScalarOrTuple[int]
    ) -> Union[Dict[ScalarOrTuple[int]], List[ScalarOrTuple[int]]]:
        enc_out_size = self.encoder.output_size(in_size)
        dec_out_sizes = self.decoder.output_sizes(enc_out_size)
        if self.output_is_tensor():
            return dec_out_sizes[-1:]
        if self.output_is_tuple():
            return dec_out_sizes
        out_sizes = {}
        for name, module in self.output_modules.items():
            out_sizes[name] = module_output_size(module, dec_out_sizes)
        return out_sizes

    def forward(self, x: Tensor) -> Union[Tensor, Dict[str, Tensor], List[Tensor]]:
        features = self.encoder(x)
        features = self.decoder(features)
        outputs = {}
        for name, output in self.output_modules.items():
            outputs[name] = output(features)
        if not outputs:
            return features
        if len(outputs) == 1 and self.out_channels:
            return next(iter(outputs.values()))
        return outputs
