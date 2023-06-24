r"""Image and spatial transformer network."""

from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Union

import torch
from torch import Size, Tensor
from torch.nn import Module

from deepali.core import DataclassConfig, Grid, PaddingMode, Sampling
from deepali.core import functional as U
from deepali.spatial import GenericSpatialTransform, ImageTransformer, ParametricTransform

from .itn import ImageTransformerConfig, ImageTransformerNetwork
from .stn import SpatialTransformerConfig, SpatialTransformerNetwork


@dataclass
class InputConfig(DataclassConfig):
    size: Sequence[int]
    channels: int = 1

    @property
    def spatial_dims(self) -> int:
        return len(self.size)

    @property
    def shape(self) -> Size:
        return Size(reversed(self.size))


@dataclass
class ImageAndSpatialTransformerConfig(DataclassConfig):
    input: InputConfig
    itn: Optional[Union[str, ImageTransformerConfig]] = "miccai2019"
    stn: SpatialTransformerConfig = SpatialTransformerConfig()


class ImageAndSpatialTransformerNetwork(Module):
    r"""Image and spatial transformer network."""

    def __init__(
        self,
        stn: SpatialTransformerNetwork,
        itn: Optional[ImageTransformerNetwork] = None,
        sampling: Union[Sampling, str] = Sampling.LINEAR,
        padding: Union[PaddingMode, str, float] = PaddingMode.BORDER,
    ) -> None:
        super().__init__()
        if stn.in_channels < 2 or stn.in_channels % 2 != 0:
            raise ValueError(
                f"{type(self).__name__}() 'stn.in_channels' must be positive even number"
            )
        self.itn = itn
        self.stn = stn
        grid = Grid(size=stn.in_size)
        transform = GenericSpatialTransform(grid, params=None, config=stn.config)
        self.warp = ImageTransformer(transform, sampling=sampling, padding=padding)

    @property
    def config(self) -> ImageAndSpatialTransformerConfig:
        stn: SpatialTransformerNetwork = self.stn
        itn: Optional[ImageTransformerNetwork] = self.itn
        return ImageAndSpatialTransformerConfig(
            input=InputConfig(
                size=tuple(stn.in_size),
                channels=stn.in_channels // 2,
            ),
            itn=None if itn is None else itn.config,
            stn=stn.config,
        )

    @property
    def transform(self) -> GenericSpatialTransform:
        r"""Reference to spatial coordinates transformation."""
        return self.warp.transform

    def forward(
        self, source_img: Tensor, target_img: Tensor, apply: bool = True
    ) -> Dict[str, Tensor]:
        output: Dict[str, Tensor] = {}
        # Image transformation
        itn: Optional[Module] = self.itn
        source_soi = source_img
        target_soi = target_img
        if itn is not None:
            with torch.set_grad_enabled(itn.training):
                source_soi = itn(source_img)
                target_soi = itn(target_img)
            output["source_soi"] = source_soi
            output["target_soi"] = target_soi
        else:
            output["source_soi"] = source_img
            output["target_soi"] = target_img
        # Spatial transformation
        stn: Module = self.stn
        stn_input = torch.cat([source_soi, target_soi], dim=1)
        params: Dict[str, Tensor] = stn(stn_input)
        vfield_params: Optional[Tensor] = params.get("vfield")
        nonrigid_transform: Optional[ParametricTransform] = self.transform.get("nonrigid")
        if vfield_params is None:
            assert nonrigid_transform is None
        else:
            assert nonrigid_transform is not None
            vfield_shape = nonrigid_transform.data_shape[1:]
            vfield_params = U.grid_reshape(vfield_params, vfield_shape, align_corners=False)
            params["vfield"] = vfield_params
        self.transform.params = params
        if apply:
            output["warped_img"] = self.warp(source_img)
        return output


def create_istn(config: ImageAndSpatialTransformerConfig) -> ImageAndSpatialTransformerNetwork:
    itn = ImageTransformerNetwork(config.input.spatial_dims, config.input.channels, config.itn)
    itn_output_size = itn.output_size(config.input.size)
    stn = SpatialTransformerNetwork(2 * itn.out_channels, itn_output_size, config.stn)
    return ImageAndSpatialTransformerNetwork(itn=itn, stn=stn)
