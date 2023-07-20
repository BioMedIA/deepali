from typing import Tuple

import pytest

import torch
from torch import Tensor

from deepali.core.typing import is_namedtuple
from deepali.networks.layers import convolution
from deepali.networks.unet import UNet, UNetConfig, last_num_channels


@pytest.fixture(scope="function")
def input_tensor(request) -> Tensor:
    if request.param == 2:
        return torch.rand((1, 1, 128, 128))
    elif request.param == 3:
        return torch.rand((1, 1, 64, 128, 128))
    else:
        raise ValueError("input_tensor() 'request.param' must be 2 or 3")


@pytest.mark.parametrize("input_tensor", [2, 3], indirect=True)
def test_unet_without_output_layer(input_tensor: Tensor) -> None:
    spatial_dims = input_tensor.ndim - 2
    in_channels = input_tensor.shape[1]

    model = UNet(spatial_dims=spatial_dims, in_channels=in_channels)
    model.eval()
    print(model)

    assert model.output_is_tensor() is False
    assert model.output_is_dict() is False
    assert model.output_is_tuple() is True

    output: Tuple[Tensor, ...] = model(input_tensor)
    assert isinstance(output, tuple)
    assert len(output) == 4
    assert isinstance(output[0], Tensor)
    assert isinstance(output[1], Tensor)
    assert isinstance(output[2], Tensor)
    assert isinstance(output[3], Tensor)

    for i, output_tensor in enumerate(output):
        ds = 2 ** (model.num_levels - 1 - i)
        channels = last_num_channels(model.num_channels[i])
        shape = (input_tensor.shape[0], channels) + tuple(n // ds for n in input_tensor.shape[2:])
        assert output_tensor.shape == shape


@pytest.mark.parametrize("input_tensor", [2, 3], indirect=True)
def test_unet_with_single_output_layer(input_tensor: Tensor) -> None:
    spatial_dims = input_tensor.ndim - 2
    in_channels = input_tensor.shape[1]
    out_channels = in_channels

    model = UNet(spatial_dims=spatial_dims, in_channels=in_channels, out_channels=out_channels)
    print(model)

    assert model.output_is_tensor() is True
    assert model.output_is_dict() is False
    assert model.output_is_tuple() is False

    output_tensor = model(input_tensor)
    assert isinstance(output_tensor, Tensor)
    assert output_tensor.shape[0] == input_tensor.shape[0]
    assert output_tensor.shape[1] == out_channels
    assert output_tensor.shape[2:] == input_tensor.shape[2:]


@pytest.mark.parametrize("input_tensor", [2, 3], indirect=True)
def test_unet_with_multiple_output_layers(input_tensor: Tensor) -> None:
    spatial_dims = input_tensor.ndim - 2
    config = UNetConfig()

    output_modules = {}
    output_indices = {}
    for i, channels in enumerate(config.decoder.num_channels):
        name = f"output_{i + 1}"
        output = convolution(
            spatial_dims=spatial_dims,
            in_channels=channels,
            out_channels=1,
            kernel_size=1,
            bias=False,
        )
        output_modules[name] = output
        output_indices[name] = i

    model = UNet(
        spatial_dims=spatial_dims,
        in_channels=1,
        output_modules=output_modules,
        output_indices=output_indices,
    )
    model.eval()
    print(model)

    assert model.output_is_tensor() is False
    assert model.output_is_dict() is True
    assert model.output_is_tuple() is False

    output = model(input_tensor)
    assert is_namedtuple(output)
    assert len(output) == len(output_modules)
    assert set(getattr(output, "_fields", [])) == set(output_modules.keys())
    assert set(output.keys()) == set(output_modules.keys())
    assert output["output_1"] is output.output_1
    assert all(isinstance(x, Tensor) for x in output)
