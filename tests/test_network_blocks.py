r"""Test basic network blocks."""

import pytest

import torch
from torch import Tensor, nn

from deepali.networks.blocks import DenseBlock
from deepali.networks.blocks import ResidualUnit
from deepali.networks.blocks import SkipConnection
from deepali.networks.layers import ConvLayer, JoinLayer, LambdaLayer


def test_dense_block() -> None:
    r"""Test block with dense skip connections."""

    x = torch.tensor([[1.0, -2.0, 3.0, 4.0, -5.0]])

    scale = LambdaLayer(lambda a: 2 * a)
    square = LambdaLayer(lambda a: a.square())
    clamp = LambdaLayer(lambda a: a.clamp(min=0))

    block = DenseBlock(scale, square, clamp)
    assert isinstance(block.layers, nn.ModuleDict)
    assert "0" in block.layers
    assert "1" in block.layers
    assert "2" in block.layers
    assert block.layers["0"] is scale
    assert block.layers["1"] is square
    assert block.layers["2"] is clamp

    block = DenseBlock({"scale": scale, "square": square, "clamp": clamp}, join="concat")
    assert isinstance(block.layers, nn.ModuleDict)
    assert "scale" in block.layers
    assert "square" in block.layers
    assert "clamp" in block.layers
    assert block.layers["scale"] is scale
    assert block.layers["square"] is square
    assert block.layers["clamp"] is clamp

    y = block(x)
    assert isinstance(y, Tensor)

    a = scale(x)
    b = square(torch.cat([x, a], dim=1))
    c = clamp(torch.cat([x, a, b], dim=1))
    assert isinstance(c, Tensor)
    assert y.shape == c.shape
    assert y.allclose(c)

    block = DenseBlock({"scale": scale, "square": square, "clamp": clamp}, join="add")

    y = block(x)
    assert isinstance(y, Tensor)
    assert y.shape == x.shape

    a = scale(x)
    b = square(x + a)
    c = clamp(x + a + b)
    assert isinstance(c, Tensor)
    assert y.shape == c.shape
    assert y.allclose(c)


def test_residual_unit() -> None:
    r"""Test convoluational residual block."""

    with pytest.raises(TypeError):
        ResidualUnit(2, 1, 1, num_layers=1.0)
    with pytest.raises(ValueError):
        ResidualUnit(2, 1, 1, num_layers=0)
    with pytest.raises(ValueError):
        ResidualUnit(2, in_channels=64, out_channels=64, num_channels=32, num_layers=2)

    x = torch.tensor([[[1.0, -2.0, 3.0, 4.0, -5.0]]])

    block = ResidualUnit(spatial_dims=1, in_channels=1, order="nac")
    assert isinstance(block, nn.Module)
    assert isinstance(block, SkipConnection)
    assert isinstance(block.func, nn.Sequential)
    assert len(block.func) == 2
    assert isinstance(block.func[0], ConvLayer)
    assert isinstance(block.func[1], ConvLayer)
    assert isinstance(block.skip, nn.Identity)

    y = block.join([x, x])
    assert isinstance(y, Tensor)
    assert y.eq(x + x).all()

    block = ResidualUnit(spatial_dims=1, in_channels=1, out_channels=2)
    assert block.spatial_dims == 1
    assert block.in_channels == 1
    assert block.out_channels == 2
    assert isinstance(block.skip, nn.Conv1d)
    assert block.skip.out_channels == 2

    block = ResidualUnit(
        spatial_dims=1,
        in_channels=1,
        out_channels=2,
        acti="relu",
        norm="group",
        num_layers=1,
        order="nac",
    )
    assert len(block.func) == 1
    assert isinstance(block.func[0].conv, nn.Conv1d)
    assert isinstance(block.func[0].norm, nn.GroupNorm)
    assert isinstance(block.func[0].acti, nn.ReLU)
    assert block.func[0].order == "NAC"
    assert not isinstance(block.join, nn.Sequential)

    block = ResidualUnit(
        spatial_dims=1,
        in_channels=1,
        out_channels=2,
        acti="relu",
        norm="group",
        num_layers=3,
        order="cna",
    )
    assert len(block.func) == 3
    assert isinstance(block.func[0].conv, nn.Conv1d)
    assert isinstance(block.func[0].norm, nn.GroupNorm)
    assert isinstance(block.func[0].acti, nn.ReLU)
    assert isinstance(block.func[1].conv, nn.Conv1d)
    assert isinstance(block.func[1].norm, nn.GroupNorm)
    assert isinstance(block.func[1].acti, nn.ReLU)
    assert isinstance(block.func[2].conv, nn.Conv1d)
    assert isinstance(block.func[2].norm, nn.GroupNorm)
    assert block.func[2].acti is None
    assert block.func[0].order == "CNA"
    assert isinstance(block.join, nn.Sequential)
    assert isinstance(block.join[0], JoinLayer)
    assert isinstance(block.join[1], nn.ReLU)

    y = block(x)
    assert isinstance(y, Tensor)
    assert y.shape == (x.shape[0], 2) + x.shape[2:]


def test_skip_connection() -> None:
    r"""Test skip connection block."""

    x = torch.tensor([[1.0, 2.0, 3.0]])
    module = LambdaLayer(lambda a: a)

    # skip: identity, join: cat
    block = SkipConnection(module)
    assert isinstance(block, nn.Module)
    assert isinstance(block.skip, nn.Identity)
    assert block.func is module

    y = block.join([x, x])
    assert isinstance(y, Tensor)
    assert y.shape == (x.shape[0], 2 * x.shape[1])

    y = block(x)
    assert isinstance(y, Tensor)
    assert y.shape == (x.shape[0], 2 * x.shape[1])
    assert y.eq(torch.cat([x, x], dim=1)).all()

    # skip: identity, join: add
    block = SkipConnection(module, name="residual", join="add")
    assert isinstance(block, nn.Module)
    assert isinstance(block.skip, nn.Identity)
    assert block.func is module

    y = block(x)
    assert isinstance(y, Tensor)
    assert y.shape == x.shape
    assert y.eq(x + x).all()

    # skip: identity, join: mul
    block = SkipConnection(module, skip="identity", join="mul")
    assert isinstance(block, nn.Module)
    assert isinstance(block.skip, nn.Identity)
    assert block.func is module

    y = block(x)
    assert isinstance(y, Tensor)
    assert y.shape == x.shape
    assert y.eq(x * x).all()

    # skip: custom, join: cat
    skip = LambdaLayer(lambda b: 2 * b)
    block = SkipConnection(module, skip=skip, join="cat")
    assert block.skip is skip
    assert block.func is module

    y = block(x)
    assert isinstance(y, Tensor)
    assert y.shape == (x.shape[0], 2 * x.shape[1])
    assert y.eq(torch.cat([x, 2 * x], dim=1)).all()
