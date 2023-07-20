r"""Test basic network layers."""

import pytest

import torch
from torch import Tensor, nn

from deepali.core import PaddingMode
from deepali.networks.layers import Activation, activation
from deepali.networks.layers import Conv2d, Conv3d, ConvLayer, convolution, conv_module
from deepali.networks.layers import JoinLayer, join_func
from deepali.networks.layers import LambdaLayer
from deepali.networks.layers import NormLayer, normalization, norm_layer


def test_activation() -> None:
    r"""Test construction of non-linear activations."""

    act = activation("relu")
    assert isinstance(act, nn.ReLU)
    act = activation("lrelu", 0.1)
    assert isinstance(act, nn.LeakyReLU)
    assert act.negative_slope == 0.1
    assert act.inplace is False
    act = activation(("LeakyReLU", {"negative_slope": 0.05}), inplace=True)
    assert isinstance(act, nn.LeakyReLU)
    assert act.negative_slope == 0.05
    assert act.inplace is True
    act = activation("softmax", dim=2)
    assert isinstance(act, nn.Softmax)
    assert act.dim == 2

    def act_func(x: Tensor) -> Tensor:
        return x.sigmoid()

    act = activation(act_func)
    assert isinstance(act, Activation)
    assert isinstance(act, nn.Module)
    assert act.func is act_func

    x = torch.tensor(2.0)
    y = act(x)
    assert isinstance(y, Tensor)
    assert y.allclose(act_func(x))

    act = Activation("elu")
    assert isinstance(act.func, nn.ELU)


def test_convolution() -> None:
    r"""Test construction of simple convolutional layer."""

    conv = convolution(1, 3, 16, 1)
    assert isinstance(conv, nn.Conv1d)
    assert conv.weight.shape == (16, 3, 1)
    assert isinstance(conv.bias, Tensor)

    conv = convolution(2, 3, 32, kernel_size=1)
    assert isinstance(conv, nn.Conv2d)
    assert isinstance(conv, Conv2d)
    assert conv.weight.shape == (32, 3, 1, 1)
    assert isinstance(conv.bias, Tensor)

    conv = convolution(3, 3, 8, kernel_size=1)
    assert isinstance(conv, nn.Conv3d)
    assert isinstance(conv, Conv3d)
    assert conv.weight.shape == (8, 3, 1, 1, 1)
    assert isinstance(conv.bias, Tensor)

    conv = convolution(3, 3, 8, 5, bias=False)
    assert isinstance(conv, nn.Conv3d)
    assert conv.weight.shape == (8, 3, 5, 5, 5)
    assert conv.bias is None
    assert conv.output_padding == (0, 0, 0)

    conv = convolution(3, 3, 8, kernel_size=(5, 3, 1), output_padding=0)
    assert isinstance(conv, nn.Conv3d)
    assert conv.weight.shape == (8, 3, 5, 3, 1)
    assert isinstance(conv.bias, Tensor)
    assert conv.output_padding == (0, 0, 0)

    conv = conv_module(3, 3, 16, kernel_size=3, stride=2, output_padding=1, transposed=True)
    assert isinstance(conv, nn.ConvTranspose3d)
    assert conv.transposed is True
    assert conv.kernel_size == (3, 3, 3)
    assert conv.stride == (2, 2, 2)
    assert conv.output_padding == (1, 1, 1)

    conv = conv_module(
        spatial_dims=3,
        in_channels=6,
        out_channels=16,
        kernel_size=3,
        stride=2,
        dilation=4,
        padding=1,
        padding_mode=PaddingMode.REFLECT,
        groups=2,
        init="xavier",
        bias="zeros",
    )
    assert isinstance(conv, Conv3d)
    assert conv.weight_init == "xavier"
    assert conv.bias_init == "zeros"
    assert conv.in_channels == 6
    assert conv.out_channels == 16
    assert conv.groups == 2
    assert conv.padding_mode == "reflect"
    assert conv.kernel_size == (3, 3, 3)
    assert conv.stride == (2, 2, 2)
    assert conv.dilation == (4, 4, 4)
    assert conv.padding == (1, 1, 1)
    assert conv.output_padding == (0, 0, 0)
    assert conv.transposed is False


def test_conv_layer() -> None:
    r"""Test convolutional layer with optional normalization and/or activation."""
    layer = ConvLayer(2, 1, 8, 3)
    assert isinstance(layer, nn.Module)
    assert hasattr(layer, "acti")
    assert hasattr(layer, "norm")
    assert hasattr(layer, "conv")
    assert layer.acti is None
    assert layer.norm is None
    assert isinstance(layer.conv, nn.Conv2d)

    layer = ConvLayer(
        1, in_channels=1, out_channels=16, kernel_size=3, acti=("lrelu", {"negative_slope": 0.1})
    )
    assert isinstance(layer.acti, nn.LeakyReLU)
    assert layer.norm is None
    assert isinstance(layer.conv, nn.Conv1d)
    assert layer.order == "CNA"
    assert layer.acti.negative_slope == 0.1

    layer = ConvLayer(2, in_channels=1, out_channels=16, kernel_size=3, acti="relu")
    assert isinstance(layer.acti, nn.ReLU)
    assert layer.norm is None
    assert isinstance(layer.conv, nn.Conv2d)
    assert layer.order == "CNA"

    layer = ConvLayer(3, in_channels=1, out_channels=16, kernel_size=3, acti="prelu", norm="batch")
    assert isinstance(layer.acti, nn.PReLU)
    assert isinstance(layer.norm, nn.BatchNorm3d)
    assert isinstance(layer.conv, nn.Conv3d)
    assert layer.order == "CNA"

    layer = ConvLayer(
        3, in_channels=1, out_channels=16, kernel_size=3, acti="relu", norm="instance", order="nac"
    )
    assert isinstance(layer.acti, nn.ReLU)
    assert isinstance(layer.norm, nn.InstanceNorm3d)
    assert isinstance(layer.conv, nn.Conv3d)
    assert layer.order == "NAC"

    x = torch.randn((2, 1, 7, 9, 11))
    y = layer(x)
    z = layer.conv(layer.acti(layer.norm(x)))
    assert isinstance(y, Tensor)
    assert isinstance(z, Tensor)
    assert y.allclose(z)

    layer = ConvLayer(
        3, in_channels=1, out_channels=16, kernel_size=3, acti="relu", norm="instance", order="CNA"
    )
    assert isinstance(layer.acti, nn.ReLU)
    assert isinstance(layer.norm, nn.InstanceNorm3d)
    assert isinstance(layer.conv, nn.Conv3d)
    assert layer.order == "CNA"

    y = layer(x)
    assert isinstance(y, Tensor)
    assert not y.allclose(z)
    z = layer.acti(layer.norm(layer.conv(x)))
    assert isinstance(z, Tensor)
    assert y.allclose(z)


def test_join_layer() -> None:
    r"""Test layer which joins features of one or more input tensors."""

    with pytest.raises(ValueError):
        join_func("foo")
    with pytest.raises(ValueError):
        JoinLayer("bar")

    x = torch.tensor([[1.0, 1.5, 2.0]])
    y = torch.tensor([[0.4, 2.5, -0.1]])

    func = join_func("add")
    z = func([x, y])
    assert isinstance(z, Tensor)
    assert z.allclose(x + y)

    func = join_func("mul")
    z = func([x, y])
    assert isinstance(z, Tensor)
    assert z.allclose(x * y)

    func = join_func("cat")
    z = func([x, y])
    assert isinstance(z, Tensor)
    assert z.allclose(torch.cat([x, y], dim=1))

    func = join_func("concat")
    z = func([x, y])
    assert isinstance(z, Tensor)
    assert z.allclose(torch.cat([x, y], dim=1))

    join = JoinLayer("add", dim=0)
    z = join([x, y])
    assert isinstance(z, Tensor)
    assert z.allclose(x + y)

    join = JoinLayer("mul")
    z = join([x, y])
    assert isinstance(z, Tensor)
    assert z.allclose(x * y)

    join = JoinLayer("cat", dim=0)
    z = join([x, y])
    assert isinstance(z, Tensor)
    assert z.allclose(torch.cat([x, y], dim=0))

    join = JoinLayer("concat", dim=1)
    z = join([x, y])
    assert isinstance(z, Tensor)
    assert z.allclose(torch.cat([x, y], dim=1))


def test_lambda_layer() -> None:
    def square_func(x: Tensor) -> Tensor:
        return x.square()

    square_layer = LambdaLayer(square_func)
    assert isinstance(square_layer, nn.Module)
    assert square_layer.func is square_func

    x = torch.tensor([[2.0, 0.5]])
    y = square_layer(x)
    assert isinstance(y, Tensor)
    assert y.allclose(torch.tensor([[4.0, 0.25]]))


def test_norm_layer() -> None:
    r"""Test construction of normalization layers."""

    # Batch normalization
    with pytest.raises(ValueError):
        norm_layer("batch", spatial_dims=3)
    with pytest.raises(ValueError):
        normalization("batch", spatial_dims=1)
    with pytest.raises(ValueError):
        normalization("batch", num_features=32)
    norm = norm_layer("batch", spatial_dims=1, num_features=16)
    assert isinstance(norm, nn.BatchNorm1d)
    assert norm.affine is True
    assert norm.num_features == 16
    assert norm.bias.shape == (16,)
    norm = normalization("batch", spatial_dims=2, num_features=32)
    assert isinstance(norm, nn.BatchNorm2d)
    assert norm.affine is True
    assert norm.num_features == 32
    assert norm.bias.shape == (32,)
    norm = norm_layer("BatchNorm", spatial_dims=3, num_features=64)
    assert isinstance(norm, nn.BatchNorm3d)
    assert norm.affine is True
    assert norm.num_features == 64
    assert norm.bias.shape == (64,)

    # Group normalization
    with pytest.raises(ValueError):
        norm_layer("group", 3)
    with pytest.raises(ValueError):
        norm_layer("group", spatial_dims=1)
    norm = norm_layer("group", num_features=32)
    assert isinstance(norm, nn.GroupNorm)
    assert norm.num_groups == 1
    assert norm.num_channels == 32
    assert norm.affine is True
    norm = norm_layer("GroupNorm", num_channels=16)
    assert isinstance(norm, nn.GroupNorm)
    assert norm.num_groups == 1
    assert norm.num_channels == 16
    assert norm.affine is True
    norm = normalization("group", num_groups=8, num_channels=64, affine=False)
    assert isinstance(norm, nn.GroupNorm)
    assert norm.num_groups == 8
    assert norm.num_channels == 64
    assert norm.affine is False

    # Layer normalization
    with pytest.raises(ValueError):
        norm_layer("layer")
    with pytest.raises(ValueError):
        norm_layer("layer", spatial_dims=1)
    norm = normalization("layer", num_features=32)
    assert isinstance(norm, nn.GroupNorm)
    assert norm.num_groups == 1
    assert norm.num_channels == 32
    assert norm.affine is True
    norm = norm_layer("LayerNorm", spatial_dims=1, num_features=64, affine=False)
    assert isinstance(norm, nn.GroupNorm)
    assert norm.num_groups == 1
    assert norm.num_channels == 64
    assert norm.affine is False

    # Instance normalization
    with pytest.raises(ValueError):
        norm_layer("instance", 2)
    with pytest.raises(ValueError):
        norm_layer("instance", spatial_dims=1)
    with pytest.raises(ValueError):
        normalization("instance", num_features=32)
    norm = norm_layer("instance", spatial_dims=1, num_features=32)
    assert isinstance(norm, nn.InstanceNorm1d)
    norm = norm_layer("instance", spatial_dims=2, num_features=32)
    assert isinstance(norm, nn.InstanceNorm2d)
    norm = normalization("instance", spatial_dims=3, num_features=32)
    assert isinstance(norm, nn.InstanceNorm3d)
    assert norm.num_features == 32
    assert norm.affine is False
    norm = normalization("InstanceNorm", spatial_dims=3, num_features=32, affine=True)
    assert isinstance(norm, nn.InstanceNorm3d)
    assert norm.num_features == 32
    assert norm.affine is True
    norm = NormLayer("instance", spatial_dims=3, num_features=64)
    assert isinstance(norm.func, nn.InstanceNorm3d)
    assert norm.func.num_features == 64
    assert norm.func.affine is False

    # Custom normalization
    def norm_func(x: Tensor) -> Tensor:
        return x

    norm = normalization(norm_func)
    assert isinstance(norm, nn.Module)
    assert isinstance(norm, NormLayer)
    assert norm.func is norm_func

    norm = NormLayer(norm_func)
    assert norm.func is norm_func
