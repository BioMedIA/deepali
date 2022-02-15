# %%
import math
from timeit import default_timer as timer
from typing import Callable, Tuple

import torch
import torch.nn.functional as F
from torch import Size, Tensor

from deepali.core import functional as U
from deepali.core import bspline as B
from deepali.core.enum import SpatialDim


# %% Vector field control point coefficients
# device = torch.device("cuda:0")
device = torch.device("cpu")

in_size = Size((21,))  # (X, ...)
stride = (5,) * len(in_size)

cp_size = B.cubic_bspline_control_point_grid_size(in_size, stride)
cp_data = torch.arange(len(in_size) * cp_size.numel(), dtype=torch.float32, device=device)
cp_data = cp_data.reshape(1, len(cp_size), *tuple(reversed(cp_size)))


# %% Reference implementation based on MIRTK C++ code


def compute_bspline_indices_and_weights_1d(x: float, degree: int = 3) -> Tuple[Tensor, Tensor]:
    if degree & 1:
        i = int(math.floor(x)) - degree // 2
    else:
        i = int(math.floor(x + 0.5)) - degree // 2
    i = torch.arange(i, i + degree + 1, 1, dtype=torch.int)
    wx = torch.empty(4, dtype=torch.float)
    if degree == 3:
        w = x - i[1]
        wx[3] = (1 / 6) * w * w * w
        wx[0] = (1 / 6) + (1 / 2) * w * (w - 1) - wx[3]
        wx[2] = w + wx[0] - 2 * wx[3]
        wx[1] = 1 - wx[0] - wx[2] - wx[3]
    else:
        raise NotImplementedError(f"compute_bspline_indices_and_weights_1d() for degree={degree}")
    return i, wx


def compute_bspline_indices_and_weights_2d(
    x: float, y: float, degree: int = 3
) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    if degree & 1:
        i = int(math.floor(x)) - degree // 2
        j = int(math.floor(y)) - degree // 2
    else:
        i = int(math.floor(x + 0.5)) - degree // 2
        j = int(math.floor(y + 0.5)) - degree // 2
    i = torch.arange(i, i + degree + 1, 1, dtype=torch.int)
    j = torch.arange(j, j + degree + 1, 1, dtype=torch.int)
    wx = torch.empty(4, dtype=torch.float)
    wy = torch.empty(4, dtype=torch.float)
    if degree == 3:
        # x
        w = x - i[1]
        wx[3] = (1 / 6) * w * w * w
        wx[0] = (1 / 6) + (1 / 2) * w * (w - 1) - wx[3]
        wx[2] = w + wx[0] - 2 * wx[3]
        wx[1] = 1 - wx[0] - wx[2] - wx[3]
        # y
        w = y - j[1]
        wy[3] = (1 / 6) * w * w * w
        wy[0] = (1 / 6) + (1 / 2) * w * (w - 1) - wy[3]
        wy[2] = w + wy[0] - 2 * wy[3]
        wy[1] = 1 - wy[0] - wy[2] - wy[3]
    else:
        raise NotImplementedError(f"compute_bspline_indices_and_weights_2d() for degree={degree}")
    return i, j, wx, wy


def interpolate_cubic_bspline_1d(data: Tensor, x: float) -> Tensor:
    degree = 3
    i, w = compute_bspline_indices_and_weights_1d(x, degree=degree)
    w = w.to(data)
    val = torch.zeros(data.shape[:2]).to(data)
    for a in range(degree + 1):
        ia: int = max(0, min(i[a].item(), data.shape[2] - 1))
        val += data[:, :, ia].mul(w[a])
    return val


def interpolate_cubic_bspline_2d(data: Tensor, x: float, y: float) -> Tensor:
    degree = 3
    i, j, wx, wy = compute_bspline_indices_and_weights_2d(x, y, degree=degree)
    wx = wx.to(data)
    wy = wy.to(data)
    val = torch.zeros(data.shape[:2]).to(data)
    for b in range(degree + 1):
        jb: int = max(0, min(j[b].item(), data.shape[2] - 1))
        for a in range(degree + 1):
            ia: int = max(0, min(i[a].item(), data.shape[3] - 1))
            val += data[..., jb, ia].mul(wx[a] * wy[b])
    return val


# %% Evaluate B-spline values at output points
D = cp_data.ndim - 2
N = cp_data.shape[0]
C = cp_data.shape[1]

conv_fn: Callable[..., Tensor] = [F.conv1d, F.conv2d, F.conv3d][D - 1]
kernels = B.bspline_interpolation_weights(
    degree=3, stride=stride, dtype=cp_data.dtype, device=cp_data.device
)

start = timer()
output = cp_data
for dim, kernel in zip((SpatialDim(dim).tensor_dim(cp_data.ndim) for dim in range(D)), kernels):
    weight = kernel.reshape((kernel.shape[0], 1, kernel.shape[1]) + (1,) * (D - 1))
    weight = weight.tile((C,) + (1,) * (weight.ndim - 1))
    output = U.move_dim(output, dim, 2)
    output = conv_fn(output, weight, groups=C)
    output = output.reshape((N, C, kernel.shape[0]) + (output.shape[2:]))
    output = output.transpose(2, 3).flatten(2, 3)
    print(output)
    output = U.move_dim(output, 2, dim)
output = output[(slice(0, N), slice(0, C)) + tuple(slice(0, n) for n in reversed(in_size))]
output = output.contiguous()
print(f"Elapsed time: {timer() - start:.3f}s")

assert output.shape[0] == N
assert output.shape[1] == C
assert output.shape[2:] == tuple(reversed(in_size))


# %%
kernel = tuple(B.cubic_bspline1d(s) for s in stride)
for _ in range(3):
    start = timer()
    result1 = B.evaluate_cubic_bspline(
        cp_data, size=in_size, stride=stride, kernel=kernel, transpose=True
    )
    print(f"Elapsed time: {timer() - start:.3f}s")

kernel = B.bspline_interpolation_weights(
    degree=3, stride=stride, dtype=cp_data.dtype, device=cp_data.device
)
for _ in range(3):
    start = timer()
    result2 = B.evaluate_cubic_bspline(
        cp_data, size=in_size, stride=stride, kernel=kernel, transpose=False
    )
    print(f"Elapsed time: {timer() - start:.3f}s")

assert torch.allclose(result1, result2, atol=0.01)


# %% Cubic B-spline kernel and its derivatives
import matplotlib.pyplot as plt

fig, ax = plt.subplots(1, 1, figsize=(12, 10))

stride = 7

cp_data = torch.zeros((1, 1, 11))
cp_data[0, 0, (cp_data.shape[2] - 1) // 2] = 1

for derivative in range(3):
    kernel = B.cubic_bspline_interpolation_weights(stride=stride, derivative=derivative)
    values = B.evaluate_cubic_bspline(cp_data, stride=stride, kernel=kernel)
    ax.plot(values[0, 0])


# %%
