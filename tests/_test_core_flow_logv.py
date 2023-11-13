# %%
# Imports
from typing import Optional, Sequence

import matplotlib.pyplot as plt

import torch
from torch import Tensor
from torch.random import Generator

from deepali.core import Grid
import deepali.core.bspline as B
import deepali.core.functional as U


# %%
# Auxiliary functions
def random_svf(
    size: Sequence[int],
    stride: int = 1,
    generator: Optional[Generator] = None,
) -> Tensor:
    cp_grid_size = B.cubic_bspline_control_point_grid_size(size, stride=stride)
    data = torch.randn((1, 3) + cp_grid_size, generator=generator)
    data = U.fill_border(data, margin=3, value=0, inplace=True)
    return B.evaluate_cubic_bspline(data, size=size, stride=stride)


def visualize_flow(ax, flow: Tensor, label: Optional[str] = None) -> None:
    grid = Grid(shape=flow.shape[2:], align_corners=True)
    x = grid.coords(channels_last=False, dtype=u.dtype, device=u.device)
    x = U.move_dim(x.unsqueeze(0).add_(flow), 1, -1)
    target_grid = U.grid_image(shape=flow.shape[2:], inverted=True, stride=(5, 5))
    warped_grid = U.warp_image(target_grid, x)
    ax.imshow(warped_grid[0, 0, flow.shape[2] // 2], cmap="gray")
    if label:
        ax.set_title(label, fontsize=24)


# %%
# Random velocity fields
size = (128, 128, 128)
generator = torch.Generator().manual_seed(42)
v = random_svf(size, stride=8, generator=generator).mul_(0.1)


# %%
# Compute logarithm of exponentiated velocity field
bch_terms = 3
exp_steps = 5
log_steps = 5

u = U.expv(v, steps=exp_steps)
w = U.logv(u, num_iters=log_steps, bch_terms=bch_terms, exp_steps=exp_steps, sigma=1.0)

fig, axes = plt.subplots(1, 4, figsize=(40, 10))

ax = axes[0]
ax.set_title("v", fontsize=32, pad=20)
visualize_flow(ax, v)

ax = axes[1]
ax.set_title("u = exp(v)", fontsize=32, pad=20)
visualize_flow(ax, u)

ax = axes[2]
ax.set_title("log(u)", fontsize=32, pad=20)
visualize_flow(ax, w)

error = w.sub(v).norm(dim=1, keepdim=True)

ax = axes[3]
ax.set_title("|log(u) - v|", fontsize=32, pad=20)
_ = ax.imshow(error[0, 0, error.shape[2] // 2], cmap="jet", vmin=0, vmax=0.1)

print(f"Mean error:     {error.mean():.5f}")
print(f"Maximium error: {error.max():.5f}")

# %%
