import torch

from deepali.core import Grid
import deepali.core.functional as U
import deepali.losses.functional as L


def test_losses_flow_bending() -> None:
    data = torch.randn((1, 2, 16, 24))
    a = L.bending_energy(data, mode="bspline", stride=1, reduction="none")
    b = L.bspline_bending_energy(data, stride=1, reduction="none")
    assert torch.allclose(a, b)


def test_losses_flow_curvature() -> None:
    grid = Grid(size=(16, 14))
    offset = U.translation([0.1, 0.2]).unsqueeze_(0)
    flow = U.affine_flow(offset, grid)
    flow.requires_grad = True
    loss = L.curvature_loss(flow)
    assert loss.requires_grad
    loss = loss.detach()
    assert loss.abs().max().lt(1e-5)
