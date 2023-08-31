import torch
import torch.nn.functional as F
from torch import Tensor

from deepali.core import Grid
from deepali.core.enum import FlowDerivativeKeys
import deepali.core.functional as U


PERIODIC_FLOW_X_SCALE = 2 * torch.pi
PERIODIC_FLOW_U_SCALE = 0.1


def periodic_flow(p: Tensor) -> Tensor:
    q = p.mul(PERIODIC_FLOW_X_SCALE)
    u = q.narrow(1, 0, 1).sin().neg_()
    v = q.narrow(1, 1, 1).cos()
    f = torch.cat([u, v], dim=1)
    return f.mul_(PERIODIC_FLOW_U_SCALE)


def periodic_flow_du_dx(p: Tensor) -> Tensor:
    q = p.mul(PERIODIC_FLOW_X_SCALE)
    g = q.narrow(1, 0, 1).cos()
    g = g.mul_(-PERIODIC_FLOW_X_SCALE * PERIODIC_FLOW_U_SCALE)
    return g


def periodic_flow_du_dy(p: Tensor) -> Tensor:
    return torch.zeros((p.shape[0], 1) + p.shape[2:], dtype=p.dtype, device=p.device)


def periodic_flow_du_dxx(p: Tensor) -> Tensor:
    q = p.mul(PERIODIC_FLOW_X_SCALE)
    g = q.narrow(1, 0, 1).sin()
    g = g.mul_(PERIODIC_FLOW_X_SCALE * PERIODIC_FLOW_X_SCALE * PERIODIC_FLOW_U_SCALE)
    return g


def periodic_flow_du_dyy(p: Tensor) -> Tensor:
    return torch.zeros((p.shape[0], 1) + p.shape[2:], dtype=p.dtype, device=p.device)


def periodic_flow_dv_dx(p: Tensor) -> Tensor:
    return torch.zeros((p.shape[0], 1) + p.shape[2:], dtype=p.dtype, device=p.device)


def periodic_flow_dv_dy(p: Tensor) -> Tensor:
    q = p.mul(PERIODIC_FLOW_X_SCALE)
    g = q.narrow(1, 1, 1).sin()
    g = g.mul_(-PERIODIC_FLOW_X_SCALE * PERIODIC_FLOW_U_SCALE)
    return g


def periodic_flow_dv_dxx(p: Tensor) -> Tensor:
    return torch.zeros((p.shape[0], 1) + p.shape[2:], dtype=p.dtype, device=p.device)


def periodic_flow_dv_dyy(p: Tensor) -> Tensor:
    q = p.mul(PERIODIC_FLOW_X_SCALE)
    g = q.narrow(1, 1, 1).cos()
    g = g.mul_(-PERIODIC_FLOW_X_SCALE * PERIODIC_FLOW_X_SCALE * PERIODIC_FLOW_U_SCALE)
    return g


def periodic_flow_deriv(p: Tensor, which: str) -> Tensor:
    deriv_fn = {
        "du/dx": periodic_flow_du_dx,
        "du/dy": periodic_flow_du_dy,
        "dv/dx": periodic_flow_dv_dx,
        "dv/dy": periodic_flow_dv_dy,
        "du/dxx": periodic_flow_du_dxx,
        "du/dyy": periodic_flow_du_dyy,
        "dv/dxx": periodic_flow_dv_dxx,
        "dv/dyy": periodic_flow_dv_dyy,
    }
    return deriv_fn[which](p)


def periodic_flow_divergence(p: Tensor) -> Tensor:
    du_dx = periodic_flow_du_dx(p)
    dv_dy = periodic_flow_dv_dy(p)
    return du_dx.add(dv_dy)


def difference(a: Tensor, b: Tensor, margin: int = 0) -> Tensor:
    assert a.shape == b.shape
    i = [
        slice(0, n, 1) if dim < 2 else slice(margin, n - margin, 1) for dim, n in enumerate(a.shape)
    ]
    return a[i].sub(b[i])


def test_flow_curl() -> None:
    # 2-dimensional vector field
    p = U.move_dim(Grid(size=(32, 24)).coords().unsqueeze_(0), -1, 1)

    x = p.narrow(1, 0, 1)
    y = p.narrow(1, 1, 1)

    flow = torch.cat([y.neg(), x], dim=1)

    curl = U.curl(flow)
    assert isinstance(curl, Tensor)
    assert curl.shape == (flow.shape[0], 1) + flow.shape[2:]
    assert curl.dtype == flow.dtype
    assert curl.device == flow.device
    assert curl.sub(2).abs().lt(1e-6).all()

    # 3-dimensional vector field
    p = U.move_dim(Grid(size=(64, 32, 16)).coords().unsqueeze_(0), -1, 1)

    x = p.narrow(1, 0, 1)
    y = p.narrow(1, 1, 1)
    z = p.narrow(1, 2, 1)

    flow = torch.cat([x.mul(z), y.mul(z), x.mul(y)], dim=1)

    curl = U.curl(flow)
    assert isinstance(curl, Tensor)
    assert curl.shape == flow.shape
    assert curl.dtype == flow.dtype
    assert curl.device == flow.device

    expected = x.sub(y)
    expected = torch.cat([expected, expected, torch.zeros_like(expected)], dim=1)
    error = difference(curl, expected, margin=1)
    assert error.abs().max().lt(1e-5)

    div = U.divergence(curl)
    assert div.abs().max().lt(1e-5)


def test_flow_derivatives() -> None:
    # 2D vector field defined by periodic functions
    p = U.move_dim(Grid(size=(120, 100)).coords().unsqueeze_(0), -1, 1)

    flow = periodic_flow(p)

    which = FlowDerivativeKeys.all(spatial_dims=2, order=1)
    which.append("du/dxx")
    which.append("dv/dyy")

    deriv = U.flow_derivatives(flow, which=which)
    assert isinstance(deriv, dict)
    assert all(isinstance(k, str) for k in deriv.keys())
    assert all(isinstance(v, Tensor) for v in deriv.values())
    assert all(v.shape == (flow.shape[0], 1) + flow.shape[2:] for v in deriv.values())

    for key, value in deriv.items():
        expected = periodic_flow_deriv(p, key)
        order = FlowDerivativeKeys.order(key)
        dif = difference(value, expected, margin=order)
        tol = 0.003 * (10 ** (order - 1))
        assert dif.abs().max().lt(tol), f"flow derivative {key}"

    # 3D vector field
    p = U.move_dim(Grid(size=(64, 32, 16)).coords().unsqueeze_(0), -1, 1)

    x = p.narrow(1, 0, 1)
    y = p.narrow(1, 1, 1)
    z = p.narrow(1, 2, 1)

    flow = torch.cat([x.mul(z), y.mul(z), x.mul(y)], dim=1)

    deriv = U.flow_derivatives(flow, order=1)

    assert difference(deriv["du/dx"], z).abs().max().lt(1e-5)
    assert deriv["du/dy"].abs().max().lt(1e-5)
    assert difference(deriv["du/dz"], x).abs().max().lt(1e-5)

    assert deriv["dv/dx"].abs().max().lt(1e-5)
    assert difference(deriv["dv/dy"], z).abs().max().lt(1e-5)
    assert difference(deriv["dv/dz"], y).abs().max().lt(1e-5)

    assert difference(deriv["dw/dx"], y).abs().max().lt(1e-5)
    assert difference(deriv["dw/dy"], x).abs().max().lt(1e-5)
    assert deriv["dw/dz"].abs().max().lt(1e-5)


def test_flow_divergence() -> None:
    grid = Grid(size=(16, 14))
    offset = U.translation([0.1, 0.2]).unsqueeze_(0)
    flow = U.affine_flow(offset, grid)
    div = U.divergence(flow)
    assert isinstance(div, Tensor)
    assert div.shape == (flow.shape[0], 1) + flow.shape[2:]
    assert div.abs().max().lt(1e-5)

    points = U.move_dim(Grid(size=(64, 64)).coords().unsqueeze_(0), -1, 1)
    flow = periodic_flow(points)
    div = U.divergence(flow)
    assert isinstance(div, Tensor)
    assert div.shape == (flow.shape[0], 1) + flow.shape[2:]
    expected = periodic_flow_divergence(points)
    dif = difference(div, expected, margin=1)
    assert dif.abs().max().lt(0.01)

    p = U.move_dim(Grid(size=(64, 32, 16)).coords().unsqueeze_(0), -1, 1)
    x = p.narrow(1, 0, 1)
    y = p.narrow(1, 1, 1)
    z = p.narrow(1, 2, 1)
    flow = torch.cat([x.mul(z), y.mul(z), x.mul(y)], dim=1)
    div = U.divergence(flow)
    assert isinstance(div, Tensor)
    assert div.shape == (flow.shape[0], 1) + flow.shape[2:]
    error = difference(div, z.mul(2))
    assert error.abs().max().lt(1e-5)


def test_flow_divergence_free() -> None:
    data = torch.randn((1, 1, 16, 24)).mul_(0.01)
    flow = U.divergence_free_flow(data)
    assert flow.shape == (data.shape[0], 2) + data.shape[2:]
    div = U.divergence(flow)
    assert div.abs().max().lt(1e-5)

    data = torch.randn((3, 2, 16, 24, 32)).mul_(0.01)
    flow = U.divergence_free_flow(data, sigma=2.0)
    assert flow.shape == (data.shape[0], 3) + data.shape[2:]
    div = U.divergence(flow)
    assert div[0, 0, 1:-1, 1:-1, 1:-1].abs().max().lt(1e-4)

    coef = F.pad(data, (1, 2, 1, 2, 1, 2))
    flow = U.divergence_free_flow(coef, mode="bspline", sigma=0.8)
    assert flow.shape == (data.shape[0], 3) + data.shape[2:]
    div = U.divergence(flow)
    assert div[0, 0, 1:-1, 1:-1, 1:-1].abs().max().lt(1e-4)

    # constructing a divergence-free field using curl() seems to work best given
    # the higher magnitude and no need for Gaussian blurring of the random field
    # where each component is sampled i.i.d. from a normal distribution
    data = torch.randn((5, 3, 16, 24, 32)).mul_(0.2)
    flow = U.divergence_free_flow(data)
    assert flow.shape == data.shape
    div = U.divergence(flow)
    assert div.abs().max().lt(1e-4)
