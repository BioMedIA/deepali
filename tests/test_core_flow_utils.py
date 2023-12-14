from typing import Optional, Sequence

import pytest
import torch
from torch import Tensor
from torch.random import Generator

from deepali.core import Grid
from deepali.core.enum import FlowDerivativeKeys
import deepali.core.bspline as B
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
    g = p.narrow(1, 0, 1).mul(PERIODIC_FLOW_X_SCALE).cos()
    g = g.mul_(-PERIODIC_FLOW_X_SCALE * PERIODIC_FLOW_U_SCALE)
    return g


def periodic_flow_du_dy(p: Tensor) -> Tensor:
    return torch.zeros((p.shape[0], 1) + p.shape[2:], dtype=p.dtype, device=p.device)


def periodic_flow_du_dxx(p: Tensor) -> Tensor:
    g = p.narrow(1, 0, 1).mul(PERIODIC_FLOW_X_SCALE).sin()
    g = g.mul_(PERIODIC_FLOW_X_SCALE * PERIODIC_FLOW_X_SCALE * PERIODIC_FLOW_U_SCALE)
    return g


def periodic_flow_du_dyy(p: Tensor) -> Tensor:
    return torch.zeros((p.shape[0], 1) + p.shape[2:], dtype=p.dtype, device=p.device)


def periodic_flow_dv_dx(p: Tensor) -> Tensor:
    return torch.zeros((p.shape[0], 1) + p.shape[2:], dtype=p.dtype, device=p.device)


def periodic_flow_dv_dy(p: Tensor) -> Tensor:
    g = p.narrow(1, 1, 1).mul(PERIODIC_FLOW_X_SCALE).sin()
    g = g.mul_(-PERIODIC_FLOW_X_SCALE * PERIODIC_FLOW_U_SCALE)
    return g


def periodic_flow_dv_dxx(p: Tensor) -> Tensor:
    return torch.zeros((p.shape[0], 1) + p.shape[2:], dtype=p.dtype, device=p.device)


def periodic_flow_dv_dyy(p: Tensor) -> Tensor:
    g = p.narrow(1, 1, 1).mul(PERIODIC_FLOW_X_SCALE).cos()
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


def random_svf(
    size: Sequence[int],
    stride: int = 1,
    generator: Optional[Generator] = None,
) -> Tensor:
    cp_grid_size = B.cubic_bspline_control_point_grid_size(size, stride=stride)
    data = torch.randn((1, 3) + cp_grid_size, generator=generator)
    data = U.fill_border(data, margin=3, value=0, inplace=True)
    return B.evaluate_cubic_bspline(data, size=size, stride=stride)


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

    deriv = U.flow_derivatives(flow, which=["du/dxz", "dv/dzy", "dw/dxy"])
    assert deriv["du/dxz"].sub(1).abs().max().lt(1e-4)
    assert deriv["dv/dzy"].sub(1).abs().max().lt(1e-4)
    assert deriv["dw/dxy"].sub(1).abs().max().lt(1e-4)


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
    assert div[0, 0, 1:-1, 1:-1, 1:-1].abs().max().lt(1e-3)

    # coef = F.pad(data, (1, 2, 1, 2, 1, 2))
    # flow = U.divergence_free_flow(coef, mode="bspline", sigma=1.0)
    # assert flow.shape == (data.shape[0], 3) + data.shape[2:]
    # div = U.divergence(flow)
    # assert div[0, 0, 1:-1, 1:-1, 1:-1].abs().max().lt(1e-4)

    # flow = U.divergence_free_flow(data, mode="gaussian", sigma=0.7355)
    # assert flow.shape == (data.shape[0], 3) + data.shape[2:]
    # div = U.divergence(flow)
    # assert div[0, 0, 1:-1, 1:-1, 1:-1].abs().max().lt(1e-4)

    # constructing a divergence-free field using curl() seems to work best given
    # the higher magnitude and no need for Gaussian blurring of the random field
    # where each component is sampled i.i.d. from a normal distribution
    data = torch.randn((5, 3, 16, 24, 32)).mul_(0.2)
    flow = U.divergence_free_flow(data)
    assert flow.shape == data.shape
    div = U.divergence(flow)
    assert div.abs().max().lt(1e-4)


def test_flow_jacobian() -> None:
    # 2D flow field
    p = Grid(size=(64, 32)).coords(channels_last=False).unsqueeze_(0)

    x = p.narrow(1, 0, 1)
    y = p.narrow(1, 1, 1)

    interior = [slice(1, n - 1) for n in p.shape[2:]]

    # u = [x^2, xy]
    flow = torch.cat([x.square(), x.mul(y)], dim=1)

    jac = torch.zeros((p.shape[0],) + p.shape[2:] + (2, 2))
    jac[..., 0, 0] = x.squeeze(1).mul(2)
    jac[..., 1, 0] = y.squeeze(1)
    jac[..., 1, 1] = x.squeeze(1)

    derivs = U.jacobian_dict(flow)
    for (i, j), deriv in derivs.items():
        atol = 1e-5
        error = difference(jac[..., i, j].unsqueeze(1), deriv)
        if (i, j) == (0, 0):
            error = error[[slice(None), slice(None)] + interior]
        if error.abs().max().gt(atol):
            raise AssertionError(f"max absolute difference of jac[{i}, {j}] > {atol}")

    mat = U.jacobian_matrix(flow)
    assert torch.allclose(
        mat[[slice(None)] + interior],
        jac[[slice(None)] + interior],
        atol=1e-5,
    )

    jac[..., 0, 0] += 1
    jac[..., 1, 1] += 1

    mat = U.jacobian_matrix(flow, add_identity=True)
    assert torch.allclose(
        mat[[slice(None)] + interior],
        jac[[slice(None)] + interior],
        atol=1e-5,
    )

    det = U.jacobian_det(flow)
    assert torch.allclose(
        det[[slice(None), 0] + interior],
        jac[[slice(None)] + interior].det(),
        atol=1e-5,
    )

    # 3D flow field
    p = Grid(size=(64, 32, 16)).coords(channels_last=False).unsqueeze_(0)

    x = p.narrow(1, 0, 1)
    y = p.narrow(1, 1, 1)
    z = p.narrow(1, 2, 1)

    interior = [slice(1, n - 1) for n in p.shape[2:]]

    # u = [z^2, 0, xy]
    flow = torch.cat([z.square(), torch.zeros_like(y), x.mul(y)], dim=1)

    jac = torch.zeros((p.shape[0],) + p.shape[2:] + (3, 3))
    jac[..., 0, 2] = z.squeeze(1).mul(2)
    jac[..., 2, 0] = y.squeeze(1)
    jac[..., 2, 1] = x.squeeze(1)

    derivs = U.jacobian_dict(flow)
    for (i, j), deriv in derivs.items():
        atol = 1e-5
        error = difference(jac[..., i, j].unsqueeze(1), deriv)
        if (i, j) == (0, 2):
            error = error[[slice(None), slice(None)] + interior]
        if error.abs().max().gt(atol):
            raise AssertionError(f"max absolute difference of jac[{i}, {j}] > {atol}")

    mat = U.jacobian_matrix(flow)
    assert torch.allclose(
        mat[[slice(None)] + interior],
        jac[[slice(None)] + interior],
        atol=1e-5,
    )

    jac[..., 0, 0] += 1
    jac[..., 1, 1] += 1
    jac[..., 2, 2] += 1

    mat = U.jacobian_matrix(flow, add_identity=True)
    assert torch.allclose(
        mat[[slice(None)] + interior],
        jac[[slice(None)] + interior],
        atol=1e-5,
    )

    det = U.jacobian_det(flow)
    assert torch.allclose(
        det[[slice(None), 0] + interior],
        jac[[slice(None)] + interior].det(),
        atol=1e-5,
    )

    # u = [0, x + y^3, yz]
    flow = torch.cat([torch.zeros_like(x), x.add(y.pow(3)), y.mul(z)], dim=1)

    jac = torch.zeros((p.shape[0],) + p.shape[2:] + (3, 3))
    jac[..., 1, 0] = 1
    jac[..., 1, 1] = y.squeeze(1).square().mul(3)
    jac[..., 2, 1] = z.squeeze(1)
    jac[..., 2, 2] = y.squeeze(1)

    derivs = U.jacobian_dict(flow)
    for (i, j), deriv in derivs.items():
        atol = 1e-5
        error = difference(jac[..., i, j].unsqueeze(1), deriv)
        if (i, j) == (1, 1):
            atol = 0.005
            error = error[[slice(None), slice(None)] + interior]
        if error.abs().max().gt(atol):
            raise AssertionError(f"max absolute difference of jac[{i}, {j}] > {atol}")

    mat = U.jacobian_matrix(flow)
    assert torch.allclose(
        mat[[slice(None)] + interior],
        jac[[slice(None)] + interior],
        atol=0.005,
    )

    jac[..., 0, 0] += 1
    jac[..., 1, 1] += 1
    jac[..., 2, 2] += 1

    mat = U.jacobian_matrix(flow, add_identity=True)
    assert torch.allclose(
        mat[[slice(None)] + interior],
        jac[[slice(None)] + interior],
        atol=0.005,
    )

    det = U.jacobian_det(flow)
    assert torch.allclose(
        det[[slice(None), 0] + interior],
        jac[[slice(None)] + interior].det(),
        atol=0.01,
    )


def test_flow_lie_bracket() -> None:
    p = U.move_dim(Grid(size=(64, 32, 16)).coords().unsqueeze_(0), -1, 1)

    x = p.narrow(1, 0, 1)
    y = p.narrow(1, 1, 1)
    z = p.narrow(1, 2, 1)

    # u = [yz, xz, xy] and v = [x, y, z]
    u = torch.cat([y.mul(z), x.mul(z), x.mul(y)], dim=1)
    v = torch.cat([x, y, z], dim=1)
    w = u

    lb_uv = U.lie_bracket(u, v)
    assert torch.allclose(U.lie_bracket(v, u), lb_uv.neg())
    assert U.lie_bracket(u, u).abs().lt(1e-6).all()
    assert torch.allclose(lb_uv, w, atol=1e-6)

    # u = [z^2, 0, xy] and v = [0, x + y^3, yz]
    u = torch.cat([z.square(), torch.zeros_like(y), x.mul(y)], dim=1)
    v = torch.cat([torch.zeros_like(x), x.add(y.pow(3)), y.mul(z)], dim=1)
    w = torch.cat([-2 * y * z**2, z**2, x * y**2 - x**2 - x * y**3], dim=1).neg_()

    lb_uv = U.lie_bracket(u, v)
    assert torch.allclose(U.lie_bracket(v, u), lb_uv.neg())
    assert U.lie_bracket(u, u).abs().lt(1e-6).all()
    error = difference(lb_uv, w).abs()
    assert error[:, :, 1:-1, 1:-1, 1:-1].max().lt(1e-5)
    assert error.max().lt(0.134)


def test_flow_logv() -> None:
    size = (128, 128, 128)
    generator = torch.Generator().manual_seed(42)
    v = random_svf(size, stride=8, generator=generator).mul_(0.1)
    u = U.expv(v)
    w = U.logv(u)
    error = w.sub(v).norm(dim=1, keepdim=True)
    assert error.mean().lt(0.001)
    assert error.max().lt(0.02)


def test_flow_compose_svfs() -> None:
    # 3D flow fields
    p = U.move_dim(Grid(size=(64, 32, 16)).coords().unsqueeze_(0), -1, 1)

    x = p.narrow(1, 0, 1)
    y = p.narrow(1, 1, 1)
    z = p.narrow(1, 2, 1)

    with pytest.raises(ValueError):
        U.compose_svfs(p, p, bch_terms=-1)
    with pytest.raises(NotImplementedError):
        U.compose_svfs(p, p, bch_terms=6)

    # u = [yz, xz, xy] and v = u
    u = v = torch.cat([y.mul(z), x.mul(z), x.mul(y)], dim=1)

    w = U.compose_svfs(u, v, bch_terms=0)
    assert torch.allclose(w, u.add(v))
    w = U.compose_svfs(u, v, bch_terms=1)
    assert torch.allclose(w, u.add(v))
    w = U.compose_svfs(u, v, bch_terms=2)
    assert torch.allclose(w, u.add(v))
    w = U.compose_svfs(u, v, bch_terms=3)
    assert torch.allclose(w, u.add(v))
    w = U.compose_svfs(u, v, bch_terms=4)
    assert torch.allclose(w, u.add(v), atol=1e-5)
    w = U.compose_svfs(u, v, bch_terms=5)
    assert torch.allclose(w, u.add(v), atol=1e-5)

    # u = [yz, xz, xy] and v = [x, y, z]
    u = torch.cat([y.mul(z), x.mul(z), x.mul(y)], dim=1)
    v = torch.cat([x, y, z], dim=1)

    w = U.compose_svfs(u, v, bch_terms=0)
    assert torch.allclose(w, u.add(v))
    w = U.compose_svfs(u, v, bch_terms=1)
    assert torch.allclose(w, u.mul(0.5).add(v), atol=1e-6)

    # u = random_svf(), u -> 0 at boundary
    # v = random_svf(), v -> 0 at boundary
    size = (64, 64, 64)
    generator = torch.Generator().manual_seed(42)
    u = random_svf(size, stride=4, generator=generator).mul_(0.1)
    v = random_svf(size, stride=4, generator=generator).mul_(0.05)
    w = U.compose_svfs(u, v, bch_terms=5)

    flow_u = U.expv(u)
    flow_v = U.expv(v)
    flow_w = U.expv(w)
    flow = U.compose_flows(flow_u, flow_v)

    error = flow_w.sub(flow).norm(dim=1)
    assert error.max().lt(0.01)
