import pytest

import numpy as np
import torch
from torch import Tensor

from deepali.core import Cube, Grid
from deepali.core import affine as A


@pytest.fixture
def default_angle() -> Tensor:
    return torch.deg2rad(torch.tensor(33.0))


@pytest.fixture
def default_cube(default_angle: Tensor) -> Cube:
    direction = A.euler_rotation_matrix(default_angle)
    cube = Cube(extent=(34, 42), center=(7, 4), direction=direction)
    return cube


def test_cube_is_grid_with_three_points(default_cube: Cube) -> None:
    cube = default_cube
    grid = cube.grid(size=3, align_corners=True)
    assert type(grid) is Grid
    assert grid.ndim == cube.ndim
    assert grid.device == cube.device
    assert grid.size() == (3,) * grid.ndim
    assert grid.align_corners() is True
    assert torch.allclose(grid.cube_extent(), cube.extent())
    assert torch.allclose(grid.affine(), cube.affine())
    assert torch.allclose(grid.inverse_affine(), cube.inverse_affine())
    assert torch.allclose(grid.spacing(), cube.spacing())
    assert torch.allclose(grid.transform("cube_corners", "world"), cube.transform("cube", "world"))
    assert torch.allclose(grid.transform("world", "cube_corners"), cube.transform("world", "cube"))
    assert torch.allclose(grid.transform(), cube.transform())
    assert torch.allclose(grid.inverse_transform(), cube.inverse_transform())


def test_cube_to_from_numpy(default_cube: Cube) -> None:
    r"""Test converting a Cube to a 1-dimensional NumPy array and constructing a new one from such array."""
    dim = default_cube.ndim
    arr = default_cube.numpy()
    assert isinstance(arr, np.ndarray)
    assert arr.ndim == 1
    assert arr.dtype == (np.float32 if default_cube.dtype == torch.float32 else np.float64)
    assert arr.shape == ((dim + 2) * dim,)
    cube = Cube.from_numpy(arr)
    assert cube == default_cube


def test_cube_eq(default_cube: Cube, default_angle: Tensor) -> None:
    r"""Test comparison of different Cube instances for equality."""
    # Same instance
    assert default_cube == default_cube

    # Different instance, same atributes
    other_cube = Cube(
        extent=default_cube.extent(),
        center=default_cube.center(),
        direction=default_cube.direction(),
    )
    assert default_cube == other_cube

    # Different extent
    other_cube = Cube(
        extent=default_cube.extent().add(1),
        center=default_cube.center(),
        direction=default_cube.direction(),
    )
    assert default_cube != other_cube

    # Different center
    other_cube = Cube(
        extent=default_cube.extent(),
        center=default_cube.center().add(0.001),
        direction=default_cube.direction(),
    )
    assert default_cube != other_cube

    # Different direction
    other_direction = A.euler_rotation_matrix(default_angle.add(0.001))
    other_cube = Cube(
        extent=default_cube.extent(),
        center=default_cube.center(),
        direction=other_direction,
    )
    assert default_cube != other_cube
