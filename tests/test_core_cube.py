import torch

from deepali.core import Cube, Grid
from deepali.core import affine as A


def test_cube_is_grid_with_three_points() -> None:
    angle = torch.deg2rad(torch.tensor(33.0))
    direction = A.euler_rotation_matrix(angle)
    cube = Cube(extent=(34, 42), center=(7, 4), direction=direction)
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
