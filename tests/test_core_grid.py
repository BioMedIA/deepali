import torch

from deepali.core.grid import Grid


def test_grid_crop():
    grid = Grid((10, 7, 5))

    new_grid = grid.crop(2, -1)
    assert isinstance(new_grid, Grid)
    assert new_grid is not grid
    assert new_grid.size() == (6, 9, 5)
    assert torch.allclose(new_grid.center(), grid.center())
    assert torch.allclose(new_grid.index_to_world((-2, 1, 0)), grid.origin())

    new_grid_2 = grid.crop((2, -1))
    assert new_grid_2 == new_grid

    new_grid_2 = grid.crop(margin=(2, -1))
    assert new_grid_2 == new_grid

    new_grid_2 = grid.crop(num=(2, 2, -1, -1))
    assert new_grid_2 == new_grid


def test_grid_pad():
    grid = Grid((10, 7, 5))

    new_grid = grid.pad(4, -1)
    assert isinstance(new_grid, Grid)
    assert new_grid is not grid
    assert new_grid.size() == (18, 5, 5)
    assert torch.allclose(new_grid.center(), grid.center())
    assert torch.allclose(new_grid.index_to_world((4, -1, 0)), grid.origin())

    new_grid_2 = grid.pad((4, -1))
    assert new_grid_2 == new_grid

    new_grid_2 = grid.pad(margin=(4, -1))
    assert new_grid_2 == new_grid

    new_grid_2 = grid.pad(num=(4, 4, -1, -1))
    assert new_grid_2 == new_grid
