import numpy as np
import pytest

from curvenav.data.goal_distance import GoalDistanceQuery
from curvenav.data.privileged import SourceConfigurationGrid, SourceConfigurationSpaceQuery


@pytest.mark.parametrize("seed", range(6))
def test_implicit_grid_matches_scipy_including_blocked_corners(seed, monkeypatch):
    pytest.importorskip("numba")
    from curvenav.data.grid_dijkstra import grid_distances
    rng = np.random.default_rng(seed)
    free = rng.random((55, 71)) > 0.25
    free[20, :] = False
    free[0, 0] = True
    free[0, 1] = free[1, 0] = False
    grid = SourceConfigurationGrid(np.where(free, 1., -1.).astype(np.float32), np.zeros(2), 0.025)
    source = SourceConfigurationSpaceQuery((grid,))
    cell = np.argwhere(free)[100]
    monkeypatch.setenv("CURVENAV_GOAL_DISTANCE_BACKEND", "scipy")
    reference = GoalDistanceQuery(source)._field(0, cell)
    monkeypatch.setenv("CURVENAV_GOAL_DISTANCE_BACKEND", "numba")
    fast = GoalDistanceQuery(source)
    np.testing.assert_array_equal(fast._field(0, cell), reference)
    assert not fast.graphs
    isolated = grid_distances(free, grid.cell_size_m, 0, 0)
    assert np.isfinite(isolated).sum() == 1
