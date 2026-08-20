import numpy as np

from curvenav.data_generation.audit_hssd_runs import (
    _continuous_clearance,
    audit_dataset_root,
)
from curvenav.data_generation.occupancy import NavigationGrid


def test_hssd_audit_checks_between_decoded_vertices():
    free = np.ones((10, 3), dtype=bool)
    free[4, 1] = False
    clearance = np.full(free.shape, 0.5, dtype=np.float32)
    clearance[4, 1] = 0.0
    grid = NavigationGrid(
        free=free,
        clearance_m=clearance,
        origin_xy=np.zeros(2),
        cell_size_m=0.05,
    )
    paths = np.array([[[0.025, 0.075], [0.475, 0.075]]])

    sampled = _continuous_clearance(grid, paths)

    assert sampled.min() == -np.inf


def test_hssd_audit_checks_every_curve_window(tmp_path):
    dataset = tmp_path / "dataset_hssd_test"
    run = dataset / "run_0000"
    run.mkdir(parents=True)

    free = np.ones((80, 30), dtype=bool)
    clearance = np.full(free.shape, 0.5, dtype=np.float32)
    np.savez_compressed(
        dataset / "navigation_grid.npz",
        free=free,
        clearance_m=clearance,
        origin_xy=np.array([-1.0, -1.0]),
        cell_size_m=0.05,
    )
    x = np.linspace(0.0, 2.0, 15, dtype=np.float32)
    np.save(run / "traj_xyz.npy", np.column_stack([x, np.zeros_like(x), x * 0.0]))
    np.save(run / "traj_yaw.npy", np.zeros_like(x))

    report = audit_dataset_root(tmp_path, min_gap=5, max_gap=8)

    assert report["total_windows"] == sum(min(8, 14 - start) - 4 for start in range(10))
    assert report["unsafe_windows"] == 0
    assert report["contract"]["minimum_clearance_m"] == 0.1
    assert report["contract"]["clearance_sample_step_m"] == 0.025
    assert report["runs"][0]["dataset"] == "dataset_hssd_test"
    assert report["runs"][0]["maximum_endpoint_error_m"] == 0.0
