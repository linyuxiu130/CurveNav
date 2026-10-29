"""Shared source-planner geometry and simulator pose contracts."""
import numpy as np
import pytest
from curvenav.data_generation.generate import base_position_from_navmesh
from curvenav.data_generation.geometry import Grid,path_length,plan_route
from curvenav.physical import ROBOT_BASE_HEIGHT_ABOVE_GROUND_M


def test_clearance_aware_planner_is_safe_and_deterministic() -> None:
    free = np.ones((80, 80), dtype=bool)
    free[25:55, 36:44] = False
    from scipy.ndimage import distance_transform_edt

    clearance = (
        distance_transform_edt(np.pad(free, 1, constant_values=False))[1:-1, 1:-1]
        * 0.05
    )
    clearance[~free] = 0.0
    grid = Grid(free, clearance.astype(np.float32), np.zeros(2), 0.05)

    first = plan_route(grid, np.array([0.6, 2.0]), np.array([3.4, 2.0]), 0.)
    second = plan_route(grid, np.array([0.6, 2.0]), np.array([3.4, 2.0]), 0.)

    assert grid.safe(first.path_xy)
    assert np.allclose(first.path_xy, second.path_xy)
    assert path_length(first.path_xy) < 4.5


def test_render_pose_uses_the_dingo_base_link_height() -> None:
    floor_position = np.array([1.0, 2.0, 3.0], dtype=np.float32)

    base_position = base_position_from_navmesh(floor_position)

    assert np.array_equal(floor_position, np.array([1.0, 2.0, 3.0]))
    assert base_position == pytest.approx(
        [1.0, 2.0 + ROBOT_BASE_HEIGHT_ABOVE_GROUND_M, 3.0]
    )


def test_curve_clock_preserves_forward_turning_and_sensor_rate():
    from scipy.interpolate import BSpline
    from curvenav.data_generation.geometry import timed_route
    # Convex-hull cubic turns to a rear endpoint without reversing the body.
    curve = BSpline([0, 0, 0, 0, 1, 1, 1, 1],
                    [[0, 0], [2, 0], [2, 3], [-1, 3]], 3)
    xy, yaw, controls = timed_route(curve, .1, .3, .5)
    np.testing.assert_allclose(xy[[0, -1]], [[0, 0], [-1, 3]], atol=1e-10)
    assert np.all(controls[:, 0] > 0)
    assert np.max(controls[:, 0]) <= .3 + 1e-8
    assert np.max(abs(controls[:, 1])) <= .5 + 1e-8
    assert np.max(abs(np.diff(np.unwrap(yaw)))) <= .05 + 1e-6
    assert np.max(np.linalg.norm(np.diff(xy, axis=0), axis=1)) <= .03 + 1e-6
    # Central finite differences converge to the analytic forward-only velocity.
    velocity = (xy[2:] - xy[:-2]) / .2
    lateral = -np.sin(yaw[1:-1])*velocity[:, 0] + np.cos(yaw[1:-1])*velocity[:, 1]
    assert np.max(abs(lateral)) < 1e-4


def test_clearance_curve_energy_gradient_matches_finite_difference():
    from scipy.interpolate import BSpline
    from scipy.ndimage import distance_transform_edt
    from scipy.optimize._numdiff import approx_derivative
    from curvenav.data_generation.geometry import _travel_objective, _bending_objective
    free = np.ones((80,80),bool)
    free[30:40,30:40] = False
    distance = distance_transform_edt(np.pad(free,1))[1:-1,1:-1]*.05
    grid = Grid(free,distance,np.zeros(2),.05)
    controls = np.array([[.617,.793],[1.037,.719],[1.843,1.017],[2.413,.613]])
    u = np.linspace(0,1,37)
    identity = BSpline([0,0,0,0,1,1,1,1],np.eye(4),3)
    def objective(controls):
        travel, travel_gradient = _travel_objective(
            controls, identity(u), identity.derivative()(u), np.ones(37)/37, grid)
        bending, bending_gradient = _bending_objective(
            controls, identity.derivative()(u), identity.derivative(2)(u), np.ones(37)/37)
        return travel + bending, travel_gradient + bending_gradient
    value, gradient = objective(controls)
    numerical = approx_derivative(lambda x:objective(x.reshape(4,2))[0],controls.ravel(),method='3-point',abs_step=1e-6)
    assert np.isfinite(value)
    np.testing.assert_allclose(gradient.ravel(),numerical.ravel(),rtol=2e-4,atol=2e-5)


def test_clearance_optimization_keeps_a_narrow_feasible_corridor():
    from scipy.ndimage import distance_transform_edt
    free = np.zeros((80,80),bool)
    free[:,28:34] = True
    distance = distance_transform_edt(np.pad(free,1))[1:-1,1:-1]*.05
    grid = Grid(free,distance,np.zeros(2),.05)
    plan = plan_route(grid,np.array([.5,1.55]),np.array([3.5,1.55]), 0.)
    assert grid.safe(plan.path_xy)
    assert grid.clearance(plan.path_xy).min() <= .15+1e-6


@pytest.mark.parametrize("goal", [[2.5, 3.5], [1.5, 2.5], [2.3, 2.5]])
def test_prescribed_heading_reaches_side_and_rear_goals_forward(goal):
    from scipy.ndimage import distance_transform_edt
    from curvenav.data_generation.geometry import timed_route
    free = np.ones((100, 100), bool)
    clearance = distance_transform_edt(np.pad(free, 1))[1:-1, 1:-1] * .05
    grid = Grid(free, clearance, np.zeros(2), .05)
    plan = plan_route(grid, np.array([2.5, 2.5]), np.array(goal), 0.)
    xy, yaw, control = timed_route(plan.curve, .1, .3, .5)
    np.testing.assert_allclose(xy[[0, -1]], [[2.5, 2.5], goal], atol=1e-9)
    assert abs(yaw[0]) < 1e-10
    assert grid.safe(xy)
    assert np.all(control[:, 0] > 0)
    assert np.max(control[:, 0]) <= .3 + 1e-8
    assert np.max(abs(control[:, 1])) <= .5 + 1e-8
    assert np.max(abs(np.diff(np.unwrap(yaw)))) <= .05 + 1e-6
    assert np.max(np.linalg.norm(np.diff(xy, axis=0), axis=1)) <= .03 + 1e-6


def test_endpoint_sampling_uses_the_route_clearance_contract():
    from scipy.ndimage import distance_transform_edt
    from curvenav.data_generation.geometry import candidate_pairs
    free = np.zeros((80, 80), bool)
    free[:, 28:34] = True
    clearance = distance_transform_edt(np.pad(free, 1))[1:-1, 1:-1] * .05
    grid = Grid(free, clearance, np.zeros(2), .05)
    pairs = candidate_pairs(grid, (.5, 2.), 42, 4)
    assert len(pairs) == 4
    assert all(grid.safe(np.stack(pair)) for pair in pairs)


@pytest.mark.parametrize('narrow', [False, True])
def test_spatial_teacher_can_start_with_reverse_tangent_in_open_or_straight_corridor(narrow):
    from scipy.ndimage import distance_transform_edt
    from curvenav.data_generation.geometry import source_route

    free = np.ones((80, 40), dtype=bool)
    if narrow:
        free[:, :12] = False
        free[:, 28:] = False
    distance = distance_transform_edt(np.pad(free, 1, constant_values=False))[1:-1, 1:-1]*.05
    grid = Grid(free, np.where(free, distance, 0).astype(np.float32), np.zeros(2), .05)
    start, goal = np.array([3.2, 1.]), np.array([.8, 1.])
    spatial = source_route(grid, start, goal, None)
    prescribed = source_route(grid, start, goal, np.pi)
    np.testing.assert_array_equal(spatial.curve(0.), start)
    tangent = spatial.curve.derivative()(0.)
    assert tangent[0] < 0 and abs(tangent[1]) < 1e-8
    assert grid.safe(spatial.path_xy)
    # Explicit numeric yaw retains the same original smoothing path.
    np.testing.assert_array_equal(spatial.path_xy, prescribed.path_xy)


def test_near_wall_recovery_uses_explicit_grid_margin_without_changing_default():
    from dataclasses import replace
    from curvenav.data_generation.geometry import source_route, PlanningError, MIN_CLEARANCE_M

    free = np.ones((80, 40), dtype=bool)
    clearance = np.full(free.shape, .085, dtype=np.float32)
    grid = Grid(free, clearance, np.zeros(2), .05)
    start, goal = np.array([.5, 1.]), np.array([2.5, 1.])
    assert grid.minimum_clearance_m == MIN_CLEARANCE_M == .1
    assert not grid.safe(np.stack((start, goal)))
    with pytest.raises(PlanningError):
        source_route(grid, start, goal, None)
    recovery = replace(grid, minimum_clearance_m=0.)
    plan = source_route(recovery, start, goal, None)
    assert recovery.safe(plan.path_xy)
    assert not recovery.safe(plan.path_xy, minimum_m=.1)
    assert grid.minimum_clearance_m == .1
