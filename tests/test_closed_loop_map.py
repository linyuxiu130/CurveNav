import numpy as np
import pytest

from curvenav.evaluation.closed_loop_map import (
    NavigationMap,
    _closed_loop_plan_diagnostics,
    _draw_navigation_map,
    _focus_map_view,
    _plan_indices,
    _query_navigation_map,
    _rasterize_navigation_map,
    _render_episode,
    _render_scene_overview,
    _robot_to_world,
    _summarize_closed_loop,
)


def test_closed_loop_map_uses_robot_x_forward_y_left_contract() -> None:
    half_turn_left = np.array([0.0, 0.0, np.sin(np.pi / 4.0), np.cos(np.pi / 4.0)])
    world = _robot_to_world(
        np.array([[1.0, 0.0], [0.0, 1.0]]),
        np.array([3.0, -2.0, 0.0]),
        half_turn_left,
    )
    np.testing.assert_allclose(world, [[3.0, -1.0], [2.0, -2.0]], atol=1e-12)

    indices = _plan_indices(321)
    assert indices.shape == (160,)
    assert indices[0] == 0
    assert indices[-1] == 320
    assert np.all(np.diff(indices) > 0)


def test_closed_loop_map_has_explicit_free_and_obstacle_layers() -> None:
    matplotlib = pytest.importorskip("matplotlib")

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots()
    navigation = _rasterize_navigation_map(
        np.array([[-2.0, -1.0], [3.0, 4.0]])
    )
    legend = _draw_navigation_map(axis, navigation)
    assert axis.images[0].cmap.colors == ["#d7dce2", "#fbfcfe"]
    assert axis.get_xlim() == (-2.025, 3.025)
    assert axis.get_ylim() == (-1.025, 4.025)
    assert [item.get_label() for item in legend] == [
        "navigable robot-center space",
        "obstacle / non-navigable space",
    ]
    plt.close(figure)


def test_closed_loop_focus_is_task_local_and_clipped_to_the_map() -> None:
    class Axis:
        xlim = None
        ylim = None

        def set_xlim(self, lower: float, upper: float) -> None:
            self.xlim = (lower, upper)

        def set_ylim(self, lower: float, upper: float) -> None:
            self.ylim = (lower, upper)

    axis = Axis()
    navigation = NavigationMap(
        free=np.ones((200, 200), dtype=np.bool_),
        extent=(-5.0, 5.0, -5.0, 5.0),
    )
    _focus_map_view(axis, navigation, np.array([[4.5, 0.0], [4.8, 0.5]]))
    assert axis.xlim == pytest.approx((2.65, 5.0))
    assert axis.ylim == pytest.approx((-1.75, 2.25))


def test_navigation_ply_samples_are_queried_as_native_cell_centres() -> None:
    navigation = _rasterize_navigation_map(
        np.array([[0.0, 0.0], [0.10, 0.0]])
    )
    free = _query_navigation_map(
        navigation,
        np.array([[0.0, 0.0], [0.05, 0.0], [0.10, 0.0]]),
    )
    np.testing.assert_array_equal(free, [True, False, True])

    x_min, x_max, y_min, y_max = navigation.extent
    boundary = _query_navigation_map(
        navigation,
        np.array([[x_min, 0.0], [x_max, 0.0], [0.0, y_min], [0.0, y_max]]),
    )
    np.testing.assert_array_equal(boundary, [True, False, True, False])


def test_closed_loop_diagnostics_score_the_executed_prefix_and_replanning() -> None:
    free = np.ones((40, 40), dtype=np.bool_)
    free[:, 20] = False
    navigation = NavigationMap(free=free, extent=(0.0, 2.0, 0.0, 2.0))
    plan = np.column_stack((np.linspace(0.1, 1.5, 64), np.ones(64)))

    metrics = _closed_loop_plan_diagnostics(
        [plan, plan.copy()], [plan, plan.copy()], navigation
    )

    assert metrics["plan_origin_free_fraction"] == 1.0
    assert metrics["free_origin_plan_count"] == 2
    assert metrics["future_plan_collision_fraction"] == 1.0
    assert metrics["future_execution_prefix_0p5m_collision_fraction"] == 0.0
    assert metrics["future_execution_prefix_1p0m_collision_fraction"] == 1.0
    assert metrics["first_plan_execution_prefix_0p5m_collision"] is False
    assert metrics["first_plan_execution_prefix_1p0m_collision"] is True
    assert metrics["adjacent_plan_pairs"] == 1
    assert metrics["adjacent_plan_first1m_world_disagreement_m_mean"] < 1e-12


def test_closed_loop_diagnostics_do_not_count_the_current_origin_as_future() -> None:
    free = np.ones((40, 40), dtype=np.bool_)
    free[0, 0] = False
    navigation = NavigationMap(free=free, extent=(0.0, 2.0, 0.0, 2.0))
    plan = np.column_stack((np.linspace(0.025, 0.50, 64), np.full(64, 0.025)))

    metrics = _closed_loop_plan_diagnostics([plan], [plan], navigation)

    assert metrics["plan_origin_free_fraction"] == 0.0
    assert metrics["free_origin_plan_count"] == 0
    assert metrics["future_plan_collision_fraction"] == 0.0
    assert metrics["future_execution_prefix_0p5m_collision_fraction"] == 0.0


def test_closed_loop_summary_weights_diagnostics_by_plans() -> None:
    records = [
        {
            "success": True,
            "total_plan_count": 1,
            "free_origin_plan_count": 1,
            "adjacent_plan_pairs": 0,
            "plan_origin_free_fraction": 1.0,
            "future_plan_collision_fraction": 0.0,
            "future_execution_prefix_0p5m_collision_fraction": 0.0,
            "future_execution_prefix_1p0m_collision_fraction": 0.0,
            "future_execution_prefix_0p5m_collision_given_free_origin_fraction": 0.0,
            "future_execution_prefix_1p0m_collision_given_free_origin_fraction": 0.0,
            "first_plan_execution_prefix_0p5m_collision": False,
            "first_plan_execution_prefix_1p0m_collision": False,
            "future_planned_point_free_fraction": 1.0,
            "actual_position_free_fraction": 1.0,
            "mpc_desired_speed_mps_mean": 0.5,
            "mpc_curvature_limited_fraction": 0.0,
            "adjacent_plan_first1m_world_disagreement_m_mean": 0.0,
            "stalled_step_fraction": 0.0,
        },
        {
            "success": False,
            "total_plan_count": 3,
            "free_origin_plan_count": 0,
            "adjacent_plan_pairs": 2,
            "plan_origin_free_fraction": 0.0,
            "future_plan_collision_fraction": 1.0,
            "future_execution_prefix_0p5m_collision_fraction": 1.0,
            "future_execution_prefix_1p0m_collision_fraction": 1.0,
            "future_execution_prefix_0p5m_collision_given_free_origin_fraction": 0.0,
            "future_execution_prefix_1p0m_collision_given_free_origin_fraction": 0.0,
            "first_plan_execution_prefix_0p5m_collision": True,
            "first_plan_execution_prefix_1p0m_collision": True,
            "future_planned_point_free_fraction": 0.0,
            "actual_position_free_fraction": 0.5,
            "mpc_desired_speed_mps_mean": 0.1,
            "mpc_curvature_limited_fraction": 1.0,
            "adjacent_plan_first1m_world_disagreement_m_mean": 0.2,
            "stalled_step_fraction": 0.5,
        },
    ]

    summary = _summarize_closed_loop(records)

    assert summary["successes"] == 1
    assert summary["free_origin_plans"] == 1
    assert summary["future_plan_collision_fraction"] == pytest.approx(0.75)
    assert summary[
        "future_execution_prefix_1p0m_collision_given_free_origin_fraction"
    ] == 0.0
    assert summary["mpc_desired_speed_mps_plan_mean"] == pytest.approx(0.2)
    assert summary["adjacent_plan_first1m_world_disagreement_m_mean"] == 0.2


def test_closed_loop_map_renders_trace_and_scene_overview(tmp_path) -> None:
    pytest.importorskip("matplotlib")
    trace_path = tmp_path / "episode-000.npz"
    positions = np.array([[0.0, 0.0, 0.0], [0.2, 0.0, 0.0]])
    quaternions = np.array([[0.0, 0.0, 0.0, 1.0]] * 2)
    np.savez(
        trace_path,
        episode_idx=np.array(0),
        success=np.array(True),
        step_robot_position_world_m=positions,
        terminal_pre_step_robot_position_world_m=np.array([0.4, 0.0, 0.0]),
        step_planar_speed_mps=np.array([0.2, 0.2]),
        step_mpc_command=np.array([[0.3, 0.0], [0.3, 0.0]]),
        step_point_goal_robot_m=np.array([[1.0, 0.0, 0.0]] * 2),
        step_robot_quaternion_xyzw=quaternions,
        plan_robot_position_world_m=positions[:1],
        plan_robot_quaternion_xyzw=quaternions[:1],
        plan_local_trajectory=np.array([[[0.0, 0.0], [0.2, 0.0], [0.4, 0.0]]]),
        plan_local_trajectory_length=np.array([3]),
        terminal_pre_step_point_goal_robot_m=np.array([0.6, 0.0, 0.0]),
        executed_path_length_m=np.array(0.4),
    )
    navigation_map = _rasterize_navigation_map(
        np.array([[-1.0, -1.0], [0.0, 0.0], [1.0, 1.0]])
    )
    episode_image = tmp_path / "episode-000.png"
    overview_image = tmp_path / "scene-overview.png"
    episode = _render_episode(trace_path, navigation_map, episode_image)
    overview = _render_scene_overview([trace_path], navigation_map, overview_image)
    assert episode_image.stat().st_size > 0
    assert overview_image.stat().st_size > 0
    assert episode["success"]
    assert overview["success_count"] == 1
