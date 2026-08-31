import numpy as np
import pytest

from curvenav.evaluation.closed_loop_map import (
    _draw_navigation_map,
    _plan_indices,
    _rasterize_navigation_map,
    _render_episode,
    _render_scene_overview,
    _robot_to_world,
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
    assert axis.images[0].cmap.colors == ["#4a1717", "#e5e7eb"]
    assert axis.get_xlim() == (-2.0, 3.0)
    assert axis.get_ylim() == (-1.0, 4.0)
    assert [item.get_label() for item in legend] == [
        "navigable robot-center space",
        "obstacle / non-navigable space",
    ]
    plt.close(figure)


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
