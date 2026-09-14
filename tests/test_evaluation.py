from pathlib import Path

import numpy as np
import pytest
import torch

from curvenav.evaluation.compare import (
    CommonSourceConfiguration,
    load_common_protocol,
    validate_reference_safety,
)
from curvenav.evaluation.metrics import (
    collision_visibility_attribution,
    configuration_space_safety_metrics,
    controller_tracking_metrics,
    paired_path_change_m,
    resample_path_at_distance,
    source_execution_prefix_metrics,
    summarize_metrics,
    trajectory_metrics,
)
from curvenav.data.privileged import SourcePathQuery
from curvenav.evaluation.offline import (
    _collision_detection_summary,
    _cached_interventions,
)
from curvenav.evaluation.report import _write_case_visualization, select_cases
from curvenav.evaluation.protocol import evaluation_strata
from curvenav.trajectory import resample_path_to_horizon


def test_cross_model_set_requires_explicit_axis_and_source_geometry(
    tmp_path: Path,
) -> None:
    path = tmp_path / "common.npz"
    np.savez(path, point_goal=np.zeros((1, 2), dtype=np.float32))
    common = np.load(path, allow_pickle=False)

    with pytest.raises(ValueError, match="lacks metric protocol"):
        load_common_protocol(common)


@pytest.mark.parametrize("batch", [1, 3, 4])
@torch.no_grad()
def test_cached_interventions_match_raw_input_permutations(batch):
    from dataclasses import fields, replace
    from test_policy import tiny_config, make_condition
    from curvenav.factory import build_policy

    policy = build_policy(tiny_config()).eval()
    condition = make_condition(batch)
    permutation = torch.arange(batch).roll(batch // 2)
    depth_swapped = replace(condition, **{
        f.name: getattr(condition, f.name)[permutation]
        for f in fields(condition) if f.name != "point_goal"
    })
    goal_swapped = replace(condition, point_goal=condition.point_goal[permutation])
    actual = _cached_interventions(policy, policy.encode_condition(condition), condition.point_goal)
    for prediction, raw in zip(actual, (depth_swapped, goal_swapped)):
        expected = policy.sample(raw)
        for f in fields(expected):
            torch.testing.assert_close(getattr(prediction, f.name), getattr(expected, f.name))


def test_cross_model_safety_queries_frozen_source_grid(tmp_path: Path) -> None:
    route = "validation/dataset/route"
    grid_root = tmp_path / "source/validation/dataset"
    grid_root.mkdir(parents=True)
    free = np.ones((9, 9), dtype=np.bool_)
    free[6, 4] = False
    clearance = np.ones((9, 9), dtype=np.float32)
    clearance[~free] = 0.0
    np.savez(
        grid_root / "navigation_grid.npz",
        free=free,
        clearance_m=clearance,
        origin_xy=np.asarray([-1.0, -1.0]),
        cell_size_m=np.asarray(0.25),
    )
    common_path = tmp_path / "common.npz"
    np.savez(
        common_path,
        route_id=np.asarray([route]),
        origin_xy=np.zeros((1, 2), dtype=np.float32),
        route_yaw=np.zeros(1, dtype=np.float32),
    )
    common = np.load(common_path, allow_pickle=False)
    safety = CommonSourceConfiguration(common, tmp_path / "source")

    metrics = safety.measure(
        torch.tensor([[[0.0, 0.0], [0.5, 0.0], [1.0, 0.0]]]),
        horizon_m=1.0,
    )

    assert metrics["footprint_collision"].item()
    assert metrics["path_field_coverage_fraction"].item() == 1.0

    stationary = safety.measure(torch.zeros(1, 3, 2), horizon_m=1.0)
    assert not stationary["footprint_collision"].item()

    outside = safety.measure(
        torch.tensor([[[0.0, 0.0], [0.0, 2.0], [0.0, 3.0]]]),
        horizon_m=3.0,
    )
    assert outside["footprint_collision"].item()
    assert outside["path_field_coverage_fraction"].item() < 1.0

    with pytest.raises(ValueError, match="do not match the frozen source geometry"):
        validate_reference_safety(
            torch.tensor([[[0.0, 0.0], [0.5, 0.0], [1.0, 0.0]]]),
            safety,
            planning_horizon_m=1.0,
        )
    contract = validate_reference_safety(
        torch.zeros(1, 3, 2), safety, planning_horizon_m=1.0
    )
    assert contract["expert_path_field_coverage_fraction"] == 1.0


def test_fixed_distance_metrics_match_identical_paths() -> None:
    target = torch.tensor([[[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]]])
    metrics = trajectory_metrics(
        target,
        target,
        torch.tensor([[3.0, 0.0]]),
    )

    assert metrics["fixed_horizon_ade_m"].item() == 0
    assert metrics["fixed_horizon_fde_m"].item() == 0
    assert metrics["goal_progress_m"].item() == 2
    assert metrics["horizon_coverage_fraction"].item() == 1


def test_short_prediction_is_held_to_score_the_missing_horizon() -> None:
    reference = torch.tensor([[[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]]])
    short = torch.tensor([[[0.0, 0.0], [0.5, 0.0], [1.0, 0.0]]])
    metrics = trajectory_metrics(short, reference, torch.tensor([[3.0, 0.0]]))

    assert metrics["horizon_coverage_fraction"].item() == 0.5
    assert metrics["fixed_horizon_fde_m"].item() == 1.0
    assert metrics["fixed_horizon_ade_m"].item() > 0.25

    change = paired_path_change_m(short, reference)
    torch.testing.assert_close(change, metrics["fixed_horizon_ade_m"])


def test_resampling_uses_physical_arc_distance() -> None:
    path = torch.tensor([[[0.0, 0.0], [0.25, 0.0], [2.0, 0.0]]])
    sampled = resample_path_at_distance(path, torch.tensor([[0.0, 1.0, 3.0]]))

    assert torch.allclose(sampled[0, :, 0], torch.tensor([0.0, 1.0, 2.0]))


def test_physical_horizon_sampler_masks_short_endpoint_repetitions() -> None:
    points, active = resample_path_to_horizon(
        torch.tensor([[[0.0, 0.0], [0.05, 0.0]]]),
        horizon_m=0.10,
        spacing_m=0.025,
    )

    assert active.tolist() == [[True, True, True, False, False]]
    assert torch.allclose(points[0, 2:], torch.tensor([[0.05, 0.0]]).expand(3, -1))


def test_physical_horizon_sampler_includes_a_non_grid_terminal_endpoint() -> None:
    points, active = resample_path_to_horizon(
        torch.tensor([[[0.0, 0.0], [0.055, 0.0]]]),
        horizon_m=0.10,
        spacing_m=0.025,
    )

    assert active.tolist() == [[True, True, True, True, False]]
    torch.testing.assert_close(points[0, 3], torch.tensor([0.055, 0.0]))


def test_summary_groups_available_observation_frames() -> None:
    reference = torch.tensor(
        [
            [[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]],
            [[0.0, 0.0], [0.8, 0.2], [1.5, 0.8]],
            [[0.0, 0.0], [0.7, -0.3], [1.2, -1.0]],
        ]
    )
    predicted = reference.clone()
    predicted[0, :, 1] += 0.1
    metrics = trajectory_metrics(
        predicted,
        reference,
        torch.tensor([[3.0, 0.0], [3.0, 1.0], [3.0, -1.0]]),
    )
    metrics["valid_observation_frames"] = torch.tensor([1, 4, 1])

    summary = summarize_metrics(metrics)

    assert summary["samples_with_1_frames"] == 2
    assert summary["samples_with_4_frames"] == 1
    assert summary["high_turn_samples"] == 1


def test_configuration_safety_uses_robot_configuration_space() -> None:
    path = torch.tensor(
        [
            [[0.0, 0.0], [0.5, 0.0], [1.0, 0.0]],
            [[0.0, 0.0], [0.5, 0.0], [1.0, 0.0]],
            [[0.0, 0.0], [0.5, 0.0], [1.0, 0.0]],
        ]
    )
    field = torch.zeros(3, 5, 9, 9)
    field[:, 0] = torch.tensor([-0.05, 0.2, 1.0])[:, None, None]
    field[:2, 3] = 1.0

    metrics = configuration_space_safety_metrics(path, field, 1.0)

    assert metrics["footprint_collision"].tolist() == [True, False, False]
    assert metrics["safety_margin_violation"].tolist() == [True, True, False]
    assert metrics["path_field_coverage_fraction"].tolist() == [1.0, 1.0, 0.0]
    assert torch.isinf(metrics["min_clearance_m"][-1])


def test_source_safety_reports_the_executed_prefix_without_another_query() -> None:
    local = torch.tensor(
        [
            [[0.0, 0.0], [0.25, 0.0], [0.50, 0.0], [0.75, 0.0], [1.0, 0.0]],
            [[0.0, 0.0], [0.25, 0.0], [0.50, 0.0], [0.75, 0.0], [1.0, 0.0]],
        ]
    )
    query = SourcePathQuery(
        local_path=local,
        grid_cells=torch.zeros_like(local, dtype=torch.long),
        clearance_m=torch.tensor(
            [[0.2, 0.2, -0.01, -0.02, -0.02], [0.2, 0.2, 0.2, 0.05, 0.2]]
        ),
        in_world_bounds=torch.ones(2, 5, dtype=torch.bool),
        active=torch.ones(2, 5, dtype=torch.bool),
    )

    metrics = source_execution_prefix_metrics(query)

    torch.testing.assert_close(
        metrics["distance_to_first_collision_m"], torch.tensor([0.5, 1.0])
    )
    torch.testing.assert_close(
        metrics["distance_to_first_margin_violation_m"], torch.tensor([0.5, 0.75])
    )
    assert metrics["execution_prefix_0p5m_collision"].tolist() == [True, False]
    assert metrics["execution_prefix_1p0m_collision"].tolist() == [True, False]
    assert metrics["execution_prefix_0p5m_margin_violation"].tolist() == [True, False]
    assert metrics["execution_prefix_1p0m_margin_violation"].tolist() == [True, True]
    assert metrics["path_endpoint_collision"].tolist() == [True, False]
    assert metrics["terminal_0p25m_collision"].tolist() == [True, False]
    assert metrics["collision_confined_to_terminal_0p25m"].tolist() == [False, False]


def test_controller_metrics_match_the_fixed_benchmark_speed_law() -> None:
    straight = torch.stack((torch.linspace(0.0, 2.0, 64), torch.zeros(64)), dim=-1)
    angle = torch.linspace(0.0, torch.pi / 2.0, 64)
    sharp = torch.stack((0.25 * angle.sin(), 0.25 * (1.0 - angle.cos())), dim=-1)

    metrics = controller_tracking_metrics(torch.stack((straight, sharp)))

    torch.testing.assert_close(
        metrics["mpc_max_curvature_lookahead_inv_m"], torch.tensor([0.0, 4.0]),
        atol=0.001, rtol=0.001,
    )
    torch.testing.assert_close(
        metrics["mpc_curvature_limited_speed_mps"], torch.tensor([0.5, 0.125]),
        atol=0.0001, rtol=0.001,
    )
    # This short quarter circle slows down for its remaining length first.
    expected_speed = 0.5 * torch.linalg.vector_norm(sharp.diff(dim=0), dim=-1).sum() / 2
    torch.testing.assert_close(metrics["mpc_desired_speed_mps"][1], expected_speed)
    stopped = controller_tracking_metrics(torch.zeros(1, 64, 2))
    assert stopped["mpc_desired_speed_mps"].item() == 0


def test_collision_detection_separates_perception_from_generation() -> None:
    metrics = {
        "truth_collision_point_count": torch.tensor([2, 1, 0, 0]),
        "truth_collision_point_raw_depth_count": torch.tensor([1, 0, 0, 0]),
        "truth_collision_point_raw_ray_coverage_count": torch.tensor([2, 0, 0, 0]),
        "truth_collision_point_raw_ray_coverage_missed_count": torch.tensor(
            [1, 0, 0, 0]
        ),
        "truth_collision_point_outside_raw_ray_coverage_count": torch.tensor(
            [0, 1, 0, 0]
        ),
        "raw_depth_false_collision_point_count": torch.tensor([0, 0, 0, 0]),
        "truth_collision_trajectory": torch.tensor([True, True, False, False]),
        "truth_collision_trajectory_recognized_by_raw_depth": torch.tensor(
            [True, False, False, False]
        ),
        "truth_collision_point_current_depth_count": torch.tensor([0, 0, 0, 0]),
        "truth_collision_point_history_only_depth_count": torch.tensor([1, 0, 0, 0]),
        "truth_collision_point_unrecognized_by_full_depth_count": torch.tensor(
            [1, 1, 0, 0]
        ),
        "truth_collision_trajectory_recognized_by_current_depth": torch.tensor(
            [False, False, False, False]
        ),
        "truth_collision_trajectory_has_additional_history_evidence": torch.tensor(
            [True, False, False, False]
        ),
        "truth_collision_trajectory_recognized_only_with_history": torch.tensor(
            [True, False, False, False]
        ),
        "truth_collision_trajectory_unrecognized_by_full_depth": torch.tensor(
            [False, True, False, False]
        ),
        "first_collision_current_depth_visible": torch.tensor(
            [False, False, False, False]
        ),
        "first_collision_history_only_depth_visible": torch.tensor(
            [True, False, False, False]
        ),
        "first_collision_unrecognized_by_full_depth": torch.tensor(
            [False, True, False, False]
        ),
        "execution_prefix_1p0m_collision": torch.tensor([True, True, False, False]),
        "path_endpoint_collision": torch.tensor([False, True, False, False]),
        "terminal_0p25m_collision": torch.tensor([True, True, False, False]),
        "collision_confined_to_terminal_0p25m": torch.tensor(
            [True, False, False, False]
        ),
        "path_endpoint_collision_current_depth_visible": torch.tensor(
            [False, False, False, False]
        ),
        "path_endpoint_collision_history_only_depth_visible": torch.tensor(
            [False, True, False, False]
        ),
        "path_endpoint_collision_unrecognized_by_full_depth": torch.tensor(
            [False, False, False, False]
        ),
    }

    summary = _collision_detection_summary(metrics)

    assert summary["ground_truth_collision_trajectory_count"] == 2
    assert summary["ground_truth_collision_point_count"] == 3
    assert summary["ground_truth_collision_points_recognized_by_raw_depth_count"] == 1
    assert summary[
        "ground_truth_collision_points_recognized_by_raw_depth_fraction"
    ] == pytest.approx(1 / 3)
    assert summary[
        "ground_truth_collision_points_inside_raw_ray_coverage_fraction"
    ] == pytest.approx(2 / 3)
    assert (
        summary["ground_truth_collision_points_raw_ray_coverage_but_missed_count"] == 1
    )
    assert summary["collision_points_visible_only_through_history_count"] == 1
    assert summary["collision_trajectories_recognized_only_through_history_count"] == 1
    assert summary["first_collision_visible_only_through_history_count"] == 1
    assert summary["first_collision_unrecognized_by_all_history_frames_count"] == 1
    assert summary["execution_prefix_1p0m_collision_count"] == 2
    assert summary[
        "first_collision_within_1p0m_visible_only_through_history_fraction"
    ] == pytest.approx(0.5)
    assert summary[
        "first_collision_within_1p0m_unrecognized_by_all_history_frames_fraction"
    ] == pytest.approx(0.5)
    assert summary["path_endpoint_collision_count"] == 1
    assert summary["endpoint_collision_point_visible_only_through_history_count"] == 1


def test_collision_attribution_separates_current_history_and_unseen_points() -> None:
    local_path = torch.tensor(
        [
            [[0.0, 0.0], [0.5, 0.0], [1.0, 0.0]],
            [[0.0, 0.0], [0.5, 0.0], [1.0, 0.0]],
        ]
    )
    full = torch.zeros(2, 5, 9, 9)
    current = torch.zeros_like(full)
    full[:, 0] = 1.0
    current[:, 0] = 1.0
    full[:, 3] = 1.0
    current[:, 3] = 1.0
    y, x = torch.meshgrid(
        torch.linspace(-1, 1, 9), torch.linspace(-1, 1, 9), indexing="ij"
    )
    # A metric distance field must obey the 1-Lipschitz distance invariant.
    obstacle_distance = torch.sqrt((x - 0.5).square() + y.square()) - 0.05
    full[:, 0] = obstacle_distance
    current[1, 0] = obstacle_distance
    source_query = SourcePathQuery(
        local_path=local_path,
        grid_cells=torch.zeros_like(local_path, dtype=torch.long),
        clearance_m=torch.tensor([[1.0, -1.0, 1.0], [1.0, -1.0, -1.0]]),
        in_world_bounds=torch.ones(2, 3, dtype=torch.bool),
        active=torch.ones(2, 3, dtype=torch.bool),
    )

    attribution = collision_visibility_attribution(source_query, full, current, 1.0)
    metrics = attribution.metrics

    assert metrics["truth_collision_point_count"].tolist() == [1, 2]
    assert metrics["truth_collision_point_current_depth_count"].tolist() == [0, 1]
    assert metrics["truth_collision_point_history_only_depth_count"].tolist() == [1, 0]
    assert metrics[
        "truth_collision_point_unrecognized_by_full_depth_count"
    ].tolist() == [0, 1]
    assert metrics[
        "truth_collision_trajectory_recognized_only_with_history"
    ].tolist() == [True, False]
    assert metrics["first_collision_history_only_depth_visible"].tolist() == [
        True,
        False,
    ]
    assert metrics["first_collision_current_depth_visible"].tolist() == [
        False,
        True,
    ]
    assert metrics["first_collision_unrecognized_by_full_depth"].tolist() == [
        False,
        False,
    ]
    assert attribution.history_only_visible_points[0, 1]
    assert attribution.unrecognized_points[1, 2]
    assert metrics["path_endpoint_collision_unrecognized_by_full_depth"].tolist() == [
        False,
        True,
    ]


def test_case_selection_includes_the_worst_source_collision() -> None:
    metrics = {
        "point_goal_is_behind": torch.tensor([False] * 4),
        "reference_safety_margin_violation": torch.tensor([False] * 4),
        "straight_safety_margin_violation": torch.tensor([False] * 4),
        "reference_goal_progress_m": torch.ones(4),
        "fixed_horizon_ade_m": torch.arange(4, dtype=torch.float32),
        "footprint_collision": torch.tensor([True, True, False, False]),
        "min_clearance_m": torch.tensor([-0.2, -0.3, 0.2, 0.3]),
        "execution_prefix_1p0m_collision": torch.tensor([False, False, True, False]),
        "distance_to_first_collision_m": torch.tensor([0.25, 1.5, 0.10, 2.0]),
        "first_collision_current_depth_visible": torch.tensor(
            [True, False, False, False]
        ),
        "first_collision_history_only_depth_visible": torch.tensor(
            [False, True, False, False]
        ),
        "first_collision_unrecognized_by_full_depth": torch.tensor(
            [False, False, False, False]
        ),
        "path_endpoint_collision": torch.tensor([False, True, False, False]),
    }

    selected = dict(select_cases(metrics))

    assert selected["current_visible_collision"] == 0
    assert selected["history_only_visible_collision"] == 1


def test_case_visualization_is_one_dependency_free_metric_svg(tmp_path: Path) -> None:
    record = {
        "label": "forward_detour_hard",
        "configuration_extent_m": 1.0,
        "configuration_ray_coverage": [[True, True], [False, True]],
        "current_configuration_ray_coverage": [[True, False], [False, True]],
        "configuration_forbidden": [[False, True], [False, False]],
        "current_configuration_forbidden": [[False, False], [False, False]],
        "source_clearance_m": [[0.2, -0.1], [0.05, 0.2]],
        "reference_path": [[0.0, 0.0], [0.8, 0.2]],
        "predicted_path": [[0.0, 0.0], [0.8, -0.2]],
        "current_frame_predicted_path": [[0.0, 0.0], [0.7, -0.1]],
        "collision_points_current_visible": [[0.8, -0.2]],
        "collision_points_history_only_visible": [],
        "collision_points_unrecognized": [],
        "current_visible_collision_points": 1,
        "history_only_visible_collision_points": 0,
        "unrecognized_collision_points": 0,
        "point_goal": [2.0, 0.0],
        "fixed_horizon_ade_m": 0.2,
        "predicted_min_clearance_m": -0.1,
        "footprint_collision": True,
        "distance_to_first_collision_m": 0.75,
        "path_endpoint_collision": True,
    }
    path = tmp_path / "offline-cases.svg"

    _write_case_visualization([record], path)

    text = path.read_text(encoding="utf-8")
    assert text.startswith('<svg id="curvenav-collision-bev"')
    assert "forward detour hard" in text
    assert "#16a34a" in text
    assert "#2563eb" in text
    assert "history obstacle" in text
    assert "unseen hit" in text
    assert "first hit 0.75m" in text
    assert 'transform="translate(12,12)"' not in text


def test_safety_sampling_detects_obstacle_between_output_points() -> None:
    path = torch.tensor([[[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]]])
    field = torch.zeros(1, 5, 9, 9)
    field[:, 0] = 1.0
    y, x = torch.meshgrid(
        torch.linspace(-2, 2, 9), torch.linspace(-2, 2, 9), indexing="ij"
    )
    field[:, 0] = torch.sqrt((x - 0.5).square() + y.square()) - 0.05
    field[:, 3] = 1.0

    metrics = configuration_space_safety_metrics(path, field, 2.0)

    assert metrics["footprint_collision"].item()


def test_strata_separate_visible_detours_and_rear_goals() -> None:
    metrics = {
        "point_goal_is_behind": torch.tensor([False, False, True]),
        "reference_safety_margin_violation": torch.tensor([False, False, False]),
        "straight_safety_margin_violation": torch.tensor([False, True, True]),
        "reference_goal_progress_m": torch.tensor([1.0, 0.5, -0.1]),
    }

    strata = evaluation_strata(metrics)

    assert strata["forward_direct"].tolist() == [True, False, False]
    assert strata["forward_detour"].tolist() == [False, True, False]
    assert strata["rear_goal"].tolist() == [False, False, True]
    assert strata["expert_moves_away_from_goal"].tolist() == [False, False, True]
