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
    configuration_space_collision_attribution,
    configuration_space_safety_metrics,
    resample_path_at_distance,
    summarize_metrics,
    trajectory_metrics,
)
from curvenav.data.privileged import SourcePathQuery
from curvenav.evaluation.offline import (
    _collision_detection_summary,
)
from curvenav.evaluation.report import select_cases
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
    field[:, 0] = torch.tensor([-0.05, 0.05, 1.0])[:, None, None]
    field[:2, 3] = 1.0

    metrics = configuration_space_safety_metrics(path, field, 1.0)

    assert metrics["footprint_collision"].tolist() == [True, False, False]
    assert metrics["safety_margin_violation"].tolist() == [True, True, False]
    assert metrics["path_field_coverage_fraction"].tolist() == [1.0, 1.0, 0.0]
    assert torch.isinf(metrics["min_clearance_m"][-1])


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
    }

    summary = _collision_detection_summary(metrics)

    assert summary["ground_truth_collision_trajectory_count"] == 2
    assert summary["ground_truth_collision_point_count"] == 3
    assert summary[
        "ground_truth_collision_points_recognized_by_raw_depth_count"
    ] == 1
    assert summary[
        "ground_truth_collision_points_recognized_by_raw_depth_fraction"
    ] == pytest.approx(1 / 3)
    assert summary[
        "ground_truth_collision_points_inside_raw_ray_coverage_fraction"
    ] == pytest.approx(2 / 3)
    assert summary[
        "ground_truth_collision_points_raw_ray_coverage_but_missed_count"
    ] == 1


def test_collision_attribution_queries_identical_path_points() -> None:
    local_path = torch.tensor([[[0.0, 0.0], [0.5, 0.0], [1.0, 0.0]]])
    raw = torch.zeros(1, 5, 9, 9)
    raw[:, 0] = 1.0
    raw[:, 3] = 1.0
    raw[:, 0, 4, 6] = -1.0
    source_query = SourcePathQuery(
        local_path=local_path,
        clearance_m=torch.tensor([[1.0, -1.0, 1.0]]),
        in_world_bounds=torch.ones(1, 3, dtype=torch.bool),
        active=torch.ones(1, 3, dtype=torch.bool),
    )

    metrics = configuration_space_collision_attribution(source_query, raw, 1.0)

    assert metrics["truth_collision_point_count"].item() > 0
    assert torch.equal(
        metrics["truth_collision_point_count"],
        metrics["truth_collision_point_raw_depth_count"],
    )
    assert metrics["truth_collision_trajectory_recognized_by_raw_depth"].item()


def test_case_selection_includes_the_worst_source_collision() -> None:
    metrics = {
        "point_goal_is_behind": torch.tensor([False] * 4),
        "reference_safety_margin_violation": torch.tensor([False] * 4),
        "straight_safety_margin_violation": torch.tensor([False] * 4),
        "reference_goal_progress_m": torch.ones(4),
        "fixed_horizon_ade_m": torch.arange(4, dtype=torch.float32),
        "footprint_collision": torch.tensor([True, True, False, False]),
        "min_clearance_m": torch.tensor([-0.2, -0.3, 0.2, 0.3]),
    }

    selected = dict(select_cases(metrics))

    assert selected["source_collision_worst"] == 1


def test_safety_sampling_detects_obstacle_between_output_points() -> None:
    path = torch.tensor([[[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]]])
    field = torch.zeros(1, 5, 9, 9)
    field[:, 0] = 1.0
    field[:, 0, 4, 6] = -0.05
    field[:, 3] = 1.0

    metrics = configuration_space_safety_metrics(path, field, 2.0)

    assert metrics["footprint_collision"].item()


def test_strata_separate_visible_detours_and_rear_goals() -> None:
    metrics = {
        "point_goal_is_behind": torch.tensor([False, False, True]),
        "reference_safety_margin_violation": torch.tensor(
            [False, False, False]
        ),
        "straight_safety_margin_violation": torch.tensor(
            [False, True, True]
        ),
        "reference_goal_progress_m": torch.tensor([1.0, 0.5, -0.1]),
    }

    strata = evaluation_strata(metrics)

    assert strata["forward_direct"].tolist() == [True, False, False]
    assert strata["forward_detour"].tolist() == [False, True, False]
    assert strata["rear_goal"].tolist() == [False, False, True]
    assert strata["expert_moves_away_from_goal"].tolist() == [False, False, True]
