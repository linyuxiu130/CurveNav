import torch

from curvenav.evaluation.offline import (
    observed_obstacle_metrics,
    summarize_policy_metrics,
    trajectory_batch_metrics,
)
from curvenav.physical import EXTRA_CLEARANCE_M, ROBOT_FOOTPRINT_RADIUS_M


def test_deterministic_trajectory_metrics() -> None:
    target = torch.tensor([[[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]]])
    path = target.clone()
    curvature = torch.zeros(1, 3)
    metrics = trajectory_batch_metrics(
        path,
        curvature,
        target,
        torch.zeros(1, 3),
        torch.tensor([[3.0, 0.0]]),
    )
    assert metrics["ade_m"].item() == 0
    assert metrics["goal_progress_m"].item() == 2


def test_validation_ade_is_grouped_by_available_observation_frames() -> None:
    metrics = {
        "ade_m": torch.tensor([0.3, 0.1, 0.2]),
        "rmse_m": torch.ones(3),
        "arc_length_m": torch.ones(3),
        "arc_length_error_m": torch.zeros(3),
        "reference_arc_length_m": torch.ones(3),
        "goal_progress_m": torch.ones(3),
        "reference_goal_progress_m": torch.ones(3),
        "max_abs_curvature_inv_m": torch.tensor([0.4, 0.8, 1.2]),
        "reference_max_abs_curvature_inv_m": torch.tensor([0.5, 1.0, 2.0]),
        "total_abs_heading_change_rad": torch.tensor([0.2, 0.3, 0.8]),
        "reference_total_abs_heading_change_rad": torch.tensor([0.1, 0.4, 1.0]),
        "terminal_heading_error_rad": torch.tensor([0.1, 0.2, 0.3]),
        "has_tangent_reversal": torch.zeros(3, dtype=torch.bool),
        "valid_observation_frames": torch.tensor([1, 4, 1]),
        "point_goal_distance_m": torch.tensor([2.0, 4.0, 9.0]),
        "point_goal_is_behind": torch.tensor([False, True, False]),
    }

    summary = summarize_policy_metrics(metrics)

    assert summary["samples_with_1_frames"] == 2
    assert summary["ade_m_with_1_frames"] == torch.tensor(0.25).item()
    assert summary["samples_with_4_frames"] == 1
    assert summary["ade_m_with_4_frames"] == torch.tensor(0.1).item()
    assert summary["high_turn_samples"] == 1
    assert summary["ade_m_high_turn_10pct"] == torch.tensor(0.2).item()
    assert summary["samples_point_goal_lt_3m"] == 1
    assert summary["samples_point_goal_3_to_6m"] == 1
    assert summary["samples_point_goal_ge_8_5m"] == 1
    assert summary["point_goal_behind_samples"] == 1
    assert summary["ade_m_point_goal_behind"] == torch.tensor(0.1).item()


def test_observed_obstacle_metrics_use_robot_configuration_space() -> None:
    path = torch.tensor(
        [
            [[0.0, 0.0], [1.0, 0.0]],
            [[0.0, 0.0], [1.0, 0.0]],
            [[0.0, 0.0], [1.0, 0.0]],
        ]
    )
    obstacle = torch.tensor(
        [
            [[1.0, ROBOT_FOOTPRINT_RADIUS_M * 0.5]],
            [[1.0, ROBOT_FOOTPRINT_RADIUS_M + EXTRA_CLEARANCE_M * 0.5]],
            [[0.0, 0.0]],
        ]
    )
    valid = torch.tensor([[True], [True], [False]])

    metrics = observed_obstacle_metrics(path, obstacle, valid)

    assert metrics["observed_obstacle_available"].tolist() == [True, True, False]
    assert metrics["observed_footprint_collision"].tolist() == [True, False, False]
    assert metrics["observed_safety_margin_violation"].tolist() == [True, True, False]
    assert torch.isinf(metrics["observed_min_clearance_m"][-1])
