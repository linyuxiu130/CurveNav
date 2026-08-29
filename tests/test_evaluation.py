import torch

from curvenav.evaluation.offline import (
    observed_obstacle_metrics,
    summarize_policy_metrics,
    trajectory_batch_metrics,
)
from curvenav.evaluation.protocol import evaluation_strata
def test_deterministic_trajectory_metrics() -> None:
    target = torch.tensor([[[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]]])
    path = target.clone()
    metrics = trajectory_batch_metrics(
        path,
        target,
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
    field = torch.zeros(3, 5, 9, 9)
    field[:, 0] = torch.tensor([-0.05, 0.05, 1.0])[:, None, None]
    field[:2, 3] = 1.0
    field[:2, 4, 4, 4] = 1.0

    metrics = observed_obstacle_metrics(path, field, 1.0)

    assert metrics["observed_obstacle_available"].tolist() == [True, True, False]
    assert metrics["observed_footprint_collision"].tolist() == [True, False, False]
    assert metrics["observed_safety_margin_violation"].tolist() == [True, True, False]
    assert torch.isinf(metrics["observed_min_clearance_m"][-1])


def test_obstacle_clearance_covers_dense_path_samples() -> None:
    path = torch.tensor([[[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]]])
    field = torch.zeros(1, 5, 9, 9)
    field[:, 0] = 1.0
    field[:, 0, 4, 6] = -0.05
    field[:, 3] = 1.0
    field[:, 4, 4, 6] = 1.0
    metrics = observed_obstacle_metrics(path, field, 2.0)

    assert metrics["observed_footprint_collision"].item()


def test_strict_strata_separate_visible_detours_and_rear_goals() -> None:
    metrics = {
        "point_goal_is_behind": torch.tensor([False, False, True]),
        "observed_obstacle_available": torch.tensor([True, True, True]),
        "reference_observed_safety_margin_violation": torch.tensor(
            [False, False, False]
        ),
        "straight_observed_safety_margin_violation": torch.tensor(
            [False, True, True]
        ),
        "reference_goal_progress_m": torch.tensor([1.0, 0.5, -0.1]),
    }

    strata = evaluation_strata(metrics)

    assert strata["forward_open"].tolist() == [True, False, False]
    assert strata["forward_visible_detour"].tolist() == [False, True, False]
    assert strata["rear_goal"].tolist() == [False, False, True]
    assert strata["expert_moves_away_from_goal"].tolist() == [False, False, True]
