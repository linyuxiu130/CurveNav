from pathlib import Path

import numpy as np
import pytest
import torch

from curvenav.evaluation.compare import load_common_geometry
from curvenav.evaluation.metrics import (
    observed_safety_metrics,
    resample_path_at_distance,
    summarize_metrics,
    trajectory_metrics,
)
from curvenav.evaluation.protocol import evaluation_strata


def test_cross_model_set_requires_frozen_configuration_geometry(
    tmp_path: Path,
) -> None:
    path = tmp_path / "common.npz"
    np.savez(path, point_goal=np.zeros((1, 2), dtype=np.float32))
    common = np.load(path, allow_pickle=False)

    with pytest.raises(ValueError, match="lacks frozen metric geometry"):
        load_common_geometry(common, 3.6)


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


def test_observed_safety_uses_robot_configuration_space() -> None:
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

    metrics = observed_safety_metrics(path, field, 1.0)

    assert metrics["observed_footprint_collision"].tolist() == [True, False, False]
    assert metrics["observed_safety_margin_violation"].tolist() == [True, True, False]
    assert metrics["observed_path_fraction"].tolist() == [1.0, 1.0, 0.0]
    assert torch.isinf(metrics["observed_min_clearance_m"][-1])


def test_safety_sampling_detects_obstacle_between_output_points() -> None:
    path = torch.tensor([[[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]]])
    field = torch.zeros(1, 5, 9, 9)
    field[:, 0] = 1.0
    field[:, 0, 4, 6] = -0.05
    field[:, 3] = 1.0

    metrics = observed_safety_metrics(path, field, 2.0)

    assert metrics["observed_footprint_collision"].item()


def test_strata_separate_visible_detours_and_rear_goals() -> None:
    metrics = {
        "point_goal_is_behind": torch.tensor([False, False, True]),
        "reference_observed_safety_margin_violation": torch.tensor(
            [False, False, False]
        ),
        "straight_observed_safety_margin_violation": torch.tensor(
            [False, True, True]
        ),
        "reference_goal_progress_m": torch.tensor([1.0, 0.5, -0.1]),
    }

    strata = evaluation_strata(metrics)

    assert strata["forward_direct"].tolist() == [True, False, False]
    assert strata["forward_visible_detour"].tolist() == [False, True, False]
    assert strata["rear_goal"].tolist() == [False, False, True]
    assert strata["expert_moves_away_from_goal"].tolist() == [False, False, True]
