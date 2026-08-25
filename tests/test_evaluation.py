import torch

from curvenav.evaluation.offline import (
    candidate_batch_metrics,
    trajectory_batch_metrics,
)


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


def test_candidate_metrics_expose_oracle_diversity_and_margin() -> None:
    target = torch.zeros(1, 3, 2)
    candidates = torch.zeros(1, 8, 3, 2)
    candidates[0, :, -1, 0] = torch.arange(8)
    costs = torch.arange(8, dtype=torch.float32).unsqueeze(0)
    clearance = torch.ones(1, 8)

    metrics = candidate_batch_metrics(candidates, costs, clearance, target, 0.35)

    assert metrics["oracle_ade_m"].item() == 0.0
    assert metrics["candidate_endpoint_diversity_m"].item() > 0.0
    assert metrics["geometric_cost_margin"].item() == 1.0
    assert metrics["selected_minimum_clearance_m"].item() == 1.0
