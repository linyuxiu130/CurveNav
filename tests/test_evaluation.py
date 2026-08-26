import torch

from curvenav.evaluation.offline import trajectory_batch_metrics


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
