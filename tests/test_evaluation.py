import torch

from curvenav.evaluation.sand import trajectory_batch_metrics


def test_trajectory_metrics_select_the_best_candidate() -> None:
    target = torch.tensor([[[0.0, 0.0], [0.5, 0.0], [1.0, 0.0]]])
    shifted = torch.tensor([[[0.0, 0.0], [0.5, 0.4], [1.0, 0.0]]])
    paths = torch.stack((target, shifted), dim=1)
    curvature = torch.zeros(1, 2, 3)
    metrics = trajectory_batch_metrics(
        paths,
        curvature,
        target,
        torch.zeros(1, 3),
        selected_indices=torch.tensor([1], dtype=torch.int64),
    )
    assert metrics["oracle_ade_m"].item() == 0
    assert metrics["oracle_rmse_m"].item() == 0
    assert metrics["target_endpoint_error_m"].item() == 0
    assert metrics["candidate_pairwise_ade_m"].item() > 0
    assert metrics["selector_ade_m"].item() > 0
    assert metrics["selector_arc_length_error_m"].item() > 0
