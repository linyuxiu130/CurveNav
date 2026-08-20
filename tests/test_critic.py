import json

import numpy as np
import torch

from curvenav.critic import (
    CriticPrediction,
    TrajectoryCritic,
    critic_metric_counts,
    critic_loss,
    load_critic_sidecar_table,
    select_critic_indices,
)
from curvenav.trajectory import PlanarBSplineCodec


def test_critic_is_equivariant_to_candidate_order() -> None:
    torch.manual_seed(7)
    critic = TrajectoryCritic(
        num_control_points=12,
        scale_xy=(5.6, 2.5),
        model_dim=32,
        layers=2,
        heads=4,
        dropout=0.0,
    ).eval()
    controls = torch.randn(2, 5, 12, 2)
    memory = torch.randn(2, 18, 32)
    order = torch.tensor([3, 0, 4, 1, 2])

    original = critic(controls, memory)
    permuted = critic(controls[:, order], memory)

    torch.testing.assert_close(permuted.score, original.score[:, order])
    torch.testing.assert_close(
        permuted.collision_logit, original.collision_logit[:, order]
    )


def test_pairwise_loss_rewards_the_declared_winner() -> None:
    zero = torch.zeros(1, 2)
    preference = torch.tensor([[[False, True], [False, False]]])
    labels = {
        "preference": preference,
        "candidate_valid": torch.ones(1, 2, dtype=torch.bool),
        "collision": torch.zeros(1, 2, dtype=torch.bool),
        "margin_violation": torch.zeros(1, 2, dtype=torch.bool),
        "progress_m": zero,
        "progress_valid": torch.ones(1, 2, dtype=torch.bool),
        "clearance_m": zero,
        "auxiliary_weight": 0.25,
    }

    def prediction(score: torch.Tensor) -> CriticPrediction:
        return CriticPrediction(score, zero, zero, zero, zero)

    correct = critic_loss(prediction(torch.tensor([[2.0, -2.0]])), **labels)
    reversed_order = critic_loss(prediction(torch.tensor([[-2.0, 2.0]])), **labels)

    assert correct.pairwise < reversed_order.pairwise
    assert correct.loss < reversed_order.loss


def test_sidecar_table_is_keyed_by_global_sample_index(tmp_path) -> None:
    root = tmp_path
    (root / "scene_shards").mkdir()
    (root / "manifest.json").write_text(
        json.dumps({"states": 2, "shards": {"unit.npz": {}}})
    )
    offsets = np.array([0, 10, 20], dtype=np.int64)
    pair_offsets = np.array([0, 1, 2], dtype=np.int64)
    controls = np.zeros((20, 12, 2), dtype=np.float32)
    controls[:10, :, 0] = 1.0
    controls[10:, :, 0] = 2.0
    np.savez(
        root / "scene_shards/unit.npz",
        state_sample_index=np.array([1, 0], dtype=np.int64),
        candidate_offsets=offsets,
        control_points_local_xy_m=controls,
        candidate_kind=np.tile(
            np.array([0] * 8 + [1, 2], dtype=np.uint8), 2
        ),
        control_valid=np.ones(20, dtype=np.bool_),
        footprint_collision=np.zeros(20, dtype=np.bool_),
        safety_margin_violation=np.zeros(20, dtype=np.bool_),
        progress_m=np.arange(20, dtype=np.float32),
        geodesic_valid=np.ones(20, dtype=np.bool_),
        minimum_extra_clearance_m=np.ones(20, dtype=np.float32),
        pairwise_preference_offsets=pair_offsets,
        pairwise_winner_local_index=np.array([2, 4], dtype=np.int32),
        pairwise_loser_local_index=np.array([3, 5], dtype=np.int32),
    )

    table = load_critic_sidecar_table(
        root,
        PlanarBSplineCodec(num_control_points=12, degree=3, num_path_points=64),
        validate=False,
    )

    assert table.controls_m.shape == (2, 10, 12, 2)
    assert table.controls_m[0, 0, 0, 0].item() == 2.0
    assert table.preference[1, 2, 3]
    assert table.preference[0, 4, 5]


def test_critic_metrics_rank_only_policy_candidates() -> None:
    score = torch.tensor([[0.0, 3.0, 2.0, 1.0, 0.0, 0.0, 0.0, 0.0, 99.0, 98.0]])
    zero = torch.zeros_like(score)
    preference = torch.zeros(1, 10, 10, dtype=torch.bool)
    preference[0, 2, 1] = True
    collision = torch.zeros(1, 10, dtype=torch.bool)
    collision[0, 1] = True
    margin = torch.zeros_like(collision)
    prediction = CriticPrediction(score, zero, zero, zero, zero)

    metrics = critic_metric_counts(
        prediction,
        preference=preference,
        collision=collision,
        margin_violation=margin,
    ).summary()

    assert metrics["selected_dominated_fraction"] == 1.0
    assert metrics["collision_opportunity_loss_fraction"] == 1.0


def test_critic_selection_is_lexicographic_without_scalar_weights() -> None:
    score = torch.tensor([[9.0, 8.0, 7.0, 1.0, 0.0, 0.0, 0.0, 0.0]])
    collision_logit = torch.tensor([[-1.0, -1.0, 1.0, -1.0, 1.0, 1.0, 1.0, 1.0]])
    margin_logit = torch.tensor([[1.0, -1.0, -1.0, -1.0, 1.0, 1.0, 1.0, 1.0]])
    zero = torch.zeros_like(score)
    prediction = CriticPrediction(
        score, collision_logit, margin_logit, zero, zero
    )

    selected = select_critic_indices(prediction)

    assert selected.item() == 1
