"""Checkpoint and distributed-training contract tests."""

from dataclasses import replace

import pytest
import torch
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

from curvenav.config import CurveNavConfig
from curvenav.training.checkpoint import (
    build_policy_contract,
    build_training_checkpoint,
    build_training_contract,
    restore_training_state,
    validate_policy_contract,
    validate_training_resume,
)
from curvenav.training.ema import ExponentialMovingAverage


BF16 = "bf16_neural_fp32_geometry_flow_jvp"
FP16 = "fp16_neural_fp32_geometry_flow_jvp"


def distributed_state(config: CurveNavConfig, precision: str = BF16):
    contract = build_training_contract(config, 2, precision)
    cpu = torch.get_rng_state()
    return contract, {
        "cpu": torch.stack((cpu, cpu)),
        "cuda": torch.zeros(2, 64, dtype=torch.uint8),
    }


def checkpoint(config: CurveNavConfig, precision: str = BF16, amp_state=None):
    model = nn.Linear(2, 2)
    optimizer = AdamW(model.parameters())
    scheduler = LambdaLR(optimizer, lambda _: 1.0)
    contract, rng = distributed_state(config, precision)
    value = build_training_checkpoint(
        model,
        optimizer,
        scheduler,
        ExponentialMovingAverage(model),
        config,
        0,
        training_contract=contract,
        rng_states=rng,
        amp_state=amp_state,
    )
    return value, model, optimizer, scheduler


def test_checkpoint_records_the_clean_depth_grounded_flow_contract() -> None:
    config = CurveNavConfig()
    value, _, _, _ = checkpoint(config)
    contract = value["policy_contract"]
    assert value["checkpoint_type"] == "curvenav_metric_curve_mean_flow_policy"
    assert contract["trajectory_decoder_type"] == (
        "single_call_global_geometry_then_clean_proposal_cspace_improved_mean_flow"
    )
    assert contract["training_objective"] == (
        "standardized_euclidean_mean_flow_plus_deployed_"
        "strict_observed_clearance_risk"
    )
    assert contract["curve_coordinates"] == (
        "standardized_physical_bspline_control_increments"
    )
    assert contract["num_curve_tokens"] == 7
    assert contract["curve_coordinate_dim"] == 14
    assert contract["num_control_points"] == 8
    assert contract["condition_token_count"] == 259
    assert contract["source_configuration_space_role"] == (
        "dataset_certificate_and_evaluation_only"
    )
    assert contract["depth_configuration_space_role"] == (
        "target_independent_observed_bev_plus_flow_candidate_curve_query_"
        "plus_deployed_curve_training_risk"
    )
    assert contract["trajectory_condition_interaction"] == (
        "goal_independent_global_scene_then_clean_estimate_query_observed_cspace_and_bev"
    )
    assert contract["path_relative_geometry"] == (
        "robot_origin_scene_then_learned_clean_control_to_bev_metric_attention_bias"
    )
    assert contract["decoder_flow_fields"] == 2
    assert contract["flow_solver"] == "none_direct_average_velocity"
    assert "clearance_risk" in contract["training_objective"]
    assert len(contract["control_increment_mean_xy_m"]) == 14
    assert len(contract["control_increment_std_xy_m"]) == 14


def test_policy_contract_catches_geometry_mismatch() -> None:
    config = CurveNavConfig()
    value = {
        "checkpoint_type": "curvenav_metric_curve_mean_flow_policy",
        "policy_contract": build_policy_contract(config),
    }
    validate_policy_contract(value, config)
    changed = replace(config, data=replace(config.data, max_depth_m=6.0))
    with pytest.raises(ValueError, match="max_depth_m"):
        validate_policy_contract(value, changed)


def test_complete_optimizer_state_roundtrip() -> None:
    config = CurveNavConfig()
    value, model, optimizer, scheduler = checkpoint(config)
    validate_training_resume(value, config, 2, BF16)
    restored = nn.Linear(2, 2)
    restored_optimizer = AdamW(restored.parameters())
    restored_scheduler = LambdaLR(restored_optimizer, lambda _: 1.0)
    step = restore_training_state(
        value,
        restored,
        restored_optimizer,
        restored_scheduler,
        ExponentialMovingAverage(restored),
        config,
        scaler=None,
    )
    assert step == 0
    for left, right in zip(model.parameters(), restored.parameters(), strict=True):
        torch.testing.assert_close(left, right)
    assert restored_scheduler.state_dict() == scheduler.state_dict()
    assert restored_optimizer.state_dict()["param_groups"] == (
        optimizer.state_dict()["param_groups"]
    )


def test_fp16_requires_and_restores_scaler_state() -> None:
    config = CurveNavConfig()
    amp = {"scale": 1024.0}
    value, _, _, _ = checkpoint(config, FP16, amp)
    validate_training_resume(value, config, 2, FP16)

    class Scaler:
        def __init__(self):
            self.state = None

        def load_state_dict(self, state):
            self.state = state

    restored = nn.Linear(2, 2)
    optimizer = AdamW(restored.parameters())
    scheduler = LambdaLR(optimizer, lambda _: 1.0)
    scaler = Scaler()
    restore_training_state(
        value,
        restored,
        optimizer,
        scheduler,
        ExponentialMovingAverage(restored),
        config,
        scaler,
    )
    assert scaler.state == amp


def test_training_contract_preserves_global_batch_for_supported_world_sizes() -> None:
    config = CurveNavConfig()
    for world_size in range(1, 9):
        contract = build_training_contract(config, world_size, BF16)
        assert contract["global_batch_size"] == 1792
        assert contract["steps_per_epoch"] == 23
        assert contract["total_steps"] == 4600


def test_resume_rejects_configuration_and_topology_changes() -> None:
    config = CurveNavConfig()
    value, _, _, _ = checkpoint(config)
    changed = replace(
        config,
        training=replace(config.training, learning_rate=1e-4),
    )
    with pytest.raises(ValueError, match="configuration"):
        validate_training_resume(value, changed, 2, BF16)
    with pytest.raises(ValueError, match="topology"):
        validate_training_resume(value, config, 4, BF16)


def test_checkpoint_type_is_strict() -> None:
    config = CurveNavConfig()
    with pytest.raises(ValueError, match="checkpoint_type"):
        validate_policy_contract(
            {
                "checkpoint_type": "old",
                "policy_contract": build_policy_contract(config),
            },
            config,
        )
