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


BF16 = "bf16_neural_fp32_geometry_flow_accumulation"
FP16 = "fp16_neural_fp32_geometry_flow_jvp"


def distributed_state(config: CurveNavConfig, precision: str = BF16):
    contract = build_training_contract(config, 2, precision)
    cpu = torch.get_rng_state()
    return contract, {
        "cpu": torch.stack((cpu, cpu)),
        "cuda": torch.zeros(2, 64, dtype=torch.uint8),
    }


def checkpoint(config: CurveNavConfig, precision: str = BF16):
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
    )
    return value, model, optimizer, scheduler


def test_checkpoint_records_the_clean_depth_grounded_flow_contract() -> None:
    config = CurveNavConfig()
    value, _, _, _ = checkpoint(config)
    contract = value["policy_contract"]
    assert value["checkpoint_type"] == "curvenav_metric_curve_flow_policy"
    assert contract["trajectory_decoder_type"] == (
        "pointwise_geometry_increment_curve_flow_transformer"
    )
    assert contract["training_objective"] == (
        "conditional_flow_matching_plus_route_utility_regression_and_pairwise_ranking"
    )
    assert contract["curve_coordinates"] == (
        "standardized_physical_bspline_control_increments"
    )
    assert contract["num_curve_tokens"] == 7
    assert contract["curve_coordinate_dim"] == 14
    assert contract["num_control_points"] == 8
    assert contract["condition_token_count"] == 259
    assert contract["source_configuration_space_role"] == (
        "dataset_certificate_critic_supervision_and_evaluation"
    )
    assert contract["depth_configuration_space_role"] == (
        "target_independent_observed_bev_plus_current_curve_query"
    )
    assert contract["trajectory_condition_interaction"] == (
        "cached_bev_kv_state_dependent_curve_geometry_attention"
    )
    assert contract["path_relative_geometry"] == (
        "increment_effect_weighted_current_curve_to_bev_attention_bias"
    )
    assert contract["goal_conditioning"] == (
        "terminal_local_goal_vector_without_straight_template_matching"
    )
    assert contract["decoder_flow_fields"] == 1
    assert contract["flow_solver"] == "explicit_euler_noise_to_data"
    assert len(contract["control_increment_mean_xy_m"]) == 14
    assert len(contract["control_increment_std_xy_m"]) == 14


def test_policy_contract_catches_geometry_mismatch() -> None:
    config = CurveNavConfig()
    value = {
        "checkpoint_type": "curvenav_metric_curve_flow_policy",
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
    )
    assert step == 0
    for left, right in zip(model.parameters(), restored.parameters(), strict=True):
        torch.testing.assert_close(left, right)
    assert restored_scheduler.state_dict() == scheduler.state_dict()
    assert restored_optimizer.state_dict()["param_groups"] == (
        optimizer.state_dict()["param_groups"]
    )


def test_fp16_training_contract_is_rejected() -> None:
    with pytest.raises(ValueError, match="unsupported CurveNav precision"):
        build_training_contract(CurveNavConfig(), 2, FP16)


def test_training_contract_derives_global_batch_from_devices_and_accumulation() -> None:
    config = CurveNavConfig()
    for world_size in range(1, 9):
        contract = build_training_contract(config, world_size, BF16)
        batch = config.training.per_device_batch_size * world_size
        assert contract["global_batch_size"] == batch
        assert contract["steps_per_epoch"] == config.training.samples_per_epoch // batch
        assert contract["total_steps"] == config.training.epochs * contract["steps_per_epoch"]
        assert contract["samples_per_epoch"] == contract["steps_per_epoch"] * batch
    accumulated = replace(config, training=replace(config.training, gradient_accumulation_steps=2))
    contract = build_training_contract(accumulated, 2, BF16)
    assert contract["global_batch_size"] == config.training.per_device_batch_size * 4
    assert contract["micro_batches_per_step"] == 2


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


def test_resume_allows_moving_artifacts_without_changing_training():
    config = CurveNavConfig()
    value, _, _, _ = checkpoint(config)
    moved = replace(config, training=replace(config.training, output_dir="outputs/resumed"))
    validate_training_resume(value, moved, 2, BF16)
    changed = replace(moved, training=replace(moved.training, learning_rate=1e-4))
    with pytest.raises(ValueError, match="configuration"):
        validate_training_resume(value, changed, 2, BF16)
