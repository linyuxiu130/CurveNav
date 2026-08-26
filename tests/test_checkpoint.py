from dataclasses import replace

import pytest
import torch
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

from curvenav.config import CurveNavConfig
from curvenav.training.checkpoint import (
    build_policy_contract,
    build_training_contract,
    build_training_checkpoint,
    restore_training_state,
    validate_policy_contract,
    validate_training_resume,
)
from curvenav.training.ema import ExponentialMovingAverage
from curvenav.training.train import _optimizer_step_succeeded


class _TestGradScaler:
    def __init__(self, scale: float) -> None:
        self.scale = scale

    def get_scale(self) -> float:
        return self.scale


class _TestOptimizer:
    def __init__(self, scaler: _TestGradScaler, next_scale: float) -> None:
        self.scaler = scaler
        self.next_scale = next_scale

    def step(self) -> None:
        self.scaler.scale = self.next_scale


def _distributed_state(
    config: CurveNavConfig,
) -> tuple[dict[str, int], dict[str, torch.Tensor]]:
    training_contract = build_training_contract(config, world_size=2)
    cpu_state = torch.get_rng_state()
    rng_states = {
        "cpu": torch.stack((cpu_state, cpu_state)),
        "cuda": torch.zeros(2, 64, dtype=torch.uint8),
    }
    return training_contract, rng_states


@pytest.mark.parametrize(
    ("next_scale", "succeeded"),
    ((32768.0, False), (65536.0, True), (131072.0, True)),
)
def test_optimizer_step_uses_loss_scale_as_the_overflow_signal(
    next_scale: float,
    succeeded: bool,
) -> None:
    scaler = _TestGradScaler(65536.0)
    optimizer = _TestOptimizer(scaler, next_scale)
    assert _optimizer_step_succeeded(optimizer, scaler) is succeeded


def test_checkpoint_records_the_bounded_curvature_flow_contract() -> None:
    config = CurveNavConfig()
    model = nn.Linear(2, 2)
    optimizer = AdamW(model.parameters())
    scheduler = LambdaLR(optimizer, lambda _: 1.0)
    training_contract, rng_states = _distributed_state(config)
    checkpoint = build_training_checkpoint(
        model,
        optimizer,
        scheduler,
        ExponentialMovingAverage(model),
        torch.amp.GradScaler("cuda", enabled=False),
        config,
        step=0,
        training_contract=training_contract,
        rng_states=rng_states,
    )
    contract = checkpoint["policy_contract"]
    assert (
        checkpoint["checkpoint_type"]
        == "curvenav_gaussian_flow_zero_mode_bounded_curvature_policy"
    )
    assert {"model", "optimizer", "scheduler", "ema", "grad_scaler"} <= set(checkpoint)
    assert "extra" not in checkpoint
    assert (
        contract["trajectory_flow_type"]
        == "normalized_gaussian_source_future_bounded_curvature_rectified_flow_adarmszero_heun"
    )
    assert contract["flow_curve_coordinate_scale"] == pytest.approx(8.0)
    assert contract["flow_training_source_type"] == "masked_isotropic_gaussian"
    assert contract["flow_inference_source_type"] == "zero_prior_mode"
    assert (
        contract["training_objective"]
        == "gaussian_source_future_flow_plus_metric_path_tangent_subgoal"
    )
    assert (
        contract["trajectory_prediction"]
        == "single_zero_prior_mode_heun_trajectory"
    )
    assert "trajectory_candidate_samples" not in contract
    assert contract["camera_extrinsics"] == {
        "forward_offset_m": pytest.approx(0.28618),
        "height_m": pytest.approx(0.62532),
        "downward_pitch_degrees": pytest.approx(10.0),
    }
    assert contract["num_curve_tokens"] == 8
    assert contract["num_curvature_control_points"] == 7
    assert contract["path_sampling"] == "fixed_uniform_metric_arc_progress"
    assert contract["curve_coordinates"] == (
        "pointgoal_scaled_arc_length_cubic_curvature_bspline"
    )
    assert contract["curve_planning_horizon_m"] == pytest.approx(3.6)
    assert contract["maximum_continuous_curvature_inv_m"] == pytest.approx(8.0)
    assert contract["model_architecture"] == {
        "model_dim": 384,
        "depth_token_grid": [8, 12],
        "depth_dropout": 0.0,
        "point_goal_hidden_dim": 384,
        "condition_layers": 4,
        "condition_heads": 8,
        "condition_dropout": 0.0,
        "flow_layers": 8,
        "flow_heads": 8,
        "flow_dropout": 0.0,
    }
    assert (
        contract["visual_compression"] == "64_goal_independent_metric_geometry_queries"
    )
    assert (
        contract["observation_to_current"]
        == "planar_rigid_transform_used_for_depth_token_alignment"
    )


def test_checkpoint_contract_catches_geometry_mismatch() -> None:
    config = CurveNavConfig()
    checkpoint = {
        "checkpoint_type": "curvenav_gaussian_flow_zero_mode_bounded_curvature_policy",
        "policy_contract": build_policy_contract(config),
    }
    validate_policy_contract(checkpoint, config)
    changed = replace(config, data=replace(config.data, max_depth_m=6.0))
    with pytest.raises(ValueError, match="max_depth_m"):
        validate_policy_contract(checkpoint, changed)

    changed_width = replace(
        config,
        depth_encoder=replace(config.depth_encoder, model_dim=512),
        point_goal_encoder=replace(config.point_goal_encoder, model_dim=512),
        condition_encoder=replace(config.condition_encoder, model_dim=512),
        trajectory_flow=replace(config.trajectory_flow, model_dim=512),
    )
    with pytest.raises(ValueError, match="model_architecture"):
        validate_policy_contract(checkpoint, changed_width)


def test_training_checkpoint_restores_the_complete_optimizer_state() -> None:
    config = CurveNavConfig()
    model = nn.Linear(2, 2)
    optimizer = AdamW(model.parameters())
    scheduler = LambdaLR(optimizer, lambda _: 1.0)
    ema = ExponentialMovingAverage(model)
    scaler = torch.amp.GradScaler("cuda", enabled=False)
    training_contract, rng_states = _distributed_state(config)
    checkpoint = build_training_checkpoint(
        model,
        optimizer,
        scheduler,
        ema,
        scaler,
        config,
        step=37,
        training_contract=training_contract,
        rng_states=rng_states,
    )
    validate_training_resume(checkpoint, config, world_size=2)

    restored_model = nn.Linear(2, 2)
    restored_optimizer = AdamW(restored_model.parameters())
    restored_scheduler = LambdaLR(restored_optimizer, lambda _: 1.0)
    restored_ema = ExponentialMovingAverage(restored_model)
    restored_scaler = torch.amp.GradScaler("cuda", enabled=False)
    restored_step = restore_training_state(
        checkpoint,
        restored_model,
        restored_optimizer,
        restored_scheduler,
        restored_ema,
        config,
        restored_scaler,
    )

    assert restored_step == 37
    for actual, expected in zip(
        restored_model.parameters(), model.parameters(), strict=True
    ):
        torch.testing.assert_close(actual, expected)
    assert restored_ema.decay == ema.decay
    assert restored_ema.num_updates == ema.num_updates
    for name, shadow in ema.shadow.items():
        torch.testing.assert_close(restored_ema.shadow[name], shadow)


def test_resume_rejects_changed_configuration_or_world_size() -> None:
    config = CurveNavConfig()
    model = nn.Linear(2, 2)
    optimizer = AdamW(model.parameters())
    scheduler = LambdaLR(optimizer, lambda _: 1.0)
    training_contract, rng_states = _distributed_state(config)
    checkpoint = build_training_checkpoint(
        model,
        optimizer,
        scheduler,
        ExponentialMovingAverage(model),
        torch.amp.GradScaler("cuda", enabled=False),
        config,
        step=10,
        training_contract=training_contract,
        rng_states=rng_states,
    )

    changed = replace(
        config,
        training=replace(config.training, learning_rate=1e-4),
    )
    with pytest.raises(ValueError, match="configuration"):
        validate_training_resume(checkpoint, changed, world_size=2)
    with pytest.raises(ValueError, match="topology"):
        validate_training_resume(checkpoint, config, world_size=4)


def test_training_contract_preserves_global_optimization_across_one_to_eight_gpus() -> (
    None
):
    for world_size in range(1, 9):
        contract = build_training_contract(CurveNavConfig(), world_size)
        assert (
            contract["minimum_per_rank_batch_size"] * world_size
            <= 1024
            <= contract["maximum_per_rank_batch_size"] * world_size
        )
        assert contract["per_device_batch_size"] == 112
        assert contract["global_batch_size"] == 1024
        assert contract["steps_per_epoch"] == 40
        assert contract["total_steps"] == 8000

    six_gpu = build_training_contract(CurveNavConfig(), 6)
    assert six_gpu["minimum_per_rank_batch_size"] == 170
    assert six_gpu["maximum_per_rank_batch_size"] == 171
    assert six_gpu["micro_batches_per_step"] == 2

    with pytest.raises(ValueError, match="world_size must be one of"):
        build_training_contract(CurveNavConfig(), world_size=9)


def test_checkpoint_requires_current_policy_type() -> None:
    with pytest.raises(ValueError, match="checkpoint_type"):
        validate_policy_contract(
            {
                "checkpoint_type": "unrelated_policy",
                "policy_contract": build_policy_contract(CurveNavConfig()),
            },
            CurveNavConfig(),
        )
