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
from curvenav.training.train import _advance_schedule_and_ema


BF16 = "bf16_primal_fp32_detached_meanflow_jvp"
FP16 = "fp16_primal_fp32_detached_meanflow_jvp"


def _distributed_state(
    config: CurveNavConfig,
) -> tuple[dict[str, int | str], dict[str, torch.Tensor]]:
    training_contract = build_training_contract(config, world_size=2, mixed_precision=BF16)
    cpu_state = torch.get_rng_state()
    rng_states = {
        "cpu": torch.stack((cpu_state, cpu_state)),
        "cuda": torch.zeros(2, 64, dtype=torch.uint8),
    }
    return training_contract, rng_states


def test_checkpoint_records_the_regular_heading_flow_contract() -> None:
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
        config,
        step=0,
        training_contract=training_contract,
        rng_states=rng_states,
        amp_state=None,
    )
    contract = checkpoint["policy_contract"]
    assert (
        checkpoint["checkpoint_type"]
        == "curvenav_metric_curve_mean_flow_policy"
    )
    assert set(checkpoint) == {
        "checkpoint_type",
        "step",
        "model",
        "optimizer",
        "scheduler",
        "ema",
        "config",
        "policy_contract",
        "training_contract",
        "amp_state",
        "rng_states",
    }
    assert checkpoint["training_contract"]["mixed_precision"] == (
        BF16
    )
    assert (
        contract["trajectory_decoder_type"]
        == "pointgoal_and_configuration_field_conditioned_curve_mean_flow_transformer"
    )
    assert contract["flow_path"] == (
        "data_anchored_linear_stochastic_interpolant"
    )
    assert contract["flow_solver"] == "none_direct_average_velocity_transport"
    assert contract["flow_source"] == (
        "standard_gaussian_flow_training_plus_exact_fixed_typical_"
        "deployment_boundary"
    )
    assert contract["flow_time_embedding"] == "end_time_and_interval_width_mlp"
    assert contract["flow_time_sampling"] == (
        "half_diagonal_quarter_logit_normal_interval_quarter_deployment_boundary"
    )
    assert contract["mean_flow_identity"] == (
        "auxiliary_instantaneous_velocity_jvp_reparameterized_average_v_loss"
    )
    assert contract["training_objective"] == (
        "standardized_euclidean_improved_mean_flow_plus_deployment_raw_"
        "observed_signed_clearance_soft_margin"
    )
    assert (
        contract["trajectory_prediction"]
        == "single_mean_flow_generated_regular_metric_heading_curve"
    )
    assert contract["curve_boundary_conditions"] == (
        "origin_and_robot_longitudinal_initial_heading"
    )
    assert contract["trajectory_endpoint_policy"] == (
        "single_evaluation_conditional_average_flow_curve"
    )
    assert contract["body_obstacle_selection"] == (
        "robot_collision_height_band_excluding_local_traversable_surface_triangles"
    )
    assert contract["temporal_modeling"] == (
        "aligned_four_frame_learned_depth_evidence_plus_causal_se2_motion_tokens"
    )
    assert "history_training_distribution" not in contract
    assert contract["state_token_features"] == "normalized_xy_sine_cosine"
    assert contract["state_translation_scale_m"] == pytest.approx(1.35)
    assert contract["state_token_count"] == 3
    assert "flow_steps" not in contract
    assert "inference_selection_type" not in contract
    assert contract["camera_extrinsics"] == {
        "forward_offset_m": pytest.approx(0.28618),
        "height_m": pytest.approx(0.62532),
        "downward_pitch_degrees": pytest.approx(10.0),
    }
    assert contract["num_curve_tokens"] == 8
    assert "route_query_count" not in contract
    assert contract["condition_token_count"] == 164
    assert contract["decoder_refinement_stages"] == 3
    assert contract["decoder_stage_supervision"] == (
        "separate_average_and_instantaneous_velocity_readouts_on_all_three_stages"
    )
    assert contract["configuration_encoder_type"] == (
        "metric_splat_observed_configuration_space_encoder"
    )
    assert contract["configuration_token_grid"] == [8, 8]
    assert contract["configuration_space_field"] == {
        "grid_size": 64,
        "extent_m": pytest.approx(3.6),
        "base_channels": [
            "measured_signed_clearance_m",
            "measured_gradient_x",
            "measured_gradient_y",
            "measured_ray_coverage_probability",
            "measured_forbidden_probability",
        ],
        "footprint_inflated": True,
        "training_supervision": (
            "deployment_generated_path_raw_observed_signed_clearance_soft_margin"
        ),
        "inference_input": "aligned_depth_evidence_only",
    }
    assert contract["source_configuration_space_truth"] == (
        "native_navigation_grid_dense_0.025m_oob_non_executable"
    )
    assert contract["body_obstacle_geometry"] == {
        "footprint_radius_m": pytest.approx(0.167584539),
        "extra_clearance_m": pytest.approx(0.10),
        "collision_bottom_z_m": pytest.approx(-0.044000001),
        "collision_top_z_m": pytest.approx(0.117981499),
        "collision_height_m": pytest.approx(0.161981500),
        "body_obstacle_min_z_m": pytest.approx(0.005999999),
        "maximum_traversable_height_m": pytest.approx(0.05),
        "maximum_traversable_slope_degrees": pytest.approx(45.0),
    }
    assert contract["num_heading_control_points"] == 8
    assert contract["path_sampling"] == "fixed_uniform_arc_progress"
    assert contract["curve_coordinates"] == (
        "positive_softplus_arc_length_and_cubic_heading_increment_coordinates"
    )
    assert contract["visual_planning_scale_m"] == pytest.approx(3.6)
    assert contract["curve_value_semantics"] == (
        "metric_arc_length_then_seven_cubic_heading_control_increments_rad"
    )
    assert contract["flow_coordinate_transform"] == (
        "standardized_softplus_length_pre_activation_and_heading_increments"
    )
    assert contract["length_pre_activation_mean"] == pytest.approx(2.77612897)
    assert contract["length_pre_activation_std"] == pytest.approx(1.33101579)
    assert len(contract["heading_increment_mean_rad"]) == 7
    assert len(contract["heading_increment_std_rad"]) == 7
    assert contract["inference_source_seed"] == 20_260_828
    assert "inference_candidate_count" not in contract
    assert "deployment_boundary_fraction" not in contract
    assert "training_source_endpoint_probability" not in contract
    assert "maximum_local_detour_ratio" not in contract
    assert "maximum_continuous_curvature_inv_m" not in contract
    assert contract["model_architecture"] == {
        "model_dim": 384,
        "depth_token_grid": [8, 12],
        "configuration_token_grid": [8, 8],
        "depth_dropout": 0.0,
        "condition_heads": 8,
        "condition_layers": 4,
        "condition_dropout": 0.0,
        "trajectory_decoder_layers": 12,
        "trajectory_decoder_heads": 8,
        "trajectory_path_tokens": 32,
        "trajectory_decoder_dropout": 0.0,
    }
    assert (
        contract["visual_compression"]
        == "shared_resnet_all_frames_metric_splat_to_8x8"
    )
    assert (
        contract["observation_to_current"]
        == "planar_rigid_transform_used_for_metric_xyz_alignment_and_motion_state"
    )


def test_checkpoint_contract_catches_geometry_mismatch() -> None:
    config = CurveNavConfig()
    checkpoint = {
        "checkpoint_type": "curvenav_metric_curve_mean_flow_policy",
        "policy_contract": build_policy_contract(config),
    }
    validate_policy_contract(checkpoint, config)
    changed = replace(config, data=replace(config.data, max_depth_m=6.0))
    with pytest.raises(ValueError, match="max_depth_m"):
        validate_policy_contract(checkpoint, changed)

    changed_width = replace(
        config,
        depth_encoder=replace(config.depth_encoder, model_dim=512),
        condition_encoder=replace(config.condition_encoder, model_dim=512),
        trajectory_decoder=replace(config.trajectory_decoder, model_dim=512),
    )
    with pytest.raises(ValueError, match="model_architecture"):
        validate_policy_contract(checkpoint, changed_width)


def test_training_checkpoint_restores_the_complete_optimizer_state() -> None:
    config = CurveNavConfig()
    model = nn.Linear(2, 2)
    optimizer = AdamW(model.parameters())
    scheduler = LambdaLR(optimizer, lambda _: 1.0)
    ema = ExponentialMovingAverage(model)
    training_contract, rng_states = _distributed_state(config)
    checkpoint = build_training_checkpoint(
        model,
        optimizer,
        scheduler,
        ema,
        config,
        step=37,
        training_contract=training_contract,
        rng_states=rng_states,
        amp_state=None,
    )
    validate_training_resume(checkpoint, config, world_size=2, mixed_precision=BF16)

    restored_model = nn.Linear(2, 2)
    restored_optimizer = AdamW(restored_model.parameters())
    restored_scheduler = LambdaLR(restored_optimizer, lambda _: 1.0)
    restored_ema = ExponentialMovingAverage(restored_model)
    restored_step = restore_training_state(
        checkpoint,
        restored_model,
        restored_optimizer,
        restored_scheduler,
        restored_ema,
        config,
        scaler=None,
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


def test_fp16_checkpoint_preserves_the_scaler_state() -> None:
    class Scaler:
        def __init__(self) -> None:
            self.state = {"scale": 512.0}

        def state_dict(self):
            return self.state

        def load_state_dict(self, state) -> None:
            self.state = dict(state)

    config = CurveNavConfig()
    model = nn.Linear(2, 2)
    optimizer = AdamW(model.parameters())
    scheduler = LambdaLR(optimizer, lambda _: 1.0)
    _, rng_states = _distributed_state(config)
    checkpoint = build_training_checkpoint(
        model,
        optimizer,
        scheduler,
        ExponentialMovingAverage(model),
        config,
        step=0,
        training_contract=build_training_contract(config, 2, FP16),
        rng_states=rng_states,
        amp_state={"scale": 512.0},
    )
    validate_training_resume(checkpoint, config, 2, FP16)
    scaler = Scaler()
    scaler.state = {"scale": 1.0}
    restored_model = nn.Linear(2, 2)
    restored_optimizer = AdamW(restored_model.parameters())
    restored_scheduler = LambdaLR(restored_optimizer, lambda _: 1.0)
    restore_training_state(
        checkpoint,
        restored_model,
        restored_optimizer,
        restored_scheduler,
        ExponentialMovingAverage(restored_model),
        config,
        scaler,
    )
    assert scaler.state == {"scale": 512.0}


def test_amp_skipped_update_does_not_advance_scheduler_or_ema() -> None:
    model = nn.Linear(2, 2)
    optimizer = AdamW(model.parameters())
    scheduler = LambdaLR(optimizer, lambda _: 1.0)
    ema = ExponentialMovingAverage(model)

    optimizer.step()
    _advance_schedule_and_ema(
        scheduler,
        ema,
        optimizer_step_was_skipped=True,
    )

    assert scheduler.last_epoch == 0
    assert ema.num_updates == 0
    optimizer.step()
    _advance_schedule_and_ema(
        scheduler,
        ema,
        optimizer_step_was_skipped=False,
    )
    assert scheduler.last_epoch == 1
    assert ema.num_updates == 1


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
        config,
        step=10,
        training_contract=training_contract,
        rng_states=rng_states,
        amp_state=None,
    )

    changed = replace(
        config,
        training=replace(config.training, learning_rate=1e-4),
    )
    with pytest.raises(ValueError, match="configuration"):
        validate_training_resume(
            checkpoint, changed, world_size=2, mixed_precision=BF16
        )
    with pytest.raises(ValueError, match="topology"):
        validate_training_resume(
            checkpoint, config, world_size=4, mixed_precision=BF16
        )


def test_training_contract_preserves_global_optimization_across_one_to_eight_gpus() -> (
    None
):
    for world_size in range(1, 9):
        contract = build_training_contract(CurveNavConfig(), world_size, BF16)
        assert (
            contract["minimum_per_rank_batch_size"] * world_size
            <= 1024
            <= contract["maximum_per_rank_batch_size"] * world_size
        )
        assert contract["per_device_batch_size"] == 342
        assert contract["mixed_precision"] == (
            BF16
        )
        assert contract["global_batch_size"] == 1024
        assert contract["flow_interval_assignment"] == (
            "global_sample_stream_index_mod_four"
        )
        assert contract["steps_per_epoch"] == 40
        assert contract["total_steps"] == 8000

    six_gpu = build_training_contract(CurveNavConfig(), 6, BF16)
    assert six_gpu["minimum_per_rank_batch_size"] == 170
    assert six_gpu["maximum_per_rank_batch_size"] == 171
    assert six_gpu["micro_batches_per_step"] == 1

    with pytest.raises(ValueError, match="world_size must be one of"):
        build_training_contract(CurveNavConfig(), world_size=9, mixed_precision=BF16)


def test_checkpoint_requires_current_policy_type() -> None:
    with pytest.raises(ValueError, match="checkpoint_type"):
        validate_policy_contract(
            {
                "checkpoint_type": "unrelated_policy",
                "policy_contract": build_policy_contract(CurveNavConfig()),
            },
            CurveNavConfig(),
        )
