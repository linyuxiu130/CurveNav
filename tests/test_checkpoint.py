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
    restore_training_state,
    validate_policy_contract,
)
from curvenav.training.ema import ExponentialMovingAverage


def test_checkpoint_records_the_flow_and_geometric_evaluator_contract() -> None:
    config = CurveNavConfig()
    model = nn.Linear(2, 2)
    optimizer = AdamW(model.parameters())
    scheduler = LambdaLR(optimizer, lambda _: 1.0)
    checkpoint = build_training_checkpoint(
        model,
        optimizer,
        scheduler,
        ExponentialMovingAverage(model),
        torch.amp.GradScaler("cuda", enabled=False),
        config,
        step=0,
    )
    contract = checkpoint["policy_contract"]
    assert checkpoint["checkpoint_type"] == "curvenav_local_policy"
    assert {"model", "optimizer", "scheduler", "ema", "grad_scaler"} <= set(checkpoint)
    assert "extra" not in checkpoint
    assert contract["trajectory_flow_candidates"] == 16
    assert (
        contract["trajectory_flow_type"]
        == "conditional_bspline_rectified_flow_heun"
    )
    assert (
        contract["trajectory_evaluator_type"]
        == "depth_surface_clearance_length_goal"
    )
    assert (
        contract["training_objective"]
        == "rectified_flow_plus_arc_path_and_tangent"
    )
    assert contract["trajectory_safe_center_distance_m"] == pytest.approx(0.35)
    assert contract["camera_extrinsics"] == {
        "forward_offset_m": pytest.approx(0.0),
        "height_m": pytest.approx(0.40),
        "downward_pitch_degrees": pytest.approx(0.0),
    }
    assert contract["num_control_points"] == 8
    assert contract["path_sampling"] == "uniform_metric_arc_progress"
    assert contract["visual_compression"] == "learned_queries_16_tokens_per_depth_frame"
    assert contract["observation_to_current"] == "planar_rigid_transform_used_for_depth_token_alignment"


def test_checkpoint_contract_catches_geometry_mismatch() -> None:
    config = CurveNavConfig()
    checkpoint = {
        "checkpoint_type": "curvenav_local_policy",
        "policy_contract": build_policy_contract(config),
    }
    validate_policy_contract(checkpoint, config)
    changed = replace(
        config, trajectory=replace(config.trajectory, normalization_scale_m=5.0)
    )
    with pytest.raises(ValueError, match="trajectory_scale_xy"):
        validate_policy_contract(checkpoint, changed)


def test_training_checkpoint_restores_the_complete_optimizer_state() -> None:
    config = CurveNavConfig()
    model = nn.Linear(2, 2)
    optimizer = AdamW(model.parameters())
    scheduler = LambdaLR(optimizer, lambda _: 1.0)
    ema = ExponentialMovingAverage(model)
    scaler = torch.amp.GradScaler("cuda", enabled=False)
    checkpoint = build_training_checkpoint(
        model, optimizer, scheduler, ema, scaler, config, step=37
    )

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


def test_checkpoint_requires_current_policy_type() -> None:
    with pytest.raises(ValueError, match="checkpoint_type"):
        validate_policy_contract(
            {
                "checkpoint_type": "unrelated_policy",
                "policy_contract": build_policy_contract(CurveNavConfig()),
            },
            CurveNavConfig(),
        )
