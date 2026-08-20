from dataclasses import replace

import pytest
import torch
from torch import nn

from curvenav.config import CurveNavConfig, RectifiedFlowConfig
from curvenav.training import (
    checkpoint_state,
    policy_contract,
    restore_training_state,
    validate_policy_contract,
)
from curvenav.training.ema import ExponentialMovingAverage
from curvenav.training.train import _advance_optimizer_state, _save_checkpoint


def test_checkpoint_uses_temporal_stack_depth_encoder_format() -> None:
    config = CurveNavConfig()
    checkpoint = checkpoint_state(nn.Linear(2, 2), config, step=0)
    assert checkpoint["format_version"] == 11
    assert (
        checkpoint["policy_contract"]["goal_semantics"]
        == "sampled_future_point_goal_robot_xy"
    )
    assert (
        checkpoint["policy_contract"]["sand_supervision"]
        == "target_endpoint_equals_point_goal"
    )
    assert (
        checkpoint["policy_contract"]["endpoint_policy"]
        == "learned_local_endpoint_origin_only"
    )
    assert (
        checkpoint["policy_contract"]["depth_encoder_revision"]
        == "stride4_temporal_stack_dilated_residual_v4"
    )


def test_checkpoint_contract_catches_scale_mismatch() -> None:
    config = CurveNavConfig()
    checkpoint = {"format_version": 11, "policy_contract": policy_contract(config)}
    validate_policy_contract(checkpoint, config)

    changed = replace(config, trajectory=replace(config.trajectory, scale_xy=(4.0, 2.0)))
    with pytest.raises(ValueError, match="scale_xy"):
        validate_policy_contract(checkpoint, changed)


def test_checkpoint_contract_rejects_pre_unconstrained_format() -> None:
    config = CurveNavConfig()
    old_contract = policy_contract(config)
    del old_contract["arc_length_policy"]
    with pytest.raises(ValueError, match="arc_length_policy"):
        validate_policy_contract(
            {"format_version": 11, "policy_contract": old_contract}, config
        )


def test_checkpoint_contract_rejects_missing_depth_encoder_revision() -> None:
    config = CurveNavConfig()
    old_contract = policy_contract(config)
    del old_contract["depth_encoder_revision"]
    with pytest.raises(ValueError, match="depth_encoder_revision"):
        validate_policy_contract(
            {"format_version": 11, "policy_contract": old_contract}, config
        )


def test_checkpoint_contract_catches_solver_step_mismatch() -> None:
    config = CurveNavConfig()
    checkpoint = {"format_version": 11, "policy_contract": policy_contract(config)}
    changed = replace(config, rectified_flow=RectifiedFlowConfig(inference_steps=4))
    with pytest.raises(ValueError, match="inference_steps"):
        validate_policy_contract(checkpoint, changed)


def test_checkpoint_contract_catches_source_scale_mismatch() -> None:
    config = CurveNavConfig()
    checkpoint = {"format_version": 11, "policy_contract": policy_contract(config)}
    changed = replace(
        config,
        rectified_flow=replace(config.rectified_flow, source_std_xy=(0.1, 0.1)),
    )
    with pytest.raises(ValueError, match="source_std_xy"):
        validate_policy_contract(checkpoint, changed)


def test_policy_contract_catches_depth_preprocessing_mismatch() -> None:
    config = CurveNavConfig()
    checkpoint = {"format_version": 11, "policy_contract": policy_contract(config)}
    changed = replace(config, data=replace(config.data, max_depth_m=10.0))
    with pytest.raises(ValueError, match="max_depth_m"):
        validate_policy_contract(checkpoint, changed)


def test_checkpoint_contract_rejects_old_goal_semantics_format() -> None:
    with pytest.raises(ValueError, match="format_version 11"):
        validate_policy_contract(
            {
                "format_version": 10,
                "policy_contract": policy_contract(CurveNavConfig()),
            },
            CurveNavConfig(),
        )


def test_training_checkpoint_atomically_replaces_one_recovery_file(tmp_path) -> None:
    class FakeAccelerator:
        is_main_process = True
        num_processes = 2
        mixed_precision = "fp16"
        scaler = None

        @staticmethod
        def wait_for_everyone() -> None:
            return None

        @staticmethod
        def unwrap_model(model):
            return model

        @staticmethod
        def save(state, path) -> None:
            torch.save(state, path)

    config = replace(
        CurveNavConfig(),
        training=replace(CurveNavConfig().training, output_dir=str(tmp_path)),
    )
    model = nn.Linear(2, 2)
    optimizer = torch.optim.AdamW(model.parameters())
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
    ema = ExponentialMovingAverage(model)
    accelerator = FakeAccelerator()

    _save_checkpoint(
        accelerator, model, optimizer, scheduler, ema, config, epoch=20, step=1500
    )
    _save_checkpoint(
        accelerator, model, optimizer, scheduler, ema, config, epoch=40, step=3000
    )

    files = sorted(path.name for path in tmp_path.iterdir())
    assert files == ["checkpoint.pt"]
    state = torch.load(tmp_path / "checkpoint.pt", weights_only=False)
    assert state["step"] == 3000
    assert state["extra"]["epoch"] == 40


def test_restore_training_state_recovers_all_update_state() -> None:
    config = CurveNavConfig()
    source = nn.Linear(2, 2)
    source_optimizer = torch.optim.AdamW(source.parameters())
    source_scheduler = torch.optim.lr_scheduler.LambdaLR(
        source_optimizer, lambda _: 1.0
    )
    source_ema = ExponentialMovingAverage(source)
    loss = source(torch.ones(1, 2)).sum()
    loss.backward()
    source_optimizer.step()
    source_scheduler.step()
    source_ema.update()
    checkpoint = checkpoint_state(
        source, config, step=7, optimizer=source_optimizer
    )
    checkpoint["scheduler"] = source_scheduler.state_dict()
    checkpoint["ema"] = source_ema.state_dict()

    target = nn.Linear(2, 2)
    target_optimizer = torch.optim.AdamW(target.parameters())
    target_scheduler = torch.optim.lr_scheduler.LambdaLR(
        target_optimizer, lambda _: 1.0
    )
    target_ema = ExponentialMovingAverage(target)
    step = restore_training_state(
        checkpoint,
        target,
        target_optimizer,
        target_scheduler,
        target_ema,
        config,
    )

    assert step == 7
    assert target_scheduler.state_dict() == source_scheduler.state_dict()
    assert target_ema.num_updates == source_ema.num_updates
    for target_parameter, source_parameter in zip(
        target.parameters(), source.parameters(), strict=True
    ):
        torch.testing.assert_close(target_parameter, source_parameter)


def test_skipped_optimizer_step_does_not_advance_scheduler_or_ema() -> None:
    class FakeAccelerator:
        optimizer_step_was_skipped = True

    model = nn.Linear(2, 2)
    optimizer = torch.optim.AdamW(model.parameters())
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
    ema = ExponentialMovingAverage(model)

    advanced = _advance_optimizer_state(FakeAccelerator(), scheduler, ema)

    assert not advanced
    assert ema.num_updates == 0
    assert scheduler.last_epoch == 0
