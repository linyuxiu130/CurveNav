from dataclasses import replace

import pytest

from curvenav.config import CurveNavConfig
from curvenav.config_io import config_from_mapping


def test_base_mapping_has_one_prepared_dataset_and_fixed_future_contract() -> None:
    config = config_from_mapping(
        {
            "data": {
                "root": "data/policy_dataset",
                "frame_spacing_m": 0.45,
                "future_steps": 24,
            },
            "model": {
                "trajectory": {"normalization_scale_m": 4.0},
                "trajectory_flow": {"transformer_layers": 3},
            },
        }
    )
    assert config.data.root == "data/policy_dataset"
    assert config.data.frame_spacing_m == 0.45
    assert config.data.future_steps == 24
    assert config.trajectory.normalization_scale_m == 4.0
    assert config.trajectory_flow.transformer_layers == 3


def test_rejects_removed_source_specific_data_config() -> None:
    with pytest.raises(TypeError, match="training_sources"):
        config_from_mapping({"data": {"training_sources": []}})
    with pytest.raises(TypeError, match="frame_skip"):
        config_from_mapping({"data": {"frame_skip": 2}})


def test_rejects_removed_architecture_switches() -> None:
    for key in ("generator", "fusion", "flow_matching", "trajectory_scorer"):
        with pytest.raises(ValueError, match="unknown model config keys"):
            config_from_mapping({"model": {key: {}}})
    with pytest.raises(TypeError, match="scale_xy"):
        config_from_mapping({"model": {"trajectory": {"scale_xy": [3.0, 3.0]}}})


def test_rejects_invalid_overfit_gate() -> None:
    with pytest.raises(ValueError, match="overfit_max_loss_ratio"):
        config_from_mapping({"training": {"overfit_max_loss_ratio": 1.0}})


def test_rejects_invalid_trajectory_contract() -> None:
    with pytest.raises(ValueError, match="cubic"):
        config_from_mapping({"model": {"trajectory": {"degree": 2}}})
    with pytest.raises(ValueError, match="num_path_points"):
        config_from_mapping(
            {"model": {"trajectory": {"num_control_points": 8, "num_path_points": 4}}}
        )
    with pytest.raises(ValueError, match="normalization_scale_m"):
        replace(CurveNavConfig(), trajectory=replace(CurveNavConfig().trajectory, normalization_scale_m=0)).validate()
    with pytest.raises(ValueError, match="exactly eight"):
        replace(
            CurveNavConfig(),
            trajectory=replace(CurveNavConfig().trajectory, num_control_points=5),
        ).validate()


def test_rejects_non_production_observation_frame_count() -> None:
    with pytest.raises(ValueError, match="four depth observations"):
        config_from_mapping({"data": {"observation_frames": 3}})
    with pytest.raises(ValueError, match="Dingo camera calibration"):
        config_from_mapping({"data": {"camera_height_m": 0.30}})
