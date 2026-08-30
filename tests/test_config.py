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
                "trajectory": {
                    "num_heading_control_points": 8,
                },
                "trajectory_decoder": {"transformer_layers": 3},
            },
        }
    )
    assert config.data.root == "data/policy_dataset"
    assert config.data.frame_spacing_m == 0.45
    assert config.data.future_steps == 24
    assert config.trajectory.num_heading_control_points == 8
    assert config.trajectory_decoder.transformer_layers == 3


def test_rejects_removed_source_specific_data_config() -> None:
    with pytest.raises(TypeError, match="training_sources"):
        config_from_mapping({"data": {"training_sources": []}})
    with pytest.raises(TypeError, match="frame_skip"):
        config_from_mapping({"data": {"frame_skip": 2}})


def test_rejects_removed_architecture_switches() -> None:
    for key in (
        "generator",
        "fusion",
        "flow_matching",
        "trajectory_flow",
        "trajectory_scorer",
    ):
        with pytest.raises(ValueError, match="unknown model config keys"):
            config_from_mapping({"model": {key: {}}})
    with pytest.raises(TypeError, match="scale_xy"):
        config_from_mapping({"model": {"trajectory": {"scale_xy": [3.0, 3.0]}}})
    with pytest.raises(TypeError, match="normalization_scale_m"):
        config_from_mapping({"model": {"trajectory": {"normalization_scale_m": 4.0}}})
    with pytest.raises(TypeError, match="control_point_mean_xy_m"):
        config_from_mapping(
            {"model": {"trajectory": {"control_point_mean_xy_m": [0.0] * 13}}}
        )
    with pytest.raises(TypeError, match="maximum_curvature_inv_m"):
        config_from_mapping(
            {"model": {"trajectory": {"maximum_curvature_inv_m": 4.0}}}
        )
    with pytest.raises(ValueError, match="unknown model config keys"):
        config_from_mapping({"model": {"trajectory_evaluator": {}}})


def test_rejects_removed_diagnostic_training_options() -> None:
    with pytest.raises(TypeError, match="overfit_steps"):
        config_from_mapping({"training": {"overfit_steps": 1}})
    with pytest.raises(TypeError, match="checkpoint_path"):
        config_from_mapping({"training": {"checkpoint_path": "unused.pt"}})


def test_training_batch_contract_is_global_and_exact() -> None:
    config = config_from_mapping(
        {
            "training": {
                "global_batch_size": 1024,
                "samples_per_epoch": 40960,
            }
        }
    )
    assert config.training.global_batch_size == 1024
    assert config.training.per_device_batch_size == 256
    with pytest.raises(TypeError, match="micro_batch_size"):
        config_from_mapping({"training": {"micro_batch_size": 128}})
    with pytest.raises(ValueError, match="samples_per_epoch"):
        config_from_mapping({"training": {"samples_per_epoch": 40000}})
    with pytest.raises(ValueError, match="cannot exceed"):
        config_from_mapping(
            {"training": {"global_batch_size": 32, "per_device_batch_size": 64}}
        )


def test_rejects_invalid_trajectory_contract() -> None:
    with pytest.raises(TypeError, match="target_spline_degree"):
        config_from_mapping({"model": {"trajectory": {"target_spline_degree": 2}}})
    with pytest.raises(ValueError, match="clamped cubic"):
        config_from_mapping({"model": {"trajectory": {"spline_degree": 2}}})
    with pytest.raises(ValueError, match="cover"):
        config_from_mapping({"model": {"trajectory": {"num_path_points": 4}}})
    with pytest.raises(ValueError, match="exactly eight"):
        replace(
            CurveNavConfig(),
            trajectory=replace(
                CurveNavConfig().trajectory,
                num_heading_control_points=7,
            ),
        ).validate()

    with pytest.raises(TypeError, match="flow_steps"):
        config_from_mapping(
            {"model": {"trajectory_decoder": {"flow_steps": 7}}}
        )
    with pytest.raises(ValueError, match="three equal refinement stages"):
        config_from_mapping(
            {"model": {"trajectory_decoder": {"transformer_layers": 11}}}
        )


def test_rejects_non_production_observation_frame_count() -> None:
    with pytest.raises(ValueError, match="four depth observations"):
        config_from_mapping({"data": {"observation_frames": 3}})
