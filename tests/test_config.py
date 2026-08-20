import pytest

from curvenav.config_io import config_from_mapping


def test_base_mapping_is_planar() -> None:
    config = config_from_mapping(
        {
            "model": {
                "trajectory": {"scale_xy": [5.6, 2.5]},
                "rectified_flow": {
                    "inference_steps": 4,
                    "source_std_xy": [0.04, 0.12],
                },
            }
        }
    )
    assert config.rectified_flow.inference_steps == 4
    assert config.trajectory.scale_xy == (5.6, 2.5)
    assert config.rectified_flow.source_std_xy == (0.04, 0.12)


def test_explicit_data_sources_are_typed_and_weighted() -> None:
    config = config_from_mapping(
        {
            "data": {
                "training_sources": [
                    {"root": "sand", "split": "train", "weight": 0.5},
                    {"root": "hssd/train", "split": "all", "weight": 0.5},
                ],
                "validation_sources": [
                    {"root": "hssd/validation", "split": "all", "weight": 1.0}
                ],
            }
        }
    )
    assert tuple(source.root for source in config.data.training_sources) == (
        "sand",
        "hssd/train",
    )
    assert tuple(source.weight for source in config.data.training_sources) == (
        0.5,
        0.5,
    )


def test_rejects_removed_architecture_switches() -> None:
    with pytest.raises(ValueError, match="unknown model config keys"):
        config_from_mapping({"model": {"generator": {"backend": "ddpm"}}})

    with pytest.raises(ValueError, match="unknown model config keys"):
        config_from_mapping({"model": {"fusion": {"transformer_layers": 2}}})

    with pytest.raises(ValueError, match="unknown model config keys"):
        config_from_mapping({"model": {"flow_matching": {"inference_steps": 8}}})

    with pytest.raises(TypeError, match="control_scale_xy"):
        config_from_mapping(
            {"model": {"trajectory": {"control_scale_xy": [5.6, 2.5]}}}
        )

    with pytest.raises(TypeError, match="goal_scale_xy"):
        config_from_mapping({"model": {"trajectory": {"goal_scale_xy": [5.6, 2.5]}}})


def test_rejects_invalid_overfit_gate() -> None:
    with pytest.raises(ValueError, match="overfit_max_loss_ratio"):
        config_from_mapping({"training": {"overfit_max_loss_ratio": 1.0}})


def test_rejects_invalid_source_scale() -> None:
    with pytest.raises(ValueError, match="source_std_xy"):
        config_from_mapping(
            {"model": {"rectified_flow": {"source_std_xy": [0.04, 0.0]}}}
        )


def test_rejects_non_cubic_or_underdetermined_trajectory_contract() -> None:
    with pytest.raises(ValueError, match="cubic"):
        config_from_mapping({"model": {"trajectory": {"degree": 2}}})
    with pytest.raises(ValueError, match="num_path_points"):
        config_from_mapping(
            {"model": {"trajectory": {"num_control_points": 12, "num_path_points": 8}}}
        )


def test_rejects_non_production_depth_sequence_length() -> None:
    with pytest.raises(ValueError, match="exactly four"):
        config_from_mapping({"data": {"sequence_length": 3}})


def test_rejects_removed_max_frames_switch() -> None:
    with pytest.raises(TypeError, match="max_frames"):
        config_from_mapping({"model": {"depth_encoder": {"max_frames": 8}}})
