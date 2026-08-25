from dataclasses import replace

import pytest
from torch import nn

from curvenav.config import CurveNavConfig
from curvenav.training import checkpoint_state, policy_contract, validate_policy_contract


def test_checkpoint_records_the_flow_and_scorer_contract() -> None:
    config = CurveNavConfig()
    checkpoint = checkpoint_state(nn.Linear(2, 2), config, step=0)
    contract = checkpoint["policy_contract"]
    assert checkpoint["checkpoint_type"] == "curvenav_local_policy"
    assert contract["trajectory_flow_candidates"] == 8
    assert (
        contract["trajectory_flow_type"]
        == "conditional_bspline_rectified_flow_heun"
    )
    assert (
        contract["trajectory_scorer_type"]
        == "conditional_trajectory_group_quality"
    )
    assert (
        contract["training_objective"]
        == "rectified_flow_plus_arc_path_tangent_and_group_quality"
    )
    assert contract["num_control_points"] == 8
    assert contract["path_sampling"] == "uniform_metric_arc_progress"
    assert contract["visual_compression"] == "learned_queries_16_tokens_per_depth_frame"
    assert contract["observation_to_current"] == "planar_rigid_transform_used_for_depth_token_alignment"


def test_checkpoint_contract_catches_geometry_mismatch() -> None:
    config = CurveNavConfig()
    checkpoint = {
        "checkpoint_type": "curvenav_local_policy",
        "policy_contract": policy_contract(config),
    }
    validate_policy_contract(checkpoint, config)
    changed = replace(
        config, trajectory=replace(config.trajectory, normalization_scale_m=5.0)
    )
    with pytest.raises(ValueError, match="trajectory_scale_xy"):
        validate_policy_contract(checkpoint, changed)


def test_checkpoint_requires_current_policy_type() -> None:
    with pytest.raises(ValueError, match="checkpoint_type"):
        validate_policy_contract(
            {"checkpoint_type": "unrelated_policy", "policy_contract": policy_contract(CurveNavConfig())},
            CurveNavConfig(),
        )
