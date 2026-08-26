"""YAML-to-dataclass loading with an explicit, reviewable contract."""

from pathlib import Path
from typing import Any, Mapping

import yaml

from curvenav.config import (
    CurveNavConfig,
    DataConfig,
    DepthEncoderConfig,
    TrajectoryFlowConfig,
    ConditionEncoderConfig,
    PointGoalEncoderConfig,
    TrainingConfig,
    TrajectoryConfig,
)


def config_from_mapping(raw: Mapping[str, Any]) -> CurveNavConfig:
    """Build a validated config without silently accepting unknown sections."""
    allowed_top_level = {"data", "model", "training"}
    unknown_top_level = set(raw) - allowed_top_level
    if unknown_top_level:
        raise ValueError(f"unknown top-level config keys: {sorted(unknown_top_level)}")

    model = raw.get("model", {})
    if not isinstance(model, Mapping):
        raise TypeError("model config must be a mapping")
    allowed_model = {
        "trajectory",
        "depth_encoder",
        "point_goal_encoder",
        "condition_encoder",
        "trajectory_flow",
    }
    unknown_model = set(model) - allowed_model
    if unknown_model:
        raise ValueError(f"unknown model config keys: {sorted(unknown_model)}")

    trajectory = dict(model.get("trajectory", {}))
    config = CurveNavConfig(
        data=DataConfig(**raw.get("data", {})),
        trajectory=TrajectoryConfig(**trajectory),
        depth_encoder=DepthEncoderConfig(**model.get("depth_encoder", {})),
        point_goal_encoder=PointGoalEncoderConfig(
            **model.get("point_goal_encoder", {})
        ),
        condition_encoder=ConditionEncoderConfig(**model.get("condition_encoder", {})),
        trajectory_flow=TrajectoryFlowConfig(**model.get("trajectory_flow", {})),
        training=TrainingConfig(**raw.get("training", {})),
    )
    config.validate()
    return config


def load_config(path: str | Path) -> CurveNavConfig:
    """Load a CurveNav YAML file."""
    with Path(path).open("r", encoding="utf-8") as stream:
        raw = yaml.safe_load(stream)
    if not isinstance(raw, Mapping):
        raise TypeError("the YAML root must be a mapping")
    return config_from_mapping(raw)
