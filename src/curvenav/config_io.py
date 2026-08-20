"""YAML-to-dataclass loading with an explicit, reviewable schema."""

from pathlib import Path
from typing import Any, Mapping

import yaml

from curvenav.config import (
    CurveNavConfig,
    ConditionEncoderConfig,
    DataConfig,
    DataSourceConfig,
    DepthEncoderConfig,
    FieldConfig,
    GoalEncoderConfig,
    MotionEncoderConfig,
    RectifiedFlowConfig,
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
        "goal_encoder",
        "motion_encoder",
        "condition_encoder",
        "field",
        "rectified_flow",
    }
    unknown_model = set(model) - allowed_model
    if unknown_model:
        raise ValueError(f"unknown model config keys: {sorted(unknown_model)}")

    trajectory = dict(model.get("trajectory", {}))
    if "scale_xy" in trajectory:
        trajectory["scale_xy"] = tuple(trajectory["scale_xy"])
    rectified_flow = dict(model.get("rectified_flow", {}))
    if "source_std_xy" in rectified_flow:
        rectified_flow["source_std_xy"] = tuple(rectified_flow["source_std_xy"])

    data = dict(raw.get("data", {}))
    for key in ("training_sources", "validation_sources"):
        if key in data:
            sources = data[key]
            if not isinstance(sources, list):
                raise TypeError(f"data.{key} must be a list")
            data[key] = tuple(DataSourceConfig(**source) for source in sources)

    config = CurveNavConfig(
        data=DataConfig(**data),
        trajectory=TrajectoryConfig(**trajectory),
        depth_encoder=DepthEncoderConfig(**model.get("depth_encoder", {})),
        goal_encoder=GoalEncoderConfig(**model.get("goal_encoder", {})),
        motion_encoder=MotionEncoderConfig(**model.get("motion_encoder", {})),
        condition_encoder=ConditionEncoderConfig(**model.get("condition_encoder", {})),
        field=FieldConfig(**model.get("field", {})),
        rectified_flow=RectifiedFlowConfig(**rectified_flow),
        training=TrainingConfig(**raw.get("training", {})),
    )
    config.validate()
    return config


def load_config(path: str | Path) -> CurveNavConfig:
    """Load a CurveNav YAML file."""
    with Path(path).open("r", encoding="utf-8") as stream:
        raw = yaml.safe_load(stream) or {}
    if not isinstance(raw, Mapping):
        raise TypeError("the YAML root must be a mapping")
    return config_from_mapping(raw)
