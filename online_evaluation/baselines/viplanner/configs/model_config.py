"""Minimal ViPlanner model configuration used during evaluation."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass
class DataConfig:
    max_depth: float = 15.0
    max_goal_distance: float = 15.0


@dataclass
class ModelConfig:
    sem: bool = True
    rgb: bool = False
    img_input_size: list[int] = field(default_factory=lambda: [360, 640])
    in_channel: int = 16
    knodes: int = 5
    pre_train_sem: bool = True
    pre_train_freeze: bool = True
    decoder_small: bool = False
    data_cfg: DataConfig | list[DataConfig] = field(default_factory=DataConfig)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "ModelConfig":
        with Path(path).open() as handle:
            values = yaml.safe_load(handle)["config"]
        data_values = values.get("data_cfg", {})
        if isinstance(data_values, list):
            values["data_cfg"] = [DataConfig(**item) for item in data_values]
        else:
            values["data_cfg"] = DataConfig(**data_values)
        return cls(**values)
