"""Strict configuration for the standalone stage-two trajectory critic."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import yaml


@dataclass(frozen=True)
class CriticPaths:
    policy_config: Path
    policy_checkpoint: Path
    dataset_root: Path
    sidecar_root: Path
    output_dir: Path


@dataclass(frozen=True)
class CriticTrainingConfig:
    seed: int = 42
    batch_size: int = 128
    epochs: int = 40
    num_workers: int = 8
    learning_rate: float = 3e-4
    weight_decay: float = 1e-2
    warmup_epochs: int = 2
    minimum_learning_rate_factor: float = 0.05
    grad_clip_norm: float = 1.0
    auxiliary_weight: float = 0.25
    overfit_steps: int = 800
    overfit_batch_size: int = 8
    overfit_max_loss_ratio: float = 0.1
    log_every_steps: int = 20

    def validate(self) -> None:
        positive_integers = {
            "batch_size": self.batch_size,
            "epochs": self.epochs,
            "num_workers": self.num_workers,
            "overfit_steps": self.overfit_steps,
            "overfit_batch_size": self.overfit_batch_size,
            "log_every_steps": self.log_every_steps,
        }
        invalid = [name for name, value in positive_integers.items() if value < 1]
        if invalid:
            raise ValueError(f"critic training values must be positive: {invalid}")
        if not 0 <= self.seed < 2**32:
            raise ValueError("critic seed must fit uint32")
        if not 0 <= self.warmup_epochs < self.epochs:
            raise ValueError("critic warmup_epochs must be in [0, epochs)")
        if not 0 < self.minimum_learning_rate_factor <= 1:
            raise ValueError("critic minimum_learning_rate_factor must be in (0, 1]")
        for name, value in (
            ("learning_rate", self.learning_rate),
            ("grad_clip_norm", self.grad_clip_norm),
            ("auxiliary_weight", self.auxiliary_weight),
        ):
            if value <= 0:
                raise ValueError(f"critic {name} must be positive")
        if self.weight_decay < 0:
            raise ValueError("critic weight_decay cannot be negative")
        if not 0 < self.overfit_max_loss_ratio < 1:
            raise ValueError("critic overfit_max_loss_ratio must be in (0, 1)")


@dataclass(frozen=True)
class CriticExperimentConfig:
    paths: CriticPaths
    training: CriticTrainingConfig

    def validate(self) -> None:
        self.training.validate()
        for name, path in (
            ("policy_config", self.paths.policy_config),
            ("policy_checkpoint", self.paths.policy_checkpoint),
            ("dataset_root", self.paths.dataset_root),
            ("sidecar_root", self.paths.sidecar_root),
        ):
            if not path.exists():
                raise FileNotFoundError(f"critic {name} does not exist: {path}")


def _reject_unknown(raw: Mapping[str, Any], allowed: set[str], context: str) -> None:
    unknown = set(raw) - allowed
    if unknown:
        raise ValueError(f"unknown {context} keys: {sorted(unknown)}")


def load_critic_config(path: str | Path) -> CriticExperimentConfig:
    """Load one critic experiment without extending the policy config schema."""
    config_path = Path(path).expanduser().resolve()
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, Mapping):
        raise TypeError("critic YAML root must be a mapping")
    _reject_unknown(raw, {"paths", "training"}, "critic top-level")
    paths_raw = raw.get("paths", {})
    training_raw = raw.get("training", {})
    if not isinstance(paths_raw, Mapping) or not isinstance(training_raw, Mapping):
        raise TypeError("critic paths and training sections must be mappings")
    required_paths = {
        "policy_config",
        "policy_checkpoint",
        "dataset_root",
        "sidecar_root",
        "output_dir",
    }
    _reject_unknown(paths_raw, required_paths, "critic paths")
    missing = required_paths - set(paths_raw)
    if missing:
        raise ValueError(f"missing critic paths: {sorted(missing)}")
    training_fields = set(CriticTrainingConfig.__dataclass_fields__)
    _reject_unknown(training_raw, training_fields, "critic training")

    def resolve(value: Any) -> Path:
        candidate = Path(str(value)).expanduser()
        return (config_path.parent / candidate).resolve() if not candidate.is_absolute() else candidate

    config = CriticExperimentConfig(
        paths=CriticPaths(**{name: resolve(paths_raw[name]) for name in required_paths}),
        training=CriticTrainingConfig(**training_raw),
    )
    config.validate()
    return config
