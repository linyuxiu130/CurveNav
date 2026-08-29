"""Compare mapped local-trajectory outputs under one geometry contract."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
from torch import Tensor

from curvenav.config import CurveNavConfig
from curvenav.config_io import load_config
from curvenav.evaluation.metrics import (
    observed_safety_metrics,
    summarize_metrics,
    summarize_paired_safety,
    trajectory_metrics,
)
from curvenav.evaluation.protocol import summarize_strata


def load_common_geometry(
    common: np.lib.npyio.NpzFile,
    planning_horizon_m: float,
) -> tuple[Tensor, Tensor, Tensor, np.ndarray]:
    required = {
        "axis",
        "configuration_field",
        "configuration_channels",
        "configuration_extent_m",
        "point_goal",
        "reference_path",
        "scene_id",
    }
    missing = sorted(required - set(common.files))
    if missing:
        raise ValueError(
            "common evaluation set lacks frozen metric geometry: "
            f"{missing}; regenerate it instead of reconstructing camera semantics "
            "inside the comparator"
        )
    if str(common["axis"]) != "x_forward_y_left":
        raise ValueError("common evaluation axis must be x_forward_y_left")
    if not math.isclose(
        float(common["configuration_extent_m"]),
        planning_horizon_m,
        rel_tol=0.0,
        abs_tol=1e-6,
    ):
        raise ValueError("common configuration-space extent does not match config")
    channels = [str(value) for value in common["configuration_channels"].tolist()]
    if channels != [
        "signed_clearance_m",
        "gradient_x",
        "gradient_y",
        "observed",
        "forbidden",
    ]:
        raise ValueError("common configuration-space channels are invalid")
    field = np.asarray(common["configuration_field"], dtype=np.float32)
    reference = np.asarray(common["reference_path"], dtype=np.float32)
    point_goal = np.asarray(common["point_goal"], dtype=np.float32)
    scene_id = common["scene_id"]
    if field.shape != (len(reference), 5, 64, 64):
        raise ValueError("common configuration_field must have shape [N,5,64,64]")
    return (
        torch.from_numpy(field),
        torch.from_numpy(reference),
        torch.from_numpy(point_goal),
        scene_id,
    )


def _canonical_paths(result_path: Path, samples: int) -> tuple[str, Tensor, str]:
    result = np.load(result_path, allow_pickle=False)
    if "axis" not in result.files or str(result["axis"]) != "x_forward_y_left":
        raise ValueError(f"mapped output axis is missing or invalid: {result_path}")
    path = np.asarray(result["base_path"], dtype=np.float32).copy()
    length = np.asarray(result["base_length"], dtype=np.int64)
    if path.shape[0] != samples or length.shape != (samples,):
        raise ValueError(f"sample count mismatch in {result_path}")
    if np.any(length < 3) or np.any(length > path.shape[1]):
        raise ValueError(f"invalid path lengths in {result_path}")
    for index, count in enumerate(length):
        path[index, count:] = path[index, count - 1]
    if not np.isfinite(path).all():
        raise ValueError(f"non-finite trajectory in {result_path}")
    return str(result["model"]), torch.from_numpy(path), str(result["checkpoint"])


def _measure(
    path: Tensor,
    reference: Tensor,
    point_goal: Tensor,
    configuration_field: Tensor,
    planning_horizon_m: float,
) -> dict[str, Tensor]:
    metrics = trajectory_metrics(path, reference, point_goal)
    metrics.update(
        observed_safety_metrics(path, configuration_field, planning_horizon_m)
    )
    reference_safety = observed_safety_metrics(
        reference, configuration_field, planning_horizon_m
    )
    metrics.update(
        {f"reference_{name}": value for name, value in reference_safety.items()}
    )
    progress = torch.linspace(0.0, 1.0, reference.shape[1])[None, :, None]
    straight = reference[:, :1] + progress * (
        reference[:, -1:] - reference[:, :1]
    )
    straight_safety = observed_safety_metrics(
        straight, configuration_field, planning_horizon_m
    )
    metrics.update(
        {f"straight_{name}": value for name, value in straight_safety.items()}
    )
    return metrics


def _summary(metrics: dict[str, Tensor]) -> dict[str, object]:
    return {
        **summarize_metrics(metrics),
        **summarize_paired_safety(metrics),
        "strata": summarize_strata(metrics),
    }


def _select(metrics: dict[str, Tensor], selected: Tensor) -> dict[str, Tensor]:
    return {name: value[selected] for name, value in metrics.items()}


def _scene_robustness(by_scene: dict[str, dict[str, object]]) -> dict[str, float]:
    definitions = {
        "fixed_horizon_ade_m_mean": "max",
        "fixed_horizon_fde_m_mean": "max",
        "goal_progress_regret_m_mean": "max",
        "observed_footprint_collision_fraction": "max",
        "extra_footprint_collision_fraction_over_reference": "max",
        "horizon_coverage_fraction_mean": "min",
    }
    result = {}
    for metric, worst_direction in definitions.items():
        values = [float(summary[metric]) for summary in by_scene.values()]
        result[f"{metric}_scene_macro"] = sum(values) / len(values)
        result[f"{metric}_worst_scene"] = (
            max(values) if worst_direction == "max" else min(values)
        )
    return result


def compare_outputs(
    config: CurveNavConfig,
    common_path: Path,
    result_paths: list[Path],
    output_path: Path,
) -> dict[str, object]:
    common = np.load(common_path, allow_pickle=False)
    horizon = config.data.future_steps * config.data.expert_waypoint_spacing_m
    field, reference, point_goal, scene_id = load_common_geometry(common, horizon)

    models: dict[str, object] = {}
    for result_path in result_paths:
        name, path, checkpoint = _canonical_paths(result_path, len(reference))
        if name in models:
            raise ValueError(f"duplicate model output: {name}")
        metrics = _measure(path, reference, point_goal, field, horizon)
        scenes = {}
        for scene in sorted(np.unique(scene_id).tolist()):
            selected = torch.from_numpy(scene_id == scene)
            scenes[str(scene)] = _summary(_select(metrics, selected))
        models[name] = {
            "checkpoint": checkpoint,
            "samples": len(path),
            **_summary(metrics),
            "scene_robustness": _scene_robustness(scenes),
            "by_scene": scenes,
        }

    report = {
        "protocol": "curvenav_common_metric_local_validation",
        "common_dataset": str(common_path.resolve()),
        "models": models,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, sort_keys=True), flush=True)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("common", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("results", type=Path, nargs="+")
    args = parser.parse_args()
    compare_outputs(
        load_config(args.config),
        args.common,
        args.results,
        args.output,
    )


if __name__ == "__main__":
    main()
