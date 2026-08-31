"""Compare mapped local-trajectory outputs under one geometry contract."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch import Tensor

from curvenav.config import CurveNavConfig
from curvenav.config_io import load_config
from curvenav.data.privileged import SourceConfigurationSpaceQuery
from curvenav.evaluation.metrics import (
    summarize_configuration_safety,
    summarize_metrics,
    trajectory_metrics,
)
from curvenav.evaluation.protocol import summarize_strata


def load_common_protocol(
    common: np.lib.npyio.NpzFile,
) -> tuple[Tensor, Tensor, np.ndarray]:
    required = {
        "axis",
        "origin_xy",
        "point_goal",
        "reference_path",
        "route_id",
        "route_yaw",
        "scene_id",
    }
    missing = sorted(required - set(common.files))
    if missing:
        raise ValueError(
            "common evaluation set lacks metric protocol: "
            f"{missing}; regenerate its protocol metadata instead of guessing axes"
        )
    if str(common["axis"]) != "x_forward_y_left":
        raise ValueError("common evaluation axis must be x_forward_y_left")
    reference = np.asarray(common["reference_path"], dtype=np.float32)
    point_goal = np.asarray(common["point_goal"], dtype=np.float32)
    scene_id = common["scene_id"]
    return torch.from_numpy(reference), torch.from_numpy(point_goal), scene_id


class CommonSourceConfiguration:
    """Common-set provenance adapter over the one source C-space query."""

    def __init__(self, common: np.lib.npyio.NpzFile, source_root: Path) -> None:
        routes = tuple(sorted(str(route_id) for route_id in np.unique(common["route_id"])))
        route_index = {route: index for index, route in enumerate(routes)}
        self.grid_index = torch.tensor(
            [route_index[str(route_id)] for route_id in common["route_id"]],
            dtype=torch.int64,
        )
        self.origin_xy = torch.from_numpy(
            np.asarray(common["origin_xy"], dtype=np.float32)
        )
        self.yaw_rad = torch.from_numpy(
            np.asarray(common["route_yaw"], dtype=np.float32)
        )
        self.query = SourceConfigurationSpaceQuery.from_paths(
            tuple((source_root / route).parent / "navigation_grid.npz" for route in routes)
        )

    def measure(self, paths: Tensor, horizon_m: float) -> dict[str, Tensor]:
        return self.query.safety_metrics(
            paths,
            self.grid_index.to(paths.device),
            self.origin_xy.to(paths.device),
            self.yaw_rad.to(paths.device),
            horizon_m,
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
    safety: CommonSourceConfiguration,
    planning_horizon_m: float,
) -> dict[str, Tensor]:
    metrics = trajectory_metrics(path, reference, point_goal)
    metrics.update(safety.measure(path, planning_horizon_m))
    reference_safety = safety.measure(reference, planning_horizon_m)
    metrics.update(
        {f"reference_{name}": value for name, value in reference_safety.items()}
    )
    progress = torch.linspace(0.0, 1.0, reference.shape[1])[None, :, None]
    straight = reference[:, :1] + progress * (
        reference[:, -1:] - reference[:, :1]
    )
    straight_safety = safety.measure(straight, planning_horizon_m)
    metrics.update(
        {f"straight_{name}": value for name, value in straight_safety.items()}
    )
    return metrics


def validate_reference_safety(
    reference: Tensor,
    safety: CommonSourceConfiguration,
    planning_horizon_m: float,
) -> dict[str, float]:
    """Reject a common set whose expert route disagrees with source geometry."""
    metrics = safety.measure(reference, planning_horizon_m)
    collision = float(metrics["footprint_collision"].float().mean())
    margin = float(metrics["safety_margin_violation"].float().mean())
    coverage = float(metrics["path_field_coverage_fraction"].float().mean())
    if collision > 0.0 or margin > 0.0 or coverage < 1.0:
        raise ValueError(
            "common expert paths do not match the frozen source geometry: "
            f"collision_fraction={collision:.6f}, "
            f"margin_violation_fraction={margin:.6f}, "
            f"path_field_coverage_fraction={coverage:.6f}"
        )
    return {
        "expert_footprint_collision_fraction": collision,
        "expert_safety_margin_violation_fraction": margin,
        "expert_path_field_coverage_fraction": coverage,
    }


def _summary(metrics: dict[str, Tensor]) -> dict[str, object]:
    return {
        **summarize_metrics(metrics),
        **summarize_configuration_safety(metrics),
        "strata": summarize_strata(metrics),
    }


def _select(metrics: dict[str, Tensor], selected: Tensor) -> dict[str, Tensor]:
    return {name: value[selected] for name, value in metrics.items()}


def _scene_robustness(by_scene: dict[str, dict[str, object]]) -> dict[str, float]:
    definitions = {
        "fixed_horizon_ade_m_mean": "max",
        "fixed_horizon_fde_m_mean": "max",
        "goal_progress_regret_m_mean": "max",
        "footprint_collision_fraction": "max",
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
    source_root: Path,
    result_paths: list[Path],
    output_path: Path,
) -> dict[str, object]:
    common = np.load(common_path, allow_pickle=False)
    horizon = config.data.future_steps * config.data.expert_waypoint_spacing_m
    reference, point_goal, scene_id = load_common_protocol(common)
    safety = CommonSourceConfiguration(common, source_root)
    reference_contract = validate_reference_safety(reference, safety, horizon)

    models: dict[str, object] = {}
    for result_path in result_paths:
        name, path, checkpoint = _canonical_paths(result_path, len(reference))
        if name in models:
            raise ValueError(f"duplicate model output: {name}")
        metrics = _measure(path, reference, point_goal, safety, horizon)
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
        "safety_source": str(source_root.resolve()),
        "reference_contract": reference_contract,
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
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("results", type=Path, nargs="+")
    args = parser.parse_args()
    compare_outputs(
        load_config(args.config),
        args.common,
        args.source,
        args.results,
        args.output,
    )


if __name__ == "__main__":
    main()
