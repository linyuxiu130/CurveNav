"""Compact artifacts for inspecting strict offline evaluation cases."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from torch import Tensor

from curvenav.evaluation.protocol import evaluation_strata


def _representative_index(mask: Tensor, score: Tensor, quantile: float) -> int | None:
    indices = mask.nonzero(as_tuple=False).flatten()
    if not len(indices):
        return None
    ordered = indices[score[indices].argsort()]
    position = round(quantile * (len(ordered) - 1))
    return int(ordered[position].item())


def select_cases(metrics: dict[str, Tensor]) -> list[tuple[str, int]]:
    """Choose deterministic behavior and collision-diagnosis cases."""
    strata = evaluation_strata(metrics)
    requested = (
        ("forward_direct_median", "forward_direct", 0.5),
        ("forward_detour_median", "forward_detour", 0.5),
        ("forward_detour_hard", "forward_detour", 1.0),
        ("rear_goal_median", "rear_goal", 0.5),
        ("expert_moves_away_median", "expert_moves_away_from_goal", 0.5),
    )
    selected = []
    for label, stratum, quantile in requested:
        index = _representative_index(
            strata[stratum], metrics["fixed_horizon_ade_m"], quantile
        )
        if index is not None and index not in {item[1] for item in selected}:
            selected.append((label, index))
    index = _representative_index(
        metrics["footprint_collision"].bool(),
        metrics["min_clearance_m"],
        0.0,
    )
    if index is not None:
        selected.append(("source_collision_worst", index))
    return selected


def write_case_report(
    output_dir: Path,
    metrics: dict[str, Tensor],
    sample_data: dict[str, Tensor],
) -> Path:
    """Write small, renderer-independent JSON containing representative geometry."""
    output_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for label, index in select_cases(metrics):
        goal = sample_data["point_goal"][index]
        field = sample_data["configuration_field"][index].float()
        records.append(
            {
                "label": label,
                "sample_index": index,
                "fixed_horizon_ade_m": float(
                    metrics["fixed_horizon_ade_m"][index]
                ),
                "fixed_horizon_fde_m": float(
                    metrics["fixed_horizon_fde_m"][index]
                ),
                "horizon_coverage_fraction": float(
                    metrics["horizon_coverage_fraction"][index]
                ),
                "goal_distance_m": float(metrics["point_goal_distance_m"][index]),
                "goal_bearing_deg": float(
                    np.degrees(np.arctan2(float(goal[1]), float(goal[0])))
                ),
                "predicted_min_clearance_m": float(
                    metrics["min_clearance_m"][index]
                ),
                "reference_min_clearance_m": float(
                    metrics["reference_min_clearance_m"][index]
                ),
                "depth_min_clearance_m": float(
                    metrics["depth_min_clearance_m"][index]
                ),
                "configuration_extent_m": float(
                    sample_data["configuration_extent_m"][index]
                ),
                "point_goal": goal.tolist(),
                "predicted_path": sample_data["predicted_path"][index].tolist(),
                "reference_path": sample_data["reference_path"][index].tolist(),
                "configuration_clearance_m": field[0].tolist(),
                "configuration_ray_coverage": (field[1] > 0.5).tolist(),
                "configuration_forbidden": (field[2] > 0.5).tolist(),
            }
        )
    path = output_dir / "offline-cases.json"
    path.write_text(json.dumps(records, indent=2) + "\n", encoding="utf-8")
    return path
