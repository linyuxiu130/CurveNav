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
    """Choose deterministic median cases plus the hardest visible detour."""
    strata = evaluation_strata(metrics)
    requested = (
        ("forward_open_median", "forward_open", 0.5),
        ("forward_visible_detour_median", "forward_visible_detour", 0.5),
        ("forward_visible_detour_hard", "forward_visible_detour", 1.0),
        ("rear_goal_median", "rear_goal", 0.5),
        ("expert_moves_away_median", "expert_moves_away_from_goal", 0.5),
    )
    selected = []
    for label, stratum, quantile in requested:
        index = _representative_index(strata[stratum], metrics["ade_m"], quantile)
        if index is not None and index not in {item[1] for item in selected}:
            selected.append((label, index))
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
        valid = sample_data["obstacle_valid"][index]
        goal = sample_data["point_goal"][index]
        records.append(
            {
                "label": label,
                "sample_index": index,
                "ade_m": float(metrics["ade_m"][index]),
                "goal_distance_m": float(metrics["point_goal_distance_m"][index]),
                "goal_bearing_deg": float(
                    np.degrees(np.arctan2(float(goal[1]), float(goal[0])))
                ),
                "predicted_min_clearance_m": float(
                    metrics["observed_min_clearance_m"][index]
                ),
                "reference_min_clearance_m": float(
                    metrics["reference_observed_min_clearance_m"][index]
                ),
                "point_goal": goal.tolist(),
                "predicted_path": sample_data["predicted_path"][index].tolist(),
                "reference_path": sample_data["reference_path"][index].tolist(),
                "obstacle_points": sample_data["obstacle_points"][index][valid].tolist(),
            }
        )
    path = output_dir / "strict-offline-cases.json"
    path.write_text(json.dumps(records, indent=2) + "\n", encoding="utf-8")
    return path
