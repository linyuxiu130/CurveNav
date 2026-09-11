#!/usr/bin/env python3
"""Aggregate closed-loop trajectory traces without changing benchmark metrics."""

from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import json
import math
import os
from pathlib import Path
from typing import Iterable

import numpy as np

from navbench.metrics import success_weighted_path_length
from navbench.trajectory_trace import TRACE_SCHEMA_VERSION


EPISODE_FIELDS = (
    "run", "model", "split", "scene", "episode_idx", "success", "spl",
    "initial_goal_distance_m", "final_goal_distance_m",
    "executed_path_length_m", "elapsed_simulation_time_s",
    "net_goal_progress_m", "goal_progress_fraction", "progress_efficiency",
    "path_length_efficiency", "detour_overhead_m",
    "goal_distance_backtrack_m", "goal_distance_total_variation_m",
    "progress_monotonicity", "net_displacement_m", "path_tortuosity",
    "mean_speed_mps", "rms_speed_mps", "p95_speed_mps",
    "mean_abs_goal_bearing_rad", "p95_abs_goal_bearing_rad",
    "mpc_linear_command_rms", "mpc_angular_command_rms",
    "mpc_angular_sign_changes", "plan_count", "mean_policy_ms",
    "p95_policy_ms", "mean_mpc_ms", "p95_mpc_ms",
    "mean_total_plan_ms", "p95_total_plan_ms", "mean_plan_arc_length_m",
    "mean_plan_endpoint_goal_gap_m",
    "mean_plan_endpoint_angular_error_rad",
    "p95_plan_peak_curvature_inv_m", "mean_plan_curvature_energy_inv_m",
)

SUMMARY_FIELDS = (
    "run", "model", "episodes", "success_rate", "mean_spl",
    "mean_elapsed_simulation_time_s", "mean_success_time_s",
    "median_final_goal_distance_m", "p95_final_goal_distance_m",
    "mean_goal_progress_fraction", "mean_progress_efficiency",
    "mean_path_length_efficiency", "mean_detour_overhead_m",
    "mean_goal_distance_backtrack_m", "mean_progress_monotonicity",
    "mean_path_tortuosity", "mean_speed_mps", "mean_plan_count",
    "median_episode_mean_total_plan_ms", "p95_episode_mean_total_plan_ms",
    "mean_plan_arc_length_m", "mean_plan_endpoint_goal_gap_m",
    "p95_plan_peak_curvature_inv_m", "mean_plan_curvature_energy_inv_m",
)


def _as_float(value: object) -> float:
    return float(np.asarray(value).item())


def _safe_ratio(numerator: float, denominator: float) -> float:
    if denominator == 0.0:
        return math.nan
    return numerator / denominator


def _finite(values: Iterable[float]) -> np.ndarray:
    array = np.asarray(list(values), dtype=np.float64)
    return array[np.isfinite(array)]


def _mean(values: Iterable[float]) -> float:
    array = _finite(values)
    return float(array.mean()) if array.size else math.nan


def _percentile(values: Iterable[float], percentile: float) -> float:
    array = _finite(values)
    return float(np.percentile(array, percentile)) if array.size else math.nan


def _rms(values: np.ndarray) -> float:
    array = np.asarray(values, dtype=np.float64)
    return float(np.sqrt(np.mean(np.square(array)))) if array.size else math.nan


def _plan_geometry(
    trajectories: np.ndarray,
    trajectory_lengths: np.ndarray,
    point_goals: np.ndarray,
) -> tuple[list[float], list[float], list[float], list[float], list[float]]:
    arc_lengths: list[float] = []
    endpoint_gaps: list[float] = []
    endpoint_angles: list[float] = []
    peak_curvatures: list[float] = []
    curvature_energies: list[float] = []
    if trajectories.ndim != 3 or trajectories.shape[-1] < 2:
        return (
            arc_lengths, endpoint_gaps, endpoint_angles,
            peak_curvatures, curvature_energies,
        )

    for padded, length, point_goal in zip(
        trajectories[..., :2], trajectory_lengths, point_goals[..., :2]
    ):
        trajectory = padded[:int(length)]
        if trajectory.shape[0] == 0:
            continue
        segments = np.diff(trajectory.astype(np.float64), axis=0)
        lengths = np.linalg.norm(segments, axis=-1)
        arc_lengths.append(float(lengths.sum()))
        endpoint = trajectory[-1].astype(np.float64)
        goal = point_goal.astype(np.float64)
        endpoint_gaps.append(float(np.linalg.norm(endpoint - goal)))
        endpoint_norm = float(np.linalg.norm(endpoint))
        goal_norm = float(np.linalg.norm(goal))
        if endpoint_norm > 0.0 and goal_norm > 0.0:
            cosine = float(np.dot(endpoint, goal) / (endpoint_norm * goal_norm))
            endpoint_angles.append(math.acos(float(np.clip(cosine, -1.0, 1.0))))

        if len(segments) < 2:
            continue
        first = segments[:-1]
        second = segments[1:]
        first_length = lengths[:-1]
        second_length = lengths[1:]
        spacing = 0.5 * (first_length + second_length)
        valid = (first_length > 0.0) & (second_length > 0.0) & (spacing > 0.0)
        if not np.any(valid):
            continue
        cross = first[:, 0] * second[:, 1] - first[:, 1] * second[:, 0]
        dot = np.sum(first * second, axis=-1)
        turning = np.arctan2(cross[valid], dot[valid])
        curvature = turning / spacing[valid]
        peak_curvatures.append(float(np.max(np.abs(curvature))))
        curvature_energies.append(float(np.sum(np.square(curvature) * spacing[valid])))
    return (
        arc_lengths, endpoint_gaps, endpoint_angles,
        peak_curvatures, curvature_energies,
    )


def _load_metric_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"success", "spl", "distance", "episode_idx"}
        if not reader.fieldnames or not required.issubset(reader.fieldnames):
            raise ValueError(f"invalid metric schema: {path}")
        rows = list(reader)
    episode_ids = [int(row["episode_idx"]) for row in rows]
    if any(episode_id < 0 for episode_id in episode_ids) or len(
        set(episode_ids)
    ) != len(episode_ids):
        raise ValueError(f"metric episode ids must be unique and non-negative: {path}")
    return sorted(rows, key=lambda row: int(row["episode_idx"]))


def _episode_row(
    run: str,
    model: str,
    split: str,
    scene: str,
    metric: dict[str, str],
    trace_path: Path,
) -> dict[str, object]:
    with np.load(trace_path, allow_pickle=False) as trace:
        schema = int(trace["schema_version"].item())
        if schema != TRACE_SCHEMA_VERSION:
            raise ValueError(
                f"trace schema {schema} != {TRACE_SCHEMA_VERSION}: {trace_path}"
            )
        episode_idx = int(trace["episode_idx"].item())
        if episode_idx != int(metric["episode_idx"]):
            raise ValueError(f"trace/metric episode mismatch: {trace_path}")
        success = _as_float(trace["success"])
        initial = _as_float(trace["initial_goal_distance_m"])
        path_length = _as_float(trace["executed_path_length_m"])
        elapsed = _as_float(trace["elapsed_simulation_time_s"])
        final_goal_vector = np.asarray(
            trace["terminal_pre_step_point_goal_robot_m"], dtype=np.float64
        )
        final_goal = float(np.linalg.norm(final_goal_vector[:2]))
        final_position = np.asarray(
            trace["terminal_pre_step_robot_position_world_m"], dtype=np.float64
        )
        step_positions = np.asarray(
            trace["step_robot_position_world_m"], dtype=np.float64
        )
        step_goals = np.asarray(trace["step_point_goal_robot_m"], dtype=np.float64)
        speeds = np.asarray(trace["step_planar_speed_mps"], dtype=np.float64)
        commands = np.asarray(trace["step_mpc_command"], dtype=np.float64)
        policy_seconds = np.asarray(trace["plan_policy_seconds"], dtype=np.float64)
        mpc_seconds = np.asarray(trace["plan_mpc_seconds"], dtype=np.float64)
        trajectories = np.asarray(trace["plan_local_trajectory"], dtype=np.float64)
        trajectory_lengths = np.asarray(
            trace["plan_local_trajectory_length"], dtype=np.int32
        )
        plan_goals = np.asarray(trace["plan_point_goal_robot_m"], dtype=np.float64)

    metric_success = float(metric["success"])
    metric_distance = float(metric["distance"])
    metric_spl = float(metric["spl"])
    expected_spl = success_weighted_path_length(success, initial, path_length)
    for name, actual, expected in (
        ("success", success, metric_success),
        ("distance", initial, metric_distance),
        ("spl", expected_spl, metric_spl),
    ):
        if not math.isclose(actual, expected, rel_tol=1e-5, abs_tol=1e-5):
            raise ValueError(f"trace/metric {name} mismatch: {trace_path}")

    goal_distances = (
        np.linalg.norm(step_goals[:, :2], axis=-1)
        if step_goals.ndim == 2 and step_goals.shape[0]
        else np.empty((0,), dtype=np.float64)
    )
    goal_distances = np.concatenate((goal_distances, np.asarray([final_goal])))
    goal_changes = np.diff(goal_distances)
    backtrack = float(np.maximum(goal_changes, 0.0).sum())
    total_variation = float(np.abs(goal_changes).sum())
    net_progress = initial - final_goal
    displacement = (
        float(np.linalg.norm(final_position[:2] - step_positions[0, :2]))
        if step_positions.ndim == 2 and step_positions.shape[0]
        else math.nan
    )
    bearings = (
        np.abs(np.arctan2(step_goals[:, 1], step_goals[:, 0]))
        if step_goals.ndim == 2 and step_goals.shape[0]
        else np.empty((0,), dtype=np.float64)
    )
    linear_commands = (
        commands[:, 0] if commands.ndim == 2 and commands.shape[1] >= 2
        else np.empty((0,), dtype=np.float64)
    )
    angular_commands = (
        commands[:, 1] if commands.ndim == 2 and commands.shape[1] >= 2
        else np.empty((0,), dtype=np.float64)
    )
    nonzero_signs = np.sign(angular_commands)
    nonzero_signs = nonzero_signs[nonzero_signs != 0.0]
    sign_changes = int(np.sum(nonzero_signs[1:] * nonzero_signs[:-1] < 0.0))
    total_plan_seconds = policy_seconds + mpc_seconds
    (
        arc_lengths, endpoint_gaps, endpoint_angles,
        peak_curvatures, curvature_energies,
    ) = _plan_geometry(trajectories, trajectory_lengths, plan_goals)

    return {
        "run": run,
        "model": model,
        "split": split,
        "scene": scene,
        "episode_idx": episode_idx,
        "success": success,
        "spl": metric_spl,
        "initial_goal_distance_m": initial,
        "final_goal_distance_m": final_goal,
        "executed_path_length_m": path_length,
        "elapsed_simulation_time_s": elapsed,
        "net_goal_progress_m": net_progress,
        "goal_progress_fraction": _safe_ratio(net_progress, initial),
        "progress_efficiency": _safe_ratio(net_progress, path_length),
        "path_length_efficiency": initial / max(initial, path_length),
        "detour_overhead_m": max(path_length + final_goal - initial, 0.0),
        "goal_distance_backtrack_m": backtrack,
        "goal_distance_total_variation_m": total_variation,
        "progress_monotonicity": _safe_ratio(net_progress, total_variation),
        "net_displacement_m": displacement,
        "path_tortuosity": _safe_ratio(path_length, displacement),
        "mean_speed_mps": _mean(speeds),
        "rms_speed_mps": _rms(speeds),
        "p95_speed_mps": _percentile(speeds, 95.0),
        "mean_abs_goal_bearing_rad": _mean(bearings),
        "p95_abs_goal_bearing_rad": _percentile(bearings, 95.0),
        "mpc_linear_command_rms": _rms(linear_commands),
        "mpc_angular_command_rms": _rms(angular_commands),
        "mpc_angular_sign_changes": sign_changes,
        "plan_count": int(policy_seconds.size),
        "mean_policy_ms": 1000.0 * _mean(policy_seconds),
        "p95_policy_ms": 1000.0 * _percentile(policy_seconds, 95.0),
        "mean_mpc_ms": 1000.0 * _mean(mpc_seconds),
        "p95_mpc_ms": 1000.0 * _percentile(mpc_seconds, 95.0),
        "mean_total_plan_ms": 1000.0 * _mean(total_plan_seconds),
        "p95_total_plan_ms": 1000.0 * _percentile(total_plan_seconds, 95.0),
        "mean_plan_arc_length_m": _mean(arc_lengths),
        "mean_plan_endpoint_goal_gap_m": _mean(endpoint_gaps),
        "mean_plan_endpoint_angular_error_rad": _mean(endpoint_angles),
        "p95_plan_peak_curvature_inv_m": _percentile(peak_curvatures, 95.0),
        "mean_plan_curvature_energy_inv_m": _mean(curvature_energies),
    }


def analyze_roots(roots: list[Path]) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    episodes: list[dict[str, object]] = []
    run_models: dict[str, str] = {}
    for root in roots:
        root = root.resolve()
        metadata_path = root / "run.json"
        if not metadata_path.is_file():
            raise ValueError(f"run has no metadata: {root}")
        metadata = json.loads(metadata_path.read_text())
        model = str(metadata["model"])
        run = root.name
        if run in run_models:
            raise ValueError(f"duplicate run label: {run}")
        run_models[run] = model
        metric_paths = sorted(root.glob("scenes/*/*/metric.csv"))
        if not metric_paths:
            raise ValueError(f"run has no scene metrics: {root}")
        for metric_path in metric_paths:
            relative = metric_path.relative_to(root)
            split, scene = relative.parts[1:3]
            trace_root = metric_path.parent / "trajectory_traces"
            for metric in _load_metric_rows(metric_path):
                episode_idx = int(metric["episode_idx"])
                trace_path = trace_root / f"episode-{episode_idx:03d}.npz"
                if not trace_path.is_file():
                    raise ValueError(f"missing trajectory trace: {trace_path}")
                episodes.append(_episode_row(
                    run, model, split, scene, metric, trace_path,
                ))

    episodes.sort(key=lambda row: (
        str(row["run"]), str(row["split"]), str(row["scene"]),
        int(row["episode_idx"]),
    ))
    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in episodes:
        grouped[str(row["run"])].append(row)
    summaries: list[dict[str, object]] = []
    for run in sorted(grouped):
        rows = grouped[run]
        successful = [row for row in rows if float(row["success"]) == 1.0]
        summaries.append({
            "run": run,
            "model": run_models[run],
            "episodes": len(rows),
            "success_rate": _mean(float(row["success"]) for row in rows),
            "mean_spl": _mean(float(row["spl"]) for row in rows),
            "mean_elapsed_simulation_time_s": _mean(
                float(row["elapsed_simulation_time_s"]) for row in rows
            ),
            "mean_success_time_s": _mean(
                float(row["elapsed_simulation_time_s"]) for row in successful
            ),
            "median_final_goal_distance_m": _percentile(
                (float(row["final_goal_distance_m"]) for row in rows), 50.0
            ),
            "p95_final_goal_distance_m": _percentile(
                (float(row["final_goal_distance_m"]) for row in rows), 95.0
            ),
            "mean_goal_progress_fraction": _mean(
                float(row["goal_progress_fraction"]) for row in rows
            ),
            "mean_progress_efficiency": _mean(
                float(row["progress_efficiency"]) for row in rows
            ),
            "mean_path_length_efficiency": _mean(
                float(row["path_length_efficiency"]) for row in rows
            ),
            "mean_detour_overhead_m": _mean(
                float(row["detour_overhead_m"]) for row in rows
            ),
            "mean_goal_distance_backtrack_m": _mean(
                float(row["goal_distance_backtrack_m"]) for row in rows
            ),
            "mean_progress_monotonicity": _mean(
                float(row["progress_monotonicity"]) for row in rows
            ),
            "mean_path_tortuosity": _mean(
                float(row["path_tortuosity"]) for row in rows
            ),
            "mean_speed_mps": _mean(float(row["mean_speed_mps"]) for row in rows),
            "mean_plan_count": _mean(float(row["plan_count"]) for row in rows),
            "median_episode_mean_total_plan_ms": _percentile(
                (float(row["mean_total_plan_ms"]) for row in rows), 50.0
            ),
            "p95_episode_mean_total_plan_ms": _percentile(
                (float(row["mean_total_plan_ms"]) for row in rows), 95.0
            ),
            "mean_plan_arc_length_m": _mean(
                float(row["mean_plan_arc_length_m"]) for row in rows
            ),
            "mean_plan_endpoint_goal_gap_m": _mean(
                float(row["mean_plan_endpoint_goal_gap_m"]) for row in rows
            ),
            "p95_plan_peak_curvature_inv_m": _percentile(
                (float(row["p95_plan_peak_curvature_inv_m"]) for row in rows), 95.0
            ),
            "mean_plan_curvature_energy_inv_m": _mean(
                float(row["mean_plan_curvature_energy_inv_m"]) for row in rows
            ),
        })
    return episodes, summaries


def _write_csv(path: Path, fields: tuple[str, ...], rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".incoming.{os.getpid()}")
    with temporary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def write_analysis(roots: list[Path], output_dir: Path) -> tuple[Path, Path]:
    episodes, summaries = analyze_roots(roots)
    episode_path = output_dir / "trajectory_episodes.csv"
    summary_path = output_dir / "trajectory_summary.csv"
    _write_csv(episode_path, EPISODE_FIELDS, episodes)
    _write_csv(summary_path, SUMMARY_FIELDS, summaries)
    return episode_path, summary_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, action="append", type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    episode_path, summary_path = write_analysis(args.root, args.output_dir)
    print(f"[trajectory] {episode_path}")
    print(f"[trajectory] {summary_path}")


if __name__ == "__main__":
    main()
