"""Audit the CurveNav v2 pilot without importing model or training code."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from curvenav.data_generation.hssd_v2 import (
    DESCRIPTOR_VERSION,
    HISTORY_OFFSETS,
    MIN_TARGET_GAP,
    MAX_TARGET_GAP,
    SCHEMA_VERSION,
    _local_xy,
    _target_gap,
)
from curvenav.data_generation.occupancy import NavigationGrid, resample_polyline


CLEARANCE_SAMPLE_STEP_M = 0.025
MINIMUM_EXTRA_CLEARANCE_M = 0.10
ROBOT_RADIUS_M = 0.25


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _distribution(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0, "min": None, "p01": None, "p05": None, "p50": None, "p95": None, "p99": None, "max": None, "mean": None}
    array = np.asarray(values, dtype=np.float64)
    percentiles = np.percentile(array, [1, 5, 50, 95, 99])
    return {
        "count": len(array),
        "min": float(array.min()),
        "p01": float(percentiles[0]),
        "p05": float(percentiles[1]),
        "p50": float(percentiles[2]),
        "p95": float(percentiles[3]),
        "p99": float(percentiles[4]),
        "max": float(array.max()),
        "mean": float(array.mean()),
    }


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _grid(path: Path) -> NavigationGrid:
    with np.load(path) as values:
        return NavigationGrid(
            free=values["free"],
            clearance_m=values["clearance_m"],
            origin_xy=values["origin_xy"],
            cell_size_m=float(values["cell_size_m"]),
        )


def _maximum_curvature(path: np.ndarray) -> float:
    points = np.asarray(path, dtype=np.float64)
    if len(points) < 3:
        return 0.0
    a = np.linalg.norm(points[1:-1] - points[:-2], axis=1)
    b = np.linalg.norm(points[2:] - points[1:-1], axis=1)
    c = np.linalg.norm(points[2:] - points[:-2], axis=1)
    cross = np.abs(
        (points[1:-1, 0] - points[:-2, 0]) * (points[2:, 1] - points[:-2, 1])
        - (points[1:-1, 1] - points[:-2, 1]) * (points[2:, 0] - points[:-2, 0])
    )
    denominator = a * b * c
    curvature = np.divide(
        2.0 * cross,
        denominator,
        out=np.zeros_like(cross),
        where=denominator > 1e-9,
    )
    return float(curvature.max(initial=0.0))


def _path_hash(world_xy: np.ndarray) -> str:
    normalized = np.round(np.asarray(world_xy, dtype=np.float64), 4).astype("<f8")
    return hashlib.sha256(normalized.tobytes()).hexdigest()


def _angle_degrees(first: np.ndarray, second: np.ndarray) -> float | None:
    first_norm = float(np.linalg.norm(first))
    second_norm = float(np.linalg.norm(second))
    if first_norm < 1e-8 or second_norm < 1e-8:
        return None
    cosine = float(np.dot(first, second) / (first_norm * second_norm))
    return math.degrees(math.acos(np.clip(cosine, -1.0, 1.0)))


def _episode_path(root: Path, episode: dict[str, Any]) -> Path:
    return root / episode["split"] / f"dataset_hssd_{episode['scene_id']}" / episode["run_id"]


def _audit_episode(root: Path, episode: dict[str, Any], camera_contract: dict[str, Any]) -> dict[str, Any]:
    episode_dir = _episode_path(root, episode)
    metadata = _read_json(episode_dir / "metadata.json")
    with np.load(episode_dir / "route.npz") as route_file:
        world_xy = route_file["world_xy"].astype(np.float64)
        yaw = route_file["yaw_rad"].astype(np.float64)
        cumulative = route_file["cumulative_arc_length_m"].astype(np.float64)
        task_goal = route_file["task_goal_world_xy"].astype(np.float64)
    with np.load(episode_dir / "poses.npz") as poses_file:
        timestamps = poses_file["nominal_timestamp_s"].astype(np.float64)
    grid = _grid(episode_dir.parent / "navigation_grid.npz")
    dense = resample_polyline(world_xy, CLEARANCE_SAMPLE_STEP_M)
    clearance = grid.sample_clearance(np.concatenate([world_xy, dense], axis=0))
    depth_path = episode_dir / "depth_m.npy"
    depth = np.load(depth_path, mmap_mode="r")
    invalid = ~np.isfinite(depth) | (depth <= 0.0)
    violations = []
    if episode["schema_version"] != SCHEMA_VERSION:
        violations.append("schema_version")
    if metadata["camera"] != camera_contract:
        violations.append("camera_contract")
    expected_shape = (
        len(world_xy),
        camera_contract["image"]["height"],
        camera_contract["image"]["width"],
    )
    if depth.dtype != np.float32 or depth.shape != expected_shape:
        violations.append("depth_dtype_or_shape")
    if metadata["depth_sha256"] != _sha256_file(depth_path):
        violations.append("depth_hash")
    invalid_fraction = float(invalid.mean())
    if abs(invalid_fraction - metadata["depth_statistics"]["invalid_fraction"]) > 1e-12:
        violations.append("depth_invalid_fraction")
    if not np.allclose(task_goal, world_xy[-1], atol=1e-7):
        violations.append("task_goal_not_final_episode_goal")
    minimum_extra_clearance = float(clearance.min())
    if not np.isfinite(clearance).all() or minimum_extra_clearance + 1e-8 < MINIMUM_EXTRA_CLEARANCE_M:
        violations.append("continuous_footprint_clearance")
    segment = np.linalg.norm(np.diff(world_xy, axis=0), axis=1)
    straight = float(np.linalg.norm(world_xy[-1] - world_xy[0]))
    return {
        "episode_id": episode["episode_id"],
        "split": episode["split"],
        "scene_id": episode["scene_id"],
        "source_family": episode["source_family"],
        "difficulty": episode["difficulty"],
        "frames": len(world_xy),
        "path_arc_length_m": float(cumulative[-1]),
        "endpoint_distance_m": straight,
        "inefficiency_ratio": float(cumulative[-1] / max(straight, 1e-9)),
        "source_reference_length_ratio": float(episode["source_geometry"]["metrics"]["reference_length_ratio"]),
        "maximum_curvature_per_m": _maximum_curvature(world_xy),
        "minimum_extra_center_clearance_m": minimum_extra_clearance,
        "minimum_total_footprint_clearance_m": ROBOT_RADIUS_M + minimum_extra_clearance,
        "raw_pose_spacing_m": float(np.median(segment)),
        "raw_nominal_dt_s": float(np.median(np.diff(timestamps))),
        "depth_invalid_fraction": invalid_fraction,
        "depth_sha256": metadata["depth_sha256"],
        "path_sha256": _path_hash(world_xy),
        "violations": violations,
    }


def _audit_samples(
    root: Path,
    episodes: list[dict[str, Any]],
    samples: list[dict[str, Any]],
    descriptor_seed: int,
) -> tuple[dict[str, Any], list[dict[str, str]]]:
    episode_map = {episode["episode_id"]: episode for episode in episodes}
    route_cache: dict[str, dict[str, np.ndarray]] = {}
    angles: list[float] = []
    target_arcs: list[float] = []
    endpoint_distances: list[float] = []
    remaining_distances: list[float] = []
    selected_history_spacing: list[float] = []
    selected_nominal_dt: list[float] = []
    failures: list[dict[str, str]] = []
    near_goal = 0
    for sample in samples:
        episode_id = sample["episode_id"]
        episode = episode_map.get(episode_id)
        if episode is None:
            failures.append({"sample_id": sample["sample_id"], "reason": "unknown_episode"})
            continue
        if episode_id not in route_cache:
            with np.load(_episode_path(root, episode) / "route.npz") as route_file:
                route_cache[episode_id] = {key: route_file[key] for key in route_file.files}
        route = route_cache[episode_id]
        world_xy = route["world_xy"].astype(np.float64)
        yaw = route["yaw_rad"].astype(np.float64)
        cumulative = route["cumulative_arc_length_m"].astype(np.float64)
        anchor = int(sample["anchor_index"])
        target_end = int(sample["target_end_index"])
        expected_end = min(
            anchor + _target_gap(descriptor_seed, episode_id, anchor), len(world_xy) - 1
        )
        expected_history = [max(0, anchor + offset) for offset in HISTORY_OFFSETS]
        expected_goal = _local_xy(world_xy[-1], world_xy[anchor], yaw[anchor])
        expected_endpoint = _local_xy(world_xy[target_end], world_xy[anchor], yaw[anchor])
        reasons = []
        if sample["descriptor_version"] != DESCRIPTOR_VERSION or sample["descriptor_seed"] != descriptor_seed:
            reasons.append("descriptor")
        if target_end != expected_end or sample["target_start_index"] != anchor:
            reasons.append("target_indices")
        if sample["history_indices"] != expected_history:
            reasons.append("history_indices")
        if not np.allclose(sample["task_goal_local_xy"], expected_goal, atol=1e-6):
            reasons.append("task_goal_local")
        if not np.allclose(sample["target_endpoint_local_xy"], expected_endpoint, atol=1e-6):
            reasons.append("target_endpoint_local")
        expected_arc = float(cumulative[target_end] - cumulative[anchor])
        if abs(sample["target_arc_length_m"] - expected_arc) > 1e-6:
            reasons.append("target_arc")
        if sample["alternatives"] or sample["critic_labels"] is not None:
            reasons.append("fabricated_critic_supervision")
        if reasons:
            failures.append({"sample_id": sample["sample_id"], "reason": ",".join(reasons)})
        angle = _angle_degrees(expected_endpoint, expected_goal)
        if angle is not None:
            angles.append(angle)
        target_arcs.append(expected_arc)
        endpoint_distances.append(float(np.linalg.norm(expected_endpoint)))
        remaining_distances.append(float(np.linalg.norm(expected_goal)))
        near_goal += int(bool(sample["near_goal"]))
        for previous, current in zip(expected_history[:-1], expected_history[1:]):
            distance = float(cumulative[current] - cumulative[previous])
            selected_history_spacing.append(distance)
            selected_nominal_dt.append(distance / 0.5)
    return (
        {
            "samples": len(samples),
            "near_goal_samples": near_goal,
            "local_endpoint_to_task_goal_angle_degrees": _distribution(angles),
            "target_arc_length_m": _distribution(target_arcs),
            "target_endpoint_distance_m": _distribution(endpoint_distances),
            "remaining_task_goal_distance_m": _distribution(remaining_distances),
            "selected_history_spacing_m": _distribution(selected_history_spacing),
            "selected_history_nominal_dt_s": _distribution(selected_nominal_dt),
        },
        failures,
    )


def _duplicates_and_leakage(episode_reports: list[dict[str, Any]], samples: list[dict[str, Any]]) -> dict[str, Any]:
    path_groups: dict[str, list[str]] = defaultdict(list)
    depth_groups: dict[str, list[str]] = defaultdict(list)
    for report in episode_reports:
        path_groups[report["path_sha256"]].append(report["episode_id"])
        depth_groups[report["depth_sha256"]].append(report["episode_id"])
    duplicate_paths = [group for group in path_groups.values() if len(group) > 1]
    duplicate_depth = [group for group in depth_groups.values() if len(group) > 1]
    episode_ids = [report["episode_id"] for report in episode_reports]
    sample_ids = [sample["sample_id"] for sample in samples]
    train = [report for report in episode_reports if report["split"] == "train"]
    validation = [report for report in episode_reports if report["split"] == "validation"]
    scene_overlap = sorted({report["scene_id"] for report in train} & {report["scene_id"] for report in validation})
    family_overlap = sorted({report["source_family"] for report in train} & {report["source_family"] for report in validation})
    return {
        "duplicate_episode_ids": len(episode_ids) - len(set(episode_ids)),
        "duplicate_sample_ids": len(sample_ids) - len(set(sample_ids)),
        "duplicate_path_groups": duplicate_paths,
        "duplicate_depth_groups": duplicate_depth,
        "train_validation_scene_overlap": scene_overlap,
        "train_validation_source_family_overlap": family_overlap,
    }


def audit(root: Path) -> dict[str, Any]:
    root = root.resolve()
    manifest = _read_json(root / "dataset_manifest.json")
    episodes = _read_jsonl(root / "episodes.jsonl")
    samples = _read_jsonl(root / "samples.jsonl")
    generation_failures = _read_jsonl(root / "failures.jsonl")
    audit_dir = root / "audit"
    audit_dir.mkdir(exist_ok=True)
    episode_reports = [
        _audit_episode(root, episode, manifest["camera"]) for episode in episodes
    ]
    sample_report, sample_failures = _audit_samples(
        root, episodes, samples, int(manifest["descriptor_seed"])
    )
    duplicates = _duplicates_and_leakage(episode_reports, samples)
    episode_failures = [
        {"episode_id": report["episode_id"], "reasons": report["violations"]}
        for report in episode_reports
        if report["violations"]
    ]
    split_counts = Counter(episode["split"] for episode in episodes)
    difficulty_counts = Counter(episode["difficulty"] for episode in episodes)
    distributions = {
        "path_arc_length_m": _distribution([report["path_arc_length_m"] for report in episode_reports]),
        "endpoint_distance_m": _distribution([report["endpoint_distance_m"] for report in episode_reports]),
        "inefficiency_ratio": _distribution([report["inefficiency_ratio"] for report in episode_reports]),
        "source_reference_length_ratio": _distribution([report["source_reference_length_ratio"] for report in episode_reports]),
        "maximum_curvature_per_m": _distribution([report["maximum_curvature_per_m"] for report in episode_reports]),
        "minimum_extra_center_clearance_m": _distribution([report["minimum_extra_center_clearance_m"] for report in episode_reports]),
        "minimum_total_footprint_clearance_m": _distribution([report["minimum_total_footprint_clearance_m"] for report in episode_reports]),
        "raw_pose_spacing_m": _distribution([report["raw_pose_spacing_m"] for report in episode_reports]),
        "raw_nominal_dt_s": _distribution([report["raw_nominal_dt_s"] for report in episode_reports]),
        "depth_invalid_fraction": _distribution([report["depth_invalid_fraction"] for report in episode_reports]),
        **sample_report,
    }
    leakage_failed = any(
        [
            duplicates["duplicate_episode_ids"],
            duplicates["duplicate_sample_ids"],
            duplicates["duplicate_path_groups"],
            duplicates["duplicate_depth_groups"],
            duplicates["train_validation_scene_overlap"],
            duplicates["train_validation_source_family_overlap"],
        ]
    )
    summary = {
        "schema_version": manifest["schema_version"],
        "root": str(root),
        "episodes_valid": len(episodes) - len(episode_failures),
        "episodes_invalid": len(episode_failures) + len(generation_failures),
        "generation_failures": len(generation_failures),
        "sample_failures": len(sample_failures),
        "split_counts": dict(split_counts),
        "difficulty_counts": dict(difficulty_counts),
        "near_goal_samples": sample_report["near_goal_samples"],
        "camera_contract": manifest["camera"],
        "continuous_clearance_contract": {
            "interpretation": "extra clearance inside an already robot-footprint-safe center set",
            "robot_radius_m": ROBOT_RADIUS_M,
            "minimum_extra_clearance_m": MINIMUM_EXTRA_CLEARANCE_M,
            "sample_step_m": CLEARANCE_SAMPLE_STEP_M,
        },
        "leakage_or_duplicate_failure": leakage_failed,
        "passed": not episode_failures and not generation_failures and not sample_failures and not leakage_failed,
    }
    failures = {
        "generation": generation_failures,
        "episodes": episode_failures,
        "samples": sample_failures,
    }
    _write_json(audit_dir / "summary.json", summary)
    _write_json(audit_dir / "distributions.json", distributions)
    _write_json(audit_dir / "duplicates_and_leakage.json", duplicates)
    _write_json(audit_dir / "episode_metrics.json", episode_reports)
    _write_json(audit_dir / "failures.json", failures)
    report = f"""# CurveNav v2 pilot audit

- Passed: **{summary['passed']}**
- Valid / invalid episodes: **{summary['episodes_valid']} / {summary['episodes_invalid']}**
- Split: train **{split_counts.get('train', 0)}**, validation **{split_counts.get('validation', 0)}**
- Samples / near-goal samples: **{len(samples)} / {sample_report['near_goal_samples']}**
- Difficulty: `{dict(difficulty_counts)}`
- Minimum continuous extra clearance: **{distributions['minimum_extra_center_clearance_m']['min']:.4f} m**
- Minimum total footprint clearance: **{distributions['minimum_total_footprint_clearance_m']['min']:.4f} m**
- Target arc length p05 / p50 / p95: **{distributions['target_arc_length_m']['p05']:.3f} / {distributions['target_arc_length_m']['p50']:.3f} / {distributions['target_arc_length_m']['p95']:.3f} m**
- Endpoint-to-task-goal angle p50 / p95: **{distributions['local_endpoint_to_task_goal_angle_degrees']['p50']:.2f} / {distributions['local_endpoint_to_task_goal_angle_degrees']['p95']:.2f} deg**
- Train/validation scene overlap: `{duplicates['train_validation_scene_overlap']}`
- Train/validation source-family overlap: `{duplicates['train_validation_source_family_overlap']}`
- Duplicate paths / depth tensors: **{len(duplicates['duplicate_path_groups'])} / {len(duplicates['duplicate_depth_groups'])}**
"""
    (audit_dir / "report.md").write_text(report, encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return summary


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset_root", type=Path)
    args = parser.parse_args(argv)
    summary = audit(args.dataset_root)
    if not summary["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
