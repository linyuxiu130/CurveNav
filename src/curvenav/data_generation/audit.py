"""Audit configured expert routes and their training-format depth."""

from __future__ import annotations

from collections import Counter
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from curvenav.data.history import validate_transform
from curvenav.data.contracts import expert_navigation_geometry_contract
from curvenav.data_generation.geometry import (
    Grid,
    path_length,
    resample,
    sha256_file,
)
from curvenav.data_generation.generate import SCHEMA


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _distribution(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "min": float(array.min()),
        "p05": float(np.percentile(array, 5)),
        "p50": float(np.percentile(array, 50)),
        "mean": float(array.mean()),
        "p95": float(np.percentile(array, 95)),
        "max": float(array.max()),
    }


def validate_route_pose(xy: np.ndarray, yaw: np.ndarray, poses: np.ndarray) -> None:
    """Check that label XZ coordinates and depth SE(3) describe the same body."""
    if not np.isfinite(xy).all() or not np.isfinite(yaw).all():
        raise ValueError("route positions and headings must be finite")
    validate_transform(poses)
    expected_rotation = np.zeros((len(yaw), 3, 3))
    c, s = np.cos(yaw), np.sin(yaw)
    expected_rotation[:, 0, 0] = expected_rotation[:, 1, 1] = c
    expected_rotation[:, 0, 1], expected_rotation[:, 1, 0] = s, -s
    expected_rotation[:, 2, 2] = 1
    if (
        not np.allclose(poses[:, :2, 3], xy * [1, -1], atol=1e-6, rtol=0)
        or not np.allclose(poses[:, :3, :3], expected_rotation, atol=1e-6, rtol=0)
    ):
        raise ValueError("depth poses and expert XY/yaw disagree")


def validate_route_spacing(xy: np.ndarray, maximum_step: float) -> None:
    """Check speed allowing only the rounding bounds of stored coordinates."""
    spacing = np.linalg.norm(np.diff(xy.astype(np.float64), axis=0), axis=1)
    # Each coordinate was rounded once on storage. Subtraction combines the
    # two endpoint errors; the norm is 1-Lipschitz in that displacement.
    half_ulp = np.spacing(np.abs(xy)).astype(np.float64) / 2
    rounding_bound = np.linalg.norm(half_ulp[:-1] + half_ulp[1:], axis=1)
    if np.any(spacing > maximum_step + rounding_bound):
        raise ValueError("route spacing exceeds the forward speed contract")


def _dataset_sha(root: Path, audit_dir: Path) -> tuple[str, int]:
    files = sorted(
        path
        for path in root.rglob("*")
        if path.is_file() and audit_dir not in path.parents
    )
    lines = [f"{sha256_file(path)}  {path.relative_to(root)}\n" for path in files]
    index = "".join(lines)
    (audit_dir / "data_files.sha256").write_text(index, encoding="utf-8")
    return hashlib.sha256(index.encode()).hexdigest(), sum(
        path.stat().st_size for path in files
    )


def audit_dataset(root: Path) -> dict[str, Any]:
    root = root.resolve()
    manifest = _read_json(root / "dataset_manifest.json")
    if manifest["schema"] != SCHEMA:
        raise ValueError("expected forward policy-format depth route schema v5")
    records = _read_jsonl(root / "routes.jsonl")
    families = {s["scene_id"]: s["source_family"] for s in manifest["scenes"]}
    observation = manifest["observation"]
    expected_image_shape = (observation["image_height"], observation["image_width"])
    grids: dict[tuple[str, str], Grid] = {}
    metrics = []
    metadata_mismatches = []
    route_violations: dict[str, list[str]] = {}
    signatures = []
    for record in records:
        route_id = record["route_id"]
        directory = root / record["route_directory"]
        metadata = _read_json(directory / "metadata.json")
        if metadata != record:
            metadata_mismatches.append(route_id)
        xy = np.load(directory / "traj_xy.npy")
        yaw = np.load(directory / "traj_yaw.npy")
        depth_path = directory / "depth.npy"
        depth = np.load(depth_path, mmap_mode="r")
        reasons = []
        frames = len(xy)
        if (
            xy.dtype != np.float32
            or yaw.dtype != np.float32
            or xy.shape != (frames, 2)
            or yaw.shape != (frames,)
            or frames < 3
        ):
            reasons.append("trajectory_shape_or_dtype")
        if depth.dtype != np.float16 or depth.shape != (frames, *expected_image_shape):
            reasons.append("depth_shape_or_dtype")
        poses = np.load(directory / "body_to_world.npy")
        times = np.load(directory / "timestamps.npy")
        validate_route_pose(xy, yaw, poses)
        if (
            poses.shape != (frames, 4, 4)
            or times.shape != (frames,)
            or not np.isfinite(times).all()
            or np.any(np.diff(times) <= 0)
        ):
            reasons.append("pose_or_time_contract")
        depth_sha = sha256_file(depth_path)
        invalid_fraction = float((depth == 0).mean())
        if not np.isfinite(depth).all() or np.any(depth < 0) or np.any(depth > 1):
            reasons.append("normalized_depth_range")
        if (
            depth_sha != record["depth"]["sha256"]
            or not math.isclose(
                invalid_fraction, record["depth"]["invalid_fraction"], abs_tol=1e-12
            )
            or invalid_fraction >= 1.0
        ):
            reasons.append("depth_content")
        key = (record["split"], record["scene_id"])
        if key not in grids:
            grids[key] = Grid.load(
                directory.parent / "navigation_grid.npz"
            )
        grid = grids[key]
        if not grid.safe(xy):
            reasons.append("route_clearance")
        period = manifest["route_contract"]["observation_period_s"]
        maximum_step = (
            period * manifest["route_contract"]["expert_speed_m_s"]
        )
        try:
            validate_route_spacing(xy, maximum_step)
        except ValueError:
            reasons.append("route_spacing")
        controls = np.load(directory / "expert_controls.npy")
        angular_limit = manifest["route_contract"]["expert_angular_speed_rad_s"]
        if (controls.shape != (frames, 2) or not np.isfinite(controls).all()
                or np.any(controls[:, 0] < 0)
                or np.any(controls[:, 0] > manifest["route_contract"]["expert_speed_m_s"] + 1e-6)
                or np.any(abs(controls[:, 1]) > angular_limit + 1e-6)):
            reasons.append("forward_control_domain")
        if abs(math.remainder(float(yaw[0]) - record["initial_yaw_rad"], 2 * math.pi)) > 1e-6:
            reasons.append("initial_heading")
        yaw_change = np.arctan2(np.sin(np.diff(yaw)), np.cos(np.diff(yaw)))
        if np.any(abs(yaw_change) > period * angular_limit + 1e-6):
            reasons.append("angular_clock")
        if not np.allclose(np.diff(times), period, atol=1e-6, rtol=0):
            reasons.append("sensor_clock")
        route_arc = path_length(xy)
        endpoint_distance = float(np.linalg.norm(xy[-1] - xy[0]))
        lower, upper = record["endpoint_distance_range_m"]
        if (
            not math.isclose(route_arc, record["route_arc_m"], abs_tol=2e-5)
            or not math.isclose(
                endpoint_distance, record["endpoint_distance_m"], abs_tol=2e-5
            )
            or not lower <= endpoint_distance < upper
        ):
            reasons.append("route_metadata")
        if record["source_family"] != families[record["scene_id"]]:
            reasons.append("source_family")
        signatures.append(
            (record["scene_id"], *np.round(np.concatenate((xy[0], xy[-1])), 4))
        )
        metrics.append(
            {
                "route_id": route_id,
                "split": record["split"],
                "scene_id": record["scene_id"],
                "frames": frames,
                "route_arc_m": route_arc,
                "endpoint_distance_m": endpoint_distance,
                "minimum_clearance_m": float(grid.clearance(resample(xy, 0.025)).min()),
                "depth_invalid_fraction": invalid_fraction,
            }
        )
        if reasons:
            route_violations[route_id] = reasons

    split_counts = Counter(item["split"] for item in records)
    scene_counts = Counter((item["split"], item["scene_id"]) for item in records)
    band_counts = Counter(item["endpoint_distance_band"] for item in records)
    train_scenes = {item["scene_id"] for item in records if item["split"] == "train"}
    validation_scenes = {
        item["scene_id"] for item in records if item["split"] == "validation"
    }
    quotas = manifest["route_contract"]["routes_per_scene_by_distance"]
    expected_scenes = {(s["split"], s["scene_id"]): sum(quotas[s["scene_id"]].values())
                       for s in manifest["scenes"]}
    expected_splits, expected_bands = Counter(), Counter()
    for (split, scene_id), count in expected_scenes.items():
        expected_splits[split] += count
        expected_bands.update(quotas[scene_id])
    expected_routes = sum(expected_scenes.values())
    scene_band_counts = Counter((r["scene_id"], r["endpoint_distance_band"]) for r in records)
    expected_scene_bands = {(scene_id, band): count for scene_id, bands in quotas.items()
                            for band, count in bands.items()}
    failures = {
        "schema": manifest.get("schema") != SCHEMA,
        "navigation_geometry": manifest.get("route_contract", {}).get(
            "navigation_geometry"
        )
        != expert_navigation_geometry_contract(),
        "route_count": len(records) != expected_routes or manifest.get("routes") != expected_routes,
        "split_counts": dict(split_counts) != dict(expected_splits),
        "scene_route_counts": dict(scene_counts) != expected_scenes,
        "distance_band_counts": dict(band_counts) != dict(expected_bands) or dict(scene_band_counts) != expected_scene_bands,
        "duplicate_route_ids": len(records)
        - len({item["route_id"] for item in records}),
        "duplicate_routes": len(signatures) - len(set(signatures)),
        "metadata_mismatches": metadata_mismatches,
        "route_violations": route_violations,
        "scene_overlap": sorted(train_scenes & validation_scenes),
        "source_family_overlap": sorted(
            {families[item] for item in train_scenes}
            & {families[item] for item in validation_scenes}
        ),
        "partial_paths": [
            str(path.relative_to(root))
            for path in root.rglob("*")
            if ".partial" in path.name
        ],
    }
    audit_dir = root / "audit"
    audit_dir.mkdir(exist_ok=True)
    dataset_sha, size_bytes = _dataset_sha(root, audit_dir)
    summary = {
        "passed": not any(bool(value) for value in failures.values()),
        "routes": len(records),
        "frames": sum(item["frames"] for item in metrics),
        "local_trajectory_samples": sum(item["frames"] - 1 for item in metrics),
        "split_counts": dict(split_counts),
        "scene_split_counts": {
            "train": len(train_scenes),
            "validation": len(validation_scenes),
        },
        "distance_band_counts": dict(band_counts),
        "frames_per_route": _distribution([item["frames"] for item in metrics]),
        "route_arc_m": _distribution([item["route_arc_m"] for item in metrics]),
        "endpoint_distance_m": _distribution(
            [item["endpoint_distance_m"] for item in metrics]
        ),
        "minimum_clearance_m": _distribution(
            [item["minimum_clearance_m"] for item in metrics]
        ),
        "depth_invalid_fraction": _distribution(
            [item["depth_invalid_fraction"] for item in metrics]
        ),
        "dataset_sha256": dataset_sha,
        "source_size_bytes": size_bytes,
    }
    _write_json(audit_dir / "summary.json", summary)
    _write_json(audit_dir / "failures.json", failures)
    _write_json(audit_dir / "route_metrics.json", metrics)
    if not summary["passed"]:
        raise RuntimeError(f"dataset audit failed: {failures}")
    return summary
