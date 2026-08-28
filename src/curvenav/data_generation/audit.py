"""Hard gates for the 500-route HSSD expert dataset."""

from __future__ import annotations

from collections import Counter
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from curvenav.data.contracts import expert_navigation_geometry_contract
from curvenav.data_generation.geometry import (
    Grid,
    path_length,
    resample,
    sha256_file,
    source_family,
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
    records = _read_jsonl(root / "routes.jsonl")
    camera = manifest["camera"]
    expected_image_shape = (camera["image"]["height"], camera["image"]["width"])
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
        depth_path = directory / "depth_m.npy"
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
        if depth.dtype != np.float32 or depth.shape != (frames, *expected_image_shape):
            reasons.append("depth_shape_or_dtype")
        depth_sha = sha256_file(depth_path)
        invalid_fraction = float((~np.isfinite(depth) | (depth <= 0)).mean())
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
                root
                / record["split"]
                / f"dataset_hssd_{record['scene_id']}"
                / "navigation_grid.npz"
            )
        grid = grids[key]
        if not grid.safe(xy):
            reasons.append("route_clearance")
        spacing = np.linalg.norm(np.diff(xy, axis=0), axis=1)
        if (
            np.any(spacing <= 0.02)
            or np.any(spacing > 0.20)
            or not 0.14 <= float(np.median(spacing)) <= 0.151
        ):
            reasons.append("route_spacing")
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
        if record["source_family"] != source_family(record["scene_id"]):
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
    failures = {
        "schema": manifest.get("schema") != SCHEMA,
        "navigation_geometry": manifest.get("route_contract", {}).get(
            "navigation_geometry"
        )
        != expert_navigation_geometry_contract(),
        "route_count": len(records) != 500 or manifest.get("routes") != 500,
        "split_counts": dict(split_counts) != {"train": 400, "validation": 100},
        "scene_route_counts": any(count != 25 for count in scene_counts.values())
        or len(scene_counts) != 20,
        "distance_band_counts": dict(band_counts)
        != {"near": 100, "middle": 200, "far": 200},
        "duplicate_route_ids": len(records)
        - len({item["route_id"] for item in records}),
        "duplicate_routes": len(signatures) - len(set(signatures)),
        "metadata_mismatches": metadata_mismatches,
        "route_violations": route_violations,
        "scene_overlap": sorted(train_scenes & validation_scenes),
        "source_family_overlap": sorted(
            {source_family(item) for item in train_scenes}
            & {source_family(item) for item in validation_scenes}
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
