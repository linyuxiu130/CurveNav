"""Hard publication gates for the official HSSD policy dataset."""

from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from curvenav.data_generation.geometry import (
    SCHEMA,
    Grid,
    sample_contract,
    sha256_file,
    source_family,
)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _distribution(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "min": float(array.min()),
        "p05": float(np.percentile(array, 5)),
        "p50": float(np.percentile(array, 50)),
        "p95": float(np.percentile(array, 95)),
        "max": float(array.max()),
    }


def _dataset_sha(root: Path, audit_dir: Path) -> tuple[str, int]:
    files = sorted(
        path for path in root.rglob("*") if path.is_file() and audit_dir not in path.parents
    )
    lines = [f"{sha256_file(path)}  {path.relative_to(root)}\n" for path in files]
    index = "".join(lines)
    (audit_dir / "data_files.sha256").write_text(index, encoding="utf-8")
    return hashlib.sha256(index.encode()).hexdigest(), sum(path.stat().st_size for path in files)


def audit_dataset(root: Path) -> dict[str, Any]:
    root = root.resolve()
    manifest = _read_json(root / "dataset_manifest.json")
    records = _read_jsonl(root / "samples.jsonl")
    variants = set(manifest["variants"])
    metrics: list[dict[str, Any]] = []
    metadata_mismatches: list[str] = []
    grid_cache: dict[tuple[str, str], Grid] = {}
    for record in records:
        key = (record["split"], record["scene_id"])
        if key not in grid_cache:
            grid_cache[key] = Grid.load(
                root
                / record["split"]
                / f"dataset_hssd_{record['scene_id']}"
                / "navigation_grid.npz"
            )
        metadata = _read_json(root / record["sample_directory"] / "metadata.json")
        if metadata != record:
            metadata_mismatches.append(record["sample_id"])
        metrics.append(sample_contract(root, record, grid_cache[key], manifest["camera"]))

    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    routes: dict[str, dict[str, str]] = defaultdict(dict)
    for record in records:
        groups[record["anchor_group_id"]].append(record)
        routes[record["source_route_id"]][record["anchor_group_id"]] = record["goal_band"]
    incomplete_groups = [
        group_id
        for group_id, group in groups.items()
        if len(group) != 5 or {item["variant"] for item in group} != variants
    ]
    mismatched_goals = [
        group_id
        for group_id, group in groups.items()
        if len({tuple(np.round(item["task_goal"]["world_xy_m"], 5)) for item in group}) != 1
    ]
    route_quota_failures = [
        route_id
        for route_id, anchors in routes.items()
        if Counter(anchors.values()) != {"near": 1, "middle": 2, "far": 2}
    ]
    route_signatures = []
    duplicate_anchor_arcs = []
    mismatched_route_goals = []
    for route_id, anchors in routes.items():
        split, scene_id, route_name = route_id.split("/")
        with np.load(
            root / split / f"dataset_hssd_{scene_id}" / "source_routes" / route_name / "route.npz"
        ) as route:
            path = route["route_world_xy"]
            arcs = route["anchor_arc_m"]
            route_goal = route["task_goal_world_xy"]
        route_signatures.append((scene_id, *np.round(np.concatenate([path[0], path[-1]]), 4)))
        record_arcs = np.asarray(
            [groups[anchor_id][0]["anchor_arc_m"] for anchor_id in anchors], dtype=np.float64
        )
        if (
            len(arcs) != len(anchors)
            or len(np.unique(np.round(arcs, 4))) != len(arcs)
            or not np.allclose(np.sort(arcs), np.sort(record_arcs), atol=1e-4)
        ):
            duplicate_anchor_arcs.append(route_id)
        route_goals = {
            tuple(np.round(record["task_goal"]["world_xy_m"], 5))
            for anchor_id in anchors
            for record in groups[anchor_id]
        }
        if len(route_goals) != 1 or not np.allclose(
            next(iter(route_goals)), route_goal, atol=2e-5
        ):
            mismatched_route_goals.append(route_id)

    train_scenes = {item["scene_id"] for item in records if item["split"] == "train"}
    validation_scenes = {item["scene_id"] for item in records if item["split"] == "validation"}
    train_families = {source_family(scene_id) for scene_id in train_scenes}
    validation_families = {source_family(scene_id) for scene_id in validation_scenes}
    sample_ids = [item["sample_id"] for item in records]
    depth_hashes = [item["depth_sha256"] for item in metrics]
    group_bands = Counter(group[0]["goal_band"] for group in groups.values())
    sample_bands = Counter(item["goal_band"] for item in records)
    split_counts = Counter(item["split"] for item in records)
    variant_counts = Counter(item["variant"] for item in records)
    category_counts = Counter(item["category"] for item in records)

    subsets = {
        name: _read_jsonl(root / "subsets" / f"{name}.jsonl") for name in ("standard", "perturbed")
    }
    subset_failures = [
        name
        for name, subset in subsets.items()
        if {item["sample_id"] for item in subset}
        != {item["sample_id"] for item in records if item["category"] == name}
    ]
    partial_paths = [
        str(path.relative_to(root)) for path in root.rglob("*") if ".partial" in path.name
    ]
    violations = [item for item in metrics if item["violations"]]
    contract_failures = {
        "schema": manifest["schema"] != SCHEMA,
        "sample_count": len(records) != 1000,
        "split_counts": dict(split_counts) != {"train": 800, "validation": 200},
        "scene_counts": (len(train_scenes), len(validation_scenes)) != (16, 4),
        "route_count": len(routes) != 40,
        "anchor_group_count": len(groups) != 200,
        "variant_counts": dict(variant_counts) != {name: 200 for name in manifest["variants"]},
        "category_counts": dict(category_counts) != {"standard": 200, "perturbed": 800},
        "group_band_counts": dict(group_bands) != {"near": 40, "middle": 80, "far": 80},
        "sample_band_counts": dict(sample_bands) != {"near": 200, "middle": 400, "far": 400},
    }
    failures = {
        "contract": [name for name, failed in contract_failures.items() if failed],
        "sample_violations": [item["sample_id"] for item in violations],
        "metadata_mismatches": metadata_mismatches,
        "duplicate_sample_ids": len(sample_ids) - len(set(sample_ids)),
        "duplicate_depth_tensors": len(depth_hashes) - len(set(depth_hashes)),
        "incomplete_groups": incomplete_groups,
        "mismatched_group_goals": mismatched_goals,
        "route_quota_failures": route_quota_failures,
        "duplicate_source_routes": len(route_signatures) - len(set(route_signatures)),
        "duplicate_anchor_arcs": duplicate_anchor_arcs,
        "mismatched_route_goals": mismatched_route_goals,
        "scene_overlap": sorted(train_scenes & validation_scenes),
        "source_family_overlap": sorted(train_families & validation_families),
        "subset_failures": subset_failures,
        "partial_paths": partial_paths,
    }
    audit_dir = root / "audit"
    audit_dir.mkdir(exist_ok=True)
    dataset_sha, size_bytes = _dataset_sha(root, audit_dir)
    cross_counts = Counter(
        f"{item['variant']}|{item['route_geometry']['turning_sign']}|"
        f"{item['route_geometry']['difficulty']}"
        for item in records
    )
    passed = not any(bool(value) for value in failures.values())
    summary = {
        "passed": passed,
        "samples_valid": len(records) - len(violations),
        "samples_invalid": len(violations),
        "samples_total": len(records),
        "anchor_groups_complete": len(groups) - len(incomplete_groups),
        "anchor_groups_total": len(groups),
        "source_routes": len(routes),
        "split_counts": dict(split_counts),
        "scene_split_counts": {"train": len(train_scenes), "validation": len(validation_scenes)},
        "variant_counts": dict(variant_counts),
        "category_counts": dict(category_counts),
        "goal_band_counts": dict(sample_bands),
        "goal_distance_m": _distribution([item["goal_distance_m"] for item in metrics]),
        "target_arc_m": _distribution([item["target_arc_m"] for item in metrics]),
        "target_curvature_p95": _distribution([item["target_curvature_p95"] for item in metrics]),
        "target_curvature_max": _distribution([item["target_curvature_max"] for item in metrics]),
        "target_clearance_m": _distribution([item["target_clearance_m"] for item in metrics]),
        "history_clearance_m": _distribution([item["history_clearance_m"] for item in metrics]),
        "variant_turn_difficulty_counts": dict(cross_counts),
        "dataset_sha256": dataset_sha,
        "source_size_bytes": size_bytes,
    }
    _write_json(audit_dir / "summary.json", summary)
    _write_json(audit_dir / "failures.json", failures)
    _write_json(audit_dir / "sample_metrics.json", metrics)
    if not passed:
        raise RuntimeError(f"dataset audit failed: {failures}")
    return summary
