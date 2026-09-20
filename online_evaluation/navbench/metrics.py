#!/usr/bin/env python3
"""Validate and aggregate official X-NavDP PointGoal metrics."""

from __future__ import annotations

import argparse
from collections import Counter
import csv
import json
import math
from pathlib import Path
import statistics


def success_weighted_path_length(
    success: float, shortest_path_m: float, actual_path_m: float
) -> float:
    """Compute the SPL formula used by the released X-NavDP evaluator."""
    shortest = float(shortest_path_m)
    actual = float(actual_path_m)
    if not math.isfinite(shortest) or shortest <= 0.0:
        raise ValueError("shortest path must be positive and finite")
    if not math.isfinite(actual) or actual < 0.0:
        raise ValueError("actual path must be non-negative and finite")
    return float(success) * shortest / max(shortest, actual)


def _metric_identity(root: Path, path: Path) -> tuple[str, str]:
    relative = path.relative_to(root)
    parts = relative.parts
    if len(parts) != 4 or parts[0] != "scenes" or parts[-1] != "metric.csv":
        raise SystemExit(f"Unexpected metric path: {relative}")
    return parts[1], parts[2]


def _mean(rows: list[dict[str, str]], field: str) -> float:
    return statistics.fmean(float(row[field]) for row in rows)


def _validate_distributed_roots(roots: list[Path]) -> int | None:
    if len(roots) == 1:
        return None
    metadata = []
    for root in roots:
        path = root / "run.json"
        if not path.is_file():
            raise SystemExit(f"Distributed shard has no run.json: {root}")
        metadata.append(json.loads(path.read_text()))

    identity_fields = (
        "suite", "suite_definition_sha256", "model", "precision", "num_envs",
        "runtime_revision", "runtime_patch_sha256", "checkpoint_sha256", "model_config_sha256", "seed",
        "episodes_per_scene", "full_suite_scene_count", "shard_count",
        "shard_weights",
    )
    reference = metadata[0]
    for index, item in enumerate(metadata[1:], 1):
        mismatches = [
            field for field in identity_fields
            if item.get(field) != reference.get(field)
        ]
        if mismatches:
            raise SystemExit(
                f"Distributed shard identity mismatch at root {index}: "
                + ", ".join(mismatches)
            )

    shard_count = int(reference["shard_count"])
    shard_indices = {int(item["shard_index"]) for item in metadata}
    if len(roots) != shard_count or shard_indices != set(range(shard_count)):
        raise SystemExit("Distributed merge requires every shard exactly once")

    declared_scenes: set[str] = set()
    for root, item in zip(roots, metadata):
        scene_keys = set(item["scene_keys"])
        overlap = declared_scenes & scene_keys
        if overlap:
            raise SystemExit("Distributed run.json files contain overlapping scenes")
        declared_scenes.update(scene_keys)
        actual_scenes = {
            "/".join(_metric_identity(root, path))
            for path in root.glob("scenes/*/*/metric.csv")
        }
        if actual_scenes != scene_keys:
            raise SystemExit(f"Distributed shard is incomplete: {root}")
    if len(declared_scenes) != int(reference["full_suite_scene_count"]):
        raise SystemExit("Distributed shards do not cover the frozen suite")
    return sum(int(item["episode_count"]) for item in metadata)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root", required=True, action="append", type=Path,
        help="run root; repeat for deterministic scene shards",
    )
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--summary", required=True, type=Path)
    parser.add_argument("--expected", type=int, default=0)
    args = parser.parse_args()

    roots = [root.resolve() for root in args.root]
    distributed_expected = _validate_distributed_roots(roots)
    if args.expected and distributed_expected is not None:
        if args.expected != distributed_expected:
            raise SystemExit(
                f"Expected {args.expected} rows, distributed run declares "
                f"{distributed_expected}"
            )

    inputs = [
        (root, path)
        for root in roots
        for path in sorted(root.glob("scenes/*/*/metric.csv"))
    ]
    if not inputs:
        roots = ", ".join(str(root) for root in args.root)
        raise SystemExit(f"No scene metrics found under: {roots}")

    rows: list[dict[str, str]] = []
    metric_fields: list[str] | None = None
    identities: set[tuple[str, str, int]] = set()
    for root, path in inputs:
        split, scene = _metric_identity(root, path)
        with path.open(newline="") as handle:
            reader = csv.DictReader(handle)
            required = {"success", "spl", "distance", "episode_idx"}
            if not reader.fieldnames or not required.issubset(reader.fieldnames):
                raise SystemExit(f"Missing official episode metrics in {path}")
            if metric_fields is None:
                metric_fields = reader.fieldnames
            elif reader.fieldnames != metric_fields:
                raise SystemExit(f"Metric schema mismatch in {path}")
            scene_rows = list(reader)
        ids = [int(row["episode_idx"]) for row in scene_rows]
        duplicates = [value for value, count in Counter(ids).items() if count > 1]
        if duplicates:
            raise SystemExit(f"Duplicate episode IDs in {scene}: {sorted(duplicates)}")
        if ids and set(ids) != set(range(len(ids))):
            raise SystemExit(f"Non-contiguous official episode prefix in {scene}")
        scene_identities = {(split, scene, value) for value in ids}
        overlap = identities & scene_identities
        if overlap:
            raise SystemExit(
                "Duplicate scene/episode identities across metric roots: "
                + ", ".join(
                    f"{domain}/{name}/{episode}"
                    for domain, name, episode in sorted(overlap)
                )
            )
        identities.update(scene_identities)
        rows.extend({"split": split, "scene": scene, **row} for row in scene_rows)

    if args.expected and len(rows) != args.expected:
        raise SystemExit(f"Expected {args.expected} rows, found {len(rows)}")
    rows.sort(key=lambda row: (row["split"], row["scene"], int(row["episode_idx"])))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    output_fields = ["split", "scene", *(metric_fields or [])]
    with args.output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=output_fields)
        writer.writeheader()
        writer.writerows(rows)

    groups = [
        (split, [row for row in rows if row["split"] == split])
        for split in sorted({row["split"] for row in rows})
    ]

    summary_rows: list[dict[str, object]] = []
    for domain, group in groups:
        row: dict[str, object] = {
            "domain": domain,
            "scenes": len({item["scene"] for item in group}),
            "episodes": len(group),
            "success_rate": _mean(group, "success"),
            "mean_spl": _mean(group, "spl"),
        }
        summary_rows.append(row)

    fieldnames: list[str] = []
    for row in summary_rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with args.summary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)
    print(f"Merged {len(rows)} episodes from {len(inputs)} scenes into {args.output}")


if __name__ == "__main__":
    main()
