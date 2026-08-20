#!/usr/bin/env python3
"""Select a feature-diverse HSSD train/validation scene batch."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import yaml


def _polygon_area(points: list[list[float]]) -> float:
    polygon = np.asarray(points, dtype=np.float64)[:, [0, 2]]
    return float(
        abs(
            np.dot(polygon[:, 0], np.roll(polygon[:, 1], -1))
            - np.dot(polygon[:, 1], np.roll(polygon[:, 0], -1))
        )
        / 2.0
    )


def _scene_record(
    asset_root: Path,
    scene_id: str,
    official_split: str,
    available_object_configs: set[str],
) -> dict[str, object]:
    scene = json.loads(
        (asset_root / "scenes" / f"{scene_id}.scene_instance.json").read_text()
    )
    semantics = json.loads(
        (
            asset_root
            / "semantics"
            / "scenes"
            / f"{scene_id}.semantic_config.json"
        ).read_text()
    )
    regions = semantics["region_annotations"]
    labels = {region["label"] for region in regions}
    areas = [_polygon_area(region["poly_loop"]) for region in regions]
    handles = {item["template_name"] for item in scene.get("object_instances", [])}
    missing_handles = sorted(
        handle
        for handle in handles
        if f"{handle}.object_config.json" not in available_object_configs
    )
    return {
        "scene_id": scene_id,
        "official_split": official_split,
        "source_family": scene_id[:6],
        "objects": len(scene.get("object_instances", [])),
        "regions": len(regions),
        "region_labels": len(labels),
        "annotated_floor_area_m2": float(sum(areas)),
        "largest_region_m2": float(max(areas)),
        "missing_object_handles": missing_handles,
    }


def _features(records: list[dict[str, object]]) -> np.ndarray:
    continuous = np.asarray(
        [
            [
                np.log1p(float(record["objects"])),
                np.log1p(float(record["regions"])),
                np.log1p(float(record["region_labels"])),
                np.log1p(float(record["annotated_floor_area_m2"])),
                np.log1p(float(record["largest_region_m2"])),
            ]
            for record in records
        ]
    )
    center = np.median(continuous, axis=0)
    scale = np.percentile(continuous, 75, axis=0) - np.percentile(
        continuous, 25, axis=0
    )
    continuous = (continuous - center) / np.maximum(scale, 1e-6)
    families = sorted({str(record["source_family"]) for record in records})
    one_hot = np.zeros((len(records), len(families)), dtype=np.float64)
    family_index = {family: index for index, family in enumerate(families)}
    for row, record in enumerate(records):
        one_hot[row, family_index[str(record["source_family"])]] = 0.75
    return np.concatenate([continuous, one_hot], axis=1)


def _diversity_order(records: list[dict[str, object]], forced_scene: str | None) -> list[int]:
    features = _features(records)
    if forced_scene is None:
        selected = [int(np.argmin(np.linalg.norm(features, axis=1)))]
    else:
        selected = [
            next(
                index
                for index, record in enumerate(records)
                if record["scene_id"] == forced_scene
            )
        ]
    remaining = set(range(len(records))) - set(selected)
    while remaining:
        candidates = np.asarray(sorted(remaining))
        distances = np.linalg.norm(
            features[candidates, None] - features[np.asarray(selected)][None], axis=2
        )
        next_index = int(candidates[np.argmax(distances.min(axis=1))])
        selected.append(next_index)
        remaining.remove(next_index)
    return selected


def _unique_family_first(
    records: list[dict[str, object]], order: list[int]
) -> list[int]:
    seen: set[str] = set()
    unique = []
    repeated = []
    for index in order:
        family = str(records[index]["source_family"])
        target = repeated if family in seen else unique
        target.append(index)
        seen.add(family)
    return unique + repeated


def _family_disjoint_queues(
    train_records: list[dict[str, object]],
    validation_records: list[dict[str, object]],
    *,
    validation_scenes: int,
    forced_train_scene: str,
) -> tuple[dict[str, list[str]], list[str]]:
    forced_record = next(
        record for record in train_records if record["scene_id"] == forced_train_scene
    )
    forced_family = str(forced_record["source_family"])
    validation_candidates = [
        record
        for record in validation_records
        if record["source_family"] != forced_family
    ]
    validation_order = _unique_family_first(
        validation_candidates,
        _diversity_order(validation_candidates, forced_scene=None),
    )
    held_out_families = []
    for index in validation_order:
        family = str(validation_candidates[index]["source_family"])
        if family not in held_out_families:
            held_out_families.append(family)
        if len(held_out_families) == validation_scenes:
            break
    if len(held_out_families) < validation_scenes:
        raise ValueError("not enough HSSD families for family-disjoint validation")

    held_out = set(held_out_families)
    train_pool = [
        record for record in train_records if record["source_family"] not in held_out
    ]
    validation_pool = [
        record
        for record in validation_candidates
        if record["source_family"] in held_out
    ]
    train_order = _diversity_order(train_pool, forced_scene=forced_train_scene)
    validation_order = _unique_family_first(
        validation_pool,
        _diversity_order(validation_pool, forced_scene=None),
    )
    return (
        {
            "train": [train_pool[index]["scene_id"] for index in train_order],
            "validation": [
                validation_pool[index]["scene_id"] for index in validation_order
            ],
        },
        held_out_families,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("asset_root", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--train-scenes", type=int, default=45)
    parser.add_argument("--validation-scenes", type=int, default=5)
    parser.add_argument("--routes-per-scene", type=int, default=20)
    args = parser.parse_args()

    asset_root = args.asset_root.expanduser().resolve()
    official = yaml.safe_load((asset_root / "scene_splits.yaml").read_text())
    inventory = json.loads(
        (asset_root / "repository_files.json").read_text(encoding="utf-8")
    )
    available_object_configs = {
        Path(path).name
        for path in inventory["files"]
        if path.endswith(".object_config.json")
    }
    catalog = {
        split: [
            _scene_record(asset_root, scene_id, split, available_object_configs)
            for scene_id in scene_ids
        ]
        for split, scene_ids in official.items()
    }
    records = {
        split: [record for record in items if not record["missing_object_handles"]]
        for split, items in catalog.items()
    }
    queues, held_out_families = _family_disjoint_queues(
        records["train"],
        records["val"],
        validation_scenes=args.validation_scenes,
        forced_train_scene="102344280",
    )
    if len(queues["train"]) < args.train_scenes:
        raise ValueError("not enough family-disjoint HSSD training scenes")
    selected = {
        "train": queues["train"][: args.train_scenes],
        "validation": queues["validation"][: args.validation_scenes],
    }
    plan = {
        "selection_method": "family-disjoint split, then greedy k-center over room, area, object and family features",
        "held_out_families": held_out_families,
        "target": {
            "train_scenes": args.train_scenes,
            "validation_scenes": args.validation_scenes,
            "routes_per_scene": args.routes_per_scene,
            "runs": (args.train_scenes + args.validation_scenes)
            * args.routes_per_scene,
        },
        "selected": selected,
        "queues": queues,
        "catalog": catalog,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(plan, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"target": plan["target"], "selected": selected}, indent=2))


if __name__ == "__main__":
    main()
