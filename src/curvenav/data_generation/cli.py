"""Command-line entry point for building expert episode manifests."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from curvenav.data_generation.manifest import (
    ManifestConfig,
    balanced_scene_type_pair_caps,
    plan_scenes,
    select_balanced_episodes,
    write_manifest_bundle,
)
from curvenav.data_generation.planner import PlannerConfig
from curvenav.data_generation.scene_catalog import (
    assign_group_disjoint_validation,
    discover_training_scenes,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build a leakage-safe, safety/efficiency-audited expert episode manifest "
            "from X-NavDP navigation metadata."
        )
    )
    parser.add_argument("data_root", help="Root containing scene_split.json and navigation_metadata")
    parser.add_argument("output_dir", help="Output directory for manifest and audit files")
    parser.add_argument("--scene-split-file", default=None)
    parser.add_argument(
        "--cache-dir",
        default=None,
        help="Per-scene planning cache; defaults to <output_dir>/planning_cache",
    )
    parser.add_argument("--target-episodes", type=int, default=500)
    parser.add_argument("--pairs-per-scene", type=int, default=100)
    parser.add_argument(
        "--no-balance-scene-types",
        action="store_true",
        help="Use the same pair cap for every type instead of oversampling scarce types",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=max(1, min(8, (os.cpu_count() or 2) // 2)),
        help="Independent CPU scene planners; rendering is a later stage",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--validation-family-fraction", type=float, default=0.15)
    parser.add_argument("--grid-cell-size", type=float, default=0.1)
    parser.add_argument("--minimum-clearance", type=float, default=0.1)
    parser.add_argument("--preferred-clearance", type=float, default=0.3)
    parser.add_argument("--maximum-detour-ratio", type=float, default=1.2)
    parser.add_argument("--output-spacing", type=float, default=0.05)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    scenes = discover_training_scenes(
        args.data_root,
        scene_split_file=args.scene_split_file,
    )
    scenes = assign_group_disjoint_validation(
        scenes,
        validation_fraction=args.validation_family_fraction,
        seed=args.seed,
    )
    planner_config = PlannerConfig(
        preferred_clearance_m=args.preferred_clearance,
        minimum_clearance_m=args.minimum_clearance,
        maximum_safe_detour_ratio=args.maximum_detour_ratio,
        output_spacing_m=args.output_spacing,
    )
    pair_caps = (
        ()
        if args.no_balance_scene_types
        else balanced_scene_type_pair_caps(
            scenes,
            base_pairs_per_scene=args.pairs_per_scene,
        )
    )
    manifest_config = ManifestConfig(
        grid_cell_size_m=args.grid_cell_size,
        pairs_per_scene=args.pairs_per_scene,
        target_episodes=args.target_episodes,
        seed=args.seed,
        workers=args.workers,
        scene_type_pair_caps=pair_caps,
        cache_dir=str(
            Path(args.cache_dir).expanduser().resolve()
            if args.cache_dir is not None
            else (Path(args.output_dir).expanduser().resolve() / "planning_cache")
        ),
    )
    scene_stats, candidates, rejected = plan_scenes(
        scenes,
        planner_config=planner_config,
        manifest_config=manifest_config,
    )
    selected = select_balanced_episodes(
        candidates,
        target_episodes=manifest_config.target_episodes,
        seed=manifest_config.seed,
    )
    summary = write_manifest_bundle(
        Path(args.output_dir),
        scenes=scenes,
        scene_stats=scene_stats,
        candidates=candidates,
        selected=selected,
        rejected=rejected,
        planner_config=planner_config,
        manifest_config=manifest_config,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
