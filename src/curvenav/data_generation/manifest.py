"""Build deterministic expert episode manifests before launching IsaacLab."""

from __future__ import annotations

from collections import Counter, defaultdict, deque
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Iterable

import numpy as np

from curvenav.data_generation.occupancy import build_navigation_grid
from curvenav.data_generation.planner import (
    PLANNER_ALGORITHM_VERSION,
    PlannerConfig,
    PlanningError,
    SafeEfficientPlanner,
)
from curvenav.data_generation.scene_catalog import SceneRecord


@dataclass(frozen=True)
class ManifestConfig:
    grid_cell_size_m: float = 0.1
    pairs_per_scene: int = 100
    target_episodes: int = 500
    seed: int = 42
    workers: int = 1
    scene_type_pair_caps: tuple[tuple[str, int], ...] = ()
    cache_dir: str | None = None

    def validate(self) -> None:
        if self.grid_cell_size_m <= 0:
            raise ValueError("grid_cell_size_m must be positive")
        if self.pairs_per_scene <= 0 or self.target_episodes <= 0:
            raise ValueError("pairs_per_scene and target_episodes must be positive")
        if self.workers <= 0:
            raise ValueError("workers must be positive")


def _stable_hash(seed: int, *parts: object) -> str:
    value = ":".join([str(seed), *(str(part) for part in parts)])
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _task_order_hash(seed: int, episode: dict[str, object]) -> str:
    return _stable_hash(
        seed,
        episode["scene_type"],
        episode["scene_id"],
        episode["pair_index"],
    )


def _episode_id(
    scene: SceneRecord,
    pair_index: int,
    planner: PlannerConfig,
    grid_cell_size_m: float,
) -> str:
    contract = json.dumps(
        {
            "algorithm_version": PLANNER_ALGORITHM_VERSION,
            "planner": asdict(planner),
            "grid_cell_size_m": grid_cell_size_m,
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    digest = hashlib.sha256(
        f"{scene.scene_type}:{scene.scene_id}:{pair_index}:{contract}".encode("utf-8")
    ).hexdigest()[:16]
    return f"{scene.scene_type}-{scene.scene_id.removesuffix('_usd')}-{pair_index:04d}-{digest}"


def _plan_scene(
    scene: SceneRecord,
    planner_config: PlannerConfig,
    manifest_config: ManifestConfig,
) -> tuple[dict[str, object], list[dict[str, object]], list[dict[str, object]]]:
    pair_cap = dict(manifest_config.scene_type_pair_caps).get(
        scene.scene_type, manifest_config.pairs_per_scene
    )
    cache_path: Path | None = None
    if manifest_config.cache_dir is not None:
        source_state = []
        for source in (scene.navigable_ply, scene.pointgoal_npy):
            stat = Path(source).stat()
            source_state.append((source, stat.st_size, stat.st_mtime_ns))
        fingerprint = hashlib.sha256(
            json.dumps(
                {
                    "cache_version": 2,
                    "algorithm_version": PLANNER_ALGORITHM_VERSION,
                    "scene": scene.to_dict(),
                    "sources": source_state,
                    "planner": asdict(planner_config),
                    "grid_cell_size_m": manifest_config.grid_cell_size_m,
                    "pair_cap": pair_cap,
                    "seed": manifest_config.seed,
                },
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()[:20]
        safe_scene_id = scene.scene_id.replace("/", "_")
        cache_path = (
            Path(manifest_config.cache_dir)
            / f"{scene.scene_type}-{safe_scene_id}-{fingerprint}.json"
        )
        if cache_path.is_file():
            try:
                cached = json.loads(cache_path.read_text(encoding="utf-8"))
                stats = dict(cached["scene_stats"])
                stats["planning_cache_hit"] = True
                return stats, list(cached["accepted"]), list(cached["rejected"])
            except (KeyError, TypeError, ValueError, json.JSONDecodeError):
                cache_path.unlink(missing_ok=True)

    grid = build_navigation_grid(
        scene.navigable_ply,
        cell_size_m=manifest_config.grid_cell_size_m,
    )
    pairs = np.load(scene.pointgoal_npy, allow_pickle=False)
    if pairs.ndim != 2 or pairs.shape[1] < 4:
        raise ValueError(f"point-goal array must have shape [N, >=4], got {pairs.shape}")
    order = sorted(
        range(len(pairs)),
        key=lambda index: _stable_hash(
            manifest_config.seed, scene.scene_type, scene.scene_id, index
        ),
    )
    order = order[:pair_cap]
    planner = SafeEfficientPlanner(grid, planner_config)
    accepted: list[dict[str, object]] = []
    rejected: list[dict[str, object]] = []
    for pair_index in order:
        pair = np.asarray(pairs[pair_index], dtype=np.float64)
        try:
            route = planner.plan(pair[:2], pair[2:4])
        except (PlanningError, ValueError) as error:
            rejected.append(
                {
                    "scene_id": scene.scene_id,
                    "scene_type": scene.scene_type,
                    "pair_index": int(pair_index),
                    "reason": str(error),
                }
            )
            continue
        accepted.append(
            {
                "episode_id": _episode_id(
                    scene,
                    pair_index,
                    planner_config,
                    manifest_config.grid_cell_size_m,
                ),
                "scene_id": scene.scene_id,
                "scene_type": scene.scene_type,
                "scene_family": scene.family_id,
                "official_split": scene.official_split,
                "internal_split": scene.internal_split,
                "pair_index": int(pair_index),
                "initial_yaw": float(pair[4]) if len(pair) > 4 else None,
                "navigable_ply": scene.navigable_ply,
                "pointgoal_npy": scene.pointgoal_npy,
                "usd_path": scene.usd_path,
                **route.to_manifest_fields(),
            }
        )

    finite_clearance = grid.clearance_m[grid.free]
    scene_stats = {
        **scene.to_dict(),
        "grid_shape": list(grid.free.shape),
        "grid_cell_size_m": grid.cell_size_m,
        "navigable_area_m2": float(grid.free.sum() * grid.cell_size_m**2),
        "clearance_p10_m": float(np.percentile(finite_clearance, 10)),
        "clearance_p50_m": float(np.percentile(finite_clearance, 50)),
        "clearance_p90_m": float(np.percentile(finite_clearance, 90)),
        "pairs_considered": len(order),
        "pairs_accepted": len(accepted),
        "pairs_rejected": len(rejected),
        "planning_cache_hit": False,
    }
    if cache_path is not None:
        _atomic_write_json(
            cache_path,
            {"scene_stats": scene_stats, "accepted": accepted, "rejected": rejected},
        )
    return scene_stats, accepted, rejected


def plan_scenes(
    scenes: Iterable[SceneRecord],
    *,
    planner_config: PlannerConfig,
    manifest_config: ManifestConfig,
) -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
    """Plan scenes in independent processes and return deterministic ordering."""

    manifest_config.validate()
    scenes = list(scenes)
    scene_stats: list[dict[str, object]] = []
    candidates: list[dict[str, object]] = []
    rejected: list[dict[str, object]] = []
    if manifest_config.workers == 1:
        results = [
            _plan_scene(scene, planner_config, manifest_config) for scene in scenes
        ]
    else:
        results = []
        with ProcessPoolExecutor(max_workers=manifest_config.workers) as pool:
            futures = {
                pool.submit(_plan_scene, scene, planner_config, manifest_config): scene
                for scene in scenes
            }
            for future in as_completed(futures):
                results.append(future.result())
    for stats, accepted, failed in results:
        scene_stats.append(stats)
        candidates.extend(accepted)
        rejected.extend(failed)
    key = lambda item: (
        str(item.get("scene_type", "")),
        str(item.get("scene_id", "")),
        int(item.get("pair_index", -1)),
    )
    return sorted(scene_stats, key=key), sorted(candidates, key=key), sorted(rejected, key=key)


def balanced_scene_type_pair_caps(
    scenes: Iterable[SceneRecord],
    *,
    base_pairs_per_scene: int,
    maximum_pairs_per_scene: int = 100,
) -> tuple[tuple[str, int], ...]:
    """Oversample scarce scene types before expensive route planning.

    Equal per-scene sampling would reproduce the 48:9 home/commercial scene
    count imbalance.  Caps inversely proportional to scene count produce a
    sufficiently large candidate pool for equal type quotas without planning
    every pair in every home scene.
    """

    counts = Counter(scene.scene_type for scene in scenes)
    if not counts:
        return ()
    largest = max(counts.values())
    return tuple(
        (
            scene_type,
            min(
                maximum_pairs_per_scene,
                int(math.ceil(base_pairs_per_scene * largest / count)),
            ),
        )
        for scene_type, count in sorted(counts.items())
    )


def select_balanced_episodes(
    candidates: Iterable[dict[str, object]],
    *,
    target_episodes: int,
    seed: int,
) -> list[dict[str, object]]:
    """Round-robin scene type, difficulty and scene to limit dataset dominance."""

    candidates = list(candidates)
    if len(candidates) <= target_episodes:
        return candidates
    by_type: defaultdict[str, list[dict[str, object]]] = defaultdict(list)
    for episode in candidates:
        by_type[str(episode["scene_type"])].append(episode)

    def round_robin_type(
        episodes: list[dict[str, object]], quota: int
    ) -> list[dict[str, object]]:
        grouped: defaultdict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
        for episode in episodes:
            grouped[(str(episode["difficulty"]), str(episode["scene_id"]))].append(episode)
        groups: dict[tuple[str, str], deque[dict[str, object]]] = {}
        for key, values in grouped.items():
            values.sort(key=lambda item: _task_order_hash(seed, item))
            groups[key] = deque(values)
        chosen: list[dict[str, object]] = []
        active = sorted(groups)
        while active and len(chosen) < quota:
            next_active = []
            for key in active:
                queue = groups[key]
                if queue and len(chosen) < quota:
                    chosen.append(queue.popleft())
                if queue:
                    next_active.append(key)
            active = next_active
        return chosen

    scene_types = sorted(by_type)
    base_quota, remainder = divmod(target_episodes, len(scene_types))
    selected: list[dict[str, object]] = []
    for index, scene_type in enumerate(scene_types):
        quota = base_quota + int(index < remainder)
        selected.extend(round_robin_type(by_type[scene_type], quota))

    # If one type has too few valid tasks, fill the remainder without changing
    # deterministic ordering.  This is reported in summary.json for visibility.
    if len(selected) < target_episodes:
        selected_ids = {str(item["episode_id"]) for item in selected}
        remaining = [
            item for item in candidates if str(item["episode_id"]) not in selected_ids
        ]
        remaining.sort(key=lambda item: _task_order_hash(seed, item))
        selected.extend(remaining[: target_episodes - len(selected)])
    return sorted(selected, key=lambda item: str(item["episode_id"]))


def _atomic_write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", delete=False
    ) as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, sort_keys=True)
        stream.write("\n")
        temporary = Path(stream.name)
    os.replace(temporary, path)


def _atomic_write_jsonl(path: Path, rows: Iterable[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", delete=False
    ) as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")))
            stream.write("\n")
        temporary = Path(stream.name)
    os.replace(temporary, path)


def write_manifest_bundle(
    output_dir: str | Path,
    *,
    scenes: Iterable[SceneRecord],
    scene_stats: list[dict[str, object]],
    candidates: list[dict[str, object]],
    selected: list[dict[str, object]],
    rejected: list[dict[str, object]],
    planner_config: PlannerConfig,
    manifest_config: ManifestConfig,
) -> dict[str, object]:
    """Atomically write the simulator queue plus its complete audit trail."""

    output = Path(output_dir).expanduser().resolve()
    selected_ids = {str(item["episode_id"]) for item in selected}
    difficulty = Counter(str(item["difficulty"]) for item in selected)
    scene_type = Counter(str(item["scene_type"]) for item in selected)
    internal_split = Counter(str(item["internal_split"]) for item in selected)
    refined_candidates = [
        item
        for item in candidates
        if bool(dict(item.get("refinement", {})).get("accepted", False))
    ]
    refined_selected = [
        item
        for item in selected
        if bool(dict(item.get("refinement", {})).get("accepted", False))
    ]

    def has_dominated_route(item: dict[str, object]) -> bool:
        return any(
            bool(candidate.get("safety_efficiency_dominated", False))
            for candidate in item.get("candidates", [])
        )

    dominated_route_count = sum(
        bool(candidate.get("safety_efficiency_dominated", False))
        for item in candidates
        for candidate in item.get("candidates", [])
    )

    def mean_refinement_reduction(rows: list[dict[str, object]], metric: str) -> float | None:
        reductions = []
        for item in rows:
            refinement = dict(item["refinement"])
            before = float(refinement[f"baseline_{metric}"])
            after = float(refinement[f"refined_{metric}"])
            if abs(before) > 1e-12:
                reductions.append((before - after) / before)
        return float(np.mean(reductions)) if reductions else None

    summary: dict[str, object] = {
        "contract_version": 2,
        "planner_algorithm_version": PLANNER_ALGORITHM_VERSION,
        "source_semantics": "navigable.ply contains footprint-safe robot-center positions",
        "planner": asdict(planner_config),
        "manifest": asdict(manifest_config),
        "scene_count": len(scene_stats),
        "candidate_count": len(candidates),
        "selected_count": len(selected),
        "rejected_count": len(rejected),
        "selected_by_scene_type": dict(sorted(scene_type.items())),
        "selected_by_difficulty": dict(sorted(difficulty.items())),
        "selected_by_internal_split": dict(sorted(internal_split.items())),
        "missing_usd_scene_count": sum(stats["usd_path"] is None for stats in scene_stats),
        "planning_cache_hit_scene_count": sum(
            bool(stats.get("planning_cache_hit", False)) for stats in scene_stats
        ),
        "safety_efficiency_frontier": {
            "dominated_route_count": dominated_route_count,
            "candidate_episode_count_with_dominated_route": sum(
                has_dominated_route(item) for item in candidates
            ),
            "selected_episode_count_with_dominated_route": sum(
                has_dominated_route(item) for item in selected
            ),
        },
        "scan_refinement": {
            "candidate_accepted_count": len(refined_candidates),
            "selected_accepted_count": len(refined_selected),
            "selected_mean_spatial_jerk_reduction_fraction": mean_refinement_reduction(
                refined_selected, "spatial_jerk_rms"
            ),
            "selected_mean_peak_curvature_reduction_fraction": mean_refinement_reduction(
                refined_selected, "maximum_curvature"
            ),
        },
    }
    candidates_with_selection = [
        {**item, "selected": str(item["episode_id"]) in selected_ids}
        for item in candidates
    ]
    _atomic_write_json(output / "summary.json", summary)
    _atomic_write_json(output / "scenes.json", [scene.to_dict() for scene in scenes])
    _atomic_write_json(output / "scene_stats.json", scene_stats)
    _atomic_write_jsonl(output / "manifest.jsonl", selected)
    _atomic_write_jsonl(output / "candidates.jsonl", candidates_with_selection)
    _atomic_write_jsonl(output / "rejected.jsonl", rejected)
    return summary
