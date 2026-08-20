"""Audit DepthSafetySelector choices against offline HSSD candidate labels."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
import hashlib
import json
import math
import multiprocessing
import os
from pathlib import Path
import shutil
import time
from typing import Any

import cv2
import numpy as np

from curvenav.config_io import load_config
from curvenav.data_generation import hssd_policy_labels as label_ops
from curvenav.data_generation.hssd_policy_sidecar_io import validate_sidecar
from curvenav.deployment.runtime import DepthSafetySelector


POLICY_CANDIDATES = 8
EXPECTED_DEPTH_SHAPE = (360, 640)
RANK_TOLERANCE = 1e-6
REASON_BITS = {
    "collision": label_ops.PREFERENCE_COLLISION,
    "safety_margin": label_ops.PREFERENCE_SAFETY_MARGIN,
    "progress": label_ops.PREFERENCE_PROGRESS,
    "clearance": label_ops.PREFERENCE_CLEARANCE,
}
SHARD_FIELDS = (
    "state_sample_index",
    "candidate_offsets",
    "candidate_kind",
    "point_offsets",
    "path_local_xy_m",
    "footprint_collision",
    "minimum_extra_clearance_m",
    "clearance_p05_m",
    "safety_margin_violation",
    "endpoint_geodesic_distance_m",
    "progress_m",
    "progress_per_arc",
    "arc_length_m",
    "curvature_p95_per_m",
    "maximum_curvature_per_m",
    "kinematic_speed_cap_mps",
    "geodesic_valid",
    "pairwise_preference_offsets",
    "pairwise_winner_local_index",
    "pairwise_loser_local_index",
    "pairwise_reason_mask",
)


@dataclass(frozen=True)
class SceneTask:
    dataset_root: str
    sidecar_root: str
    shard_name: str
    maximum_depth_m: float
    batch_size: int
    local_rows: tuple[int, ...] | None


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]


def _distribution(values: list[float | None]) -> dict[str, float | int | None]:
    finite = np.asarray(
        [value for value in values if value is not None and math.isfinite(value)],
        dtype=np.float64,
    )
    if not len(finite):
        return {
            "count": 0,
            "min": None,
            "p05": None,
            "p50": None,
            "p95": None,
            "p99": None,
            "max": None,
            "mean": None,
        }
    return {
        "count": len(finite),
        "min": float(finite.min()),
        "p05": float(np.percentile(finite, 5)),
        "p50": float(np.percentile(finite, 50)),
        "p95": float(np.percentile(finite, 95)),
        "p99": float(np.percentile(finite, 99)),
        "max": float(finite.max()),
        "mean": float(finite.mean()),
    }


def _fraction(count: int, denominator: int) -> float | None:
    return count / denominator if denominator else None


def _average_ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and values[order[end]] == values[order[start]]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + end - 1)
        start = end
    return ranks


def _rank_comparison(
    euclidean_progress: np.ndarray,
    geodesic_progress: np.ndarray,
    geodesic_valid: np.ndarray,
) -> dict[str, float | int | bool | None]:
    valid_indices = np.flatnonzero(geodesic_valid)
    if len(valid_indices) < 2:
        return {
            "valid_candidates": len(valid_indices),
            "comparable_pairs": 0,
            "concordant_pairs": 0,
            "pairwise_agreement": None,
            "spearman": None,
            "top1_agreement": None,
        }
    euclidean = euclidean_progress[valid_indices].astype(np.float64)
    geodesic = geodesic_progress[valid_indices].astype(np.float64)
    comparable = 0
    concordant = 0
    for first in range(len(valid_indices)):
        for second in range(first + 1, len(valid_indices)):
            euclidean_delta = euclidean[first] - euclidean[second]
            geodesic_delta = geodesic[first] - geodesic[second]
            if (
                abs(euclidean_delta) <= RANK_TOLERANCE
                or abs(geodesic_delta) <= RANK_TOLERANCE
            ):
                continue
            comparable += 1
            concordant += int(euclidean_delta * geodesic_delta > 0.0)
    euclidean_rank = _average_ranks(euclidean)
    geodesic_rank = _average_ranks(geodesic)
    if euclidean_rank.std() <= 1e-12 or geodesic_rank.std() <= 1e-12:
        spearman = None
    else:
        spearman = float(np.corrcoef(euclidean_rank, geodesic_rank)[0, 1])
    return {
        "valid_candidates": len(valid_indices),
        "comparable_pairs": comparable,
        "concordant_pairs": concordant,
        "pairwise_agreement": (
            concordant / comparable if comparable else None
        ),
        "spearman": spearman,
        "top1_agreement": bool(
            valid_indices[int(np.argmax(euclidean))]
            == valid_indices[int(np.argmax(geodesic))]
        ),
    }


def _policy_paths(shard: dict[str, np.ndarray], row: int) -> np.ndarray:
    candidate_begin = int(shard["candidate_offsets"][row])
    candidate_end = int(shard["candidate_offsets"][row + 1])
    if candidate_end - candidate_begin != 10:
        raise ValueError("selector audit requires eight policy, expert and hold")
    kinds = shard["candidate_kind"][candidate_begin:candidate_end].tolist()
    if kinds != [0] * POLICY_CANDIDATES + [1, 2]:
        raise ValueError("sidecar candidate order does not match the frozen contract")
    paths = []
    for local_index in range(POLICY_CANDIDATES):
        candidate = candidate_begin + local_index
        point_begin = int(shard["point_offsets"][candidate])
        point_end = int(shard["point_offsets"][candidate + 1])
        paths.append(shard["path_local_xy_m"][point_begin:point_end])
    if len({path.shape for path in paths}) != 1:
        raise ValueError("policy candidate paths do not share one dense shape")
    return np.stack(paths).astype(np.float32, copy=False)


def _candidate_labels(
    shard: dict[str, np.ndarray], row: int
) -> list[dict[str, Any]]:
    begin = int(shard["candidate_offsets"][row])
    labels = []
    for local_index in range(POLICY_CANDIDATES):
        candidate = begin + local_index
        labels.append(
            {
                "footprint_collision": bool(
                    shard["footprint_collision"][candidate]
                ),
                "minimum_extra_clearance_m": float(
                    shard["minimum_extra_clearance_m"][candidate]
                ),
                "safety_margin_violation": bool(
                    shard["safety_margin_violation"][candidate]
                ),
                "progress_m": float(shard["progress_m"][candidate]),
                "geodesic_valid": bool(shard["geodesic_valid"][candidate]),
            }
        )
    return labels


def _stored_preferences(
    shard: dict[str, np.ndarray], row: int
) -> list[tuple[int, int, int]]:
    begin = int(shard["pairwise_preference_offsets"][row])
    end = int(shard["pairwise_preference_offsets"][row + 1])
    return [
        (
            int(shard["pairwise_winner_local_index"][index]),
            int(shard["pairwise_loser_local_index"][index]),
            int(shard["pairwise_reason_mask"][index]),
        )
        for index in range(begin, end)
    ]


def _analyze_state(
    shard: dict[str, np.ndarray],
    row: int,
    record: dict[str, Any],
    paths: np.ndarray,
    selected_index: int,
    task_goal: np.ndarray,
) -> dict[str, Any]:
    candidate_begin = int(shard["candidate_offsets"][row])
    policy_indices = np.arange(candidate_begin, candidate_begin + POLICY_CANDIDATES)
    labels = _candidate_labels(shard, row)
    stored = _stored_preferences(shard, row)
    stored_policy = {
        preference
        for preference in stored
        if preference[0] < POLICY_CANDIDATES
        and preference[1] < POLICY_CANDIDATES
    }
    computed_policy = set(label_ops.pareto_preferences(labels))
    if stored_policy != computed_policy:
        raise ValueError(f"policy Pareto mismatch: {record['sample_id']}")
    dominators = [
        preference
        for preference in computed_policy
        if preference[1] == selected_index
    ]
    dominating_masks = sorted(preference[2] for preference in dominators)
    reason_union = 0
    for mask in dominating_masks:
        reason_union |= mask

    task_distance = np.linalg.norm(task_goal)
    euclidean_progress = task_distance - np.linalg.norm(
        task_goal[None] - paths[:, -1], axis=1
    )
    geodesic_progress = shard["progress_m"][policy_indices]
    geodesic_valid = shard["geodesic_valid"][policy_indices]
    rank = _rank_comparison(
        euclidean_progress,
        geodesic_progress,
        geodesic_valid,
    )
    selected = candidate_begin + selected_index
    collisions = shard["footprint_collision"][policy_indices]
    violations = shard["safety_margin_violation"][policy_indices]
    selected_geodesic_valid = bool(shard["geodesic_valid"][selected])
    return {
        "sample_index": int(record["sample_index"]),
        "sample_id": record["sample_id"],
        "split": record["split"],
        "scene_id": record["scene_id"],
        "episode_id": record["episode_id"],
        "anchor_index": int(record["anchor_index"]),
        "selected_policy_index": selected_index,
        "selected_is_policy_pareto_dominated": bool(dominators),
        "policy_dominator_count": len(dominators),
        "dominating_reason_masks": dominating_masks,
        "dominating_reason_union": reason_union,
        "selected_footprint_collision": bool(
            shard["footprint_collision"][selected]
        ),
        "selected_safety_margin_violation": bool(
            shard["safety_margin_violation"][selected]
        ),
        "selected_geodesic_valid": selected_geodesic_valid,
        "selected_progress_m": (
            float(shard["progress_m"][selected])
            if selected_geodesic_valid
            else None
        ),
        "selected_minimum_extra_clearance_m": float(
            shard["minimum_extra_clearance_m"][selected]
        ),
        "selected_clearance_p05_m": float(
            shard["clearance_p05_m"][selected]
        ),
        "selected_curvature_p95_per_m": float(
            shard["curvature_p95_per_m"][selected]
        ),
        "selected_maximum_curvature_per_m": float(
            shard["maximum_curvature_per_m"][selected]
        ),
        "selected_kinematic_speed_cap_mps": float(
            shard["kinematic_speed_cap_mps"][selected]
        ),
        "selected_euclidean_progress_m": float(
            euclidean_progress[selected_index]
        ),
        "noncollision_policy_available": bool((~collisions).any()),
        "margin_safe_policy_available": bool((~violations).any()),
        "collision_opportunity_loss": bool(
            collisions[selected_index] and (~collisions).any()
        ),
        "margin_opportunity_loss": bool(
            violations[selected_index] and (~violations).any()
        ),
        "expert_dominates_selected_policy": any(
            winner == 8 and loser == selected_index
            for winner, loser, _ in stored
        ),
        "hold_dominates_selected_policy": any(
            winner == 9 and loser == selected_index
            for winner, loser, _ in stored
        ),
        "rank_valid_policy_candidates": rank["valid_candidates"],
        "rank_comparable_pairs": rank["comparable_pairs"],
        "rank_concordant_pairs": rank["concordant_pairs"],
        "rank_pairwise_agreement": rank["pairwise_agreement"],
        "rank_spearman": rank["spearman"],
        "rank_top1_agreement": rank["top1_agreement"],
    }


def _episode_dir(dataset_root: Path, record: dict[str, Any]) -> Path:
    return (
        dataset_root
        / record["split"]
        / f"dataset_hssd_{record['scene_id']}"
        / record["episode_id"].rsplit("/", 1)[-1]
    )


def _audit_scene(task: SceneTask) -> dict[str, Any]:
    cv2.setNumThreads(0)
    started = time.perf_counter()
    dataset_root = Path(task.dataset_root)
    sidecar_root = Path(task.sidecar_root)
    samples = _read_jsonl(dataset_root / "samples.jsonl")
    shard_path = sidecar_root / "scene_shards" / task.shard_name
    with np.load(shard_path, allow_pickle=False) as archive:
        shard = {name: archive[name] for name in SHARD_FIELDS}
    total_rows = len(shard["state_sample_index"])
    rows = (
        list(range(total_rows))
        if task.local_rows is None
        else list(task.local_rows)
    )
    if len(rows) != len(set(rows)) or any(
        row < 0 or row >= total_rows for row in rows
    ):
        raise ValueError(f"invalid local row selection: {task.shard_name}")
    records: dict[int, dict[str, Any]] = {}
    by_episode: dict[str, list[int]] = defaultdict(list)
    for row in rows:
        sample_index = int(shard["state_sample_index"][row])
        source = dict(samples[sample_index])
        source["sample_index"] = sample_index
        records[row] = source
        by_episode[source["episode_id"]].append(row)

    selector = DepthSafetySelector(task.maximum_depth_m)
    states = []
    depth_files = 0
    for episode_id in sorted(by_episode):
        episode_rows = sorted(by_episode[episode_id])
        depth_path = _episode_dir(dataset_root, records[episode_rows[0]]) / "depth_m.npy"
        depth = np.load(depth_path, mmap_mode="r")
        depth_files += 1
        if depth.dtype != np.float32 or depth.shape[1:] != EXPECTED_DEPTH_SHAPE:
            raise ValueError(f"raw depth contract mismatch: {depth_path}")
        for start in range(0, len(episode_rows), task.batch_size):
            batch_rows = episode_rows[start : start + task.batch_size]
            paths = np.stack([_policy_paths(shard, row) for row in batch_rows])
            goals = np.asarray(
                [records[row]["task_goal_local_xy"] for row in batch_rows],
                dtype=np.float32,
            )
            anchors = [int(records[row]["anchor_index"]) for row in batch_rows]
            if any(anchor < 0 or anchor >= len(depth) for anchor in anchors):
                raise ValueError(f"anchor outside raw depth run: {episode_id}")
            current_depth = np.stack([depth[anchor] for anchor in anchors])[
                ..., None
            ]
            if (
                current_depth.dtype != np.float32
                or current_depth.shape[1:] != (*EXPECTED_DEPTH_SHAPE, 1)
            ):
                raise ValueError(f"selector depth batch mismatch: {episode_id}")
            selected, _ = selector.select_indices(paths, current_depth, goals)
            for offset, row in enumerate(batch_rows):
                states.append(
                    _analyze_state(
                        shard,
                        row,
                        records[row],
                        paths[offset],
                        int(selected[offset]),
                        goals[offset],
                    )
                )
        del depth
    states.sort(key=lambda state: state["sample_index"])
    split, scene_id = task.shard_name.removesuffix(".npz").split("__", 1)
    return {
        "shard": task.shard_name,
        "split": split,
        "scene_id": scene_id,
        "states": states,
        "depth_files_mapped": depth_files,
        "shard_decompressions": 1,
        "fallbacks": 0,
        "wall_seconds": time.perf_counter() - started,
    }


def _aggregate(states: list[dict[str, Any]]) -> dict[str, Any]:
    count = len(states)
    dominated = sum(state["selected_is_policy_pareto_dominated"] for state in states)
    reason_state_counts = {
        name: sum(bool(state["dominating_reason_union"] & bit) for state in states)
        for name, bit in REASON_BITS.items()
    }
    reason_pair_counts = {
        name: sum(
            bool(mask & bit)
            for state in states
            for mask in state["dominating_reason_masks"]
        )
        for name, bit in REASON_BITS.items()
    }
    noncollision_opportunities = sum(
        state["noncollision_policy_available"] for state in states
    )
    margin_opportunities = sum(
        state["margin_safe_policy_available"] for state in states
    )
    collision_losses = sum(state["collision_opportunity_loss"] for state in states)
    margin_losses = sum(state["margin_opportunity_loss"] for state in states)
    comparable_pairs = sum(state["rank_comparable_pairs"] for state in states)
    concordant_pairs = sum(state["rank_concordant_pairs"] for state in states)
    rank_top1_states = [
        state for state in states if state["rank_top1_agreement"] is not None
    ]
    selected_counts = Counter(state["selected_policy_index"] for state in states)
    exact_reason_masks = Counter(
        state["dominating_reason_union"]
        for state in states
        if state["selected_is_policy_pareto_dominated"]
    )
    return {
        "states": count,
        "selector_policy_index_counts": {
            str(index): selected_counts[index] for index in range(POLICY_CANDIDATES)
        },
        "selected_policy_pareto_dominated": {
            "count": dominated,
            "fraction": _fraction(dominated, count),
            "reason_state_counts": reason_state_counts,
            "reason_dominating_pair_counts": reason_pair_counts,
            "reason_union_mask_counts": {
                str(mask): value for mask, value in sorted(exact_reason_masks.items())
            },
        },
        "selected_offline_labels": {
            "footprint_collision_count": sum(
                state["selected_footprint_collision"] for state in states
            ),
            "footprint_collision_fraction": _fraction(
                sum(state["selected_footprint_collision"] for state in states),
                count,
            ),
            "safety_margin_violation_count": sum(
                state["selected_safety_margin_violation"] for state in states
            ),
            "safety_margin_violation_fraction": _fraction(
                sum(
                    state["selected_safety_margin_violation"]
                    for state in states
                ),
                count,
            ),
            "geodesic_valid_count": sum(
                state["selected_geodesic_valid"] for state in states
            ),
            "geodesic_valid_fraction": _fraction(
                sum(state["selected_geodesic_valid"] for state in states),
                count,
            ),
            "progress_m": _distribution(
                [state["selected_progress_m"] for state in states]
            ),
            "minimum_extra_clearance_m": _distribution(
                [state["selected_minimum_extra_clearance_m"] for state in states]
            ),
            "clearance_p05_m": _distribution(
                [state["selected_clearance_p05_m"] for state in states]
            ),
            "curvature_p95_per_m": _distribution(
                [state["selected_curvature_p95_per_m"] for state in states]
            ),
            "maximum_curvature_per_m": _distribution(
                [state["selected_maximum_curvature_per_m"] for state in states]
            ),
            "kinematic_speed_cap_mps": _distribution(
                [state["selected_kinematic_speed_cap_mps"] for state in states]
            ),
            "euclidean_progress_m": _distribution(
                [state["selected_euclidean_progress_m"] for state in states]
            ),
        },
        "opportunity_losses": {
            "states_with_noncollision_policy": noncollision_opportunities,
            "selected_collision_despite_noncollision_count": collision_losses,
            "selected_collision_despite_noncollision_fraction_all": _fraction(
                collision_losses, count
            ),
            "selected_collision_despite_noncollision_fraction_opportunity": (
                _fraction(collision_losses, noncollision_opportunities)
            ),
            "states_with_margin_safe_policy": margin_opportunities,
            "selected_violation_despite_margin_safe_count": margin_losses,
            "selected_violation_despite_margin_safe_fraction_all": _fraction(
                margin_losses, count
            ),
            "selected_violation_despite_margin_safe_fraction_opportunity": (
                _fraction(margin_losses, margin_opportunities)
            ),
        },
        "euclidean_vs_geodesic_progress_rank": {
            "states_with_at_least_two_geodesic_valid_policy_candidates": len(
                rank_top1_states
            ),
            "state_top1_agreement_count": sum(
                state["rank_top1_agreement"] for state in rank_top1_states
            ),
            "state_top1_agreement_fraction": _fraction(
                sum(state["rank_top1_agreement"] for state in rank_top1_states),
                len(rank_top1_states),
            ),
            "pooled_comparable_pairs": comparable_pairs,
            "pooled_concordant_pairs": concordant_pairs,
            "pooled_pairwise_order_agreement": _fraction(
                concordant_pairs, comparable_pairs
            ),
            "state_pairwise_order_agreement": _distribution(
                [state["rank_pairwise_agreement"] for state in states]
            ),
            "state_spearman": _distribution(
                [state["rank_spearman"] for state in states]
            ),
        },
        "supervision_reference_not_selector_candidates": {
            "expert_dominates_selected_policy_count": sum(
                state["expert_dominates_selected_policy"] for state in states
            ),
            "expert_dominates_selected_policy_fraction": _fraction(
                sum(
                    state["expert_dominates_selected_policy"]
                    for state in states
                ),
                count,
            ),
            "hold_dominates_selected_policy_count": sum(
                state["hold_dominates_selected_policy"] for state in states
            ),
            "hold_dominates_selected_policy_fraction": _fraction(
                sum(state["hold_dominates_selected_policy"] for state in states),
                count,
            ),
        },
    }


def _fixed_smoke_rows(
    shard_states: dict[str, int], total: int = 128
) -> dict[str, tuple[int, ...]]:
    names = sorted(shard_states)
    if total < len(names):
        raise ValueError("smoke total must cover every scene shard")
    allocation = {name: total // len(names) for name in names}
    remainder = total - sum(allocation.values())
    validation = [name for name in names if name.startswith("validation__")]
    priority = validation or names
    for index in range(remainder):
        allocation[priority[index % len(priority)]] += 1
    selected = {}
    for name in names:
        count = allocation[name]
        rows = np.linspace(0, shard_states[name] - 1, count, dtype=np.int64)
        if len(np.unique(rows)) != count:
            raise ValueError(f"smoke row selection is not unique: {name}")
        selected[name] = tuple(int(row) for row in rows)
    return selected


def _run_tasks(
    tasks: list[SceneTask], workers: int, phase: str
) -> list[dict[str, Any]]:
    results = []
    context = multiprocessing.get_context("fork")
    with ProcessPoolExecutor(max_workers=workers, mp_context=context) as executor:
        futures = {executor.submit(_audit_scene, task): task for task in tasks}
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            print(
                json.dumps(
                    {
                        "phase": phase,
                        "shard": result["shard"],
                        "states": len(result["states"]),
                        "wall_seconds": result["wall_seconds"],
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
    return sorted(results, key=lambda result: result["shard"])


def _render_report(summary: dict[str, Any]) -> str:
    full = summary["full"]
    dominated = full["selected_policy_pareto_dominated"]
    labels = full["selected_offline_labels"]
    losses = full["opportunity_losses"]
    rank = full["euclidean_vs_geodesic_progress_rank"]
    runtime = summary["runtime"]
    return f"""# HSSD v2 DepthSafetySelector gap audit

## Scope

This audit replays the current `DepthSafetySelector` over the same eight policy candidates using each anchor's original 360x640 float32 metre depth frame and original task goal. It does not resize depth, change selector thresholds, construct a scalar oracle, or include expert/hold in selector metrics.

All collision, clearance and geodesic values below are offline navigation-grid labels. They are not simulator truth, MPC rollout labels or closed-loop results.

## Main result

- Selected policy candidate is strictly Pareto dominated by another policy candidate in **{dominated['count']} / {full['states']} ({dominated['fraction']:.2%})** states.
- Selected offline collision fraction: **{labels['footprint_collision_fraction']:.2%}**.
- Selected safety-margin violation fraction: **{labels['safety_margin_violation_fraction']:.2%}**.
- Selected geodesic-valid fraction: **{labels['geodesic_valid_fraction']:.2%}**.
- Collision opportunity loss: **{losses['selected_collision_despite_noncollision_count']}** states, **{losses['selected_collision_despite_noncollision_fraction_opportunity']:.2%}** of states with a non-collision policy alternative.
- Margin opportunity loss: **{losses['selected_violation_despite_margin_safe_count']}** states, **{losses['selected_violation_despite_margin_safe_fraction_opportunity']:.2%}** of states with a margin-safe policy alternative.

## Progress ranking

Among states with at least two geodesic-valid policy candidates:

- Euclidean-progress top-1 agrees with geodesic-progress top-1 in **{rank['state_top1_agreement_fraction']:.2%}**.
- Pooled comparable-pair order agreement: **{rank['pooled_pairwise_order_agreement']:.2%}**.
- State-wise Spearman mean / p50: **{rank['state_spearman']['mean']:.3f} / {rank['state_spearman']['p50']:.3f}**.

## Runtime and interpretation

- Fixed smoke: **128 states**, {runtime['smoke_wall_seconds']:.2f} s.
- Full audit: **{full['states']} states**, {runtime['full_wall_seconds']:.2f} s wall time with {runtime['workers']} scene workers.
- Fallbacks: **0**; each shard was decompressed once per smoke/full pass and raw depth was mmaped once per used episode.

Strict Pareto-dominated selections and safety opportunity losses quantify offline learnable room for a critic. They do not by themselves justify keeping or discarding a learned selector. That decision remains gated on matched quick100 current-selector versus oracle-selector closed-loop evaluation.
"""


def audit(
    dataset_root: Path,
    sidecar_root: Path,
    config_path: Path,
    output_dir: Path,
    *,
    workers: int,
    batch_size: int,
) -> dict[str, Any]:
    dataset_root = dataset_root.resolve()
    sidecar_root = sidecar_root.resolve()
    config_path = config_path.resolve()
    output_dir = output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(f"selector gap audit already exists: {output_dir}")
    sidecar_validation = validate_sidecar(
        sidecar_root,
        source_dataset_root=dataset_root,
        verify_hashes=True,
    )
    manifest = json.loads(
        (sidecar_root / "manifest.json").read_text(encoding="utf-8")
    )
    config = load_config(config_path)
    config.validate()
    shard_states = {
        name: int(record["states"])
        for name, record in manifest["shards"].items()
    }
    worker_count = min(workers, len(shard_states))
    common = {
        "dataset_root": str(dataset_root),
        "sidecar_root": str(sidecar_root),
        "maximum_depth_m": config.data.max_depth_m,
        "batch_size": batch_size,
    }

    smoke_rows = _fixed_smoke_rows(shard_states)
    smoke_tasks = [
        SceneTask(
            shard_name=name,
            local_rows=smoke_rows[name],
            **common,
        )
        for name in sorted(shard_states)
    ]
    smoke_started = time.perf_counter()
    smoke_scene_results = _run_tasks(smoke_tasks, worker_count, "smoke128")
    smoke_wall = time.perf_counter() - smoke_started
    smoke_states = sorted(
        [
            state
            for scene in smoke_scene_results
            for state in scene["states"]
        ],
        key=lambda state: state["sample_index"],
    )
    if (
        len(smoke_states) != 128
        or len({state["sample_id"] for state in smoke_states}) != 128
        or {state["split"] for state in smoke_states} != {"train", "validation"}
        or len({state["scene_id"] for state in smoke_states}) != len(shard_states)
        or any(scene["fallbacks"] for scene in smoke_scene_results)
    ):
        raise RuntimeError("fixed selector smoke failed its coverage contract")
    smoke_summary = _aggregate(smoke_states)
    estimated_full_seconds = smoke_wall * sum(shard_states.values()) / 128
    print(
        json.dumps(
            {
                "smoke_states": len(smoke_states),
                "smoke_wall_seconds": smoke_wall,
                "estimated_full_seconds": estimated_full_seconds,
                "dominated_fraction": smoke_summary[
                    "selected_policy_pareto_dominated"
                ]["fraction"],
            },
            sort_keys=True,
        ),
        flush=True,
    )

    full_tasks = [
        SceneTask(
            shard_name=name,
            local_rows=None,
            **common,
        )
        for name in sorted(shard_states)
    ]
    full_started = time.perf_counter()
    full_scene_results = _run_tasks(full_tasks, worker_count, "full")
    full_wall = time.perf_counter() - full_started
    full_states = sorted(
        [state for scene in full_scene_results for state in scene["states"]],
        key=lambda state: state["sample_index"],
    )
    if (
        [state["sample_index"] for state in full_states]
        != list(range(manifest["states"]))
        or any(scene["fallbacks"] for scene in full_scene_results)
    ):
        raise RuntimeError("full selector audit did not cover stable sample indices")
    full_summary = _aggregate(full_states)
    scene_summaries = {
        scene["shard"]: {
            "split": scene["split"],
            "scene_id": scene["scene_id"],
            "wall_seconds": scene["wall_seconds"],
            "depth_files_mapped": scene["depth_files_mapped"],
            "shard_decompressions": scene["shard_decompressions"],
            "fallbacks": scene["fallbacks"],
            "metrics": _aggregate(scene["states"]),
        }
        for scene in full_scene_results
    }
    runtime_path = Path(__file__).parents[1] / "deployment" / "runtime.py"
    summary = {
        "audit_version": "hssd_v2_selector_gap_v1",
        "scope": (
            "offline navigation-grid comparison of current DepthSafetySelector; "
            "not simulator truth or closed-loop"
        ),
        "inputs": {
            "dataset_root": str(dataset_root),
            "dataset_bundle_sha256": _sha256_file(
                dataset_root / "audit" / "data_files.sha256"
            ),
            "dataset_manifest_sha256": _sha256_file(
                dataset_root / "dataset_manifest.json"
            ),
            "sidecar_root": str(sidecar_root),
            "sidecar_sha256sums_sha256": _sha256_file(
                sidecar_root / "SHA256SUMS"
            ),
            "sidecar_manifest_sha256": _sha256_file(
                sidecar_root / "manifest.json"
            ),
            "selector_source": str(runtime_path),
            "selector_source_sha256": _sha256_file(runtime_path),
            "config": str(config_path),
            "config_sha256": _sha256_file(config_path),
            "maximum_depth_m": config.data.max_depth_m,
        },
        "input_contract": {
            "depth": (
                "anchor frame from raw depth_m.npy, float32 metres, [360,640]; "
                "no resize"
            ),
            "task_goal": "original samples.jsonl task_goal_local_xy",
            "candidates": "policy local indices 0..7 from each sidecar state",
            "selector": (
                "curvenav.deployment.runtime.DepthSafetySelector.select_indices"
            ),
            "fallbacks": 0,
        },
        "definitions": {
            "pareto_dominated": (
                "another policy candidate is no worse in collision, 0.1 m "
                "margin, geodesic progress and extra clearance, and strictly "
                "better in at least one"
            ),
            "rank_pairwise_agreement": (
                "fraction of non-tied policy pairs whose Euclidean and geodesic "
                "progress order has the same sign"
            ),
            "rank_spearman": (
                "Pearson correlation of state-wise average ranks over geodesic-"
                "valid policy candidates"
            ),
            "no_scalar_oracle": True,
        },
        "sidecar_validation": sidecar_validation,
        "smoke128": {
            "sample_indices": [state["sample_index"] for state in smoke_states],
            "metrics": smoke_summary,
        },
        "full": full_summary,
        "per_scene": scene_summaries,
        "runtime": {
            "workers": worker_count,
            "batch_size": batch_size,
            "smoke_wall_seconds": smoke_wall,
            "smoke_linear_full_estimate_seconds": estimated_full_seconds,
            "full_wall_seconds": full_wall,
            "states_per_second": len(full_states) / full_wall,
        },
        "interpretation": (
            "Measures offline learnable selector room only; final keep/discard "
            "requires matched quick100 current-selector versus oracle-selector."
        ),
    }

    temporary = output_dir.with_name(f".{output_dir.name}.tmp-{os.getpid()}")
    if temporary.exists():
        raise FileExistsError(f"selector audit temporary directory exists: {temporary}")
    temporary.mkdir(parents=True)
    try:
        (temporary / "scenes").mkdir()
        (temporary / "summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        (temporary / "smoke128.json").write_text(
            json.dumps(summary["smoke128"], indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        (temporary / "report.md").write_text(
            _render_report(summary), encoding="utf-8"
        )
        (temporary / "state_metrics.jsonl").write_text(
            "".join(
                json.dumps(state, sort_keys=True) + "\n" for state in full_states
            ),
            encoding="utf-8",
        )
        for shard, scene in scene_summaries.items():
            (temporary / "scenes" / f"{shard.removesuffix('.npz')}.json").write_text(
                json.dumps(scene, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
        files = sorted(path for path in temporary.rglob("*") if path.is_file())
        (temporary / "SHA256SUMS").write_text(
            "".join(
                f"{_sha256_file(path)}  {path.relative_to(temporary).as_posix()}\n"
                for path in files
            ),
            encoding="utf-8",
        )
        output_dir.parent.mkdir(parents=True, exist_ok=True)
        os.replace(temporary, output_dir)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    print(json.dumps(full_summary, indent=2, sort_keys=True))
    return summary


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset_root", type=Path)
    parser.add_argument("sidecar_root", type=Path)
    parser.add_argument("config", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args(argv)
    audit(
        args.dataset_root,
        args.sidecar_root,
        args.config,
        args.output_dir,
        workers=args.workers,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()
