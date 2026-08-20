"""Audit whether the current HSSD planner supplies same-state route alternatives.

The audit is deliberately read-only with respect to the v2 dataset.  It reruns
the existing planner on a small, deterministic set of already-rendered states
and writes only an independent design report.  Geometric path differences are
reported as diagnostics and are never promoted to certified topology labels.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import time
from typing import Any

import numpy as np
from scipy.spatial import cKDTree

from curvenav.data_generation.occupancy import NavigationGrid, resample_polyline
from curvenav.data_generation.planner import SafeEfficientPlanner, _path_length


# One fixed state per episode covers every pilot route and scene without paying
# to rerun five full smooth/validation passes for all 12,210 training windows.
ANCHOR_FRACTIONS = (0.0,)
GEOMETRY_DIAGNOSTIC_SEPARATION_M = 0.5
TARGET_TRAIN_EPISODES = 1000
TARGET_CANDIDATES_PER_STATE = 4
POLICY_PATH_POINTS = 64


SOURCE_EVIDENCE = [
    {
        "path": "src/curvenav/data_generation/planner.py",
        "lines": "154-166, 215-256",
        "fact": "one deterministic plan call varies only clearance weight and returns one selected route",
    },
    {
        "path": "src/curvenav/data_generation/planner.py",
        "lines": "258-305",
        "fact": "all alternatives use the same 8-neighbour A* graph, endpoints and deterministic tie order",
    },
    {
        "path": "src/curvenav/data_generation/hssd_pilot.py",
        "lines": "150-225",
        "fact": "route sampling draws different start/goal pairs and does not request multiple routes for one pair",
    },
    {
        "path": "src/curvenav/data_generation/hssd_v2.py",
        "lines": "157-265, 389-450",
        "fact": "v2 stores one route per episode; every sample reserves empty alternatives and null critic labels",
    },
    {
        "path": "src/curvenav/data_generation/occupancy.py",
        "lines": "97-178",
        "fact": "the navigation mask is already footprint-center safe and clearance is extra distance to its boundary",
    },
]


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def _load_grid(path: Path) -> NavigationGrid:
    with np.load(path) as values:
        return NavigationGrid(
            free=values["free"].astype(bool),
            clearance_m=values["clearance_m"].astype(np.float32),
            origin_xy=values["origin_xy"].astype(np.float64),
            cell_size_m=float(values["cell_size_m"]),
        )


def _path_digest(path: np.ndarray) -> str:
    canonical = np.round(resample_polyline(path, 0.05), 4).astype("<f4")
    return hashlib.sha256(canonical.tobytes()).hexdigest()


def _symmetric_hausdorff(first: np.ndarray, second: np.ndarray) -> float:
    first = resample_polyline(first, 0.05)
    second = resample_polyline(second, 0.05)
    forward = cKDTree(second).query(first, k=1)[0].max(initial=0.0)
    backward = cKDTree(first).query(second, k=1)[0].max(initial=0.0)
    return float(max(forward, backward))


def _current_planner_candidates(
    planner: SafeEfficientPlanner, start_xy: np.ndarray, goal_xy: np.ndarray
) -> list[dict[str, Any]]:
    """Reproduce candidate construction without changing the production planner."""

    config = planner.config
    start = planner.grid.snap_to_free(start_xy, config.snap_distance_m)
    goal = planner.grid.snap_to_free(goal_xy, config.snap_distance_m)
    reference_length: float | None = None
    candidates: list[dict[str, Any]] = []
    for clearance_weight in config.clearance_weights:
        indices = planner._astar(start, goal, clearance_weight)
        if indices is None:
            continue
        path = planner.grid.grid_to_world(indices)
        path = np.vstack([start_xy, path, goal_xy])
        path = planner._shortcut(path)
        path, _ = planner._safe_smooth(path)
        if not planner.grid.path_is_safe(
            path,
            minimum_clearance_m=config.minimum_clearance_m,
            sample_step_m=config.validation_step_m,
        ):
            continue
        length = _path_length(path)
        if clearance_weight == 0.0:
            reference_length = length
        if reference_length is None:
            continue
        metrics = planner._metrics(path, reference_length)
        if (
            metrics.maximum_curvature > config.maximum_curvature
            or metrics.curvature_p95 > config.maximum_curvature_p95
            or metrics.reference_length_ratio > config.maximum_safe_detour_ratio + 1e-6
        ):
            continue
        candidates.append(
            {
                "clearance_weight": float(clearance_weight),
                "path": path,
                "path_sha256": _path_digest(path),
                "length_m": metrics.length_m,
                "clearance_p05_m": metrics.clearance_p05_m,
            }
        )
    return candidates


def _state_probe(
    episode: dict[str, Any], anchor_index: int, world_xy: np.ndarray, planner: SafeEfficientPlanner
) -> dict[str, Any]:
    started = time.monotonic()
    candidates = _current_planner_candidates(
        planner, world_xy[anchor_index], world_xy[-1]
    )
    unique: dict[str, np.ndarray] = {}
    for candidate in candidates:
        unique.setdefault(candidate["path_sha256"], candidate["path"])
    paths = list(unique.values())
    separations = [
        _symmetric_hausdorff(paths[first], paths[second])
        for first in range(len(paths))
        for second in range(first + 1, len(paths))
    ]
    maximum_separation = max(separations, default=0.0)
    return {
        "sample_id": f"{episode['episode_id']}:{anchor_index:04d}",
        "episode_id": episode["episode_id"],
        "split": episode["split"],
        "scene_id": episode["scene_id"],
        "difficulty": episode["difficulty"],
        "anchor_index": anchor_index,
        "same_observation_and_task_goal": True,
        "eligible_clearance_weight_candidates": len(candidates),
        "unique_smoothed_geometry_count": len(unique),
        "maximum_pairwise_hausdorff_m": maximum_separation,
        "has_geometrically_separated_pair": (
            maximum_separation >= GEOMETRY_DIAGNOSTIC_SEPARATION_M
        ),
        "certified_topology_count": None,
        "certified_topology_reason": (
            "the current planner records neither a corridor graph nor a homotopy signature"
        ),
        "elapsed_seconds": time.monotonic() - started,
    }


def _logical_schema() -> dict[str, Any]:
    return {
        "schema_version": "curvenav_critic_sidecar_v1",
        "storage": (
            "per-scene ragged NPZ shards plus JSONL state index; observations and task_goal "
            "remain referenced by immutable v2 sample_id"
        ),
        "state": {
            "sample_id": "string, foreign key to v2 samples.jsonl",
            "episode_id": "string",
            "anchor_index": "int32",
            "candidate_offsets": "int64 [num_states + 1]",
            "pairwise_preference_offsets": "int64 [num_states + 1]",
        },
        "candidate": {
            "candidate_id": "string, deterministic producer/revision/seed hash",
            "kind": "enum: expert | topology | hold | policy",
            "producer_revision": "string",
            "producer_seed": "uint64",
            "topology_class_id": "nullable string, hash of ordered corridor-graph edge ids",
            "corridor_edge_ids": "ragged int32; empty for hold and unclassified policy candidates",
            "first_branch_side": "nullable enum: left | right | straight; null for hold",
            "point_offsets": "int64 [num_candidates + 1]",
            "path_local_xy_m": "float32 [num_points, 2], current body frame, first point [0, 0]",
            "arc_length_m": "float32",
        },
        "offline_labels": {
            "footprint_collision": "bool",
            "minimum_extra_clearance_m": "float32",
            "clearance_p05_m": "float32",
            "safety_margin_violation": "bool for minimum_extra_clearance_m < 0.1 m",
            "endpoint_geodesic_distance_m": "float32",
            "progress_m": "float32",
            "progress_per_arc": "float32 dimensionless",
            "curvature_p95_per_m": "float32 m^-1",
            "maximum_curvature_per_m": "float32 m^-1",
            "nominal_peak_angular_rate_radps": "float32 geometric proxy",
            "kinematic_speed_cap_mps": "float32 geometric proxy",
        },
        "closed_loop_labels": {
            "status": "enum: not_run | solved | solver_failed | rollout_failed",
            "collision": "nullable bool",
            "executed_progress_m": "nullable float32",
            "tracking_rmse_m": "nullable float32",
            "tracking_p95_m": "nullable float32",
            "endpoint_error_m": "nullable float32",
            "linear_saturation_fraction": "nullable float32 dimensionless",
            "angular_saturation_fraction": "nullable float32 dimensionless",
            "rollout_duration_s": "nullable float32",
        },
        "pairwise_preference": {
            "winner_candidate_index": "int32",
            "loser_candidate_index": "int32",
            "reason_mask": (
                "collision, safety margin, progress, clearance, or closed-loop tracking dominance"
            ),
            "rule": "store only Pareto-dominant pairs; do not invent a weighted scalar utility",
        },
    }


def _definitions() -> dict[str, str]:
    return {
        "resampling": "R_0.025(tau): uniform 0.025 m arc-length samples plus every original vertex",
        "footprint_collision": "exists q in R_0.025(tau) with q outside the footprint-center-safe free mask F",
        "minimum_extra_clearance_m": "min over q in R_0.025(tau) of distance from q to boundary of F; zero when outside F",
        "clearance_p05_m": "5th percentile of the same extra-clearance samples",
        "progress_m": "D_F0.1(start, goal) - D_F0.1(endpoint, goal), using 8-neighbour metric geodesic distance in clearance>=0.1 m cells",
        "progress_per_arc": "progress_m / max(arc_length_m, 1e-6 m); hold is defined as zero",
        "curvature": "three-point Menger curvature on 0.05 m resampled path, in m^-1",
        "nominal_peak_angular_rate_radps": "0.5 m/s * maximum_curvature_per_m",
        "kinematic_speed_cap_mps": "min(0.5 m/s, 0.5 rad/s / maximum_curvature_per_m); 0.5 m/s for zero curvature",
        "topology_class": "ordered start-to-goal corridor-edge sequence on a clearance-constrained medial-axis graph",
        "first_branch_side": "sign of the 2-D cross product between body-forward and the first differing corridor edge; positive is left, negative is right",
        "offline_pareto_preference": "a dominates b only if no worse in collision, safety margin, progress and clearance, and strictly better in at least one",
    }


def audit(dataset_root: Path, output_dir: Path) -> dict[str, Any]:
    dataset_root = dataset_root.resolve()
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    episodes = _read_jsonl(dataset_root / "episodes.jsonl")
    samples = _read_jsonl(dataset_root / "samples.jsonl")
    dataset_manifest = json.loads((dataset_root / "dataset_manifest.json").read_text())
    started = time.monotonic()
    grids: dict[tuple[str, str], NavigationGrid] = {}
    state_metrics = []
    route_frames = 0

    for episode in episodes:
        scene_key = (episode["split"], episode["scene_id"])
        scene_dir = (
            dataset_root
            / episode["split"]
            / f"dataset_hssd_{episode['scene_id']}"
        )
        if scene_key not in grids:
            grids[scene_key] = _load_grid(scene_dir / "navigation_grid.npz")
        planner = SafeEfficientPlanner(grids[scene_key])
        episode_dir = scene_dir / episode["run_id"]
        with np.load(episode_dir / "route.npz") as route:
            world_xy = route["world_xy"].astype(np.float64)
        route_frames += len(world_xy)
        anchors = sorted(
            {
                min(int(round(fraction * (len(world_xy) - 2))), len(world_xy) - 2)
                for fraction in ANCHOR_FRACTIONS
            }
        )
        for anchor in anchors:
            try:
                state_metrics.append(_state_probe(episode, anchor, world_xy, planner))
            except Exception as error:
                state_metrics.append(
                    {
                        "sample_id": f"{episode['episode_id']}:{anchor:04d}",
                        "episode_id": episode["episode_id"],
                        "split": episode["split"],
                        "scene_id": episode["scene_id"],
                        "difficulty": episode["difficulty"],
                        "anchor_index": anchor,
                        "error": f"{type(error).__name__}: {error}",
                    }
                )

    successful = [row for row in state_metrics if "error" not in row]
    elapsed = time.monotonic() - started
    depth_bytes = sum(path.stat().st_size for path in dataset_root.rglob("depth_m.npy"))
    dataset_bytes = sum(
        path.stat().st_size for path in dataset_root.rglob("*") if path.is_file()
    )
    train_samples = len(samples) * TARGET_TRAIN_EPISODES / len(episodes)
    split_scale = TARGET_TRAIN_EPISODES / len(episodes)
    split_estimate = {
        split: {
            "episodes": round(sum(row["split"] == split for row in episodes) * split_scale),
            "samples": round(sum(row["split"] == split for row in samples) * split_scale),
            "frames": round(
                sum(row["frames"] for row in episodes if row["split"] == split)
                * split_scale
            ),
        }
        for split in ("train", "validation")
    }
    fixed_candidate_bytes = (
        TARGET_CANDIDATES_PER_STATE
        * POLICY_PATH_POINTS
        * 2
        * np.dtype(np.float32).itemsize
    )
    labels_and_masks_bytes = TARGET_CANDIDATES_PER_STATE * (
        POLICY_PATH_POINTS + 16 * np.dtype(np.float32).itemsize
    )
    estimated_sidecar_bytes = int(
        train_samples * (fixed_candidate_bytes + labels_and_masks_bytes)
    )

    summary = {
        "dataset_root": str(dataset_root),
        "read_only_inputs": ["episodes.jsonl", "samples.jsonl", "route.npz", "navigation_grid.npz"],
        "probe": {
            "episodes": len(episodes),
            "states_requested": len(state_metrics),
            "states_successful": len(successful),
            "states_failed": len(state_metrics) - len(successful),
            "anchor_fractions": list(ANCHOR_FRACTIONS),
            "elapsed_seconds": elapsed,
            "mean_seconds_per_state": elapsed / max(len(successful), 1),
            "states_with_multiple_unique_geometries": sum(
                row["unique_smoothed_geometry_count"] > 1 for row in successful
            ),
            "states_with_geometrically_separated_pair": sum(
                row["has_geometrically_separated_pair"] for row in successful
            ),
            "geometric_separation_diagnostic_m": GEOMETRY_DIAGNOSTIC_SEPARATION_M,
            "certified_multi_topology_states": None,
            "certification_failure": (
                "current planner has no corridor graph/homotopy signature and discards candidate geometry"
            ),
        },
        "current_dataset": {
            "episodes": len(episodes),
            "samples": len(samples),
            "frames": route_frames,
            "dataset_bytes": dataset_bytes,
            "depth_bytes": depth_bytes,
            "samples_with_nonempty_alternatives": sum(bool(row["alternatives"]) for row in samples),
            "samples_with_critic_labels": sum(row["critic_labels"] is not None for row in samples),
        },
        "scale_estimate": {
            "recommended_episode_target": TARGET_TRAIN_EPISODES,
            "estimated_samples": round(train_samples),
            "estimated_frames": round(route_frames * TARGET_TRAIN_EPISODES / len(episodes)),
            "estimated_split": split_estimate,
            "estimated_dataset_bytes_linear": round(dataset_bytes * TARGET_TRAIN_EPISODES / len(episodes)),
            "estimated_depth_bytes_linear": round(depth_bytes * TARGET_TRAIN_EPISODES / len(episodes)),
            "measured_pilot_render_wall_seconds": dataset_manifest["elapsed_seconds"],
            "estimated_render_wall_seconds_same_two_worker_setup": (
                dataset_manifest["elapsed_seconds"]
                * TARGET_TRAIN_EPISODES
                / len(episodes)
            ),
            "candidate_sidecar_assumption": (
                f"{TARGET_CANDIDATES_PER_STATE} candidates/state, {POLICY_PATH_POINTS} points, float32 xy plus masks and 16 scalar slots"
            ),
            "estimated_uncompressed_candidate_sidecar_bytes": estimated_sidecar_bytes,
            "estimated_offline_candidate_audit_seconds": (
                elapsed * train_samples / max(len(successful), 1)
            ),
            "closed_loop_runtime_seconds": None,
            "closed_loop_runtime_reason": (
                "must benchmark the active MPC/simulator rollout; no measured per-candidate wall time is available"
            ),
        },
        "verdict": {
            "current_planner_is_multitopology_generator": False,
            "reason": (
                "clearance-weight variants may differ geometrically but are not enumerated, stored, "
                "or certified as distinct corridor classes"
            ),
            "do_not_use": ["rotated copies", "noisy copies", "warped expert copies"],
            "recommended_generator": (
                "clearance-constrained medial-axis corridor graph, loopless K-shortest edge sequences, "
                "then the existing safe smoothing and continuous validation"
            ),
            "candidate_policy": (
                "expert forward + at most two genuine corridor classes + one hold; fewer candidates "
                "when the scene has no qualifying alternative"
            ),
            "hard_negative_policy": (
                "after an oracle-selector gap, ingest actual checkpoint proposals and label them; "
                "never manufacture collision negatives by perturbing the expert"
            ),
        },
        "offline_available": [
            "footprint collision on the footprint-center-safe navigation mask",
            "extra clearance and safety-margin violation",
            "geodesic progress and path efficiency",
            "path length and geometric curvature proxies",
            "corridor topology id after the new graph enumerator exists",
        ],
        "requires_active_mpc_closed_loop": [
            "actual trackability and solver failure",
            "executed collision and executed progress",
            "tracking RMSE/p95/endpoint error",
            "linear/angular saturation fractions and rollout time",
        ],
        "source_evidence": SOURCE_EVIDENCE,
        "label_definitions": _definitions(),
    }
    (output_dir / "probe_state_metrics.json").write_text(
        json.dumps(state_metrics, indent=2, sort_keys=True) + "\n"
    )
    (output_dir / "critic_sidecar_schema_v1.json").write_text(
        json.dumps(_logical_schema(), indent=2, sort_keys=True) + "\n"
    )
    (output_dir / "multitopology_audit_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    _write_report(output_dir / "multitopology_design_audit.md", summary)
    return summary


def _write_report(path: Path, summary: dict[str, Any]) -> None:
    probe = summary["probe"]
    current = summary["current_dataset"]
    scale = summary["scale_estimate"]
    source_lines = "\n".join(
        f"- `{item['path']}:{item['lines']}`: {item['fact']}."
        for item in summary["source_evidence"]
    )
    offline_lines = "\n".join(f"- {item}." for item in summary["offline_available"])
    closed_loop_lines = "\n".join(
        f"- {item}." for item in summary["requires_active_mpc_closed_loop"]
    )
    report = f"""# CurveNav same-state multi-topology and critic-label design audit

## Verdict

The current HSSD generator is **not** a certified same-state multi-topology generator. It samples different start/goal episodes and selects one route. Clearance-weight variants can differ geometrically, but their paths are discarded and no corridor or homotopy identifier exists. Rotating, warping or adding noise to the selected route is explicitly rejected.

The executable replacement is one sidecar pipeline: build a clearance-constrained medial-axis corridor graph, enumerate loopless K-shortest edge sequences, lift each sequence to the existing grid planner, then reuse the current shortcut, smooth, curvature and continuous-clearance validation. A state stores expert forward, at most two different corridor classes, and hold. It stores fewer candidates when no true alternative exists.

## Current-pipeline probe

- Episodes / fixed same-state probes: **{current['episodes']} / {probe['states_successful']}**.
- Failed probes: **{probe['states_failed']}**.
- States with more than one unique smoothed geometry: **{probe['states_with_multiple_unique_geometries']} / {probe['states_successful']}**.
- States with a pair at least {probe['geometric_separation_diagnostic_m']:.1f} m apart: **{probe['states_with_geometrically_separated_pair']} / {probe['states_successful']}**.
- Certified multi-topology states: **unknown**, not zero: the current representation cannot certify them.
- Existing samples with alternatives / critic labels: **{current['samples_with_nonempty_alternatives']} / {current['samples_with_critic_labels']}** out of **{current['samples']}**.
- CPU probe time: **{probe['elapsed_seconds']:.1f} s**, {probe['mean_seconds_per_state']:.3f} s/state.

The 0.5 m separation count is diagnostic only. It is not used as a topology label or acceptance threshold.

## Candidate contract

The immutable v2 `sample_id` is the state key, so observation history and final `task_goal` are not duplicated. Candidate paths use the current body frame and metric float32 XY. Candidate count is ragged.

- `expert`: the existing executable forward prefix.
- `topology`: up to two different ordered corridor-edge sequences whose first differing branch occurs inside the policy horizon.
- `hold`: one point at `[0, 0]`, zero progress and no fabricated path.
- `policy`: actual checkpoint proposals captured after an oracle-selector gap; these are the preferred hard negatives.

The exact machine-readable schema and mathematical definitions are in `critic_sidecar_schema_v1.json` and `multitopology_audit_summary.json`. Pairwise critic supervision contains only Pareto-dominant pairs; no arbitrary weighted scalar score is baked into the data.

## Label boundary

Available offline from the current route/navigation-grid contract:

{offline_lines}

Must be measured with the active MPC closed loop:

{closed_loop_lines}

Kinematic curvature speed caps remain proxies and must not be named trackability labels.

## Scale estimate

- Recommended ceiling for the next training corpus: **{scale['recommended_episode_target']} episodes**, approximately **{scale['estimated_samples']} states** and **{scale['estimated_frames']} frames** at pilot density.
- Preserve the pilot split ratio: **{scale['estimated_split']['train']['episodes']} train / {scale['estimated_split']['validation']['episodes']} scene-disjoint validation episodes**, approximately **{scale['estimated_split']['train']['samples']} / {scale['estimated_split']['validation']['samples']} states**.
- Linear physical dataset estimate: **{scale['estimated_dataset_bytes_linear'] / 2**30:.2f} GiB**, of which depth is **{scale['estimated_depth_bytes_linear'] / 2**30:.2f} GiB**.
- Four-candidate, 64-point fixed-shape upper estimate for the uncompressed sidecar: **{scale['estimated_uncompressed_candidate_sidecar_bytes'] / 2**20:.1f} MiB**. Ragged storage should be smaller.
- Measured pilot rendering: **{scale['measured_pilot_render_wall_seconds']:.1f} wall seconds for 200 episodes with two GPU workers**; linear 1000-episode estimate under the same setup is **{scale['estimated_render_wall_seconds_same_two_worker_setup'] / 60:.1f} minutes**.
- Offline current-planner probe extrapolation: **{scale['estimated_offline_candidate_audit_seconds'] / 60:.1f} sequential wall minutes on this host**. This is too slow; the new topology graph must be cached once per scene.
- MPC rollout runtime is intentionally **not estimated** without a measured current-stack per-candidate time. Run a small rollout shard first, then use `states x selected_candidates x measured_seconds`.

Do not render extra depth for alternatives: all candidates share the same observation and task goal. Scaling depth and scaling critic candidates are separate decisions.

## Code evidence

{source_lines}
"""
    path.write_text(report, encoding="utf-8")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset_root", type=Path)
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args(argv)
    summary = audit(args.dataset_root, args.output_dir)
    print(json.dumps(summary["probe"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
