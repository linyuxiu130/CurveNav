"""Generate traced policy hard negatives for an immutable HSSD v2 dataset."""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import os
from pathlib import Path
import shutil
import time
from typing import Any

import numpy as np
import torch

from curvenav.config_io import load_config
from curvenav.data import build_hssd_v2_loader
from curvenav.data_generation import hssd_policy_labels as label_ops
from curvenav.data_generation.audit_hssd_v2_dynamics import (
    EVIDENCE as DYNAMICS_EVIDENCE,
)
from curvenav.data_generation.hssd_policy_sidecar_io import (
    deterministic_candidate_id,
    validate_sidecar,
)
from curvenav.factory import build_policy
from curvenav.training.checkpoint import validate_policy_contract
from curvenav.training.ema import ExponentialMovingAverage
from curvenav.training.prefetch import CudaPrefetchLoader
from curvenav.types import PolicyCondition


SCHEMA_VERSION = "curvenav_critic_sidecar_v1"
FROZEN_SCHEMA_SHA256 = "7462cabc90aeca434763acdfabb9fe047fc269461da1f88ff4620dd39db7673a"
CANDIDATE_COUNT = 8
KIND_POLICY = 0
KIND_EXPERT = 1
KIND_HOLD = 2
KIND_NAMES = {KIND_POLICY: "policy", KIND_EXPERT: "expert", KIND_HOLD: "hold"}
HOLD_PRODUCER_REVISION = "curvenav_hold_v1"
CLOSED_LOOP_FLOAT_FIELDS = (
    "endpoint_error_m",
    "executed_progress_m",
    "rollout_duration_s",
    "tracking_p95_m",
    "tracking_rmse_m",
    "linear_saturation_fraction",
    "angular_saturation_fraction",
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _utf8_array(values: list[str]) -> np.ndarray:
    encoded = [value.encode("utf-8") for value in values]
    width = max(1, *(len(value) for value in encoded))
    return np.asarray(encoded, dtype=f"S{width}")


def _build_state(
    sample_index: int,
    record: dict[str, Any],
    policy_controls: np.ndarray,
    policy_paths: np.ndarray,
    expert_control: np.ndarray,
    expert_path: np.ndarray,
    producer_seed: int,
    policy_revision: str,
    expert_revision: str,
    labeler: label_ops.OfflineLabeler,
) -> dict[str, Any]:
    candidates = []
    for index in range(CANDIDATE_COUNT):
        candidates.append(
            {
                "candidate_id": deterministic_candidate_id(
                    record["sample_id"], "policy", policy_revision, producer_seed, index
                ),
                "kind": KIND_POLICY,
                "policy_sample_index": index,
                "producer_candidate_index": index,
                "producer_revision": policy_revision,
                "producer_seed": producer_seed,
                "control_valid": True,
                "control": policy_controls[index].astype(np.float32),
                "path": policy_paths[index].astype(np.float32),
            }
        )
    candidates.append(
        {
            "candidate_id": deterministic_candidate_id(
                record["sample_id"], "expert", expert_revision, 0, 0
            ),
            "kind": KIND_EXPERT,
            "policy_sample_index": -1,
            "producer_candidate_index": 0,
            "producer_revision": expert_revision,
            "producer_seed": 0,
            "control_valid": True,
            "control": expert_control.astype(np.float32),
            "path": expert_path.astype(np.float32),
        }
    )
    candidates.append(
        {
            "candidate_id": deterministic_candidate_id(
                record["sample_id"], "hold", HOLD_PRODUCER_REVISION, 0, 0
            ),
            "kind": KIND_HOLD,
            "policy_sample_index": -1,
            "producer_candidate_index": 0,
            "producer_revision": HOLD_PRODUCER_REVISION,
            "producer_seed": 0,
            "control_valid": False,
            "control": np.zeros_like(expert_control, dtype=np.float32),
            "path": np.zeros((1, 2), dtype=np.float32),
        }
    )
    candidate_labels = [
        labeler.label(record, candidate["path"]) for candidate in candidates
    ]
    for candidate, label in zip(candidates, candidate_labels, strict=True):
        candidate["labels"] = label
    policy_difference = np.linalg.norm(
        policy_paths[:, None] - policy_paths[None], axis=-1
    ).mean(axis=-1)
    pair_indices = np.triu_indices(CANDIDATE_COUNT, k=1)
    expert_difference = np.linalg.norm(
        policy_paths - expert_path[None], axis=-1
    ).mean(axis=-1)
    return {
        "sample_index": sample_index,
        "sample_id": record["sample_id"],
        "split": record["split"],
        "scene_id": record["scene_id"],
        "episode_id": record["episode_id"],
        "anchor_index": int(record["anchor_index"]),
        "candidates": candidates,
        "preferences": label_ops.pareto_preferences(candidate_labels),
        "policy_pairwise_ade_m": float(policy_difference[pair_indices].mean()),
        "policy_to_expert_ade_m_min": float(expert_difference.min()),
        "policy_to_expert_ade_m_mean": float(expert_difference.mean()),
    }


def _write_scene_shard(
    path: Path, states: list[dict[str, Any]]
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    candidates = [candidate for state in states for candidate in state["candidates"]]
    candidate_lengths = np.asarray(
        [len(state["candidates"]) for state in states], dtype=np.int64
    )
    candidate_offsets = np.concatenate([[0], np.cumsum(candidate_lengths)]).astype(
        np.int64
    )
    points = [candidate["path"] for candidate in candidates]
    for candidate, candidate_path in zip(candidates, points, strict=True):
        if candidate_path.ndim != 2 or candidate_path.shape[1] != 2:
            raise ValueError(f"invalid path shape: {candidate['candidate_id']}")
        if not np.array_equal(candidate_path[0], np.zeros(2, dtype=candidate_path.dtype)):
            raise ValueError(
                "candidate path does not start exactly at origin: "
                f"{candidate['candidate_id']}"
            )
    point_lengths = np.asarray([len(value) for value in points], dtype=np.int64)
    point_offsets = np.concatenate([[0], np.cumsum(point_lengths)]).astype(np.int64)
    preferences = [preference for state in states for preference in state["preferences"]]
    preference_lengths = np.asarray(
        [len(state["preferences"]) for state in states], dtype=np.int64
    )
    preference_offsets = np.concatenate([[0], np.cumsum(preference_lengths)]).astype(
        np.int64
    )

    def label_array(name: str, dtype: Any) -> np.ndarray:
        return np.asarray([candidate["labels"][name] for candidate in candidates], dtype=dtype)

    null_strings = _utf8_array([""] * len(candidates))
    corridor_edge_offsets = np.zeros(len(candidates) + 1, dtype=np.int64)
    closed_loop_valid = np.zeros(len(candidates), dtype=bool)
    closed_loop_null_float = np.full(len(candidates), np.nan, dtype=np.float32)
    np.savez_compressed(
        path,
        state_sample_index=np.asarray([state["sample_index"] for state in states], dtype=np.int64),
        state_sample_id=_utf8_array([state["sample_id"] for state in states]),
        state_episode_id=_utf8_array([state["episode_id"] for state in states]),
        state_anchor_index=np.asarray([state["anchor_index"] for state in states], dtype=np.int32),
        candidate_offsets=candidate_offsets,
        candidate_id=_utf8_array([value["candidate_id"] for value in candidates]),
        candidate_kind=np.asarray([value["kind"] for value in candidates], dtype=np.uint8),
        policy_sample_index=np.asarray(
            [value["policy_sample_index"] for value in candidates], dtype=np.int8
        ),
        producer_candidate_index=np.asarray(
            [value["producer_candidate_index"] for value in candidates], dtype=np.int32
        ),
        producer_revision=_utf8_array([value["producer_revision"] for value in candidates]),
        producer_seed=np.asarray([value["producer_seed"] for value in candidates], dtype=np.uint64),
        topology_class_id=null_strings,
        corridor_edge_offsets=corridor_edge_offsets,
        corridor_edge_ids=np.empty(0, dtype=np.int32),
        first_branch_side=null_strings,
        control_valid=np.asarray([value["control_valid"] for value in candidates], dtype=bool),
        control_points_local_xy_m=np.stack(
            [value["control"] for value in candidates]
        ).astype(np.float32),
        point_offsets=point_offsets,
        path_local_xy_m=np.concatenate(points).astype(np.float32),
        footprint_collision=label_array("footprint_collision", bool),
        minimum_extra_clearance_m=label_array("minimum_extra_clearance_m", np.float32),
        clearance_p05_m=label_array("clearance_p05_m", np.float32),
        safety_margin_violation=label_array("safety_margin_violation", bool),
        endpoint_geodesic_distance_m=label_array("endpoint_geodesic_distance_m", np.float32),
        progress_m=label_array("progress_m", np.float32),
        progress_per_arc=label_array("progress_per_arc", np.float32),
        arc_length_m=label_array("arc_length_m", np.float32),
        curvature_p95_per_m=label_array("curvature_p95_per_m", np.float32),
        maximum_curvature_per_m=label_array("maximum_curvature_per_m", np.float32),
        nominal_peak_angular_rate_radps=label_array(
            "nominal_peak_angular_rate_radps", np.float32
        ),
        kinematic_speed_cap_mps=label_array("kinematic_speed_cap_mps", np.float32),
        geodesic_valid=label_array("geodesic_valid", bool),
        closed_loop_status=_utf8_array(["not_run"] * len(candidates)),
        closed_loop_collision=np.zeros(len(candidates), dtype=bool),
        closed_loop_collision_valid=closed_loop_valid,
        **{
            f"closed_loop_{name}": closed_loop_null_float
            for name in CLOSED_LOOP_FLOAT_FIELDS
        },
        **{
            f"closed_loop_{name}_valid": closed_loop_valid
            for name in CLOSED_LOOP_FLOAT_FIELDS
        },
        pairwise_preference_offsets=preference_offsets,
        pairwise_winner_local_index=np.asarray([value[0] for value in preferences], dtype=np.int32),
        pairwise_loser_local_index=np.asarray([value[1] for value in preferences], dtype=np.int32),
        pairwise_reason_mask=np.asarray([value[2] for value in preferences], dtype=np.uint8),
    )
    shard_record = {
        "states": len(states),
        "candidates": len(candidates),
        "preferences": len(preferences),
        "sha256": _sha256_file(path),
        "bytes": path.stat().st_size,
    }
    state_index = [
        {
            "sample_index": int(state["sample_index"]),
            "sample_id": state["sample_id"],
            "episode_id": state["episode_id"],
            "anchor_index": int(state["anchor_index"]),
            "split": state["split"],
            "scene_id": state["scene_id"],
            "shard": f"scene_shards/{path.name}",
            "local_state_index": row,
            "candidate_begin": int(candidate_offsets[row]),
            "candidate_end": int(candidate_offsets[row + 1]),
            "pairwise_preference_begin": int(preference_offsets[row]),
            "pairwise_preference_end": int(preference_offsets[row + 1]),
        }
        for row, state in enumerate(states)
    ]
    return shard_record, state_index


def _audit(states: list[dict[str, Any]], wall_seconds: float) -> dict[str, Any]:
    candidates = [candidate for state in states for candidate in state["candidates"]]
    policy_labels = [
        candidate["labels"] for candidate in candidates if candidate["kind"] == KIND_POLICY
    ]
    pairs = [preference for state in states for preference in state["preferences"]]
    return {
        "states": len(states),
        "policy_candidates": len(states) * CANDIDATE_COUNT,
        "all_candidates_including_expert_hold": len(candidates),
        "failed_samples": 0,
        "wall_seconds": wall_seconds,
        "wall_scope": (
            "CUDA policy inference plus offline geometry labeling; excludes "
            "loader/model startup and shard compression"
        ),
        "states_per_second": len(states) / wall_seconds,
        "policy_candidates_per_second": len(states) * CANDIDATE_COUNT / wall_seconds,
        "policy_pairwise_ade_m": label_ops.distribution(
            [state["policy_pairwise_ade_m"] for state in states]
        ),
        "policy_to_expert_ade_m_min": label_ops.distribution(
            [state["policy_to_expert_ade_m_min"] for state in states]
        ),
        "policy_to_expert_ade_m_mean": label_ops.distribution(
            [state["policy_to_expert_ade_m_mean"] for state in states]
        ),
        "policy_label_determinable_fraction": float(
            np.mean([label["geodesic_valid"] for label in policy_labels])
        ),
        "policy_collision_fraction": float(
            np.mean([label["footprint_collision"] for label in policy_labels])
        ),
        "policy_safety_margin_violation_fraction": float(
            np.mean([label["safety_margin_violation"] for label in policy_labels])
        ),
        "policy_progress_m": label_ops.distribution(
            [label["progress_m"] for label in policy_labels]
        ),
        "policy_minimum_extra_clearance_m": label_ops.distribution(
            [label["minimum_extra_clearance_m"] for label in policy_labels]
        ),
        "policy_maximum_curvature_per_m": label_ops.distribution(
            [label["maximum_curvature_per_m"] for label in policy_labels]
        ),
        "policy_kinematic_speed_cap_mps": label_ops.distribution(
            [label["kinematic_speed_cap_mps"] for label in policy_labels]
        ),
        "pareto_pairs": len(pairs),
        "states_with_pareto_pair": sum(bool(state["preferences"]) for state in states),
        "pareto_state_coverage": float(np.mean([bool(state["preferences"]) for state in states])),
        "topology_class_non_null": 0,
    }


def _validate_loader_batch(
    batch: dict[str, torch.Tensor], dataset: Any, expected_start: int
) -> None:
    batch_size = int(batch["sample_index"].shape[0])
    expected = torch.arange(
        expected_start,
        expected_start + batch_size,
        device=batch["sample_index"].device,
        dtype=torch.int64,
    )
    if not torch.equal(batch["sample_index"], expected):
        raise ValueError("HSSD loader sample_index is not stable and sequential")
    condition = PolicyCondition(
        depth=batch["depth"],
        task_goal=batch["task_goal"],
        motion_context=batch["motion_context"],
    )
    condition.validate()
    if condition.depth.dtype != torch.float16:
        raise TypeError("packed HSSD depth must reach CUDA as float16")
    if condition.depth.shape[1:] != (4, 1, 168, 224):
        raise ValueError("packed HSSD depth shape does not match format10")
    if not torch.isfinite(condition.depth).all() or not bool(
        ((condition.depth >= 0.0) & (condition.depth <= 1.0)).all()
    ):
        raise ValueError("packed HSSD depth is not finite normalized depth")
    indices = batch["sample_index"].detach().cpu().tolist()
    goals = batch["task_goal"].detach().cpu().numpy()
    anchors = batch["anchor_index"].detach().cpu().numpy()
    for offset, index in enumerate(indices):
        record = dataset.records[index]
        if int(anchors[offset]) != int(record["anchor_index"]):
            raise ValueError(f"anchor_index mismatch: {record['sample_id']}")
        if not np.allclose(goals[offset], record["task_goal_local_xy"], atol=1e-6):
            raise ValueError(f"task_goal mismatch: {record['sample_id']}")
        expected_motion = dataset[index]["motion_context"].numpy()
        actual_motion = batch["motion_context"][offset].detach().cpu().numpy()
        if not np.allclose(actual_motion, expected_motion, atol=1e-6):
            raise ValueError(f"motion_context mismatch: {record['sample_id']}")


def _load_ema_policy(config: Any, checkpoint_path: Path, device: torch.device):
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    validate_policy_contract(checkpoint, config)
    if checkpoint.get("format_version") != 10 or "ema" not in checkpoint:
        raise ValueError("policy sidecar requires a format10 checkpoint with EMA")
    policy = build_policy(config)
    policy.load_state_dict(checkpoint["model"], strict=True)
    ema = ExponentialMovingAverage(policy, decay=config.training.ema_decay)
    ema.load_state_dict(checkpoint["ema"])
    ema.copy_to(policy)
    return policy.eval().to(device), checkpoint


def generate(
    dataset_root: Path,
    checkpoint_path: Path,
    config_path: Path,
    schema_path: Path,
    output_dir: Path,
    *,
    device_name: str,
    batch_size: int,
    num_workers: int,
    seed: int,
    limit: int | None,
) -> dict[str, Any]:
    dataset_root = dataset_root.resolve()
    checkpoint_path = checkpoint_path.resolve()
    schema_path = schema_path.resolve()
    output_dir = output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(f"sidecar output already exists: {output_dir}")
    if not torch.cuda.is_available() or not device_name.startswith("cuda"):
        raise RuntimeError("policy candidate generation requires an available CUDA device")
    config = load_config(config_path.resolve())
    config.validate()
    if _sha256_file(schema_path) != FROZEN_SCHEMA_SHA256:
        raise ValueError("critic sidecar schema does not match the frozen v1 SHA256")
    frozen_schema = json.loads(schema_path.read_text(encoding="utf-8"))
    if frozen_schema.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("critic sidecar schema_version mismatch")
    checkpoint_sha = _sha256_file(checkpoint_path)
    dataset_bundle_sha = _sha256_file(dataset_root / "audit" / "data_files.sha256")
    policy_revision = f"sha256:{checkpoint_sha}"
    expert_revision = f"sha256:{dataset_bundle_sha}"
    cache_manifest = (
        dataset_root
        / f"curvenav_hssd_depth_{config.data.image_height}x{config.data.image_width}_float16"
        / "manifest.json"
    )
    bundle = build_hssd_v2_loader(
        dataset_root,
        config.data,
        config.trajectory,
        batch_size=batch_size,
        num_workers=num_workers,
    )
    requested = len(bundle.dataset) if limit is None else min(limit, len(bundle.dataset))
    device = torch.device(device_name)
    policy, checkpoint = _load_ema_policy(config, checkpoint_path, device)
    loader = CudaPrefetchLoader(bundle.loader, bundle.depth_bank, device)
    labeler = label_ops.OfflineLabeler(dataset_root)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    states: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    started = time.perf_counter()

    with torch.inference_mode():
        for batch in loader:
            if len(states) >= requested:
                break
            remaining = requested - len(states)
            if int(batch["sample_index"].shape[0]) > remaining:
                batch = {name: value[:remaining] for name, value in batch.items()}
            _validate_loader_batch(batch, bundle.dataset, len(states))
            condition = PolicyCondition(
                depth=batch["depth"],
                task_goal=batch["task_goal"],
                motion_context=batch["motion_context"],
            )
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                prediction = policy.sample(condition, num_samples=CANDIDATE_COUNT)
            batch_count = int(condition.task_goal.shape[0])
            controls = prediction.control_points.reshape(
                batch_count, CANDIDATE_COUNT, config.trajectory.num_control_points, 2
            ).float().cpu().numpy()
            paths = prediction.dense_path.reshape(
                batch_count, CANDIDATE_COUNT, config.trajectory.num_path_points, 2
            ).float().cpu().numpy()
            expert_controls = batch["control_points"].float().cpu().numpy()
            expert_paths = batch["canonical_path"].float().cpu().numpy()
            sample_indices = batch["sample_index"].cpu().tolist()
            for offset, sample_index in enumerate(sample_indices):
                record = bundle.dataset.records[sample_index]
                try:
                    states.append(
                        _build_state(
                            sample_index,
                            record,
                            controls[offset],
                            paths[offset],
                            expert_controls[offset],
                            expert_paths[offset],
                            seed,
                            policy_revision,
                            expert_revision,
                            labeler,
                        )
                    )
                except Exception as error:
                    failures.append(
                        {
                            "sample_id": record["sample_id"],
                            "reason": f"{type(error).__name__}: {error}",
                        }
                    )
                    raise
            if len(states) % 1024 == 0 or len(states) == requested:
                elapsed = time.perf_counter() - started
                print(
                    json.dumps(
                        {
                            "progress_states": len(states),
                            "requested_states": requested,
                            "states_per_second": len(states) / elapsed,
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
    torch.cuda.synchronize(device)
    wall_seconds = time.perf_counter() - started
    if len(states) != requested or failures:
        raise RuntimeError(
            "sidecar generation incomplete: "
            f"states={len(states)}/{requested}, failures={len(failures)}"
        )

    temporary = output_dir.with_name(f".{output_dir.name}.tmp-{os.getpid()}")
    if temporary.exists():
        raise FileExistsError(f"temporary sidecar output already exists: {temporary}")
    temporary.mkdir(parents=True)
    try:
        shard_dir = temporary / "scene_shards"
        shard_dir.mkdir()
        by_scene: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
        for state in states:
            by_scene[(state["split"], state["scene_id"])].append(state)
        shard_records = {}
        state_index = []
        for (split, scene_id), scene_states in sorted(by_scene.items()):
            name = f"{split}__{scene_id}.npz"
            shard_record, shard_index = _write_scene_shard(
                shard_dir / name, scene_states
            )
            shard_records[name] = shard_record
            state_index.extend(shard_index)
        state_index.sort(key=lambda record: record["sample_index"])
        (temporary / "state_index.jsonl").write_text(
            "".join(json.dumps(record, sort_keys=True) + "\n" for record in state_index),
            encoding="utf-8",
        )
        shutil.copyfile(schema_path, temporary / "critic_sidecar_schema_v1.json")
        audit = _audit(states, wall_seconds)
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "frozen_schema": {
                "file": "critic_sidecar_schema_v1.json",
                "sha256": FROZEN_SCHEMA_SHA256,
            },
            "state_index": "state_index.jsonl",
            "dataset_root": str(dataset_root),
            "dataset_bundle_sha256": dataset_bundle_sha,
            "packed_depth_manifest": str(cache_manifest),
            "packed_depth_manifest_sha256": _sha256_file(cache_manifest),
            "source_states": len(bundle.dataset),
            "states": len(states),
            "limit": limit,
            "loader": "curvenav.data.build_hssd_v2_loader + CudaPrefetchLoader",
            "metadata_resolution": "bundle.dataset.records[sample_index]",
            "verified_contract": {
                "checkpoint": "format10 + policy contract + EMA state loaded strictly",
                "depth": "float16 packed bank, normalized, finite, shape [B,4,1,168,224]",
                "task_goal": "batch value equals dataset.records[sample_index].task_goal_local_xy",
                "motion_context": (
                    "official HSSD v2 dataset value, checked against every "
                    "indexed sample"
                ),
            },
            "producer": {
                "kind": "policy",
                "checkpoint": str(checkpoint_path),
                "checkpoint_sha256": checkpoint_sha,
                "checkpoint_format_version": int(checkpoint["format_version"]),
                "checkpoint_step": int(checkpoint["step"]),
                "weights": "ema",
                "seed": seed,
                "policy_candidates_per_state": CANDIDATE_COUNT,
                "inference_steps": config.rectified_flow.inference_steps,
            },
            "candidate_order": ["policy/00..07", "expert", "hold"],
            "candidate_kind_codes": {str(key): value for key, value in KIND_NAMES.items()},
            "producer_revisions": {
                "policy": policy_revision,
                "expert": expert_revision,
                "hold": HOLD_PRODUCER_REVISION,
            },
            "candidate_id_contract": {
                "algorithm": "sha256",
                "canonical_payload": (
                    "schema_version\\0sample_id\\0kind\\0producer_revision\\0"
                    "producer_seed_decimal\\0producer_candidate_index_decimal"
                ),
                "stored_value": "sha256:<lowercase hex digest>",
            },
            "nullable_storage": {
                "topology_class_id": "fixed-width UTF-8 bytes; b'' is logical null",
                "first_branch_side": "fixed-width UTF-8 bytes; b'' is logical null",
                "corridor_edge_ids": (
                    "ragged int32 with offsets; equal adjacent offsets encode empty"
                ),
                "closed_loop": (
                    "status=b'not_run'; each nullable value has an explicit "
                    "false validity array"
                ),
            },
            "topology_contract": (
                "policy/expert/hold are unclassified: null topology, empty "
                "corridor edges, null first branch side"
            ),
            "label_contract": {
                "clearance_sample_step_m": label_ops.CLEARANCE_SAMPLE_STEP_M,
                "curvature_sample_step_m": label_ops.CURVATURE_SAMPLE_STEP_M,
                "minimum_extra_clearance_m": (
                    label_ops.MINIMUM_EXTRA_CLEARANCE_M
                ),
                "geodesic": "8-neighbour metric distance over clearance>=0.1 m cells",
                "progress_m": "D(start,goal)-D(endpoint,goal)",
                "kinematic_speed_cap_mps": "min(0.5 m/s, 0.5 rad/s / maximum_curvature_per_m)",
                "nominal_peak_angular_rate_radps": "0.5 m/s * maximum_curvature_per_m",
                "kinematic_constraint_evidence": [
                    item
                    for item in DYNAMICS_EVIDENCE
                    if item["scope"] == "active quick100 benchmark"
                ],
                "preference": "primitive-label Pareto dominance only",
                "preference_reason_bits": {
                    "collision": label_ops.PREFERENCE_COLLISION,
                    "safety_margin": label_ops.PREFERENCE_SAFETY_MARGIN,
                    "progress": label_ops.PREFERENCE_PROGRESS,
                    "clearance": label_ops.PREFERENCE_CLEARANCE,
                },
                "tracking_labels": "not_run",
            },
            "device": device_name,
            "batch_size": batch_size,
            "num_workers": num_workers,
            "shards": shard_records,
        }
        (temporary / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n"
        )
        (temporary / "audit.json").write_text(
            json.dumps(audit, indent=2, sort_keys=True) + "\n"
        )
        (temporary / "failures.jsonl").write_text("")
        hashed_files = sorted(
            path for path in temporary.rglob("*") if path.is_file() and path.name != "SHA256SUMS"
        )
        (temporary / "SHA256SUMS").write_text(
            "".join(
                f"{_sha256_file(path)}  {path.relative_to(temporary).as_posix()}\n"
                for path in hashed_files
            )
        )
        validate_sidecar(
            temporary,
            source_dataset_root=dataset_root,
            verify_hashes=True,
        )
        output_dir.parent.mkdir(parents=True, exist_ok=True)
        os.replace(temporary, output_dir)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    print(json.dumps(audit, indent=2, sort_keys=True))
    return {"manifest": manifest, "audit": audit}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset_root", type=Path)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("config", type=Path)
    parser.add_argument("schema", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args(argv)
    generate(
        args.dataset_root,
        args.checkpoint,
        args.config,
        args.schema,
        args.output_dir,
        device_name=args.device,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=args.seed,
        limit=args.limit,
    )


if __name__ == "__main__":
    main()
