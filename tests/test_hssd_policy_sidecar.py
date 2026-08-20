from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil

import numpy as np

from curvenav.data_generation.hssd_policy_sidecar import (
    HOLD_PRODUCER_REVISION,
    KIND_EXPERT,
    KIND_HOLD,
    KIND_POLICY,
    _write_scene_shard,
)
from curvenav.data_generation.hssd_policy_labels import (
    OfflineLabeler,
    grid_geodesic_distance,
    pareto_preferences,
)
from curvenav.data_generation.hssd_policy_sidecar_io import (
    SidecarReader,
    deterministic_candidate_id,
    validate_sidecar,
)
from curvenav.data_generation.occupancy import NavigationGrid


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_geometry_labels_and_pareto(tmp_path: Path) -> None:
    free = np.ones((21, 21), dtype=bool)
    free[[0, -1], :] = False
    free[:, [0, -1]] = False
    clearance = np.where(free, 1.0, 0.0).astype(np.float32)
    grid = NavigationGrid(free, clearance, np.zeros(2), 0.1)
    distance = grid_geodesic_distance(grid, np.array([1.55, 1.05]))
    assert np.isfinite(distance[10, 10])

    better = {
        "geodesic_valid": True,
        "footprint_collision": False,
        "safety_margin_violation": False,
        "progress_m": 1.0,
        "minimum_extra_clearance_m": 0.4,
    }
    worse = {
        "geodesic_valid": True,
        "footprint_collision": True,
        "safety_margin_violation": True,
        "progress_m": 0.5,
        "minimum_extra_clearance_m": 0.0,
    }
    tradeoff = {
        "geodesic_valid": True,
        "footprint_collision": False,
        "safety_margin_violation": False,
        "progress_m": 1.5,
        "minimum_extra_clearance_m": 0.2,
    }
    pairs = pareto_preferences([better, worse, tradeoff])
    assert (0, 1, 15) in pairs
    assert not any({winner, loser} == {0, 2} for winner, loser, _ in pairs)

    episode_dir = tmp_path / "train" / "dataset_hssd_unit" / "run_0000"
    episode_dir.mkdir(parents=True)
    np.savez(
        episode_dir.parent / "navigation_grid.npz",
        free=free,
        clearance_m=clearance,
        origin_xy=np.zeros(2),
        cell_size_m=np.array(0.1),
    )
    world_xy = np.array([[0.55, 1.05], [1.55, 1.05]], dtype=np.float32)
    np.savez(
        episode_dir / "route.npz",
        world_xy=world_xy,
        yaw_rad=np.zeros(2, dtype=np.float32),
        task_goal_world_xy=world_xy[-1],
    )
    record = {
        "sample_id": "hssd_v2/train/unit/run_0000:0000",
        "episode_id": "hssd_v2/train/unit/run_0000",
        "split": "train",
        "scene_id": "unit",
        "anchor_index": 0,
    }
    label = OfflineLabeler(tmp_path).label(
        record, np.array([[0.0, 0.0], [0.5, 0.0]], dtype=np.float32)
    )
    assert not label["footprint_collision"]
    assert label["minimum_extra_clearance_m"] == 1.0
    assert label["geodesic_valid"]
    assert 0.49 <= label["progress_m"] <= 0.51
    assert label["nominal_peak_angular_rate_radps"] == 0.0


def test_reader_and_validator_lock_v1_storage_contract(tmp_path: Path) -> None:
    sample_id = "hssd_v2/train/unit/run_0000:0000"
    policy_revision = f"sha256:{'1' * 64}"
    expert_revision = f"sha256:{'2' * 64}"
    labels = {
        "footprint_collision": False,
        "minimum_extra_clearance_m": 0.5,
        "clearance_p05_m": 0.5,
        "safety_margin_violation": False,
        "endpoint_geodesic_distance_m": 1.0,
        "progress_m": 0.5,
        "progress_per_arc": 1.0,
        "arc_length_m": 0.5,
        "curvature_p95_per_m": 0.0,
        "maximum_curvature_per_m": 0.0,
        "nominal_peak_angular_rate_radps": 0.0,
        "kinematic_speed_cap_mps": 0.5,
        "geodesic_valid": True,
    }
    candidates = []
    for index in range(8):
        candidates.append(
            {
                "candidate_id": deterministic_candidate_id(
                    sample_id, "policy", policy_revision, 42, index
                ),
                "kind": KIND_POLICY,
                "policy_sample_index": index,
                "producer_candidate_index": index,
                "producer_revision": policy_revision,
                "producer_seed": 42,
                "control_valid": True,
                "control": np.zeros((12, 2), dtype=np.float32),
                "path": np.array([[0.0, 0.0], [0.5, 0.0]], dtype=np.float32),
                "labels": labels,
            }
        )
    for kind_code, kind, revision in (
        (KIND_EXPERT, "expert", expert_revision),
        (KIND_HOLD, "hold", HOLD_PRODUCER_REVISION),
    ):
        candidates.append(
            {
                "candidate_id": deterministic_candidate_id(
                    sample_id, kind, revision, 0, 0
                ),
                "kind": kind_code,
                "policy_sample_index": -1,
                "producer_candidate_index": 0,
                "producer_revision": revision,
                "producer_seed": 0,
                "control_valid": kind == "expert",
                "control": np.zeros((12, 2), dtype=np.float32),
                "path": (
                    np.array([[0.0, 0.0], [0.5, 0.0]], dtype=np.float32)
                    if kind == "expert"
                    else np.zeros((1, 2), dtype=np.float32)
                ),
                "labels": labels,
            }
        )
    state = {
        "sample_index": 0,
        "sample_id": sample_id,
        "episode_id": "hssd_v2/train/unit/run_0000",
        "anchor_index": 0,
        "split": "train",
        "scene_id": "unit",
        "candidates": candidates,
        "preferences": [(0, 8, 4)],
    }

    root = tmp_path / "sidecar"
    shard_dir = root / "scene_shards"
    shard_dir.mkdir(parents=True)
    shard_record, state_index = _write_scene_shard(
        shard_dir / "train__unit.npz", [state]
    )
    source_root = tmp_path / "source"
    source_root.mkdir()
    source_record = {
        "sample_id": sample_id,
        "episode_id": state["episode_id"],
        "anchor_index": 0,
        "split": "train",
        "scene_id": "unit",
    }
    (source_root / "samples.jsonl").write_text(
        json.dumps(source_record) + "\n", encoding="utf-8"
    )
    (source_root / "audit").mkdir()
    (source_root / "audit/data_files.sha256").write_text(
        "immutable source bundle\n", encoding="utf-8"
    )
    packed_manifest = source_root / "packed_depth/manifest.json"
    packed_manifest.parent.mkdir()
    packed_manifest.write_text('{"format":"float16"}\n', encoding="utf-8")
    (root / "state_index.jsonl").write_text(
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in state_index),
        encoding="utf-8",
    )
    frozen = (
        Path(__file__).parents[1]
        / "outputs/audits/hssd_v2_multitopology_design_20260819/critic_sidecar_schema_v1.json"
    )
    shutil.copyfile(frozen, root / "critic_sidecar_schema_v1.json")
    manifest = {
        "schema_version": "curvenav_critic_sidecar_v1",
        "frozen_schema": {
            "file": "critic_sidecar_schema_v1.json",
            "sha256": _sha256(root / "critic_sidecar_schema_v1.json"),
        },
        "state_index": "state_index.jsonl",
        "states": 1,
        "dataset_bundle_sha256": _sha256(source_root / "audit/data_files.sha256"),
        "packed_depth_manifest": str(packed_manifest),
        "packed_depth_manifest_sha256": _sha256(packed_manifest),
        "producer": {"seed": 42},
        "producer_revisions": {
            "policy": policy_revision,
            "expert": expert_revision,
            "hold": HOLD_PRODUCER_REVISION,
        },
        "shards": {"train__unit.npz": shard_record},
    }
    (root / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    files = sorted(path for path in root.rglob("*") if path.is_file())
    (root / "SHA256SUMS").write_text(
        "".join(
            f"{_sha256(path)}  {path.relative_to(root).as_posix()}\n" for path in files
        ),
        encoding="utf-8",
    )

    report = validate_sidecar(root, source_dataset_root=source_root)
    assert report == {
        "states": 1,
        "candidates": 10,
        "preferences": 1,
        "source_foreign_keys_checked": 1,
    }
    with np.load(shard_dir / "train__unit.npz", allow_pickle=False) as shard:
        assert shard["producer_seed"].dtype == np.uint64
        assert np.all(shard["topology_class_id"] == b"")
        assert np.all(shard["first_branch_side"] == b"")
        assert len(shard["corridor_edge_ids"]) == 0
        assert np.all(np.diff(shard["candidate_offsets"]) >= 0)
        assert int(shard["point_offsets"][-1]) == len(shard["path_local_xy_m"])
    loaded = SidecarReader(root).state(sample_id)
    assert len(loaded["candidates"]) == 10
    assert loaded["candidates"][0]["producer_seed"] == 42
    assert loaded["candidates"][8]["producer_seed"] == 0
    assert loaded["candidates"][9]["topology_class_id"] is None
    assert loaded["candidates"][9]["first_branch_side"] is None
    assert loaded["candidates"][9]["closed_loop_labels"]["status"] == "not_run"
    assert loaded["candidates"][9]["closed_loop_labels"]["collision"] is None
    assert loaded["candidates"][9]["closed_loop_labels"]["tracking_rmse_m"] is None


def test_candidate_id_includes_revision_seed_and_slot() -> None:
    baseline = deterministic_candidate_id("sample", "policy", "revision-a", 42, 0)
    assert baseline == deterministic_candidate_id(
        "sample", "policy", "revision-a", 42, 0
    )
    assert baseline != deterministic_candidate_id(
        "sample", "policy", "revision-b", 42, 0
    )
    assert baseline != deterministic_candidate_id(
        "sample", "policy", "revision-a", 43, 0
    )
    assert baseline != deterministic_candidate_id(
        "sample", "policy", "revision-a", 42, 1
    )
