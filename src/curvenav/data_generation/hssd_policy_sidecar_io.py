"""Strict reader and validator for ``curvenav_critic_sidecar_v1``."""

from __future__ import annotations

from collections import defaultdict
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


SCHEMA_VERSION = "curvenav_critic_sidecar_v1"
FROZEN_SCHEMA_SHA256 = "7462cabc90aeca434763acdfabb9fe047fc269461da1f88ff4620dd39db7673a"
KIND_NAMES = {0: "policy", 1: "expert", 2: "hold"}
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


def _decode(value: np.bytes_) -> str:
    return bytes(value).decode("utf-8")


def _nullable_string(value: np.bytes_) -> str | None:
    decoded = _decode(value)
    return decoded or None


def deterministic_candidate_id(
    sample_id: str,
    kind: str,
    producer_revision: str,
    producer_seed: int,
    producer_candidate_index: int,
) -> str:
    if kind not in KIND_NAMES.values():
        raise ValueError(f"unknown candidate kind: {kind}")
    if not producer_revision:
        raise ValueError("producer_revision must be non-empty")
    if not 0 <= producer_seed < 2**64:
        raise ValueError("producer_seed must fit uint64")
    if producer_candidate_index < 0:
        raise ValueError("producer_candidate_index must be non-negative")
    payload = "\0".join(
        (
            SCHEMA_VERSION,
            sample_id,
            kind,
            producer_revision,
            str(producer_seed),
            str(producer_candidate_index),
        )
    ).encode("utf-8")
    return f"sha256:{hashlib.sha256(payload).hexdigest()}"


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]


def _validate_offsets(
    name: str, offsets: np.ndarray, outer_count: int, flat_count: int
) -> None:
    if offsets.dtype != np.int64 or offsets.shape != (outer_count + 1,):
        raise ValueError(f"{name} must be int64 [{outer_count + 1}]")
    if int(offsets[0]) != 0 or np.any(np.diff(offsets) < 0):
        raise ValueError(f"{name} must start at zero and be monotonic")
    if int(offsets[-1]) != flat_count:
        raise ValueError(f"{name} terminal offset does not match flat array")


def _verify_checksums(root: Path) -> None:
    checksum_path = root / "SHA256SUMS"
    entries: dict[str, str] = {}
    for line in checksum_path.read_text(encoding="utf-8").splitlines():
        expected, relative = line.split("  ", 1)
        path = root / relative
        if path.resolve().parent != root.resolve() and root.resolve() not in path.resolve().parents:
            raise ValueError(f"checksum path escapes sidecar root: {relative}")
        if relative in entries:
            raise ValueError(f"duplicate checksum entry: {relative}")
        entries[relative] = expected
        if not path.is_file() or _sha256_file(path) != expected:
            raise ValueError(f"checksum mismatch: {relative}")
    actual = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file() and path.name != "SHA256SUMS"
    }
    if set(entries) != actual:
        raise ValueError("SHA256SUMS does not enumerate the exact sidecar file set")


def validate_sidecar(
    root: Path,
    *,
    source_dataset_root: Path | None = None,
    verify_hashes: bool = True,
) -> dict[str, int]:
    root = root.resolve()
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("sidecar schema_version mismatch")
    schema_record = manifest.get("frozen_schema", {})
    schema_path = root / schema_record.get("file", "")
    if (
        schema_record.get("sha256") != FROZEN_SCHEMA_SHA256
        or not schema_path.is_file()
        or _sha256_file(schema_path) != FROZEN_SCHEMA_SHA256
    ):
        raise ValueError("frozen schema file/hash mismatch")
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    if schema.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("embedded frozen schema version mismatch")
    if verify_hashes:
        _verify_checksums(root)

    index = _read_jsonl(root / manifest["state_index"])
    state_count = int(manifest["states"])
    if len(index) != state_count:
        raise ValueError("state_index.jsonl length does not match manifest")
    if [int(record["sample_index"]) for record in index] != list(range(state_count)):
        raise ValueError("state index must retain stable sequential descriptor indices")
    sample_ids = [record["sample_id"] for record in index]
    if len(set(sample_ids)) != len(sample_ids):
        raise ValueError("state index contains duplicate sample_id foreign keys")

    source_records = None
    if source_dataset_root is not None:
        source_root = Path(source_dataset_root).resolve()
        bundle_manifest = source_root / "audit" / "data_files.sha256"
        if _sha256_file(bundle_manifest) != manifest.get("dataset_bundle_sha256"):
            raise ValueError("source dataset bundle checksum does not match manifest")
        packed_manifest = Path(manifest.get("packed_depth_manifest", "")).resolve()
        if not packed_manifest.is_relative_to(source_root):
            raise ValueError("packed depth manifest is outside source dataset root")
        if (
            not packed_manifest.is_file()
            or _sha256_file(packed_manifest)
            != manifest.get("packed_depth_manifest_sha256")
        ):
            raise ValueError("packed depth manifest checksum does not match manifest")
        source_records = _read_jsonl(source_root / "samples.jsonl")
        for record in index:
            source = source_records[int(record["sample_index"])]
            if (
                source["sample_id"] != record["sample_id"]
                or source["episode_id"] != record["episode_id"]
                or int(source["anchor_index"]) != int(record["anchor_index"])
                or source["split"] != record["split"]
                or source["scene_id"] != record["scene_id"]
            ):
                raise ValueError(f"source foreign-key mismatch: {record['sample_id']}")

    by_shard: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in index:
        by_shard[record["shard"]].append(record)
    expected_shards = {f"scene_shards/{name}" for name in manifest["shards"]}
    if set(by_shard) != expected_shards:
        raise ValueError("state index shard set does not match manifest")

    total_candidates = 0
    total_preferences = 0
    candidate_ids: set[str] = set()
    for relative, references in sorted(by_shard.items()):
        shard_path = root / relative
        shard_manifest = manifest["shards"][shard_path.name]
        if _sha256_file(shard_path) != shard_manifest["sha256"]:
            raise ValueError(f"manifest shard checksum mismatch: {relative}")
        with np.load(shard_path, allow_pickle=False) as archive:
            shard = {name: archive[name] for name in archive.files}
            states = len(shard["state_sample_index"])
            candidates = len(shard["candidate_id"])
            preferences = len(shard["pairwise_winner_local_index"])
            _validate_offsets("candidate_offsets", shard["candidate_offsets"], states, candidates)
            _validate_offsets(
                "point_offsets",
                shard["point_offsets"],
                candidates,
                len(shard["path_local_xy_m"]),
            )
            if np.any(np.diff(shard["point_offsets"]) <= 0):
                raise ValueError("every candidate path must contain at least one point")
            _validate_offsets(
                "corridor_edge_offsets",
                shard["corridor_edge_offsets"],
                candidates,
                len(shard["corridor_edge_ids"]),
            )
            _validate_offsets(
                "pairwise_preference_offsets",
                shard["pairwise_preference_offsets"],
                states,
                preferences,
            )
            if shard["producer_seed"].dtype != np.uint64:
                raise ValueError("producer_seed must use uint64 storage")
            if len(shard["corridor_edge_ids"]) or np.any(shard["corridor_edge_offsets"]):
                raise ValueError("unclassified candidates must have empty corridor edges")
            if any(_decode(value) for value in shard["topology_class_id"]):
                raise ValueError("unclassified topology_class_id must use b'' null encoding")
            if any(_decode(value) for value in shard["first_branch_side"]):
                raise ValueError("unclassified first_branch_side must use b'' null encoding")
            if not np.allclose(
                shard["nominal_peak_angular_rate_radps"],
                0.5 * shard["maximum_curvature_per_m"],
                rtol=1e-6,
                atol=1e-7,
            ):
                raise ValueError("nominal angular-rate proxy does not match frozen definition")
            if any(_decode(value) != "not_run" for value in shard["closed_loop_status"]):
                raise ValueError("closed-loop status must be not_run for this sidecar")
            if np.any(shard["closed_loop_collision_valid"]):
                raise ValueError("closed-loop collision validity must be false")
            if np.any(shard["closed_loop_collision"]):
                raise ValueError("invalid closed-loop collision storage must use false")
            for name in CLOSED_LOOP_FLOAT_FIELDS:
                if np.any(shard[f"closed_loop_{name}_valid"]):
                    raise ValueError(f"closed-loop {name} validity must be false")
                if not np.isnan(shard[f"closed_loop_{name}"]).all():
                    raise ValueError(f"closed-loop {name} null storage must be NaN")

            if [int(record["local_state_index"]) for record in references] != list(range(states)):
                raise ValueError(f"state index local rows are not one-to-one: {relative}")
            for row, reference in enumerate(references):
                if (
                    int(shard["state_sample_index"][row]) != int(reference["sample_index"])
                    or _decode(shard["state_sample_id"][row]) != reference["sample_id"]
                    or _decode(shard["state_episode_id"][row]) != reference["episode_id"]
                    or int(shard["state_anchor_index"][row]) != int(reference["anchor_index"])
                    or int(shard["candidate_offsets"][row]) != int(reference["candidate_begin"])
                    or int(shard["candidate_offsets"][row + 1]) != int(reference["candidate_end"])
                    or int(shard["pairwise_preference_offsets"][row])
                    != int(reference["pairwise_preference_begin"])
                    or int(shard["pairwise_preference_offsets"][row + 1])
                    != int(reference["pairwise_preference_end"])
                ):
                    raise ValueError(f"state index/shard mismatch: {reference['sample_id']}")
                begin, end = int(shard["candidate_offsets"][row]), int(
                    shard["candidate_offsets"][row + 1]
                )
                expected_kinds = [0] * 8 + [1, 2]
                if shard["candidate_kind"][begin:end].tolist() != expected_kinds:
                    raise ValueError(f"candidate order mismatch: {reference['sample_id']}")
                for candidate in range(begin, end):
                    point_begin = int(shard["point_offsets"][candidate])
                    if not np.array_equal(
                        shard["path_local_xy_m"][point_begin],
                        np.zeros(2, dtype=np.float32),
                    ):
                        raise ValueError(
                            "candidate path does not start exactly at origin: "
                            f"{reference['sample_id']}"
                        )
                    kind = KIND_NAMES[int(shard["candidate_kind"][candidate])]
                    revision = _decode(shard["producer_revision"][candidate])
                    seed = int(shard["producer_seed"][candidate])
                    producer_index = int(shard["producer_candidate_index"][candidate])
                    if revision != manifest["producer_revisions"][kind]:
                        raise ValueError(f"producer revision mismatch: {reference['sample_id']}")
                    expected_seed = int(manifest["producer"]["seed"]) if kind == "policy" else 0
                    if seed != expected_seed:
                        raise ValueError(f"producer seed mismatch: {reference['sample_id']}")
                    expected_producer_index = candidate - begin if kind == "policy" else 0
                    if producer_index != expected_producer_index:
                        raise ValueError(
                            f"producer candidate index mismatch: {reference['sample_id']}"
                        )
                    expected_id = deterministic_candidate_id(
                        reference["sample_id"], kind, revision, seed, producer_index
                    )
                    stored_id = _decode(shard["candidate_id"][candidate])
                    if stored_id != expected_id:
                        raise ValueError(f"candidate ID mismatch: {expected_id}")
                    if stored_id in candidate_ids:
                        raise ValueError(f"duplicate candidate ID: {stored_id}")
                    candidate_ids.add(stored_id)
                pair_begin = int(shard["pairwise_preference_offsets"][row])
                pair_end = int(shard["pairwise_preference_offsets"][row + 1])
                candidate_count = end - begin
                winners = shard["pairwise_winner_local_index"][pair_begin:pair_end]
                losers = shard["pairwise_loser_local_index"][pair_begin:pair_end]
                if (
                    np.any(winners < 0)
                    or np.any(winners >= candidate_count)
                    or np.any(losers < 0)
                    or np.any(losers >= candidate_count)
                    or np.any(winners == losers)
                ):
                    raise ValueError(f"invalid pairwise local index: {reference['sample_id']}")
            if states != int(shard_manifest["states"]):
                raise ValueError(f"shard state count mismatch: {relative}")
            if candidates != int(shard_manifest["candidates"]):
                raise ValueError(f"shard candidate count mismatch: {relative}")
            if preferences != int(shard_manifest["preferences"]):
                raise ValueError(f"shard preference count mismatch: {relative}")
            total_candidates += candidates
            total_preferences += preferences
    return {
        "states": state_count,
        "candidates": total_candidates,
        "preferences": total_preferences,
        "source_foreign_keys_checked": state_count if source_records is not None else 0,
    }


class SidecarReader:
    """Resolve a sidecar state by immutable v2 ``sample_id``."""

    def __init__(self, root: Path, *, validate: bool = True) -> None:
        self.root = root.resolve()
        self.manifest = json.loads(
            (self.root / "manifest.json").read_text(encoding="utf-8")
        )
        if validate:
            validate_sidecar(self.root, verify_hashes=True)
        self.records = _read_jsonl(self.root / self.manifest["state_index"])
        self.by_sample_id = {record["sample_id"]: record for record in self.records}

    def __len__(self) -> int:
        return len(self.records)

    def state(self, sample_id: str) -> dict[str, Any]:
        reference = self.by_sample_id[sample_id]
        with np.load(self.root / reference["shard"], allow_pickle=False) as archive:
            shard = {name: archive[name] for name in archive.files}
            row = int(reference["local_state_index"])
            begin = int(shard["candidate_offsets"][row])
            end = int(shard["candidate_offsets"][row + 1])
            candidates = []
            for index in range(begin, end):
                point_begin = int(shard["point_offsets"][index])
                point_end = int(shard["point_offsets"][index + 1])
                corridor_begin = int(shard["corridor_edge_offsets"][index])
                corridor_end = int(shard["corridor_edge_offsets"][index + 1])
                closed_loop = {"status": _decode(shard["closed_loop_status"][index])}
                closed_loop["collision"] = (
                    bool(shard["closed_loop_collision"][index])
                    if bool(shard["closed_loop_collision_valid"][index])
                    else None
                )
                for name in CLOSED_LOOP_FLOAT_FIELDS:
                    closed_loop[name] = (
                        float(shard[f"closed_loop_{name}"][index])
                        if bool(shard[f"closed_loop_{name}_valid"][index])
                        else None
                    )
                candidates.append(
                    {
                        "candidate_id": _decode(shard["candidate_id"][index]),
                        "kind": KIND_NAMES[int(shard["candidate_kind"][index])],
                        "producer_revision": _decode(shard["producer_revision"][index]),
                        "producer_seed": int(shard["producer_seed"][index]),
                        "producer_candidate_index": int(
                            shard["producer_candidate_index"][index]
                        ),
                        "topology_class_id": _nullable_string(
                            shard["topology_class_id"][index]
                        ),
                        "corridor_edge_ids": shard["corridor_edge_ids"][
                            corridor_begin:corridor_end
                        ].copy(),
                        "first_branch_side": _nullable_string(
                            shard["first_branch_side"][index]
                        ),
                        "path_local_xy_m": shard["path_local_xy_m"][
                            point_begin:point_end
                        ].copy(),
                        "control_points_local_xy_m": (
                            shard["control_points_local_xy_m"][index].copy()
                            if bool(shard["control_valid"][index])
                            else None
                        ),
                        "offline_labels": {
                            name: shard[name][index].item()
                            for name in (
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
                                "nominal_peak_angular_rate_radps",
                                "kinematic_speed_cap_mps",
                            )
                        },
                        "closed_loop_labels": closed_loop,
                    }
                )
            pair_begin = int(shard["pairwise_preference_offsets"][row])
            pair_end = int(shard["pairwise_preference_offsets"][row + 1])
            preferences = [
                {
                    "winner_candidate_index": int(
                        shard["pairwise_winner_local_index"][index]
                    ),
                    "loser_candidate_index": int(
                        shard["pairwise_loser_local_index"][index]
                    ),
                    "reason_mask": int(shard["pairwise_reason_mask"][index]),
                }
                for index in range(pair_begin, pair_end)
            ]
        return {**reference, "candidates": candidates, "pairwise_preferences": preferences}
