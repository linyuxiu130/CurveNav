"""In-memory hard-negative labels joined to immutable HSSD conditions."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

from curvenav.config import DataConfig
from curvenav.data.depth_bank import PackedDepthBankSpec
from curvenav.data.hssd import CurveNavHssdV2Dataset
from curvenav.data.trajectory import collate_metric_paths
from curvenav.data_generation.hssd_policy_sidecar_io import validate_sidecar
from curvenav.trajectory import PlanarBSplineCodec


POLICY_CANDIDATES = 8
TOTAL_CANDIDATES = 10


@dataclass(frozen=True)
class CriticSidecarTable:
    controls_m: Tensor
    candidate_kind: Tensor
    candidate_valid: Tensor
    preference: Tensor
    collision: Tensor
    margin_violation: Tensor
    progress_m: Tensor
    progress_valid: Tensor
    clearance_m: Tensor

    def __len__(self) -> int:
        return self.controls_m.shape[0]


def load_critic_sidecar_table(
    root: Path,
    codec: PlanarBSplineCodec,
    *,
    validate: bool = True,
) -> CriticSidecarTable:
    """Load every shard once and materialize a table keyed by global sample index."""
    root = root.expanduser().resolve()
    if validate:
        validate_sidecar(root, verify_hashes=True)
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    state_count = int(manifest["states"])
    controls = np.empty((state_count, TOTAL_CANDIDATES, 12, 2), dtype=np.float32)
    kinds = np.empty((state_count, TOTAL_CANDIDATES), dtype=np.uint8)
    candidate_valid = np.empty((state_count, TOTAL_CANDIDATES), dtype=np.bool_)
    preference = np.zeros(
        (state_count, TOTAL_CANDIDATES, TOTAL_CANDIDATES), dtype=np.bool_
    )
    collision = np.empty((state_count, TOTAL_CANDIDATES), dtype=np.bool_)
    margin = np.empty((state_count, TOTAL_CANDIDATES), dtype=np.bool_)
    progress = np.empty((state_count, TOTAL_CANDIDATES), dtype=np.float32)
    progress_valid = np.empty((state_count, TOTAL_CANDIDATES), dtype=np.bool_)
    clearance = np.empty((state_count, TOTAL_CANDIDATES), dtype=np.float32)
    seen = np.zeros(state_count, dtype=np.bool_)

    for shard_name in sorted(manifest["shards"]):
        with np.load(root / "scene_shards" / shard_name, allow_pickle=False) as archive:
            shard = {name: archive[name] for name in archive.files}
        shard_controls = shard["control_points_local_xy_m"].copy()
        missing_controls = np.flatnonzero(~shard["control_valid"])
        if len(missing_controls):
            fit_indices = []
            paths = []
            for candidate in missing_controls:
                begin = int(shard["point_offsets"][candidate])
                end = int(shard["point_offsets"][candidate + 1])
                path = shard["path_local_xy_m"][begin:end]
                if len(path) == 1:
                    if not np.array_equal(path[0], np.zeros(2, dtype=np.float32)):
                        raise ValueError("single-point hold candidate must be the origin")
                    shard_controls[candidate] = 0.0
                else:
                    fit_indices.append(candidate)
                    paths.append(torch.from_numpy(path))
            if paths:
                _, fitted = collate_metric_paths(paths, codec)
                shard_controls[fit_indices] = fitted.numpy()
        for row, sample_index_value in enumerate(shard["state_sample_index"]):
            sample_index = int(sample_index_value)
            if not 0 <= sample_index < state_count or seen[sample_index]:
                raise ValueError(f"invalid or duplicate critic sample index: {sample_index}")
            begin = int(shard["candidate_offsets"][row])
            end = int(shard["candidate_offsets"][row + 1])
            if end - begin != TOTAL_CANDIDATES:
                raise ValueError("critic sidecar must contain exactly ten ordered candidates")
            controls[sample_index] = shard_controls[begin:end]
            kinds[sample_index] = shard["candidate_kind"][begin:end]
            candidate_valid[sample_index] = True
            collision[sample_index] = shard["footprint_collision"][begin:end]
            margin[sample_index] = shard["safety_margin_violation"][begin:end]
            progress[sample_index] = np.nan_to_num(
                shard["progress_m"][begin:end], nan=0.0
            )
            progress_valid[sample_index] = shard["geodesic_valid"][begin:end]
            clearance[sample_index] = np.nan_to_num(
                shard["minimum_extra_clearance_m"][begin:end], nan=0.0
            )
            pair_begin = int(shard["pairwise_preference_offsets"][row])
            pair_end = int(shard["pairwise_preference_offsets"][row + 1])
            winners = shard["pairwise_winner_local_index"][pair_begin:pair_end]
            losers = shard["pairwise_loser_local_index"][pair_begin:pair_end]
            preference[sample_index, winners, losers] = True
            seen[sample_index] = True

    if not np.all(seen):
        missing = np.flatnonzero(~seen)
        raise ValueError(f"critic sidecar is missing sample indices: {missing[:8].tolist()}")
    expected_kinds = np.array([0] * POLICY_CANDIDATES + [1, 2], dtype=np.uint8)
    if not np.all(kinds == expected_kinds):
        raise ValueError("critic candidate order is not policy×8, expert, hold")
    return CriticSidecarTable(
        controls_m=torch.from_numpy(controls),
        candidate_kind=torch.from_numpy(kinds),
        candidate_valid=torch.from_numpy(candidate_valid),
        preference=torch.from_numpy(preference),
        collision=torch.from_numpy(collision),
        margin_violation=torch.from_numpy(margin),
        progress_m=torch.from_numpy(progress),
        progress_valid=torch.from_numpy(progress_valid),
        clearance_m=torch.from_numpy(clearance),
    )


class CriticHssdDataset(Dataset):
    """Join conditions to sidecar labels without loading or fitting expert paths."""

    def __init__(
        self,
        conditions: CurveNavHssdV2Dataset,
        table: CriticSidecarTable,
    ) -> None:
        self.conditions = conditions
        self.table = table
        if max(conditions.sample_indices) >= len(table):
            raise ValueError("HSSD sample index is outside the critic sidecar table")

    def __len__(self) -> int:
        return len(self.conditions)

    def __getitem__(self, index: int) -> dict[str, Tensor]:
        condition = self.conditions.condition_item(index)
        sample_index = int(condition["sample_index"])
        return {
            **condition,
            "controls_m": self.table.controls_m[sample_index],
            "candidate_valid": self.table.candidate_valid[sample_index],
            "preference": self.table.preference[sample_index],
            "collision": self.table.collision[sample_index],
            "margin_violation": self.table.margin_violation[sample_index],
            "progress_m": self.table.progress_m[sample_index],
            "progress_valid": self.table.progress_valid[sample_index],
            "clearance_m": self.table.clearance_m[sample_index],
        }


class CriticConditionDataset(Dataset):
    """Expose immutable policy conditions without fitting or loading target paths."""

    def __init__(self, conditions: CurveNavHssdV2Dataset) -> None:
        self.conditions = conditions

    def __len__(self) -> int:
        return len(self.conditions)

    def __getitem__(self, index: int) -> dict[str, Tensor]:
        return self.conditions.condition_item(index)


class CriticLabelDataset(Dataset):
    """Expose sidecar tensors only; frozen condition tokens are cached by sample ID."""

    def __init__(
        self,
        sample_indices: tuple[int, ...],
        table: CriticSidecarTable,
    ) -> None:
        if not sample_indices:
            raise ValueError("critic label dataset requires at least one sample")
        if min(sample_indices) < 0 or max(sample_indices) >= len(table):
            raise ValueError("critic label sample index is outside the sidecar table")
        self.sample_indices = sample_indices
        self.table = table

    def __len__(self) -> int:
        return len(self.sample_indices)

    def __getitem__(self, index: int) -> dict[str, Tensor]:
        sample_index = self.sample_indices[index]
        return {
            "sample_index": torch.tensor(sample_index, dtype=torch.int64),
            "controls_m": self.table.controls_m[sample_index],
            "candidate_valid": self.table.candidate_valid[sample_index],
            "preference": self.table.preference[sample_index],
            "collision": self.table.collision[sample_index],
            "margin_violation": self.table.margin_violation[sample_index],
            "progress_m": self.table.progress_m[sample_index],
            "progress_valid": self.table.progress_valid[sample_index],
            "clearance_m": self.table.clearance_m[sample_index],
        }


@dataclass(frozen=True)
class CriticLoaderBundle:
    loader: DataLoader
    depth_bank: PackedDepthBankSpec
    dataset: CriticHssdDataset


@dataclass(frozen=True)
class CriticConditionLoaderBundle:
    loader: DataLoader
    depth_bank: PackedDepthBankSpec
    dataset: CriticConditionDataset


def build_critic_loader(
    dataset_root: Path,
    data: DataConfig,
    table: CriticSidecarTable,
    *,
    split: str,
    batch_size: int,
    num_workers: int,
    shuffle: bool,
    seed: int,
) -> CriticLoaderBundle:
    conditions = CurveNavHssdV2Dataset(dataset_root, data, split=split)
    dataset = CriticHssdDataset(conditions, table)
    generator = torch.Generator().manual_seed(seed)
    arguments: dict[str, Any] = {
        "dataset": dataset,
        "batch_size": batch_size,
        "shuffle": shuffle,
        "generator": generator,
        "num_workers": num_workers,
        "pin_memory": True,
        "drop_last": False,
    }
    if num_workers:
        arguments.update(persistent_workers=True, prefetch_factor=2)
    return CriticLoaderBundle(
        loader=DataLoader(**arguments),
        depth_bank=conditions.depth_bank,
        dataset=dataset,
    )


def build_critic_condition_loader(
    conditions: CurveNavHssdV2Dataset,
    *,
    batch_size: int,
    num_workers: int,
) -> CriticConditionLoaderBundle:
    """Build the one-pass loader used to cache frozen condition memory."""
    dataset = CriticConditionDataset(conditions)
    arguments: dict[str, Any] = {
        "dataset": dataset,
        "batch_size": batch_size,
        "shuffle": False,
        "num_workers": num_workers,
        "pin_memory": True,
        "drop_last": False,
    }
    if num_workers:
        arguments.update(prefetch_factor=2)
    return CriticConditionLoaderBundle(
        loader=DataLoader(**arguments),
        depth_bank=conditions.depth_bank,
        dataset=dataset,
    )


def build_critic_label_loader(
    conditions: CurveNavHssdV2Dataset,
    table: CriticSidecarTable,
    *,
    batch_size: int,
    num_workers: int,
    shuffle: bool,
    seed: int,
) -> DataLoader:
    """Build the repeated critic loader without observation I/O or target fitting."""
    dataset = CriticLabelDataset(conditions.sample_indices, table)
    generator = torch.Generator().manual_seed(seed)
    arguments: dict[str, Any] = {
        "dataset": dataset,
        "batch_size": batch_size,
        "shuffle": shuffle,
        "generator": generator,
        "num_workers": num_workers,
        "pin_memory": True,
        "drop_last": False,
    }
    if num_workers:
        arguments.update(persistent_workers=True, prefetch_factor=2)
    return DataLoader(**arguments)
