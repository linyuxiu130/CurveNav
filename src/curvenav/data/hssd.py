"""Deterministic CurveNav v2 HSSD loader with the v2-A policy contract."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

from curvenav.config import DataConfig, TrajectoryConfig
from curvenav.data.depth_bank import PackedDepthBankSpec, PackedDepthRun
from curvenav.data.depth_cache import (
    HSSD_CACHE_SCHEMA_VERSION,
    hssd_depth_cache_root,
)
from curvenav.data.trajectory import collate_metric_paths
from curvenav.trajectory import PlanarBSplineCodec


HSSD_V2_SCHEMA = "curvenav_hssd_v2.0"


def _read_jsonl(path: Path) -> tuple[dict[str, Any], ...]:
    return tuple(json.loads(line) for line in path.read_text().splitlines() if line)


def _local_xy(world_xy: np.ndarray, origin_xy: np.ndarray, yaw: float) -> np.ndarray:
    delta = np.asarray(world_xy, dtype=np.float64) - np.asarray(origin_xy, dtype=np.float64)
    cosine = math.cos(yaw)
    sine = math.sin(yaw)
    return np.stack(
        (
            delta[..., 0] * cosine + delta[..., 1] * sine,
            -delta[..., 0] * sine + delta[..., 1] * cosine,
        ),
        axis=-1,
    ).astype(np.float32, copy=False)


def v2a_motion_context(
    world_xy: np.ndarray,
    yaw_rad: np.ndarray,
    anchor_index: int,
) -> np.ndarray:
    """Return anchor-1→anchor direction expressed in the current body frame."""
    context = np.zeros(3, dtype=np.float32)
    if anchor_index == 0:
        return context
    current = world_xy[anchor_index]
    previous = world_xy[anchor_index - 1]
    local = _local_xy(current, previous, float(yaw_rad[anchor_index]))
    magnitude = float(np.linalg.norm(local))
    if magnitude > 1e-6:
        context[:2] = local / magnitude
        context[2] = 1.0
    return context


@dataclass
class _EpisodeArrays:
    world_xy: np.ndarray
    yaw_rad: np.ndarray


class CurveNavHssdV2Dataset(Dataset):
    """Read immutable v2 sample descriptors without changing their supervision."""

    def __init__(self, root: Path, data: DataConfig, split: str | None = None) -> None:
        self.root = root.expanduser().resolve()
        manifest = json.loads((self.root / "dataset_manifest.json").read_text())
        if manifest.get("schema_version") != HSSD_V2_SCHEMA:
            raise ValueError("HSSD dataset is not CurveNav v2")
        history = manifest.get("history_contract", {})
        expected_offsets = tuple(
            -(data.sequence_length - 1 - index) * (data.frame_skip + 1)
            for index in range(data.sequence_length)
        )
        if history.get("frames") != data.sequence_length or tuple(
            history.get("index_offsets", ())
        ) != expected_offsets:
            raise ValueError("HSSD history contract does not match the policy config")
        camera = manifest.get("camera", {})
        if camera.get("depth", {}).get("dtype") != "float32" or camera.get(
            "depth", {}
        ).get("unit") != "m":
            raise ValueError("HSSD depth must be physical float32 metres")
        all_records = _read_jsonl(self.root / "samples.jsonl")
        selected = tuple(
            (sample_index, record)
            for sample_index, record in enumerate(all_records)
            if split is None or record["split"] == split
        )
        if not selected:
            raise ValueError("HSSD selection contains no samples")
        self.sample_indices = tuple(sample_index for sample_index, _ in selected)
        self.records = tuple(record for _, record in selected)
        self.data = data
        self._episodes: OrderedDict[str, _EpisodeArrays] = OrderedDict()

        cache_root = hssd_depth_cache_root(
            self.root,
            data.image_height,
            data.image_width,
        )
        cache_manifest_path = cache_root / "manifest.json"
        if not cache_manifest_path.is_file():
            raise RuntimeError(
                f"required packed HSSD depth cache is missing: {cache_manifest_path}; "
                "run scripts/prepare_hssd_v2_depth_cache.py first"
            )
        cache_manifest = json.loads(cache_manifest_path.read_text(encoding="utf-8"))
        expected_cache_header = {
            "schema_version": HSSD_CACHE_SCHEMA_VERSION,
            "dataset_schema_version": HSSD_V2_SCHEMA,
            "height": data.image_height,
            "width": data.image_width,
            "dtype": "float16",
            "max_depth_m": data.max_depth_m,
        }
        cache_mismatches = {
            key: (cache_manifest.get(key), value)
            for key, value in expected_cache_header.items()
            if cache_manifest.get(key) != value
        }
        if cache_mismatches:
            raise RuntimeError(
                f"packed HSSD depth cache contract mismatch: {cache_mismatches}"
            )
        manifest_runs = cache_manifest.get("runs")
        if not isinstance(manifest_runs, dict):
            raise RuntimeError("packed HSSD depth cache manifest has no run index")

        episode_records: OrderedDict[str, dict[str, Any]] = OrderedDict()
        for record in self.records:
            episode_records.setdefault(record["episode_id"], record)
        depth_runs = []
        self._depth_offsets: dict[str, int] = {}
        offset = 0
        for episode_id, record in episode_records.items():
            episode_dir = self._episode_dir(record)
            metadata = json.loads((episode_dir / "metadata.json").read_text())
            frames = int(metadata["frames"])
            run_key = episode_dir.relative_to(self.root).as_posix()
            expected_run = {
                "frames": frames,
                "source_depth_sha256": str(metadata["depth_sha256"]),
            }
            if manifest_runs.get(run_key) != expected_run:
                raise RuntimeError(f"packed HSSD depth cache run mismatch: {run_key}")
            cache_path = (
                cache_root
                / episode_dir.relative_to(self.root).parent
                / f"{episode_dir.name}.npy"
            )
            if not cache_path.is_file():
                raise RuntimeError(f"packed HSSD depth cache file is missing: {cache_path}")
            self._depth_offsets[episode_id] = offset
            depth_runs.append(
                PackedDepthRun(path=cache_path, offset=offset, frames=frames)
            )
            offset += frames
        self.depth_bank = PackedDepthBankSpec(
            runs=tuple(depth_runs),
            total_frames=offset,
            height=data.image_height,
            width=data.image_width,
        )

    def __len__(self) -> int:
        return len(self.records)

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        state["_episodes"] = OrderedDict()
        return state

    def _episode_dir(self, record: dict[str, Any]) -> Path:
        run_id = record["episode_id"].rsplit("/", 1)[-1]
        return (
            self.root
            / record["split"]
            / f"dataset_hssd_{record['scene_id']}"
            / run_id
        )

    def _episode(self, record: dict[str, Any]) -> _EpisodeArrays:
        episode_id = record["episode_id"]
        cached = self._episodes.pop(episode_id, None)
        if cached is not None:
            self._episodes[episode_id] = cached
            return cached
        episode_dir = self._episode_dir(record)
        with np.load(episode_dir / "route.npz") as route:
            world_xy = route["world_xy"].astype(np.float32)
            yaw_rad = route["yaw_rad"].astype(np.float32)
        arrays = _EpisodeArrays(
            world_xy=world_xy,
            yaw_rad=yaw_rad,
        )
        self._episodes[episode_id] = arrays
        if len(self._episodes) > 8:
            self._episodes.popitem(last=False)
        return arrays

    def condition_item(self, index: int) -> dict[str, Tensor]:
        """Return only the immutable policy condition and its global foreign key."""
        record = self.records[index]
        episode = self._episode(record)
        anchor = int(record["anchor_index"])
        expected_history = [
            max(
                0,
                anchor
                - (self.data.sequence_length - 1 - offset)
                * (self.data.frame_skip + 1),
            )
            for offset in range(self.data.sequence_length)
        ]
        if record["history_indices"] != expected_history:
            raise ValueError(f"HSSD sample has invalid history indices: {record['sample_id']}")
        return {
            "sample_index": torch.tensor(self.sample_indices[index], dtype=torch.int64),
            "anchor_index": torch.tensor(anchor, dtype=torch.int64),
            "depth_indices": torch.tensor(
                [
                    self._depth_offsets[record["episode_id"]] + frame
                    for frame in expected_history
                ],
                dtype=torch.int64,
            ),
            "task_goal": torch.tensor(record["task_goal_local_xy"], dtype=torch.float32),
            "motion_context": torch.from_numpy(
                v2a_motion_context(episode.world_xy, episode.yaw_rad, anchor)
            ),
        }

    def __getitem__(self, index: int) -> dict[str, Tensor]:
        item = self.condition_item(index)
        record = self.records[index]
        episode = self._episode(record)
        anchor = int(record["anchor_index"])
        target_end = int(record["target_end_index"])
        metric_path = _local_xy(
            episode.world_xy[anchor : target_end + 1],
            episode.world_xy[anchor],
            float(episode.yaw_rad[anchor]),
        )
        return {**item, "metric_path": torch.from_numpy(metric_path)}


@dataclass
class _HssdV2Collator:
    codec: PlanarBSplineCodec

    def __call__(self, batch: list[dict[str, Tensor]]) -> dict[str, Tensor]:
        canonical, controls = collate_metric_paths(
            [item["metric_path"] for item in batch],
            self.codec,
        )
        return {
            "sample_index": torch.stack([item["sample_index"] for item in batch]),
            "anchor_index": torch.stack([item["anchor_index"] for item in batch]),
            "depth_indices": torch.stack([item["depth_indices"] for item in batch]),
            "task_goal": torch.stack([item["task_goal"] for item in batch]),
            "motion_context": torch.stack([item["motion_context"] for item in batch]),
            "canonical_path": canonical,
            "control_points": controls,
        }


@dataclass(frozen=True)
class HssdLoaderBundle:
    loader: DataLoader
    depth_bank: PackedDepthBankSpec
    dataset: CurveNavHssdV2Dataset


def build_hssd_v2_loader(
    root: Path,
    data: DataConfig,
    trajectory: TrajectoryConfig,
    *,
    batch_size: int,
    num_workers: int,
    split: str | None = None,
) -> HssdLoaderBundle:
    dataset = CurveNavHssdV2Dataset(root, data, split)
    codec = PlanarBSplineCodec(
        num_control_points=trajectory.num_control_points,
        degree=trajectory.degree,
        num_path_points=trajectory.num_path_points,
    )
    arguments: dict[str, Any] = {
        "dataset": dataset,
        "batch_size": batch_size,
        "shuffle": False,
        "num_workers": num_workers,
        "pin_memory": True,
        "drop_last": False,
        "collate_fn": _HssdV2Collator(codec),
    }
    if num_workers:
        arguments.update(persistent_workers=True, prefetch_factor=2)
    loader = DataLoader(**arguments)
    return HssdLoaderBundle(
        loader=loader,
        depth_bank=dataset.depth_bank,
        dataset=dataset,
    )
