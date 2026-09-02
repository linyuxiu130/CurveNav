"""Immutable, source-agnostic CurveNav policy dataset."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset

from curvenav.config import DataConfig, TrajectoryConfig
from curvenav.data.contracts import expert_navigation_geometry_contract
from curvenav.data.depth import depth_camera_contract
from curvenav.data.depth_bank import PackedDepthBankSpec, PackedDepthRun
from curvenav.data.trajectory import MAXIMUM_EXPERT_PROJECTION_ADE_RATIO
from curvenav.data.privileged import (
    SOURCE_CONFIGURATION_PATH_SAMPLING,
    SOURCE_CONFIGURATION_QUERY_SPACING_M,
    SOURCE_CONFIGURATION_QUERY_TYPE,
)
from curvenav.physical import EXTRA_CLEARANCE_M


POLICY_ARRAYS = {
    "depth_indices": ("uint32", 2),
    "point_goal": ("float32", 2),
    "observation_to_current": ("float32", 3),
    "observation_valid": ("bool", 2),
    "curve_values": ("float32", 2),
    "source_grid_index": ("int64", 1),
    "source_origin_xy": ("float32", 2),
    "source_yaw_rad": ("float32", 1),
}


def flow_coordinate_statistics(trajectory: TrajectoryConfig) -> dict[str, object]:
    """Return the train-split Euclidean coordinate normalization contract."""
    return {
        "control_increment_mean_xy_m": list(
            trajectory.control_increment_mean_xy_m
        ),
        "control_increment_std_xy_m": list(
            trajectory.control_increment_std_xy_m
        ),
    }


def policy_dataset_contract(
    data: DataConfig,
    trajectory: TrajectoryConfig,
) -> dict[str, Any]:
    """Return the model-facing fields every prepared dataset must satisfy."""
    return {
        "expert_navigation_geometry": expert_navigation_geometry_contract(),
        "observation_frames": data.observation_frames,
        "frame_spacing_m": data.frame_spacing_m,
        "expert_waypoint_spacing_m": data.expert_waypoint_spacing_m,
        "future_steps": data.future_steps,
        "planar_axis_convention": "x_forward_y_left",
        "observation_to_current_semantics": (
            "planar_rigid_transform_from_observation_to_current_frame"
        ),
        **depth_camera_contract(data),
        "num_curve_values": 2 * (trajectory.num_control_points - 1),
        "num_path_points": trajectory.num_path_points,
        "curve_value_semantics": ("seven_planar_cubic_bspline_control_points_xy_m"),
        "flow_coordinate_statistics": flow_coordinate_statistics(trajectory),
        "expert_projection": "equal_arc_planar_bspline_least_squares",
        "maximum_expert_projection_ade_m": (
            data.expert_waypoint_spacing_m * MAXIMUM_EXPERT_PROJECTION_ADE_RATIO
        ),
        "production_curve_minimum_source_clearance_m": EXTRA_CLEARANCE_M,
        "trajectory_geometry_coupling": (
            "source_safe_expert_physical_increment_imitation"
        ),
        "source_configuration_space_truth": (
            "native_navigation_grid_endpoint_inclusive_dense_0.025m_"
            "oob_non_executable"
        ),
    }


def read_policy_manifest(root: str | Path) -> dict[str, Any]:
    path = Path(root).expanduser().resolve() / "manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    return manifest


class PreparedPolicyDataset(Dataset):
    """Memory-map one compiled split with no source-specific runtime logic."""

    def __init__(
        self,
        root: str | Path,
        split: str,
        data: DataConfig,
        trajectory: TrajectoryConfig,
    ) -> None:
        self.root = Path(root).expanduser().resolve()
        self.split = split
        manifest = read_policy_manifest(self.root)
        expected_contract = policy_dataset_contract(data, trajectory)
        contract = manifest.get("contract", {})
        mismatches = {
            key: (contract.get(key), value)
            for key, value in expected_contract.items()
            if contract.get(key) != value
        }
        if mismatches:
            raise ValueError(f"policy dataset contract mismatch: {mismatches}")

        split_root = self.root / split
        split_manifest = json.loads(
            (split_root / "manifest.json").read_text(encoding="utf-8")
        )
        self.count = int(split_manifest["samples"])
        if self.count < 1:
            raise ValueError(f"policy dataset split is empty: {split}")
        self.arrays: dict[str, np.ndarray] = {}
        array_manifest = split_manifest.get("arrays", {})
        for name, (dtype, rank) in POLICY_ARRAYS.items():
            metadata = array_manifest.get(name, {})
            path = split_root / str(metadata.get("file", ""))
            array = np.load(path, mmap_mode="r")
            if (
                str(array.dtype) != dtype
                or array.ndim != rank
                or len(array) != self.count
            ):
                raise ValueError(
                    f"invalid prepared array {name}: dtype={array.dtype}, shape={array.shape}"
                )
            if list(array.shape) != metadata.get("shape"):
                raise ValueError(f"prepared array manifest mismatch: {name}")
            self.arrays[name] = array
        expected_shapes = {
            "depth_indices": (self.count, data.observation_frames),
            "point_goal": (self.count, 2),
            "observation_to_current": (self.count, data.observation_frames, 4),
            "observation_valid": (self.count, data.observation_frames),
            "curve_values": (
                self.count,
                2 * (trajectory.num_control_points - 1),
            ),
            "source_grid_index": (self.count,),
            "source_origin_xy": (self.count, 2),
            "source_yaw_rad": (self.count,),
        }
        invalid_shapes = {
            name: (tuple(self.arrays[name].shape), shape)
            for name, shape in expected_shapes.items()
            if tuple(self.arrays[name].shape) != shape
        }
        if invalid_shapes:
            raise ValueError(f"prepared policy tensor shape mismatch: {invalid_shapes}")
        finite_arrays = (
            "point_goal",
            "observation_to_current",
            "curve_values",
            "source_origin_xy",
            "source_yaw_rad",
        )
        if any(not np.isfinite(self.arrays[name]).all() for name in finite_arrays):
            raise ValueError(
                f"prepared policy split contains non-finite values: {split}"
            )
        if not self.arrays["observation_valid"][:, -1].all():
            raise ValueError(
                f"prepared policy split has an invalid current frame: {split}"
            )
        observation_valid = self.arrays["observation_valid"]
        if np.any(observation_valid[:, :-1] & ~observation_valid[:, 1:]):
            raise ValueError(
                f"prepared policy split history is not a valid suffix: {split}"
            )

        source = split_manifest.get("source_configuration_space", {})
        grids = source.get("grids", ())
        if (
            source.get("query") != SOURCE_CONFIGURATION_QUERY_TYPE
            or source.get("spacing_m") != SOURCE_CONFIGURATION_QUERY_SPACING_M
            or source.get("path_sampling") != SOURCE_CONFIGURATION_PATH_SAMPLING
            or source.get("out_of_bounds") != "non_executable_negative_clearance"
            or not grids
        ):
            raise ValueError(
                f"prepared source configuration contract mismatch: {split}"
            )
        if not all(
            (split_root / str(item.get("file", ""))).is_file() for item in grids
        ):
            raise ValueError(f"prepared source configuration grid is missing: {split}")
        grid_index = self.arrays["source_grid_index"]
        if np.any(grid_index < 0) or np.any(grid_index >= len(grids)):
            raise ValueError(f"prepared source grid index is invalid: {split}")

        source_audit = split_manifest.get("audit", {}).get(
            "source_configuration_space", {}
        )
        required_source_audit = (
            "serialized_requery",
            "expert_all_margin_safe",
        )
        if not all(source_audit.get(name) is True for name in required_source_audit):
            raise ValueError(
                f"prepared source configuration audit is missing or invalid: {split}"
            )

        depth = split_manifest.get("depth", {})
        if depth.get("dtype") != "float16_normalized" or (
            depth.get("height"),
            depth.get("width"),
            depth.get("max_depth_m"),
        ) != (data.image_height, data.image_width, data.max_depth_m):
            raise ValueError(f"prepared depth contract mismatch: {split}")
        runs = tuple(
            PackedDepthRun(
                path=split_root / item["file"],
                offset=int(item["offset"]),
                frames=int(item["frames"]),
            )
            for item in depth.get("runs", ())
        )
        total_frames = sum(run.frames for run in runs)
        contiguous = all(
            run.offset == sum(previous.frames for previous in runs[:index])
            and run.path.is_file()
            for index, run in enumerate(runs)
        )
        if (
            not runs
            or not contiguous
            or total_frames != int(depth.get("total_frames", -1))
        ):
            raise ValueError(f"invalid prepared depth bank: {split}")
        self.depth_bank = PackedDepthBankSpec(
            runs=runs,
            total_frames=total_frames,
            height=data.image_height,
            width=data.image_width,
        )
        indices = self.arrays["depth_indices"]
        if int(indices.max()) >= total_frames:
            raise ValueError(f"prepared depth index exceeds split bank: {split}")

    def __len__(self) -> int:
        return self.count

    def __getitem__(self, index: int) -> dict[str, Tensor]:
        sample = {
            name: torch.from_numpy(np.array(array[index], copy=True))
            for name, array in self.arrays.items()
        }
        # This runtime-only group is assigned before any distributed shuffling.
        # Four equal global strata cover the exact deployed boundary, one
        # random interior stratum and two diagonal MeanFlow strata.  It is
        # neither a stored expert label nor a policy condition.
        sample["flow_interval_group"] = torch.tensor(index % 4, dtype=torch.uint8)
        return sample


class RepeatedPolicyDataset(Dataset):
    """Deterministically shuffle complete cycles to an exact sample budget."""

    def __init__(
        self,
        dataset: PreparedPolicyDataset,
        count: int,
        seed: int,
        start_index: int = 0,
    ) -> None:
        if count < 1:
            raise ValueError("repeated sample count must be positive")
        if start_index < 0:
            raise ValueError("repeated sample start index cannot be negative")
        self.dataset = dataset
        self.count = count
        self.seed = seed
        self.start_index = start_index
        self.depth_bank = dataset.depth_bank

    def __len__(self) -> int:
        return self.count

    def __getitem__(self, index: int) -> dict[str, Tensor]:
        absolute_index = self.start_index + index
        size = len(self.dataset)
        cycle, position = divmod(absolute_index, size)
        offset = (self.seed + cycle * 0x9E3779B1) % size
        stride = 2 * ((self.seed ^ (cycle * 0x85EBCA77)) % max(size // 2, 1)) + 1
        while math.gcd(stride, size) != 1:
            stride += 2
        sample = self.dataset[(offset + stride * position) % size]
        sample["flow_interval_group"] = torch.tensor(
            absolute_index % 4,
            dtype=torch.uint8,
        )
        return sample
