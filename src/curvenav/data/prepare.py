"""Compile public and generated routes into the one policy dataset contract."""

from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
from dataclasses import asdict, dataclass, replace
import json
import math
import os
from pathlib import Path
import shutil
from typing import Callable

import numpy as np
import torch
import yaml
from torch import Tensor

from curvenav.config import CurveNavConfig, DataConfig
from curvenav.config_io import load_config
from curvenav.data.contracts import expert_navigation_geometry_contract
from curvenav.data.depth import (
    BENCHMARK_INTRINSICS,
    CANONICAL_INTRINSICS,
    depth_camera_contract,
)
from curvenav.data.history import (
    OBSERVATION_PERIOD_S,
    ObservationHistory,
)
from curvenav.data_generation.geometry import points_at_arc
from curvenav.data_generation.audit import validate_route_pose
from curvenav.data.prepared import (
    policy_dataset_contract,
)
from curvenav.data.privileged import (
    SOURCE_CONFIGURATION_PATH_SAMPLING,
    SOURCE_CONFIGURATION_QUERY_SPACING_M,
    SOURCE_CONFIGURATION_QUERY_TYPE,
    SourceConfigurationSpaceQuery,
)
from curvenav.data.trajectory import (
    MAXIMUM_EXPERT_PROJECTION_ADE_RATIO,
    collate_metric_paths,
)
from curvenav.trajectory import IncrementalBSplineTrajectory
from curvenav.trajectory.resampling import path_arc_length
from curvenav.physical import EXTRA_CLEARANCE_M
from curvenav.data.obstacle_memory import ObstacleMemory


@dataclass(frozen=True)
class _DepthRun:
    source: Path
    frames: int
    name: str


@dataclass
class _Example:
    depth_run: _DepthRun
    depth_indices: np.ndarray
    point_goal: np.ndarray
    observation_to_current: np.ndarray
    observation_valid: np.ndarray
    observation_age_s: np.ndarray
    camera_intrinsics: np.ndarray
    camera_to_body: np.ndarray
    source_grid_path: Path
    source_origin_xy: np.ndarray
    source_yaw_rad: float
    metric_path: np.ndarray
    scene: str
    reached_goal: bool
    obstacle_memory: np.ndarray


def _arc_length(path: np.ndarray) -> float:
    return float(np.linalg.norm(np.diff(path, axis=0), axis=1).sum())


def _fixed_future(
    path: np.ndarray, future_steps: int, spacing_m: float
) -> tuple[np.ndarray, bool]:
    """Clip the recorded path to the metric horizon without discarding short turns."""
    path = np.asarray(path, dtype=np.float32)
    if path.ndim != 2 or path.shape[1] != 2 or len(path) < 2:
        raise ValueError("expert path must have shape [N,2] with N >= 2")
    length = _arc_length(path)
    horizon = future_steps * spacing_m
    arc = np.r_[0., np.cumsum(np.linalg.norm(np.diff(path, axis=0), axis=1))]
    end = min(length, horizon)
    endpoint = points_at_arc(path, np.array([end]))
    return np.vstack((path[arc < end], endpoint)).astype(np.float32), length <= horizon


def _planar_local(points: np.ndarray, origin: np.ndarray, yaw: float) -> np.ndarray:
    delta = np.asarray(points, dtype=np.float32) - np.asarray(origin, dtype=np.float32)
    cosine, sine = np.cos(-yaw), np.sin(-yaw)
    return np.stack(
        (
            delta[..., 0] * cosine - delta[..., 1] * sine,
            -(delta[..., 0] * sine + delta[..., 1] * cosine),
        ),
        axis=-1,
    ).astype(np.float32, copy=False)


def _hssd_examples(root: Path, config: CurveNavConfig) -> dict[str, list[_Example]]:
    data = config.data
    source_manifest = json.loads(
        (root / "dataset_manifest.json").read_text(encoding="utf-8")
    )
    if (
        source_manifest.get("route_contract", {}).get("navigation_geometry")
        != expert_navigation_geometry_contract()
    ):
        raise ValueError(
            "HSSD experts must be planned against the stage and static objects"
        )
    pitch = math.radians(data.camera_downward_pitch_degrees)
    sine, cosine = math.sin(pitch), math.cos(pitch)
    expected_camera_transform = np.asarray(
        [
            [0.0, -sine, cosine, data.camera_forward_offset_m],
            [-1.0, 0.0, 0.0, 0.0],
            [0.0, -cosine, -sine, data.camera_height_m],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    camera = source_manifest.get("camera", {})
    intrinsic = np.asarray(camera.get("image", {}).get("K", ()), dtype=np.float64)
    body_from_camera = np.asarray(
        camera.get("body_from_camera_optical", ()), dtype=np.float64
    )
    expected_intrinsic = BENCHMARK_INTRINSICS.matrix()
    if (
        intrinsic.shape != (3, 3)
        or body_from_camera.shape != (4, 4)
        or not np.allclose(intrinsic, expected_intrinsic, atol=1e-6)
        or not np.allclose(body_from_camera, expected_camera_transform, atol=1e-6)
    ):
        raise ValueError("HSSD camera calibration does not match CurveNav")
    intrinsic = CANONICAL_INTRINSICS.matrix()
    if (
        source_manifest.get("schema") != "curvenav_hssd_policy_depth_routes_v4"
        or source_manifest.get("observation") != depth_camera_contract(data)
    ):
        raise ValueError("HSSD must contain the exact policy depth storage contract")
    records = [
        json.loads(line)
        for line in (root / "routes.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    output: dict[str, list[_Example]] = {"train": [], "validation": []}
    with ProcessPoolExecutor(
        max_workers=8, mp_context=multiprocessing.get_context("spawn")
    ) as pool:
        tasks = ((record, root, data, intrinsic, body_from_camera) for record in records)
        for index, (split, examples) in enumerate(pool.map(_hssd_route, tasks), 1):
            output[split].extend(examples)
            if index % 25 == 0:
                print(f"Prepared causal geometry: {index}/{len(records)} routes", flush=True)
    return output


def _hssd_route(arguments):
    record, root, data, intrinsic, body_from_camera = arguments
    examples = []
    split = str(record["split"])
    if split not in ("train", "validation"):
        raise ValueError(f"invalid HSSD split: {split}")
    route_id = str(record["route_id"])
    route_root = root / str(record["route_directory"])
    source_grid_path = (route_root.parent / "navigation_grid.npz").resolve()
    xy = np.load(route_root / "traj_xy.npy").astype(np.float32)
    yaw = np.load(route_root / "traj_yaw.npy").astype(np.float32)
    poses = np.load(route_root / "body_to_world.npy")
    times = np.load(route_root / "timestamps.npy")
    validate_route_pose(xy, yaw, poses)
    if poses.shape != (len(xy), 4, 4) or times.shape != (len(xy),):
        raise ValueError("depth route poses/timestamps do not match frames")
    if (
        int(record["frames"]) != len(xy) or len(xy) != len(yaw)
    ):
        raise ValueError(f"HSSD route frame count mismatch: {route_id}")
    if not np.allclose(np.diff(times), OBSERVATION_PERIOD_S, atol=1e-6, rtol=0):
        raise ValueError(
            f"HSSD observations must follow the 10 Hz sensor clock: {route_id}"
        )
    depth = _DepthRun(
        source=route_root / "depth.npy",
        frames=len(xy),
        name=f"hssd/{route_id}",
    )
    history = ObservationHistory()
    memory = ObstacleMemory(data.future_steps * data.expert_waypoint_spacing_m, data.max_depth_m)
    depth_frames = np.load(depth.source, mmap_mode="r")
    for anchor in range(len(xy) - 1):
        frame_indices, relative, age, valid = history.update(
            anchor, poses[anchor], float(times[anchor])
        )
        transform: Callable[[np.ndarray], np.ndarray] = (
            lambda points, anchor=anchor: _planar_local(
                points, xy[anchor], float(yaw[anchor])
            )
        )
        full_local = transform(xy[anchor:])
        local_path, reached_goal = _fixed_future(
            full_local, data.future_steps, data.expert_waypoint_spacing_m
        )
        examples.append(
            _Example(
                obstacle_memory=memory.update(depth_frames[anchor], intrinsic, body_from_camera, poses[anchor]),
                depth_run=depth,
                depth_indices=frame_indices,
                point_goal=full_local[-1],
                observation_to_current=relative,
                observation_valid=valid,
                observation_age_s=age,
                camera_intrinsics=np.broadcast_to(
                    intrinsic, (data.observation_frames, 3, 3)
                ),
                camera_to_body=np.broadcast_to(
                    body_from_camera, (data.observation_frames, 4, 4)
                ),
                source_grid_path=source_grid_path,
                source_origin_xy=xy[anchor].copy(),
                source_yaw_rad=float(yaw[anchor]),
                metric_path=local_path,
                scene=f"hssd/{record['scene_id']}",
                reached_goal=reached_goal,
            )
        )
    return split, examples


def _save_array(split_root: Path, name: str, value: np.ndarray) -> dict[str, object]:
    path = split_root / f"{name}.npy"
    np.save(path, value, allow_pickle=False)
    return {"file": path.name, "shape": list(value.shape), "dtype": str(value.dtype)}


@dataclass(frozen=True)
class _SourceMetadata:
    grid_paths: tuple[Path, ...]
    grid_index: np.ndarray
    origin_xy: np.ndarray
    yaw_rad: np.ndarray
    query: SourceConfigurationSpaceQuery


def _source_metadata(examples: list[_Example]) -> _SourceMetadata:
    """Keep source C-space provenance out of the policy condition tensors."""
    grid_paths = tuple(
        sorted(
            {example.source_grid_path.resolve() for example in examples},
            key=lambda path: path.as_posix(),
        )
    )
    index = {path: value for value, path in enumerate(grid_paths)}
    return _SourceMetadata(
        grid_paths=grid_paths,
        grid_index=np.asarray(
            [index[example.source_grid_path.resolve()] for example in examples],
            dtype=np.int64,
        ),
        origin_xy=np.stack([example.source_origin_xy for example in examples]).astype(
            np.float32
        ),
        yaw_rad=np.asarray(
            [example.source_yaw_rad for example in examples], dtype=np.float32
        ),
        query=SourceConfigurationSpaceQuery.from_paths(grid_paths),
    )


def _source_minimum_clearance(
    path: Tensor,
    source: _SourceMetadata,
    planning_horizon_m: float,
) -> np.ndarray:
    """Evaluate prepared curves in bounded chunks against the source oracle."""
    values = []
    for start in range(0, len(path), 2048):
        end = min(start + 2048, len(path))
        values.append(
            source.query.query(
                path[start:end],
                torch.from_numpy(source.grid_index[start:end]),
                torch.from_numpy(source.origin_xy[start:end]),
                torch.from_numpy(source.yaw_rad[start:end]),
                max(planning_horizon_m, float(path_arc_length(path[start:end]).max())),
            ).minimum_clearance_m.numpy()
        )
    return np.concatenate(values)


def _flow_coordinate_statistics(
    curve_values: np.ndarray,
) -> dict[str, object]:
    """Measure train-split physical control-increment normalization."""
    controls = np.asarray(curve_values, dtype=np.float64).reshape(-1, 7, 2)
    increments = np.diff(
        np.concatenate((np.zeros((len(controls), 1, 2)), controls), axis=1),
        axis=1,
    )
    # One physical XY scale per control preserves the Euclidean metric and
    # remains defined when a boundary condition makes one axis deterministic.
    scale = np.sqrt(increments.var(axis=0, ddof=1).mean(axis=-1))
    return {
        "control_increment_mean_xy_m": increments.mean(axis=0).ravel().tolist(),
        "control_increment_std_xy_m": np.repeat(scale, 2).tolist(),
    }


def _copy_source_configuration_grids(
    split_root: Path,
    source: _SourceMetadata,
) -> list[dict[str, str]]:
    """Make the prepared dataset self-contained for the same source oracle."""
    geometry_root = split_root / "source_configuration"
    geometry_root.mkdir()
    entries = []
    for index, path in enumerate(source.grid_paths):
        destination = geometry_root / f"{index:05d}.npz"
        shutil.copyfile(path, destination)
        entries.append({"file": destination.relative_to(split_root).as_posix()})
    return entries


def _audit_serialized_source_contract(
    split_root: Path,
    source_grids: list[dict[str, str]],
    codec: IncrementalBSplineTrajectory,
    planning_horizon_m: float,
) -> dict[str, object]:
    """Certify serialized controls against the copied source C-space grids.

    The compiler gates curves before writing, but this second pass proves that
    the exact expert arrays and copied geometry used by a future evaluator
    preserve the same safety relation.  It is a data-contract check, not a
    training-time or inference-time safety mechanism.
    """
    query = SourceConfigurationSpaceQuery.from_paths(
        tuple(split_root / entry["file"] for entry in source_grids)
    )
    expert_values = np.load(split_root / "curve_values.npy", mmap_mode="r")
    grid_index = np.load(split_root / "source_grid_index.npy", mmap_mode="r")
    origin_xy = np.load(split_root / "source_origin_xy.npy", mmap_mode="r")
    yaw_rad = np.load(split_root / "source_yaw_rad.npy", mmap_mode="r")

    expert_minimum: list[np.ndarray] = []
    expert_oob_count = 0
    for start in range(0, len(expert_values), 2048):
        end = min(start + 2048, len(expert_values))
        indices = torch.from_numpy(
            np.array(grid_index[start:end], dtype=np.int64, copy=True)
        )
        origins = torch.from_numpy(
            np.array(origin_xy[start:end], dtype=np.float32, copy=True)
        )
        yaws = torch.from_numpy(
            np.array(yaw_rad[start:end], dtype=np.float32, copy=True)
        )
        expert_path, _ = codec.decode_values(
            torch.from_numpy(
                np.array(expert_values[start:end], dtype=np.float32, copy=True)
            )
        )
        expert_query = query.query(
            expert_path, indices, origins, yaws,
            max(planning_horizon_m, float(path_arc_length(expert_path).max())),
        )
        expert_minimum.append(expert_query.minimum_clearance_m.numpy())
        expert_oob_count += int(
            (~expert_query.in_world_bounds).any(dim=-1).sum().item()
        )

    expert_minimum_values = np.concatenate(expert_minimum)
    expert_all_margin_safe = bool(
        expert_oob_count == 0
        and np.all(expert_minimum_values + 1e-6 >= EXTRA_CLEARANCE_M)
    )
    if not expert_all_margin_safe:
        raise RuntimeError(
            "serialized expert curves violate the source C-space contract: "
            f"oob={expert_oob_count}, min={expert_minimum_values.min():.6f}"
        )
    return {
        "query": SOURCE_CONFIGURATION_QUERY_TYPE,
        "spacing_m": SOURCE_CONFIGURATION_QUERY_SPACING_M,
        "path_sampling": SOURCE_CONFIGURATION_PATH_SAMPLING,
        "out_of_bounds": "non_executable_negative_clearance",
        "serialized_requery": True,
        "expert_count": int(len(expert_values)),
        "expert_all_margin_safe": expert_all_margin_safe,
        "expert_oob_count": expert_oob_count,
        "expert_minimum_clearance_m": float(expert_minimum_values.min()),
    }


def _compile_split(
    split_root: Path,
    examples: list[_Example],
    seed: int,
    config: CurveNavConfig,
) -> dict[str, object]:
    split_root.mkdir(parents=True)
    depth_root = split_root / "depth"
    depth_root.mkdir()
    depth_offsets: dict[str, int] = {}
    depth_manifest = []
    total_frames = 0
    for example in examples:
        key = example.depth_run.name
        if key in depth_offsets:
            continue
        index = len(depth_manifest)
        destination = depth_root / f"{index:05d}.npy"
        os.link(example.depth_run.source, destination)
        depth_offsets[key] = total_frames
        depth_manifest.append(
            {
                "file": destination.relative_to(split_root).as_posix(),
                "frames": example.depth_run.frames,
                "offset": total_frames,
            }
        )
        total_frames += example.depth_run.frames

    generator = np.random.default_rng(seed)
    order = generator.permutation(len(examples))
    examples = [examples[int(index)] for index in order]
    depth_indices = np.stack(
        [
            example.depth_indices + depth_offsets[example.depth_run.name]
            for example in examples
        ]
    ).astype(np.uint32)
    point_goal = np.stack([example.point_goal for example in examples]).astype(
        np.float32
    )
    observation_to_current = np.stack(
        [example.observation_to_current for example in examples]
    ).astype(np.float32)
    observation_valid = np.stack(
        [example.observation_valid for example in examples]
    ).astype(np.bool_)

    codec = IncrementalBSplineTrajectory(
        num_control_points=config.trajectory.num_control_points,
        degree=config.trajectory.spline_degree,
        num_path_points=config.trajectory.num_path_points,
        control_increment_mean_xy_m=(config.trajectory.control_increment_mean_xy_m),
        control_increment_std_xy_m=(config.trajectory.control_increment_std_xy_m),
    )
    curve_batches = []
    reference_batches = []
    projection_error_batches = []
    for start in range(0, len(examples), 2048):
        paths = [
            torch.from_numpy(example.metric_path)
            for example in examples[start : start + 2048]
        ]
        curve_values, reference_path, projection_error = collate_metric_paths(
            paths,
            codec,
        )
        curve_batches.append(curve_values.numpy())
        reference_batches.append(reference_path.numpy())
        projection_error_batches.append(projection_error.numpy())
    curve_values = np.concatenate(curve_batches).astype(np.float32)
    reference_path = np.concatenate(reference_batches).astype(np.float32)
    projection_error = np.concatenate(projection_error_batches).astype(np.float32)
    maximum_projection_error = (
        config.data.expert_waypoint_spacing_m * MAXIMUM_EXPERT_PROJECTION_ADE_RATIO
    )
    keep = projection_error <= maximum_projection_error
    rejected_projection_count = int((~keep).sum())
    examples = [
        example for example, selected in zip(examples, keep, strict=True) if selected
    ]
    depth_indices = depth_indices[keep]
    point_goal = point_goal[keep]
    observation_to_current = observation_to_current[keep]
    observation_valid = observation_valid[keep]
    curve_values = curve_values[keep]
    reference_path = reference_path[keep]
    projection_error = projection_error[keep]
    with torch.no_grad():
        decoded, heading = codec.decode_values(
            torch.from_numpy(curve_values),
        )
        if not torch.equal(decoded, torch.from_numpy(reference_path)):
            raise RuntimeError("stored expert controls do not reproduce their path")
        heading_delta = heading[:, 1:] - heading[:, :-1]
        wrapped_turn = torch.atan2(heading_delta.sin(), heading_delta.cos())
        total_turn = wrapped_turn.abs().sum(1).numpy()
        segment = torch.linalg.vector_norm(decoded[:, 1:] - decoded[:, :-1], dim=-1)
        support = 0.5 * (segment[:, 1:] + segment[:, :-1])
        maximum_curvature = (
            (wrapped_turn[:, 1:].abs() / support.clamp_min(1e-6)).amax(1).numpy()
        )
        extent_m = config.data.future_steps * config.data.expert_waypoint_spacing_m
        source = _source_metadata(examples)
        minimum_clearance = _source_minimum_clearance(decoded, source, extent_m)

    safe = minimum_clearance + 1e-6 >= EXTRA_CLEARANCE_M
    rejected_clearance_count = int((~safe).sum())
    examples = [
        example for example, selected in zip(examples, safe, strict=True) if selected
    ]
    depth_indices = depth_indices[safe]
    point_goal = point_goal[safe]
    observation_to_current = observation_to_current[safe]
    observation_valid = observation_valid[safe]
    curve_values = curve_values[safe]
    reference_path = reference_path[safe]
    projection_error = projection_error[safe]
    decoded = decoded[safe]
    heading = heading[safe]
    total_turn = total_turn[safe]
    maximum_curvature = maximum_curvature[safe]
    minimum_clearance = minimum_clearance[safe]

    source = _source_metadata(examples)
    observed_flow_statistics = _flow_coordinate_statistics(
        curve_values,
    )

    arrays = {
        "obstacle_memory": _save_array(
            split_root, "obstacle_memory", np.stack([e.obstacle_memory for e in examples])
        ),
        "depth_indices": _save_array(split_root, "depth_indices", depth_indices),
        "point_goal": _save_array(split_root, "point_goal", point_goal),
        "observation_to_current": _save_array(
            split_root, "observation_to_current", observation_to_current
        ),
        "observation_valid": _save_array(
            split_root, "observation_valid", observation_valid
        ),
        "curve_values": _save_array(split_root, "curve_values", curve_values),
        "source_grid_index": _save_array(
            split_root,
            "source_grid_index",
            source.grid_index,
        ),
        "source_origin_xy": _save_array(
            split_root,
            "source_origin_xy",
            source.origin_xy,
        ),
        "source_yaw_rad": _save_array(
            split_root,
            "source_yaw_rad",
            source.yaw_rad,
        ),
    }
    for name in ("observation_age_s", "camera_intrinsics", "camera_to_body"):
        arrays[name] = _save_array(
            split_root,
            name,
            np.stack([getattr(e, name) for e in examples]).astype(np.float32),
        )
    source_grids = _copy_source_configuration_grids(split_root, source)
    serialized_source_audit = _audit_serialized_source_contract(
        split_root,
        source_grids,
        codec,
        extent_m,
    )
    local_arc = np.asarray([_arc_length(example.metric_path) for example in examples])
    flow_coordinates = codec.coordinates_from_values(
        torch.from_numpy(curve_values),
    ).numpy()
    goal_distance = np.linalg.norm(point_goal, axis=1)
    endpoint = reference_path[:, -1]
    goal_angle = np.degrees(
        np.arccos(
            np.clip(
                (endpoint * point_goal).sum(axis=1)
                / np.maximum(np.linalg.norm(endpoint, axis=1) * goal_distance, 1e-8),
                -1.0,
                1.0,
            )
        )
    )
    scene_counts = Counter(example.scene for example in examples)
    audit = {
        "local_arc_m": {
            "mean": float(local_arc.mean()),
            "p50": float(np.quantile(local_arc, 0.5)),
            "p95": float(np.quantile(local_arc, 0.95)),
            "max": float(local_arc.max()),
        },
        "point_goal_distance_m": {
            "mean": float(goal_distance.mean()),
            "p50": float(np.quantile(goal_distance, 0.5)),
            "p95": float(np.quantile(goal_distance, 0.95)),
            "max": float(goal_distance.max()),
        },
        "goal_prefix_angle_deg": {
            "mean": float(goal_angle.mean()),
            "p95": float(np.quantile(goal_angle, 0.95)),
            "max": float(goal_angle.max()),
        },
        "production_curve_projection_ade_m": {
            "mean": float(projection_error.mean()),
            "p95": float(np.quantile(projection_error, 0.95)),
            "max": float(projection_error.max()),
        },
        "production_curve_projection_rejected": rejected_projection_count,
        "production_curve_clearance_rejected": rejected_clearance_count,
        "source_configuration_space": {
            **serialized_source_audit,
            "grid_count": len(source_grids),
        },
        "production_curve_minimum_clearance_m": {
            "min": float(minimum_clearance.min()),
            "p05": float(np.quantile(minimum_clearance, 0.05)),
        },
        "production_curve_coordinate_statistics": {
            **observed_flow_statistics,
            "standardized_coordinate_rms": float(np.sqrt(np.mean(flow_coordinates**2))),
        },
        "production_curve_max_curvature_inv_m": {
            "p50": float(np.quantile(maximum_curvature, 0.50)),
            "p95": float(np.quantile(maximum_curvature, 0.95)),
            "p99": float(np.quantile(maximum_curvature, 0.99)),
            "max": float(maximum_curvature.max()),
        },
        "production_curve_total_turn_rad": {
            "p50": float(np.quantile(total_turn, 0.50)),
            "p95": float(np.quantile(total_turn, 0.95)),
            "p99": float(np.quantile(total_turn, 0.99)),
            "max": float(total_turn.max()),
        },
        "near_goal_fraction": float(
            np.mean([example.reached_goal for example in examples])
        ),
        "scene_samples": dict(sorted(scene_counts.items())),
    }
    split_manifest = {
        "samples": len(examples),
        "arrays": arrays,
        "depth": {
            "dtype": "depth_float16_zero_invalid",
            "height": config.data.image_height,
            "width": config.data.image_width,
            "max_depth_m": config.data.max_depth_m,
            "total_frames": total_frames,
            "runs": depth_manifest,
        },
        "source_configuration_space": {
            "query": SOURCE_CONFIGURATION_QUERY_TYPE,
            "spacing_m": SOURCE_CONFIGURATION_QUERY_SPACING_M,
            "path_sampling": SOURCE_CONFIGURATION_PATH_SAMPLING,
            "out_of_bounds": "non_executable_negative_clearance",
            "grids": source_grids,
        },
        "audit": audit,
    }
    (split_root / "manifest.json").write_text(
        json.dumps(split_manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    return split_manifest


def compile_policy_dataset(
    hssd_root: Path,
    output_root: Path,
    config: CurveNavConfig,
) -> None:
    config.validate()
    hssd = _hssd_examples(hssd_root.expanduser().resolve(), config)
    output_root = output_root.expanduser().resolve()
    building_root = output_root.with_name(output_root.name + ".building")
    replaced_root = output_root.with_name(output_root.name + ".replaced")
    if building_root.exists() or replaced_root.exists():
        raise FileExistsError(
            f"unfinished dataset transaction exists beside: {output_root}"
        )
    building_root.mkdir(parents=True)
    try:
        split_manifests = {}
        for index, split in enumerate(("train", "validation")):
            examples = hssd[split]
            if not examples:
                raise ValueError(f"compiled split has no examples: {split}")
            split_manifests[split] = _compile_split(
                building_root / split,
                examples,
                config.training.seed + index,
                config,
            )
            if split == "train":
                statistics = split_manifests[split]["audit"][
                    "production_curve_coordinate_statistics"
                ]
                fitted = {
                    name: tuple(statistics[name])
                    for name in (
                        "control_increment_mean_xy_m",
                        "control_increment_std_xy_m",
                    )
                }
                config = replace(
                    config,
                    trajectory=replace(config.trajectory, **fitted),
                    data=replace(config.data, root=str(output_root)),
                )
                config.validate()
                # Recompute this diagnostic using the fitted training scale.
                values = np.load(building_root / split / "curve_values.npy")
                controls = values.astype(np.float64).reshape(-1, 7, 2)
                increments = np.diff(
                    np.concatenate((np.zeros((len(controls), 1, 2)), controls), axis=1),
                    axis=1,
                ).reshape(-1, 14)
                statistics["standardized_coordinate_rms"] = float(
                    np.sqrt(
                        np.mean(
                            (
                                (
                                    increments
                                    - np.array(fitted["control_increment_mean_xy_m"])
                                )
                                / np.array(fitted["control_increment_std_xy_m"])
                            )
                            ** 2
                        )
                    )
                )
                (building_root / split / "manifest.json").write_text(
                    json.dumps(split_manifests[split], indent=2)
                )
        configuration = {
            "data": asdict(config.data),
            "training": asdict(config.training),
            "model": {
                name: asdict(getattr(config, name))
                for name in (
                    "trajectory",
                    "depth_encoder",
                    "condition_encoder",
                    "trajectory_decoder",
                )
            },
        }
        (building_root / "config.yaml").write_text(
            yaml.safe_dump(configuration, sort_keys=False)
        )
        manifest = {
            "contract": {
                **policy_dataset_contract(config.data, config.trajectory),
                "point_goal_semantics": "mission_destination_in_current_robot_xy",
                "expert_path_source": "metric_arc_horizon_or_true_goal",
                "expert_endpoint_policy": "implicit_local_subgoal_unless_near_goal",
                "metric_scale_forced": False,
            },
            "splits": {
                split: {
                    "samples": split_manifest["samples"],
                    "manifest": f"{split}/manifest.json",
                }
                for split, split_manifest in split_manifests.items()
            },
        }
        (building_root / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
        )
        replacing = output_root.exists()
        if replacing:
            os.replace(output_root, replaced_root)
        try:
            os.replace(building_root, output_root)
        except BaseException:
            if replacing:
                os.replace(replaced_root, output_root)
            raise
        if replacing:
            shutil.rmtree(replaced_root)
        print(json.dumps(manifest, indent=2, ensure_ascii=False))
    except BaseException:
        shutil.rmtree(building_root, ignore_errors=True)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description="Compile the CurveNav policy dataset")
    parser.add_argument("--hssd-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    compile_policy_dataset(args.hssd_root, args.output, load_config(args.config))


if __name__ == "__main__":
    main()
