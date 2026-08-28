"""Compile public and generated routes into the one policy dataset contract."""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import shutil
from typing import Callable

import numpy as np
import torch

from curvenav.config import CurveNavConfig, DataConfig
from curvenav.config_io import load_config
from curvenav.data.depth import depth_camera_contract
from curvenav.data.prepared import (
    policy_dataset_contract,
)
from curvenav.data.trajectory import (
    MAXIMUM_EXPERT_PROJECTION_ADE_RATIO,
    collate_metric_paths,
)
from curvenav.trajectory import MetricCurvatureTrajectory


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
    metric_path: np.ndarray
    scene: str
    reached_goal: bool


def _arc_length(path: np.ndarray) -> float:
    return float(np.linalg.norm(np.diff(path, axis=0), axis=1).sum())


def _fixed_future(path: np.ndarray, future_steps: int) -> tuple[np.ndarray, bool]:
    """Keep future expert transitions without imposing a metric trajectory length."""
    path = np.asarray(path, dtype=np.float32)
    if path.ndim != 2 or path.shape[1] != 2 or len(path) < 2:
        raise ValueError("expert path must have shape [N,2] with N >= 2")
    if len(path) <= future_steps + 1:
        return path, True
    return path[: future_steps + 1], False


def _cumulative_distance(positions: np.ndarray) -> np.ndarray:
    positions = np.asarray(positions, dtype=np.float64)
    if positions.ndim != 2 or len(positions) < 1:
        raise ValueError("route positions must have shape [N, D]")
    return np.concatenate(
        (
            np.zeros(1, dtype=np.float64),
            np.cumsum(np.linalg.norm(np.diff(positions, axis=0), axis=1)),
        )
    )


def _frame_indices(
    anchor: int,
    cumulative_distance: np.ndarray,
    data: DataConfig,
) -> np.ndarray:
    """Select the nearest past frames at fixed traveled-distance offsets."""
    cumulative_distance = np.asarray(cumulative_distance, dtype=np.float64)
    if cumulative_distance.ndim != 1 or not 0 <= anchor < len(cumulative_distance):
        raise ValueError("invalid route cumulative distance or anchor")
    offsets = np.arange(data.observation_frames - 1, -1, -1) * data.frame_spacing_m
    targets = cumulative_distance[anchor] - offsets
    available = cumulative_distance[: anchor + 1]
    upper = np.searchsorted(available, targets, side="right")
    upper = np.minimum(upper, anchor)
    lower = np.maximum(upper - 1, 0)
    choose_upper = np.abs(available[upper] - targets) <= np.abs(
        available[lower] - targets
    )
    return np.where(choose_upper, upper, lower).astype(np.uint32)


def _observation_valid(indices: np.ndarray, observation_frames: int) -> np.ndarray:
    """Keep only the latest occurrence of a repeated padded observation."""
    indices = np.asarray(indices)
    if indices.shape != (observation_frames,):
        raise ValueError("frame indices must match the fixed observation count")
    valid = np.ones(observation_frames, dtype=np.bool_)
    valid[:-1] = indices[:-1] != indices[1:]
    valid[-1] = True
    return valid


def _observation_to_current(
    local_origins: np.ndarray,
    frame_yaw: np.ndarray,
    current_yaw: float,
    observation_frames: int,
) -> np.ndarray:
    """Describe every selected camera pose in the current robot frame."""
    local_origins = np.asarray(local_origins, dtype=np.float32)
    frame_yaw = np.asarray(frame_yaw, dtype=np.float32)
    if local_origins.shape != (observation_frames, 2):
        raise ValueError("local frame origins do not match observation count")
    if frame_yaw.shape != (observation_frames,):
        raise ValueError("frame yaw does not match observation count")
    # Habitat routes live in the world XZ plane.  ``atan2(dz, dx)`` increases
    # towards the robot's right because world Y, not world Z, is up.  CurveNav
    # uses the ROS/Isaac body convention x-forward, y-left, so body yaw is the
    # negative of that stored route angle.
    delta_yaw = float(current_yaw) - frame_yaw
    return np.column_stack(
        (
            local_origins,
            np.sin(delta_yaw),
            np.cos(delta_yaw),
        )
    ).astype(np.float32)


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
    expected_intrinsic = np.asarray(
        [
            [data.canonical_focal_x_px, 0.0, data.image_width / 2.0],
            [0.0, data.canonical_focal_y_px, data.image_height / 2.0],
            [0.0, 0.0, 1.0],
        ]
    )
    if (
        intrinsic.shape != (3, 3)
        or body_from_camera.shape != (4, 4)
        or not np.allclose(intrinsic, expected_intrinsic, atol=1e-6)
        or not np.allclose(body_from_camera, expected_camera_transform, atol=1e-6)
    ):
        raise ValueError("HSSD camera calibration does not match CurveNav")
    cache = root / f"curvenav_hssd_depth_{data.image_height}x{data.image_width}_float16"
    manifest = json.loads((cache / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("dtype") != "float16" or manifest.get(
        "target_camera"
    ) != depth_camera_contract(data):
        raise ValueError("HSSD depth cache does not match the CurveNav data contract")
    records = [
        json.loads(line)
        for line in (root / "routes.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    output: dict[str, list[_Example]] = {"train": [], "validation": []}
    for record in records:
        split = str(record["split"])
        if split not in output:
            raise ValueError(f"invalid HSSD split: {split}")
        route_id = str(record["route_id"])
        cached = manifest.get("runs", {}).get(route_id)
        route_root = root / str(record["route_directory"])
        xy = np.load(route_root / "traj_xy.npy").astype(np.float32)
        yaw = np.load(route_root / "traj_yaw.npy").astype(np.float32)
        if (
            not isinstance(cached, dict)
            or int(cached.get("frames", -1)) != len(xy)
            or len(xy) != len(yaw)
        ):
            raise ValueError(f"HSSD route is missing from the depth cache: {route_id}")
        spacing = np.linalg.norm(np.diff(xy, axis=0), axis=1)
        if not math.isclose(
            float(np.median(spacing)),
            data.expert_waypoint_spacing_m,
            abs_tol=0.01,
        ):
            raise ValueError(
                f"HSSD waypoint spacing does not match CurveNav: {route_id}"
            )
        depth = _DepthRun(
            source=cache / str(cached["file"]),
            frames=len(xy),
            name=f"hssd/{route_id}",
        )
        cumulative_distance = _cumulative_distance(xy)
        for anchor in range(len(xy) - 1):
            frame_indices = _frame_indices(anchor, cumulative_distance, data)
            transform: Callable[[np.ndarray], np.ndarray] = (
                lambda points, anchor=anchor: _planar_local(
                    points, xy[anchor], float(yaw[anchor])
                )
            )
            full_local = transform(xy[anchor:])
            local_path, reached_goal = _fixed_future(full_local, data.future_steps)
            output[split].append(
                _Example(
                    depth_run=depth,
                    depth_indices=frame_indices,
                    point_goal=full_local[-1],
                    observation_to_current=_observation_to_current(
                        transform(xy[frame_indices]),
                        yaw[frame_indices],
                        float(yaw[anchor]),
                        data.observation_frames,
                    ),
                    observation_valid=_observation_valid(
                        frame_indices, data.observation_frames
                    ),
                    metric_path=local_path,
                    scene=f"hssd/{record['scene_id']}",
                    reached_goal=reached_goal,
                )
            )
    return output


def _save_array(split_root: Path, name: str, value: np.ndarray) -> dict[str, object]:
    path = split_root / f"{name}.npy"
    np.save(path, value, allow_pickle=False)
    return {"file": path.name, "shape": list(value.shape), "dtype": str(value.dtype)}


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

    codec = MetricCurvatureTrajectory(
        num_curvature_control_points=config.trajectory.num_curvature_control_points,
        degree=config.trajectory.curvature_spline_degree,
        num_path_points=config.trajectory.num_path_points,
        length_pretransform_mean=config.trajectory.length_pretransform_mean,
        length_pretransform_std=config.trajectory.length_pretransform_std,
        curvature_control_mean_inv_m=(
            config.trajectory.curvature_control_mean_inv_m
        ),
        curvature_control_std_inv_m=config.trajectory.curvature_control_std_inv_m,
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
    examples = [example for example, selected in zip(examples, keep, strict=True) if selected]
    depth_indices = depth_indices[keep]
    point_goal = point_goal[keep]
    observation_to_current = observation_to_current[keep]
    observation_valid = observation_valid[keep]
    curve_values = curve_values[keep]
    reference_path = reference_path[keep]
    projection_error = projection_error[keep]
    with torch.no_grad():
        decoded, heading, curvature = codec.decode_values(
            torch.from_numpy(curve_values),
        )
        if not torch.equal(decoded, torch.from_numpy(reference_path)):
            raise RuntimeError("stored expert controls do not reproduce their path")
        maximum_curvature = curvature.abs().amax(1).numpy()
        total_turn = (heading[:, 1:] - heading[:, :-1]).abs().sum(1).numpy()

    arrays = {
        "depth_indices": _save_array(split_root, "depth_indices", depth_indices),
        "point_goal": _save_array(split_root, "point_goal", point_goal),
        "observation_to_current": _save_array(
            split_root, "observation_to_current", observation_to_current
        ),
        "observation_valid": _save_array(
            split_root, "observation_valid", observation_valid
        ),
        "curve_values": _save_array(split_root, "curve_values", curve_values),
    }
    local_arc = np.asarray([_arc_length(example.metric_path) for example in examples])
    length_pretransform = curve_values[:, 0] + np.log(
        -np.expm1(-curve_values[:, 0])
    )
    curvature_controls = curve_values[:, 1:]
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
        "production_curve_coordinate_statistics": {
            "length_pretransform_mean": float(length_pretransform.mean()),
            "length_pretransform_std": float(length_pretransform.std()),
            "curvature_control_mean_inv_m": float(curvature_controls.mean()),
            "curvature_control_std_inv_m": float(curvature_controls.std()),
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
            "dtype": "float16_normalized",
            "height": config.data.image_height,
            "width": config.data.image_width,
            "max_depth_m": config.data.max_depth_m,
            "total_frames": total_frames,
            "runs": depth_manifest,
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
    if output_root.exists() or building_root.exists():
        raise FileExistsError(f"output dataset already exists: {output_root}")
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
        manifest = {
            "contract": {
                **policy_dataset_contract(config.data, config.trajectory),
                "point_goal_semantics": "mission_destination_in_current_robot_xy",
                "expert_path_source": "fixed_future_waypoints_or_true_goal",
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
        os.replace(building_root, output_root)
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
