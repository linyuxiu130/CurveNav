"""Compile public and generated routes into the one policy dataset contract."""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
import hashlib
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
from curvenav.data.prepared import (
    policy_dataset_contract,
)
from curvenav.data.trajectory import collate_metric_paths
from curvenav.trajectory import PlanarBSplineCodec


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
            np.cumsum(np.linalg.vector_norm(np.diff(positions, axis=0), axis=1)),
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
    offsets = (
        np.arange(data.observation_frames - 1, -1, -1) * data.frame_spacing_m
    )
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
    delta_yaw = frame_yaw - float(current_yaw)
    return np.column_stack(
        (
            local_origins,
            np.sin(delta_yaw),
            np.cos(delta_yaw),
        )
    ).astype(np.float32)


def _sand_local(
    points: np.ndarray,
    origin: np.ndarray,
    yaw: float,
    pitch: float,
) -> np.ndarray:
    delta = np.asarray(points, dtype=np.float32) - np.asarray(origin, dtype=np.float32)
    cos_yaw = np.cos(-yaw)
    sin_yaw = np.sin(-yaw)
    cos_pitch = np.cos(-pitch)
    sin_pitch = np.sin(-pitch)
    forward = delta[..., 0] * cos_pitch + delta[..., 2] * sin_pitch
    lateral = delta[..., 1]
    return np.stack(
        (
            forward * cos_yaw - lateral * sin_yaw,
            forward * sin_yaw + lateral * cos_yaw,
        ),
        axis=-1,
    ).astype(np.float32, copy=False)


def _stable_sand_split(run_name: str, seed: int) -> str:
    digest = hashlib.sha256(f"{seed}:{run_name}".encode()).digest()
    return "train" if int.from_bytes(digest[:8], "little") % 10 else "validation"


def _sand_examples(root: Path, config: CurveNavConfig) -> dict[str, list[_Example]]:
    data = config.data
    cache = root / f"curvenav_depth_{data.image_height}x{data.image_width}_float16"
    manifest = json.loads((cache / "manifest.json").read_text(encoding="utf-8"))
    if (
        manifest.get("dtype") != "float16"
        or (manifest.get("height"), manifest.get("width"), manifest.get("max_depth_m"))
        != (data.image_height, data.image_width, data.max_depth_m)
    ):
        raise ValueError("SanD depth cache does not match the CurveNav data contract")
    output: dict[str, list[_Example]] = {"train": [], "validation": []}
    for run_dir in sorted(root.glob("dataset_*/run_*")):
        run_key = run_dir.relative_to(root).as_posix()
        if run_key in manifest.get("excluded_runs", {}):
            continue
        if run_key not in manifest["runs"]:
            raise ValueError(f"SanD run is missing from the depth cache: {run_key}")
        xyz = np.load(run_dir / "traj_xyz.npy").astype(np.float32)
        yaw = np.load(run_dir / "traj_yaw.npy").astype(np.float32)
        pitch = np.load(run_dir / "traj_pitch.npy").astype(np.float32)
        if not (len(xyz) == len(yaw) == len(pitch) == int(manifest["runs"][run_key])):
            raise ValueError(f"SanD route/depth length mismatch: {run_key}")
        depth = _DepthRun(
            source=cache / run_dir.parent.name / f"{run_dir.name}.npy",
            frames=len(xyz),
            name=f"sand/{run_key}",
        )
        split = _stable_sand_split(run_key, config.training.seed)
        scene = f"sand/{run_dir.parent.name}"
        cumulative_distance = _cumulative_distance(xyz)
        route_spacing = np.linalg.vector_norm(np.diff(xyz, axis=0), axis=1)
        moving_spacing = route_spacing[route_spacing > 1e-5]
        if len(moving_spacing) and not math.isclose(
            float(np.median(moving_spacing)),
            data.expert_waypoint_spacing_m,
            abs_tol=0.03,
        ):
            raise ValueError(f"SanD waypoint spacing does not match CurveNav: {run_key}")
        for anchor in range(len(xyz) - 1):
            frame_indices = _frame_indices(anchor, cumulative_distance, data)
            transform: Callable[[np.ndarray], np.ndarray] = (
                lambda points, anchor=anchor: _sand_local(
                    points, xyz[anchor], float(yaw[anchor]), float(pitch[anchor])
                )
            )
            full_local = transform(xyz[anchor:])
            local_path, reached_goal = _fixed_future(full_local, data.future_steps)
            output[split].append(
                _Example(
                    depth_run=depth,
                    depth_indices=frame_indices,
                    point_goal=full_local[-1],
                    observation_to_current=_observation_to_current(
                        transform(xyz[frame_indices]),
                        yaw[frame_indices],
                        float(yaw[anchor]),
                        data.observation_frames,
                    ),
                    observation_valid=_observation_valid(
                        frame_indices,
                        data.observation_frames,
                    ),
                    metric_path=local_path,
                    scene=scene,
                    reached_goal=reached_goal,
                )
            )
    return output


def _hssd_examples(root: Path, config: CurveNavConfig) -> dict[str, list[_Example]]:
    data = config.data
    cache = root / f"curvenav_hssd_depth_{data.image_height}x{data.image_width}_float16"
    manifest = json.loads((cache / "manifest.json").read_text(encoding="utf-8"))
    if (
        manifest.get("dtype") != "float16"
        or (
            manifest.get("height"),
            manifest.get("width"),
            manifest.get("max_depth_m"),
        )
        != (data.image_height, data.image_width, data.max_depth_m)
    ):
        raise ValueError("HSSD depth cache does not match the CurveNav data contract")
    records = [
        json.loads(line)
        for line in (root / "samples.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    output: dict[str, list[_Example]] = {"train": [], "validation": []}
    for record in records:
        split = str(record["split"])
        if split not in output:
            raise ValueError(f"invalid HSSD split: {split}")
        sample_id = str(record["sample_id"])
        cached = manifest.get("samples", {}).get(sample_id)
        if not isinstance(cached, dict) or int(cached.get("frames", -1)) != data.observation_frames:
            raise ValueError(f"HSSD sample is missing from the depth cache: {sample_id}")
        with np.load(root / str(record["sample_directory"]) / "geometry.npz") as values:
            point_goal = values["task_goal_local_xy"].astype(np.float32)
            observation_to_current = values["observation_to_current"].astype(np.float32)
            metric_path = values["target_path_local_xy"].astype(np.float32)
        if observation_to_current.shape != (data.observation_frames, 4):
            raise ValueError(f"HSSD observation transform does not match CurveNav: {sample_id}")
        if point_goal.shape != (2,) or metric_path.ndim != 2 or metric_path.shape[1] != 2:
            raise ValueError(f"invalid HSSD navigation geometry: {sample_id}")
        reached_goal = bool(np.linalg.norm(metric_path[-1] - point_goal) <= 0.05)
        output[split].append(
            _Example(
                depth_run=_DepthRun(
                    source=cache / split / str(cached["file"]),
                    frames=data.observation_frames,
                    name=sample_id,
                ),
                depth_indices=np.arange(4, dtype=np.uint32),
                point_goal=point_goal,
                observation_to_current=observation_to_current,
                observation_valid=np.ones(data.observation_frames, dtype=np.bool_),
                metric_path=metric_path,
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
        [example.depth_indices + depth_offsets[example.depth_run.name] for example in examples]
    ).astype(np.uint32)
    point_goal = np.stack([example.point_goal for example in examples]).astype(np.float32)
    observation_to_current = np.stack(
        [example.observation_to_current for example in examples]
    ).astype(np.float32)
    observation_valid = np.stack(
        [example.observation_valid for example in examples]
    ).astype(np.bool_)

    codec = PlanarBSplineCodec(
        num_control_points=config.trajectory.num_control_points,
        degree=config.trajectory.degree,
        num_path_points=config.trajectory.num_path_points,
    )
    reference_batches = []
    control_batches = []
    for start in range(0, len(examples), 2048):
        paths = [
            torch.from_numpy(example.metric_path)
            for example in examples[start : start + 2048]
        ]
        reference_path, controls = collate_metric_paths(paths, codec)
        reference_batches.append(reference_path.numpy())
        control_batches.append(controls.numpy())
    reference_path = np.concatenate(reference_batches).astype(np.float32)
    control_points = np.concatenate(control_batches).astype(np.float32)
    with torch.no_grad():
        decoded, _, curvature = codec(torch.from_numpy(control_points))
        reference_tensor = torch.from_numpy(reference_path)
        fit_rmse = (decoded - reference_tensor).square().mean((1, 2)).sqrt().numpy()
        maximum_curvature = curvature.abs().amax(1).numpy()

    arrays = {
        "depth_indices": _save_array(split_root, "depth_indices", depth_indices),
        "point_goal": _save_array(split_root, "point_goal", point_goal),
        "observation_to_current": _save_array(
            split_root, "observation_to_current", observation_to_current
        ),
        "observation_valid": _save_array(
            split_root, "observation_valid", observation_valid
        ),
        "control_points": _save_array(split_root, "control_points", control_points),
        "reference_path": _save_array(
            split_root,
            "reference_path",
            reference_path,
        ),
    }
    local_arc = np.asarray([_arc_length(example.metric_path) for example in examples])
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
        "bspline_fit_rmse_m": {
            "mean": float(fit_rmse.mean()),
            "p95": float(np.quantile(fit_rmse, 0.95)),
            "max": float(fit_rmse.max()),
        },
        "bspline_max_curvature_inv_m": {
            "p50": float(np.quantile(maximum_curvature, 0.50)),
            "p95": float(np.quantile(maximum_curvature, 0.95)),
            "p99": float(np.quantile(maximum_curvature, 0.99)),
            "max": float(maximum_curvature.max()),
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
    sand_root: Path,
    hssd_root: Path,
    output_root: Path,
    config: CurveNavConfig,
) -> None:
    config.validate()
    sand = _sand_examples(sand_root.expanduser().resolve(), config)
    hssd = _hssd_examples(hssd_root.expanduser().resolve(), config)
    output_root = output_root.expanduser().resolve()
    building_root = output_root.with_name(output_root.name + ".building")
    if output_root.exists() or building_root.exists():
        raise FileExistsError(f"output dataset already exists: {output_root}")
    building_root.mkdir(parents=True)
    try:
        split_manifests = {}
        for index, split in enumerate(("train", "validation")):
            examples = sand[split] + hssd[split]
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
                "reference_path_source": "fixed_future_expert_waypoints_or_true_goal",
                "reference_endpoint_policy": "implicit_local_subgoal_unless_near_goal",
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
    parser.add_argument("--sand-root", type=Path, required=True)
    parser.add_argument("--hssd-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    compile_policy_dataset(args.sand_root, args.hssd_root, args.output, load_config(args.config))


if __name__ == "__main__":
    main()
