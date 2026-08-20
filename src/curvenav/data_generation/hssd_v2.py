"""Build the CurveNav v2 HSSD pilot from audited v1 route geometry.

The v1 depth images are deliberately ignored.  Every selected pose is rendered
again with the policy camera, and Habitat's native float32 metric depth is
stored without clipping, invalid-value replacement, or integer quantization.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
import hashlib
import json
import math
import multiprocessing
from pathlib import Path
import shutil
import time
from typing import Any

import numpy as np

from curvenav.data_generation.hssd_pilot import HssdPilotConfig, _create_simulator


SCHEMA_VERSION = "curvenav_hssd_v2.0"
DEPTH_CONTRACT_VERSION = "physical_depth_float32_m_v1"
DESCRIPTOR_VERSION = "fixed_expert_prefix_v1"
HISTORY_OFFSETS = (-9, -6, -3, 0)
MIN_TARGET_GAP = 5
MAX_TARGET_GAP = 42
NOMINAL_SPEED_MPS = 0.5


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in records),
        encoding="utf-8",
    )


def source_family(scene_id: str) -> str:
    return scene_id.split("_", 1)[0][:6]


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _camera_contract(config: HssdPilotConfig) -> dict[str, Any]:
    fx = float(config.depth_focal_px)
    fy = fx
    cx = config.depth_width / 2.0
    cy = config.depth_height / 2.0
    body_from_optical = np.array(
        [
            [0.0, 0.0, 1.0, config.camera_forward_m],
            [-1.0, 0.0, 0.0, 0.0],
            [0.0, -1.0, 0.0, config.camera_height_m],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    return {
        "contract_version": DEPTH_CONTRACT_VERSION,
        "sensor": "Habitat-Sim pinhole depth camera",
        "measurement_type": "distance_to_image_plane",
        "depth": {
            "file": "depth_m.npy",
            "dtype": "float32",
            "unit": "m",
            "encoding": "NumPy .npy, C-order [frame, row, column]",
            "invalid": "not isfinite(value) or value <= 0",
            "near_m": 0.01,
            "far_m": 100.0,
        },
        "image": {
            "width": config.depth_width,
            "height": config.depth_height,
            "horizontal_fov_degrees": config.horizontal_fov_degrees,
            "K": [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]],
            "pixel_convention": "OpenCV optical frame; origin at top-left",
        },
        "frames": {
            "body": "x forward, y left, z up",
            "camera_optical": "x right, y down, z forward",
            "body_from_camera_optical": body_from_optical.tolist(),
        },
        "pose_on_body_m": {
            "forward": config.camera_forward_m,
            "left": 0.0,
            "up": config.camera_height_m,
            "pitch_degrees": config.camera_pitch_degrees,
        },
    }


def _sand_to_habitat_positions(world_xyz: np.ndarray) -> np.ndarray:
    return np.column_stack(
        [world_xyz[:, 0], world_xyz[:, 2], world_xyz[:, 1]]
    ).astype(np.float32)


def _set_agent_pose(simulator: Any, position: np.ndarray, heading: float) -> None:
    import habitat_sim

    habitat_yaw = math.atan2(-math.cos(heading), -math.sin(heading))
    state = habitat_sim.AgentState()
    state.position = position
    state.rotation = np.quaternion(
        math.cos(habitat_yaw / 2.0),
        0.0,
        math.sin(habitat_yaw / 2.0),
        0.0,
    )
    simulator.get_agent(0).set_state(state, reset_sensors=True)


def _depth_statistics(samples: list[np.ndarray], invalid_count: int, count: int) -> dict[str, Any]:
    valid_samples = np.concatenate(samples) if samples else np.empty(0, dtype=np.float32)
    if len(valid_samples):
        percentiles = np.percentile(valid_samples, [1, 50, 99]).tolist()
        sampled_min = float(valid_samples.min())
        sampled_max = float(valid_samples.max())
    else:
        percentiles = [None, None, None]
        sampled_min = None
        sampled_max = None
    return {
        "values": count,
        "invalid_values": invalid_count,
        "invalid_fraction": invalid_count / count,
        "sample_stride_pixels": 8,
        "sampled_valid_min_m": sampled_min,
        "sampled_valid_p01_m": percentiles[0],
        "sampled_valid_p50_m": percentiles[1],
        "sampled_valid_p99_m": percentiles[2],
        "sampled_valid_max_m": sampled_max,
    }


def _render_episode(
    simulator: Any,
    source_run: Path,
    episode_dir: Path,
    split: str,
    scene_id: str,
    camera: dict[str, Any],
    gpu_device: int,
) -> dict[str, Any]:
    source_metadata = _read_json(source_run / "metadata.json")
    world_xyz = np.load(source_run / "traj_xyz.npy").astype(np.float32)
    yaw = np.load(source_run / "traj_yaw.npy").astype(np.float32)
    if world_xyz.shape != (len(yaw), 3) or len(yaw) < 2:
        raise ValueError("source trajectory has inconsistent pose arrays")

    episode_dir.mkdir(parents=True)
    world_xy = world_xyz[:, :2]
    segment_m = np.linalg.norm(np.diff(world_xy, axis=0), axis=1)
    cumulative_m = np.concatenate([[0.0], np.cumsum(segment_m)]).astype(np.float32)
    timestamps_s = (cumulative_m / NOMINAL_SPEED_MPS).astype(np.float32)
    np.savez(
        episode_dir / "route.npz",
        world_xyz=world_xyz,
        world_xy=world_xy,
        yaw_rad=yaw,
        segment_length_m=segment_m.astype(np.float32),
        cumulative_arc_length_m=cumulative_m,
        task_goal_world_xy=world_xy[-1],
    )
    np.savez(
        episode_dir / "poses.npz",
        world_xyz=world_xyz,
        yaw_rad=yaw,
        nominal_timestamp_s=timestamps_s,
    )

    height = int(camera["image"]["height"])
    width = int(camera["image"]["width"])
    depth_path = episode_dir / "depth_m.npy"
    depth_stack = np.lib.format.open_memmap(
        depth_path,
        mode="w+",
        dtype=np.float32,
        shape=(len(yaw), height, width),
    )
    positions = _sand_to_habitat_positions(world_xyz)
    invalid_count = 0
    value_count = 0
    valid_samples: list[np.ndarray] = []
    for frame_index, (position, heading) in enumerate(zip(positions, yaw)):
        _set_agent_pose(simulator, position, float(heading))
        depth = np.asarray(simulator.get_sensor_observations()["depth"])
        if depth.dtype != np.float32 or depth.shape != (height, width):
            raise ValueError(f"unexpected native depth {depth.dtype} {depth.shape}")
        depth_stack[frame_index] = depth
        invalid = ~np.isfinite(depth) | (depth <= 0.0)
        invalid_count += int(invalid.sum())
        value_count += int(depth.size)
        sampled = depth[::8, ::8]
        sampled = sampled[np.isfinite(sampled) & (sampled > 0.0)]
        if len(sampled):
            valid_samples.append(sampled.copy())
    depth_stack.flush()
    del depth_stack

    run_id = source_run.name
    episode_id = f"hssd_v2/{split}/{scene_id}/{run_id}"
    metadata = {
        "schema_version": SCHEMA_VERSION,
        "episode_id": episode_id,
        "split": split,
        "scene_id": scene_id,
        "source_family": source_family(scene_id),
        "run_id": run_id,
        "frames": len(yaw),
        "task_goal": {
            "definition": "final pose of the complete expert episode",
            "world_xy": world_xy[-1].astype(float).tolist(),
        },
        "expert_target": {
            "definition": "deterministic executable prefix of this unwarped route",
            "variable_arc_length": True,
            "forced_to_task_goal": False,
            "forced_arc_length_m": None,
        },
        "history": {
            "selection": "spatial trajectory indices",
            "raw_pose_spacing_median_m": float(np.median(segment_m)),
            "index_offsets": list(HISTORY_OFFSETS),
            "nominal_speed_mps": NOMINAL_SPEED_MPS,
            "nominal_timestamps_only": True,
        },
        "camera": camera,
        "depth_statistics": _depth_statistics(valid_samples, invalid_count, value_count),
        "depth_sha256": _sha256_file(depth_path),
        "difficulty": source_metadata["difficulty"],
        "difficulty_tags": source_metadata["difficulty_tags"],
        "source_geometry": {
            "dataset": "HSSD",
            "root": str(source_run.parent.parent.parent.resolve()),
            "episode": str(source_run.resolve()),
            "source_revision": source_metadata.get("source_revision"),
            "metrics": source_metadata["metrics"],
            "reused": ["trajectory geometry", "navigation grid", "difficulty label"],
            "not_reused": ["depth observations", "camera metadata"],
        },
        "render": {"gpu_device": gpu_device},
    }
    _write_json(episode_dir / "metadata.json", metadata)
    return metadata


def _render_scene(
    scene: dict[str, Any],
    source_root: str,
    asset_root: str,
    output_root: str,
    runs_per_scene: int,
    gpu_device: int,
) -> dict[str, Any]:
    source_root_path = Path(source_root)
    asset_root_path = Path(asset_root)
    output_root_path = Path(output_root)
    split = scene["split"]
    scene_id = scene["scene_id"]
    source_scene = source_root_path / scene["source_split"] / f"dataset_hssd_{scene_id}"
    output_scene = output_root_path / split / f"dataset_hssd_{scene_id}"
    output_scene.mkdir(parents=True, exist_ok=False)
    shutil.copy2(source_scene / "navigation_grid.npz", output_scene / "navigation_grid.npz")
    source_bev = source_scene / "bev_routes.png"
    if source_bev.exists():
        shutil.copy2(source_bev, output_scene / "bev_routes_source_geometry.png")

    config = HssdPilotConfig()
    camera = _camera_contract(config)
    valid: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    started = time.monotonic()
    simulator = None
    try:
        simulator = _create_simulator(asset_root_path, scene_id, gpu_device, config)
        source_runs = sorted(source_scene.glob("run_*"))[:runs_per_scene]
        if len(source_runs) != runs_per_scene:
            raise RuntimeError(
                f"{source_scene} has {len(source_runs)} runs, need {runs_per_scene}"
            )
        for source_run in source_runs:
            try:
                valid.append(
                    _render_episode(
                        simulator,
                        source_run,
                        output_scene / source_run.name,
                        split,
                        scene_id,
                        camera,
                        gpu_device,
                    )
                )
            except Exception as error:
                failures.append(
                    {
                        "split": split,
                        "scene_id": scene_id,
                        "run_id": source_run.name,
                        "reason": f"{type(error).__name__}: {error}",
                    }
                )
    except Exception as error:
        completed = {record["run_id"] for record in valid}
        for source_run in sorted(source_scene.glob("run_*"))[:runs_per_scene]:
            if source_run.name not in completed:
                failures.append(
                    {
                        "split": split,
                        "scene_id": scene_id,
                        "run_id": source_run.name,
                        "reason": f"scene render failed: {type(error).__name__}: {error}",
                    }
                )
    finally:
        if simulator is not None:
            simulator.close()
    return {
        "scene_id": scene_id,
        "split": split,
        "valid": valid,
        "failures": failures,
        "elapsed_seconds": time.monotonic() - started,
    }


def _render_scene_batch(
    scenes: list[dict[str, Any]],
    source_root: str,
    asset_root: str,
    output_root: str,
    runs_per_scene: int,
    gpu_device: int,
) -> list[dict[str, Any]]:
    return [
        _render_scene(
            scene,
            source_root,
            asset_root,
            output_root,
            runs_per_scene,
            gpu_device,
        )
        for scene in scenes
    ]


def _local_xy(world_xy: np.ndarray, origin_xy: np.ndarray, yaw: float) -> np.ndarray:
    delta = np.asarray(world_xy, dtype=np.float64) - np.asarray(origin_xy, dtype=np.float64)
    cosine = math.cos(yaw)
    sine = math.sin(yaw)
    return np.array(
        [
            delta[0] * cosine + delta[1] * sine,
            -delta[0] * sine + delta[1] * cosine,
        ],
        dtype=np.float64,
    )


def _target_gap(seed: int, episode_id: str, anchor: int) -> int:
    key = f"{DESCRIPTOR_VERSION}:{seed}:{episode_id}:{anchor}".encode()
    value = int.from_bytes(hashlib.sha256(key).digest()[:8], "little")
    return MIN_TARGET_GAP + value % (MAX_TARGET_GAP - MIN_TARGET_GAP + 1)


def make_sample_records(
    episode: dict[str, Any], route: dict[str, np.ndarray], descriptor_seed: int
) -> list[dict[str, Any]]:
    world_xy = route["world_xy"].astype(np.float64)
    yaw = route["yaw_rad"].astype(np.float64)
    cumulative = route["cumulative_arc_length_m"].astype(np.float64)
    goal = world_xy[-1]
    records: list[dict[str, Any]] = []
    for anchor in range(len(world_xy) - 1):
        target_end = min(anchor + _target_gap(descriptor_seed, episode["episode_id"], anchor), len(world_xy) - 1)
        history = [max(0, anchor + offset) for offset in HISTORY_OFFSETS]
        motion = [[0.0, 0.0, 0.0, 1.0, 0.0]]
        for previous, current in zip(history[:-1], history[1:]):
            if previous == current:
                motion.append([0.0, 0.0, 0.0, 1.0, 0.0])
                continue
            delta_yaw = math.atan2(
                math.sin(yaw[current] - yaw[previous]),
                math.cos(yaw[current] - yaw[previous]),
            )
            local_delta = _local_xy(world_xy[current], world_xy[previous], yaw[previous])
            motion.append(
                [
                    float(local_delta[0]),
                    float(local_delta[1]),
                    math.sin(delta_yaw),
                    math.cos(delta_yaw),
                    1.0,
                ]
            )
        task_goal_local = _local_xy(goal, world_xy[anchor], yaw[anchor])
        endpoint_local = _local_xy(
            world_xy[target_end], world_xy[anchor], yaw[anchor]
        )
        target_arc = float(cumulative[target_end] - cumulative[anchor])
        remaining_arc = float(cumulative[-1] - cumulative[anchor])
        records.append(
            {
                "schema_version": SCHEMA_VERSION,
                "descriptor_version": DESCRIPTOR_VERSION,
                "descriptor_seed": descriptor_seed,
                "sample_id": f"{episode['episode_id']}:{anchor:04d}",
                "episode_id": episode["episode_id"],
                "split": episode["split"],
                "scene_id": episode["scene_id"],
                "anchor_index": anchor,
                "history_indices": history,
                "history_motion_se2": motion,
                "history_motion_fields": ["dx", "dy", "sin_dyaw", "cos_dyaw", "valid"],
                "task_goal_local_xy": task_goal_local.astype(float).tolist(),
                "target_start_index": anchor,
                "target_end_index": target_end,
                "target_num_points": target_end - anchor + 1,
                "target_arc_length_m": target_arc,
                "target_endpoint_local_xy": endpoint_local.astype(float).tolist(),
                "remaining_task_arc_length_m": remaining_arc,
                "near_goal": remaining_arc <= 1.5,
                "alternatives": [],
                "critic_labels": None,
            }
        )
    return records


def _dataset_manifest(
    config: dict[str, Any], output_root: Path, elapsed_seconds: float
) -> dict[str, Any]:
    selected = config["selected_scenes"]
    return {
        "schema_version": SCHEMA_VERSION,
        "created_unix_time": time.time(),
        "output_root": str(output_root),
        "source_root": str(Path(config["source_root"]).resolve()),
        "asset_root": str(Path(config["asset_root"]).resolve()),
        "descriptor_seed": config["descriptor_seed"],
        "target_gap_indices": [MIN_TARGET_GAP, MAX_TARGET_GAP],
        "selected_scenes": selected,
        "source_families": {
            split: sorted({source_family(scene["scene_id"]) for scene in selected if scene["split"] == split})
            for split in ("train", "validation")
        },
        "camera": _camera_contract(HssdPilotConfig()),
        "history_contract": {
            "frames": 4,
            "index_offsets": list(HISTORY_OFFSETS),
            "spatial_sampling_is_primary": True,
            "raw_expert_pose_spacing_m": 0.15,
            "selected_nominal_spacing_m": 0.45,
            "nominal_speed_mps": NOMINAL_SPEED_MPS,
            "motion_label": "adjacent selected-frame relative SE(2): dx,dy,sin(dyaw),cos(dyaw),valid",
        },
        "supervision_contract": {
            "task_goal": "complete episode final goal transformed into current body frame",
            "target": "unwarped deterministic variable-length expert prefix",
            "target_forced_to_task_goal": False,
            "target_forced_arc_length_m": None,
            "training_time_target_randomization": False,
        },
        "critic_schema_reservation": {
            "alternatives": "list of same-state candidates; empty in pilot",
            "candidate_fields": ["path", "collision", "progress", "clearance"],
            "critic_labels": "null in pilot",
        },
        "reuse_policy": {
            "reused": ["v1 HSSD route geometry", "v1 footprint-safe navigation grid"],
            "rerendered": ["all physical depth frames with the v2 camera"],
            "forbidden": ["v1 approximately 89-degree depth images"],
        },
        "elapsed_seconds": elapsed_seconds,
    }


def generate(config_path: Path, output_override: Path | None, workers: int, gpu_devices: list[int]) -> dict[str, Any]:
    config = _read_json(config_path)
    output_root = (output_override or Path(config["output_root"])).expanduser().resolve()
    source_root = Path(config["source_root"]).expanduser().resolve()
    asset_root = Path(config["asset_root"]).expanduser().resolve()
    selected = config["selected_scenes"]
    output_root.mkdir(parents=True, exist_ok=False)
    _write_json(output_root / "config_resolved.json", {**config, "output_root": str(output_root)})

    started = time.monotonic()
    results = []
    worker_count = min(workers, len(gpu_devices), len(selected))
    scene_batches = [selected[index::worker_count] for index in range(worker_count)]
    context = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(max_workers=worker_count, mp_context=context) as executor:
        futures = {
            executor.submit(
                _render_scene_batch,
                batch,
                str(source_root),
                str(asset_root),
                str(output_root),
                int(config["runs_per_scene"]),
                gpu_devices[index],
            ): batch
            for index, batch in enumerate(scene_batches)
        }
        for future in as_completed(futures):
            batch_results = future.result()
            results.extend(batch_results)
            for result in batch_results:
                print(
                    json.dumps(
                        {
                            "scene_id": result["scene_id"],
                            "split": result["split"],
                            "valid": len(result["valid"]),
                            "invalid": len(result["failures"]),
                            "elapsed_seconds": round(result["elapsed_seconds"], 1),
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )

    episodes = sorted(
        [record for result in results for record in result["valid"]],
        key=lambda record: record["episode_id"],
    )
    failures = sorted(
        [record for result in results for record in result["failures"]],
        key=lambda record: (record["split"], record["scene_id"], record["run_id"]),
    )
    samples = []
    for episode in episodes:
        episode_dir = output_root / episode["split"] / f"dataset_hssd_{episode['scene_id']}" / episode["run_id"]
        with np.load(episode_dir / "route.npz") as route_file:
            route = {key: route_file[key] for key in route_file.files}
        samples.extend(make_sample_records(episode, route, int(config["descriptor_seed"])))
    samples.sort(key=lambda record: record["sample_id"])
    _write_jsonl(output_root / "episodes.jsonl", episodes)
    _write_jsonl(output_root / "samples.jsonl", samples)
    _write_jsonl(output_root / "failures.jsonl", failures)
    elapsed = time.monotonic() - started
    manifest = _dataset_manifest(config, output_root, elapsed)
    manifest.update(
        {
            "episodes_valid": len(episodes),
            "episodes_invalid": len(failures),
            "samples": len(samples),
            "frames": sum(int(episode["frames"]) for episode in episodes),
            "scene_timings_seconds": {
                result["scene_id"]: result["elapsed_seconds"] for result in results
            },
        }
    )
    _write_json(output_root / "dataset_manifest.json", manifest)
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return manifest


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--gpu-devices", default="1,3,4,7")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    gpu_devices = [int(value) for value in args.gpu_devices.split(",")]
    generate(args.config.resolve(), args.output_root, args.workers, gpu_devices)


if __name__ == "__main__":
    main()
