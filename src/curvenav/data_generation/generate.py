"""Generate 500 continuous, unperturbed HSSD expert routes."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
import math
import multiprocessing
import os
from pathlib import Path
import shutil
from typing import Any

import numpy as np
from scipy.ndimage import distance_transform_edt, label

from curvenav.data_generation.assets import (
    HSSD_COMMIT,
    HSSD_REPOSITORY,
    validate_assets,
)
from curvenav.data_generation.geometry import (
    ENDPOINT_CLEARANCE_M,
    MAX_SNAP_M,
    MIN_CLEARANCE_M,
    SAFETY_STEP_M,
    Grid,
    PlanningError,
    candidate_pairs,
    headings,
    native_route,
    path_length,
    points_at_arc,
    sha256_file,
    source_family,
    source_route,
)
from curvenav.physical import ROBOT_HEIGHT_M, ROBOT_RADIUS_M


GRID_CELL_M = 0.05
SCHEMA = "curvenav_hssd_expert_routes"


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(item, sort_keys=True) + "\n" for item in records),
        encoding="utf-8",
    )


def stable_seed(seed: int, *parts: object) -> int:
    value = ":".join([str(seed), *(str(item) for item in parts)]).encode()
    return int.from_bytes(hashlib.sha256(value).digest()[:8], "little")


def camera_contract(camera: dict[str, float | int]) -> dict[str, Any]:
    width = int(camera["image_width"])
    height = int(camera["image_height"])
    focal_x = float(camera["focal_x_px"])
    focal_y = float(camera["focal_y_px"])
    forward_offset = float(camera["forward_offset_m"])
    camera_height = float(camera["height_m"])
    pitch = math.radians(float(camera["downward_pitch_degrees"]))
    sine, cosine = math.sin(pitch), math.cos(pitch)
    horizontal_fov = math.degrees(2 * math.atan(width / (2 * focal_x)))
    return {
        "sensor": "Habitat-Sim pinhole depth camera",
        "measurement_type": "distance_to_image_plane",
        "depth": {
            "dtype": "float32",
            "unit": "m",
            "invalid": "not isfinite(value) or value <= 0",
            "near_m": 0.01,
            "far_m": 100.0,
        },
        "image": {
            "width": width,
            "height": height,
            "horizontal_fov_degrees": horizontal_fov,
            "K": [
                [focal_x, 0.0, width / 2],
                [0.0, focal_y, height / 2],
                [0.0, 0.0, 1.0],
            ],
        },
        "body_from_camera_optical": [
            [0.0, -sine, cosine, forward_offset],
            [-1.0, 0.0, 0.0, 0.0],
            [0.0, -cosine, -sine, camera_height],
            [0.0, 0.0, 0.0, 1.0],
        ],
    }


def create_simulator(
    asset_root: Path,
    scene_id: str,
    gpu: int,
    camera: dict[str, float | int],
):
    import habitat_sim

    settings = habitat_sim.SimulatorConfiguration()
    settings.scene_dataset_config_file = str(
        asset_root / "hssd-hab.scene_dataset_config.json"
    )
    settings.scene_id = str(asset_root / "scenes" / f"{scene_id}.scene_instance.json")
    settings.gpu_device_id, settings.enable_physics = gpu, False
    depth = habitat_sim.CameraSensorSpec()
    depth.uuid, depth.sensor_type = "depth", habitat_sim.SensorType.DEPTH
    depth.resolution, depth.hfov, depth.near, depth.far = (
        [int(camera["image_height"]), int(camera["image_width"])],
        camera_contract(camera)["image"]["horizontal_fov_degrees"],
        0.01,
        100.0,
    )
    depth.position = [
        0.0,
        float(camera["height_m"]),
        -float(camera["forward_offset_m"]),
    ]
    depth.orientation = [
        -math.radians(float(camera["downward_pitch_degrees"])),
        0.0,
        0.0,
    ]
    agent = habitat_sim.agent.AgentConfiguration()
    agent.height, agent.radius, agent.sensor_specifications = (
        ROBOT_HEIGHT_M,
        ROBOT_RADIUS_M,
        [depth],
    )
    return habitat_sim.Simulator(habitat_sim.Configuration(settings, [agent]))


def build_grid(simulator: Any, seed: int) -> tuple[Grid, float]:
    import habitat_sim

    settings = habitat_sim.NavMeshSettings()
    settings.set_defaults()
    settings.agent_radius, settings.agent_height = ROBOT_RADIUS_M, ROBOT_HEIGHT_M
    settings.cell_size = settings.cell_height = GRID_CELL_M
    if not simulator.recompute_navmesh(simulator.pathfinder, settings):
        raise RuntimeError("Habitat navmesh construction failed")
    pathfinder = simulator.pathfinder
    pathfinder.seed(seed)
    heights = np.asarray(
        [pathfinder.get_random_navigable_point()[1] for _ in range(1024)]
    )
    bins = np.round(heights / 0.1).astype(np.int32)
    floor_bin = int(np.bincount(bins - bins.min()).argmax() + bins.min())
    floor = float(np.median(heights[bins == floor_bin]))
    all_free = pathfinder.get_topdown_view(GRID_CELL_M, floor).T.copy()
    components, count = label(
        all_free, np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=np.uint8)
    )
    if not count:
        raise RuntimeError("scene has no navigable component")
    sizes = np.bincount(components.ravel())
    sizes[0] = 0
    free = components == int(sizes.argmax())
    if free.sum() / all_free.sum() < 0.8:
        raise RuntimeError("dominant navigable component is below 80 percent")
    clearance = (
        distance_transform_edt(np.pad(free, 1, constant_values=False))[1:-1, 1:-1]
        * GRID_CELL_M
    )
    clearance[~free] = 0.0
    lower, _ = pathfinder.get_bounds()
    origin = np.array([lower[0] - GRID_CELL_M / 2, lower[2] - GRID_CELL_M / 2])
    return Grid(free, clearance.astype(np.float32), origin, GRID_CELL_M), floor


def set_pose(simulator: Any, position: np.ndarray, heading: float) -> None:
    import habitat_sim

    habitat_yaw = math.atan2(-math.cos(heading), -math.sin(heading))
    state = habitat_sim.AgentState()
    state.position = position
    state.rotation = np.quaternion(
        math.cos(habitat_yaw / 2), 0.0, math.sin(habitat_yaw / 2), 0.0
    )
    simulator.get_agent(0).set_state(state, reset_sensors=True)


def render_depth(
    simulator: Any,
    xyz: np.ndarray,
    yaw: np.ndarray,
    path: Path,
    image_height: int,
    image_width: int,
) -> tuple[str, float]:
    frames = len(xyz)
    stack = np.lib.format.open_memmap(
        path,
        mode="w+",
        dtype=np.float32,
        shape=(frames, image_height, image_width),
    )
    invalid = 0
    for index, (position, heading) in enumerate(zip(xyz, yaw, strict=True)):
        set_pose(simulator, position, float(heading))
        depth = np.asarray(simulator.get_sensor_observations()["depth"])
        if depth.dtype != np.float32 or depth.shape != (image_height, image_width):
            raise RuntimeError("Habitat returned an unexpected depth tensor")
        stack[index] = depth
        invalid += int((~np.isfinite(depth) | (depth <= 0)).sum())
    stack.flush()
    del stack
    return sha256_file(path), invalid / (frames * image_height * image_width)


def sampled_route(
    simulator: Any,
    planned_xy: np.ndarray,
    floor_m: float,
    spacing_m: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    length = path_length(planned_xy)
    arcs = np.linspace(0.0, length, math.ceil(length / spacing_m) + 1)
    requested_xy = points_at_arc(planned_xy, arcs)
    requested_xyz = np.column_stack(
        (requested_xy[:, 0], np.full(len(requested_xy), floor_m), requested_xy[:, 1])
    )
    habitat_xyz = np.asarray(
        [simulator.pathfinder.snap_point(point) for point in requested_xyz],
        dtype=np.float32,
    )
    if not np.isfinite(habitat_xyz).all():
        raise PlanningError("route contains an invalid Habitat pose")
    route_xy = habitat_xyz[:, [0, 2]].astype(np.float32)
    snap_error = np.linalg.norm(route_xy - requested_xy, axis=1).astype(np.float32)
    if float(snap_error.max()) > MAX_SNAP_M:
        raise PlanningError("route pose snap exceeds the physical contract")
    route_arcs = np.concatenate(
        ([0.0], np.cumsum(np.linalg.norm(np.diff(route_xy, axis=0), axis=1)))
    )
    route_yaw = headings(route_xy, route_arcs)
    return route_xy, route_yaw.astype(np.float32), habitat_xyz, snap_error


def generate_route(
    simulator: Any,
    grid: Grid,
    root: Path,
    scene_dir: Path,
    scene: dict[str, str],
    route_index: int,
    distance_band: str,
    distance_range_m: list[float],
    config: dict[str, Any],
    floor_m: float,
    seed: int,
) -> dict[str, Any]:
    route_name = f"run_{route_index + 1:04d}"
    pairs = candidate_pairs(
        grid,
        tuple(distance_range_m),
        stable_seed(seed, scene["scene_id"], route_name),
        config["candidate_limit"],
    )
    directory = scene_dir / route_name
    for start, goal in pairs:
        try:
            _, snapped_start, snapped_goal, _ = native_route(
                simulator, start, goal, floor_m
            )
            goal_xy = snapped_goal[[0, 2]].astype(np.float64)
            if grid.clearance(goal_xy[None])[0] < ENDPOINT_CLEARANCE_M:
                raise PlanningError("PointGoal endpoint lacks clearance")
            plan = source_route(
                grid,
                snapped_start[[0, 2]],
                goal_xy,
                MIN_CLEARANCE_M,
            )
            route_xy, route_yaw, habitat_xyz, snap_error = sampled_route(
                simulator,
                plan.path_xy,
                floor_m,
                config["route_sample_spacing_m"],
            )
            if not grid.safe(route_xy):
                raise PlanningError("sampled expert route violates clearance")
            endpoint_distance = float(np.linalg.norm(route_xy[-1] - route_xy[0]))
            if not distance_range_m[0] <= endpoint_distance < distance_range_m[1]:
                raise PlanningError("sampled endpoint left its distance band")
            directory.mkdir(parents=True, exist_ok=False)
            np.save(directory / "traj_xy.npy", route_xy, allow_pickle=False)
            np.save(directory / "traj_yaw.npy", route_yaw, allow_pickle=False)
            depth_sha, invalid_fraction = render_depth(
                simulator,
                habitat_xyz,
                route_yaw,
                directory / "depth_m.npy",
                int(config["camera"]["image_height"]),
                int(config["camera"]["image_width"]),
            )
            route_id = f"{scene['split']}/dataset_hssd_{scene['scene_id']}/{route_name}"
            record = {
                "route_id": route_id,
                "route_directory": str(directory.relative_to(root)),
                "split": scene["split"],
                "scene_id": scene["scene_id"],
                "source_family": source_family(scene["scene_id"]),
                "frames": len(route_xy),
                "route_arc_m": path_length(route_xy),
                "endpoint_distance_m": endpoint_distance,
                "endpoint_distance_band": distance_band,
                "endpoint_distance_range_m": distance_range_m,
                "maximum_navmesh_snap_m": float(snap_error.max()),
                "planner": {
                    "difficulty": plan.difficulty,
                    "difficulty_tags": plan.difficulty_tags,
                    "metrics": plan.metrics,
                },
                "depth": {
                    "sha256": depth_sha,
                    "invalid_fraction": invalid_fraction,
                },
            }
            write_json(directory / "metadata.json", record)
            return record
        except PlanningError:
            shutil.rmtree(directory, ignore_errors=True)
    raise RuntimeError(
        f"{scene['scene_id']} {route_name} exhausted endpoint candidates"
    )


def route_bands(config: dict[str, Any]) -> list[str]:
    return [
        name
        for name, count in config["routes_per_scene_by_distance"].items()
        for _ in range(count)
    ]


def generate_scene(
    scene: dict[str, str], config: dict[str, Any], root: str, gpu: int
) -> list[dict[str, Any]]:
    root_path = Path(root)
    scene_dir = root_path / scene["split"] / f"dataset_hssd_{scene['scene_id']}"
    scene_dir.mkdir(parents=True, exist_ok=False)
    seed = stable_seed(config["seed"], scene["scene_id"]) % (2**31 - 1)
    simulator = create_simulator(
        Path(config["asset_root"]),
        scene["scene_id"],
        gpu,
        config["camera"],
    )
    try:
        grid, floor = build_grid(simulator, seed)
        np.savez_compressed(
            scene_dir / "navigation_grid.npz",
            free=grid.free,
            clearance_m=grid.clearance_m,
            origin_xy=grid.origin_xy,
            cell_size_m=grid.cell_size_m,
            floor_height_m=floor,
        )
        records = [
            generate_route(
                simulator,
                grid,
                root_path,
                scene_dir,
                scene,
                route_index,
                band,
                config["endpoint_distance_bands_m"][band],
                config,
                floor,
                seed,
            )
            for route_index, band in enumerate(route_bands(config))
        ]
        print(
            json.dumps(
                {
                    "scene_id": scene["scene_id"],
                    "routes": len(records),
                    "frames": sum(item["frames"] for item in records),
                }
            ),
            flush=True,
        )
        return records
    finally:
        simulator.close()


def scene_batch(
    batch: list[dict[str, str]], config: dict[str, Any], root: str, gpu: int
) -> list[dict[str, Any]]:
    records = []
    for scene in batch:
        records.extend(generate_scene(scene, config, root, gpu))
    return records


def validate_config(config: dict[str, Any]) -> None:
    scenes = config["selected_scenes"]
    train = [item for item in scenes if item["split"] == "train"]
    validation = [item for item in scenes if item["split"] == "validation"]
    if len(scenes) != 20 or len(train) != 16 or len(validation) != 4:
        raise ValueError(
            "selected_scenes must contain 16 train and 4 validation scenes"
        )
    if {item["scene_id"] for item in train} & {item["scene_id"] for item in validation}:
        raise ValueError("scene split leakage")
    if {source_family(item["scene_id"]) for item in train} & {
        source_family(item["scene_id"]) for item in validation
    }:
        raise ValueError("source-family split leakage")
    bands = config["endpoint_distance_bands_m"]
    quotas = config["routes_per_scene_by_distance"]
    if list(bands) != ["near", "middle", "far"] or set(quotas) != set(bands):
        raise ValueError("endpoint distance bands must be near, middle, and far")
    if sum(quotas.values()) != config["routes_per_scene"]:
        raise ValueError("route distance quotas must equal routes_per_scene")
    if config["routes_per_scene"] * len(scenes) != config["expected_routes"]:
        raise ValueError("route count contract does not match selected scenes")
    if not math.isclose(config["route_sample_spacing_m"], 0.15):
        raise ValueError(
            "expert route sampling must match the 0.15 m training contract"
        )
    if config["workers"] < 1 or config["gpu_device"] < 0:
        raise ValueError("workers must be positive and gpu_device must be non-negative")
    camera = config["camera"]
    numeric_camera = (
        camera["image_width"],
        camera["image_height"],
        camera["focal_x_px"],
        camera["focal_y_px"],
        camera["height_m"],
    )
    if not all(math.isfinite(float(value)) and float(value) > 0 for value in numeric_camera):
        raise ValueError("camera dimensions, focal lengths and height must be positive")


def generate(config_path: Path) -> dict[str, Any]:
    config_path = config_path.resolve()
    config = read_json(config_path)
    validate_config(config)
    project_root = config_path.parent.parent
    output = (project_root / config["output_root"]).resolve()
    asset_root = (project_root / config["asset_root"]).resolve()
    asset_manifest = read_json(asset_root / "download_manifest.json")
    selected_scene_ids = {item["scene_id"] for item in config["selected_scenes"]}
    if (
        asset_manifest.get("repository") != HSSD_REPOSITORY
        or asset_manifest.get("commit") != HSSD_COMMIT
        or not selected_scene_ids.issubset(asset_manifest.get("scenes", []))
    ):
        raise ValueError("HSSD assets do not match the frozen source contract")
    validate_assets(asset_root, sorted(selected_scene_ids))
    if output.exists():
        raise FileExistsError(output)
    partial = output.with_name(output.name + f".partial.{os.getpid()}")
    partial.mkdir(parents=True)
    try:
        config = {**config, "asset_root": str(asset_root), "output_root": str(output)}
        write_json(partial / "config.json", config)
        batches = [
            config["selected_scenes"][index :: config["workers"]]
            for index in range(config["workers"])
        ]
        context = multiprocessing.get_context("spawn")
        records = []
        with ProcessPoolExecutor(
            max_workers=config["workers"], mp_context=context
        ) as executor:
            futures = [
                executor.submit(
                    scene_batch, batch, config, str(partial), config["gpu_device"]
                )
                for batch in batches
            ]
            for future in as_completed(futures):
                records.extend(future.result())
        records.sort(key=lambda item: item["route_id"])
        if len(records) != config["expected_routes"]:
            raise RuntimeError(f"generated {len(records)} routes in {partial}")
        write_jsonl(partial / "routes.jsonl", records)
        manifest = {
            "schema": SCHEMA,
            "seed": config["seed"],
            "asset_root": str(asset_root),
            "asset_repository": HSSD_REPOSITORY,
            "asset_commit": HSSD_COMMIT,
            "output_root": str(output),
            "routes": len(records),
            "frames": sum(item["frames"] for item in records),
            "scenes": config["selected_scenes"],
            "camera": camera_contract(config["camera"]),
            "route_contract": {
                "unperturbed": True,
                "sample_spacing_m": config["route_sample_spacing_m"],
                "continuous_safety_step_m": SAFETY_STEP_M,
                "minimum_clearance_m": MIN_CLEARANCE_M,
                "maximum_navmesh_snap_m": MAX_SNAP_M,
                "endpoint_distance_bands_m": config["endpoint_distance_bands_m"],
                "routes_per_scene_by_distance": config["routes_per_scene_by_distance"],
            },
        }
        write_json(partial / "dataset_manifest.json", manifest)
        from curvenav.data_generation.audit import audit_dataset

        summary = audit_dataset(partial)
        partial.rename(output)
        return summary
    except BaseException:
        shutil.rmtree(partial, ignore_errors=True)
        raise


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    args = parser.parse_args(argv)
    print(json.dumps(generate(args.config), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
