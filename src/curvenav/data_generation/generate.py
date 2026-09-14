"""Generate scene-balanced depth directly in the policy storage format."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import fcntl
import hashlib
import json
import math
import multiprocessing
import os
import shutil
import time
from pathlib import Path
from typing import Any

import numpy as np
from scipy.ndimage import distance_transform_edt, label

from curvenav.data.depth import (
    BENCHMARK_INTRINSICS,
    depth_camera_contract,
    preprocess_depth,
)
from curvenav.config_io import load_config
from curvenav.data.history import OBSERVATION_PERIOD_S
from curvenav.config import DataConfig
from curvenav.data.contracts import expert_navigation_geometry_contract
from curvenav.data_generation.assets import (
    validate_assets,
)
from curvenav.data_generation.geometry import (
    MIN_CLEARANCE_M,
    SAFETY_STEP_M,
    Grid,
    PlanningError,
    candidate_pairs,
    Plan,
    timed_route,
    path_length,
    sha256_file,
    source_route,
)
from curvenav.physical import (
    MAXIMUM_TRAVERSABLE_HEIGHT_M,
    MAXIMUM_TRAVERSABLE_SLOPE_DEGREES,
    ROBOT_COLLISION_HEIGHT_M,
    ROBOT_BASE_HEIGHT_ABOVE_GROUND_M,
    ROBOT_FOOTPRINT_RADIUS_M,
)


GRID_CELL_M = 0.05
SCHEMA = "curvenav_policy_depth_routes_v5"


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
    scene_dataset: str,
):
    import habitat_sim

    settings = habitat_sim.SimulatorConfiguration()
    settings.scene_dataset_config_file = str(
        asset_root / scene_dataset
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
        ROBOT_COLLISION_HEIGHT_M,
        ROBOT_FOOTPRINT_RADIUS_M,
        [depth],
    )
    return habitat_sim.Simulator(habitat_sim.Configuration(settings, [agent]))


def configure_navmesh_settings(settings: Any) -> None:
    """Set the one robot configuration-space contract used by expert planning."""
    settings.set_defaults()
    settings.agent_radius = ROBOT_FOOTPRINT_RADIUS_M
    settings.agent_height = ROBOT_COLLISION_HEIGHT_M
    settings.agent_max_climb = MAXIMUM_TRAVERSABLE_HEIGHT_M
    settings.agent_max_slope = MAXIMUM_TRAVERSABLE_SLOPE_DEGREES
    settings.cell_size = GRID_CELL_M
    settings.cell_height = GRID_CELL_M
    settings.include_static_objects = True


def build_grid(simulator: Any, seed: int) -> tuple[Grid, float]:
    import habitat_sim

    settings = habitat_sim.NavMeshSettings()
    configure_navmesh_settings(settings)
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
    clearance = (
        distance_transform_edt(np.pad(free, 1, constant_values=False))[1:-1, 1:-1]
        * GRID_CELL_M
    )
    clearance[~free] = 0.0
    lower, _ = pathfinder.get_bounds()
    origin = np.array([lower[0] - GRID_CELL_M / 2, lower[2] - GRID_CELL_M / 2])
    return Grid(free, clearance.astype(np.float32), origin, GRID_CELL_M), floor


def base_position_from_navmesh(position: np.ndarray) -> np.ndarray:
    """Lift a floor contact point to the benchmark Dingo base-link origin."""
    base_position = np.asarray(position, dtype=np.float32).copy()
    base_position[1] += ROBOT_BASE_HEIGHT_ABOVE_GROUND_M
    return base_position


def set_pose(simulator: Any, position: np.ndarray, heading: float) -> None:
    import habitat_sim

    habitat_yaw = math.atan2(-math.cos(heading), -math.sin(heading))
    state = habitat_sim.AgentState()
    state.position = base_position_from_navmesh(position)
    state.rotation = np.quaternion(
        math.cos(habitat_yaw / 2), 0.0, math.sin(habitat_yaw / 2), 0.0
    )
    simulator.get_agent(0).set_state(state, reset_sensors=True)


def render_depth(
    simulator: Any,
    xyz: np.ndarray,
    yaw: np.ndarray,
    path: Path,
    data: DataConfig,
) -> tuple[str, float]:
    frames = len(xyz)
    if len(yaw) != frames:
        raise ValueError("route positions and headings must have equal length")
    invalid = 0
    image_height, image_width = data.image_height, data.image_width
    # Sequential writes avoid remote mmap page faults on shared filesystems.
    with path.open("wb") as depth_file:
        np.lib.format.write_array_header_1_0(depth_file, {
            "descr": np.lib.format.dtype_to_descr(np.dtype(np.float16)),
            "fortran_order": False,
            "shape": (frames, image_height, image_width),
        })
        for position, heading in zip(xyz, yaw):
            set_pose(simulator, position, float(heading))
            depth = np.asarray(simulator.get_sensor_observations()["depth"])
            if depth.dtype != np.float32 or depth.shape != (BENCHMARK_INTRINSICS.height, BENCHMARK_INTRINSICS.width):
                raise RuntimeError("Habitat returned an unexpected depth tensor")
            depth, _ = preprocess_depth(
                depth, source_intrinsics=BENCHMARK_INTRINSICS,
                maximum_m=data.max_depth_m, height=image_height, width=image_width,
            )
            packed = depth.astype(np.float16)
            depth_file.write(packed.tobytes())
            invalid += int((packed == 0).sum())
    return sha256_file(path), invalid / (frames * image_height * image_width)


def sampled_route(
    simulator: Any,
    plan: Plan,
    floor_m: float,
    period_s: float,
    speed_m_s: float,
    angular_speed_rad_s: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    route_xy, route_yaw, controls = timed_route(
        plan.curve, period_s, speed_m_s, angular_speed_rad_s
    )
    requested_xyz = np.column_stack(
        (route_xy[:, 0], np.full(len(route_xy), floor_m), route_xy[:, 1])
    )
    snapped = np.asarray(
        [simulator.pathfinder.snap_point(point) for point in requested_xyz],
        dtype=np.float64,
    )
    if not np.isfinite(snapped).all():
        raise PlanningError("route contains an invalid Habitat pose")
    snap_error = np.linalg.norm(snapped[:, [0, 2]] - route_xy, axis=1)
    # Horizontal snapping changes the analytic trajectory and its heading.
    # Only floating-point/navmesh storage error is permitted; height comes from the floor.
    tolerance = 8 * np.finfo(np.float32).eps * max(1., abs(route_xy).max())
    if snap_error.max() > tolerance:
        raise PlanningError("expert curve leaves the Habitat navigation surface")
    requested_xyz[:, 1] = snapped[:, 1]
    return route_xy, route_yaw, requested_xyz, snap_error, controls


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
    data: DataConfig,
) -> dict[str, Any]:
    route_name = f"{distance_band}_{route_index + 1:04d}"
    pairs = candidate_pairs(
        grid,
        tuple(distance_range_m),
        stable_seed(seed, scene["scene_id"], route_name),
        config["candidate_limit"],
    )
    destination = scene_dir / route_name
    if (destination / "metadata.json").is_file():
        record = read_json(destination / "metadata.json")
        if record["source_family"] != scene["source_family"]:
            raise ValueError("cached source family differs from generation config")
        return record
    directory = scene_dir / (route_name + ".partial")
    if directory.exists():
        shutil.rmtree(directory)
    # Independent of the path optimizer and fixed across endpoint retries:
    # rejecting a difficult route must not silently resample an easier heading.
    start_yaw = float(np.random.default_rng(
        stable_seed(seed, scene["scene_id"], route_name + ":heading")
    ).uniform(-math.pi, math.pi))
    for candidate_index, (start, goal) in enumerate(pairs):
        try:
            plan = source_route(
                grid,
                start,
                goal,
                start_yaw,
            )
            route_xy, route_yaw, habitat_xyz, snap_error, controls = sampled_route(
                simulator,
                plan,
                floor_m,
                config["observation_period_s"],
                config["expert_speed_m_s"],
                config["expert_angular_speed_rad_s"],
            )
            if not grid.safe(route_xy):
                raise PlanningError("sampled expert route violates clearance")
            endpoint_distance = float(np.linalg.norm(route_xy[-1] - route_xy[0]))
            if not distance_range_m[0] <= endpoint_distance < distance_range_m[1]:
                raise PlanningError("sampled endpoint left its distance band")
        except PlanningError:
            continue
        directory.mkdir(parents=True, exist_ok=False)
        np.save(directory / "expert_controls.npy", controls.astype(np.float32), allow_pickle=False)
        np.save(directory / "traj_xy.npy", route_xy.astype(np.float32), allow_pickle=False)
        np.save(directory / "traj_yaw.npy", route_yaw.astype(np.float32), allow_pickle=False)
        poses = np.broadcast_to(np.eye(4), (len(route_xy), 4, 4)).copy()
        c, sn = np.cos(route_yaw), np.sin(route_yaw)
        poses[:, 0, 0], poses[:, 0, 1] = c, sn
        poses[:, 1, 0], poses[:, 1, 1] = -sn, c
        base = np.stack([base_position_from_navmesh(p) for p in habitat_xyz])
        poses[:, :3, 3] = base[:, [0, 2, 1]] * [1, -1, 1]
        np.save(
            directory / "body_to_world.npy",
            poses.astype(np.float32),
            allow_pickle=False,
        )
        # Uniform observation clock along the sampled kinematic route, not wall time.
        route_time = (
            np.arange(len(route_xy), dtype=np.float64)
            * config["observation_period_s"]
        )
        np.save(
            directory / "timestamps.npy",
            route_time.astype(np.float64),
            allow_pickle=False,
        )

        depth_sha, invalid_fraction = render_depth(
            simulator,
            habitat_xyz,
            route_yaw,
            directory / "depth.npy",
            data,
        )
        route_id = str(destination.relative_to(root))
        record = {
            "route_id": route_id,
            "route_directory": route_id,
            "split": scene["split"],
            "scene_id": scene["scene_id"],
            "source": config["source"],
            "source_family": scene["source_family"],
            "frames": len(route_xy),
            "initial_yaw_rad": start_yaw,
            "timestamp_semantics": "uniform_sensor_clock",
            "motion_model": "forward_differential_drive_curve_clock",
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
        directory.rename(destination)
        return record
    raise RuntimeError(
        f"{scene['scene_id']} {route_name} exhausted endpoint candidates"
    )


def route_quota(config: dict[str, Any], scene: dict[str, str]) -> dict[str, int]:
    """Apportion exact split totals evenly over scenes, then over distance bands."""
    scene_ids = sorted(s["scene_id"] for s in config["selected_scenes"] if s["split"] == scene["split"])
    count, extra = divmod(config["routes_per_split"][scene["split"]], len(scene_ids))
    count += scene_ids.index(scene["scene_id"]) < extra
    bands = ("near", "middle", "far")
    weights = config["distance_band_weights"]
    denominator = sum(weights.values())
    fractions = {band: divmod(count * weights[band], denominator) for band in bands}
    quota = {band: fractions[band][0] for band in bands}
    remainder = sorted(bands, key=lambda band: -fractions[band][1])
    for band in remainder[:count - sum(quota.values())]:
        quota[band] += 1
    return quota


def endpoint_distance_ranges(grid: Grid, quantiles: dict[str, list[float]], seed: int) -> dict[str, list[float]]:
    """Stratify by a deterministic estimate of this scene's endpoint distribution.

    The 8,192 pairs estimate quantiles, not a finite set of training tasks.
    Actual route candidates are sampled independently. Zero and the bounding-box
    diagonal include the complete distance support, including small scenes.
    """
    eligible = np.argwhere(grid.free & (grid.clearance_m + 1e-9 >= MIN_CLEARANCE_M))
    if len(eligible) < 2:
        raise PlanningError("scene has fewer than two safe endpoints")
    rng = np.random.default_rng(stable_seed(seed, "distance_quantiles"))
    pairs = eligible[rng.integers(len(eligible), size=(8192, 2))]
    distances = np.linalg.norm(pairs[:, 1] - pairs[:, 0], axis=1) * grid.cell_size_m
    distances = distances[distances > 0]
    cuts = [quantiles[band][0] for band in ("near", "middle", "far")] + [1.]
    edges = np.quantile(distances, cuts)
    edges[0] = 0.
    edges[-1] = np.nextafter(np.linalg.norm(np.ptp(eligible, axis=0)) * grid.cell_size_m, np.inf)
    if np.any(np.diff(edges) <= 0):
        raise PlanningError("scene has insufficient endpoint distance variation")
    return {band: edges[i:i+2].tolist() for i, band in enumerate(("near", "middle", "far"))}


def generate_scene(
    scene: dict[str, str], config: dict[str, Any], root: str, gpu: int,
    data: DataConfig,
) -> list[dict[str, Any]]:
    root_path = Path(root)
    scene_dir = root_path / scene["split"] / f"{config['source']}_{scene['scene_id']}"
    scene_dir.mkdir(parents=True, exist_ok=True)
    quota = route_quota(config, scene)
    expected = [scene_dir / f"{band}_{i + 1:04d}" / "metadata.json"
                for band, count in quota.items()
                for i in range(count)]
    if all(p.is_file() for p in expected):
        records = [read_json(p) for p in expected]
        if any(r["source_family"] != scene["source_family"] for r in records):
            raise ValueError("cached source family differs from generation config")
        return records
    seed = stable_seed(config["seed"], scene["scene_id"]) % (2**31 - 1)
    simulator = create_simulator(
        Path(config["asset_root"]),
        scene["scene_id"],
        gpu,
        config["camera"],
        config["scene_dataset"],
    )
    try:
        grid_path = scene_dir / "navigation_grid.npz"
        navmesh_path = scene_dir / "scene.navmesh"
        if grid_path.exists() and navmesh_path.exists():
            if not simulator.pathfinder.load_nav_mesh(str(navmesh_path)):
                raise RuntimeError(f"invalid cached navmesh: {navmesh_path}")
            grid = Grid.load(grid_path)
            with np.load(grid_path) as stored:
                floor = float(stored["floor_height_m"])
        else:
            grid, floor = build_grid(simulator, seed)
            simulator.pathfinder.save_nav_mesh(str(navmesh_path))
            temporary = grid_path.with_name("navigation_grid.partial.npz")
            np.savez_compressed(temporary, free=grid.free, clearance_m=grid.clearance_m,
                               origin_xy=grid.origin_xy, cell_size_m=grid.cell_size_m,
                               floor_height_m=floor)
            os.replace(temporary, grid_path)
        distance_ranges = endpoint_distance_ranges(grid, config["endpoint_distance_quantiles"], seed)
        records = []
        for band, count in quota.items():
            for route_index in range(count):
                cached = (scene_dir / f"{band}_{route_index + 1:04d}" / "metadata.json").is_file()
                started = time.monotonic()
                record = generate_route(
                    simulator, grid, root_path, scene_dir, scene, route_index,
                    band, distance_ranges[band], config, floor, seed, data,
                )
                records.append(record)
                print(json.dumps({"route_id": record["route_id"], "frames": record["frames"],
                                  "cached": cached, "seconds": round(time.monotonic() - started, 3),
                                  "scene_completed": len(records), "scene_target": sum(quota.values())}),
                      flush=True)
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


def validate_config(config: dict[str, Any], data: DataConfig) -> None:
    if not math.isfinite(config["expert_angular_speed_rad_s"]) or config["expert_angular_speed_rad_s"] <= 0:
        raise ValueError("expert_angular_speed_rad_s must be positive")
    if not math.isfinite(config["expert_speed_m_s"]) or config["expert_speed_m_s"] <= 0:
        raise ValueError("expert_speed_m_s must be positive")
    if config["source"] not in ("hssd", "grscenes"):
        raise ValueError("unsupported scene source")
    scenes = config["selected_scenes"]
    train = [item for item in scenes if item["split"] == "train"]
    validation = [item for item in scenes if item["split"] == "validation"]
    if not train or not validation or len(train) + len(validation) != len(scenes):
        raise ValueError("selected_scenes must contain train and validation scenes")
    if len({item["scene_id"] for item in scenes}) != len(scenes):
        raise ValueError("selected_scenes must be unique")
    if {item["source_family"] for item in train} & {
        item["source_family"] for item in validation
    }:
        raise ValueError("source-family split leakage")
    bands = config["endpoint_distance_quantiles"]
    weights = config["distance_band_weights"]
    if set(bands) != {"near", "middle", "far"} or set(weights) != set(bands):
        raise ValueError("endpoint distance bands must be near, middle, and far")
    distance_ranges = [bands[band] for band in ("near", "middle", "far")]
    if not all(
        len(bounds) == 2
        and all(math.isfinite(float(value)) for value in bounds)
        and 0 <= bounds[0] < bounds[1] <= 1
        for bounds in distance_ranges
    ):
        raise ValueError("distance quantiles must be finite increasing intervals in [0, 1]")
    if distance_ranges[0][0] != 0 or distance_ranges[-1][1] != 1:
        raise ValueError("distance quantiles must cover all endpoint distances")
    if not all(
        math.isclose(left[1], right[0])
        for left, right in zip(distance_ranges[:-1], distance_ranges[1:])
    ):
        raise ValueError("endpoint distance ranges must be contiguous")
    if any(type(weight) is not int or weight < 1 for weight in weights.values()):
        raise ValueError("distance band weights must be positive integers")
    totals = config["routes_per_split"]
    if set(totals) != {"train", "validation"} or any(type(n) is not int or n < 1 for n in totals.values()):
        raise ValueError("routes_per_split must give positive train and validation totals")
    if any(min(route_quota(config, scene).values()) < 1 for scene in scenes):
        raise ValueError("route totals must cover every scene and distance band")
    if config["observation_period_s"] != OBSERVATION_PERIOD_S:
        raise ValueError("expert observations must match the benchmark 10 Hz camera")
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
    if not all(
        math.isfinite(float(value)) and float(value) > 0 for value in numeric_camera
    ):
        raise ValueError("camera dimensions, focal lengths and height must be positive")
    expected_camera = {
        "image_width": BENCHMARK_INTRINSICS.width,
        "image_height": BENCHMARK_INTRINSICS.height,
        "focal_x_px": BENCHMARK_INTRINSICS.fx,
        "focal_y_px": BENCHMARK_INTRINSICS.fy,
        "forward_offset_m": data.camera_forward_offset_m,
        "height_m": data.camera_height_m,
        "downward_pitch_degrees": data.camera_downward_pitch_degrees,
    }
    if camera != expected_camera:
        raise ValueError("generation camera must match the benchmark Dingo policy")


def _generate(config_path: Path) -> dict[str, Any]:
    config_path = config_path.resolve()
    config = read_json(config_path)
    project_root = config_path.parent.parent
    policy_config = load_config(project_root / config["policy_config"])
    data = policy_config.data
    validate_config(config, data)
    output = (project_root / config["output_root"]).resolve()
    asset_root = (project_root / config["asset_root"]).resolve()
    asset_manifest = read_json(asset_root / "download_manifest.json")
    selected_scene_ids = {item["scene_id"] for item in config["selected_scenes"]}
    if (asset_manifest["repository"] != config["asset_repository"]
            or asset_manifest["commit"] != config["asset_commit"]
            or not selected_scene_ids.issubset(asset_manifest["scenes"])):
        raise ValueError("scene assets do not match the frozen source contract")
    if config["source"] == "hssd":
        validate_assets(asset_root, sorted(selected_scene_ids))
    else:
        from curvenav.data_generation.grscenes import validate_prepared_assets
        validate_prepared_assets(asset_root, config["selected_scenes"])
    if output.exists():
        raise FileExistsError(output)
    packed_output = (project_root / config["packed_output_root"]).resolve()
    if packed_output.exists():
        raise FileExistsError(packed_output)
    # Only observation/planning semantics enter the key. More routes or workers
    # reuse the exact same per-band endpoint seeds and completed render files.
    semantics = {key: config[key] for key in (
        "seed", "source", "asset_repository", "asset_commit", "candidate_limit",
        "camera", "observation_period_s", "expert_speed_m_s",
        "expert_angular_speed_rad_s", "endpoint_distance_quantiles",
    )}
    semantics["observation"] = depth_camera_contract(data)
    semantics["navigation_geometry"] = expert_navigation_geometry_contract()
    semantics["code"] = {
        name: sha256_file(Path(__file__).parent.parent / name)
        for name in ("data_generation/generate.py", "data_generation/geometry.py",
                     "data/depth.py", "physical.py")
    }
    semantics["asset_converter"] = asset_manifest.get("converter_sha256")
    cache_key = hashlib.sha256(json.dumps(semantics, sort_keys=True).encode()).hexdigest()
    cache = (project_root / config["route_cache_root"]).resolve() / cache_key
    cache.mkdir(parents=True, exist_ok=True)
    write_json(cache / "contract.json", semantics)
    partial = output.with_name(output.name + f".partial.{os.getpid()}")
    partial.mkdir(parents=True)
    config = {**config, "asset_root": str(asset_root), "output_root": str(output)}
    write_json(partial / "config.json", config)
    context = multiprocessing.get_context("spawn")
    records = []
    with ProcessPoolExecutor(
        max_workers=config["workers"], mp_context=context
    ) as executor:
        futures = [
            executor.submit(
                generate_scene,
                scene,
                config,
                str(cache),
                config["gpu_device"],
                data,
            )
            for scene in config["selected_scenes"]
        ]
        for future in as_completed(futures):
            records.extend(future.result())
    # Publish only requested routes. Hard links share immutable depth bytes;
    # the cache remains useful when extending scene/route quotas later.
    for record in records:
        relative = Path(record["route_directory"])
        destination = partial / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        grid = destination.parent / "navigation_grid.npz"
        if not grid.exists():
            os.link(cache / relative.parent / grid.name, grid)
        shutil.copytree(cache / relative, destination, copy_function=os.link)
    records.sort(key=lambda item: item["route_id"])
    if len(records) != sum(config["routes_per_split"].values()):
        raise RuntimeError(f"generated {len(records)} routes in {partial}")
    write_jsonl(partial / "routes.jsonl", records)
    manifest = {
        "schema": SCHEMA,
        "seed": config["seed"],
        "asset_root": str(asset_root),
        "asset_repository": asset_manifest["repository"],
        "asset_commit": asset_manifest["commit"],
        "output_root": str(output),
        "routes": len(records),
        "frames": sum(item["frames"] for item in records),
        "scenes": config["selected_scenes"],
        "camera": camera_contract(config["camera"]),
        "observation": depth_camera_contract(data),
        "route_contract": {
            "navigation_geometry": expert_navigation_geometry_contract(),
            "unperturbed": True,
            "initial_heading": "uniform_world_yaw_fixed_before_endpoint_retries",
            "observation_period_s": config["observation_period_s"],
            "expert_speed_m_s": config["expert_speed_m_s"],
            "expert_angular_speed_rad_s": config["expert_angular_speed_rad_s"],
            "continuous_safety_step_m": SAFETY_STEP_M,
            "minimum_clearance_m": MIN_CLEARANCE_M,
            "horizontal_snap": "float32_storage_tolerance_only",
            "endpoint_distance_quantiles": config["endpoint_distance_quantiles"],
            "routes_per_scene_by_distance": {scene["scene_id"]: route_quota(config, scene)
                                             for scene in config["selected_scenes"]},
        },
    }
    write_json(partial / "dataset_manifest.json", manifest)
    from curvenav.data_generation.audit import audit_dataset

    summary = audit_dataset(partial)
    partial.rename(output)
    from curvenav.data.prepare import compile_policy_dataset
    compile_policy_dataset(output, project_root / config["packed_output_root"], policy_config)
    return summary


def generate(config_path: Path) -> dict[str, Any]:
    config_path = config_path.resolve()
    config = read_json(config_path)
    cache = (config_path.parent.parent / config["route_cache_root"]).resolve()
    cache.mkdir(parents=True, exist_ok=True)
    with (cache / ".generation.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return _generate(config_path)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    args = parser.parse_args(argv)
    print(json.dumps(generate(args.config), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
