"""Generate the single official HSSD policy dataset."""

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

from curvenav.data_generation.assets import HSSD_COMMIT, HSSD_REPOSITORY
from curvenav.data_generation.geometry import (
    ENDPOINT_CLEARANCE_M,
    FRAMES,
    HISTORY_ARCS_M,
    MAX_SNAP_M,
    MIN_CLEARANCE_M,
    NOMINAL_SPEED_MPS,
    PERTURBATION_PROFILE,
    SAFETY_STEP_M,
    SCHEMA,
    TARGET_ARC_M,
    Grid,
    Plan,
    PlanningError,
    candidate_pairs,
    observation_to_current,
    local_prefix,
    local_xy,
    native_route,
    path_length,
    points_at_arc,
    route_turn,
    sample_contract,
    sha256_file,
    snapped_history,
    source_family,
    source_route,
    variant_history,
)


GRID_CELL_M = 0.05
ROBOT_RADIUS_M = 0.25
ROBOT_HEIGHT_M = 0.70
DEPTH_HEIGHT = 360
DEPTH_WIDTH = 640
DEPTH_FOCAL_PX = 1.4 / 1.88 * DEPTH_WIDTH
CAMERA_HEIGHT_M = 0.30
def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(item, sort_keys=True) + "\n" for item in records), encoding="utf-8"
    )


def stable_seed(seed: int, *parts: object) -> int:
    value = ":".join([str(seed), *(str(item) for item in parts)]).encode()
    return int.from_bytes(hashlib.sha256(value).digest()[:8], "little")


def camera_contract() -> dict[str, Any]:
    hfov = math.degrees(2 * math.atan(DEPTH_WIDTH / (2 * DEPTH_FOCAL_PX)))
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
            "width": DEPTH_WIDTH,
            "height": DEPTH_HEIGHT,
            "horizontal_fov_degrees": hfov,
            "K": [
                [DEPTH_FOCAL_PX, 0.0, DEPTH_WIDTH / 2],
                [0.0, DEPTH_FOCAL_PX, DEPTH_HEIGHT / 2],
                [0.0, 0.0, 1.0],
            ],
        },
        "body_from_camera_optical": [
            [0.0, 0.0, 1.0, 0.0],
            [-1.0, 0.0, 0.0, 0.0],
            [0.0, -1.0, 0.0, CAMERA_HEIGHT_M],
            [0.0, 0.0, 0.0, 1.0],
        ],
    }


def create_simulator(asset_root: Path, scene_id: str, gpu: int):
    import habitat_sim

    settings = habitat_sim.SimulatorConfiguration()
    settings.scene_dataset_config_file = str(asset_root / "hssd-hab.scene_dataset_config.json")
    settings.scene_id = str(asset_root / "scenes" / f"{scene_id}.scene_instance.json")
    settings.gpu_device_id, settings.enable_physics = gpu, False
    depth = habitat_sim.CameraSensorSpec()
    depth.uuid, depth.sensor_type = "depth", habitat_sim.SensorType.DEPTH
    depth.resolution, depth.hfov, depth.near, depth.far = (
        [DEPTH_HEIGHT, DEPTH_WIDTH],
        camera_contract()["image"]["horizontal_fov_degrees"],
        0.01,
        100.0,
    )
    depth.position = [0.0, CAMERA_HEIGHT_M, 0.0]
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
    heights = np.asarray([pathfinder.get_random_navigable_point()[1] for _ in range(1024)])
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
    if free.sum() / max(all_free.sum(), 1) < 0.8:
        raise RuntimeError("dominant navigable component is below 80 percent")
    clearance = (
        distance_transform_edt(np.pad(free, 1, constant_values=False))[1:-1, 1:-1] * GRID_CELL_M
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
    state.rotation = np.quaternion(math.cos(habitat_yaw / 2), 0.0, math.sin(habitat_yaw / 2), 0.0)
    simulator.get_agent(0).set_state(state, reset_sensors=True)


def render_depth(simulator: Any, xyz: np.ndarray, yaw: np.ndarray, path: Path) -> dict[str, Any]:
    stack = np.lib.format.open_memmap(
        path, mode="w+", dtype=np.float32, shape=(FRAMES, DEPTH_HEIGHT, DEPTH_WIDTH)
    )
    invalid = 0
    for index, (position, heading) in enumerate(zip(xyz, yaw)):
        set_pose(simulator, position, float(heading))
        depth = np.asarray(simulator.get_sensor_observations()["depth"])
        if depth.dtype != np.float32 or depth.shape != (DEPTH_HEIGHT, DEPTH_WIDTH):
            raise RuntimeError("Habitat returned an unexpected depth tensor")
        stack[index] = depth
        invalid += int((~np.isfinite(depth) | (depth <= 0)).sum())
    stack.flush()
    del stack
    return {
        "sha256": sha256_file(path),
        "invalid_fraction": invalid / (FRAMES * DEPTH_HEIGHT * DEPTH_WIDTH),
    }


def in_band(distance_m: float, band: str, bounds: list[float]) -> bool:
    return (
        bounds[0] <= distance_m <= bounds[1]
        if band == "far"
        else bounds[0] <= distance_m < bounds[1]
    )


def anchor_candidates(
    route: np.ndarray, goal: np.ndarray, bands: dict[str, list[float]], step_m: float, seed: int
) -> dict[str, np.ndarray]:
    arcs = np.arange(-HISTORY_ARCS_M[0], path_length(route) + 1e-8, step_m)
    distances = np.linalg.norm(points_at_arc(route, arcs) - goal, axis=1)
    rng = np.random.default_rng(seed)
    return {
        name: rng.permutation(
            arcs[[in_band(float(distance), name, bounds) for distance in distances]]
        )
        for name, bounds in bands.items()
    }


def precheck_anchor(
    grid: Grid,
    route: np.ndarray,
    goal: np.ndarray,
    arc_m: float,
    band: str,
    bounds: list[float],
    variants: list[dict[str, Any]],
) -> bool:
    for variant in variants:
        history = variant_history(route, arc_m, variant["lateral_m"], variant["yaw_degrees"])
        distance = float(np.linalg.norm(goal - history["world_xy"][-1]))
        if not grid.safe(history["world_xy"]) or not in_band(distance, band, bounds):
            return False
    return True


def generate_sample(
    simulator: Any,
    grid: Grid,
    root: Path,
    scene_dir: Path,
    scene: dict[str, str],
    route_name: str,
    route: np.ndarray,
    source_plan: Plan,
    goal: np.ndarray,
    anchor_id: str,
    anchor_arc_m: float,
    band: str,
    bounds: list[float],
    variant: dict[str, Any],
    floor_m: float,
) -> dict[str, Any]:
    history = snapped_history(
        simulator, grid, route, anchor_arc_m, variant["lateral_m"], variant["yaw_degrees"], floor_m
    )
    anchor_xy, anchor_yaw = history["world_xy"][-1].astype(np.float64), float(history["yaw"][-1])
    global_route, _, snapped_goal, geodesic_m = native_route(simulator, anchor_xy, goal, floor_m)
    goal = snapped_goal[[0, 2]].astype(np.float64)
    target, _ = local_prefix(grid, global_route)
    target_local = np.asarray(
        [local_xy(point, anchor_xy, anchor_yaw) for point in target], dtype=np.float32
    )
    target_local[0] = 0
    goal_local = local_xy(goal, anchor_xy, anchor_yaw).astype(np.float32)
    distance = float(np.linalg.norm(goal_local))
    if not in_band(distance, band, bounds):
        raise PlanningError("realized PointGoal left its assigned band")
    source_route_id = f"{scene['split']}/{scene['scene_id']}/{route_name}"
    anchor_group_id = f"{source_route_id}/{anchor_id}"
    sample_id = f"hssd/{anchor_group_id}/{variant['name']}"
    directory = scene_dir / "source_routes" / route_name / "anchors" / anchor_id / variant["name"]
    work = directory.with_name(directory.name + ".partial")
    work.mkdir(parents=True, exist_ok=False)
    xy, yaw = history["world_xy"].astype(np.float32), history["yaw"].astype(np.float32)
    spacing = np.linalg.norm(np.diff(xy, axis=0), axis=1)
    try:
        depth = render_depth(simulator, history["habitat_xyz"], yaw, work / "depth_m.npy")
        np.savez(
            work / "geometry.npz",
            history_world_xy=xy,
            history_yaw_rad=yaw,
            history_nominal_timestamp_s=np.concatenate(
                [[0.0], np.cumsum(spacing) / NOMINAL_SPEED_MPS]
            ).astype(np.float32),
            observation_to_current=observation_to_current(xy, yaw),
            observation_valid=np.ones(FRAMES, dtype=bool),
            task_goal_world_xy=goal.astype(np.float32),
            task_goal_local_xy=goal_local,
            replanned_global_route_world_xy=global_route.astype(np.float32),
            target_path_world_xy=target.astype(np.float32),
            target_path_local_xy=target_local,
            navmesh_snap_error_m=history["snap_error_m"],
            anchor_arc_m=np.float32(anchor_arc_m),
        )
        turn, turn_sign = route_turn(route, anchor_arc_m)
        record = {
            "schema": SCHEMA,
            "sample_id": sample_id,
            "split": scene["split"],
            "scene_id": scene["scene_id"],
            "source_family": source_family(scene["scene_id"]),
            "source_route_id": source_route_id,
            "anchor_group_id": anchor_group_id,
            "anchor_id": anchor_id,
            "anchor_arc_m": anchor_arc_m,
            "goal_band": band,
            "goal_band_m": bounds,
            "variant": variant["name"],
            "category": "standard" if variant["name"] == "standard" else "perturbed",
            "sample_directory": str(directory.relative_to(root)),
            "task_goal": {
                "world_xy_m": goal.tolist(),
                "local_xy_m": goal_local.tolist(),
                "euclidean_distance_m": distance,
                "geodesic_distance_m": geodesic_m,
            },
            "target": {
                "arc_length_m": path_length(target),
                "arc_cap_m": TARGET_ARC_M,
                "endpoint_local_xy_m": target_local[-1].tolist(),
            },
            "perturbation": {
                "lateral_m": variant["lateral_m"],
                "yaw_degrees": variant["yaw_degrees"],
                "profile": PERTURBATION_PROFILE.tolist(),
                "maximum_snap_m": float(history["snap_error_m"].max()),
            },
            "route_geometry": {
                "turning_sign": turn_sign,
                "signed_turn_degrees": turn,
                "difficulty": source_plan.difficulty,
                "difficulty_tags": source_plan.difficulty_tags,
            },
            "camera": camera_contract(),
            "depth": depth,
        }
        write_json(work / "metadata.json", record)
        temporary = dict(record)
        temporary["sample_directory"] = str(work.relative_to(root))
        violations = sample_contract(root, temporary, grid, camera_contract())["violations"]
        if violations:
            raise RuntimeError(f"sample contract failed: {violations}")
        work.rename(directory)
        return record
    except Exception:
        shutil.rmtree(work, ignore_errors=True)
        raise


def generate_route(
    simulator: Any,
    grid: Grid,
    root: Path,
    scene_dir: Path,
    scene: dict[str, str],
    route_index: int,
    config: dict[str, Any],
    floor_m: float,
    seed: int,
) -> list[dict[str, Any]]:
    route_name = f"source_route_{route_index:02d}"
    pairs = candidate_pairs(
        grid,
        tuple(config["source_endpoint_distance_range_m"]),
        stable_seed(seed, scene["scene_id"], route_name),
        config["candidate_limit"],
    )
    variants, bands, quotas = (
        config["trajectory_variants"],
        config["goal_distance_bands"],
        config["anchors_per_source_route"],
    )
    for pair_index, (start, goal) in enumerate(pairs):
        route_dir = scene_dir / "source_routes" / route_name
        shutil.rmtree(route_dir, ignore_errors=True)
        try:
            _, snapped_start, snapped_goal, _ = native_route(simulator, start, goal, floor_m)
            goal = snapped_goal[[0, 2]].astype(np.float64)
            if grid.clearance(goal[None])[0] < ENDPOINT_CLEARANCE_M:
                raise PlanningError("PointGoal endpoint lacks clearance")
            plan = source_route(
                grid,
                snapped_start[[0, 2]],
                goal,
                MIN_CLEARANCE_M + max(abs(item["lateral_m"]) for item in variants),
            )
            route = plan.path_xy
            candidates = anchor_candidates(
                route,
                goal,
                bands,
                config["anchor_candidate_step_m"],
                stable_seed(seed, scene["scene_id"], route_name, pair_index),
            )
            selected: list[tuple[str, str, float]] = []
            used: list[float] = []
            for band, count in quotas.items():
                accepted = 0
                for arc in candidates[band]:
                    arc = float(arc)
                    if any(
                        abs(arc - other) < config["minimum_anchor_separation_m"] for other in used
                    ) or not precheck_anchor(grid, route, goal, arc, band, bands[band], variants):
                        continue
                    selected.append((f"{band}_{accepted:02d}", band, arc))
                    used.append(arc)
                    accepted += 1
                    if accepted == count:
                        break
                if accepted != count:
                    raise PlanningError("source route lacks the fixed anchor quota")
            records = []
            for anchor_id, band, arc in selected:
                group_dir = route_dir / "anchors" / anchor_id
                try:
                    group = [
                        generate_sample(
                            simulator,
                            grid,
                            root,
                            scene_dir,
                            scene,
                            route_name,
                            route,
                            plan,
                            goal,
                            anchor_id,
                            arc,
                            band,
                            bands[band],
                            variant,
                            floor_m,
                        )
                        for variant in variants
                    ]
                except Exception:
                    shutil.rmtree(group_dir, ignore_errors=True)
                    raise
                records.extend(group)
            route_dir.mkdir(parents=True, exist_ok=True)
            np.savez(
                route_dir / "route.npz",
                route_world_xy=route,
                task_goal_world_xy=goal.astype(np.float32),
                anchor_arc_m=np.asarray(used, dtype=np.float32),
            )
            write_json(
                route_dir / "route.json",
                {
                    "source_route_id": f"{scene['split']}/{scene['scene_id']}/{route_name}",
                    "anchors": [
                        {"anchor_id": item[0], "goal_band": item[1], "anchor_arc_m": item[2]}
                        for item in selected
                    ],
                    "planner": {
                        "difficulty": plan.difficulty,
                        "difficulty_tags": plan.difficulty_tags,
                        "metrics": plan.metrics,
                    },
                },
            )
            return records
        except (PlanningError, RuntimeError, ValueError):
            shutil.rmtree(route_dir, ignore_errors=True)
    raise RuntimeError(f"{scene['scene_id']} {route_name} exhausted endpoint candidates")


def generate_scene(
    scene: dict[str, str], config: dict[str, Any], root: str, gpu: int
) -> list[dict[str, Any]]:
    root_path = Path(root)
    scene_dir = root_path / scene["split"] / f"dataset_hssd_{scene['scene_id']}"
    scene_dir.mkdir(parents=True, exist_ok=False)
    seed = stable_seed(config["seed"], scene["scene_id"]) % (2**31 - 1)
    simulator = create_simulator(Path(config["asset_root"]), scene["scene_id"], gpu)
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
        records = []
        for route_index in range(config["source_routes_per_scene"]):
            records.extend(
                generate_route(
                    simulator, grid, root_path, scene_dir, scene, route_index, config, floor, seed
                )
            )
        return records
    finally:
        simulator.close()


def scene_batch(
    batch: list[dict[str, str]], config: dict[str, Any], root: str, gpu: int
) -> list[dict[str, Any]]:
    output = []
    for scene in batch:
        records = generate_scene(scene, config, root, gpu)
        output.extend(records)
        print(json.dumps({"scene_id": scene["scene_id"], "samples": len(records)}), flush=True)
    return output


def validate_config(config: dict[str, Any]) -> None:
    scenes, variants = config["selected_scenes"], config["trajectory_variants"]
    train, validation = [item for item in scenes if item["split"] == "train"], [
        item for item in scenes if item["split"] == "validation"
    ]
    if len(scenes) != 20 or len(train) != 16 or len(validation) != 4:
        raise ValueError("selected_scenes must contain 16 train and 4 validation scenes")
    if {item["scene_id"] for item in train} & {item["scene_id"] for item in validation}:
        raise ValueError("scene split leakage")
    if {source_family(item["scene_id"]) for item in train} & {
        source_family(item["scene_id"]) for item in validation
    }:
        raise ValueError("source-family split leakage")
    if [item["name"] for item in variants] != [
        "standard",
        "left_mild",
        "right_mild",
        "left_hard",
        "right_hard",
    ]:
        raise ValueError("trajectory_variants do not match the five-variant contract")
    if config["goal_distance_bands"] != {
        "near": [0.5, 3.0],
        "middle": [3.0, 6.0],
        "far": [6.0, 10.0],
    } or config["anchors_per_source_route"] != {"near": 1, "middle": 2, "far": 2}:
        raise ValueError("goal bands or anchor quotas changed")
    if config["source_routes_per_scene"] != 2 or config["expected_samples"] != 1000:
        raise ValueError("dataset size contract changed")
    if config["workers"] < 1 or config["gpu_device"] < 0:
        raise ValueError("workers must be positive and gpu_device must be non-negative")


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
    if output.exists():
        raise FileExistsError(output)
    partial = output.with_name(output.name + f".partial.{os.getpid()}")
    partial.mkdir(parents=True)
    config = {**config, "asset_root": str(asset_root), "output_root": str(output)}
    write_json(partial / "config.json", config)
    workers, gpu = config["workers"], config["gpu_device"]
    batches = [config["selected_scenes"][index::workers] for index in range(workers)]
    context = multiprocessing.get_context("spawn")
    records = []
    with ProcessPoolExecutor(max_workers=workers, mp_context=context) as executor:
        futures = [
            executor.submit(scene_batch, batch, config, str(partial), gpu)
            for index, batch in enumerate(batches)
        ]
        for future in as_completed(futures):
            records.extend(future.result())
    records.sort(key=lambda item: item["sample_id"])
    if len(records) != config["expected_samples"]:
        raise RuntimeError(f"generated {len(records)} samples in {partial}")
    write_jsonl(partial / "samples.jsonl", records)
    write_jsonl(
        partial / "subsets" / "standard.jsonl",
        [item for item in records if item["category"] == "standard"],
    )
    write_jsonl(
        partial / "subsets" / "perturbed.jsonl",
        [item for item in records if item["category"] == "perturbed"],
    )
    manifest = {
        "schema": SCHEMA,
        "seed": config["seed"],
        "asset_root": str(asset_root),
        "asset_repository": HSSD_REPOSITORY,
        "asset_commit": HSSD_COMMIT,
        "output_root": str(output),
        "samples": len(records),
        "scenes": config["selected_scenes"],
        "source_routes": 40,
        "anchor_groups": 200,
        "variants": [item["name"] for item in config["trajectory_variants"]],
        "camera": camera_contract(),
        "task_contract": {
            "pointgoal_distance_m": [0.5, 10.0],
            "target_maximum_arc_m": TARGET_ARC_M,
            "history_frames": FRAMES,
            "history_arc_offsets_m": HISTORY_ARCS_M.tolist(),
            "continuous_safety_step_m": SAFETY_STEP_M,
            "minimum_extra_clearance_m": MIN_CLEARANCE_M,
            "maximum_navmesh_snap_m": MAX_SNAP_M,
        },
    }
    write_json(partial / "dataset_manifest.json", manifest)
    from curvenav.data_generation.audit import audit_dataset

    summary = audit_dataset(partial)
    partial.rename(output)
    return summary


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    args = parser.parse_args(argv)
    print(json.dumps(generate(args.config), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
