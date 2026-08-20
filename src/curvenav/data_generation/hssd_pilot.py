"""Generate a small, safety-audited HSSD expert-trajectory pilot."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.ndimage import distance_transform_edt, label

from curvenav.data_generation.occupancy import NavigationGrid, resample_polyline
from curvenav.data_generation.planner import (
    PlannerConfig,
    PlanningError,
    PlannedRoute,
    SafeEfficientPlanner,
)


@dataclass(frozen=True)
class HssdPilotConfig:
    grid_cell_m: float = 0.05
    robot_radius_m: float = 0.25
    robot_height_m: float = 0.70
    endpoint_clearance_m: float = 0.30
    minimum_clearance_m: float = 0.10
    preferred_clearance_m: float = 0.30
    minimum_route_m: float = 6.0
    maximum_route_m: float = 15.0
    trajectory_spacing_m: float = 0.15
    depth_width: int = 640
    depth_height: int = 360
    depth_focal_px: float = 1.4 / 1.88 * 640.0
    depth_limit_m: float = 8.0
    camera_height_m: float = 0.30
    camera_forward_m: float = 0.0
    camera_pitch_degrees: float = 0.0

    @property
    def horizontal_fov_degrees(self) -> float:
        return math.degrees(
            2.0 * math.atan(self.depth_width / (2.0 * self.depth_focal_px))
        )


@dataclass(frozen=True)
class SceneGrid:
    navigation: NavigationGrid
    floor_height_m: float
    connected_fraction: float


def _create_simulator(asset_root: Path, scene_id: str, gpu_device: int, config: HssdPilotConfig):
    import habitat_sim

    simulator_config = habitat_sim.SimulatorConfiguration()
    simulator_config.scene_dataset_config_file = str(
        asset_root / "hssd-hab.scene_dataset_config.json"
    )
    simulator_config.scene_id = str(
        asset_root / "scenes" / f"{scene_id}.scene_instance.json"
    )
    simulator_config.gpu_device_id = gpu_device
    simulator_config.enable_physics = False

    depth = habitat_sim.CameraSensorSpec()
    depth.uuid = "depth"
    depth.sensor_type = habitat_sim.SensorType.DEPTH
    depth.resolution = [config.depth_height, config.depth_width]
    depth.hfov = config.horizontal_fov_degrees
    depth.near = 0.01
    depth.far = 100.0
    depth.position = [0.0, config.camera_height_m, -config.camera_forward_m]
    depth.orientation = [math.radians(config.camera_pitch_degrees), 0.0, 0.0]

    agent = habitat_sim.agent.AgentConfiguration()
    agent.height = config.robot_height_m
    agent.radius = config.robot_radius_m
    agent.sensor_specifications = [depth]
    return habitat_sim.Simulator(
        habitat_sim.Configuration(simulator_config, [agent])
    )


def _build_scene_grid(simulator, config: HssdPilotConfig, seed: int) -> SceneGrid:
    import habitat_sim

    settings = habitat_sim.NavMeshSettings()
    settings.set_defaults()
    settings.agent_radius = config.robot_radius_m
    settings.agent_height = config.robot_height_m
    settings.cell_size = config.grid_cell_m
    settings.cell_height = config.grid_cell_m
    if not simulator.recompute_navmesh(simulator.pathfinder, settings):
        raise RuntimeError("Habitat failed to build the HSSD navmesh")

    pathfinder = simulator.pathfinder
    pathfinder.seed(seed)
    heights = np.asarray(
        [pathfinder.get_random_navigable_point()[1] for _ in range(1024)],
        dtype=np.float64,
    )
    height_bins = np.round(heights / 0.1).astype(np.int32)
    dominant_bin = int(np.bincount(height_bins - height_bins.min()).argmax())
    dominant_bin += int(height_bins.min())
    floor_height = float(np.median(heights[height_bins == dominant_bin]))

    topdown = pathfinder.get_topdown_view(config.grid_cell_m, floor_height)
    all_free = topdown.T.copy()
    components, count = label(
        all_free, np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=np.uint8)
    )
    if count == 0:
        raise RuntimeError("HSSD scene has no navigable component")
    sizes = np.bincount(components.ravel())
    sizes[0] = 0
    free = components == int(sizes.argmax())
    connected_fraction = float(free.sum() / max(all_free.sum(), 1))
    if connected_fraction < 0.8:
        raise RuntimeError(
            f"largest navigable component is only {connected_fraction:.1%}"
        )

    padded = np.pad(free, 1, constant_values=False)
    clearance = distance_transform_edt(padded)[1:-1, 1:-1] * config.grid_cell_m
    clearance[~free] = 0.0
    lower, _ = pathfinder.get_bounds()
    # Habitat maps pixel [z, x] to lower_bound + index * meters_per_pixel.
    # NavigationGrid addresses cell centers, hence the half-cell origin shift.
    origin = np.array(
        [lower[0] - config.grid_cell_m / 2, lower[2] - config.grid_cell_m / 2],
        dtype=np.float64,
    )
    return SceneGrid(
        navigation=NavigationGrid(
            free=free,
            clearance_m=clearance.astype(np.float32),
            origin_xy=origin,
            cell_size_m=config.grid_cell_m,
        ),
        floor_height_m=floor_height,
        connected_fraction=connected_fraction,
    )


def _sample_routes(
    scene_grid: SceneGrid,
    count: int,
    seed: int,
    config: HssdPilotConfig,
) -> list[PlannedRoute]:
    grid = scene_grid.navigation
    eligible = np.argwhere(grid.clearance_m >= config.endpoint_clearance_m)
    if len(eligible) < 2:
        raise RuntimeError("scene has too few safe endpoint cells")
    planner = SafeEfficientPlanner(
        grid,
        PlannerConfig(
            preferred_clearance_m=config.preferred_clearance_m,
            minimum_clearance_m=config.minimum_clearance_m,
            maximum_safe_detour_ratio=1.2,
            output_spacing_m=config.grid_cell_m,
        ),
    )
    rng = np.random.default_rng(seed)
    candidates: list[PlannedRoute] = []
    target_candidates = max(12, count)
    for _ in range(500):
        cells = eligible[rng.integers(len(eligible), size=2)]
        start = grid.grid_to_world(cells[0])
        goal = grid.grid_to_world(cells[1])
        direct_distance = float(np.linalg.norm(goal - start))
        if not config.minimum_route_m * 0.8 <= direct_distance <= config.maximum_route_m:
            continue
        try:
            route = planner.plan(start, goal)
        except (PlanningError, ValueError):
            continue
        if not config.minimum_route_m <= route.metrics.length_m <= config.maximum_route_m:
            continue
        if any(
            np.linalg.norm(route.start_xy - other.start_xy) < 1.0
            and np.linalg.norm(route.goal_xy - other.goal_xy) < 1.0
            for other in candidates
        ):
            continue
        candidates.append(route)
        if len(candidates) >= target_candidates:
            break
    if len(candidates) < count:
        raise RuntimeError(f"only planned {len(candidates)} valid routes, need {count}")

    selected: list[PlannedRoute] = []
    buckets = {
        difficulty: sorted(
            (route for route in candidates if route.difficulty == difficulty),
            key=lambda route: (
                route.metrics.risk_density,
                route.metrics.reference_length_ratio,
            ),
        )
        for difficulty in ("narrow", "detour", "turning", "open")
    }
    while len(selected) < count and any(buckets.values()):
        for bucket in buckets.values():
            if bucket and len(selected) < count:
                selected.append(bucket.pop(0))
    selected_ids = {id(route) for route in selected}
    remaining = sorted(
        (route for route in candidates if id(route) not in selected_ids),
        key=lambda route: (
            route.metrics.risk_density,
            route.metrics.reference_length_ratio,
        ),
    )
    selected.extend(remaining[: count - len(selected)])
    return selected


def _trajectory_positions(
    simulator, route: PlannedRoute, floor_height_m: float, config: HssdPilotConfig
) -> np.ndarray:
    path = resample_polyline(route.path_xy, config.trajectory_spacing_m)
    positions = []
    for x, z in path:
        snapped = np.asarray(
            simulator.pathfinder.snap_point([float(x), floor_height_m, float(z)]),
            dtype=np.float64,
        )
        if not np.isfinite(snapped).all():
            raise RuntimeError("trajectory point could not be snapped to the navmesh")
        positions.append(snapped)
    return np.asarray(positions, dtype=np.float32)


def _trajectory_yaw(positions: np.ndarray) -> np.ndarray:
    direction = np.diff(positions[:, [0, 2]], axis=0)
    yaw = np.arctan2(direction[:, 1], direction[:, 0])
    return np.concatenate([yaw, yaw[-1:]]).astype(np.float32)


def _render_run(
    simulator,
    run_dir: Path,
    positions: np.ndarray,
    yaw: np.ndarray,
    config: HssdPilotConfig,
) -> dict[str, float]:
    import habitat_sim

    depth_dir = run_dir / "depth"
    depth_dir.mkdir(parents=True)
    sand_xyz = np.column_stack(
        [positions[:, 0], positions[:, 2], positions[:, 1]]
    ).astype(np.float32)
    np.save(run_dir / "traj_xyz.npy", sand_xyz)
    np.save(run_dir / "traj_yaw.npy", yaw)
    np.save(run_dir / "traj_pitch.npy", np.zeros(len(yaw), dtype=np.float32))

    invalid_fractions = []
    clipped_fractions = []
    for index, (position, heading) in enumerate(zip(positions, yaw)):
        habitat_yaw = math.atan2(-math.cos(float(heading)), -math.sin(float(heading)))
        state = habitat_sim.AgentState()
        state.position = position
        state.rotation = np.quaternion(
            math.cos(habitat_yaw / 2.0),
            0.0,
            math.sin(habitat_yaw / 2.0),
            0.0,
        )
        simulator.get_agent(0).set_state(state, reset_sensors=True)
        depth = np.asarray(
            simulator.get_sensor_observations()["depth"], dtype=np.float32
        )
        invalid = ~np.isfinite(depth) | (depth <= 0.0)
        invalid_fractions.append(float(invalid.mean()))
        depth = np.nan_to_num(
            depth,
            nan=config.depth_limit_m,
            posinf=config.depth_limit_m,
            neginf=config.depth_limit_m,
        )
        depth[depth <= 0.0] = config.depth_limit_m
        clipped_fractions.append(float((depth >= config.depth_limit_m).mean()))
        depth_mm = np.rint(
            np.clip(depth, 0.0, config.depth_limit_m) * 1000.0
        ).astype(np.uint16)
        Image.fromarray(depth_mm).save(depth_dir / f"depth_{index:04d}.png")
    return {
        "invalid_depth_fraction_max": max(invalid_fractions),
        "depth_at_limit_fraction_mean": float(np.mean(clipped_fractions)),
    }


def _plot_bev(
    scene_id: str,
    scene_grid: SceneGrid,
    routes: list[PlannedRoute],
    output_path: Path,
) -> None:
    import matplotlib.pyplot as plt

    grid = scene_grid.navigation
    x0, z0 = grid.origin_xy + grid.cell_size_m / 2
    x1 = x0 + grid.free.shape[0] * grid.cell_size_m
    z1 = z0 + grid.free.shape[1] * grid.cell_size_m
    figure, axis = plt.subplots(figsize=(9, 8), constrained_layout=True)
    axis.imshow(
        (~grid.free).T,
        origin="lower",
        extent=[x0, x1, z0, z1],
        cmap="gray_r",
        interpolation="nearest",
        alpha=0.88,
    )
    colors = plt.cm.tab10(np.linspace(0.0, 1.0, len(routes)))
    for index, (route, color) in enumerate(zip(routes, colors)):
        axis.plot(
            route.path_xy[:, 0],
            route.path_xy[:, 1],
            color=color,
            linewidth=2.4,
            label=f"{index}: {route.difficulty}, {route.metrics.length_m:.1f} m",
        )
        axis.scatter(*route.start_xy, color="#14a44d", s=55, zorder=4)
        axis.scatter(*route.goal_xy, color="#dc3545", marker="*", s=95, zorder=4)
    axis.set_title(f"HSSD {scene_id}: black = obstacle / non-navigable")
    axis.set_xlabel("world x [m]")
    axis.set_ylabel("world z [m]")
    axis.set_aspect("equal")
    axis.legend(loc="upper right", fontsize=8)
    figure.savefig(output_path, dpi=170)
    plt.close(figure)


def _plot_depth_montage(dataset_dir: Path, output_path: Path) -> None:
    import matplotlib.pyplot as plt

    run_dirs = sorted(dataset_dir.glob("run_*"))
    figure, axes = plt.subplots(
        len(run_dirs), 3, figsize=(11, 3.0 * len(run_dirs)), squeeze=False
    )
    for row, run_dir in enumerate(run_dirs):
        frames = sorted((run_dir / "depth").glob("*.png"))
        indices = [0, len(frames) // 2, len(frames) - 1]
        for column, frame_index in enumerate(indices):
            depth = np.asarray(Image.open(frames[frame_index]), dtype=np.float32) / 1000.0
            axes[row, column].imshow(depth, cmap="magma_r", vmin=0.0, vmax=8.0)
            axes[row, column].set_title(f"{run_dir.name}, frame {frame_index}")
            axes[row, column].axis("off")
    figure.tight_layout()
    figure.savefig(output_path, dpi=150)
    plt.close(figure)


def generate_scene(
    asset_root: Path,
    output_root: Path,
    scene_id: str,
    routes_per_scene: int,
    gpu_device: int,
    seed: int,
    config: HssdPilotConfig,
) -> list[dict[str, object]]:
    simulator = _create_simulator(asset_root, scene_id, gpu_device, config)
    try:
        scene_grid = _build_scene_grid(simulator, config, seed)
        routes = _sample_routes(scene_grid, routes_per_scene, seed, config)
        dataset_dir = output_root / f"dataset_hssd_{scene_id}"
        dataset_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            dataset_dir / "navigation_grid.npz",
            free=scene_grid.navigation.free,
            clearance_m=scene_grid.navigation.clearance_m,
            origin_xy=scene_grid.navigation.origin_xy,
            cell_size_m=scene_grid.navigation.cell_size_m,
            floor_height_m=scene_grid.floor_height_m,
        )
        records = []
        for run_index, route in enumerate(routes):
            run_dir = dataset_dir / f"run_{run_index:04d}"
            positions = _trajectory_positions(
                simulator, route, scene_grid.floor_height_m, config
            )
            yaw = _trajectory_yaw(positions)
            depth_metrics = _render_run(simulator, run_dir, positions, yaw, config)
            record = {
                "scene_id": scene_id,
                "run_id": run_dir.name,
                "frames": len(positions),
                "difficulty": route.difficulty,
                "difficulty_tags": list(route.difficulty_tags),
                "metrics": asdict(route.metrics),
                "depth": depth_metrics,
                "camera": {
                    "width": config.depth_width,
                    "height": config.depth_height,
                    "focal_px": config.depth_focal_px,
                    "hfov_degrees": config.horizontal_fov_degrees,
                    "height_m": config.camera_height_m,
                    "forward_m": config.camera_forward_m,
                    "pitch_degrees": config.camera_pitch_degrees,
                },
            }
            (run_dir / "metadata.json").write_text(
                json.dumps(record, indent=2, sort_keys=True), encoding="utf-8"
            )
            records.append(record)
        _plot_bev(scene_id, scene_grid, routes, dataset_dir / "bev_routes.png")
        _plot_depth_montage(dataset_dir, dataset_dir / "depth_montage.png")
        return records
    finally:
        simulator.close()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("asset_root", type=Path)
    parser.add_argument("output_root", type=Path)
    parser.add_argument("--scene-id", action="append", required=True)
    parser.add_argument("--routes-per-scene", type=int, default=4)
    parser.add_argument("--gpu-device", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    asset_root = args.asset_root.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    config = HssdPilotConfig()
    records = []
    for scene_offset, scene_id in enumerate(args.scene_id):
        records.extend(
            generate_scene(
                asset_root,
                output_root,
                scene_id,
                args.routes_per_scene,
                args.gpu_device,
                args.seed + scene_offset,
                config,
            )
        )
    (output_root / "manifest.jsonl").write_text(
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in records),
        encoding="utf-8",
    )
    summary = {
        "config": asdict(config),
        "scenes": args.scene_id,
        "runs": len(records),
        "frames": sum(int(record["frames"]) for record in records),
        "maximum_invalid_depth_fraction": max(
            float(record["depth"]["invalid_depth_fraction_max"]) for record in records
        ),
    }
    (output_root / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
