"""Render closed-loop world trajectories on the frozen navigation map."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


MAX_RENDERED_PLANS = 160
GLOBAL_MAP_RESOLUTION_M = 0.05


@dataclass(frozen=True)
class NavigationMap:
    free: np.ndarray
    extent: tuple[float, float, float, float]


def _rasterize_navigation_map(navigation_xy: np.ndarray) -> NavigationMap:
    """Rasterize the frozen PLY once; every episode reuses the same exact base."""
    minimum = np.floor(navigation_xy.min(axis=0) / GLOBAL_MAP_RESOLUTION_M)
    maximum = np.ceil(navigation_xy.max(axis=0) / GLOBAL_MAP_RESOLUTION_M)
    x_edges = np.arange(minimum[0], maximum[0] + 1.0) * GLOBAL_MAP_RESOLUTION_M
    y_edges = np.arange(minimum[1], maximum[1] + 1.0) * GLOBAL_MAP_RESOLUTION_M
    counts, _, _ = np.histogram2d(
        navigation_xy[:, 1], navigation_xy[:, 0], bins=(y_edges, x_edges)
    )
    return NavigationMap(
        free=counts > 0,
        extent=(x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]),
    )


def _draw_navigation_map(axis: object, navigation_map: NavigationMap) -> list[object]:
    """Draw the frozen robot-center configuration space as the global base map."""
    from matplotlib.patches import Patch

    from matplotlib.colors import ListedColormap

    axis.imshow(
        navigation_map.free,
        origin="lower",
        extent=navigation_map.extent,
        interpolation="nearest",
        cmap=ListedColormap(["#4a1717", "#e5e7eb"]),
        zorder=1,
    )
    axis.set_xlim(navigation_map.extent[:2])
    axis.set_ylim(navigation_map.extent[2:])
    axis.set_aspect("equal", adjustable="box")
    return [
        Patch(facecolor="#e5e7eb", label="navigable robot-center space"),
        Patch(facecolor="#4a1717", label="obstacle / non-navigable space"),
    ]


def _read_open3d_navigation_ply(path: Path) -> np.ndarray:
    """Read the benchmark's fixed Open3D binary vertex contract."""
    with path.open("rb") as handle:
        header = []
        while True:
            line = handle.readline()
            if not line:
                raise ValueError(f"incomplete PLY header: {path}")
            text = line.decode("ascii").strip()
            header.append(text)
            if text == "end_header":
                break
        if "format binary_little_endian 1.0" not in header:
            raise ValueError(f"navigation PLY must be binary little-endian: {path}")
        vertex_line = next(
            (line for line in header if line.startswith("element vertex ")),
            None,
        )
        if vertex_line is None:
            raise ValueError(f"navigation PLY has no vertices: {path}")
        expected_properties = (
            "property double x",
            "property double y",
            "property double z",
            "property uchar red",
            "property uchar green",
            "property uchar blue",
        )
        properties = tuple(line for line in header if line.startswith("property "))
        if properties != expected_properties:
            raise ValueError(f"unexpected navigation PLY vertex schema: {path}")
        count = int(vertex_line.rsplit(" ", 1)[1])
        vertices = np.fromfile(
            handle,
            dtype=np.dtype([
                ("x", "<f8"), ("y", "<f8"), ("z", "<f8"),
                ("red", "u1"), ("green", "u1"), ("blue", "u1"),
            ]),
            count=count,
        )
    if vertices.shape[0] != count:
        raise ValueError(f"truncated navigation PLY: {path}")
    return np.column_stack((vertices["x"], vertices["y"])).astype(np.float64)


def _yaw_from_xyzw(quaternion: np.ndarray) -> np.ndarray:
    x, y, z, w = np.moveaxis(np.asarray(quaternion, dtype=np.float64), -1, 0)
    return np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def _robot_to_world(
    local_xy: np.ndarray,
    position_world: np.ndarray,
    quaternion_xyzw: np.ndarray,
) -> np.ndarray:
    local = np.asarray(local_xy, dtype=np.float64)
    position = np.asarray(position_world, dtype=np.float64)
    yaw = _yaw_from_xyzw(np.asarray(quaternion_xyzw, dtype=np.float64))
    cosine = np.cos(yaw)
    sine = np.sin(yaw)
    world = np.empty_like(local)
    world[..., 0] = position[..., 0] + cosine * local[..., 0] - sine * local[..., 1]
    world[..., 1] = position[..., 1] + sine * local[..., 0] + cosine * local[..., 1]
    return world


def _plan_indices(count: int) -> np.ndarray:
    if count <= MAX_RENDERED_PLANS:
        return np.arange(count, dtype=np.int64)
    return np.linspace(0, count - 1, MAX_RENDERED_PLANS).round().astype(np.int64)


def _render_episode(
    trace_path: Path,
    navigation_map: NavigationMap,
    output_path: Path,
) -> dict[str, object]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    with np.load(trace_path, allow_pickle=False) as trace:
        episode_idx = int(trace["episode_idx"].item())
        success = bool(trace["success"].item())
        positions = np.asarray(trace["step_robot_position_world_m"], dtype=np.float64)
        terminal = np.asarray(
            trace["terminal_pre_step_robot_position_world_m"], dtype=np.float64
        )[None]
        executed = np.concatenate((positions, terminal), axis=0)[:, :2]
        speeds = np.asarray(trace["step_planar_speed_mps"], dtype=np.float64)
        commands = np.asarray(trace["step_mpc_command"], dtype=np.float64)
        point_goals = np.asarray(trace["step_point_goal_robot_m"], dtype=np.float64)
        quaternions = np.asarray(trace["step_robot_quaternion_xyzw"], dtype=np.float64)
        plan_positions = np.asarray(trace["plan_robot_position_world_m"], dtype=np.float64)
        plan_quaternions = np.asarray(
            trace["plan_robot_quaternion_xyzw"], dtype=np.float64
        )
        plan_local = np.asarray(trace["plan_local_trajectory"], dtype=np.float64)
        plan_lengths = np.asarray(trace["plan_local_trajectory_length"], dtype=np.int64)
        final_goal_distance = float(np.linalg.norm(
            np.asarray(trace["terminal_pre_step_point_goal_robot_m"], dtype=np.float64)[:2]
        ))
        executed_length = float(trace["executed_path_length_m"].item())

    if executed.shape[0] < 2 or point_goals.shape[0] == 0:
        raise ValueError(f"trace has no closed-loop motion: {trace_path}")
    goal_world = _robot_to_world(point_goals[0, :2], positions[0], quaternions[0])
    stalled = (speeds < 0.02) & (commands[:, 0] > 0.2)

    figure, axis = plt.subplots(figsize=(10, 10), constrained_layout=True)
    map_handles = _draw_navigation_map(axis, navigation_map)
    selected = _plan_indices(plan_local.shape[0])
    for index in selected:
        length = int(plan_lengths[index])
        if length < 2:
            continue
        world_plan = _robot_to_world(
            plan_local[index, :length, :2],
            plan_positions[index],
            plan_quaternions[index],
        )
        axis.plot(world_plan[:, 0], world_plan[:, 1], color="#46c7e8", alpha=0.08, lw=0.7)
    axis.plot(
        executed[:, 0], executed[:, 1], color="#2457ff", lw=2.5,
        label="executed trajectory", zorder=4,
    )
    if np.any(stalled):
        axis.scatter(
            positions[stalled, 0], positions[stalled, 1], s=9, c="#ef4444",
            linewidths=0, alpha=0.65, label="stalled while commanded", zorder=5,
        )
    axis.scatter(
        executed[0, 0], executed[0, 1], marker="o", s=90, c="#22c55e",
        edgecolors="white", linewidths=0.8, label="start", zorder=6,
    )
    axis.scatter(
        goal_world[0], goal_world[1], marker="*", s=180, c="#facc15",
        edgecolors="#111827", linewidths=0.8, label="goal", zorder=6,
    )
    axis.scatter(
        executed[-1, 0], executed[-1, 1], marker="x", s=90, c="#fb7185",
        linewidths=2.0, label="terminal", zorder=6,
    )
    axis.set_xlabel("world x (m)")
    axis.set_ylabel("world y (m)")
    axis.set_title(
        f"episode {episode_idx:03d} | {'success' if success else 'timeout'} | "
        f"final goal {final_goal_distance:.2f} m | executed {executed_length:.2f} m"
    )
    handles, labels = axis.get_legend_handles_labels()
    axis.legend(
        map_handles + handles,
        [item.get_label() for item in map_handles] + labels,
        loc="best",
        fontsize=8,
        framealpha=0.9,
    )
    axis.grid(color="white", alpha=0.12, linewidth=0.5)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180)
    plt.close(figure)
    return {
        "episode_idx": episode_idx,
        "success": success,
        "trace": str(trace_path.resolve()),
        "image": str(output_path.resolve()),
        "rendered_plan_count": int(selected.size),
        "total_plan_count": int(plan_local.shape[0]),
        "stalled_step_fraction": float(stalled.mean()),
        "final_goal_distance_m": final_goal_distance,
    }


def _render_scene_overview(
    trace_paths: list[Path],
    navigation_map: NavigationMap,
    output_path: Path,
) -> dict[str, object]:
    """Overlay every executed episode on one immutable global obstacle map."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    figure, axis = plt.subplots(figsize=(10, 10), constrained_layout=True)
    map_handles = _draw_navigation_map(axis, navigation_map)
    successes = 0
    for trace_path in trace_paths:
        with np.load(trace_path, allow_pickle=False) as trace:
            episode_idx = int(trace["episode_idx"].item())
            success = bool(trace["success"].item())
            positions = np.asarray(
                trace["step_robot_position_world_m"], dtype=np.float64
            )
            terminal = np.asarray(
                trace["terminal_pre_step_robot_position_world_m"], dtype=np.float64
            )[None]
            executed = np.concatenate((positions, terminal), axis=0)[:, :2]
            point_goal = np.asarray(
                trace["step_point_goal_robot_m"][0], dtype=np.float64
            )
            quaternion = np.asarray(
                trace["step_robot_quaternion_xyzw"][0], dtype=np.float64
            )
        goal_world = _robot_to_world(point_goal[:2], positions[0], quaternion)
        color = "#16a34a" if success else "#2563eb"
        successes += int(success)
        axis.plot(executed[:, 0], executed[:, 1], color=color, lw=1.8, alpha=0.82, zorder=3)
        axis.scatter(
            executed[0, 0], executed[0, 1], marker="o", s=18,
            c="#22c55e", edgecolors="#052e16", linewidths=0.35, zorder=4,
        )
        axis.scatter(
            executed[-1, 0], executed[-1, 1], marker="x", s=24,
            c="#fb7185", linewidths=0.9, zorder=4,
        )
        axis.text(
            executed[-1, 0],
            executed[-1, 1],
            str(episode_idx),
            color="#fef2f2",
            fontsize=6,
            ha="center",
            va="center",
            zorder=5,
        )
        axis.scatter(
            goal_world[0], goal_world[1], marker="*", s=30, c="#facc15",
            edgecolors="#111827", linewidths=0.3, zorder=4,
        )
    trajectory_handles = [
        Line2D([0], [0], color="#16a34a", lw=2, label="successful execution"),
        Line2D([0], [0], color="#2563eb", lw=2, label="failed execution"),
        Line2D(
            [0], [0], marker="*", color="none", markerfacecolor="#facc15",
            markeredgecolor="#111827", markersize=9, label="PointGoal",
        ),
        Line2D(
            [0], [0], marker="o", color="none", markerfacecolor="#22c55e",
            markeredgecolor="#052e16", markersize=5, label="episode start",
        ),
        Line2D(
            [0], [0], marker="x", color="#fb7185", markersize=6,
            label="episode terminal",
        ),
    ]
    axis.set_xlabel("world x (m)")
    axis.set_ylabel("world y (m)")
    axis.set_title(
        f"closed-loop scene overview | {successes}/{len(trace_paths)} successful"
    )
    axis.legend(
        handles=map_handles + trajectory_handles,
        loc="best",
        fontsize=8,
        framealpha=0.9,
    )
    axis.grid(color="white", alpha=0.12, linewidth=0.5)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180)
    plt.close(figure)
    return {
        "image": str(output_path.resolve()),
        "episode_count": len(trace_paths),
        "success_count": successes,
    }


def render_run(checkpoint_root: Path) -> Path:
    checkpoint_root = checkpoint_root.resolve()
    records: list[dict[str, object]] = []
    scene_overviews: list[dict[str, object]] = []
    for scene_metadata_path in sorted(checkpoint_root.glob("scenes/*/*/run.json")):
        metadata = json.loads(scene_metadata_path.read_text(encoding="utf-8"))
        navigation_file = Path(metadata["navigation_file"])
        navigation_xy = _read_open3d_navigation_ply(navigation_file)
        navigation_map = _rasterize_navigation_map(navigation_xy)
        scene_root = scene_metadata_path.parent
        output_root = scene_root / "trajectory_maps"
        trace_paths = sorted((scene_root / "trajectory_traces").glob("episode-*.npz"))
        for trace_path in trace_paths:
            records.append(_render_episode(
                trace_path,
                navigation_map,
                output_root / f"{trace_path.stem}.png",
            ))
        if trace_paths:
            overview = _render_scene_overview(
                trace_paths,
                navigation_map,
                output_root / "scene-overview.png",
            )
            overview.update({
                "scene": str(scene_root.relative_to(checkpoint_root / "scenes")),
                "navigation_file": str(navigation_file.resolve()),
                "map_resolution_m": GLOBAL_MAP_RESOLUTION_M,
                "map_semantics": (
                    "frozen robot-center configuration space; occupied pixels are "
                    "static obstacle or non-navigable space"
                ),
                "dynamic_obstacle_layer": "unavailable in benchmark trace",
            })
            scene_overviews.append(overview)
    if not records:
        raise ValueError(f"run has no completed trajectory traces: {checkpoint_root}")
    manifest_path = checkpoint_root / "trajectory_maps.json"
    temporary = manifest_path.with_name(manifest_path.name + ".incoming")
    temporary.write_text(
        json.dumps(
            {"episodes": records, "scene_overviews": scene_overviews},
            ensure_ascii=False,
            indent=2,
        ) + "\n",
        encoding="utf-8",
    )
    temporary.replace(manifest_path)
    return manifest_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    args = parser.parse_args()
    print(render_run(args.root))


if __name__ == "__main__":
    main()
