"""Render closed-loop world trajectories on the frozen navigation map."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from curvenav.evaluation.metrics import controller_tracking_metrics
from curvenav.physical import PATH_CONFIGURATION_QUERY_SPACING_M


MAX_RENDERED_PLANS = 160
GLOBAL_MAP_RESOLUTION_M = 0.05


@dataclass(frozen=True)
class NavigationMap:
    free: np.ndarray
    extent: tuple[float, float, float, float]


def _rasterize_navigation_map(navigation_xy: np.ndarray) -> NavigationMap:
    """Bin navigation samples for display; empty bins do not imply obstacles."""
    lattice = np.rint(
        np.asarray(navigation_xy, dtype=np.float64) / GLOBAL_MAP_RESOLUTION_M
    ).astype(np.int64)
    minimum = lattice.min(axis=0)
    maximum = lattice.max(axis=0)
    free = np.zeros(
        (maximum[1] - minimum[1] + 1, maximum[0] - minimum[0] + 1),
        dtype=np.bool_,
    )
    free[lattice[:, 1] - minimum[1], lattice[:, 0] - minimum[0]] = True
    half_cell = 0.5 * GLOBAL_MAP_RESOLUTION_M
    return NavigationMap(
        free=free,
        extent=(
            minimum[0] * GLOBAL_MAP_RESOLUTION_M - half_cell,
            maximum[0] * GLOBAL_MAP_RESOLUTION_M + half_cell,
            minimum[1] * GLOBAL_MAP_RESOLUTION_M - half_cell,
            maximum[1] * GLOBAL_MAP_RESOLUTION_M + half_cell,
        ),
    )


def _draw_navigation_map(axis: object, navigation_map: NavigationMap) -> list[object]:
    """Draw navigation sample coverage, without inferring obstacle occupancy."""
    from matplotlib.patches import Patch

    from matplotlib.colors import ListedColormap

    axis.imshow(
        navigation_map.free,
        origin="lower",
        extent=navigation_map.extent,
        interpolation="nearest",
        cmap=ListedColormap(["#d7dce2", "#fbfcfe"]),
        zorder=1,
    )
    axis.set_xlim(navigation_map.extent[:2])
    axis.set_ylim(navigation_map.extent[2:])
    axis.set_aspect("equal", adjustable="box")
    return [
        Patch(facecolor="#fbfcfe", edgecolor="#94a3b8", label="bins containing navigation samples"),
        Patch(facecolor="#d7dce2", edgecolor="#94a3b8", label="no navigation sample (unknown)"),
    ]


def _focus_map_view(
    axis: object,
    navigation_map: NavigationMap,
    points: np.ndarray,
    *,
    padding_m: float = 1.0,
    minimum_span_m: float = 4.0,
) -> None:
    """Crop a diagnostic map to the task while retaining physical scale."""
    points = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    if points.size == 0 or not np.isfinite(points).all():
        raise ValueError("map focus points must be finite and non-empty")
    lower = points.min(axis=0) - padding_m
    upper = points.max(axis=0) + padding_m
    span = upper - lower
    expansion = np.maximum(minimum_span_m - span, 0.0) * 0.5
    lower -= expansion
    upper += expansion
    x_min, x_max, y_min, y_max = navigation_map.extent
    lower = np.maximum(lower, (x_min, y_min))
    upper = np.minimum(upper, (x_max, y_max))
    axis.set_xlim(float(lower[0]), float(upper[0]))
    axis.set_ylim(float(lower[1]), float(upper[1]))


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


def _query_navigation_map(
    navigation_map: NavigationMap,
    world_xy: np.ndarray,
) -> np.ndarray:
    """Query sample-bin coverage only, not physical traversability."""
    points = np.asarray(world_xy, dtype=np.float64)
    x_min, x_max, y_min, y_max = navigation_map.extent
    inside = (
        (points[..., 0] >= x_min)
        & (points[..., 0] < x_max)
        & (points[..., 1] >= y_min)
        & (points[..., 1] < y_max)
    )
    columns = np.floor(
        (points[..., 0] - x_min) / GLOBAL_MAP_RESOLUTION_M
    ).astype(np.int64)
    rows = np.floor(
        (points[..., 1] - y_min) / GLOBAL_MAP_RESOLUTION_M
    ).astype(np.int64)
    columns = np.clip(columns, 0, navigation_map.free.shape[1] - 1)
    rows = np.clip(rows, 0, navigation_map.free.shape[0] - 1)
    return inside & navigation_map.free[rows, columns]


def _resample_world_plan(
    path: np.ndarray,
    horizon_m: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Endpoint-inclusive physical arc sampling shared by online diagnostics."""
    path = np.asarray(path, dtype=np.float64)
    segment = np.linalg.norm(np.diff(path[:, :2], axis=0), axis=1)
    cumulative = np.concatenate(([0.0], np.cumsum(segment)))
    evaluated_length = cumulative[-1]
    if horizon_m is not None:
        evaluated_length = min(evaluated_length, horizon_m)
    distance = np.arange(
        0.0,
        evaluated_length + 0.5 * PATH_CONFIGURATION_QUERY_SPACING_M,
        PATH_CONFIGURATION_QUERY_SPACING_M,
    )
    distance = distance[distance <= evaluated_length + 1e-12]
    if distance.size == 0 or evaluated_length - distance[-1] > 1e-12:
        distance = np.append(distance, evaluated_length)
    indices = np.searchsorted(cumulative, distance, side="right") - 1
    indices = np.clip(indices, 0, len(path) - 2)
    width = segment[indices]
    ratio = np.divide(
        distance - cumulative[indices],
        width,
        out=np.zeros_like(distance),
        where=width > 1e-12,
    )
    sampled = path[indices, :2] + ratio[:, None] * (
        path[indices + 1, :2] - path[indices, :2]
    )
    return sampled, distance


def _closed_loop_plan_diagnostics(
    world_plans: list[np.ndarray],
    local_plans: list[np.ndarray],
    navigation_map: NavigationMap,
) -> dict[str, object]:
    """Summarize sample coverage, controller response and replanning stability."""
    if not world_plans or len(world_plans) != len(local_plans):
        raise ValueError("closed-loop diagnostics require aligned non-empty plans")
    future_sample_gap = []
    prefix_sample_gap = {0.5: [], 1.0: []}
    origin_free = []
    free_points = 0
    total_points = 0
    dense_world = []
    controller_values: dict[str, list[float]] = {}
    for world in world_plans:
        dense, distance = _resample_world_plan(world)
        free = _query_navigation_map(navigation_map, dense)
        dense_world.append(dense)
        future = distance > 1e-12
        free_points += int(free[future].sum())
        total_points += int(future.sum())
        origin_free.append(bool(free[0]))
        future_sample_gap.append(bool((~free[future]).any()))
        for horizon_m in prefix_sample_gap:
            selected = future & (distance <= horizon_m + 1e-12)
            prefix_sample_gap[horizon_m].append(bool((~free[selected]).any()))
    for point_count in sorted({len(plan) for plan in local_plans}):
        batch = np.stack([
            plan[:, :2] for plan in local_plans if len(plan) == point_count
        ])
        measured = controller_tracking_metrics(
            torch.from_numpy(batch).to(dtype=torch.float32)
        )
        for name, value in measured.items():
            controller_values.setdefault(name, []).extend(
                value.to(dtype=torch.float64).tolist()
            )

    disagreement = []
    for previous, current in zip(dense_world[:-1], dense_world[1:]):
        if len(current) < 2:
            current_prefix = current
        else:
            current_prefix, _ = _resample_world_plan(current, horizon_m=1.0)
        distance = np.linalg.norm(
            current_prefix[:, None, :] - previous[None, :, :], axis=-1
        )
        disagreement.append(float(distance.min(axis=1).mean()))

    def fraction(values: list[bool]) -> float:
        return float(np.mean(values)) if values else 0.0

    free_origin_indices = [index for index, value in enumerate(origin_free) if value]

    def conditional_fraction(values: list[bool]) -> float:
        selected = [values[index] for index in free_origin_indices]
        return fraction(selected)

    return {
        "plan_origin_sample_coverage": fraction(origin_free),
        "covered_origin_plan_count": len(free_origin_indices),
        "future_planned_point_sample_coverage": free_points / max(total_points, 1),
        "future_plan_sample_gap_fraction": fraction(future_sample_gap),
        "future_plan_prefix_0p5m_sample_gap_fraction": fraction(
            prefix_sample_gap[0.5]
        ),
        "future_plan_prefix_1p0m_sample_gap_fraction": fraction(
            prefix_sample_gap[1.0]
        ),
        "future_plan_prefix_0p5m_sample_gap_given_covered_origin_fraction": (
            conditional_fraction(prefix_sample_gap[0.5])
        ),
        "future_plan_prefix_1p0m_sample_gap_given_covered_origin_fraction": (
            conditional_fraction(prefix_sample_gap[1.0])
        ),
        "first_plan_prefix_0p5m_sample_gap": (
            prefix_sample_gap[0.5][0] if prefix_sample_gap[0.5] else False
        ),
        "first_plan_prefix_1p0m_sample_gap": (
            prefix_sample_gap[1.0][0] if prefix_sample_gap[1.0] else False
        ),
        "mpc_desired_speed_mps_mean": float(
            np.mean(controller_values.get("mpc_desired_speed_mps", [0.0]))
        ),
        "mpc_curvature_limited_fraction": fraction([
            bool(value)
            for value in controller_values.get("mpc_curvature_is_active", [])
        ]),
        "mpc_max_curvature_lookahead_inv_m_p95": float(np.quantile(
            controller_values.get("mpc_max_curvature_lookahead_inv_m", [0.0]), 0.95
        )),
        "adjacent_plan_first1m_world_disagreement_m_mean": (
            float(np.mean(disagreement)) if disagreement else 0.0
        ),
        "adjacent_plan_pairs": len(disagreement),
    }


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

    world_plans = []
    local_plans = []
    for index in range(plan_local.shape[0]):
        length = int(plan_lengths[index])
        if length < 3:
            continue
        local = plan_local[index, :length, :2]
        local_plans.append(local)
        world_plans.append(_robot_to_world(
            local, plan_positions[index], plan_quaternions[index]
        ))
    plan_diagnostics = _closed_loop_plan_diagnostics(
        world_plans, local_plans, navigation_map
    )
    executed_free = _query_navigation_map(navigation_map, executed)

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
        axis.plot(
            world_plan[:, 0], world_plan[:, 1],
            color="#0891b2", alpha=0.10, lw=0.75, zorder=2,
        )
        dense_plan, dense_distance = _resample_world_plan(world_plan)
        future = dense_distance > 1e-12
        unsafe = future & ~_query_navigation_map(navigation_map, dense_plan)
        if np.any(unsafe):
            axis.scatter(
                dense_plan[unsafe, 0], dense_plan[unsafe, 1],
                s=3.0, c="#dc2626", linewidths=0, alpha=0.16, zorder=3,
            )
    axis.plot(
        executed[:, 0], executed[:, 1], color="#2457ff", lw=2.5,
        label="executed trajectory", zorder=4,
    )
    if np.any(stalled):
        axis.scatter(
            positions[stalled, 0], positions[stalled, 1], s=9, c="#ef4444",
            linewidths=0, alpha=0.65, label="stalled while commanded", zorder=5,
        )
    executed_unsafe = ~executed_free
    if np.any(executed_unsafe):
        axis.scatter(
            executed[executed_unsafe, 0], executed[executed_unsafe, 1],
            s=13, c="#b91c1c", linewidths=0, alpha=0.85,
            label="executed outside sampled bins", zorder=5,
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
    _focus_map_view(
        axis,
        navigation_map,
        np.concatenate((executed, goal_world.reshape(1, 2)), axis=0),
    )
    handles, labels = axis.get_legend_handles_labels()
    axis.legend(
        map_handles + handles,
        [item.get_label() for item in map_handles] + labels,
        loc="best",
        fontsize=8,
        framealpha=0.9,
    )
    axis.grid(color="#64748b", alpha=0.22, linewidth=0.5)
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
        "actual_position_sample_coverage": float(executed_free.mean()),
        "final_goal_distance_m": final_goal_distance,
        **plan_diagnostics,
    }


def _render_scene_overview(
    trace_paths: list[Path],
    navigation_map: NavigationMap,
    output_path: Path,
) -> dict[str, object]:
    """Overlay executed episodes on the navigation-sample coverage map."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    figure, axis = plt.subplots(figsize=(10, 10), constrained_layout=True)
    map_handles = _draw_navigation_map(axis, navigation_map)
    successes = 0
    focus_points = []
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
        focus_points.extend((executed, goal_world.reshape(1, 2)))
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
            color="#111827",
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
    _focus_map_view(
        axis,
        navigation_map,
        np.concatenate(focus_points, axis=0),
        padding_m=1.5,
        minimum_span_m=6.0,
    )
    axis.legend(
        handles=map_handles + trajectory_handles,
        loc="best",
        fontsize=8,
        framealpha=0.9,
    )
    axis.grid(color="#64748b", alpha=0.22, linewidth=0.5)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180)
    plt.close(figure)
    return {
        "image": str(output_path.resolve()),
        "episode_count": len(trace_paths),
        "success_count": successes,
    }


def _summarize_closed_loop(records: list[dict[str, object]]) -> dict[str, object]:
    """Aggregate diagnostics with plans, not episodes, as the planning unit."""
    plans = sum(int(record["total_plan_count"]) for record in records)
    covered_origin_plans = sum(
        int(record["covered_origin_plan_count"]) for record in records
    )
    pairs = sum(int(record["adjacent_plan_pairs"]) for record in records)

    def episode_mean(name: str) -> float:
        return float(np.mean([float(record[name]) for record in records]))

    def plan_mean(name: str) -> float:
        return sum(
            float(record[name]) * int(record["total_plan_count"])
            for record in records
        ) / max(plans, 1)

    def covered_origin_plan_mean(name: str) -> float:
        return sum(
            float(record[name]) * int(record["covered_origin_plan_count"])
            for record in records
        ) / max(covered_origin_plans, 1)

    return {
        "episodes": len(records),
        "successes": sum(bool(record["success"]) for record in records),
        "plans": plans,
        "covered_origin_plans": covered_origin_plans,
        "plan_origin_sample_coverage": plan_mean("plan_origin_sample_coverage"),
        "future_plan_sample_gap_fraction": plan_mean(
            "future_plan_sample_gap_fraction"
        ),
        "future_plan_prefix_0p5m_sample_gap_fraction": plan_mean(
            "future_plan_prefix_0p5m_sample_gap_fraction"
        ),
        "future_plan_prefix_1p0m_sample_gap_fraction": plan_mean(
            "future_plan_prefix_1p0m_sample_gap_fraction"
        ),
        "future_plan_prefix_0p5m_sample_gap_given_covered_origin_fraction": (
            covered_origin_plan_mean(
                "future_plan_prefix_0p5m_sample_gap_given_covered_origin_fraction"
            )
        ),
        "future_plan_prefix_1p0m_sample_gap_given_covered_origin_fraction": (
            covered_origin_plan_mean(
                "future_plan_prefix_1p0m_sample_gap_given_covered_origin_fraction"
            )
        ),
        "first_plan_prefix_0p5m_sample_gap_fraction": episode_mean(
            "first_plan_prefix_0p5m_sample_gap"
        ),
        "first_plan_prefix_1p0m_sample_gap_fraction": episode_mean(
            "first_plan_prefix_1p0m_sample_gap"
        ),
        "future_planned_point_sample_coverage_episode_mean": episode_mean(
            "future_planned_point_sample_coverage"
        ),
        "actual_position_sample_coverage_episode_mean": episode_mean(
            "actual_position_sample_coverage"
        ),
        "mpc_desired_speed_mps_plan_mean": plan_mean(
            "mpc_desired_speed_mps_mean"
        ),
        "mpc_curvature_limited_fraction": plan_mean(
            "mpc_curvature_limited_fraction"
        ),
        "adjacent_plan_first1m_world_disagreement_m_mean": sum(
            float(record["adjacent_plan_first1m_world_disagreement_m_mean"])
            * int(record["adjacent_plan_pairs"])
            for record in records
        ) / max(pairs, 1),
        "stalled_step_fraction_episode_mean": episode_mean(
            "stalled_step_fraction"
        ),
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
                    "5 cm bins containing navigation PLY samples; empty bins are "
                    "unknown, not collision or non-traversability labels"
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
            {
                "summary": _summarize_closed_loop(records),
                "episodes": records,
                "scene_overviews": scene_overviews,
            },
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
