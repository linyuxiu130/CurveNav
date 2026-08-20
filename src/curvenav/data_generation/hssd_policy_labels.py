"""Offline geometry labels and Pareto rules for HSSD policy candidates."""

from __future__ import annotations

from dataclasses import dataclass
import heapq
import math
from pathlib import Path
from typing import Any

import numpy as np

from curvenav.data_generation.audit_hssd_v2_dynamics import (
    BENCHMARK_V_MAX_MPS,
    BENCHMARK_W_MAX_RADPS,
    geometric_curvature,
)
from curvenav.data_generation.occupancy import NavigationGrid, resample_polyline


MINIMUM_EXTRA_CLEARANCE_M = 0.1
CLEARANCE_SAMPLE_STEP_M = 0.025
CURVATURE_SAMPLE_STEP_M = 0.05
PREFERENCE_COLLISION = 1
PREFERENCE_SAFETY_MARGIN = 2
PREFERENCE_PROGRESS = 4
PREFERENCE_CLEARANCE = 8


def _episode_dir(root: Path, record: dict[str, Any]) -> Path:
    return (
        root
        / record["split"]
        / f"dataset_hssd_{record['scene_id']}"
        / record["episode_id"].rsplit("/", 1)[-1]
    )


def _load_grid(root: Path, record: dict[str, Any]) -> NavigationGrid:
    path = (
        root
        / record["split"]
        / f"dataset_hssd_{record['scene_id']}"
        / "navigation_grid.npz"
    )
    with np.load(path) as values:
        return NavigationGrid(
            free=values["free"].astype(bool),
            clearance_m=values["clearance_m"].astype(np.float32),
            origin_xy=values["origin_xy"].astype(np.float64),
            cell_size_m=float(values["cell_size_m"]),
        )


def _local_to_world(
    path_local: np.ndarray, origin: np.ndarray, yaw: float
) -> np.ndarray:
    cosine = math.cos(yaw)
    sine = math.sin(yaw)
    path = np.asarray(path_local, dtype=np.float64)
    return origin + np.column_stack(
        (
            path[:, 0] * cosine - path[:, 1] * sine,
            path[:, 0] * sine + path[:, 1] * cosine,
        )
    )


def _arc_length(path: np.ndarray) -> float:
    if len(path) < 2:
        return 0.0
    return float(np.linalg.norm(np.diff(path, axis=0), axis=1).sum())


def distribution(
    values: list[float] | np.ndarray,
) -> dict[str, float | int | None]:
    array = np.asarray(values, dtype=np.float64)
    array = array[np.isfinite(array)]
    if not len(array):
        return {
            "count": 0,
            "min": None,
            "p50": None,
            "p95": None,
            "p99": None,
            "max": None,
            "mean": None,
        }
    return {
        "count": len(array),
        "min": float(array.min()),
        "p50": float(np.percentile(array, 50)),
        "p95": float(np.percentile(array, 95)),
        "p99": float(np.percentile(array, 99)),
        "max": float(array.max()),
        "mean": float(array.mean()),
    }


def grid_geodesic_distance(
    grid: NavigationGrid, goal_xy: np.ndarray
) -> np.ndarray:
    safe = grid.free & (
        grid.clearance_m + 1e-7 >= MINIMUM_EXTRA_CLEARANCE_M
    )
    goal = grid.world_to_grid(goal_xy)
    if not grid.in_bounds(goal) or not safe[tuple(goal)]:
        raise ValueError(
            "episode goal is outside the clearance-constrained navigation grid"
        )
    distance = np.full(safe.shape, np.inf, dtype=np.float64)
    goal_cell = (int(goal[0]), int(goal[1]))
    distance[goal_cell] = 0.0
    queue = [(0.0, *goal_cell)]
    neighbors = (
        (-1, 0, 1.0),
        (1, 0, 1.0),
        (0, -1, 1.0),
        (0, 1, 1.0),
        (-1, -1, math.sqrt(2.0)),
        (-1, 1, math.sqrt(2.0)),
        (1, -1, math.sqrt(2.0)),
        (1, 1, math.sqrt(2.0)),
    )
    while queue:
        current, x, y = heapq.heappop(queue)
        if current > distance[x, y] + 1e-12:
            continue
        for dx, dy, step in neighbors:
            nx, ny = x + dx, y + dy
            if not (0 <= nx < safe.shape[0] and 0 <= ny < safe.shape[1]):
                continue
            if not safe[nx, ny]:
                continue
            if dx and dy and (not safe[x + dx, y] or not safe[x, y + dy]):
                continue
            candidate = current + step * grid.cell_size_m
            if candidate + 1e-12 < distance[nx, ny]:
                distance[nx, ny] = candidate
                heapq.heappush(queue, (candidate, nx, ny))
    return distance


@dataclass(frozen=True)
class EpisodeGeometry:
    world_xy: np.ndarray
    yaw_rad: np.ndarray
    geodesic_to_goal_m: np.ndarray


class OfflineLabeler:
    """Compute only labels supported by the immutable navigation-grid contract."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.grids: dict[tuple[str, str], NavigationGrid] = {}
        self.episodes: dict[str, EpisodeGeometry] = {}

    def episode(
        self, record: dict[str, Any]
    ) -> tuple[NavigationGrid, EpisodeGeometry]:
        scene_key = (record["split"], record["scene_id"])
        if scene_key not in self.grids:
            self.grids[scene_key] = _load_grid(self.root, record)
        grid = self.grids[scene_key]
        episode_id = record["episode_id"]
        if episode_id not in self.episodes:
            with np.load(_episode_dir(self.root, record) / "route.npz") as route:
                world_xy = route["world_xy"].astype(np.float64)
                yaw_rad = route["yaw_rad"].astype(np.float64)
                task_goal = route["task_goal_world_xy"].astype(np.float64)
            self.episodes[episode_id] = EpisodeGeometry(
                world_xy=world_xy,
                yaw_rad=yaw_rad,
                geodesic_to_goal_m=grid_geodesic_distance(grid, task_goal),
            )
        return grid, self.episodes[episode_id]

    @staticmethod
    def _distance_at(
        grid: NavigationGrid, distance: np.ndarray, point_xy: np.ndarray
    ) -> float:
        cell = grid.world_to_grid(np.asarray(point_xy, dtype=np.float64))
        if not grid.in_bounds(cell):
            return math.nan
        value = float(distance[tuple(cell)])
        return value if math.isfinite(value) else math.nan

    def label(
        self, record: dict[str, Any], path_local: np.ndarray
    ) -> dict[str, Any]:
        grid, episode = self.episode(record)
        anchor = int(record["anchor_index"])
        local = np.asarray(path_local, dtype=np.float64)
        if (
            local.ndim != 2
            or local.shape[1] != 2
            or not np.isfinite(local).all()
        ):
            raise ValueError(f"non-finite candidate path: {record['sample_id']}")
        if len(local) == 1:
            clearance_path = local
            curvature_path = local
        else:
            clearance_path = np.concatenate(
                [local, resample_polyline(local, CLEARANCE_SAMPLE_STEP_M)], axis=0
            )
            curvature_path = resample_polyline(local, CURVATURE_SAMPLE_STEP_M)
        world = _local_to_world(
            clearance_path,
            episode.world_xy[anchor],
            float(episode.yaw_rad[anchor]),
        )
        clearance = grid.sample_clearance(world)
        collision = bool((~np.isfinite(clearance)).any())
        finite_clearance = np.where(np.isfinite(clearance), clearance, 0.0)
        minimum_clearance = float(finite_clearance.min())
        clearance_p05 = float(np.percentile(finite_clearance, 5))
        curvature = geometric_curvature(curvature_path)
        curvature_p95 = (
            float(np.percentile(curvature, 95)) if len(curvature) else 0.0
        )
        maximum_curvature = float(curvature.max(initial=0.0))
        arc_length = _arc_length(local)
        endpoint_world = _local_to_world(
            local[-1:],
            episode.world_xy[anchor],
            float(episode.yaw_rad[anchor]),
        )[0]
        start_distance = self._distance_at(
            grid, episode.geodesic_to_goal_m, episode.world_xy[anchor]
        )
        endpoint_distance = self._distance_at(
            grid, episode.geodesic_to_goal_m, endpoint_world
        )
        progress = (
            start_distance - endpoint_distance
            if math.isfinite(start_distance) and math.isfinite(endpoint_distance)
            else math.nan
        )
        progress_per_arc = (
            0.0
            if arc_length <= 1e-6 and math.isfinite(progress)
            else progress / arc_length
            if math.isfinite(progress)
            else math.nan
        )
        speed_cap = (
            BENCHMARK_V_MAX_MPS
            if maximum_curvature
            <= BENCHMARK_W_MAX_RADPS / BENCHMARK_V_MAX_MPS
            else BENCHMARK_W_MAX_RADPS / maximum_curvature
        )
        return {
            "footprint_collision": collision,
            "minimum_extra_clearance_m": minimum_clearance,
            "clearance_p05_m": clearance_p05,
            "safety_margin_violation": (
                minimum_clearance + 1e-7 < MINIMUM_EXTRA_CLEARANCE_M
            ),
            "endpoint_geodesic_distance_m": endpoint_distance,
            "progress_m": progress,
            "progress_per_arc": progress_per_arc,
            "arc_length_m": arc_length,
            "curvature_p95_per_m": curvature_p95,
            "maximum_curvature_per_m": maximum_curvature,
            "nominal_peak_angular_rate_radps": (
                BENCHMARK_V_MAX_MPS * maximum_curvature
            ),
            "kinematic_speed_cap_mps": speed_cap,
            "geodesic_valid": (
                math.isfinite(endpoint_distance) and math.isfinite(progress)
            ),
        }


def pareto_preferences(
    candidate_labels: list[dict[str, Any]],
) -> list[tuple[int, int, int]]:
    """Return only pairs with unambiguous primitive-label Pareto dominance."""

    def dominates(
        first: dict[str, Any], second: dict[str, Any]
    ) -> tuple[bool, int]:
        if not first["geodesic_valid"] or not second["geodesic_valid"]:
            return False, 0
        no_worse = (
            int(first["footprint_collision"])
            <= int(second["footprint_collision"])
            and int(first["safety_margin_violation"])
            <= int(second["safety_margin_violation"])
            and first["progress_m"] + 1e-6 >= second["progress_m"]
            and first["minimum_extra_clearance_m"] + 1e-6
            >= second["minimum_extra_clearance_m"]
        )
        if not no_worse:
            return False, 0
        reason = 0
        reason |= PREFERENCE_COLLISION * (
            int(first["footprint_collision"])
            < int(second["footprint_collision"])
        )
        reason |= PREFERENCE_SAFETY_MARGIN * (
            int(first["safety_margin_violation"])
            < int(second["safety_margin_violation"])
        )
        reason |= PREFERENCE_PROGRESS * (
            first["progress_m"] > second["progress_m"] + 1e-6
        )
        reason |= PREFERENCE_CLEARANCE * (
            first["minimum_extra_clearance_m"]
            > second["minimum_extra_clearance_m"] + 1e-6
        )
        return reason != 0, reason

    preferences = []
    for first in range(len(candidate_labels)):
        for second in range(first + 1, len(candidate_labels)):
            first_wins, first_reason = dominates(
                candidate_labels[first], candidate_labels[second]
            )
            second_wins, second_reason = dominates(
                candidate_labels[second], candidate_labels[first]
            )
            if first_wins:
                preferences.append((first, second, first_reason))
            elif second_wins:
                preferences.append((second, first, second_reason))
    return preferences
