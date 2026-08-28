"""Shared geometry and clearance-aware route planning."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import heapq
import math
from pathlib import Path
import warnings
from typing import Any

import numpy as np
from scipy.interpolate import splev, splprep

from curvenav.physical import EXTRA_CLEARANCE_M, ROBOT_FOOTPRINT_RADIUS_M


SAFETY_STEP_M = 0.025
MIN_CLEARANCE_M = EXTRA_CLEARANCE_M
ENDPOINT_CLEARANCE_M = 0.30
MAX_SNAP_M = 0.06


class PlanningError(RuntimeError):
    pass


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def source_family(scene_id: str) -> str:
    return scene_id.split("_", 1)[0][:6]


@dataclass(frozen=True)
class Grid:
    free: np.ndarray
    clearance_m: np.ndarray
    origin_xy: np.ndarray
    cell_size_m: float

    @classmethod
    def load(cls, path: Path) -> "Grid":
        with np.load(path) as values:
            return cls(
                values["free"],
                values["clearance_m"],
                values["origin_xy"],
                float(values["cell_size_m"]),
            )

    def world_to_grid(self, xy: np.ndarray) -> np.ndarray:
        return np.floor(
            (np.asarray(xy, dtype=np.float64) - self.origin_xy) / self.cell_size_m
        ).astype(np.int32)

    def grid_to_world(self, cells: np.ndarray) -> np.ndarray:
        return (
            self.origin_xy
            + (np.asarray(cells, dtype=np.float64) + 0.5) * self.cell_size_m
        )

    def snap(self, xy: np.ndarray, radius_m: float = 0.25) -> np.ndarray:
        center = self.world_to_grid(np.asarray(xy))
        radius = max(1, math.ceil(radius_m / self.cell_size_m))
        lo = np.maximum(center - radius, 0)
        hi = np.minimum(center + radius + 1, self.free.shape)
        cells = np.argwhere(self.free[lo[0] : hi[0], lo[1] : hi[1]]) + lo
        if not len(cells):
            raise PlanningError("no navigable grid cell near endpoint")
        distance = np.linalg.norm(self.grid_to_world(cells) - xy, axis=1)
        index = int(np.argmin(distance))
        if distance[index] > radius_m:
            raise PlanningError("endpoint is too far from the navigation grid")
        return cells[index]

    def clearance(self, xy: np.ndarray) -> np.ndarray:
        cells = self.world_to_grid(xy)
        values = np.full(len(cells), -np.inf, dtype=np.float64)
        valid = (
            (cells[:, 0] >= 0)
            & (cells[:, 0] < self.free.shape[0])
            & (cells[:, 1] >= 0)
            & (cells[:, 1] < self.free.shape[1])
        )
        positions = np.flatnonzero(valid)
        inside = cells[valid]
        navigable = self.free[inside[:, 0], inside[:, 1]]
        positions, inside = positions[navigable], inside[navigable]
        values[positions] = self.clearance_m[inside[:, 0], inside[:, 1]]
        return values

    def safe(self, path: np.ndarray, minimum_m: float = MIN_CLEARANCE_M) -> bool:
        dense = resample(path, SAFETY_STEP_M)
        values = self.clearance(np.concatenate([path, dense]))
        return bool(np.isfinite(values).all() and np.all(values + 1e-9 >= minimum_m))


def polyline(path: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    path = np.asarray(path, dtype=np.float64)
    if path.ndim != 2 or path.shape[1] != 2 or len(path) < 2:
        raise ValueError("path must have shape [N,2], N >= 2")
    keep = np.concatenate(
        [[True], np.linalg.norm(np.diff(path, axis=0), axis=1) > 1e-9]
    )
    path = path[keep]
    if len(path) < 2:
        raise ValueError("path has zero length")
    return path, np.concatenate(
        [[0.0], np.cumsum(np.linalg.norm(np.diff(path, axis=0), axis=1))]
    )


def points_at_arc(path: np.ndarray, arcs_m: np.ndarray) -> np.ndarray:
    path, cumulative = polyline(path)
    query = np.clip(np.asarray(arcs_m), 0.0, cumulative[-1])
    return np.column_stack(
        [np.interp(query, cumulative, path[:, axis]) for axis in range(2)]
    )


def resample(path: np.ndarray, spacing_m: float) -> np.ndarray:
    path, cumulative = polyline(path)
    query = np.linspace(
        0.0, cumulative[-1], max(2, math.ceil(cumulative[-1] / spacing_m) + 1)
    )
    return points_at_arc(path, query)


def path_length(path: np.ndarray) -> float:
    return float(np.linalg.norm(np.diff(path, axis=0), axis=1).sum())


def curvature(path: np.ndarray) -> np.ndarray:
    path = np.asarray(path, dtype=np.float64)
    if len(path) < 3:
        return np.zeros(1)
    first, second, chord = (
        path[1:-1] - path[:-2],
        path[2:] - path[1:-1],
        path[2:] - path[:-2],
    )
    denominator = (
        np.linalg.norm(first, axis=1)
        * np.linalg.norm(second, axis=1)
        * np.linalg.norm(chord, axis=1)
    )
    cross = np.abs(first[:, 0] * second[:, 1] - first[:, 1] * second[:, 0])
    return np.divide(
        2 * cross, denominator, out=np.zeros_like(cross), where=denominator > 1e-9
    )


def headings(path: np.ndarray, arcs_m: np.ndarray) -> np.ndarray:
    delta = points_at_arc(path, arcs_m + 0.03) - points_at_arc(path, arcs_m - 0.03)
    return np.arctan2(delta[:, 1], delta[:, 0])


@dataclass(frozen=True)
class Plan:
    path_xy: np.ndarray
    metrics: dict[str, float]
    difficulty: str
    difficulty_tags: tuple[str, ...]


_NEIGHBORS = (
    (-1, 0, 1.0),
    (1, 0, 1.0),
    (0, -1, 1.0),
    (0, 1, 1.0),
    (-1, -1, math.sqrt(2)),
    (-1, 1, math.sqrt(2)),
    (1, -1, math.sqrt(2)),
    (1, 1, math.sqrt(2)),
)


def _astar(grid: Grid, start: np.ndarray, goal: np.ndarray) -> np.ndarray:
    shape = grid.free.shape
    allowed = grid.free & (grid.clearance_m + 1e-9 >= MIN_CLEARANCE_M)
    source, target = tuple(start), tuple(goal)
    cost = np.full(shape, np.inf)
    parent = np.full((*shape, 2), -1, dtype=np.int32)
    closed = np.zeros(shape, dtype=bool)
    cost[source] = 0.0
    queue = [(0.0, 0.0, *source)]
    while queue:
        _, current, x, y = heapq.heappop(queue)
        if closed[x, y] or current > cost[x, y] + 1e-12:
            continue
        if (x, y) == target:
            cells = [target]
            while cells[-1] != source:
                previous = parent[cells[-1]]
                if previous[0] < 0:
                    raise PlanningError("incomplete A* parent chain")
                cells.append((int(previous[0]), int(previous[1])))
            return np.asarray(cells[::-1], dtype=np.int32)
        closed[x, y] = True
        for dx, dy, step in _NEIGHBORS:
            nx, ny = x + dx, y + dy
            if not (0 <= nx < shape[0] and 0 <= ny < shape[1]) or not allowed[nx, ny]:
                continue
            if dx and dy and (not allowed[x + dx, y] or not allowed[x, y + dy]):
                continue
            clear = 0.5 * (grid.clearance_m[x, y] + grid.clearance_m[nx, ny])
            trial = current + step * grid.cell_size_m * (
                1 + math.exp(-clear / ROBOT_FOOTPRINT_RADIUS_M)
            )
            if trial + 1e-12 >= cost[nx, ny]:
                continue
            cost[nx, ny], parent[nx, ny] = trial, (x, y)
            heuristic = math.hypot(nx - target[0], ny - target[1]) * grid.cell_size_m
            heapq.heappush(queue, (trial + heuristic, trial, nx, ny))
    raise PlanningError("no safe grid route")


def _path_cost(grid: Grid, path: np.ndarray) -> float:
    sampled = resample(path, SAFETY_STEP_M)
    segment_length = np.linalg.norm(np.diff(sampled, axis=0), axis=1)
    clearance = grid.clearance(sampled)
    if not np.isfinite(clearance).all():
        return math.inf
    segment_clearance = 0.5 * (clearance[:-1] + clearance[1:])
    return float(
        np.sum(
            segment_length
            * (1.0 + np.exp(-segment_clearance / ROBOT_FOOTPRINT_RADIUS_M))
        )
    )


def _simplify_route(grid: Grid, path: np.ndarray) -> np.ndarray:
    output, index = [path[0]], 0
    while index < len(path) - 1:
        following = len(path) - 1
        while following > index + 1:
            direct = path[[index, following]]
            if grid.safe(direct) and _path_cost(grid, direct) <= _path_cost(
                grid, path[index : following + 1]
            ) + 1e-9:
                break
            following -= 1
        output.append(path[following])
        index = following
    return np.asarray(output)


def _smooth(grid: Grid, path: np.ndarray, spacing_m: float) -> np.ndarray:
    length = path_length(path)
    candidates: list[np.ndarray] = []
    if len(path) >= 3:
        segment = np.linalg.norm(np.diff(path, axis=0), axis=1)
        parameter = np.concatenate([[0.0], np.cumsum(segment)]) / length
        weights = np.ones(len(path))
        weights[[0, -1]] = 1e3
        for tolerance in (0.05, 0.025, 0.0):
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    spline, _ = splprep(
                        path.T,
                        u=parameter,
                        w=weights,
                        k=min(3, len(path) - 1),
                        s=len(path) * tolerance**2,
                    )
                candidate = np.column_stack(
                    splev(
                        np.linspace(
                            0.0, 1.0, max(2, math.ceil(length / spacing_m) + 1)
                        ),
                        spline,
                    )
                )
                candidate[[0, -1]] = path[[0, -1]]
                if grid.safe(candidate):
                    candidates.append(candidate)
            except (TypeError, ValueError):
                pass
    for ratio in (0.25, 0.15, 0.08):
        candidate = path.copy()
        for _ in range(3):
            pieces = [
                (1 - ratio) * a + ratio * b
                for a, b in zip(candidate[:-1], candidate[1:])
            ]
            other = [
                ratio * a + (1 - ratio) * b
                for a, b in zip(candidate[:-1], candidate[1:])
            ]
            candidate = np.vstack(
                [
                    candidate[0],
                    np.column_stack((pieces, other)).reshape(-1, 2),
                    candidate[-1],
                ]
            )
        candidate = resample(candidate, spacing_m)
        if grid.safe(candidate):
            candidates.append(candidate)
    if not candidates:
        candidate = resample(path, spacing_m)
        if not grid.safe(candidate):
            raise PlanningError("route smoothing violated clearance")
        return candidate
    return min(
        candidates,
        key=lambda item: (
            np.percentile(curvature(item), 95),
            curvature(item).max(),
            path_length(item),
        ),
    )


def plan_route(
    grid: Grid,
    start_xy: np.ndarray,
    goal_xy: np.ndarray,
    spacing_m: float = 0.05,
) -> Plan:
    start, goal = grid.snap(start_xy), grid.snap(goal_xy)
    cells = _astar(grid, start, goal)
    discrete_path = np.vstack([start_xy, grid.grid_to_world(cells), goal_xy])
    path = _smooth(
        grid,
        _simplify_route(grid, discrete_path),
        spacing_m,
    )
    length = path_length(path)
    clear, curve = grid.clearance(path), curvature(path)
    heading = np.arctan2(np.diff(path, axis=0)[:, 1], np.diff(path, axis=0)[:, 0])
    heading_change = np.arctan2(
        np.sin(np.diff(heading)),
        np.cos(np.diff(heading)),
    )
    metrics = {
        "length_m": length,
        "geodesic_ratio": length / float(np.linalg.norm(path[-1] - path[0])),
        "minimum_clearance_m": float(clear.min()),
        "clearance_p05_m": float(np.percentile(clear, 5)),
        "risk_density": float(
            np.mean(np.exp(-clear / ROBOT_FOOTPRINT_RADIUS_M))
        ),
        "curvature_p95": float(np.percentile(curve, 95)),
        "maximum_curvature": float(curve.max()),
        "total_turn_radians": float(np.abs(heading_change).sum()),
    }
    tags = []
    if metrics["clearance_p05_m"] < 0.2:
        tags.append("narrow")
    if metrics["geodesic_ratio"] > 1.25:
        tags.append("detour")
    if metrics["total_turn_radians"] > math.pi / 2 or metrics["curvature_p95"] > 1.0:
        tags.append("turning")
    tags = tags or ["open"]
    return Plan(path.astype(np.float32), metrics, tags[0], tuple(tags))


def source_route(grid: Grid, start_xy: np.ndarray, goal_xy: np.ndarray) -> Plan:
    plan = plan_route(grid, start_xy, goal_xy, grid.cell_size_m)
    if not grid.safe(plan.path_xy):
        raise PlanningError("source route violates continuous clearance")
    return plan


def candidate_pairs(
    grid: Grid, distance_range_m: tuple[float, float], seed: int, limit: int
) -> list[tuple[np.ndarray, np.ndarray]]:
    eligible = np.argwhere(grid.clearance_m >= ENDPOINT_CLEARANCE_M)
    rng, pairs, seen = np.random.default_rng(seed), [], set()
    minimum, maximum = distance_range_m
    for _ in range(40000):
        start = eligible[rng.integers(len(eligible))]
        radius, angle = math.sqrt(rng.uniform(minimum**2, maximum**2)), rng.uniform(
            -math.pi, math.pi
        )
        goal = start + np.rint(
            radius / grid.cell_size_m * np.array([math.cos(angle), math.sin(angle)])
        ).astype(int)
        if (
            np.any(goal < 0)
            or np.any(goal >= grid.free.shape)
            or grid.clearance_m[tuple(goal)] < ENDPOINT_CLEARANCE_M
        ):
            continue
        key = tuple(np.concatenate([start, goal]))
        if key in seen:
            continue
        seen.add(key)
        pair = grid.grid_to_world(np.stack([start, goal]))
        if minimum <= np.linalg.norm(pair[1] - pair[0]) < maximum:
            pairs.append((pair[0], pair[1]))
            if len(pairs) == limit:
                break
    if not pairs:
        raise PlanningError("no safe endpoint pair")
    return pairs
