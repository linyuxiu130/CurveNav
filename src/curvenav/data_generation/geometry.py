"""Shared geometry, planning, and per-sample data contracts."""

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


SCHEMA = "curvenav_hssd_policy_dataset"
FRAMES = 4
HISTORY_ARCS_M = np.array([-1.35, -0.90, -0.45, 0.0], dtype=np.float64)
PERTURBATION_PROFILE = np.array([0.0, 1 / 3, 2 / 3, 1.0], dtype=np.float64)
TARGET_ARC_M = 3.0
LOOKAHEAD_M = 4.0
SAFETY_STEP_M = 0.025
MIN_CLEARANCE_M = 0.10
ENDPOINT_CLEARANCE_M = 0.30
MAX_SNAP_M = 0.06
MAX_CURVATURE = 10.0
MAX_CURVATURE_P95 = 4.0
NOMINAL_SPEED_MPS = 0.5


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
        return self.origin_xy + (np.asarray(cells, dtype=np.float64) + 0.5) * self.cell_size_m

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
    keep = np.concatenate([[True], np.linalg.norm(np.diff(path, axis=0), axis=1) > 1e-9])
    path = path[keep]
    if len(path) < 2:
        raise ValueError("path has zero length")
    return path, np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(path, axis=0), axis=1))])


def points_at_arc(path: np.ndarray, arcs_m: np.ndarray) -> np.ndarray:
    path, cumulative = polyline(path)
    query = np.clip(np.asarray(arcs_m), 0.0, cumulative[-1])
    return np.column_stack([np.interp(query, cumulative, path[:, axis]) for axis in range(2)])


def resample(path: np.ndarray, spacing_m: float) -> np.ndarray:
    path, cumulative = polyline(path)
    query = np.linspace(0.0, cumulative[-1], max(2, math.ceil(cumulative[-1] / spacing_m) + 1))
    return points_at_arc(path, query)


def prefix(path: np.ndarray, maximum_arc_m: float) -> np.ndarray:
    path, cumulative = polyline(path)
    end = min(float(cumulative[-1]), maximum_arc_m)
    middle = path[(cumulative > 0.0) & (cumulative < end - 1e-4)]
    return np.vstack([path[0], middle, points_at_arc(path, np.array([end]))]).astype(np.float32)


def path_length(path: np.ndarray) -> float:
    return float(np.linalg.norm(np.diff(path, axis=0), axis=1).sum())


def curvature(path: np.ndarray) -> np.ndarray:
    path = np.asarray(path, dtype=np.float64)
    if len(path) < 3:
        return np.zeros(1)
    first, second, chord = path[1:-1] - path[:-2], path[2:] - path[1:-1], path[2:] - path[:-2]
    denominator = (
        np.linalg.norm(first, axis=1)
        * np.linalg.norm(second, axis=1)
        * np.linalg.norm(chord, axis=1)
    )
    cross = np.abs(first[:, 0] * second[:, 1] - first[:, 1] * second[:, 0])
    return np.divide(2 * cross, denominator, out=np.zeros_like(cross), where=denominator > 1e-9)


def local_xy(world_xy: np.ndarray, origin_xy: np.ndarray, yaw: float) -> np.ndarray:
    delta = np.asarray(world_xy) - np.asarray(origin_xy)
    cosine, sine = math.cos(yaw), math.sin(yaw)
    return np.array([cosine * delta[0] + sine * delta[1], -sine * delta[0] + cosine * delta[1]])


def world_xy(local: np.ndarray, origin: np.ndarray, yaw: float) -> np.ndarray:
    local = np.asarray(local)
    cosine, sine = math.cos(yaw), math.sin(yaw)
    return (
        np.column_stack(
            (cosine * local[:, 0] - sine * local[:, 1], sine * local[:, 0] + cosine * local[:, 1])
        )
        + origin
    )


def headings(path: np.ndarray, arcs_m: np.ndarray) -> np.ndarray:
    delta = points_at_arc(path, arcs_m + 0.03) - points_at_arc(path, arcs_m - 0.03)
    return np.arctan2(delta[:, 1], delta[:, 0])


def observation_to_current(xy: np.ndarray, yaw: np.ndarray) -> np.ndarray:
    anchor_xy, anchor_yaw = xy[-1], float(yaw[-1])
    translation = np.asarray([local_xy(point, anchor_xy, anchor_yaw) for point in xy])
    delta = yaw - anchor_yaw
    return np.column_stack((translation, np.sin(delta), np.cos(delta))).astype(np.float32)


@dataclass(frozen=True)
class Plan:
    path_xy: np.ndarray
    metrics: dict[str, float]
    difficulty: str
    difficulty_tags: tuple[str, ...]
    clearance_weight: float


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


def _astar(
    grid: Grid, start: np.ndarray, goal: np.ndarray, weight: float, preferred_m: float
) -> np.ndarray:
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
                1 + weight * math.exp(-clear / preferred_m)
            )
            if trial + 1e-12 >= cost[nx, ny]:
                continue
            cost[nx, ny], parent[nx, ny] = trial, (x, y)
            heuristic = math.hypot(nx - target[0], ny - target[1]) * grid.cell_size_m
            heapq.heappush(queue, (trial + heuristic, trial, nx, ny))
    raise PlanningError("no safe grid route")


def _shortcut(grid: Grid, path: np.ndarray) -> np.ndarray:
    output, index = [path[0]], 0
    while index < len(path) - 1:
        following = len(path) - 1
        while following > index + 1 and not grid.safe(path[[index, following]]):
            following -= 1
        output.append(path[following])
        index = following
    return np.asarray(output)


def _smooth(grid: Grid, path: np.ndarray, spacing_m: float) -> np.ndarray:
    length = path_length(path)
    candidates: list[np.ndarray] = []
    if len(path) >= 3:
        segment = np.linalg.norm(np.diff(path, axis=0), axis=1)
        parameter = np.concatenate([[0.0], np.cumsum(segment)]) / max(length, 1e-9)
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
                    splev(np.linspace(0.0, 1.0, max(2, math.ceil(length / spacing_m) + 1)), spline)
                )
                candidate[[0, -1]] = path[[0, -1]]
                if grid.safe(candidate):
                    candidates.append(candidate)
            except (TypeError, ValueError):
                pass
    for ratio in (0.25, 0.15, 0.08):
        candidate = path.copy()
        for _ in range(3):
            pieces = [(1 - ratio) * a + ratio * b for a, b in zip(candidate[:-1], candidate[1:])]
            other = [ratio * a + (1 - ratio) * b for a, b in zip(candidate[:-1], candidate[1:])]
            candidate = np.vstack(
                [candidate[0], np.column_stack((pieces, other)).reshape(-1, 2), candidate[-1]]
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
    preferred_m: float = 0.30,
    spacing_m: float = 0.05,
) -> Plan:
    start, goal = grid.snap(start_xy), grid.snap(goal_xy)
    candidates: list[tuple[float, np.ndarray, dict[str, float]]] = []
    reference_length = None
    for weight in (0.0, 0.5, 1.0, 2.0, 4.0):
        try:
            cells = _astar(grid, start, goal, weight, preferred_m)
            path = _smooth(
                grid,
                _shortcut(grid, np.vstack([start_xy, grid.grid_to_world(cells), goal_xy])),
                spacing_m,
            )
        except PlanningError:
            continue
        length = path_length(path)
        if weight == 0.0:
            reference_length = length
        if reference_length is None:
            continue
        clear, curve = grid.clearance(path), curvature(path)
        metrics = {
            "length_m": length,
            "reference_length_ratio": length / max(reference_length, 1e-6),
            "geodesic_ratio": length / max(float(np.linalg.norm(path[-1] - path[0])), 1e-6),
            "minimum_clearance_m": float(clear.min()),
            "clearance_p05_m": float(np.percentile(clear, 5)),
            "risk_density": float(np.mean(np.exp(-clear / preferred_m))),
            "curvature_p95": float(np.percentile(curve, 95)),
            "maximum_curvature": float(curve.max()),
            "total_turn_radians": float(
                np.abs(
                    np.arctan2(
                        np.sin(
                            np.diff(
                                np.arctan2(
                                    np.diff(path, axis=0)[:, 1], np.diff(path, axis=0)[:, 0]
                                )
                            )
                        ),
                        np.cos(
                            np.diff(
                                np.arctan2(
                                    np.diff(path, axis=0)[:, 1], np.diff(path, axis=0)[:, 0]
                                )
                            )
                        ),
                    )
                ).sum()
            ),
        }
        if (
            metrics["maximum_curvature"] <= MAX_CURVATURE
            and metrics["curvature_p95"] <= MAX_CURVATURE_P95
        ):
            candidates.append((weight, path, metrics))
    eligible = [item for item in candidates if item[2]["reference_length_ratio"] <= 1.2 + 1e-6]
    if not eligible:
        raise PlanningError("no curvature-feasible route within the detour budget")
    frontier = [
        item
        for item in eligible
        if not any(
            other is not item
            and other[2]["length_m"] <= item[2]["length_m"] + 1e-9
            and other[2]["risk_density"] <= item[2]["risk_density"] + 1e-9
            and other[2]["clearance_p05_m"] > item[2]["clearance_p05_m"] + 1e-9
            for other in eligible
        )
    ]
    weight, path, metrics = min(
        frontier,
        key=lambda item: (
            item[2]["risk_density"]
            + 0.02 * item[2]["curvature_p95"]
            + 0.02 * item[2]["reference_length_ratio"],
            -item[2]["clearance_p05_m"],
            item[2]["length_m"],
        ),
    )
    tags = []
    if metrics["clearance_p05_m"] < 0.2:
        tags.append("narrow")
    if metrics["geodesic_ratio"] > 1.25:
        tags.append("detour")
    if metrics["total_turn_radians"] > math.pi / 2 or metrics["curvature_p95"] > 1.0:
        tags.append("turning")
    tags = tags or ["open"]
    return Plan(path.astype(np.float32), metrics, tags[0], tuple(tags), float(weight))


def native_route(
    simulator: Any, start_xy: np.ndarray, goal_xy: np.ndarray, floor_m: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    import habitat_sim

    requested = np.array(
        [[start_xy[0], floor_m, start_xy[1]], [goal_xy[0], floor_m, goal_xy[1]]], dtype=np.float32
    )
    snapped = np.asarray(
        [simulator.pathfinder.snap_point(point) for point in requested], dtype=np.float32
    )
    if (
        not np.isfinite(snapped).all()
        or np.linalg.norm(snapped[:, [0, 2]] - requested[:, [0, 2]], axis=1).max() > MAX_SNAP_M
    ):
        raise PlanningError("route endpoint snap exceeds the physical contract")
    query = habitat_sim.ShortestPath()
    query.requested_start, query.requested_end = snapped
    if not simulator.pathfinder.find_path(query):
        raise PlanningError("Habitat found no route")
    path = resample(np.asarray(query.points)[:, [0, 2]], 0.05)
    path[[0, -1]] = snapped[:, [0, 2]]
    return path.astype(np.float32), snapped[0], snapped[1], float(query.geodesic_distance)


def local_prefix(grid: Grid, global_route: np.ndarray) -> tuple[np.ndarray, Plan]:
    lookahead = points_at_arc(
        global_route, np.array([min(path_length(global_route), LOOKAHEAD_M)])
    )[0]
    cells = grid.world_to_grid(prefix(global_route, LOOKAHEAD_M))
    margin = math.ceil(2.0 / grid.cell_size_m)
    lo, hi = np.maximum(cells.min(0) - margin, 0), np.minimum(
        cells.max(0) + margin + 1, grid.free.shape
    )
    crop = Grid(
        grid.free[lo[0] : hi[0], lo[1] : hi[1]],
        grid.clearance_m[lo[0] : hi[0], lo[1] : hi[1]],
        grid.origin_xy + lo * grid.cell_size_m,
        grid.cell_size_m,
    )
    plan = plan_route(crop, global_route[0], lookahead)
    return prefix(plan.path_xy, TARGET_ARC_M), plan


def source_route(
    grid: Grid, start_xy: np.ndarray, goal_xy: np.ndarray, preferred_m: float
) -> Plan:
    plan = plan_route(grid, start_xy, goal_xy, preferred_m, grid.cell_size_m)
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


def variant_history(
    base_route: np.ndarray, anchor_arc_m: float, lateral_m: float, yaw_degrees: float
) -> dict[str, np.ndarray]:
    arcs = anchor_arc_m + HISTORY_ARCS_M
    base_xy, base_yaw = points_at_arc(base_route, arcs), headings(base_route, arcs)
    normal = np.array([-math.sin(base_yaw[-1]), math.cos(base_yaw[-1])])
    return {
        "base_xy": base_xy.astype(np.float32),
        "base_yaw": base_yaw.astype(np.float32),
        "world_xy": (base_xy + (PERTURBATION_PROFILE * lateral_m)[:, None] * normal).astype(
            np.float32
        ),
        "yaw": (base_yaw + np.deg2rad(PERTURBATION_PROFILE * yaw_degrees)).astype(np.float32),
        "normal": normal.astype(np.float32),
    }


def snapped_history(
    simulator: Any,
    grid: Grid,
    base_route: np.ndarray,
    anchor_arc_m: float,
    lateral_m: float,
    yaw_degrees: float,
    floor_m: float,
) -> dict[str, np.ndarray]:
    result = variant_history(base_route, anchor_arc_m, lateral_m, yaw_degrees)
    requested = result["world_xy"]
    xyz = np.column_stack((requested[:, 0], np.full(FRAMES, floor_m), requested[:, 1]))
    snapped = np.asarray(
        [simulator.pathfinder.snap_point(point) for point in xyz], dtype=np.float32
    )
    error = np.linalg.norm(snapped[:, [0, 2]] - requested, axis=1)
    result.update(
        world_xy=snapped[:, [0, 2]], habitat_xyz=snapped, snap_error_m=error.astype(np.float32)
    )
    spacing, yaw_step = np.linalg.norm(np.diff(result["world_xy"], axis=0), axis=1), np.abs(
        np.arctan2(np.sin(np.diff(result["yaw"])), np.cos(np.diff(result["yaw"])))
    )
    if (
        not np.isfinite(snapped).all()
        or error.max() > MAX_SNAP_M
        or not grid.safe(result["world_xy"])
        or np.any((spacing < 0.2) | (spacing > 0.75))
        or np.any(yaw_step > math.radians(30))
    ):
        raise PlanningError("variant history violates pose or clearance contract")
    return result


def route_turn(path: np.ndarray, anchor_arc_m: float) -> tuple[float, str]:
    end = min(path_length(path), anchor_arc_m + 3.0)
    values = headings(
        path, np.linspace(anchor_arc_m, end, max(2, math.ceil((end - anchor_arc_m) / 0.05) + 1))
    )
    degrees = math.degrees(
        float(np.arctan2(np.sin(np.diff(values)), np.cos(np.diff(values))).sum())
    )
    return degrees, "left" if degrees > 1 else "right" if degrees < -1 else "straight"


def sample_contract(
    root: Path, record: dict[str, Any], grid: Grid, camera: dict[str, Any]
) -> dict[str, Any]:
    directory = root / record["sample_directory"]
    reasons: list[str] = []
    depth_path, geometry_path = directory / "depth_m.npy", directory / "geometry.npz"
    depth = np.load(depth_path, mmap_mode="r")
    expected_shape = (FRAMES, camera["image"]["height"], camera["image"]["width"])
    depth_hash = sha256_file(depth_path)
    invalid = ~np.isfinite(depth) | (depth <= 0)
    if depth.dtype != np.float32 or depth.shape != expected_shape:
        reasons.append("depth_shape_or_dtype")
    if (
        depth_hash != record["depth"]["sha256"]
        or abs(float(invalid.mean()) - record["depth"]["invalid_fraction"]) > 1e-12
        or invalid.all()
    ):
        reasons.append("depth_content")
    with np.load(geometry_path) as archive:
        values = {name: archive[name] for name in archive.files}
    history_xy, history_yaw = values["history_world_xy"], values["history_yaw_rad"]
    goal_world, goal_local = values["task_goal_world_xy"], values["task_goal_local_xy"]
    target_world, target_local = values["target_path_world_xy"], values["target_path_local_xy"]
    if record["schema"] != SCHEMA or record["camera"] != camera:
        reasons.append("schema")
    if history_xy.shape != (FRAMES, 2) or history_yaw.shape != (FRAMES,):
        reasons.append("history_shape")
    if not np.allclose(
        goal_local, local_xy(goal_world, history_xy[-1], float(history_yaw[-1])), atol=2e-5
    ):
        reasons.append("goal_frame")
    goal_distance = float(np.linalg.norm(goal_local))
    if (
        not np.allclose(goal_world, record["task_goal"]["world_xy_m"], atol=2e-5)
        or not np.allclose(goal_local, record["task_goal"]["local_xy_m"], atol=2e-5)
        or not math.isclose(
            goal_distance, record["task_goal"]["euclidean_distance_m"], abs_tol=2e-5
        )
    ):
        reasons.append("goal_metadata")
    lower, upper = record["goal_band_m"]
    band_ok = (
        lower <= goal_distance <= upper
        if record["goal_band"] == "far"
        else lower <= goal_distance < upper
    )
    if not (0.5 <= goal_distance <= 10.0 and band_ok):
        reasons.append("goal_range")
    if grid.clearance(goal_world[None])[0] + 1e-8 < ENDPOINT_CLEARANCE_M:
        reasons.append("goal_clearance")
    if (
        target_local.ndim != 2
        or target_local.shape[1] != 2
        or not np.allclose(target_local[0], 0, atol=1e-7)
    ):
        reasons.append("target_shape")
    if not np.allclose(
        target_world, world_xy(target_local, history_xy[-1], float(history_yaw[-1])), atol=2e-5
    ):
        reasons.append("target_frame")
    curve = curvature(target_world)
    target_arc = path_length(target_world)
    if not math.isclose(
        target_arc, record["target"]["arc_length_m"], abs_tol=2e-5
    ) or not np.allclose(target_local[-1], record["target"]["endpoint_local_xy_m"], atol=2e-5):
        reasons.append("target_metadata")
    if (
        target_arc > TARGET_ARC_M + 2e-5
        or not grid.safe(target_world)
        or curve.max() > MAX_CURVATURE + 1e-6
        or np.percentile(curve, 95) > MAX_CURVATURE_P95 + 1e-6
    ):
        reasons.append("target_geometry")
    spacing = np.linalg.norm(np.diff(history_xy, axis=0), axis=1)
    timestamps = values["history_nominal_timestamp_s"]
    if (
        not grid.safe(history_xy)
        or np.any((spacing < 0.33) | (spacing > 0.60))
        or not np.array_equal(values["observation_valid"], np.ones(FRAMES, dtype=bool))
        or not np.allclose(
            timestamps,
            np.concatenate([[0.0], np.cumsum(spacing) / NOMINAL_SPEED_MPS]),
            atol=2e-5,
        )
        or not np.allclose(
            values["observation_to_current"],
            observation_to_current(history_xy, history_yaw),
            atol=2e-5,
        )
    ):
        reasons.append("history_geometry")
    return {
        "sample_id": record["sample_id"],
        "split": record["split"],
        "scene_id": record["scene_id"],
        "source_family": record["source_family"],
        "source_route_id": record["source_route_id"],
        "anchor_group_id": record["anchor_group_id"],
        "variant": record["variant"],
        "goal_band": record["goal_band"],
        "goal_distance_m": goal_distance,
        "target_arc_m": target_arc,
        "target_curvature_p95": float(np.percentile(curve, 95)),
        "target_curvature_max": float(curve.max()),
        "target_clearance_m": float(
            grid.clearance(
                np.concatenate([target_world, resample(target_world, SAFETY_STEP_M)])
            ).min()
        ),
        "history_clearance_m": float(
            grid.clearance(np.concatenate([history_xy, resample(history_xy, SAFETY_STEP_M)])).min()
        ),
        "depth_sha256": depth_hash,
        "depth_invalid_fraction": float(invalid.mean()),
        "violations": reasons,
    }
