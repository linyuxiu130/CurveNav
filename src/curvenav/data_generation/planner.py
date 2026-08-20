"""Safety-constrained, near-shortest expert route planning."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import heapq
import math
from typing import Union
import warnings

import numpy as np
from scipy.interpolate import splev, splprep

from curvenav.data_generation.occupancy import NavigationGrid, resample_polyline
from curvenav.data_generation.scan_refinement import refine_scan_style_trajectory


class PlanningError(RuntimeError):
    """Raised when no route satisfies the expert quality contract."""


PLANNER_ALGORITHM_VERSION = "scan-style-refinement-v3"


@dataclass(frozen=True)
class PlannerConfig:
    clearance_weights: tuple[float, ...] = (0.0, 0.5, 1.0, 2.0, 4.0)
    preferred_clearance_m: float = 0.3
    minimum_clearance_m: float = 0.1
    maximum_safe_detour_ratio: float = 1.2
    validation_step_m: float = 0.025
    output_spacing_m: float = 0.05
    snap_distance_m: float = 0.25
    chaikin_iterations: int = 3
    spline_smoothing_m: float = 0.05
    curvature_weight: float = 0.02
    length_tiebreak_weight: float = 0.02
    # X-NavDP's MPC allows w_max=0.5 rad/s and v_min=0.05 m/s in high
    # curvature regions, corresponding to a practical kappa ceiling of 10 1/m.
    maximum_curvature: float = 10.0
    maximum_curvature_p95: float = 4.0

    def validate(self) -> None:
        if not self.clearance_weights or self.clearance_weights[0] != 0.0:
            raise ValueError("clearance_weights must start with 0.0 for the reference path")
        if self.preferred_clearance_m <= 0 or self.minimum_clearance_m < 0:
            raise ValueError("clearance thresholds must be non-negative and preferred > 0")
        if self.maximum_safe_detour_ratio < 1.0:
            raise ValueError("maximum_safe_detour_ratio must be at least 1.0")
        if self.maximum_curvature <= 0 or self.maximum_curvature_p95 <= 0:
            raise ValueError("curvature limits must be positive")


@dataclass(frozen=True)
class RouteMetrics:
    length_m: float
    euclidean_distance_m: float
    geodesic_ratio: float
    reference_length_ratio: float
    minimum_clearance_m: float
    clearance_p05_m: float
    mean_clearance_m: float
    risk_density: float
    preferred_clearance_exposure: float
    total_turn_radians: float
    curvature_p95: float
    maximum_curvature: float


RouteCandidate = tuple[
    float,
    np.ndarray,
    RouteMetrics,
    dict[str, Union[float, int, bool, str]],
]


def _safety_efficiency_frontier(
    candidates: list[RouteCandidate],
) -> list[RouteCandidate]:
    """Prefer a wider bottleneck only when risk and path length do not increase."""

    tolerance = 1e-9

    def dominates(challenger: RouteCandidate, incumbent: RouteCandidate) -> bool:
        better = challenger[2]
        worse = incumbent[2]
        no_worse = (
            better.length_m <= worse.length_m + tolerance
            and better.risk_density <= worse.risk_density + tolerance
            and better.clearance_p05_m >= worse.clearance_p05_m - tolerance
        )
        wider_bottleneck = (
            better.clearance_p05_m > worse.clearance_p05_m + tolerance
        )
        return no_worse and wider_bottleneck

    return [
        candidate
        for candidate in candidates
        if not any(
            dominates(other, candidate)
            for other in candidates
            if other is not candidate
        )
    ]


@dataclass(frozen=True)
class PlannedRoute:
    path_xy: np.ndarray
    start_xy: np.ndarray
    goal_xy: np.ndarray
    selected_clearance_weight: float
    difficulty: str
    difficulty_tags: tuple[str, ...]
    metrics: RouteMetrics
    candidates: tuple[dict[str, float | bool], ...]
    refinement: dict[str, float | int | bool | str]

    def to_manifest_fields(self) -> dict[str, object]:
        return {
            "start_xy": np.round(self.start_xy, 5).tolist(),
            "goal_xy": np.round(self.goal_xy, 5).tolist(),
            "path_xy": np.round(self.path_xy, 5).tolist(),
            "selected_clearance_weight": self.selected_clearance_weight,
            "difficulty": self.difficulty,
            "difficulty_tags": list(self.difficulty_tags),
            "metrics": asdict(self.metrics),
            "candidates": list(self.candidates),
            "refinement": dict(self.refinement),
        }


class SafeEfficientPlanner:
    """Choose the safest executable route inside a bounded detour budget."""

    _NEIGHBORS = (
        (-1, 0, 1.0),
        (1, 0, 1.0),
        (0, -1, 1.0),
        (0, 1, 1.0),
        (-1, -1, math.sqrt(2.0)),
        (-1, 1, math.sqrt(2.0)),
        (1, -1, math.sqrt(2.0)),
        (1, 1, math.sqrt(2.0)),
    )

    def __init__(self, grid: NavigationGrid, config: PlannerConfig | None = None) -> None:
        self.grid = grid
        self.config = config or PlannerConfig()
        self.config.validate()

    def plan(self, start_xy: np.ndarray, goal_xy: np.ndarray) -> PlannedRoute:
        start_xy = np.asarray(start_xy, dtype=np.float64)
        goal_xy = np.asarray(goal_xy, dtype=np.float64)
        start = self.grid.snap_to_free(start_xy, self.config.snap_distance_m)
        goal = self.grid.snap_to_free(goal_xy, self.config.snap_distance_m)
        if np.array_equal(start, goal):
            raise PlanningError("start and goal snap to the same grid cell")

        candidates: list[RouteCandidate] = []
        failures: list[dict[str, float | bool]] = []
        reference_length: float | None = None
        for clearance_weight in self.config.clearance_weights:
            indices = self._astar(start, goal, clearance_weight)
            if indices is None:
                failures.append({"clearance_weight": clearance_weight, "valid": False})
                continue
            path = self.grid.grid_to_world(indices)
            path = np.vstack([start_xy, path, goal_xy])
            path = self._shortcut(path)
            path, refinement = self._safe_smooth(path)
            if not self.grid.path_is_safe(
                path,
                minimum_clearance_m=self.config.minimum_clearance_m,
                sample_step_m=self.config.validation_step_m,
            ):
                failures.append({"clearance_weight": clearance_weight, "valid": False})
                continue
            length = _path_length(path)
            if clearance_weight == 0.0:
                reference_length = length
            if reference_length is None:
                continue
            metrics = self._metrics(path, reference_length)
            if (
                metrics.maximum_curvature > self.config.maximum_curvature
                or metrics.curvature_p95 > self.config.maximum_curvature_p95
            ):
                failures.append(
                    {
                        "clearance_weight": clearance_weight,
                        "valid": False,
                        "curvature_rejected": True,
                    }
                )
                continue
            candidates.append((clearance_weight, path, metrics, refinement))

        if reference_length is None:
            raise PlanningError("no collision-free reference path")
        eligible = [
            candidate
            for candidate in candidates
            if candidate[2].reference_length_ratio
            <= self.config.maximum_safe_detour_ratio + 1e-6
        ]
        if not candidates:
            raise PlanningError("no curvature-feasible smoothed path")
        if not eligible:
            raise PlanningError("all candidate paths exceed the safe detour budget")
        frontier = _safety_efficiency_frontier(eligible)

        def score(candidate: RouteCandidate) -> tuple[float, float, float]:
            metrics = candidate[2]
            quality = (
                metrics.risk_density
                + self.config.curvature_weight * metrics.curvature_p95
                + self.config.length_tiebreak_weight * metrics.reference_length_ratio
            )
            return quality, -metrics.clearance_p05_m, metrics.length_m

        selected_weight, selected_path, selected_metrics, selected_refinement = min(
            frontier, key=score
        )
        eligible_weights = {candidate[0] for candidate in eligible}
        frontier_weights = {candidate[0] for candidate in frontier}
        candidate_summary = tuple(
            {
                "clearance_weight": float(weight),
                "valid": True,
                "length_m": float(metrics.length_m),
                "reference_length_ratio": float(metrics.reference_length_ratio),
                "clearance_p05_m": float(metrics.clearance_p05_m),
                "risk_density": float(metrics.risk_density),
                "scan_refinement_accepted": bool(refinement["accepted"]),
                "detour_eligible": weight in eligible_weights,
                "safety_efficiency_dominated": (
                    weight in eligible_weights and weight not in frontier_weights
                ),
            }
            for weight, _, metrics, refinement in candidates
        ) + tuple(failures)
        difficulty, difficulty_tags = _difficulty_labels(selected_metrics)
        return PlannedRoute(
            path_xy=selected_path.astype(np.float32),
            start_xy=start_xy.astype(np.float32),
            goal_xy=goal_xy.astype(np.float32),
            selected_clearance_weight=float(selected_weight),
            difficulty=difficulty,
            difficulty_tags=difficulty_tags,
            metrics=selected_metrics,
            candidates=candidate_summary,
            refinement=selected_refinement,
        )

    def _astar(
        self,
        start: np.ndarray,
        goal: np.ndarray,
        clearance_weight: float,
    ) -> np.ndarray | None:
        free = self.grid.free
        clearance = self.grid.clearance_m
        shape = free.shape
        start_cell = (int(start[0]), int(start[1]))
        goal_cell = (int(goal[0]), int(goal[1]))
        g_score = np.full(shape, np.inf, dtype=np.float64)
        parent = np.full((*shape, 2), -1, dtype=np.int32)
        closed = np.zeros(shape, dtype=bool)
        g_score[start_cell] = 0.0
        heap: list[tuple[float, float, int, int]] = [
            (self._heuristic(start_cell, goal_cell), 0.0, *start_cell)
        ]
        preferred = self.config.preferred_clearance_m

        while heap:
            _, current_g, x, y = heapq.heappop(heap)
            if closed[x, y] or current_g > g_score[x, y] + 1e-12:
                continue
            if (x, y) == goal_cell:
                return _reconstruct(parent, start_cell, goal_cell)
            closed[x, y] = True
            for dx, dy, step_cells in self._NEIGHBORS:
                nx, ny = x + dx, y + dy
                if not (0 <= nx < shape[0] and 0 <= ny < shape[1]) or not free[nx, ny]:
                    continue
                if dx and dy and (not free[x + dx, y] or not free[x, y + dy]):
                    continue
                local_clearance = 0.5 * (float(clearance[x, y]) + float(clearance[nx, ny]))
                risk = math.exp(-local_clearance / preferred)
                step_cost = (
                    step_cells
                    * self.grid.cell_size_m
                    * (1.0 + clearance_weight * risk)
                )
                tentative = current_g + step_cost
                if tentative + 1e-12 >= g_score[nx, ny]:
                    continue
                g_score[nx, ny] = tentative
                parent[nx, ny] = (x, y)
                heuristic = self._heuristic((nx, ny), goal_cell)
                heapq.heappush(heap, (tentative + heuristic, tentative, nx, ny))
        return None

    def _heuristic(self, cell: tuple[int, int], goal: tuple[int, int]) -> float:
        return math.hypot(cell[0] - goal[0], cell[1] - goal[1]) * self.grid.cell_size_m

    def _shortcut(self, path: np.ndarray) -> np.ndarray:
        result = [path[0]]
        index = 0
        while index < len(path) - 1:
            next_index = len(path) - 1
            while next_index > index + 1:
                segment = np.vstack([path[index], path[next_index]])
                if self.grid.path_is_safe(
                    segment,
                    minimum_clearance_m=self.config.minimum_clearance_m,
                    sample_step_m=self.config.validation_step_m,
                ):
                    break
                next_index -= 1
            result.append(path[next_index])
            index = next_index
        return np.asarray(result, dtype=np.float64)

    def _safe_smooth(
        self, path: np.ndarray
    ) -> tuple[np.ndarray, dict[str, float | int | bool | str]]:
        safe_candidates: list[np.ndarray] = []
        if len(path) >= 3:
            segment = np.linalg.norm(np.diff(path, axis=0), axis=1)
            cumulative = np.concatenate([[0.0], np.cumsum(segment)])
            if cumulative[-1] > 1e-8:
                u = cumulative / cumulative[-1]
                weights = np.ones(len(path), dtype=np.float64)
                weights[[0, -1]] = 1e3
                for tolerance in (
                    self.config.spline_smoothing_m,
                    self.config.spline_smoothing_m / 2.0,
                    0.0,
                ):
                    try:
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore", RuntimeWarning)
                            spline, _ = splprep(
                                path.T,
                                u=u,
                                w=weights,
                                k=min(3, len(path) - 1),
                                s=len(path) * tolerance**2,
                            )
                        count = max(
                            2,
                            int(np.ceil(cumulative[-1] / self.config.output_spacing_m)) + 1,
                        )
                        smoothed = np.column_stack(
                            splev(np.linspace(0.0, 1.0, count), spline)
                        )
                        smoothed[[0, -1]] = path[[0, -1]]
                    except (TypeError, ValueError):
                        continue
                    if self.grid.path_is_safe(
                        smoothed,
                        minimum_clearance_m=self.config.minimum_clearance_m,
                        sample_step_m=self.config.validation_step_m,
                    ):
                        safe_candidates.append(smoothed)

        for ratio in (0.25, 0.15, 0.08):
            smoothed = path.copy()
            for _ in range(self.config.chaikin_iterations):
                smoothed = _chaikin(smoothed, ratio)
            smoothed = resample_polyline(smoothed, self.config.output_spacing_m)
            if self.grid.path_is_safe(
                smoothed,
                minimum_clearance_m=self.config.minimum_clearance_m,
                sample_step_m=self.config.validation_step_m,
            ):
                safe_candidates.append(smoothed)
        baseline = (
            resample_polyline(path, self.config.output_spacing_m)
            if not safe_candidates
            else min(
                safe_candidates,
                key=lambda candidate: (
                    float(np.percentile(_curvature(candidate), 95)),
                    _path_length(candidate),
                ),
            )
        )
        result = refine_scan_style_trajectory(
            baseline,
            self.grid,
            minimum_clearance_m=self.config.minimum_clearance_m,
            validation_step_m=self.config.validation_step_m,
            output_spacing_m=self.config.output_spacing_m,
        )
        return result.path_xy, result.diagnostics

    def _metrics(self, path: np.ndarray, reference_length: float) -> RouteMetrics:
        length = _path_length(path)
        euclidean = float(np.linalg.norm(path[-1] - path[0]))
        clearance = self.grid.sample_clearance(path)
        if not np.isfinite(clearance).all():
            raise PlanningError("non-finite clearance in a supposedly safe path")
        trim = max(1, int(round(len(clearance) * 0.05)))
        clearance_core = (
            clearance[trim:-trim] if len(clearance) > 2 * trim + 2 else clearance
        )
        segment = np.diff(path, axis=0)
        ds = np.linalg.norm(segment, axis=1).clip(1e-8)
        heading = np.arctan2(segment[:, 1], segment[:, 0])
        turn = np.arctan2(np.sin(np.diff(heading)), np.cos(np.diff(heading)))
        curvature = _curvature(path)
        risk = np.exp(-clearance / self.config.preferred_clearance_m)
        return RouteMetrics(
            length_m=float(length),
            euclidean_distance_m=euclidean,
            geodesic_ratio=float(length / max(euclidean, 1e-6)),
            reference_length_ratio=float(length / max(reference_length, 1e-6)),
            minimum_clearance_m=float(np.min(clearance)),
            clearance_p05_m=float(np.percentile(clearance_core, 5)),
            mean_clearance_m=float(np.mean(clearance)),
            risk_density=float(np.mean(risk)),
            preferred_clearance_exposure=float(
                np.mean(clearance < self.config.preferred_clearance_m)
            ),
            total_turn_radians=float(np.sum(np.abs(turn))),
            curvature_p95=float(np.percentile(curvature, 95)),
            maximum_curvature=float(np.max(curvature)),
        )


def _reconstruct(
    parent: np.ndarray,
    start: tuple[int, int],
    goal: tuple[int, int],
) -> np.ndarray:
    path = [goal]
    cell = goal
    while cell != start:
        previous = parent[cell]
        if previous[0] < 0:
            raise PlanningError("A* parent chain is incomplete")
        cell = (int(previous[0]), int(previous[1]))
        path.append(cell)
    path.reverse()
    return np.asarray(path, dtype=np.int32)


def _chaikin(path: np.ndarray, ratio: float) -> np.ndarray:
    if len(path) < 3:
        return path.copy()
    output = [path[0]]
    for first, second in zip(path[:-1], path[1:]):
        output.append((1.0 - ratio) * first + ratio * second)
        output.append(ratio * first + (1.0 - ratio) * second)
    output.append(path[-1])
    return np.asarray(output, dtype=np.float64)


def _path_length(path: np.ndarray) -> float:
    return float(np.linalg.norm(np.diff(path, axis=0), axis=1).sum())


def _curvature(path: np.ndarray) -> np.ndarray:
    if len(path) < 3:
        return np.zeros(1, dtype=np.float64)
    first = path[1:-1] - path[:-2]
    second = path[2:] - path[1:-1]
    chord = path[2:] - path[:-2]
    denominator = (
        np.linalg.norm(first, axis=1)
        * np.linalg.norm(second, axis=1)
        * np.linalg.norm(chord, axis=1)
    ).clip(1e-8)
    twice_area = 2.0 * np.abs(first[:, 0] * second[:, 1] - first[:, 1] * second[:, 0])
    curvature = twice_area / denominator
    trim = max(1, int(round(len(curvature) * 0.03)))
    return curvature[trim:-trim] if len(curvature) > 2 * trim + 2 else curvature


def _difficulty_labels(metrics: RouteMetrics) -> tuple[str, tuple[str, ...]]:
    severity: dict[str, float] = {}
    if metrics.preferred_clearance_exposure > 0.4 or metrics.clearance_p05_m < 0.2:
        severity["narrow"] = max(
            metrics.preferred_clearance_exposure / 0.4,
            0.2 / max(metrics.clearance_p05_m, 1e-3),
        )
    if metrics.geodesic_ratio > 1.25:
        severity["detour"] = (metrics.geodesic_ratio - 1.0) / 0.25
    if metrics.total_turn_radians > math.pi / 2 or metrics.curvature_p95 > 1.0:
        severity["turning"] = max(
            metrics.total_turn_radians / (math.pi / 2),
            metrics.curvature_p95,
        )
    if not severity:
        return "open", ("open",)
    tags = tuple(sorted(severity))
    primary = max(severity, key=lambda key: (severity[key], key))
    return primary, tags
