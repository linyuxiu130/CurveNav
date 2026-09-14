"""Shared geometry and clearance-aware route planning."""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
import hashlib
import heapq
import math
from pathlib import Path
from typing import Any

import numpy as np
from scipy.interpolate import BSpline, PPoly, RectBivariateSpline
from scipy.integrate import quad_vec, solve_ivp
from scipy.optimize import brentq, minimize
from scipy.ndimage import distance_transform_edt

from curvenav.physical import EXTRA_CLEARANCE_M, ROBOT_FOOTPRINT_RADIUS_M


SAFETY_STEP_M = 0.025
MIN_CLEARANCE_M = EXTRA_CLEARANCE_M


class PlanningError(RuntimeError):
    pass


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


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

    @cached_property
    def clearance_spline(self):
        # Smooth optimization surrogate. Acceptance still uses native grid cells.
        signed = np.where(
            self.free, self.clearance_m,
            (1 - distance_transform_edt(~self.free)) * self.cell_size_m,
        )
        axes = [
            self.origin_xy[k] + (np.arange(n) + 0.5) * self.cell_size_m
            for k, n in enumerate(self.free.shape)
        ]
        return RectBivariateSpline(*axes, signed)

    def clearance_gradient(self, xy: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Continuous surrogate value and its exact spline derivative."""
        x, y = xy.T
        field = self.clearance_spline
        return field.ev(x, y), np.column_stack(
            (field.ev(x, y, dx=1), field.ev(x, y, dy=1))
        )

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


@dataclass(frozen=True)
class Plan:
    curve: BSpline
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
            trial = current + step * grid.cell_size_m * clearance_density(clear)
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
    return float(np.sum(segment_length * clearance_density(segment_clearance)))


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


def clearance_density(clearance):
    """Dimensionless travel cost; extra clearance is scaled by body radius."""
    return 1.0 + np.exp(1.0 - clearance / ROBOT_FOOTPRINT_RADIUS_M)


def _travel_objective(controls, basis, first, weights, grid):
    """Integral rho(clearance) ds and its control-point gradient."""
    points, tangent = basis @ controls, first @ controls
    speed = np.linalg.norm(tangent, axis=1)
    if np.any(speed <= np.finfo(float).eps):
        raise PlanningError("curve optimization encountered a stationary tangent")
    distance, distance_gradient = grid.clearance_gradient(points)
    density = clearance_density(distance)
    dp = -(density - 1)[:, None] / ROBOT_FOOTPRINT_RADIUS_M * distance_gradient * speed[:, None]
    dv = density[:, None] * tangent / speed[:, None]
    return float(weights @ (density * speed)), (
        basis.T @ (weights[:, None] * dp) + first.T @ (weights[:, None] * dv)
    )


def _bending_objective(controls, first, second, weights):
    """Integral r_body^2 * curvature^2 ds, independent of parameterization."""
    tangent, acceleration = first @ controls, second @ controls
    speed = np.linalg.norm(tangent, axis=1)
    cross = tangent[:, 0] * acceleration[:, 1] - tangent[:, 1] * acceleration[:, 0]
    radius2 = ROBOT_FOOTPRINT_RADIUS_M**2
    energy = radius2 * cross**2 / speed**5
    dv = ((2 * radius2 * cross / speed**5)[:, None]
          * np.column_stack((acceleration[:, 1], -acceleration[:, 0]))
          - (5 * energy / speed**2)[:, None] * tangent)
    da = (2 * radius2 * cross / speed**5)[:, None] * np.column_stack(
        (-tangent[:, 1], tangent[:, 0]))
    return float(weights @ energy), (
        first.T @ (weights[:, None] * dv) + second.T @ (weights[:, None] * da)
    )


def _smooth(grid: Grid, path: np.ndarray, start_yaw: float) -> tuple[BSpline, np.ndarray]:
    """Optimize one continuous curve for length, clearance and turning effort."""
    length = path_length(path)
    # One cubic representation at every length, including sub-body-length
    # routes. Four controls are the cubic basis minimum, not a turn constraint.
    count = max(4, math.ceil(length / (2 * ROBOT_FOOTPRINT_RADIUS_M)) + 1)
    controls = points_at_arc(path, np.linspace(0., length, count))
    degree = 3
    knots = np.r_[
        np.zeros(degree), np.linspace(0, 1, len(controls) - degree + 1), np.ones(degree)
    ]
    parameter = np.linspace(0, 1, max(2, math.ceil(length / SAFETY_STEP_M) + 1))
    identity = BSpline(knots, np.eye(len(controls)), degree)
    basis = identity(parameter)
    first, second = identity.derivative(), identity.derivative(2)
    # Gauss quadrature integrates straight spline segments exactly; trapezoids
    # can otherwise lower the apparent length by collapsing control spacing.
    edges = np.unique(np.r_[parameter, knots])
    nodes, gauss_weights = np.polynomial.legendre.leggauss(4)
    half = np.diff(edges) / 2
    travel_u = ((edges[:-1] + half)[:, None] + half[:, None] * nodes).ravel()
    weights = (half[:, None] * gauss_weights).ravel()
    travel_basis, travel_first = identity(travel_u), first(travel_u)

    # A clamped cubic has p'(0) parallel to P1-P0. Parameterize this
    # handle by a positive length, so the prescribed heading cannot flip.
    direction = np.array([math.cos(start_yaw), math.sin(start_yaw)])
    offset = controls[1:] - controls[0]
    seed_yaw = math.atan2(offset[0, 1], offset[0, 0])
    turn = math.atan2(math.sin(start_yaw - seed_yaw), math.cos(start_yaw - seed_yaw))
    # Preserve the obstacle-avoiding A* seed away from the initial turn.
    # This initializes controls only; every interior control remains optimized.
    arcs = np.arange(count - 1) * length / (count - 1)
    angles = turn * np.linspace(1., 0., count - 1) * np.exp(-arcs / (2 * ROBOT_FOOTPRINT_RADIUS_M))
    c, s = np.cos(angles), np.sin(angles)
    controls[1:] = controls[0] + np.column_stack(
        (c * offset[:, 0] - s * offset[:, 1], s * offset[:, 0] + c * offset[:, 1])
    )

    def assemble(values):
        handle = math.exp(values[0]) * direction
        return np.vstack((controls[0], controls[0] + handle,
                          values[1:].reshape(-1, 2), controls[-1]))

    def objective(values):
        assembled = assemble(values)
        # Split at every tangent-speed extremum: a narrow curvature peak
        # must not hide between fixed quadrature nodes.
        tangent = BSpline(knots, assembled, degree).derivative()
        pieces = [PPoly.from_spline((tangent.t, tangent.c[:, k], tangent.k))
                  for k in range(2)]
        breaks = list(np.unique(knots))
        for i, (lo, hi) in enumerate(zip(pieces[0].x[:-1], pieces[0].x[1:])):
            if hi <= lo:
                continue
            squared = np.polyadd(*(np.polymul(piece.c[:, i], piece.c[:, i]) for piece in pieces))
            for root in np.roots(np.polyder(squared)):
                if np.isreal(root) and 0 < root.real < hi - lo:
                    breaks.append(lo + float(root.real))

        scale = np.linalg.norm(tangent.c, axis=1).max()
        if np.any(np.linalg.norm(tangent(breaks), axis=1) <= np.finfo(float).eps * scale):
            # Infinite energy outside the regular-curve domain lets the
            # optimizer shorten its trial step without accepting a cusp.
            return math.inf, np.zeros_like(values)

        def integrand(u):
            value, gradient = _bending_objective(
                assembled, first([u]), second([u]), np.ones(1)
            )
            return np.r_[value, gradient.ravel()]

        integral, error = quad_vec(integrand, 0., 1., points=sorted(set(breaks)),
                                   epsabs=1e-7, epsrel=1e-6)
        if not np.isfinite(integral).all() or error > 1e-7 + 1e-6 * np.linalg.norm(integral):
            raise PlanningError("curve energy quadrature did not converge")
        # Clearance uses the navigation grid's existing spatial quadrature;
        # only curvature needs adaptive integration around tangent extrema.
        cost, gradient = _travel_objective(assembled, travel_basis, travel_first, weights, grid)
        cost += integral[0]
        gradient += integral[1:].reshape(assembled.shape)
        return cost, np.r_[gradient[1] @ (assembled[1] - assembled[0]),
                           gradient[2:-1].ravel()]

    def clearance(values):
        assembled = assemble(values)
        distance, gradient = grid.clearance_gradient(basis @ assembled)
        jacobian = np.column_stack((
            basis[:, 1] * (gradient @ (assembled[1] - assembled[0])),
            (basis[:, 2:-1, None] * gradient[:, None, :]).reshape(len(parameter), -1),
        ))
        return distance - MIN_CLEARANCE_M, jacobian

    lower = grid.origin_xy + grid.cell_size_m * 0.5
    upper = grid.origin_xy + (np.array(grid.free.shape) - 0.5) * grid.cell_size_m
    reach = min((upper[k] - controls[0, k]) / direction[k] if direction[k] > 0
                else (lower[k] - controls[0, k]) / direction[k]
                for k in range(2) if abs(direction[k]) > np.finfo(float).eps)
    initial = np.r_[math.log(np.linalg.norm(controls[1] - controls[0])),
                    controls[2:-1].ravel()]
    result = minimize(
        objective, initial, jac=True, method="SLSQP",
        bounds=[(math.log(np.finfo(float).eps * length), math.log(reach))]
               + list(zip(lower, upper)) * (len(controls) - 3),
        constraints={
            "type": "ineq",
            "fun": lambda x: clearance(x)[0],
            "jac": lambda x: clearance(x)[1],
        },
        options={"maxiter": 300, "ftol": 1e-6},
    )
    if not result.success:
        raise PlanningError(f"clearance curve optimization failed: {result.message}")
    curve = BSpline(knots, assemble(result.x), degree)
    # Derivative control hull bounds arc distance between certification samples.
    bound = np.linalg.norm(curve.derivative().c, axis=1).max()
    sampled = curve(np.linspace(0, 1, max(2, math.ceil(bound / SAFETY_STEP_M) + 1)))
    if not grid.safe(sampled):
        raise PlanningError("optimized curve leaves the source configuration space")
    return curve, sampled


def timed_route(curve: BSpline, period_s: float, speed_m_s: float,
                angular_speed_rad_s: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Time-parameterize the same analytic curve used for planning and rendering.

    dt/du=max(|p'|/v_max, |dtheta/du|/omega_max). Thus v>=0,
    lateral body velocity is zero, and both velocity limits hold pointwise.
    This is a kinematic model; no uncalibrated acceleration limit is invented.
    """
    first = curve.derivative()
    second = curve.derivative(2) if curve.k >= 2 else None

    def clock_rate(u, _):
        tangent = first(u)
        norm2 = float(tangent @ tangent)
        if norm2 <= np.finfo(float).eps ** 2:
            raise PlanningError("expert curve has a stationary tangent")
        acceleration = second(u) if second is not None else np.zeros(2)
        turn = (tangent[0] * acceleration[1] - tangent[1] * acceleration[0]) / norm2
        return [max(math.sqrt(norm2) / speed_m_s, abs(turn) / angular_speed_rad_s)]

    clock = solve_ivp(clock_rate, (0., 1.), [0.], rtol=1e-11, atol=1e-12,
                      max_step=1 / (8 * len(curve.c)), dense_output=True)
    if not clock.success:
        raise PlanningError(clock.message)
    duration = float(clock.y[0, -1])
    intervals = math.ceil(duration / period_s)
    # Stretch to the next sensor tick; never append a shorter final time step.
    ticks = np.linspace(0., duration, intervals + 1)
    u = np.array([0., *[brentq(lambda x: clock.sol(x)[0] - t, 0., 1.)
                         for t in ticks[1:-1]], 1.])
    tangent = first(u)
    xy = curve(u)
    yaw = np.arctan2(tangent[:, 1], tangent[:, 0])
    rates = np.array([clock_rate(x, None)[0] for x in u])
    du_dt = duration / (intervals * period_s) / rates
    v = np.linalg.norm(tangent, axis=1) * du_dt
    acceleration = second(u) if second is not None else np.zeros_like(tangent)
    w = (tangent[:, 0]*acceleration[:, 1]-tangent[:, 1]*acceleration[:, 0]) / (
        tangent*tangent).sum(1) * du_dt
    # Habitat XZ yaw is opposite to the policy's planning-world yaw.
    return xy, yaw, np.column_stack((v, -w))


def plan_route(
    grid: Grid,
    start_xy: np.ndarray,
    goal_xy: np.ndarray,
    start_yaw: float,
) -> Plan:
    start, goal = grid.snap(start_xy), grid.snap(goal_xy)
    cells = _astar(grid, start, goal)
    discrete_path = np.vstack([start_xy, grid.grid_to_world(cells), goal_xy])
    curve_spline, path = _smooth(
        grid,
        _simplify_route(grid, discrete_path),
        start_yaw,
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
            np.mean(clearance_density(clear) - 1)
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
    return Plan(curve_spline, path.astype(np.float32), metrics, tags[0], tuple(tags))


def source_route(grid: Grid, start_xy: np.ndarray, goal_xy: np.ndarray, start_yaw: float) -> Plan:
    plan = plan_route(grid, start_xy, goal_xy, start_yaw)
    if not grid.safe(plan.path_xy):
        raise PlanningError("source route violates continuous clearance")
    return plan


def candidate_pairs(
    grid: Grid, distance_range_m: tuple[float, float], seed: int, limit: int
) -> list[tuple[np.ndarray, np.ndarray]]:
    eligible = np.argwhere(grid.free & (grid.clearance_m + 1e-9 >= MIN_CLEARANCE_M))
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
            np.array_equal(start, goal)
            or np.any(goal < 0)
            or np.any(goal >= grid.free.shape)
            or not grid.free[tuple(goal)]
            or grid.clearance_m[tuple(goal)] + 1e-9 < MIN_CLEARANCE_M
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
