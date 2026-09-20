"""One source configuration-space oracle for supervision and evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from scipy.ndimage import distance_transform_edt
from torch import Tensor

from curvenav.physical import EXTRA_CLEARANCE_M, PATH_CONFIGURATION_QUERY_SPACING_M
from curvenav.trajectory import path_arc_length


SOURCE_CONFIGURATION_QUERY_SPACING_M = PATH_CONFIGURATION_QUERY_SPACING_M
SOURCE_CONFIGURATION_QUERY_TYPE = "source_dingo_signed_clearance_cell_lookup"
SOURCE_CONFIGURATION_PATH_SAMPLING = "closed_cell_supercover_max_spacing_v2"


def _closed_grid_cells(coordinates: Tensor) -> Tensor:
    """All cells touching a point; occupied cells include their edges and corners."""
    nearest = coordinates.round()
    # FP64 transform/interpolation roundoff at an analytically exact grid crossing.
    tolerance = 4 * torch.finfo(coordinates.dtype).eps * coordinates.abs().clamp_min(1)
    coordinates = torch.where((coordinates - nearest).abs() <= tolerance, nearest, coordinates)
    high = coordinates.floor().long()
    low = coordinates.ceil().long() - 1
    return torch.stack((
        high,
        torch.stack((low[..., 0], high[..., 1]), -1),
        torch.stack((high[..., 0], low[..., 1]), -1),
        low,
    ), -2)


@dataclass(frozen=True)
class SourceConfigurationGrid:
    """Dingo centre-clearance values on one immutable world XZ grid."""

    signed_clearance_m: np.ndarray
    origin_xy: np.ndarray
    cell_size_m: float

    @classmethod
    def load(cls, path: Path) -> "SourceConfigurationGrid":
        with np.load(path) as values:
            free = np.asarray(values["free"], dtype=np.bool_)
            clearance = np.asarray(values["clearance_m"], dtype=np.float32)
            origin = np.asarray(values["origin_xy"], dtype=np.float64)
            cell_size = float(values["cell_size_m"])
        if free.shape != clearance.shape or free.ndim != 2:
            raise ValueError(f"invalid navigation grid arrays: {path}")
        if origin.shape != (2,) or cell_size <= 0:
            raise ValueError(f"invalid navigation grid metric frame: {path}")
        distance_inside_obstacle = distance_transform_edt(~free).astype(np.float32)
        signed = np.where(
            free,
            clearance,
            -distance_inside_obstacle * cell_size,
        ).astype(np.float32)
        return cls(signed, origin, cell_size)

    def query_world(self, world_xy: np.ndarray) -> np.ndarray:
        """Use the same closed-cell clearance as continuous trajectory queries."""
        world_xy = np.asarray(world_xy, dtype=np.float64)
        if world_xy.ndim != 2 or world_xy.shape[1] != 2:
            raise ValueError("world_xy must have shape [N,2]")
        cells = _closed_grid_cells(torch.from_numpy(
            (world_xy - self.origin_xy) / self.cell_size_m
        )).numpy()
        values = np.full(cells.shape[:-1], -np.inf, dtype=np.float32)
        inside = (
            (cells[..., 0] >= 0)
            & (cells[..., 0] < self.signed_clearance_m.shape[0])
            & (cells[..., 1] >= 0)
            & (cells[..., 1] < self.signed_clearance_m.shape[1])
        )
        values[inside] = self.signed_clearance_m[
            cells[inside, 0], cells[inside, 1]
        ]
        return values.min(-1)

@dataclass(frozen=True)
class SourcePathQuery:
    """One dense source-grid measurement of a robot-frame trajectory."""

    local_path: Tensor
    clearance_m: Tensor
    in_world_bounds: Tensor
    active: Tensor
    grid_cells: Tensor

    @property
    def minimum_clearance_m(self) -> Tensor:
        return self.clearance_m.masked_fill(~self.active, torch.inf).amin(dim=-1)

    @property
    def path_field_coverage_fraction(self) -> Tensor:
        return (self.in_world_bounds & self.active).sum(dim=-1) / self.active.sum(
            dim=-1
        ).clamp_min(1)


class SourceConfigurationSpaceQuery:
    """Query immutable source C-space at one spacing with one OOB definition.

    The oracle validates expert B-splines, supervises candidate scores, and
    independently evaluates trajectories.  It is never part of the deployed
    depth/PointGoal policy input.
    """

    def __init__(self, grids: tuple[SourceConfigurationGrid, ...]) -> None:
        if not grids:
            raise ValueError("source configuration query requires at least one grid")
        self.grids = grids
        self._atlases: dict[torch.device, tuple[Tensor, ...]] = {}

    @classmethod
    def from_paths(cls, paths: tuple[Path, ...]) -> "SourceConfigurationSpaceQuery":
        return cls(tuple(SourceConfigurationGrid.load(path) for path in paths))

    @classmethod
    def from_prepared_split(
        cls,
        root: str | Path,
        split: str,
    ) -> "SourceConfigurationSpaceQuery":
        split_root = Path(root).expanduser().resolve() / split
        import json

        manifest = json.loads((split_root / "manifest.json").read_text(encoding="utf-8"))
        source = manifest.get("source_configuration_space", {})
        if (
            source.get("query") != SOURCE_CONFIGURATION_QUERY_TYPE
            or source.get("spacing_m") != SOURCE_CONFIGURATION_QUERY_SPACING_M
            or source.get("path_sampling") != SOURCE_CONFIGURATION_PATH_SAMPLING
            or source.get("out_of_bounds") != "non_executable_negative_clearance"
        ):
            raise ValueError("prepared source configuration-space contract mismatch")
        paths = tuple(
            split_root / str(entry["file"])
            for entry in source.get("grids", ())
        )
        if not all(path.is_file() for path in paths):
            raise ValueError("prepared source configuration grid is missing")
        return cls.from_paths(paths)

    def _atlas(self, device: torch.device) -> tuple[Tensor, ...]:
        """Pack immutable maps once; all scenes share one batched lookup."""
        if device not in self._atlases:
            sizes = np.array([grid.signed_clearance_m.shape for grid in self.grids], dtype=np.int64)
            offsets = np.r_[0, np.prod(sizes, axis=1).cumsum()[:-1]]
            self._atlases[device] = tuple(torch.as_tensor(value, device=device) for value in (
                np.concatenate([grid.signed_clearance_m.ravel() for grid in self.grids]),
                np.stack([grid.origin_xy for grid in self.grids]),
                np.array([grid.cell_size_m for grid in self.grids], dtype=np.float64),
                sizes, offsets,
            ))
        return self._atlases[device]

    def grid_coordinates(
        self, local: Tensor, indices: Tensor, origin: Tensor, yaw: Tensor,
    ) -> Tensor:
        """One FP64 robot-to-native-grid transform, outside neural autocast."""
        _, grid_origins, cell_sizes, _, _ = self._atlas(local.device)
        local, origin, yaw = local.double(), origin.double(), yaw.double()
        cosine, sine = yaw.cos()[:, None], yaw.sin()[:, None]
        world = torch.stack((
            origin[:, None, 0] + cosine * local[..., 0] + sine * local[..., 1],
            origin[:, None, 1] + sine * local[..., 0] - cosine * local[..., 1],
        ), -1)
        return (world - grid_origins[indices, None]) / cell_sizes[indices, None, None]

    def point_cells(self, local: Tensor, indices: Tensor, origin: Tensor, yaw: Tensor) -> Tensor:
        return _closed_grid_cells(self.grid_coordinates(local, indices, origin, yaw))[..., 0, :]

    @staticmethod
    def _trace_segments(path: Tensor, grid_path: Tensor, horizon: float) -> tuple[Tensor, Tensor, Tensor]:
        """Sample every cell interior and boundary crossed, preserving vertices.

        Grid-boundary crossings partition each original segment into intervals
        lying in a single cell. Their midpoints cannot miss a short corner cut.
        Metric subdivisions also bound the spacing of visibility diagnostics.
        """
        delta = path[:, 1:].double() - path[:, :-1].double()
        length = delta.norm(dim=-1)
        start_arc = length.cumsum(-1) - length
        fraction = ((horizon - start_arc).clamp_min(0) / length.clamp_min(1e-30)).clamp_max(1)
        end = path[:, :-1].double() + fraction[..., None] * delta
        grid_start = grid_path[:, :-1]
        grid_end = grid_start + fraction[..., None] * (grid_path[:, 1:] - grid_start)
        grid_delta = grid_end - grid_start
        used_length = length * fraction
        events = [torch.zeros_like(length[..., None]), torch.ones_like(length[..., None])]
        for axis in range(2):
            low = torch.minimum(grid_start[..., axis], grid_end[..., axis]).floor()
            high = torch.maximum(grid_start[..., axis], grid_end[..., axis]).floor()
            count = int((high - low).max().item())
            boundaries = low[..., None] + torch.arange(1, count + 1, device=path.device)
            direction = grid_delta[..., axis, None]
            t = (boundaries - grid_start[..., axis, None]) / torch.where(direction != 0, direction, 1)
            events.append(torch.where((direction != 0) & (t > 0) & (t < 1), t, 1))
        spacing = SOURCE_CONFIGURATION_QUERY_SPACING_M
        count = int(torch.ceil(used_length.max() / spacing).item())
        distances = torch.arange(max(0, count - 1), device=path.device).add(1) * spacing
        events.append((distances / used_length[..., None].clamp_min(1e-30)).clamp_max(1))
        cuts = torch.cat(events, -1).sort(-1).values
        left, right = cuts[..., :-1], cuts[..., 1:]
        # Midpoint covers each crossed cell; the right endpoint also checks
        # isolated contacts at grid corners and retains every original vertex.
        t = torch.stack(((left + right) * 0.5, right), -1).flatten(-2)
        local = (path[:, :-1, None].double()
                 + t[..., None] * (end - path[:, :-1])[..., None, :]).flatten(1, 2)
        coordinates = (grid_start[..., None, :] + t[..., None] * grid_delta[..., None, :]).flatten(1, 2)
        valid = ((right > left) & (used_length[..., None] > 0)).repeat_interleave(2, -1).flatten(1, 2)
        local = torch.cat((path[:, :1].double(), local), 1)
        coordinates = torch.cat((grid_path[:, :1], coordinates), 1)
        valid = torch.cat((torch.ones_like(valid[:, :1]), valid), 1)
        # Stable compaction keeps active samples contiguous and in execution order.
        size = int(valid.sum(-1).max().item())
        order = (~valid).to(torch.int32).argsort(dim=-1, stable=True)[:, :size]
        active = valid.gather(1, order)
        local = local.gather(1, order[..., None].expand(-1, -1, 2))
        coordinates = coordinates.gather(1, order[..., None].expand(-1, -1, 2))
        last = (active.sum(-1) - 1)[:, None, None].expand(-1, 1, 2)
        local = torch.where(active[..., None], local, local.gather(1, last))
        coordinates = torch.where(active[..., None], coordinates, coordinates.gather(1, last))
        return local.float(), coordinates, active

    def query(
        self, path: Tensor, grid_index: Tensor, world_origin_xy: Tensor,
        world_yaw_rad: Tensor, planning_horizon_m: float,
    ) -> SourcePathQuery:
        """Trace the polyline against closed occupied cells, including corner contact."""
        if path.ndim != 3 or path.shape[-1] != 2 or path.shape[1] < 2:
            raise ValueError("path must have shape [B,P,2], P >= 2")
        batch = path.shape[0]
        if grid_index.shape != (batch,) or world_origin_xy.shape != (batch, 2) or world_yaw_rad.shape != (batch,):
            raise ValueError("source grid poses must match the path batch")
        if planning_horizon_m <= 0:
            raise ValueError("planning horizon must be positive")
        indices = grid_index.long()
        if (indices < 0).any() or (indices >= len(self.grids)).any():
            raise ValueError("source grid index is outside the prepared grid table")
        coordinates = self.grid_coordinates(path, indices, world_origin_xy, world_yaw_rad)
        local, coordinates, active = self._trace_segments(path, coordinates, planning_horizon_m)
        cells = _closed_grid_cells(coordinates)
        values, _, _, sizes, offsets = self._atlas(path.device)
        shape = sizes[indices, None, None]
        inside = (cells >= 0).all(-1) & (cells < shape).all(-1)
        address = offsets[indices, None, None] + cells[..., 0] * shape[..., 1] + cells[..., 1]
        # Select a valid address before the gather; out-of-map values remain unsafe.
        clearance = values[torch.where(inside, address, 0)].masked_fill(~inside, -planning_horizon_m)
        # Clearance covers every touching cell; progress uses the canonical cell
        # only after the common collision-free prefix has been established.
        return SourcePathQuery(local, clearance.amin(-1), inside.all(-1), active, cells[..., 0, :])

    def safety_metrics(
        self,
        path: Tensor,
        grid_index: Tensor,
        world_origin_xy: Tensor,
        world_yaw_rad: Tensor,
        planning_horizon_m: float,
    ) -> dict[str, Tensor]:
        """Use the source oracle as the sole physical safety metric."""
        query = self.query(
            path,
            grid_index,
            world_origin_xy,
            world_yaw_rad,
            planning_horizon_m,
        )
        return self.safety_metrics_from_query(path, query, planning_horizon_m)

    @staticmethod
    def safety_metrics_from_query(
        path: Tensor,
        query: SourcePathQuery,
        planning_horizon_m: float,
    ) -> dict[str, Tensor]:
        """Summarize one previously sampled source-C-space trajectory."""
        minimum = query.minimum_clearance_m
        return {
            "path_field_coverage_fraction": query.path_field_coverage_fraction,
            "min_clearance_m": minimum,
            "footprint_collision": minimum < 0.0,
            "safety_margin_violation": minimum < EXTRA_CLEARANCE_M,
            "max_margin_violation_m": torch.relu(EXTRA_CLEARANCE_M - minimum),
            "arc_length_beyond_local_horizon_m": torch.relu(
                path_arc_length(path.float()) - planning_horizon_m
            ),
        }
