"""One source configuration-space oracle for data preparation and evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from scipy.ndimage import distance_transform_edt
from torch import Tensor

from curvenav.physical import EXTRA_CLEARANCE_M, PATH_CONFIGURATION_QUERY_SPACING_M
from curvenav.trajectory import path_arc_length, resample_path_to_horizon


SOURCE_CONFIGURATION_QUERY_SPACING_M = PATH_CONFIGURATION_QUERY_SPACING_M
SOURCE_CONFIGURATION_QUERY_TYPE = "source_dingo_signed_clearance_cell_lookup"
SOURCE_CONFIGURATION_PATH_SAMPLING = "endpoint_inclusive_max_spacing"


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
        """Look up the source grid with its native piecewise-constant cells."""
        world_xy = np.asarray(world_xy, dtype=np.float64)
        if world_xy.ndim != 2 or world_xy.shape[1] != 2:
            raise ValueError("world_xy must have shape [N,2]")
        cells = np.floor(
            (world_xy - self.origin_xy) / self.cell_size_m
        ).astype(np.int64)
        values = np.full(len(cells), -np.inf, dtype=np.float32)
        inside = (
            (cells[:, 0] >= 0)
            & (cells[:, 0] < self.signed_clearance_m.shape[0])
            & (cells[:, 1] >= 0)
            & (cells[:, 1] < self.signed_clearance_m.shape[1])
        )
        values[inside] = self.signed_clearance_m[
            cells[inside, 0], cells[inside, 1]
        ]
        return values

@dataclass(frozen=True)
class SourcePathQuery:
    """One dense source-grid measurement of a robot-frame trajectory."""

    local_path: Tensor
    clearance_m: Tensor
    in_world_bounds: Tensor
    active: Tensor

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

    The oracle is training-only provenance: it validates expert B-splines and
    independently evaluates trajectories.  It is never part of the deployed
    depth/PointGoal policy input.
    """

    def __init__(self, grids: tuple[SourceConfigurationGrid, ...]) -> None:
        if not grids:
            raise ValueError("source configuration query requires at least one grid")
        self.grids = grids
        self._grid_tensors: dict[
            tuple[int, str, int | None], tuple[Tensor, Tensor]
        ] = {}

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

    def _grid_tensors_for(
        self,
        grid_index: int,
        device: torch.device,
    ) -> tuple[Tensor, Tensor]:
        cache_key = (grid_index, device.type, device.index)
        cached = self._grid_tensors.get(cache_key)
        if cached is not None:
            return cached
        grid = self.grids[grid_index]
        values = torch.from_numpy(grid.signed_clearance_m).to(device=device)
        origin = torch.as_tensor(grid.origin_xy, dtype=torch.float32, device=device)
        self._grid_tensors[cache_key] = (values, origin)
        return values, origin

    def query(
        self,
        path: Tensor,
        grid_index: Tensor,
        world_origin_xy: Tensor,
        world_yaw_rad: Tensor,
        planning_horizon_m: float,
    ) -> SourcePathQuery:
        """Return native-grid clearance; any source-grid OOB point is unsafe."""
        if path.ndim != 3 or path.shape[-1] != 2:
            raise ValueError("path must have shape [B,P,2]")
        batch = path.shape[0]
        if grid_index.shape != (batch,):
            raise ValueError("grid_index must have shape [B]")
        if world_origin_xy.shape != (batch, 2):
            raise ValueError("world_origin_xy must have shape [B,2]")
        if world_yaw_rad.shape != (batch,):
            raise ValueError("world_yaw_rad must have shape [B]")
        indices = grid_index.long()
        if (indices < 0).any() or (indices >= len(self.grids)).any():
            raise ValueError("source grid index is outside the prepared grid table")
        local, active = resample_path_to_horizon(
            path.float(),
            planning_horizon_m,
            SOURCE_CONFIGURATION_QUERY_SPACING_M,
        )
        origin = world_origin_xy.float()
        yaw = world_yaw_rad.float()
        cosine, sine = yaw.cos(), yaw.sin()
        world = torch.stack(
            (
                origin[:, None, 0]
                + cosine[:, None] * local[..., 0]
                + sine[:, None] * local[..., 1],
                origin[:, None, 1]
                + sine[:, None] * local[..., 0]
                - cosine[:, None] * local[..., 1],
            ),
            dim=-1,
        )
        clearance = torch.full(
            world.shape[:2],
            -float(planning_horizon_m),
            device=world.device,
            dtype=torch.float32,
        )
        in_bounds = torch.zeros_like(clearance, dtype=torch.bool)
        for index in indices.unique(sorted=True).tolist():
            selected = indices == index
            values, grid_origin = self._grid_tensors_for(int(index), world.device)
            cells = torch.floor(
                (world[selected] - grid_origin) / self.grids[int(index)].cell_size_m
            ).long()
            inside = (
                (cells[..., 0] >= 0)
                & (cells[..., 0] < values.shape[0])
                & (cells[..., 1] >= 0)
                & (cells[..., 1] < values.shape[1])
            )
            selected_clearance = torch.full_like(cells[..., 0], -float(planning_horizon_m), dtype=torch.float32)
            selected_clearance[inside] = values[
                cells[..., 0][inside], cells[..., 1][inside]
            ]
            clearance[selected] = selected_clearance
            in_bounds[selected] = inside
        return SourcePathQuery(local, clearance, in_bounds, active)

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
