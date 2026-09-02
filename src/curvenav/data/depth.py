"""Calibrated metric-depth reprojection shared by data preparation and deployment."""

from dataclasses import dataclass
from functools import lru_cache

import numpy as np

from curvenav.config import DataConfig


@dataclass(frozen=True)
class PinholeIntrinsics:
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float

    @classmethod
    def from_matrix(
        cls,
        matrix: np.ndarray,
        *,
        width: int,
        height: int,
    ) -> "PinholeIntrinsics":
        value = np.asarray(matrix, dtype=np.float64)
        if value.shape != (3, 3) or not np.isfinite(value).all():
            raise ValueError("camera intrinsic matrix must be finite [3,3]")
        expected_last_row = np.array([0.0, 0.0, 1.0])
        if not np.allclose(value[2], expected_last_row, atol=1e-8, rtol=0.0):
            raise ValueError("camera intrinsic matrix must be pinhole-normalized")
        if value[0, 1] != 0.0 or value[1, 0] != 0.0:
            raise ValueError("CurveNav requires a zero-skew pinhole camera")
        if value[0, 0] <= 0.0 or value[1, 1] <= 0.0:
            raise ValueError("camera focal lengths must be positive")
        return cls(
            width=int(width),
            height=int(height),
            fx=float(value[0, 0]),
            fy=float(value[1, 1]),
            cx=float(value[0, 2]),
            cy=float(value[1, 2]),
        )

    def matrix(self) -> np.ndarray:
        return np.array(
            ((self.fx, 0.0, self.cx), (0.0, self.fy, self.cy), (0.0, 0.0, 1.0)),
            dtype=np.float64,
        )

    def at_resolution(self, width: int, height: int) -> "PinholeIntrinsics":
        """Resample the same horizontal and vertical field of view."""
        if width < 1 or height < 1:
            raise ValueError("pinhole resolution must be positive")
        scale_x = width / self.width
        scale_y = height / self.height
        return PinholeIntrinsics(
            width=width,
            height=height,
            fx=self.fx * scale_x,
            fy=self.fy * scale_y,
            cx=self.cx * scale_x,
            cy=self.cy * scale_y,
        )


BENCHMARK_INTRINSICS = PinholeIntrinsics(
    width=640,
    height=360,
    fx=326.398559570312,
    fy=326.398559570312,
    cx=320.0,
    cy=180.0,
)
CANONICAL_INTRINSICS = PinholeIntrinsics(
    width=224,
    height=126,
    fx=166.80851063829786,
    fy=166.80851063829786,
    cx=112.0,
    cy=63.0,
)


def depth_camera_contract(data: DataConfig) -> dict[str, int | float]:
    """Return the one metric-depth calibration used by every data source."""
    return {
        "image_height": data.image_height,
        "image_width": data.image_width,
        "max_depth_m": data.max_depth_m,
        "canonical_focal_x_px": data.canonical_focal_x_px,
        "canonical_focal_y_px": data.canonical_focal_y_px,
        "camera_forward_offset_m": data.camera_forward_offset_m,
        "camera_height_m": data.camera_height_m,
        "camera_downward_pitch_degrees": data.camera_downward_pitch_degrees,
    }


@lru_cache(maxsize=8)
def _pinhole_remap(
    source: PinholeIntrinsics,
    target: PinholeIntrinsics,
) -> tuple[np.ndarray, np.ndarray]:
    row, column = np.meshgrid(
        np.arange(target.height, dtype=np.float32),
        np.arange(target.width, dtype=np.float32),
        indexing="ij",
    )
    ray_x = (column - target.cx) / target.fx
    ray_y = (row - target.cy) / target.fy
    source_x = source.fx * ray_x + source.cx
    source_y = source.fy * ray_y + source.cy
    return source_x.astype(np.float32), source_y.astype(np.float32)


def preprocess_metric_depth(
    depth_m: np.ndarray,
    *,
    source_intrinsics: PinholeIntrinsics,
    maximum_m: float,
) -> np.ndarray:
    """Reproject one calibrated FoV at its delivered sampling resolution."""
    frame = np.asarray(depth_m, dtype=np.float32)
    if frame.ndim != 2:
        raise ValueError(f"depth must be a two-dimensional image, got {frame.shape}")
    sampled_intrinsics = source_intrinsics.at_resolution(
        width=frame.shape[1],
        height=frame.shape[0],
    )
    invalid = ~np.isfinite(frame) | (frame <= 0.0)
    frame = frame.copy()
    frame[invalid] = maximum_m
    map_x, map_y = _pinhole_remap(sampled_intrinsics, CANONICAL_INTRINSICS)
    source_x = np.floor(map_x + 0.5).astype(np.int64)
    source_y = np.floor(map_y + 0.5).astype(np.int64)
    inside = (
        (source_x >= 0)
        & (source_x < frame.shape[1])
        & (source_y >= 0)
        & (source_y < frame.shape[0])
    )
    canonical = np.full(map_x.shape, maximum_m, dtype=np.float32)
    canonical[inside] = frame[source_y[inside], source_x[inside]]
    return np.ascontiguousarray(
        np.clip(canonical, 0.0, maximum_m) / maximum_m,
        dtype=np.float32,
    )
