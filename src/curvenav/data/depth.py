"""Calibrated metric-depth reprojection shared by data preparation and deployment."""

from dataclasses import dataclass
from functools import lru_cache

import cv2
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


BENCHMARK_INTRINSICS = PinholeIntrinsics(
    width=640,
    height=360,
    fx=326.398559570312,
    fy=326.398559570312,
    cx=321.792145,
    cy=181.007690,
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
    """Reproject physical depth into CurveNav's fixed pinhole camera and normalize."""
    frame = np.asarray(depth_m, dtype=np.float32)
    if frame.shape != (source_intrinsics.height, source_intrinsics.width):
        raise ValueError(
            "depth shape does not match source intrinsics: "
            f"{frame.shape} != {(source_intrinsics.height, source_intrinsics.width)}"
        )
    invalid = ~np.isfinite(frame) | (frame <= 0.0)
    frame = frame.copy()
    frame[invalid] = maximum_m
    map_x, map_y = _pinhole_remap(source_intrinsics, CANONICAL_INTRINSICS)
    canonical = cv2.remap(
        frame,
        map_x,
        map_y,
        interpolation=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=maximum_m,
    )
    return np.ascontiguousarray(
        np.clip(canonical, 0.0, maximum_m) / maximum_m,
        dtype=np.float32,
    )
