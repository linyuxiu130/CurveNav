"""Calibrated metric-depth reprojection shared by data preparation and deployment."""

from dataclasses import dataclass
from functools import lru_cache

import cv2
import numpy as np


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
SAND_INTRINSICS = PinholeIntrinsics(
    width=640,
    height=480,
    fx=389.551,
    fy=389.551,
    cx=324.211,
    cy=235.656,
)
CANONICAL_INTRINSICS = PinholeIntrinsics(
    width=224,
    height=126,
    fx=166.80851063829786,
    fy=166.80851063829786,
    cx=112.0,
    cy=63.0,
)


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
