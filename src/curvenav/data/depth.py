"""Calibrated metric-depth reprojection shared by data preparation and deployment."""

from dataclasses import dataclass

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
CANONICAL_INTRINSICS = BENCHMARK_INTRINSICS.at_resolution(224, 126)


def depth_camera_contract(data: DataConfig) -> dict[str, int | float | str]:
    """Return the one metric-depth calibration used by every data source."""
    return {
        "depth_encoding": "fp16_zero_unknown_one_censored_interior_hit_v2",
        "image_height": data.image_height,
        "image_width": data.image_width,
        "max_depth_m": data.max_depth_m,
        "intrinsic_matrix": (CANONICAL_INTRINSICS.matrix().tolist() if data.embodiment == "dingo" else "per_frame_calibrated"),
        "pixel_centres": "half_integer",
        "camera_forward_offset_m": data.camera_forward_offset_m,
        "camera_height_m": data.camera_height_m,
        "camera_downward_pitch_degrees": data.camera_downward_pitch_degrees,
    }


def preprocess_depth(
    depth_m: np.ndarray,
    *,
    source_intrinsics: PinholeIntrinsics,
    maximum_m: float,
    height: int,
    width: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Resize optical-Z; preserve FoV and invalid depth.

    Pixels have half-integer centres, as in the simulator rasterizer;
    K follows that same mapping. No depth interpolation crosses object edges.
    """
    if depth_m.ndim != 2 or depth_m.shape != (
        source_intrinsics.height,
        source_intrinsics.width,
    ):
        raise ValueError("depth and calibration image dimensions must agree")
    if not np.isfinite(maximum_m) or maximum_m <= 0:
        raise ValueError("maximum depth must be finite and positive")
    y = np.minimum(
        ((np.arange(height) + 0.5) * depth_m.shape[0] / height).astype(int),
        depth_m.shape[0] - 1,
    )
    x = np.minimum(
        ((np.arange(width) + 0.5) * depth_m.shape[1] / width).astype(int),
        depth_m.shape[1] - 1,
    )
    depth = depth_m[y[:, None], x].astype(np.float32)
    valid = np.isfinite(depth) & (depth > 0)
    hit = valid & (depth < maximum_m)
    # Reserve 0 for unknown and 1 for finite measurements at/beyond the
    # planning range. FP16 rounding must not change a surface into either.
    smallest = float(np.nextafter(np.float16(0), np.float16(1)))
    largest = float(np.nextafter(np.float16(1), np.float16(0)))
    depth = np.where(
        hit, np.clip(depth / maximum_m, smallest, largest), np.where(valid, 1, 0)
    ).astype(np.float32)
    intrinsic = (
        source_intrinsics.at_resolution(width, height).matrix().astype(np.float32)
    )
    return depth, intrinsic
