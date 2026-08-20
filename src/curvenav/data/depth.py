"""One physical-depth transform shared by offline data and deployment."""

import cv2
import numpy as np


def preprocess_metric_depth(
    depth_m: np.ndarray,
    *,
    height: int,
    width: int,
    maximum_m: float,
) -> np.ndarray:
    """Convert one physical-metre depth frame to the model's normalized image."""
    frame = np.asarray(depth_m, dtype=np.float32)
    if frame.ndim != 2:
        raise ValueError("depth_m must be a two-dimensional frame")
    invalid = ~np.isfinite(frame) | (frame <= 0.0)
    frame = frame.copy()
    frame[invalid] = maximum_m
    frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_NEAREST)
    return np.ascontiguousarray(np.clip(frame, 0.0, maximum_m) / maximum_m)
