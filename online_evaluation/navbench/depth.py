"""One meter-valued depth contract shared by transport and model adapters."""

from __future__ import annotations

import numpy as np


def as_depth_meters(values: np.ndarray) -> np.ndarray:
    """Return contiguous float32 meters without quantizing or replacing sentinels."""
    return np.ascontiguousarray(np.asarray(values, dtype=np.float32))


def replace_invalid_depth(values: np.ndarray, replacement_m: float) -> np.ndarray:
    """Apply a model's explicit invalid-depth convention after wire decoding."""
    depth = as_depth_meters(values).copy()
    depth[~np.isfinite(depth)] = np.float32(replacement_m)
    return depth
