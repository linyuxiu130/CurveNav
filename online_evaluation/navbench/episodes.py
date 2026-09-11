"""Validation helpers for the official X-NavDP PointGoal episode files."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


EPISODE_COLUMNS = (
    "start_x_m",
    "start_y_m",
    "goal_x_m",
    "goal_y_m",
    "start_yaw_rad",
)


def sha256_file(path: Path) -> str:
    """Return the SHA-256 digest of one file."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_json(value: Any) -> str:
    """Hash a JSON value using a stable compact encoding."""
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def validate_official_episode_file(
    path: Path,
    *,
    expected_count: int,
    expected_dtype: str,
    expected_sha256: str,
    actual_sha256: str | None = None,
) -> list[str]:
    """Validate one immutable episode file released by X-NavDP."""
    errors: list[str] = []
    if (actual_sha256 if actual_sha256 is not None else sha256_file(path)) != expected_sha256:
        errors.append("SHA-256 differs from the pinned X-NavDP release")
        return errors
    try:
        episodes = np.load(path, mmap_mode="r", allow_pickle=False)
    except (OSError, ValueError) as exc:
        return [f"cannot load episode array: {exc}"]
    if episodes.shape != (expected_count, len(EPISODE_COLUMNS)):
        errors.append(
            f"expected shape {(expected_count, len(EPISODE_COLUMNS))}, "
            f"found {episodes.shape}"
        )
        return errors
    if episodes.dtype.str != expected_dtype:
        errors.append(f"expected dtype {expected_dtype}, found {episodes.dtype.str}")
    values = np.asarray(episodes)
    if not np.isfinite(values).all():
        errors.append("episode array contains non-finite values")
    distances = np.linalg.norm(values[:, 2:4] - values[:, :2], axis=1)
    if np.any(distances <= 0.0):
        errors.append("episode array contains a zero-length task")
    return errors
