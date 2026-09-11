"""Shared causal sliding history and rigid geometry for preparation and deployment."""

from collections import deque
import numpy as np


def validate_transform(transform: np.ndarray) -> None:
    if transform.shape[-2:] != (4, 4) or not np.isfinite(transform).all():
        raise ValueError("rigid transforms must be finite [...,4,4]")
    rotation = transform[..., :3, :3]
    if (
        not np.allclose(transform[..., 3, :], [0, 0, 0, 1], atol=1e-5, rtol=0)
        or not np.allclose(
            np.swapaxes(rotation, -1, -2) @ rotation, np.eye(3), atol=1e-4, rtol=0
        )
        or not np.allclose(np.linalg.det(rotation), 1, atol=1e-4, rtol=0)
    ):
        raise ValueError("transforms must belong to SE(3), without scale or reflection")


def inverse_rigid(transform: np.ndarray) -> np.ndarray:
    result = np.zeros_like(transform)
    result[..., :3, :3] = np.swapaxes(transform[..., :3, :3], -1, -2)
    result[..., :3, 3] = -(result[..., :3, :3] @ transform[..., :3, 3, None])[..., 0]
    result[..., 3, 3] = 1
    return result


OBSERVATION_PERIOD_S = 0.1
HISTORY_WINDOW = 16
HISTORY_SLOTS = np.arange(3) * (HISTORY_WINDOW - 1) // 2


def history_contract() -> dict:
    return {
        "window": HISTORY_WINDOW,
        "slots": HISTORY_SLOTS.tolist(),
        "clock": "sensor_sequence",
        "period_s": OBSERVATION_PERIOD_S,
        "padding": "left_invalid",
    }


class ObservationHistory:
    """Three fixed samples of the previous sixteen observations, plus current.

    Slots are floor(j * 15 / 2), j=0..2, using XNavDP-style sampling. Missing slots
    reference the earliest available image but are masked, never duplicated
    evidence. Sampling happens before appending the current observation.
    """

    def __init__(self):
        self.frames = deque(maxlen=HISTORY_WINDOW)
        self.last_time = -np.inf

    def update(self, index: int, pose: np.ndarray, timestamp: float):
        if not np.isfinite(timestamp) or timestamp <= self.last_time:
            raise ValueError(
                "observation timestamps must increase strictly within an episode"
            )
        self.last_time = timestamp
        current = (index, pose.copy(), timestamp)
        past = list(self.frames)
        padding = HISTORY_WINDOW - len(past)
        padded = [past[0] if past else current] * padding + past
        selected = [padded[slot] for slot in HISTORY_SLOTS] + [current]
        indices = np.array([item[0] for item in selected], dtype=np.uint32)
        poses = np.stack([item[1] for item in selected])
        age = np.array([timestamp - item[2] for item in selected], dtype=np.float32)
        valid = np.r_[HISTORY_SLOTS >= padding, True]
        self.frames.append(current)
        return indices, inverse_rigid(pose) @ poses, age, valid
