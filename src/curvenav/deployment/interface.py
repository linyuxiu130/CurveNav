"""Strict in-memory NPZ boundary used by benchmark policy servers."""

from __future__ import annotations

from io import BytesIO

import numpy as np

from curvenav.deployment.runtime import CurveNavRuntime, load_policy


REQUEST_FIELDS = frozenset(
    {"point_goal", "depth_m", "robot_position", "robot_quaternion", "reset"}
)
RESPONSE_FIELDS = frozenset(
    {"path", "candidate_paths", "candidate_log_probabilities"}
)


def _read_request(payload: bytes) -> dict[str, np.ndarray]:
    with np.load(BytesIO(payload), allow_pickle=False) as archive:
        fields = frozenset(archive.files)
        if fields != REQUEST_FIELDS:
            raise ValueError(
                f"CurveNav request fields must be {sorted(REQUEST_FIELDS)}, got {sorted(fields)}"
            )
        request = {name: np.array(archive[name], copy=True) for name in archive.files}
    for name in REQUEST_FIELDS - {"reset"}:
        if request[name].dtype != np.float32:
            raise TypeError(f"{name} must be float32")
    if request["reset"].dtype != np.bool_:
        raise TypeError("reset must be bool")
    return request


def _write_response(
    path: np.ndarray,
    candidate_paths: np.ndarray,
    candidate_log_probabilities: np.ndarray,
) -> bytes:
    if path.ndim != 3 or path.shape[-1] != 3:
        raise ValueError("path must have shape [B,P,3]")
    batch, path_points, _ = path.shape
    if (
        candidate_paths.ndim != 4
        or candidate_paths.shape[0] != batch
        or candidate_paths.shape[2:] != (path_points, 3)
    ):
        raise ValueError("candidate_paths must have shape [B,C,P,3]")
    if candidate_log_probabilities.shape != candidate_paths.shape[:2]:
        raise ValueError("candidate_log_probabilities must have shape [B,C]")
    stream = BytesIO()
    np.savez(
        stream,
        path=np.asarray(path, dtype=np.float32),
        candidate_paths=np.asarray(candidate_paths, dtype=np.float32),
        candidate_log_probabilities=np.asarray(
            candidate_log_probabilities, dtype=np.float32
        ),
    )
    return stream.getvalue()


class CurveNavNpzInterface:
    """Translate one exact benchmark request into one stateful policy step."""

    def __init__(self, runtime: CurveNavRuntime) -> None:
        self.runtime = runtime

    def predict(self, payload: bytes) -> bytes:
        request = _read_request(payload)
        batch_size = int(request["point_goal"].shape[0])
        reset = request["reset"]
        if reset.shape != (batch_size,):
            raise ValueError(f"reset must have shape [{batch_size}]")
        if self.runtime.batch_size != batch_size:
            self.runtime.reset(batch_size)
        for env_id in np.flatnonzero(reset):
            self.runtime.reset_env(int(env_id))
        path, candidate_paths, candidate_log_probabilities = self.runtime.step(
            request["point_goal"],
            request["depth_m"],
            request["robot_position"],
            request["robot_quaternion"],
        )
        return _write_response(path, candidate_paths, candidate_log_probabilities)


def load_npz_interface(
    checkpoint_path: str,
    config_path: str,
    device: str = "cuda",
) -> CurveNavNpzInterface:
    """Load the deployable EMA policy behind the only benchmark boundary."""
    config, policy = load_policy(checkpoint_path, config_path, device)
    return CurveNavNpzInterface(
        CurveNavRuntime(config, policy, device=device)
    )
