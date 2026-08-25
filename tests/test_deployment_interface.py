from io import BytesIO

import numpy as np
import pytest

from curvenav.deployment.interface import CurveNavNpzInterface, RESPONSE_FIELDS
from curvenav.deployment.runtime import RuntimePrediction


class _FakeRuntime:
    batch_size = 0

    def __init__(self) -> None:
        self.reset_ids = []

    def reset(self, batch_size: int) -> None:
        self.batch_size = batch_size

    def reset_env(self, env_id: int) -> None:
        self.reset_ids.append(env_id)

    def step(self, point_goals, depth, positions, quaternions):
        del depth, positions, quaternions
        batch = len(point_goals)
        candidate_values = np.zeros((batch, 16), np.float32)
        return RuntimePrediction(
            path=np.zeros((batch, 64, 3), np.float32),
            candidate_paths=np.zeros((batch, 16, 64, 3), np.float32),
            candidate_costs=candidate_values,
            candidate_clearance_costs=candidate_values,
            candidate_length_costs=candidate_values,
            candidate_goal_costs=candidate_values,
            candidate_minimum_clearance_m=candidate_values,
        )


def _payload(**updates) -> bytes:
    values = {
        "point_goal": np.zeros((2, 2), np.float32),
        "depth_m": np.ones((2, 360, 640, 1), np.float32),
        "robot_position": np.zeros((2, 3), np.float32),
        "robot_quaternion": np.tile(
            np.array([0.0, 0.0, 0.0, 1.0], np.float32), (2, 1)
        ),
        "reset": np.array([True, False]),
    }
    values.update(updates)
    stream = BytesIO()
    np.savez(stream, **values)
    return stream.getvalue()


def test_npz_interface_has_one_strict_request_and_response_contract() -> None:
    runtime = _FakeRuntime()
    response = CurveNavNpzInterface(runtime).predict(_payload())

    assert runtime.batch_size == 2
    assert runtime.reset_ids == [0]
    with np.load(BytesIO(response), allow_pickle=False) as archive:
        assert frozenset(archive.files) == RESPONSE_FIELDS
        assert archive["path"].shape == (2, 64, 3)
        assert archive["candidate_paths"].shape == (2, 16, 64, 3)
        assert archive["candidate_costs"].shape == (2, 16)
        assert archive["candidate_minimum_clearance_m"].shape == (2, 16)
        assert all(archive[name].dtype == np.float32 for name in RESPONSE_FIELDS)


def test_npz_interface_requires_exact_fields_and_dtypes() -> None:
    interface = CurveNavNpzInterface(_FakeRuntime())
    with pytest.raises(ValueError, match="request fields"):
        interface.predict(_payload(unexpected=np.zeros(1)))
    with pytest.raises(TypeError, match="point_goal"):
        interface.predict(_payload(point_goal=np.zeros((2, 2), np.float64)))
