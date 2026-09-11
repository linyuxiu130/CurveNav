from types import SimpleNamespace
import warnings
import numpy as np
import pytest
import torch
from curvenav.deployment.runtime import CurveNavRuntime
from curvenav.data.observation import DepthContextBuffer
from test_depth_memory import config, condition


class RecordingPolicy:
    def __init__(self):
        self.conditions = []

    def sample(self, condition):
        self.conditions.append(condition)
        return SimpleNamespace(path=torch.zeros(len(condition.point_goal), 64, 2))


def test_sensor_history_is_independent_of_async_inference_and_resets_per_env():
    policy = RecordingPolicy()
    cfg = config()
    runtime = CurveNavRuntime(cfg, policy, "cpu")
    runtime.reset(2)
    buffer = DepthContextBuffer(cfg.data)
    buffer.reset(2)
    c = condition(2)
    args = [
        np.full((2, 126, 224, 1), 2.0, np.float32),
        np.tile(np.eye(4, dtype=np.float32), (2, 1, 1)),
        c.camera_intrinsics[:, 0].numpy(),
        c.camera_to_body[:, 0].numpy(),
        np.zeros(2, np.float64),
    ]
    snapshots = []
    # The sensor runs 21 times; the planner only consumes ticks 0, 6, 20.
    for tick in range(21):
        args[-1][:] = tick * 0.1
        snapshot = buffer.update(*args)
        if tick in (0, 6, 20):
            snapshots.append(snapshot)
            result = runtime.step(np.zeros((2, 2), np.float32), snapshot)
            assert result.path.shape == (2, 63, 3)
    np.testing.assert_allclose(
        snapshots[-1]["observation_age_s"], [[1.6, 0.9, 0.1, 0]] * 2, atol=1e-6
    )
    assert snapshots[0]["observation_valid"].sum() == 2  # immutable in-flight snapshot
    buffer.reset_env(0)
    args[-1][:] = 2.1
    snapshot = buffer.update(*args)
    assert snapshot["observation_valid"][0].sum() == 1
    assert snapshot["observation_valid"][1].sum() == 4
    # Replaying a snapshot cannot mutate the model's history.
    for value in snapshots[-1].values():
        value.setflags(write=False)  # raw HTTP buffers are immutable
    with warnings.catch_warnings(action="error", category=UserWarning):
        runtime.step(np.zeros((2, 2), np.float32), snapshots[-1])
    torch.testing.assert_close(
        policy.conditions[-1].observation_age_s, policy.conditions[-2].observation_age_s
    )
    policy.conditions[-1].observation_age_s.zero_()
    np.testing.assert_allclose(
        snapshots[-1]["observation_age_s"], [[1.6, 0.9, 0.1, 0]] * 2, atol=1e-6
    )


def test_sensor_boundary_rejects_bad_calibration_and_dropped_clock_ticks():
    buffer = DepthContextBuffer(config().data)
    buffer.reset(1)
    c = condition()
    args = [
        np.ones((1, 126, 224, 1), np.float32),
        np.eye(4, dtype=np.float32)[None],
        c.camera_intrinsics[:, 0].numpy(),
        c.camera_to_body[:, 0].numpy(),
        np.zeros(1, np.float64),
    ]
    buffer.update(*args)
    args[-1][:] = 0.2
    with pytest.raises(ValueError, match="10 Hz"):
        buffer.update(*args)
    args[-1][:] = 0.1
    args[2][:, 0, 0] = 0
    with pytest.raises(ValueError, match="focal"):
        buffer.update(*args)
