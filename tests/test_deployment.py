import numpy as np
import pytest
import torch

from curvenav.config import CurveNavConfig
from curvenav.data.depth import BENCHMARK_INTRINSICS, preprocess_metric_depth
from curvenav.deployment.runtime import (
    CurveNavRuntime,
    DepthContextBuffer,
)


class _RecordingPolicy:
    def __init__(self) -> None:
        self.conditions = []

    def __call__(self, condition):
        self.conditions.append(condition)


def test_invalid_depth_is_encoded_as_sensor_limit():
    depth = np.full((360, 640), 4.0, dtype=np.float32)
    depth[round(BENCHMARK_INTRINSICS.cy), round(BENCHMARK_INTRINSICS.cx)] = np.nan

    normalized = preprocess_metric_depth(
        depth, source_intrinsics=BENCHMARK_INTRINSICS, maximum_m=5.0
    )

    assert normalized.shape == (126, 224)
    assert normalized[63, 112] == 1.0
    np.testing.assert_allclose(normalized[0, 0], 0.8)


def test_benchmark_depth_resolution_rescales_the_same_camera_rays():
    half = BENCHMARK_INTRINSICS.at_resolution(width=320, height=180)
    assert half.fx == BENCHMARK_INTRINSICS.fx / 2
    assert half.fy == BENCHMARK_INTRINSICS.fy / 2
    assert half.cx == 160.0
    assert half.cy == 90.0

    full_depth = np.full((360, 640), 2.0, dtype=np.float32)
    half_depth = np.full((180, 320), 2.0, dtype=np.float32)
    full = preprocess_metric_depth(
        full_depth, source_intrinsics=BENCHMARK_INTRINSICS, maximum_m=5.0
    )
    downsampled = preprocess_metric_depth(
        half_depth, source_intrinsics=BENCHMARK_INTRINSICS, maximum_m=5.0
    )
    np.testing.assert_array_equal(downsampled, full)


def test_depth_preprocessing_rejects_non_image_input():
    with pytest.raises(ValueError, match="two-dimensional"):
        preprocess_metric_depth(
            np.ones((1, 180, 320), dtype=np.float32),
            source_intrinsics=BENCHMARK_INTRINSICS,
            maximum_m=5.0,
        )


def test_runtime_reset_warms_the_actual_batch_execution() -> None:
    policy = _RecordingPolicy()
    runtime = CurveNavRuntime(CurveNavConfig(), policy, device="cpu")

    runtime.reset(3)

    assert runtime.batch_size == 3
    assert len(policy.conditions) == 1
    condition = policy.conditions[0]
    assert condition.depth.shape == (3, 4, 1, 126, 224)
    assert condition.point_goal.shape == (3, 2)
    assert torch.all(condition.observation_valid)
    assert torch.all(condition.observation_to_current[..., 3] == 1.0)


def test_depth_context_uses_expert_spatial_offsets():
    context_buffer = DepthContextBuffer(CurveNavConfig())
    context_buffer.reset(1)
    selected = None
    for index in range(15):
        depth = np.full((1, 360, 640, 1), 1.0 + index * 0.2, dtype=np.float32)
        position = np.array([[index * 0.1, 0.0]], dtype=np.float32)
        selected = context_buffer.update(depth, position, np.zeros(1, dtype=np.float32))

    assert selected is not None
    selected_indices = (
        selected.depth[0, :, 0].mean(axis=(1, 2)) * 5.0 - 1.0
    ) / 0.2
    selected_distances = selected_indices * 0.1
    np.testing.assert_allclose(
        selected_distances,
        [0.05, 0.50, 0.95, 1.40],
        atol=0.051,
    )


def test_stationary_context_uses_the_current_observation():
    context_buffer = DepthContextBuffer(CurveNavConfig())
    context_buffer.reset(1)
    context_buffer.update(
        np.full((1, 360, 640, 1), 1.0, dtype=np.float32),
        np.zeros((1, 2), dtype=np.float32),
        np.zeros(1, dtype=np.float32),
    )
    selected = context_buffer.update(
        np.full((1, 360, 640, 1), 4.0, dtype=np.float32),
        np.zeros((1, 2), dtype=np.float32),
        np.zeros(1, dtype=np.float32),
    )

    np.testing.assert_allclose(selected.depth, 0.8)


def test_depth_context_reports_observation_transforms_in_current_coordinates():
    context_buffer = DepthContextBuffer(CurveNavConfig())
    context_buffer.reset(1)
    depth = np.ones((1, 360, 640, 1), dtype=np.float32)
    context_buffer.update(
        depth,
        np.array([[0.0, 0.0]], dtype=np.float32),
        np.array([0.0], dtype=np.float32),
    )
    selected = context_buffer.update(
        depth,
        np.array([[1.0, 0.0]], dtype=np.float32),
        np.array([np.pi / 2], dtype=np.float32),
    )

    np.testing.assert_allclose(
        selected.observation_to_current[-1, -1], [0.0, 0.0, 0.0, 1.0]
    )
    np.testing.assert_allclose(
        selected.observation_to_current[0, 0], [0.0, 1.0, -1.0, 0.0], atol=1e-6
    )
