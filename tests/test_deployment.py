import numpy as np

from curvenav.config import CurveNavConfig
from curvenav.data_generation.hssd_pilot import HssdPilotConfig
from curvenav.data.depth import preprocess_metric_depth
from curvenav.deployment.runtime import (
    BENCHMARK_CAMERA_HEIGHT_M,
    BENCHMARK_FOCAL_PX,
    DepthSafetySelector,
    SpatialDepthHistory,
)


def test_hssd_camera_matches_the_final_benchmark():
    camera = HssdPilotConfig()

    assert camera.depth_focal_px == BENCHMARK_FOCAL_PX
    assert camera.camera_height_m == BENCHMARK_CAMERA_HEIGHT_M
    assert camera.camera_forward_m == 0.0
    assert camera.camera_pitch_degrees == 0.0


def test_invalid_depth_is_encoded_as_sensor_limit():
    depth = np.array([[0.0, np.nan], [np.inf, 4.0]], dtype=np.float32)

    normalized = preprocess_metric_depth(depth, height=2, width=2, maximum_m=8.0)

    np.testing.assert_allclose(normalized, [[1.0, 1.0], [1.0, 0.5]])


def test_depth_history_uses_expert_spatial_offsets():
    history = SpatialDepthHistory(CurveNavConfig())
    history.reset(1)
    selected = None
    for index in range(15):
        depth = np.full((1, 2, 2, 1), 1.0 + index * 0.3, dtype=np.float32)
        position = np.array([[index * 0.1, 0.0]], dtype=np.float32)
        selected = history.update(depth, position)

    assert selected is not None
    selected_indices = (
        selected[0, :, 0].mean(axis=(1, 2)) * 8.0 - 1.0
    ) / 0.3
    selected_distances = selected_indices * 0.1
    np.testing.assert_allclose(
        selected_distances,
        [0.05, 0.50, 0.95, 1.40],
        atol=0.051,
    )


def test_stationary_history_uses_the_current_observation():
    history = SpatialDepthHistory(CurveNavConfig())
    history.reset(1)
    history.update(
        np.full((1, 2, 2, 1), 1.0, dtype=np.float32),
        np.zeros((1, 2), dtype=np.float32),
    )
    selected = history.update(
        np.full((1, 2, 2, 1), 4.0, dtype=np.float32),
        np.zeros((1, 2), dtype=np.float32),
    )

    np.testing.assert_allclose(selected, 0.5)


def test_depth_selector_prefers_observed_safe_detour():
    selector = DepthSafetySelector(maximum_depth_m=8.0)
    depth = np.full((1, 360, 640, 1), 8.0, dtype=np.float32)
    depth[0, 80:280, 300:340, 0] = 1.0
    parameter = np.linspace(0.0, 1.0, 64, dtype=np.float32)
    straight = np.column_stack((2.0 * parameter, np.zeros_like(parameter)))
    detour = np.column_stack(
        (2.0 * parameter, 0.8 * np.sin(np.pi * parameter))
    )
    candidates = np.stack((straight, detour))[None]

    selected, values = selector.select(
        candidates,
        depth,
        np.array([[3.0, 0.0]], dtype=np.float32),
    )

    np.testing.assert_allclose(selected[0], detour)
    assert values[0, 1] > values[0, 0]


def test_depth_selector_does_not_prefer_a_short_free_endpoint():
    selector = DepthSafetySelector(maximum_depth_m=8.0)
    depth = np.full((1, 360, 640, 1), 8.0, dtype=np.float32)
    parameter = np.linspace(0.0, 1.0, 64, dtype=np.float32)
    short = np.column_stack((0.5 * parameter, np.zeros_like(parameter)))
    progress = np.column_stack((1.5 * parameter, np.zeros_like(parameter)))
    candidates = np.stack((short, progress))[None]

    selected, values = selector.select(
        candidates,
        depth,
        np.array([[4.0, 0.0]], dtype=np.float32),
    )

    np.testing.assert_allclose(selected[0], progress)
    assert values[0, 1] > values[0, 0]
