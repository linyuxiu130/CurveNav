"""Fast, simulator-free checks for the CurveNav v2 supervision contract."""

from __future__ import annotations

import numpy as np

from curvenav.data_generation.hssd_pilot import HssdPilotConfig
from curvenav.data_generation.hssd_v2 import (
    DEPTH_CONTRACT_VERSION,
    SCHEMA_VERSION,
    _camera_contract,
    make_sample_records,
)


def main() -> None:
    world_xy = np.column_stack(
        [np.arange(60, dtype=np.float32) * 0.15, np.zeros(60, dtype=np.float32)]
    )
    segment = np.linalg.norm(np.diff(world_xy, axis=0), axis=1)
    cumulative = np.concatenate([[0.0], np.cumsum(segment)]).astype(np.float32)
    route = {
        "world_xy": world_xy,
        "yaw_rad": np.zeros(60, dtype=np.float32),
        "cumulative_arc_length_m": cumulative,
    }
    episode = {
        "episode_id": "hssd_v2/train/unit/run_0000",
        "split": "train",
        "scene_id": "unit",
    }
    first = make_sample_records(episode, route, 17)
    second = make_sample_records(episode, route, 17)
    assert first == second
    assert len(first) == 59
    assert first[0]["task_goal_local_xy"] == [float(world_xy[-1, 0]), 0.0]
    assert first[0]["target_end_index"] < len(world_xy) - 1
    assert first[0]["target_endpoint_local_xy"] != first[0]["task_goal_local_xy"]
    assert len({round(sample["target_arc_length_m"], 3) for sample in first}) > 10
    assert any(sample["near_goal"] for sample in first)
    sample = first[12]
    assert sample["history_indices"] == [3, 6, 9, 12]
    assert np.allclose(np.asarray(sample["history_motion_se2"])[1:, 0], 0.45)
    assert np.allclose(np.asarray(sample["history_motion_se2"])[1:, 4], 1.0)

    camera = _camera_contract(HssdPilotConfig())
    assert camera["contract_version"] == DEPTH_CONTRACT_VERSION
    assert camera["image"]["width"] == 640
    assert camera["image"]["height"] == 360
    assert abs(camera["image"]["horizontal_fov_degrees"] - 67.7570796) < 1e-6
    assert camera["depth"]["dtype"] == "float32"
    assert camera["depth"]["unit"] == "m"
    assert camera["depth"]["invalid"] == "not isfinite(value) or value <= 0"
    assert "transport_note" not in camera
    assert "integer_quantization" not in camera["depth"]
    assert "clipped" not in camera["depth"]
    assert "invalid_values_replaced" not in camera["depth"]
    assert SCHEMA_VERSION == "curvenav_hssd_v2.0"
    print("CurveNav v2 contract checks passed")


if __name__ == "__main__":
    main()
