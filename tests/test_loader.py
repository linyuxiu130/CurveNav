import json

import numpy as np
import pytest
import torch

from curvenav.config import CurveNavConfig, DataConfig, TrajectoryConfig
from curvenav.data.contracts import expert_navigation_geometry_contract
from curvenav.data.depth_bank import gather_depth_observations, load_packed_depth_bank
from curvenav.data.loader import (
    build_policy_training_loader,
    build_policy_validation_loader,
)
from curvenav.data.prepare import (
    _cumulative_distance,
    _frame_indices,
    _fixed_future,
    _observation_to_current,
    _planar_local,
)
from curvenav.data.prepared import PreparedPolicyDataset, RepeatedPolicyDataset
from curvenav.deployment.runtime import DepthContextBuffer
from curvenav.training.batching import (
    DistributedStepBatchSampler,
    build_distributed_batch_layout,
)


def _write_dataset(root, count: int = 4) -> None:
    contract = {
        "expert_navigation_geometry": expert_navigation_geometry_contract(),
        "observation_frames": 4,
        "frame_spacing_m": 0.45,
        "expert_waypoint_spacing_m": 0.15,
        "future_steps": 24,
        "planar_axis_convention": "x_forward_y_left",
        "observation_to_current_semantics": "planar_rigid_transform_from_observation_to_current_frame",
        "image_height": 126,
        "image_width": 224,
        "max_depth_m": 5.0,
        "canonical_focal_x_px": 166.80851063829786,
        "canonical_focal_y_px": 166.80851063829786,
        "camera_forward_offset_m": 0.28618,
        "camera_height_m": 0.62532,
        "camera_downward_pitch_degrees": 10.0,
        "num_curve_values": 8,
        "num_path_points": 64,
        "curve_value_semantics": "metric_arc_length_then_seven_cubic_heading_control_increments_rad",
        "flow_coordinate_transform": "standardized_log_length_and_heading_increments",
        "log_length_mean": TrajectoryConfig().log_length_mean,
        "log_length_std": TrajectoryConfig().log_length_std,
        "heading_increment_mean_rad": list(
            TrajectoryConfig().heading_increment_mean_rad
        ),
        "heading_increment_std_rad": list(
            TrajectoryConfig().heading_increment_std_rad
        ),
        "expert_projection": "equal_arc_heading_field_least_squares",
        "maximum_expert_projection_ade_m": 0.03,
    }
    root.mkdir()
    (root / "manifest.json").write_text(json.dumps({"contract": contract}))
    for split in ("train", "validation"):
        split_root = root / split
        (split_root / "depth").mkdir(parents=True)
        np.save(split_root / "depth/00000.npy", np.ones((8, 126, 224), np.float16))
        arrays = {
            "depth_indices": np.tile(np.arange(4, dtype=np.uint32), (count, 1)),
            "point_goal": np.ones((count, 2), np.float32),
            "observation_to_current": np.tile(
                np.array([0.0, 0.0, 0.0, 1.0], np.float32), (count, 4, 1)
            ),
            "observation_valid": np.ones((count, 4), np.bool_),
            "curve_values": np.tile(
                np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], np.float32),
                (count, 1),
            ),
        }
        metadata = {}
        for name, value in arrays.items():
            np.save(split_root / f"{name}.npy", value)
            metadata[name] = {
                "file": f"{name}.npy",
                "shape": list(value.shape),
                "dtype": str(value.dtype),
            }
        manifest = {
            "samples": count,
            "arrays": metadata,
            "depth": {
                "dtype": "float16_normalized",
                "height": 126,
                "width": 224,
                "max_depth_m": 5.0,
                "total_frames": 8,
                "runs": [{"file": "depth/00000.npy", "offset": 0, "frames": 8}],
            },
        }
        (split_root / "manifest.json").write_text(json.dumps(manifest))


def test_prepared_dataset_has_one_fixed_tensor_contract(tmp_path) -> None:
    root = tmp_path / "policy"
    _write_dataset(root)
    data = DataConfig(root=str(root))
    trajectory = TrajectoryConfig()
    dataset = PreparedPolicyDataset(root, "train", data, trajectory)

    sample = dataset[0]
    assert set(sample) == {
        "depth_indices",
        "point_goal",
        "observation_to_current",
        "observation_valid",
        "curve_values",
    }
    assert sample["depth_indices"].dtype == torch.uint32
    assert sample["curve_values"].shape == (8,)
    bank = load_packed_depth_bank(dataset.depth_bank, torch.device("cpu"))
    depth = gather_depth_observations(bank, sample["depth_indices"].unsqueeze(0))
    assert depth.shape == (1, 4, 1, 126, 224)


def test_habitat_xz_routes_are_converted_to_x_forward_y_left() -> None:
    # Facing world +X, world -Z is physically left and world +Z is right.
    points = np.array([[1.0, -2.0], [1.0, 2.0]], dtype=np.float32)
    local = _planar_local(points, np.zeros(2, dtype=np.float32), 0.0)
    np.testing.assert_allclose(local, [[1.0, 2.0], [1.0, -2.0]])

    # A past pose whose route angle is +90 degrees is a physical right turn,
    # hence its body-yaw delta in the current left-positive frame is -90.
    transform = _observation_to_current(
        np.zeros((4, 2), dtype=np.float32),
        np.array([np.pi / 2, 0.0, 0.0, 0.0], dtype=np.float32),
        0.0,
        4,
    )
    np.testing.assert_allclose(transform[0, 2:], [-1.0, 0.0], atol=1e-6)


def test_training_and_deployment_history_transforms_are_identical() -> None:
    route_xz = np.array(
        [[0.0, 0.0], [0.45, 0.0], [0.45, 0.45], [0.9, 0.45]],
        dtype=np.float32,
    )
    route_yaw = np.array([0.0, np.pi / 2, 0.0, -np.pi / 2], dtype=np.float32)
    local_origins = _planar_local(route_xz, route_xz[-1], float(route_yaw[-1]))
    prepared_transform = _observation_to_current(
        local_origins,
        route_yaw,
        float(route_yaw[-1]),
        4,
    )

    context = DepthContextBuffer(CurveNavConfig())
    context.reset(1)
    depth = np.ones((1, 360, 640, 1), dtype=np.float32)
    selected = None
    for position_xz, yaw in zip(route_xz, route_yaw, strict=True):
        # Habitat +Z is physical right, while deployment world +Y is left.
        position_xy = np.array([[position_xz[0], -position_xz[1]]], dtype=np.float32)
        selected = context.update(
            depth,
            position_xy,
            np.array([-yaw], dtype=np.float32),
        )
    assert selected is not None
    np.testing.assert_allclose(
        selected.observation_to_current[0],
        prepared_transform,
        atol=1e-6,
    )


def test_prepared_dataset_rejects_geometry_contract_mismatch(tmp_path) -> None:
    root = tmp_path / "policy"
    _write_dataset(root)
    data = DataConfig(root=str(root))
    with pytest.raises(ValueError, match="num_path_points"):
        PreparedPolicyDataset(
            root,
            "train",
            data,
            TrajectoryConfig(num_path_points=65),
        )


def test_prepared_dataset_rejects_non_positive_arc_length(tmp_path) -> None:
    root = tmp_path / "policy"
    _write_dataset(root)
    curve_values_path = root / "train" / "curve_values.npy"
    curve_values = np.load(curve_values_path)
    curve_values[0, 0] = 0.0
    np.save(curve_values_path, curve_values)

    with pytest.raises(ValueError, match="non-positive arc length"):
        PreparedPolicyDataset(root, "train", DataConfig(root=str(root)), TrajectoryConfig())


def test_training_and_validation_preserve_deterministic_batch_order(tmp_path) -> None:
    root = tmp_path / "policy"
    _write_dataset(root)
    data = DataConfig(root=str(root))
    trajectory = TrajectoryConfig()

    validation = build_policy_validation_loader(
        data, trajectory, batch_size=2, num_workers=1
    )
    training = build_policy_training_loader(
        data,
        trajectory,
        optimizer_steps=1,
        global_batch_size=4,
        per_device_batch_size=2,
        rank=0,
        world_size=1,
        num_workers=1,
        prefetch_factor=2,
        seed=42,
    )

    assert validation.loader.in_order
    assert training.loader.in_order


def test_training_sampler_covers_each_cycle_once_and_resume_continues() -> None:
    class IndexedDataset:
        depth_bank = None

        def __len__(self):
            return 7

        def __getitem__(self, index):
            return index

    base = IndexedDataset()
    complete = RepeatedPolicyDataset(base, count=14, seed=42)
    first_cycle = [complete[index] for index in range(7)]
    second_cycle = [complete[index] for index in range(7, 14)]
    assert sorted(first_cycle) == list(range(7))
    assert sorted(second_cycle) == list(range(7))
    resumed = RepeatedPolicyDataset(base, count=6, seed=42, start_index=8)
    assert [resumed[index] for index in range(6)] == [
        complete[index] for index in range(8, 14)
    ]


def test_six_rank_batches_cover_exact_global_step_without_padding() -> None:
    layout = build_distributed_batch_layout(1024, 171, 6)
    assert layout.rank_batch_sizes == (171, 171, 171, 171, 170, 170)
    assert layout.micro_batches_per_step == 1

    rank_batches = [
        list(DistributedStepBatchSampler(1, 1024, 171, rank, 6))
        for rank in range(6)
    ]
    assert [list(map(len, batches)) for batches in rank_batches] == [
        [171],
        [171],
        [171],
        [171],
        [170],
        [170],
    ]
    covered = [index for batches in rank_batches for batch in batches for index in batch]
    assert sorted(covered) == list(range(1024))
    assert len(set(covered)) == 1024
    ddp_weight = sum(6 * len(batch) / 1024 for batches in rank_batches for batch in batches)
    assert ddp_weight / 6 == pytest.approx(1.0)


def test_micro_batches_are_balanced_for_one_static_shape() -> None:
    four_gpu = DistributedStepBatchSampler(1, 1024, 192, rank=0, world_size=4)
    assert list(map(len, four_gpu)) == [128, 128]

    two_gpu = DistributedStepBatchSampler(1, 1024, 192, rank=0, world_size=2)
    assert list(map(len, two_gpu)) == [171, 171, 170]


def test_fixed_future_uses_steps_without_rescaling_metric_length() -> None:
    far = np.array([[0.0, 0.0], [2.0, 0.0], [4.0, 0.0]], np.float32)
    near = np.array([[0.0, 0.0], [1.2, 0.2]], np.float32)
    far_prefix, far_reached = _fixed_future(far, 1)
    near_prefix, near_reached = _fixed_future(near, 24)

    np.testing.assert_allclose(far_prefix[-1], [2.0, 0.0])
    np.testing.assert_allclose(near_prefix[-1], near[-1])
    assert not far_reached
    assert near_reached


def test_fixed_future_preserves_stationary_steps_in_the_prediction_window() -> None:
    path = np.array(
        [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.15, 0.0]],
        np.float32,
    )
    prefix, reached = _fixed_future(path, 2)
    np.testing.assert_array_equal(prefix, path[:3])
    assert not reached


def test_observation_frames_use_distance_and_keep_current_when_stationary() -> None:
    positions = np.array(
        [[0.0, 0.0], [0.2, 0.0], [0.4, 0.0], [0.4, 0.0], [0.9, 0.0]],
        np.float32,
    )
    cumulative = _cumulative_distance(positions)

    data = DataConfig()
    moving = _frame_indices(4, cumulative, data)
    stationary = _frame_indices(3, cumulative, data)

    np.testing.assert_array_equal(moving, [0, 0, 3, 4])
    assert stationary[-1] == 3
