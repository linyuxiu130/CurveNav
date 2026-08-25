import json

import numpy as np
import pytest
import torch

from curvenav.config import DataConfig, TrajectoryConfig
from curvenav.data import (
    PreparedPolicyDataset,
    build_policy_overfit_loader,
    build_policy_training_loader,
    build_policy_validation_loader,
)
from curvenav.data.depth_bank import gather_depth_observations, load_packed_depth_bank
from curvenav.data.prepare import (
    _cumulative_distance,
    _frame_indices,
    _fixed_future,
)
from curvenav.data.prepared import RepeatedPolicyDataset


def _write_dataset(root, count: int = 4) -> None:
    contract = {
        "observation_frames": 4,
        "frame_spacing_m": 0.45,
        "expert_waypoint_spacing_m": 0.15,
        "future_steps": 24,
        "observation_to_current_semantics": "planar_rigid_transform_from_observation_to_current_frame",
        "image_height": 126,
        "image_width": 224,
        "max_depth_m": 5.0,
        "canonical_focal_x_px": 166.80851063829786,
        "canonical_focal_y_px": 166.80851063829786,
        "camera_forward_offset_m": 0.28618,
        "camera_height_m": 0.62532,
        "camera_downward_pitch_degrees": 10.0,
        "num_control_points": 8,
        "num_path_points": 64,
        "bspline_bending_regularization_m4": 1e-5,
    }
    root.mkdir()
    (root / "manifest.json").write_text(
        json.dumps({"contract": contract})
    )
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
            "control_points": np.zeros((count, 8, 2), np.float32),
            "reference_path": np.zeros((count, 64, 2), np.float32),
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
        "control_points",
        "reference_path",
    }
    assert sample["depth_indices"].dtype == torch.uint32
    assert sample["control_points"].shape == (8, 2)
    bank = load_packed_depth_bank(dataset.depth_bank, torch.device("cpu"))
    depth = gather_depth_observations(bank, sample["depth_indices"].unsqueeze(0))
    assert depth.shape == (1, 4, 1, 126, 224)


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


def test_overfit_loader_repeats_only_the_prepared_dataset(tmp_path) -> None:
    root = tmp_path / "policy"
    _write_dataset(root, count=2)
    data = DataConfig(root=str(root))
    bundle = build_policy_overfit_loader(data, TrajectoryConfig(), 8)
    batch = next(iter(bundle.loader))
    assert batch["point_goal"].shape == (8, 2)
    assert bundle.samples == 8


def test_validation_is_ordered_while_training_keeps_out_of_order_throughput(tmp_path) -> None:
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
        batch_size=2,
        samples_per_epoch=4,
        num_workers=1,
        prefetch_factor=2,
        seed=42,
    )

    assert validation.loader.in_order
    assert not training.loader.in_order


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
    assert [resumed[index] for index in range(6)] == [complete[index] for index in range(8, 14)]


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
