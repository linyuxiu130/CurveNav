import json

import numpy as np
import pytest
import torch

from curvenav.config import CurveNavConfig, DataConfig, TrajectoryConfig
from curvenav.data.batch import unpack_policy_batch
from curvenav.data.depth_bank import gather_depth_observations, load_packed_depth_bank
from curvenav.data.loader import (
    build_policy_training_loader,
    build_policy_validation_loader,
)
from curvenav.data.prepare import (
    _cumulative_distance,
    _flow_coordinate_statistics,
    _frame_indices,
    _fixed_future,
    _observation_to_current,
    _planar_local,
    _validate_flow_coordinate_statistics,
)
from curvenav.data.prepared import (
    PreparedPolicyDataset,
    RepeatedPolicyDataset,
    flow_coordinate_statistics,
    policy_dataset_contract,
)
from curvenav.data.privileged import (
    SourceConfigurationSpaceQuery,
)
from curvenav.deployment.runtime import DepthContextBuffer
from curvenav.factory import build_policy
from curvenav.training.batching import (
    DistributedStepBatchSampler,
    build_distributed_batch_layout,
)


def _write_dataset(root, count: int = 4) -> None:
    contract = policy_dataset_contract(DataConfig(), TrajectoryConfig())
    root.mkdir()
    (root / "manifest.json").write_text(json.dumps({"contract": contract}))
    for split in ("train", "validation"):
        split_root = root / split
        (split_root / "depth").mkdir(parents=True)
        (split_root / "source_configuration").mkdir()
        np.save(split_root / "depth/00000.npy", np.ones((8, 126, 224), np.float16))
        np.savez(
            split_root / "source_configuration/00000.npz",
            free=np.ones((9, 9), dtype=np.bool_),
            clearance_m=np.ones((9, 9), dtype=np.float32),
            origin_xy=np.asarray([-1.0, -1.0], dtype=np.float64),
            cell_size_m=np.asarray(0.25),
        )
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
            "source_grid_index": np.zeros(count, np.int64),
            "source_origin_xy": np.zeros((count, 2), np.float32),
            "source_yaw_rad": np.zeros(count, np.float32),
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
            "source_configuration_space": {
                "query": "source_dingo_signed_clearance_cell_lookup",
                "spacing_m": 0.025,
                "out_of_bounds": "non_executable_negative_clearance",
                "grids": [{"file": "source_configuration/00000.npz"}],
            },
            "audit": {
                "source_configuration_space": {
                    "serialized_requery": True,
                    "expert_all_margin_safe": True,
                }
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
        "source_grid_index",
        "source_origin_xy",
        "source_yaw_rad",
        "flow_interval_group",
    }
    assert sample["depth_indices"].dtype == torch.uint32
    assert sample["curve_values"].shape == (8,)
    bank = load_packed_depth_bank(dataset.depth_bank, torch.device("cpu"))
    depth = gather_depth_observations(bank, sample["depth_indices"].unsqueeze(0))
    assert depth.shape == (1, 4, 1, 126, 224)


def test_unpack_policy_batch_accepts_integer_global_interval_groups() -> None:
    batch_size = 2
    prepared = unpack_policy_batch(
        {
            "depth": torch.zeros((batch_size, 4, 1, 126, 224)),
            "point_goal": torch.zeros((batch_size, 2)),
            "observation_to_current": torch.zeros((batch_size, 4, 4)),
            "observation_valid": torch.ones((batch_size, 4), dtype=torch.bool),
            "curve_values": torch.zeros((batch_size, 8)),
            "flow_interval_group": torch.tensor([0, 3], dtype=torch.uint8),
        }
    )
    torch.testing.assert_close(
        prepared.flow_interval_group,
        torch.tensor([0, 3], dtype=torch.uint8),
    )


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


def test_prepared_dataset_requires_serialized_source_safety_certificate(tmp_path) -> None:
    root = tmp_path / "policy"
    _write_dataset(root)
    manifest_path = root / "train" / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["audit"]["source_configuration_space"][
        "expert_all_margin_safe"
    ] = False
    manifest_path.write_text(json.dumps(manifest))

    with pytest.raises(ValueError, match="source configuration audit"):
        PreparedPolicyDataset(root, "train", DataConfig(root=str(root)), TrajectoryConfig())


def test_source_gated_flow_statistics_reject_stale_trajectory_scale() -> None:
    config = CurveNavConfig()
    _validate_flow_coordinate_statistics(
        flow_coordinate_statistics(config.trajectory), config
    )
    codec = build_policy(config).curve_codec
    values = codec.values_from_coordinates(torch.zeros(32, 8)).numpy()
    observed = _flow_coordinate_statistics(values)

    with pytest.raises(ValueError, match="trajectory normalization"):
        _validate_flow_coordinate_statistics(observed, config)


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
            return {"index": torch.tensor(index)}

    base = IndexedDataset()
    complete = RepeatedPolicyDataset(base, count=14, seed=42)
    first_cycle = [int(complete[index]["index"]) for index in range(7)]
    second_cycle = [int(complete[index]["index"]) for index in range(7, 14)]
    assert sorted(first_cycle) == list(range(7))
    assert sorted(second_cycle) == list(range(7))
    resumed = RepeatedPolicyDataset(base, count=6, seed=42, start_index=8)
    assert [int(resumed[index]["index"]) for index in range(6)] == [
        int(complete[index]["index"]) for index in range(8, 14)
    ]
    assert [int(complete[index]["flow_interval_group"]) for index in range(8)] == [
        0,
        1,
        2,
        3,
        0,
        1,
        2,
        3,
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


def test_global_flow_interval_groups_are_exactly_quartered_for_all_topologies() -> None:
    for world_size in range(1, 9):
        batches = [
            batch
            for rank in range(world_size)
            for batch in DistributedStepBatchSampler(
                1,
                1024,
                342,
                rank,
                world_size,
            )
        ]
        groups = torch.tensor([index % 4 for batch in batches for index in batch])
        torch.testing.assert_close(torch.bincount(groups, minlength=4), torch.full((4,), 256))


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


def test_final_production_curve_is_checked_in_source_configuration_space(
    tmp_path,
) -> None:
    path = torch.tensor([[[-1.0, 0.0], [0.0, 0.0], [1.0, 0.0]]])
    grid_path = tmp_path / "navigation_grid.npz"
    clearance = np.ones((9, 9), dtype=np.float32)
    clearance[4, 4] = 0.05
    np.savez(
        grid_path,
        free=np.ones((9, 9), dtype=np.bool_),
        clearance_m=clearance,
        origin_xy=np.asarray([-1.0, -1.0]),
        cell_size_m=np.asarray(0.25),
    )
    query = SourceConfigurationSpaceQuery.from_paths((grid_path,))
    minimum = query.query(
        path,
        torch.zeros(1, dtype=torch.int64),
        torch.zeros(1, 2),
        torch.zeros(1),
        1.0,
    ).minimum_clearance_m.numpy()
    np.testing.assert_allclose(minimum, [0.05])


def test_source_query_detects_an_obstacle_between_sparse_curve_points(tmp_path) -> None:
    grid_path = tmp_path / "navigation_grid.npz"
    free = np.ones((41, 41), dtype=np.bool_)
    free[29, 20] = False
    np.savez(
        grid_path,
        free=free,
        clearance_m=np.ones((41, 41), dtype=np.float32),
        origin_xy=np.asarray([-1.0, -1.0]),
        cell_size_m=np.asarray(0.05),
    )
    query = SourceConfigurationSpaceQuery.from_paths((grid_path,))
    result = query.query(
        torch.tensor([[[-1.0, 0.0], [1.0, 0.0]]]),
        torch.zeros(1, dtype=torch.int64),
        torch.zeros(1, 2),
        torch.zeros(1),
        2.0,
    )

    assert result.minimum_clearance_m.item() < 0.0


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
