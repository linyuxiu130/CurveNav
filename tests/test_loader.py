import json

import numpy as np
import pytest
import torch

from curvenav.config import CurveNavConfig, DataConfig, TrajectoryConfig
from curvenav.data.batch import unpack_policy_batch
from curvenav.data.depth import BENCHMARK_INTRINSICS
from curvenav.data.depth_bank import gather_depth_observations, load_packed_depth_bank
from test_depth_memory import condition as depth_condition
from dataclasses import fields
from curvenav.data.loader import (
    build_policy_training_loader,
    build_policy_validation_loader,
    source_balanced_indices,
)
from curvenav.data.prepare import (
    _flow_coordinate_statistics,
    _fixed_future,
    _planar_local,
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
from curvenav.factory import build_policy
from curvenav.training.batching import (
    DistributedStepBatchSampler,
    build_distributed_batch_layout,
)


def test_source_balanced_sampling_is_fixed_unique_and_covers_small_scenes():
    sources = np.repeat([0, 2, 7], [100, 40, 3])
    indices = source_balanced_indices(sources, 8)
    assert indices == source_balanced_indices(sources, 8)
    assert indices == sorted(set(indices))
    assert np.unique(sources[indices], return_counts=True)[1].tolist() == [8, 8, 3]
    assert source_balanced_indices(sources, 200) == list(range(len(sources)))


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
            "obstacle_memory": depth_condition(count).obstacle_memory.numpy(),
            "depth_indices": np.tile(np.arange(4, dtype=np.uint32), (count, 1)),
            "point_goal": np.ones((count, 2), np.float32),
            "observation_to_current": np.broadcast_to(
                np.eye(4, dtype=np.float32), (count, 4, 4, 4)
            ),
            "observation_age_s": np.tile(
                np.array([1.6, 0.9, 0.1, 0], np.float32), (count, 1)
            ),
            "camera_intrinsics": depth_condition(count).camera_intrinsics.numpy(),
            "camera_to_body": depth_condition(count).camera_to_body.numpy(),
            "observation_valid": np.ones((count, 4), np.bool_),
            "curve_values": np.tile(
                np.linspace(0.1, 1.4, 14, dtype=np.float32),
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
                "dtype": "depth_float16_zero_invalid",
                "height": 126,
                "width": 224,
                "max_depth_m": 5.0,
                "total_frames": 8,
                "runs": [
                    {
                        "file": "depth/00000.npy",
                        "offset": 0,
                        "frames": 8,
                    }
                ],
            },
            "source_configuration_space": {
                "query": "source_dingo_signed_clearance_cell_lookup",
                "spacing_m": 0.025,
                "path_sampling": "closed_cell_supercover_max_spacing_v2",
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
        "obstacle_memory",
        "depth_indices",
        "camera_intrinsics",
        "camera_to_body",
        "observation_age_s",
        "point_goal",
        "observation_to_current",
        "observation_valid",
        "curve_values",
        "source_grid_index",
        "source_origin_xy",
        "source_yaw_rad",
    }
    assert sample["depth_indices"].dtype == torch.uint32
    assert sample["curve_values"].shape == (14,)
    bank = load_packed_depth_bank(dataset.depth_bank)
    depth = gather_depth_observations(bank, sample["depth_indices"].unsqueeze(0))
    assert depth.shape == (1, 4, 1, 126, 224)


def test_unpack_policy_batch_preserves_training_inputs() -> None:
    batch_size = 2
    c = depth_condition(batch_size)
    prepared = unpack_policy_batch(
        {
            **{f.name: getattr(c, f.name) for f in fields(c)},
            "curve_values": torch.zeros(batch_size, 14),
        }
    )
    torch.testing.assert_close(prepared.target.curve_values, torch.zeros(batch_size, 14))


def test_habitat_xz_routes_are_converted_to_x_forward_y_left() -> None:
    # Facing world +X, world -Z is physically left and world +Z is right.
    points = np.array([[1.0, -2.0], [1.0, 2.0]], dtype=np.float32)
    local = _planar_local(points, np.zeros(2, dtype=np.float32), 0.0)
    np.testing.assert_allclose(local, [[1.0, 2.0], [1.0, -2.0]])


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


def test_prepared_dataset_rejects_non_finite_control(tmp_path) -> None:
    root = tmp_path / "policy"
    _write_dataset(root)
    curve_values_path = root / "train" / "curve_values.npy"
    curve_values = np.load(curve_values_path)
    curve_values[0, 0] = np.nan
    np.save(curve_values_path, curve_values)

    with pytest.raises(ValueError, match="non-finite values"):
        PreparedPolicyDataset(
            root, "train", DataConfig(root=str(root)), TrajectoryConfig()
        )


def test_prepared_dataset_requires_serialized_source_safety_certificate(
    tmp_path,
) -> None:
    root = tmp_path / "policy"
    _write_dataset(root)
    manifest_path = root / "train" / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["audit"]["source_configuration_space"]["expert_all_margin_safe"] = False
    manifest_path.write_text(json.dumps(manifest))

    with pytest.raises(ValueError, match="source configuration audit"):
        PreparedPolicyDataset(
            root, "train", DataConfig(root=str(root)), TrajectoryConfig()
        )


def test_flow_statistics_are_measured_from_physical_control_increments() -> None:
    torch.manual_seed(44)
    codec = build_policy(CurveNavConfig()).curve_codec
    values = codec.values_from_coordinates(torch.randn(32, 14)).numpy()
    observed = _flow_coordinate_statistics(values)
    controls = values.astype(np.float64).reshape(32, 7, 2)
    increments = np.diff(
        np.concatenate((np.zeros((32, 1, 2)), controls), axis=1), axis=1
    ).reshape(32, 14)
    np.testing.assert_allclose(
        observed["control_increment_mean_xy_m"], increments.mean(0)
    )
    np.testing.assert_allclose(
        observed["control_increment_std_xy_m"],
        np.repeat(np.sqrt(increments.reshape(-1, 7, 2).var(0, ddof=1).mean(-1)), 2)
    )


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

        def __getitems__(self, indices):
            return {"index": torch.tensor(indices)}

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
    # A worker batch can cross a shuffle-cycle boundary, including on resume.
    batched = resumed.__getitems__(list(range(6)))
    for name, values in batched.items():
        assert torch.equal(values, torch.stack([resumed[i][name] for i in range(6)]))



def test_fixed_micro_batches_cover_global_stream() -> None:
    for world_size in range(1, 9):
        micro_batch = 319
        accumulation = 2
        global_batch = micro_batch * world_size * accumulation
        layout = build_distributed_batch_layout(global_batch, micro_batch, world_size)
        assert layout.rank_batch_sizes == (micro_batch * accumulation,) * world_size
        assert layout.micro_batches_per_step == accumulation
        batches = [
            batch
            for rank in range(world_size)
            for batch in DistributedStepBatchSampler(2, global_batch, micro_batch, rank, world_size)
        ]
        assert all(len(batch) == micro_batch for batch in batches)
        covered = [index for batch in batches for index in batch]
        assert sorted(covered) == list(range(2 * global_batch))
    with pytest.raises(ValueError, match="complete fixed-size"):
        build_distributed_batch_layout(1000, 320, 2)


def test_fixed_future_metric_horizon_is_independent_of_observation_density() -> None:
    far = np.array([[0.0, 0.0], [2.0, 0.0], [4.0, 0.0]], np.float32)
    near = np.array([[0.0, 0.0], [1.2, 0.2]], np.float32)
    far_prefix, far_reached = _fixed_future(far, 1, 0.15)
    near_prefix, near_reached = _fixed_future(near, 24, 0.15)

    np.testing.assert_allclose(far_prefix[-1], [0.15, 0.0])
    np.testing.assert_allclose(near_prefix[-1], near[-1])
    assert not far_reached
    assert near_reached


def test_final_production_curve_is_checked_in_source_configuration_space(
    tmp_path,
) -> None:
    path = torch.tensor([[[-0.9, 0.0], [0.0, 0.0], [1.0, 0.0]]])
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


def test_fixed_future_does_not_spend_metric_horizon_on_stationary_observations() -> (
    None
):
    path = np.array(
        [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.15, 0.0]],
        np.float32,
    )
    prefix, reached = _fixed_future(path, 2, 0.15)
    np.testing.assert_allclose(prefix[-1], path[-1])
    assert reached
