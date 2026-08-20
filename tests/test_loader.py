import random
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import torch

from curvenav.data.depth_cache import prepare_depth_cache
from curvenav.data.depth_bank import (
    PackedDepthBankSpec,
    PackedDepthRun,
    gather_depth_sequences,
    load_packed_depth_bank,
)
from curvenav.data.loader import (
    CurveNavSandDataset,
    _FixedSandValidationDataset,
    _PreparedSandCollator,
    _WeightedSandDataset,
)
from curvenav.trajectory import PlanarBSplineCodec, resample_path_by_arc_length


def test_packed_amp_depth_matches_online_transform_bitwise(tmp_path) -> None:
    dataset_root = tmp_path / "dataset_root"
    run_path = dataset_root / "dataset_avoid" / "run_0001"
    depth_dir = run_path / "depth"
    depth_dir.mkdir(parents=True)
    generator = np.random.default_rng(42)
    image = generator.integers(0, 12_000, size=(480, 640), dtype=np.uint16)
    for frame_index in range(4):
        assert cv2.imwrite(
            str(depth_dir / f"depth_{frame_index:04d}.png"),
            image,
        )
    prepare_depth_cache(
        dataset_root,
        height=168,
        width=224,
        depth_units_per_m=1000.0,
        max_depth_m=8.0,
        workers=1,
    )
    upstream = SimpleNamespace(
        dataset_root=str(dataset_root),
        run_info={
            "run": {
                "dataset_name": "dataset_avoid",
                "run_dir": "run_0001",
                "length": 4,
            }
        },
        image_size=(168, 224),
        depth_scale=1000.0,
        max_depth=8.0,
        normalize_depth=True,
        sequence_length=4,
        frame_skip=0,
    )
    dataset = CurveNavSandDataset(upstream)

    bank = load_packed_depth_bank(dataset.depth_bank, torch.device("cpu"))
    indices = dataset._depth_indices("run", 3).unsqueeze(0)
    actual = gather_depth_sequences(bank, indices)[0, -1, 0].numpy()
    reference = image.astype(np.float32) / 1000.0
    reference = np.clip(reference, 0, 8.0)
    reference = cv2.resize(reference, (224, 168), interpolation=cv2.INTER_NEAREST)
    reference = (reference / 8.0).astype(np.float16)
    assert actual.dtype == np.float16
    assert np.array_equal(actual, reference)


def test_vectorized_planar_transform_matches_scalar_route_bitwise() -> None:
    generator = np.random.default_rng(7)
    translation = generator.normal(size=(42, 3)).astype(np.float32)
    yaw = np.float32(0.37)
    pitch = np.float32(-0.19)
    reference = []
    for value in translation:
        cos_yaw = np.cos(-yaw)
        sin_yaw = np.sin(-yaw)
        cos_pitch = np.cos(-pitch)
        sin_pitch = np.sin(-pitch)
        x_after_pitch = value[0] * cos_pitch + value[2] * sin_pitch
        y_after_pitch = value[1]
        reference.append(
            [
                x_after_pitch * cos_yaw - y_after_pitch * sin_yaw,
                x_after_pitch * sin_yaw + y_after_pitch * cos_yaw,
            ]
        )
    expected = np.asarray(reference, dtype=np.float32)
    actual = CurveNavSandDataset._world_to_planar(translation, yaw, pitch)
    assert np.array_equal(actual, expected)


def test_sample_uses_sampled_endpoint_as_point_goal() -> None:
    dataset = object.__new__(CurveNavSandDataset)
    dataset.upstream = SimpleNamespace(
        run_info={
            "run": {
                "traj_xyz": np.array(
                    [[0.0, 0.0, 0.0], [1.0, 0.2, 0.0], [2.0, 3.0, 0.0]],
                    dtype=np.float32,
                ),
                "traj_yaw": np.zeros(3, dtype=np.float32),
                "traj_pitch": np.zeros(3, dtype=np.float32),
            }
        },
        sequence_length=4,
        frame_skip=0,
    )
    dataset._depth_offsets = {"run": 0}

    sample = dataset._prepare_sample("run", start_index=0, end_index=1)

    torch.testing.assert_close(sample["metric_path"][-1], torch.tensor([1.0, 0.2]))
    torch.testing.assert_close(sample["task_goal"], torch.tensor([1.0, 0.2]))
    torch.testing.assert_close(sample["metric_path"][-1], sample["task_goal"])


def test_weighted_sources_rebase_depth_indices(monkeypatch) -> None:
    class FakeSource:
        def __init__(self, path: str, frames: int, indices: list[int]) -> None:
            self.depth_bank = PackedDepthBankSpec(
                runs=(PackedDepthRun(Path(path), offset=0, frames=frames),),
                total_frames=frames,
                height=8,
                width=8,
            )
            self.indices = torch.tensor(indices, dtype=torch.int64)

        def __getitem__(self, index: int) -> dict[str, object]:
            del index
            return {"depth_indices": self.indices.clone()}

    sources = (FakeSource("a.npy", 10, [1, 2]), FakeSource("b.npy", 20, [3, 4]))
    dataset = _WeightedSandDataset(sources, (0.5, 0.5), count=4)  # type: ignore[arg-type]
    monkeypatch.setattr(random, "choices", lambda *args, **kwargs: [1])
    sample = dataset[0]

    assert torch.equal(sample["depth_indices"], torch.tensor([13, 14]))
    assert dataset.depth_bank.total_frames == 30
    assert dataset.depth_bank.runs[1].offset == 10


def test_ragged_collator_matches_per_path_resampling() -> None:
    codec = PlanarBSplineCodec(num_control_points=8, num_path_points=32)
    generator = torch.Generator().manual_seed(11)
    batch = []
    for length in (6, 9, 17, 31):
        path = torch.randn(length, 2, generator=generator).cumsum(dim=0)
        path[0] = 0
        task_goal = torch.tensor([8.0, -3.0])
        batch.append(
            {
                "depth_indices": torch.arange(2, dtype=torch.int64) + len(batch) * 2,
                "task_goal": task_goal,
                "motion_context": torch.zeros(3),
                "metric_path": path,
            }
        )

    actual = _PreparedSandCollator(codec)(batch)
    expected_path = torch.cat(
        [resample_path_by_arc_length(item["metric_path"].unsqueeze(0), 32) for item in batch]
    )
    torch.testing.assert_close(actual["canonical_path"], expected_path, rtol=0, atol=1e-7)
    torch.testing.assert_close(
        actual["control_points"],
        codec.encode(expected_path),
        rtol=0,
        atol=1e-6,
    )
    assert torch.equal(actual["depth_indices"], torch.arange(8).view(4, 2))
    assert torch.equal(actual["task_goal"], torch.tensor([[8.0, -3.0]]).repeat(4, 1))


def test_fixed_validation_descriptors_are_stable_and_cover_runs() -> None:
    class FakeBase:
        upstream = SimpleNamespace(
            run_info={
                "run_a": {"length": 20},
                "run_b": {"length": 24},
                "run_c": {"length": 28},
            },
            min_gap=5,
            max_gap=10,
        )

        @staticmethod
        def _prepare_sample(run_id, start_index, end_index):
            return {"descriptor": (run_id, start_index, end_index)}

    first = _FixedSandValidationDataset(FakeBase(), count=12, random_seed=42)
    second = _FixedSandValidationDataset(FakeBase(), count=12, random_seed=42)
    assert first.descriptors == second.descriptors
    assert {run_id for run_id, _, _ in first.descriptors} == {
        "run_a",
        "run_b",
        "run_c",
    }
    assert first[3]["descriptor"] == first.descriptors[3]
