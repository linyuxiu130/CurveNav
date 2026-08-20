"""High-throughput SanD loading for the single CurveNav training route."""

import json
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

from curvenav.config import DataConfig, DataSourceConfig, TrajectoryConfig
from curvenav.data.depth_bank import PackedDepthBankSpec, PackedDepthRun
from curvenav.data.depth_cache import CACHE_SCHEMA_VERSION, depth_cache_root
from curvenav.data.sand_index import SandRunIndex
from curvenav.data.trajectory import collate_metric_paths
from curvenav.trajectory import PlanarBSplineCodec


class CurveNavSandDataset(Dataset):
    """Sample official SanD runs without its redundant intermediate spline fit."""

    def __init__(self, upstream: Dataset) -> None:
        self.upstream = upstream
        height, width = self.upstream.image_size  # type: ignore[attr-defined]
        cache_root = depth_cache_root(  # type: ignore[attr-defined]
            self.upstream.dataset_root,
            height,
            width,
        )
        manifest_path = cache_root / "manifest.json"
        if not manifest_path.is_file():
            raise RuntimeError(
                f"required packed depth cache is missing: {manifest_path}; "
                "run scripts/prepare_sand_depth_cache.py first"
            )
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        expected_header = {
            "schema_version": CACHE_SCHEMA_VERSION,
            "height": height,
            "width": width,
            "dtype": "float16",
            "depth_units_per_m": float(self.upstream.depth_scale),  # type: ignore[attr-defined]
            "max_depth_m": float(self.upstream.max_depth),  # type: ignore[attr-defined]
        }
        mismatches = {
            key: (manifest.get(key), value)
            for key, value in expected_header.items()
            if manifest.get(key) != value
        }
        if mismatches:
            raise RuntimeError(f"packed depth cache contract mismatch: {mismatches}")
        manifest_runs = manifest.get("runs")
        if not isinstance(manifest_runs, dict):
            raise RuntimeError("packed depth cache manifest has no run index")
        depth_runs = []
        self._depth_offsets: dict[str, int] = {}
        offset = 0
        for run_id, run in self.upstream.run_info.items():  # type: ignore[attr-defined]
            run_key = f'{run["dataset_name"]}/{run["run_dir"]}'
            if manifest_runs.get(run_key) != int(run["length"]):
                raise RuntimeError(f"packed depth cache run mismatch: {run_key}")
            path = (
                cache_root / run["dataset_name"] / f'{run["run_dir"]}.npy'
            )
            if not path.is_file():
                raise RuntimeError(f"packed depth cache file is missing: {path}")
            frames = int(run["length"])
            self._depth_offsets[run_id] = offset
            depth_runs.append(PackedDepthRun(path=path, offset=offset, frames=frames))
            offset += frames
        self.depth_bank = PackedDepthBankSpec(
            runs=tuple(depth_runs),
            total_frames=offset,
            height=height,
            width=width,
        )

    def __len__(self) -> int:
        return len(self.upstream)

    def _depth_indices(self, run_id: str, start_index: int) -> Tensor:
        """Return the four immutable frame-bank indices for one observation."""
        sequence_length = self.upstream.sequence_length  # type: ignore[attr-defined]
        frame_step = self.upstream.frame_skip + 1  # type: ignore[attr-defined]
        run_offset = self._depth_offsets[run_id]
        return torch.tensor(
            [
                run_offset
                + max(0, start_index - (sequence_length - 1 - offset) * frame_step)
                for offset in range(sequence_length)
            ],
            dtype=torch.int64,
        )

    @staticmethod
    def _world_to_planar(
        world_translation: np.ndarray,
        yaw: np.floating,
        pitch: np.floating,
    ) -> np.ndarray:
        cos_yaw = np.cos(-yaw)
        sin_yaw = np.sin(-yaw)
        cos_pitch = np.cos(-pitch)
        sin_pitch = np.sin(-pitch)
        x_after_pitch = (
            world_translation[:, 0] * cos_pitch
            + world_translation[:, 2] * sin_pitch
        )
        y_after_pitch = world_translation[:, 1]
        relative_x = x_after_pitch * cos_yaw - y_after_pitch * sin_yaw
        relative_y = x_after_pitch * sin_yaw + y_after_pitch * cos_yaw
        return np.stack((relative_x, relative_y), axis=-1).astype(
            np.float32,
            copy=False,
        )

    def _prepare_sample(
        self,
        run_id: str,
        start_index: int,
        end_index: int,
    ) -> dict[str, object]:
        run = self.upstream.run_info[run_id]  # type: ignore[attr-defined]
        trajectory_xyz = run["traj_xyz"]
        start_xyz = trajectory_xyz[start_index]
        start_yaw = run["traj_yaw"][start_index]
        start_pitch = run["traj_pitch"][start_index]
        metric_path = self._world_to_planar(
            trajectory_xyz[start_index : end_index + 1] - start_xyz,
            start_yaw,
            start_pitch,
        )
        task_goal = self._world_to_planar(
            (trajectory_xyz[end_index] - start_xyz)[None],
            start_yaw,
            start_pitch,
        )[0]

        context = np.zeros(3, dtype=np.float32)
        if start_index > 0:
            previous_in_current = self._world_to_planar(
                (trajectory_xyz[start_index - 1] - start_xyz)[None],
                start_yaw,
                start_pitch,
            )[0]
            executed_displacement = -previous_in_current
            magnitude = float(np.linalg.norm(executed_displacement))
            if magnitude > 1e-6:
                context[:2] = executed_displacement / magnitude
                context[2] = 1.0
        return {
            "depth_indices": self._depth_indices(run_id, start_index),
            "task_goal": torch.from_numpy(task_goal),
            "motion_context": torch.from_numpy(context),
            "metric_path": torch.from_numpy(metric_path),
        }

    def __getitem__(self, index: int) -> dict[str, object]:
        del index
        scene_name = self.upstream._sample_scene()  # type: ignore[attr-defined]
        run_id = random.choice(self.upstream.scene_runs[scene_name])  # type: ignore[attr-defined]
        run = self.upstream.run_info[run_id]  # type: ignore[attr-defined]
        run_length = int(run["length"])
        start_index = random.randint(0, run_length - self.upstream.min_gap - 1)  # type: ignore[attr-defined]
        end_index = random.randint(
            start_index + self.upstream.min_gap,  # type: ignore[attr-defined]
            min(start_index + self.upstream.max_gap, run_length - 1),  # type: ignore[attr-defined]
        )
        return self._prepare_sample(run_id, start_index, end_index)


def _combine_depth_banks(
    datasets: tuple[CurveNavSandDataset, ...],
) -> tuple[PackedDepthBankSpec, tuple[int, ...]]:
    if not datasets:
        raise ValueError("at least one dataset is required")
    first = datasets[0].depth_bank
    runs = []
    offsets = []
    total_frames = 0
    for dataset in datasets:
        spec = dataset.depth_bank
        if (spec.height, spec.width) != (first.height, first.width):
            raise ValueError("all data sources must use the same depth resolution")
        offsets.append(total_frames)
        runs.extend(
            PackedDepthRun(
                path=run.path,
                offset=run.offset + total_frames,
                frames=run.frames,
            )
            for run in spec.runs
        )
        total_frames += spec.total_frames
    return (
        PackedDepthBankSpec(
            runs=tuple(runs),
            total_frames=total_frames,
            height=first.height,
            width=first.width,
        ),
        tuple(offsets),
    )


class _WeightedSandDataset(Dataset):
    """Sample one or more SanD-compatible sources through one data path."""

    def __init__(
        self,
        datasets: tuple[CurveNavSandDataset, ...],
        weights: tuple[float, ...],
        count: int,
    ) -> None:
        if len(datasets) != len(weights):
            raise ValueError("datasets and weights must have the same length")
        if count < 1:
            raise ValueError("sample count must be positive")
        self.datasets = datasets
        self.weights = weights
        self.count = count
        self.depth_bank, self._depth_offsets = _combine_depth_banks(datasets)

    def __len__(self) -> int:
        return self.count

    def __getitem__(self, index: int) -> dict[str, object]:
        del index
        source_index = random.choices(
            range(len(self.datasets)), weights=self.weights, k=1
        )[0]
        sample = self.datasets[source_index][0]
        sample["depth_indices"] = (
            sample["depth_indices"] + self._depth_offsets[source_index]  # type: ignore[operator]
        )
        return sample


class _OffsetDataset(Dataset):
    def __init__(self, base: Dataset, depth_offset: int) -> None:
        self.base = base
        self.depth_offset = depth_offset

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, index: int) -> dict[str, object]:
        sample = self.base[index]  # type: ignore[assignment]
        sample["depth_indices"] = sample["depth_indices"] + self.depth_offset
        return sample


class _FixedSandValidationDataset(Dataset):
    """Index-stable samples that cover every held-out run uniformly."""

    def __init__(
        self,
        base: CurveNavSandDataset,
        count: int,
        random_seed: int,
    ) -> None:
        if count < 1:
            raise ValueError("validation sample count must be positive")
        self.base = base
        run_ids = sorted(base.upstream.run_info)  # type: ignore[attr-defined]
        if not run_ids:
            raise RuntimeError("validation split contains no SanD runs")
        generator = random.Random(random_seed)
        generator.shuffle(run_ids)
        descriptors = []
        for index in range(count):
            run_id = run_ids[index % len(run_ids)]
            run = base.upstream.run_info[run_id]  # type: ignore[attr-defined]
            run_length = int(run["length"])
            min_gap = int(base.upstream.min_gap)  # type: ignore[attr-defined]
            max_gap = int(base.upstream.max_gap)  # type: ignore[attr-defined]
            start_index = generator.randint(0, run_length - min_gap - 1)
            end_index = generator.randint(
                start_index + min_gap,
                min(start_index + max_gap, run_length - 1),
            )
            descriptors.append((run_id, start_index, end_index))
        self.descriptors = tuple(descriptors)

    def __len__(self) -> int:
        return len(self.descriptors)

    def __getitem__(self, index: int) -> dict[str, object]:
        return self.base._prepare_sample(*self.descriptors[index])


class _OneBatchDataset(Dataset):
    def __init__(self, base: Dataset, count: int) -> None:
        self.base = base
        self.count = count

    def __len__(self) -> int:
        return self.count

    def __getitem__(self, index: int) -> dict[str, object]:
        return self.base[index]


@dataclass
class _PreparedSandCollator:
    codec: PlanarBSplineCodec

    def __call__(self, batch: list[dict[str, object]]) -> dict[str, Tensor]:
        depth_indices = torch.stack(
            [item["depth_indices"] for item in batch]  # type: ignore[list-item]
        )
        task_goal = torch.stack(
            [item["task_goal"] for item in batch]  # type: ignore[list-item]
        )
        motion_context = torch.stack(
            [item["motion_context"] for item in batch]  # type: ignore[list-item]
        )
        metric_paths = [item["metric_path"] for item in batch]
        canonical_path, control_points = collate_metric_paths(
            metric_paths,  # type: ignore[arg-type]
            self.codec,
        )
        return {
            "depth_indices": depth_indices,
            "task_goal": task_goal,
            "motion_context": motion_context,
            "control_points": control_points,
            "canonical_path": canonical_path,
        }


def _seed_worker(worker_id: int) -> None:
    del worker_id
    worker_seed = torch.initial_seed() % 2**32
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def _build_dataset(
    config: DataConfig,
    source: DataSourceConfig,
    samples_per_epoch: int,
    random_seed: int,
) -> CurveNavSandDataset:
    upstream = SandRunIndex(
        dataset_root=source.root,
        sequence_length=config.sequence_length,
        frame_skip=config.frame_skip,
        min_gap=config.min_gap,
        max_gap=config.max_gap,
        samples_per_epoch=samples_per_epoch,
        image_size=(config.image_height, config.image_width),
        depth_scale=config.depth_units_per_m,
        max_depth=config.max_depth_m,
        split_mode=source.split,
        train_ratio=0.9,
        random_seed=random_seed,
    )
    return CurveNavSandDataset(upstream)


def _build_source_datasets(
    config: DataConfig,
    sources: tuple[DataSourceConfig, ...],
    samples: int,
    random_seed: int,
) -> tuple[CurveNavSandDataset, ...]:
    return tuple(
        _build_dataset(
            config,
            source,
            samples,
            random_seed,
        )
        for source in sources
    )


def _collator(config: TrajectoryConfig) -> _PreparedSandCollator:
    return _PreparedSandCollator(
        PlanarBSplineCodec(
            num_control_points=config.num_control_points,
            degree=config.degree,
            num_path_points=config.num_path_points,
        )
    )


@dataclass(frozen=True)
class SandLoaderBundle:
    loader: DataLoader
    depth_bank: PackedDepthBankSpec


def build_sand_training_loader(
    data: DataConfig,
    trajectory: TrajectoryConfig,
    batch_size: int,
    samples_per_epoch: int,
    num_workers: int,
    prefetch_factor: int,
    split_seed: int,
    worker_seed: int,
) -> SandLoaderBundle:
    """Build the fixed-shape, pinned-memory production training loader."""
    if num_workers < 1:
        raise ValueError("production SanD loading requires at least one worker")
    generator = torch.Generator().manual_seed(worker_seed)
    datasets = _build_source_datasets(
        data,
        data.training_sources,
        samples_per_epoch,
        split_seed,
    )
    dataset = _WeightedSandDataset(
        datasets,
        tuple(source.weight for source in data.training_sources),
        samples_per_epoch,
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=True,
        prefetch_factor=prefetch_factor,
        collate_fn=_collator(trajectory),
        worker_init_fn=_seed_worker,
        generator=generator,
        in_order=False,
    )
    return SandLoaderBundle(loader=loader, depth_bank=dataset.depth_bank)


def build_sand_overfit_loader(
    data: DataConfig,
    trajectory: TrajectoryConfig,
    batch_size: int,
    random_seed: int,
) -> SandLoaderBundle:
    """Return one deterministic CPU-prepared batch for the overfit gate."""
    datasets = _build_source_datasets(
        data,
        data.training_sources,
        batch_size,
        random_seed,
    )
    dataset = _WeightedSandDataset(
        datasets,
        tuple(source.weight for source in data.training_sources),
        batch_size,
    )
    loader = DataLoader(
        _OneBatchDataset(dataset, batch_size),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        drop_last=True,
        collate_fn=_collator(trajectory),
    )
    return SandLoaderBundle(loader=loader, depth_bank=dataset.depth_bank)


def build_sand_validation_loader(
    data: DataConfig,
    trajectory: TrajectoryConfig,
    batch_size: int,
    samples: int,
    num_workers: int,
    random_seed: int,
) -> SandLoaderBundle:
    """Build the fixed, run-disjoint validation protocol for configured sources."""
    if num_workers < 1:
        raise ValueError("SanD validation requires at least one worker")
    bases = _build_source_datasets(
        data,
        data.validation_sources,
        samples,
        random_seed,
    )
    depth_bank, offsets = _combine_depth_banks(bases)
    total_weight = sum(source.weight for source in data.validation_sources)
    counts = [
        int(samples * source.weight / total_weight)
        for source in data.validation_sources
    ]
    for index in range(samples - sum(counts)):
        counts[index % len(counts)] += 1
    fixed_sources = [
        _OffsetDataset(
            _FixedSandValidationDataset(base, count, random_seed + index),
            offsets[index],
        )
        for index, (base, count) in enumerate(zip(bases, counts, strict=True))
        if count
    ]
    fixed = torch.utils.data.ConcatDataset(fixed_sources)
    loader = DataLoader(
        fixed,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
        persistent_workers=True,
        prefetch_factor=2,
        collate_fn=_collator(trajectory),
        worker_init_fn=_seed_worker,
        generator=torch.Generator().manual_seed(random_seed),
    )
    return SandLoaderBundle(loader=loader, depth_bank=depth_bank)
