"""High-throughput loading for the one prepared CurveNav dataset format."""

from __future__ import annotations

from dataclasses import dataclass

from torch.utils.data import DataLoader

from curvenav.config import DataConfig, TrajectoryConfig
from curvenav.data.depth_bank import PackedDepthBankSpec
from curvenav.data.prepared import PreparedPolicyDataset, RepeatedPolicyDataset


@dataclass(frozen=True)
class PolicyLoaderBundle:
    loader: DataLoader
    depth_bank: PackedDepthBankSpec
    samples: int


def build_policy_loader(
    data: DataConfig,
    trajectory: TrajectoryConfig,
    *,
    split: str,
    batch_size: int,
    num_workers: int,
    prefetch_factor: int,
    samples: int | None = None,
    seed: int = 0,
    sample_offset: int = 0,
) -> PolicyLoaderBundle:
    """Load precomputed labels; workers move only small fixed-shape tensors."""
    base = PreparedPolicyDataset(data.root, split, data, trajectory)
    dataset = (
        RepeatedPolicyDataset(base, samples, seed, sample_offset)
        if samples is not None
        else base
    )
    arguments = {
        "dataset": dataset,
        "batch_size": batch_size,
        "shuffle": False,
        "num_workers": num_workers,
        "pin_memory": True,
        "drop_last": samples is not None,
    }
    if num_workers:
        arguments.update(
            persistent_workers=True,
            prefetch_factor=prefetch_factor,
            in_order=split != "train",
        )
    loader = DataLoader(**arguments)
    return PolicyLoaderBundle(
        loader=loader,
        depth_bank=base.depth_bank,
        samples=len(dataset),
    )


def build_policy_training_loader(
    data: DataConfig,
    trajectory: TrajectoryConfig,
    batch_size: int,
    samples_per_epoch: int,
    num_workers: int,
    prefetch_factor: int,
    seed: int,
    sample_offset: int = 0,
) -> PolicyLoaderBundle:
    if num_workers < 1:
        raise ValueError("production policy loading requires at least one worker")
    return build_policy_loader(
        data,
        trajectory,
        split="train",
        batch_size=batch_size,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor,
        samples=samples_per_epoch,
        seed=seed,
        sample_offset=sample_offset,
    )


def build_policy_overfit_loader(
    data: DataConfig,
    trajectory: TrajectoryConfig,
    batch_size: int,
    seed: int = 0,
) -> PolicyLoaderBundle:
    return build_policy_loader(
        data,
        trajectory,
        split="train",
        batch_size=batch_size,
        num_workers=0,
        prefetch_factor=1,
        samples=batch_size,
        seed=seed,
    )


def build_policy_validation_loader(
    data: DataConfig,
    trajectory: TrajectoryConfig,
    batch_size: int,
    num_workers: int,
    samples: int | None = None,
) -> PolicyLoaderBundle:
    return build_policy_loader(
        data,
        trajectory,
        split="validation",
        batch_size=batch_size,
        num_workers=num_workers,
        prefetch_factor=2,
        samples=samples,
    )
