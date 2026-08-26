"""High-throughput loading for the one prepared CurveNav dataset format."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.utils.data import DataLoader

from curvenav.config import DataConfig, TrajectoryConfig
from curvenav.data.depth_bank import PackedDepthBankSpec
from curvenav.data.prepared import PreparedPolicyDataset, RepeatedPolicyDataset
from curvenav.training.batching import DistributedStepBatchSampler


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
        # Worker base seeds must not advance the model process RNG.  This also
        # makes worker construction identical after an exact resume.
        "generator": torch.Generator().manual_seed(seed),
    }
    if num_workers:
        arguments.update(
            persistent_workers=True,
            prefetch_factor=prefetch_factor,
            # The resume offset denotes a prefix of the deterministic sample
            # stream, so yielded batches must preserve that order.
            in_order=True,
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
    optimizer_steps: int,
    global_batch_size: int,
    per_device_batch_size: int,
    rank: int,
    world_size: int,
    num_workers: int,
    prefetch_factor: int,
    seed: int,
    sample_offset: int = 0,
) -> PolicyLoaderBundle:
    if num_workers < 1:
        raise ValueError("production policy loading requires at least one worker")
    samples = optimizer_steps * global_batch_size
    base = PreparedPolicyDataset(data.root, "train", data, trajectory)
    dataset = RepeatedPolicyDataset(base, samples, seed, sample_offset)
    batch_sampler = DistributedStepBatchSampler(
        optimizer_steps,
        global_batch_size,
        per_device_batch_size,
        rank,
        world_size,
    )
    loader = DataLoader(
        dataset,
        batch_sampler=batch_sampler,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=prefetch_factor,
        in_order=True,
        generator=torch.Generator().manual_seed(seed),
    )
    return PolicyLoaderBundle(
        loader=loader,
        depth_bank=base.depth_bank,
        samples=samples,
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
