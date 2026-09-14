"""High-throughput loading for the one prepared CurveNav dataset format."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset, default_convert

from curvenav.config import DataConfig, TrajectoryConfig
from curvenav.data.depth_bank import PackedDepthBankSpec
from curvenav.data.prepared import PreparedPolicyDataset, RepeatedPolicyDataset
from curvenav.training.batching import DistributedStepBatchSampler


@dataclass(frozen=True)
class PolicyLoaderBundle:
    loader: DataLoader
    depth_bank: PackedDepthBankSpec
    samples: int


def source_balanced_indices(source_indices: np.ndarray, samples_per_source: int) -> list[int]:
    """Fixed uniform sampling without replacement within every source scene."""
    if samples_per_source < 1:
        raise ValueError("samples_per_source must be positive")
    rng = np.random.default_rng(0)
    selected = []
    for source in np.unique(source_indices):
        indices = np.flatnonzero(source_indices == source)
        selected.extend(rng.choice(indices, min(len(indices), samples_per_source), replace=False))
    return np.sort(selected).tolist()


def build_policy_validation_loader(
    data: DataConfig,
    trajectory: TrajectoryConfig,
    batch_size: int,
    num_workers: int,
    rank: int = 0,
    world_size: int = 1,
    samples_per_source: int | None = None,
) -> PolicyLoaderBundle:
    """Load precomputed labels; workers move only small fixed-shape tensors."""
    base = PreparedPolicyDataset(data.root, "validation", data, trajectory)
    indices = (
        range(len(base)) if samples_per_source is None else
        source_balanced_indices(base.arrays["source_grid_index"], samples_per_source)
    )
    dataset = Subset(base, indices[rank::world_size])
    arguments = {
        "dataset": dataset,
        "batch_size": batch_size,
        "shuffle": False,
        "num_workers": num_workers,
        "pin_memory": True,
        "collate_fn": default_convert,
        "drop_last": False,
        # Worker construction must not advance the model process RNG.
        "generator": torch.Generator().manual_seed(0),
    }
    if num_workers:
        arguments.update(
            persistent_workers=True,
            prefetch_factor=2,
            # Evaluation follows the stored validation order.
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
        collate_fn=default_convert,
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
