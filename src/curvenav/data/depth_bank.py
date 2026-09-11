"""Immutable packed depth metadata and device materialization."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import Tensor


@dataclass(frozen=True)
class PackedDepthRun:
    path: Path
    offset: int
    frames: int


@dataclass(frozen=True)
class PackedDepthBankSpec:
    runs: tuple[PackedDepthRun, ...]
    total_frames: int
    height: int
    width: int


def load_packed_depth_bank(
    spec: PackedDepthBankSpec,
    device: torch.device,
) -> Tensor:
    """Load normalized FP16 depth into immutable device storage."""
    bank = torch.empty(
        (spec.total_frames, spec.height, spec.width),
        dtype=torch.float16,
        device=device,
    )
    for run in spec.runs:
        packed = np.load(run.path, mmap_mode="c")
        if (
            packed.shape != (run.frames, spec.height, spec.width)
            or packed.dtype != np.float16
        ):
            raise ValueError("invalid packed depth")
        if not np.isfinite(packed).all() or np.any(packed < 0) or np.any(packed > 1):
            raise ValueError("packed depth must be normalized [0,1], with zero invalid")
        bank[run.offset : run.offset + run.frames].copy_(torch.from_numpy(packed))
    return bank


def gather_depth_observations(bank: Tensor, indices: Tensor) -> Tensor:
    """Gather ``[B,F,1,H,W]`` depth from the immutable frame bank."""
    batch_size, observation_frames = indices.shape
    return bank.index_select(0, indices.flatten().long()).view(
        batch_size,
        observation_frames,
        1,
        bank.shape[1],
        bank.shape[2],
    )
