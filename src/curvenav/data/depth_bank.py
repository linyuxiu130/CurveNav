"""Immutable packed-depth metadata and device materialization."""

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
    """Load one split's normalized FP16 frames into immutable device storage."""
    bank = torch.empty(
        (spec.total_frames, spec.height, spec.width),
        dtype=torch.float16,
        device=device,
    )
    for run in spec.runs:
        packed = np.load(run.path, mmap_mode="c")
        bank[run.offset : run.offset + run.frames].copy_(torch.from_numpy(packed))
    return bank


def gather_depth_sequences(bank: Tensor, indices: Tensor) -> Tensor:
    """Materialize ``[B,T,1,H,W]`` depth without CPU image copies or H2D payloads."""
    batch_size, sequence_length = indices.shape
    return bank.index_select(0, indices.flatten()).view(
        batch_size,
        sequence_length,
        1,
        bank.shape[1],
        bank.shape[2],
    )
