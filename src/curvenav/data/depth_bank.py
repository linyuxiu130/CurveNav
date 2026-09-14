"""Immutable packed depth metadata and device materialization."""

from dataclasses import dataclass
import fcntl
import hashlib
import os
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


class PackedDepthBank:
    """Read immutable depth frames through shared filesystem page cache."""

    def __init__(self, spec: PackedDepthBankSpec) -> None:
        self.height = spec.height
        self.width = spec.width
        digest = hashlib.sha256()
        for run in spec.runs:
            stat = run.path.stat()
            digest.update(str((run.path, run.offset, run.frames,
                               stat.st_size, stat.st_mtime_ns)).encode())
        cache_root = Path(
            os.environ.get("CURVENAV_DEPTH_CACHE_DIR", "/tmp/curvenav-depth-cache")
        )
        cache_root.mkdir(parents=True, exist_ok=True)
        cache_path = cache_root / f"{digest.hexdigest()}.npy"
        with cache_path.with_suffix(".lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            if not cache_path.exists():
                temporary = cache_path.with_suffix(f".{os.getpid()}.tmp")
                values = np.lib.format.open_memmap(
                    temporary, mode="w+", dtype=np.float16,
                    shape=(spec.total_frames, spec.height, spec.width),
                )
                for run in spec.runs:
                    source = np.load(run.path, mmap_mode="r")
                    if source.shape != (run.frames, spec.height, spec.width) or source.dtype != np.float16:
                        raise ValueError("invalid packed depth")
                    values[run.offset:run.offset + run.frames] = source
                values.flush()
                del values
                os.replace(temporary, cache_path)
        self.values = torch.from_numpy(np.load(cache_path, mmap_mode="c"))

    def gather(self, indices: Tensor, pin_memory: bool = False) -> Tensor:
        flat = indices.numpy().reshape(-1)
        if flat.min() < 0 or flat.max() >= len(self.values):
            raise IndexError("packed depth index is out of range")
        output = torch.empty(
            (*indices.shape, 1, self.height, self.width),
            dtype=torch.float16,
            pin_memory=pin_memory,
        )
        torch.index_select(
            self.values,
            0,
            indices.flatten().long(),
            out=output.view(-1, self.height, self.width),
        )
        return output


def load_packed_depth_bank(
    spec: PackedDepthBankSpec,
 ) -> PackedDepthBank:
    return PackedDepthBank(spec)


def gather_depth_observations(bank: PackedDepthBank, indices: Tensor) -> Tensor:
    """Gather ``[B,F,1,H,W]`` depth from the immutable frame bank."""
    return bank.gather(indices)
