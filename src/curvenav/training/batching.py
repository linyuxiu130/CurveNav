"""Exact global-batch partitioning for CurveNav data-parallel training."""

from collections.abc import Iterator
from dataclasses import dataclass

from torch.utils.data import Sampler


@dataclass(frozen=True)
class DistributedBatchLayout:
    """One optimizer step partitioned across ranks and bounded micro-batches."""

    rank_batch_sizes: tuple[int, ...]
    rank_offsets: tuple[int, ...]
    micro_batches_per_step: int


def build_distributed_batch_layout(
    global_batch_size: int,
    per_device_batch_size: int,
    world_size: int,
) -> DistributedBatchLayout:
    """Split every global batch without padding, duplication, or dropped samples."""
    if not 1 <= world_size <= 8:
        raise ValueError("world_size must be in [1, 8]")
    if per_device_batch_size < 1 or global_batch_size < world_size:
        raise ValueError("batch sizes must provide at least one sample per rank")
    base, remainder = divmod(global_batch_size, world_size)
    sizes = tuple(base + int(rank < remainder) for rank in range(world_size))
    offsets = tuple(rank * base + min(rank, remainder) for rank in range(world_size))
    micro_counts = tuple(
        (size + per_device_batch_size - 1) // per_device_batch_size
        for size in sizes
    )
    if len(set(micro_counts)) != 1:
        raise ValueError("all ranks must execute the same number of micro-batches")
    return DistributedBatchLayout(
        rank_batch_sizes=sizes,
        rank_offsets=offsets,
        micro_batches_per_step=micro_counts[0],
    )


class DistributedStepBatchSampler(Sampler[list[int]]):
    """Yield this rank's disjoint slice of every deterministic global batch."""

    def __init__(
        self,
        optimizer_steps: int,
        global_batch_size: int,
        per_device_batch_size: int,
        rank: int,
        world_size: int,
    ) -> None:
        if optimizer_steps < 1:
            raise ValueError("optimizer_steps must be positive")
        if not 0 <= rank < world_size:
            raise ValueError("rank must index the distributed world")
        self.optimizer_steps = optimizer_steps
        self.global_batch_size = global_batch_size
        self.per_device_batch_size = per_device_batch_size
        self.rank = rank
        self.layout = build_distributed_batch_layout(
            global_batch_size,
            per_device_batch_size,
            world_size,
        )

    def __len__(self) -> int:
        return self.optimizer_steps * self.layout.micro_batches_per_step

    def __iter__(self) -> Iterator[list[int]]:
        rank_size = self.layout.rank_batch_sizes[self.rank]
        rank_offset = self.layout.rank_offsets[self.rank]
        micro_count = self.layout.micro_batches_per_step
        base_micro_size, larger_micro_batches = divmod(rank_size, micro_count)
        for step in range(self.optimizer_steps):
            step_offset = step * self.global_batch_size + rank_offset
            micro_offset = 0
            for micro_index in range(micro_count):
                micro_size = base_micro_size + int(
                    micro_index < larger_micro_batches
                )
                yield list(
                    range(
                        step_offset + micro_offset,
                        step_offset + micro_offset + micro_size,
                    )
                )
                micro_offset += micro_size
