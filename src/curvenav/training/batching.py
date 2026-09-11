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
    micro_count, remainder = divmod(global_batch_size, world_size * per_device_batch_size)
    if remainder or micro_count < 1:
        raise ValueError("global batch must contain complete fixed-size micro-batches")
    rank_size = micro_count * per_device_batch_size
    return DistributedBatchLayout(
        rank_batch_sizes=(rank_size,) * world_size,
        rank_offsets=tuple(rank * rank_size for rank in range(world_size)),
        micro_batches_per_step=micro_count,
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
        rank_offset = self.layout.rank_offsets[self.rank]
        micro_count = self.layout.micro_batches_per_step
        for step in range(self.optimizer_steps):
            step_offset = step * self.global_batch_size + rank_offset
            for micro_index in range(micro_count):
                micro_offset = micro_index * self.per_device_batch_size
                yield list(
                    range(
                        step_offset + micro_offset,
                        step_offset + micro_offset + self.per_device_batch_size,
                    )
                )
