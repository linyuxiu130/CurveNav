"""Overlap pinned-memory batch transfer with the preceding CUDA training step."""

from collections.abc import Iterator, Mapping
from typing import Protocol

import torch
from torch import Tensor

from curvenav.data.depth_bank import (
    PackedDepthBank,
    PackedDepthBankSpec,
    load_packed_depth_bank,
)


TensorBatch = Mapping[str, Tensor]


class BatchLoader(Protocol):
    def __len__(self) -> int: ...

    def __iter__(self) -> Iterator[TensorBatch]: ...


class CudaPrefetchLoader:
    """Gather immutable depth and move one batch ahead on a CUDA stream."""

    def __init__(
        self,
        loader: BatchLoader,
        depth_bank: PackedDepthBankSpec,
        device: torch.device,
    ) -> None:
        self.loader = loader
        self.depth_bank_spec = depth_bank
        self.device = device
        self._depth_bank: PackedDepthBank | None = None

    def __len__(self) -> int:
        return len(self.loader)

    def __iter__(self) -> Iterator[dict[str, Tensor]]:
        iterator = iter(self.loader)
        cpu_batch = next(iterator)
        if self._depth_bank is None:
            self._depth_bank = load_packed_depth_bank(
                self.depth_bank_spec,
            )
        prefetch_stream = torch.cuda.Stream(device=self.device)
        with torch.cuda.stream(prefetch_stream):
            gpu_batch = self._to_device(cpu_batch)

        while True:
            current_stream = torch.cuda.current_stream(self.device)
            current_stream.wait_stream(prefetch_stream)
            current_batch = gpu_batch
            for value in current_batch.values():
                value.record_stream(current_stream)

            try:
                cpu_batch = next(iterator)
            except StopIteration:
                yield current_batch
                return
            with torch.cuda.stream(prefetch_stream):
                gpu_batch = self._to_device(cpu_batch)
            yield current_batch

    def _to_device(self, batch: TensorBatch) -> dict[str, Tensor]:
        moved = {
            name: value.to(device=self.device, non_blocking=True)
            for name, value in batch.items()
            if name != "depth_indices"
        }
        # The 10 Hz corpus grows independently of GPU capacity. Keep immutable
        # banks in host RAM and transfer only the selected four-frame batch.
        indices = batch["depth_indices"]
        depth = self._depth_bank.gather(indices, pin_memory=True)
        moved["depth"] = depth.to(self.device, non_blocking=True)
        return moved
