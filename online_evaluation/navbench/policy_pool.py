"""One logical policy batch sharded across identical policy servers."""

from __future__ import annotations

import concurrent.futures
from dataclasses import dataclass

import numpy as np

from navbench.client import navigator_reset, navigator_shutdown, pointgoal_step


@dataclass(frozen=True)
class BatchShard:
    port: int
    start: int
    stop: int


class PolicyPool:
    """Run one logical policy batch on one or more identical GPU servers."""

    def __init__(self, ports: list[int], batch_size: int) -> None:
        if not ports or len(ports) > batch_size:
            raise ValueError("policy server count must be in [1, batch_size]")
        boundaries = np.linspace(0, batch_size, len(ports) + 1, dtype=np.int64)
        self.shards = [
            BatchShard(port, int(start), int(stop))
            for port, start, stop in zip(ports, boundaries[:-1], boundaries[1:])
        ]
        self.batch_size = batch_size
        self.executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=len(self.shards), thread_name_prefix="policy-shard"
        )

    @property
    def ports(self) -> list[int]:
        return [shard.port for shard in self.shards]

    def reset_all(
        self,
        intrinsic: np.ndarray,
        sample_indices: list[int],
        scene_name: str,
    ) -> dict:
        futures = [
            self.executor.submit(
                navigator_reset,
                intrinsic,
                batch_size=shard.stop - shard.start,
                global_batch_size=self.batch_size,
                batch_start=shard.start,
                port=shard.port,
                sample_indices=sample_indices[shard.start:shard.stop],
                scene_name=scene_name,
            )
            for shard in self.shards
        ]
        responses = [future.result() for future in futures]
        if any(response != responses[0] for response in responses[1:]):
            raise ValueError("policy shards must use the same observation contract")
        return responses[0]

    def step(self, observation: dict[str, np.ndarray]) -> np.ndarray:
        futures = [
            self.executor.submit(
                pointgoal_step,
                **{
                    key: value[shard.start:shard.stop]
                    for key, value in observation.items()
                },
                port=shard.port,
            )
            for shard in self.shards
        ]
        return np.concatenate(
            [future.result()[0] for future in futures], axis=0
        )

    def reset_env(self, env_id: int, sample_idx: int, scene_name: str) -> None:
        for shard in self.shards:
            if shard.start <= env_id < shard.stop:
                navigator_reset(
                    env_id=env_id - shard.start,
                    port=shard.port,
                    sample_idx=sample_idx,
                    scene_name=scene_name,
                )
                return
        raise IndexError(env_id)

    def close(self) -> None:
        try:
            futures = [
                self.executor.submit(navigator_shutdown, port=shard.port)
                for shard in self.shards
            ]
            for future in futures:
                future.result()
        finally:
            self.executor.shutdown(wait=True)
