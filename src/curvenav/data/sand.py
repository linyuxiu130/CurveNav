"""The single SanD-to-CurveNav batch adapter."""

from dataclasses import dataclass
from typing import Mapping

from torch import Tensor

from curvenav.types import PolicyCondition, TrajectoryTarget


@dataclass
class PreparedSandBatch:
    condition: PolicyCondition
    target: TrajectoryTarget
    canonical_path: Tensor


def unpack_prepared_sand_batch(batch: Mapping[str, object]) -> PreparedSandBatch:
    """Rebuild typed model inputs from the worker-prepared tensor-only batch."""
    required = ("depth", "task_goal", "motion_context", "control_points", "canonical_path")
    missing = [key for key in required if key not in batch]
    if missing:
        raise KeyError(f"prepared SanD batch is missing fields: {missing}")
    values = [batch[key] for key in required]
    if not all(isinstance(value, Tensor) for value in values):
        raise TypeError("prepared SanD batch fields must be tensors")
    condition = PolicyCondition(
        depth=batch["depth"],  # type: ignore[arg-type]
        task_goal=batch["task_goal"],  # type: ignore[arg-type]
        motion_context=batch["motion_context"],  # type: ignore[arg-type]
    )
    target = TrajectoryTarget(control_points=batch["control_points"])  # type: ignore[arg-type]
    condition.validate()
    target.validate()
    return PreparedSandBatch(
        condition=condition,
        target=target,
        canonical_path=batch["canonical_path"],  # type: ignore[arg-type]
    )
