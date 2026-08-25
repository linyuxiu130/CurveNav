"""Tensor-only data-loader batches to typed CurveNav model inputs."""

from dataclasses import dataclass
from typing import Mapping

from torch import Tensor

from curvenav.types import PolicyCondition, TrajectoryTarget


@dataclass
class PreparedPolicyBatch:
    condition: PolicyCondition
    target: TrajectoryTarget


def unpack_policy_batch(batch: Mapping[str, object]) -> PreparedPolicyBatch:
    """Rebuild typed model inputs from the worker-prepared tensor-only batch."""
    required = (
        "depth",
        "point_goal",
        "observation_to_current",
        "observation_valid",
        "control_points",
        "reference_path",
    )
    missing = [key for key in required if key not in batch]
    if missing:
        raise KeyError(f"prepared policy batch is missing fields: {missing}")
    values = [batch[key] for key in required]
    if not all(isinstance(value, Tensor) for value in values):
        raise TypeError("prepared policy batch fields must be tensors")
    condition = PolicyCondition(
        depth=batch["depth"],  # type: ignore[arg-type]
        point_goal=batch["point_goal"],  # type: ignore[arg-type]
        observation_to_current=batch["observation_to_current"],  # type: ignore[arg-type]
        observation_valid=batch["observation_valid"],  # type: ignore[arg-type]
    )
    target = TrajectoryTarget(
        control_points=batch["control_points"],  # type: ignore[arg-type]
        reference_path=batch["reference_path"],  # type: ignore[arg-type]
    )
    condition.validate()
    target.validate()
    return PreparedPolicyBatch(
        condition=condition,
        target=target,
    )
