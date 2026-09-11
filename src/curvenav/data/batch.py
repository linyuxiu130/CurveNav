"""Tensor-only data-loader batches to typed CurveNav model inputs."""

from dataclasses import dataclass, fields
from typing import Mapping

from curvenav.types import PolicyCondition, TrajectoryTarget


@dataclass
class PreparedPolicyBatch:
    condition: PolicyCondition
    target: TrajectoryTarget


def unpack_policy_batch(batch: Mapping[str, object]) -> PreparedPolicyBatch:
    """Rebuild typed model inputs from the worker-prepared tensor-only batch."""
    names = tuple(field.name for field in fields(PolicyCondition))
    condition = PolicyCondition(**{name: batch[name] for name in names})
    target = TrajectoryTarget(
        curve_values=batch["curve_values"],  # type: ignore[arg-type]
    )
    condition.validate()
    target.validate()
    return PreparedPolicyBatch(
        condition=condition,
        target=target,
    )
