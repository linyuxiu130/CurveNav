"""Tensor-only data-loader batches to typed CurveNav model inputs."""

from dataclasses import dataclass
from typing import Mapping

from torch import Tensor

from curvenav.types import PolicyCondition, TrajectoryTarget


@dataclass
class PreparedPolicyBatch:
    condition: PolicyCondition
    target: TrajectoryTarget
    flow_interval_group: Tensor


def unpack_policy_batch(batch: Mapping[str, object]) -> PreparedPolicyBatch:
    """Rebuild typed model inputs from the worker-prepared tensor-only batch."""
    required = (
        "depth",
        "point_goal",
        "observation_to_current",
        "observation_valid",
        "curve_values",
        "flow_interval_group",
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
        curve_values=batch["curve_values"],  # type: ignore[arg-type]
    )
    flow_interval_group = batch["flow_interval_group"]
    if (
        not isinstance(flow_interval_group, Tensor)
        or flow_interval_group.shape != condition.point_goal.shape[:1]
        or flow_interval_group.dtype
        not in (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64)
    ):
        raise ValueError("flow_interval_group must be an integer tensor with shape [B]")
    condition.validate()
    target.validate()
    return PreparedPolicyBatch(
        condition=condition,
        target=target,
        flow_interval_group=flow_interval_group,
    )
