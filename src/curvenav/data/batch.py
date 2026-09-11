"""Tensor-only data-loader batches to typed CurveNav model inputs."""

from dataclasses import dataclass, fields
from typing import Mapping

import torch
from torch import Tensor

from curvenav.types import PolicyCondition, TrajectoryTarget


@dataclass
class PreparedPolicyBatch:
    condition: PolicyCondition
    target: TrajectoryTarget
    flow_interval_group: Tensor


def unpack_policy_batch(batch: Mapping[str, object]) -> PreparedPolicyBatch:
    """Rebuild typed model inputs from the worker-prepared tensor-only batch."""
    names = tuple(field.name for field in fields(PolicyCondition))
    condition = PolicyCondition(**{name: batch[name] for name in names})
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
