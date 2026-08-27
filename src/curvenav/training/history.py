"""Training distribution over physically available observation histories."""

import torch

from curvenav.types import PolicyCondition


HISTORY_TRAINING_DISTRIBUTION = "uniform_valid_observation_suffix_marginalization"


def sample_observation_history_prefix(
    condition: PolicyCondition,
) -> PolicyCondition:
    """Uniformly marginalize every sample over its valid history suffixes."""
    valid_count = condition.observation_valid.sum(dim=1)
    keep_count = (
        torch.rand(valid_count.shape, device=valid_count.device)
        * valid_count.float()
    ).floor().long() + 1
    frame = torch.arange(
        condition.observation_valid.shape[1],
        device=valid_count.device,
    )
    sampled_valid = condition.observation_valid & (
        frame[None] >= condition.observation_valid.shape[1] - keep_count[:, None]
    )
    return PolicyCondition(
        depth=condition.depth,
        point_goal=condition.point_goal,
        observation_to_current=condition.observation_to_current,
        observation_valid=sampled_valid,
    )
