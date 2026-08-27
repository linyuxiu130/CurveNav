import torch

from curvenav.training.history import sample_observation_history_prefix
from curvenav.types import PolicyCondition


def test_training_marginalizes_only_over_available_history_suffixes() -> None:
    valid = torch.tensor(
        [
            [False, False, False, True],
            [False, False, True, True],
            [False, True, True, True],
            [True, True, True, True],
        ]
    )
    condition = PolicyCondition(
        depth=torch.zeros(4, 4, 1, 2, 2),
        point_goal=torch.zeros(4, 2),
        observation_to_current=torch.zeros(4, 4, 4),
        observation_valid=valid,
    )
    torch.manual_seed(11)
    first = sample_observation_history_prefix(condition)
    torch.manual_seed(11)
    second = sample_observation_history_prefix(condition)

    assert torch.equal(first.observation_valid, second.observation_valid)
    assert torch.all(first.observation_valid[:, -1])
    assert not torch.any(first.observation_valid & ~valid)
    assert not torch.any(
        first.observation_valid[:, :-1] & ~first.observation_valid[:, 1:]
    )
