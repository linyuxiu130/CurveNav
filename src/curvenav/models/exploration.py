"""X-NavDP structured proposals, expressed exactly in physical spline controls.

Reference: InternRobotics/NavDP@878740a2011856d0e3782dd6ccd880fd2eccd70f,
baselines/x-navdp/src/x_navdp/models/x_navdp_policy.py: generate_action_mix.
Upstream copyright (c) 2026 Tianyu Yang and X-NavDP contributors, MIT;
the full notice is retained in online_evaluation/baselines/x-navdp/LICENSE.
The proposal distribution is separate from the teacher's safety/goal evaluation;
perturbing a smooth curve does not certify its safety or trackability.
"""

import torch
from torch import Tensor

from curvenav.trajectory import IncrementalBSplineTrajectory


GOAL_DROPOUT_PROBABILITY = 0.2
EXPLORATION_RANDOM_DIM = 9
EXPLORATION_TYPE = "goal_nogoal_physical_spline_mix_flip_scale_v1"


def structured_proposals(
    codec: IncrementalBSplineTrajectory,
    coordinates: Tensor,
    random: Tensor,
) -> Tensor:
    """Map [N,2,C] goal/no-goal coordinates and [N,9] U[0,1) to controls.

    All mixing takes place *after* undoing increment normalization. Since spline
    decoding is linear in physical controls, mixing and axis transformations commute
    with decoding. The origin and cubic continuity survive; curvature bounds do not.
    No truncation, collision repair, kinematic projection or refitting is performed.
    """
    controls = codec.values_from_coordinates(coordinates.flatten(0, 1)).reshape(
        -1, 2, codec.num_control_tokens, 2
    )
    paths, _ = codec.decode_values(controls.flatten(0, 1).flatten(1))
    paths = paths.unflatten(0, (-1, 2))
    goal, nogoal = controls.unbind(1)
    opposite = paths[:, 0, 1, 0] * paths[:, 1, 1, 0] < 0
    alignment = torch.stack((1 - 2 * opposite.float(), torch.ones_like(random[:, 0])), -1)
    nogoal = nogoal * alignment[:, None]
    lengths = paths.diff(dim=2).norm(dim=-1).sum(-1)
    length_ratio = lengths[:, 0] / lengths[:, 1].clamp_min(1e-6)

    # The upstream straight-path mode has a random 1.5–3.5 m endpoint.
    fraction = torch.arange(1, codec.num_control_tokens + 1, device=goal.device) / codec.num_control_tokens
    line_x = (1.5 + 2 * random[:, 1, None]) * fraction
    line = torch.stack((line_x, torch.zeros_like(line_x)), -1)
    mixed = torch.where((random[:, 0] < 0.25)[:, None, None], line, goal)
    mixed = mixed + ((random[:, 2] - 0.5) * length_ratio)[:, None, None] * nogoal
    mixed = torch.where((random[:, 3] < 0.15)[:, None, None], nogoal, mixed)
    axis_sign = 1 - 2 * (random[:, 4:6] < 0.25).float()
    mixed = mixed * (axis_sign * (0.75 + 0.5 * random[:, 6:8]))[:, None]
    return torch.where((random[:, 8] < 0.2)[:, None, None], goal, mixed).flatten(1)
