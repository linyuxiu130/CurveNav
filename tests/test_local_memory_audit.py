"""Ordering checks for the experimental target; no navigation-success claim."""

import torch

from curvenav.evaluation.local_memory_audit import progress_utility, summarize


def test_progress_target_preserves_margin_and_prefers_verified_progress_to_stop():
    progress = torch.tensor([0., .8, .8, -.2, -.2, 0.])
    length = torch.tensor([0., 1.2, 1.2, .3, 2., 1.])
    clearance = torch.tensor([.2, .11, .03, .2, .2, .2])
    score = progress_utility(progress, length, clearance, horizon=3.6, radius=.42, margin=.1)
    assert score[1] > score[0] == 0
    assert score[2] < -1  # .03 m is NOT admitted under the existing .10 m contract.
    assert 0 > score[3] > score[4] > -1
    assert score[5] < 0  # A loop without progress does not beat stopping.
    # An arbitrarily inefficient but feasible detour can have positive progress.
    old = (.2 - .5 * 1.) / 3.6
    new = progress_utility(torch.tensor(.2), torch.tensor(1.), torch.tensor(.11),
                           horizon=3.6, radius=.42, margin=.1)
    assert old < 0 < new


def test_summary_separates_coverage_selection_and_expert_diagnostics():
    import numpy as np
    arrays = {f'x_{key}': np.zeros((2, 34))
              for key in ('clearance', 'progress', 'length', 'scores', 'old_target', 'progress_target')}
    arrays['x_clearance'].fill(.2)
    arrays['x_progress'][0, 0] = 1
    arrays['x_scores'][0, 1] = 1  # Missed a useful neural candidate.
    arrays['x_scores'][:, 33] = 100  # Expert is never in the selectable bank.
    result = summarize(arrays, ['x'], .1, .2)['x']
    assert result['bank_has_safe_progress'] == 1
    assert result['selection_miss_given_coverage'] == 1
    assert result['selected_safe_progress'] == 0


def test_occupied_only_memory_aliases_past_free_and_past_unknown():
    """A history older than the image window cannot leave free evidence here.

    This records a limitation of the old contract, not a desired property of a
    future free-space memory. A known-free ray and an invalid ray are distinct.
    """
    import numpy as np
    from curvenav.data.obstacle_memory import ObstacleMemory
    k = np.array([[16., 0., 8.], [0., 16., 8.], [0., 0., 1.]], dtype=np.float32)
    # Optical Z -> body forward X; optical X/Y -> body -Y/-Z.
    camera = np.array([[0., 0., 1., 0.], [-1., 0., 0., 0.],
                       [0., -1., 0., .7], [0., 0., 0., 1.]], dtype=np.float32)
    previous_pose = np.array([[0., -1., 0., 0.], [1., 0., 0., 0.],
                              [0., 0., 1., 0.], [0., 0., 0., 1.]], dtype=np.float32)
    seen, unseen = ObstacleMemory(3.6, 5., (.42, .02, 1.5)), ObstacleMemory(3.6, 5., (.42, .02, 1.5))
    seen.update(np.ones((16, 16), np.float32), k, camera, previous_pose)
    unseen.update(np.zeros((16, 16), np.float32), k, camera, previous_pose)
    current_depth = np.full((16, 16), .2, np.float32)
    first = seen.update(current_depth, k, camera, np.eye(4))
    second = unseen.update(current_depth, k, camera, np.eye(4))
    assert first.any()
    np.testing.assert_array_equal(first, second)
    np.testing.assert_array_equal(seen.voxels, unseen.voxels)


def test_goal_sweep_keeps_connected_detour_and_rejects_clear_shortcuts():
    import numpy as np
    from dataclasses import replace
    from types import SimpleNamespace
    from curvenav.data.privileged import SourceConfigurationGrid, SourceConfigurationSpaceQuery
    from curvenav.training.critic import RouteUtilityTeacher
    from curvenav.evaluation.local_memory_audit import detour_goals
    field = np.ones((80, 80), dtype=np.float32)
    field[50:54, 35:45] = -.2  # finite wall across the forward direct segment
    grid = SourceConfigurationGrid(field, np.array([-2., -2.]), .05)
    query = SourceConfigurationSpaceQuery((grid,))
    margin_query = SourceConfigurationSpaceQuery((replace(grid, signed_clearance_m=field-.1),))
    batch = dict(source_grid_index=torch.tensor([0]), source_origin_xy=torch.zeros(1, 2),
                 source_yaw_rad=torch.zeros(1))
    dataset = SimpleNamespace(__getitems__=lambda indices: batch)
    indices, goals, rejected = detour_goals(dataset, [7], RouteUtilityTeacher(query, 3.6),
                                          RouteUtilityTeacher(margin_query, 3.6))
    assert indices == [7]
    torch.testing.assert_close(goals, torch.tensor([[1.5, 0.]]))
    assert rejected == dict(goal_not_margin_safe=0, disconnected=0, direct_path_clear=3)
