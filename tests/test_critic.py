"""Score supervision, exact candidate selection, and inference conditioning."""
import numpy as np
import torch

from curvenav.data.privileged import SourceConfigurationGrid, SourceConfigurationSpaceQuery
from curvenav.training.critic import CurveNavCriterion, RouteUtilityTeacher, candidate_query


def oracle():
    clearance = np.ones((200, 200), dtype=np.float32)
    clearance[110:120, :] = -0.2  # world x in [1,2): an impassable wall
    return SourceConfigurationSpaceQuery((SourceConfigurationGrid(clearance, np.array([-10., -10.]), 0.1),))


def source_batch(batch, device):
    return dict(source_grid_index=torch.zeros(batch, device=device, dtype=torch.long),
                source_origin_xy=torch.zeros(batch, 2, device=device),
                source_yaw_rad=torch.zeros(batch, device=device),
                point_goal=torch.tensor([[0., 3.]], device=device).expand(batch, -1))


def criterion_loss(output):
    return CurveNavCriterion(oracle(), 3.6)(output, source_batch(len(output.candidate_paths), output.candidate_paths.device))


def test_score_loss_regresses_actual_candidate_utilities():
    from curvenav.models.policy import CurveNavTrainingOutput
    paths = torch.tensor([[[[0., 0.], [0., 0.]], [[0., 0.], [0., 2.]],
                           [[0., 0.], [3., 2.]]]]).expand(2, -1, -1, -1)
    scores = torch.tensor([[1., 2., -1.], [-2., .5, 3.]], requires_grad=True)
    criterion = CurveNavCriterion(oracle(), 3.6)
    batch = source_batch(2, 'cpu')
    target = criterion.teacher(paths, batch).score
    def loss(value):
        return criterion(CurveNavTrainingOutput(torch.tensor(0.), paths, value), batch).critic_loss
    actual = loss(scores)
    expected = torch.nn.functional.smooth_l1_loss(scores, target)
    torch.testing.assert_close(actual, expected)
    gradient = torch.autograd.grad(actual, scores)[0]
    torch.testing.assert_close(gradient, (scores - target).clamp(-1, 1) / scores.numel())


def test_teacher_queries_body_clearance_whole_curve_and_is_batch_independent():
    # Endpoints clear, intermediate segment crosses wall beyond a nominal 0.5 m horizon.
    paths = torch.tensor([[[[0., 0.], [0., 3.]], [[0., 0.], [3., 0.]], [[0., 0.], [30., 0.]]]])
    teacher = RouteUtilityTeacher(oracle(), 0.5)
    truth = teacher(paths, source_batch(1, 'cpu')).clearance_m
    torch.testing.assert_close(truth, torch.tensor([[1., -0.2, -0.5]]))
    for i in range(3):
        alone = teacher(paths[:, i:i+1], source_batch(1, 'cpu')).clearance_m
        torch.testing.assert_close(alone[:, 0], truth[:, i])


def test_score_is_goal_conditioned_and_selection_returns_exact_candidate():
    from test_policy import tiny_config, make_condition
    from curvenav.factory import build_policy
    torch.manual_seed(9)
    policy = build_policy(tiny_config()).eval()
    condition = make_condition(1)
    encoded = policy.encode_condition(condition)
    evaluator = policy.trajectory_evaluator
    memory = evaluator.project_condition_memory(encoded)
    path, _ = policy.curve_codec.decode(torch.randn(1, 14))
    score = evaluator(path, condition.point_goal, encoded, memory)
    other = evaluator(path, -condition.point_goal, encoded, memory)
    assert not torch.equal(score, other)
    prediction = policy.sample(condition)
    assert prediction.candidates.shape[:2] == (1, 32)
    torch.testing.assert_close(prediction.path[0], prediction.candidates[0, prediction.scores[0].argmax()], rtol=0, atol=0)
    assert policy.inference_source.unique(dim=0).shape[0] == 32


def test_teacher_prefers_progress_to_stopping_and_does_not_credit_wall_crossing():
    teacher = RouteUtilityTeacher(oracle(), 3.6)
    batch = source_batch(1, 'cpu')
    paths = torch.tensor([[[[0., 0.], [0., 0.]], [[0., 0.], [0., 2.]], [[0., 0.], [3., 2.]]]])
    labels = teacher(paths, batch)
    assert labels.score[0, 1] > labels.score[0, 0] > labels.score[0, 2]
    assert labels.clearance_m[0, 2] < 0
    # Ranking is independent of the other trajectories in a batch.
    for i in range(3):
        alone = teacher(paths[:, i:i+1], batch)
        torch.testing.assert_close(alone.score[:, 0], labels.score[:, i])


def test_teacher_ranking_follows_actual_goal_for_identical_safe_candidates():
    teacher = RouteUtilityTeacher(oracle(), 3.6)
    batch = source_batch(1, 'cpu')
    paths = torch.tensor([[[[0., 0.], [0., 2.]], [[0., 0.], [0., -2.]]]])
    forward = teacher(paths, batch)
    backward = teacher(paths, {**batch, 'point_goal': -batch['point_goal']})
    assert forward.score.argmax(1).item() == 0
    assert backward.score.argmax(1).item() == 1
    torch.testing.assert_close(forward.clearance_m, backward.clearance_m)
    # Zero goal means arrival, not exploration: the utility favors stopping.
    stopped = torch.zeros_like(paths[:, :1])
    zero_goal = teacher(torch.cat((stopped, paths), dim=1),
                        {**batch, 'point_goal': torch.zeros_like(batch['point_goal'])})
    assert zero_goal.score.argmax(1).item() == 0


def test_geodesic_progress_rewards_moving_away_from_goal_to_exit_dead_end():
    # Wall separates start and goal except for an opening at world y > 2.
    clear = np.ones((100, 100), np.float32)
    clear[50:53, :70] = -0.1
    query = SourceConfigurationSpaceQuery((SourceConfigurationGrid(clear, np.array([-5., -5.]), .1),))
    teacher = RouteUtilityTeacher(query, 3.6)
    batch = source_batch(1, 'cpu')
    batch['source_origin_xy'][:] = torch.tensor([-1., 0.])
    batch['point_goal'][:] = torch.tensor([2., 0.])
    # local -y is positive world y at yaw=0. Both ends stay before the wall.
    paths = torch.tensor([[[[0., 0.], [0., -1.]], [[0., 0.], [.7, 0.]]]])
    labels = teacher(paths, batch)
    assert labels.progress_m[0, 0] > 0
    assert labels.score[0, 0] > labels.score[0, 1]


def test_utility_is_invariant_to_robot_frame_rotation():
    teacher = RouteUtilityTeacher(oracle(), 3.6)
    batch = source_batch(1, 'cpu')
    batch['source_origin_xy'][:] = torch.tensor([.023, .041])
    paths = torch.tensor([[[[0., 0.], [0., 2.]], [[0., 0.], [3., 1.]]]])
    expected = teacher(paths, batch)
    angle = torch.tensor(.73)
    rotation = torch.stack((torch.stack((angle.cos(), -angle.sin())),
                            torch.stack((angle.sin(), angle.cos()))))
    rotated = dict(batch)
    rotated['source_yaw_rad'] = batch['source_yaw_rad'] + angle
    rotated['point_goal'] = batch['point_goal'] @ rotation.T
    actual = teacher(paths @ rotation.T, rotated)
    torch.testing.assert_close(actual.score, expected.score, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(actual.progress_m, expected.progress_m)


def test_source_trace_detects_short_corner_crossing_and_preserves_vertices():
    clear = np.ones((20, 20), np.float32)
    clear[1, 1] = -.1
    query = SourceConfigurationSpaceQuery((SourceConfigurationGrid(clear, np.zeros(2), .1),))
    path = torch.tensor([[[0., 0.], [.101, .101]]])
    result = query.query(path, torch.tensor([0]), torch.tensor([[.05, .151]]), torch.tensor([0.]), 3.6)
    assert result.minimum_clearance_m.item() < 0
    assert ((result.grid_cells[0] == torch.tensor([1, 1])).all(-1) & result.active[0]).any()
    points = result.local_path[0, result.active[0]]
    torch.testing.assert_close(points[0], path[0, 0])
    torch.testing.assert_close(points[-1], path[0, -1])
    assert torch.diff(points, dim=0).norm(dim=-1).max() <= .025001


def test_closed_cell_contacts_agree_with_geodesic_connectivity():
    # All side-cell arrangements, both traversal directions, and exact edge contact.
    for side_x, side_y in ((.2, .2), (-.1, .2), (.2, -.1), (-.1, -.1)):
        grid = SourceConfigurationGrid(np.array([[.2, side_y], [side_x, .2]], np.float32), np.zeros(2), 1.)
        query = SourceConfigurationSpaceQuery((grid,))
        teacher = RouteUtilityTeacher(query, 3.6)
        for reverse in (False, True):
            batch = source_batch(1, 'cpu')
            batch['source_origin_xy'][:] = 1.5 if reverse else .5
            batch['point_goal'][:] = torch.tensor([[.1, 0.]])
            path = torch.tensor([[[[0., 0.], [-1., 1.] if reverse else [1., -1.]]]])
            result = teacher(path, batch)
            assert bool(result.clearance_m[0, 0] < 0) == (min(side_x, side_y) < 0)
            assert torch.isfinite(result.score).all()
        np.testing.assert_allclose(grid.query_world(np.array([[1., 1.]])), min(.2, side_x, side_y))
    # A segment lying on an occupied cell edge must also count as contact.
    path = torch.tensor([[[0., 0.], [0., -.5]]])
    result = query.query(path, torch.tensor([0]), torch.tensor([[1., .25]]), torch.tensor([0.]), 3.6)
    assert result.minimum_clearance_m < 0
    assert torch.all(result.clearance_m[result.active] < 0)


def test_teacher_reuses_queried_cells_at_large_origin_boundary():
    clear = np.ones((20, 20), np.float32)
    clear[9, :] = -.05
    query = SourceConfigurationSpaceQuery((SourceConfigurationGrid(clear, np.array([1000.00003, -1000.00003]), .05),))
    batch = source_batch(1, 'cpu')
    batch['source_origin_xy'][:] = torch.tensor([[1000.75, -999.75]])
    batch['point_goal'][:] = torch.tensor([[.1, 0.]])
    paths = torch.tensor([[[[0., 0.], [-.25, 0.]], [[0., 0.], [.1, 0.]], [[0., 0.], [0., 0.]]]])
    result, _ = candidate_query(query, paths, batch, 3.6)
    cells = query.point_cells(paths.flatten(0, 1)[:, :1], batch['source_grid_index'].repeat_interleave(3),
                             batch['source_origin_xy'].repeat_interleave(3, 0), batch['source_yaw_rad'].repeat_interleave(3))
    torch.testing.assert_close(result.grid_cells[:, :1], cells)
    labels = RouteUtilityTeacher(query, 3.6)(paths, batch)
    assert torch.isfinite(labels.score).all()
    assert labels.clearance_m[0, 0] < 0
    assert labels.clearance_m[0, 1] > 0


def test_packed_scene_queries_match_separate_maps_with_different_shapes_and_scales():
    grids = (
        SourceConfigurationGrid(np.ones((20, 30), np.float32), np.array([-.5, -.7]), .05),
        SourceConfigurationGrid(np.ones((40, 25), np.float32), np.array([-1., -.8]), .1),
    )
    grids[0].signed_clearance_m[13, :] = -.05
    grids[1].signed_clearance_m[15, :] = -.1
    path = torch.tensor([[[0., 0.], [.6, .1]], [[0., 0.], [1.5, -.3]], [[0., 0.], [0., 0.]]])
    indices = torch.tensor([1, 0, 1])
    origins, yaws = torch.zeros(3, 2), torch.zeros(3)
    result = SourceConfigurationSpaceQuery(grids).query(path, indices, origins, yaws, 3.6)
    for row, index in enumerate(indices.tolist()):
        one = SourceConfigurationSpaceQuery((grids[index],)).query(
            path[row:row+1], torch.zeros(1, dtype=torch.long), origins[row:row+1], yaws[row:row+1], 3.6,
        )
        for name in ("local_path", "grid_cells", "clearance_m", "in_world_bounds"):
            torch.testing.assert_close(getattr(result, name)[row, result.active[row]],
                                       getattr(one, name)[0, one.active[0]], rtol=0, atol=0)
