"""Independent geometry and end-to-end checks for the depth observation contract."""

from dataclasses import replace
import numpy as np
import pytest
import torch
from curvenav.config import (
    CurveNavConfig,
    DepthEncoderConfig,
    ConditionEncoderConfig,
    TrajectoryDecoderConfig,
)
from curvenav.data.depth import PinholeIntrinsics, preprocess_depth
from curvenav.data.history import ObservationHistory, inverse_rigid, validate_transform
from curvenav.data.observation import DepthContextBuffer
from curvenav.factory import build_policy, build_evaluation_projector
from curvenav.types import PolicyCondition, TrajectoryTarget


def config():
    return CurveNavConfig(
        depth_encoder=DepthEncoderConfig(
            model_dim=32, frame_tokens_height=2, frame_tokens_width=2
        ),
        condition_encoder=ConditionEncoderConfig(model_dim=32),
        trajectory_decoder=TrajectoryDecoderConfig(
            model_dim=32, transformer_layers=2, transformer_heads=4
        ),
    )


def condition(batch=1, height=126, width=224):
    k = torch.tensor([[100.0, 0, width / 2], [0, 100.0, height / 2], [0, 0, 1.0]])
    extrinsic = torch.tensor(
        [[0.0, 0, 1, 0], [-1.0, 0, 0, 0], [0, -1.0, 0, 0.6], [0, 0, 0, 1.0]]
    )
    return PolicyCondition(
        depth=torch.full((batch, 4, 1, height, width), 0.4),
        point_goal=torch.tensor([[3.0, 0.0]]).expand(batch, -1).clone(),
        camera_intrinsics=k.expand(batch, 4, 3, 3).clone(),
        camera_to_body=extrinsic.expand(batch, 4, 4, 4).clone(),
        observation_to_current=torch.eye(4).expand(batch, 4, 4, 4).clone(),
        observation_age_s=torch.tensor([[1.6, 0.9, 0.1, 0.0]])
        .expand(batch, -1)
        .clone(),
        observation_valid=torch.ones(batch, 4, dtype=torch.bool),
    )


def test_resize_preserves_rays_and_missing_depth():
    k = PinholeIntrinsics(8, 6, 5, 7, 3.2, 2.7)
    raw = np.full((6, 8), np.nan, np.float32)
    depth, resized = preprocess_depth(
        raw, source_intrinsics=k, maximum_m=5, height=3, width=4
    )
    assert depth.shape == (3, 4) and np.all(depth == 0)
    # Continuous half-pixel correspondence, independent of nearest discretization.
    for u, v in [(0.0, 0.0), (3.0, 2.0)]:
        np.testing.assert_allclose(
            [
                (u + 0.5 - resized[0, 2]) / resized[0, 0],
                (v + 0.5 - resized[1, 2]) / resized[1, 1],
            ],
            [
                (((u + 0.5) * 2) - k.cx) / k.fx,
                (((v + 0.5) * 2) - k.cy) / k.fy,
            ],
            atol=1e-7,
        )


def test_fp16_depth_encoding_preserves_unknown_hits_and_range_limits():
    raw = np.array([[np.nan, np.inf, -np.inf, -1, 0, 1e-10, 4.9999, 5, 8]], np.float32)
    depth, _ = preprocess_depth(
        raw,
        source_intrinsics=PinholeIntrinsics(9, 1, 5, 5, 4, 0),
        maximum_m=5,
        height=1,
        width=9,
    )
    stored = depth.astype(np.float16).astype(np.float32)
    assert np.all(stored[0, :5] == 0)
    assert np.all((stored[0, 5:7] > 0) & (stored[0, 5:7] < 1))
    assert np.all(stored[0, 7:] == 1)


def test_history_fixed_slots_padding_and_irregular_clock():
    history = ObservationHistory()
    pose = np.eye(4)
    times = np.cumsum(np.arange(1, 21, dtype=np.float64)) / 10
    for i, time in enumerate(times):
        indices, relative, age, valid = history.update(i, pose, time)
        assert indices.shape == (4,) and valid[-1]
        assert np.all(indices <= i)
        if i == 0:
            assert valid.sum() == 1
        if i == 1:
            np.testing.assert_array_equal(valid, [False] * 2 + [True, True])
    expected = np.array([3, 10, 18, 19])
    np.testing.assert_array_equal(indices, expected)
    np.testing.assert_allclose(age, times[-1] - times[expected], atol=1e-6)
    np.testing.assert_allclose(relative, np.broadcast_to(np.eye(4), (4, 4, 4)))


def test_static_wall_aligns_across_four_calibrated_views():
    c = condition(height=16, width=24)
    c.camera_intrinsics[..., 0, 0] = 30
    c.camera_intrinsics[..., 1, 1] = 1000
    from curvenav.physical import BODY_OBSTACLE_MIN_Z_M, ROBOT_COLLISION_TOP_Z_M

    c.camera_to_body[..., 2, 3] = (BODY_OBSTACLE_MIN_Z_M + ROBOT_COLLISION_TOP_Z_M) / 2
    # Render optical Z analytically for the same plane x_current = 2 m.
    for frame in range(4):
        angle = (3 - frame) * 0.025
        c.observation_to_current[0, frame, :2, :2] = torch.tensor(
            [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
        )
        c.observation_to_current[0, frame, 0, 3] = -(3 - frame) * 0.05
        transform = c.observation_to_current[0, frame] @ c.camera_to_body[0, frame]
        y, x = torch.meshgrid(torch.arange(16), torch.arange(24), indexing="ij")
        rays = torch.stack(
            ((x + 0.5 - 12) / 30, (y + 0.5 - 8) / 1000, torch.ones_like(x)), -1
        )
        directions = rays @ transform[:3, :3].T
        c.depth[0, frame, 0] = (2 - transform[0, 3]) / directions[..., 0] / 5
    projection = build_evaluation_projector(config())(c)
    assert projection.token_valid.all()
    torch.testing.assert_close(
        projection.points[..., 0],
        torch.full_like(projection.points[..., 0], 2),
        atol=1e-6,
        rtol=0,
    )
    assert projection.obstacle_valid.all()


def test_bev_distance_and_out_of_bounds_match_metric_grid():
    from scipy.ndimage import distance_transform_edt

    projector = build_evaluation_projector(config())
    mask = torch.zeros(1, 1, 64, 64, dtype=torch.bool)
    mask[0, 0, 12, 37] = True
    expected = distance_transform_edt(~mask[0, 0].numpy()) * (7.2 / 63)
    np.testing.assert_allclose(
        projector._euclidean_distance_transform(mask)[0], expected, atol=1e-6
    )
    # A point outside the metric domain must not round back onto the border.
    raster = projector._rasterize(torch.tensor([[[3.61, 0.0]]]), torch.tensor([[True]]))
    assert not raster.any()


def test_raster_clearance_is_a_lower_bound_on_continuous_point_clearance():
    from curvenav.physical import ROBOT_FOOTPRINT_RADIUS_M

    projector = build_evaluation_projector(config())
    # Off-grid point close to a cell corner exercises both coordinate errors.
    point = torch.tensor([0.41, 0.29, 0.06])
    field = projector._configuration_field(
        point.reshape(1, 1, 1, 1, 3),
        torch.ones(1, 1, 1, 1, dtype=torch.bool),
        point.reshape(1, 1, 1, 3),
        torch.ones(1, 1, 1, dtype=torch.bool),
        torch.eye(4).reshape(1, 1, 4, 4),
        torch.ones(1, 1, dtype=torch.bool),
    )
    axis = torch.linspace(-3.6, 3.6, 64)
    y, x = torch.meshgrid(axis, axis, indexing="ij")
    exact = (
        (x - point[0]).square() + (y - point[1]).square()
    ).sqrt() - ROBOT_FOOTPRINT_RADIUS_M
    assert torch.all(field[0, 0] <= exact + 1e-6)
    # Without the cell error bound the old calculation overestimates clearance.
    assert torch.any(
        field[0, 0] + projector.configuration_resolution_m / np.sqrt(2) > exact + 1e-4
    )


def test_occluded_history_is_retained_and_padding_cannot_add_geometry():
    c = condition()
    c.depth[:, 0] = 0.6  # Behind the newer 2 m surface: absence is unobservable.
    projector = build_evaluation_projector(config())
    assert projector(c).token_valid[:, 0].all()
    c.observation_valid[:, :-1] = False
    first = projector(c)
    c.depth[:, :-1] = 0.02
    second = projector(c)
    torch.testing.assert_close(first.configuration_field, second.configuration_field)
    assert not second.token_valid[:, :-1].any()


def test_bf16_multiframe_bev_fusion_matches_fp32_weighted_mean():
    encoder = build_policy(config()).condition_encoder.configuration_encoder
    encoder.visual_projection = torch.nn.Identity()
    values = torch.linspace(-1, 1, 768).to(torch.bfloat16)
    tokens = values.reshape(1, 768, 1).expand(-1, -1, 32).clone().requires_grad_()
    points = encoder.metric_position[:, 37:38].expand(-1, 768, -1)
    result = encoder._splat_visual(tokens, points, torch.ones(1, 768, dtype=torch.bool))
    assert result.dtype == torch.float32
    torch.testing.assert_close(
        result[0, 37], values.float().mean().expand(32), atol=1e-6, rtol=0
    )
    result.sum().backward()
    assert torch.isfinite(tokens.grad).all()


def test_all_missing_is_unknown_and_model_finite():
    c = condition()
    c.depth.zero_()
    projection = build_evaluation_projector(config())(c)
    assert not projection.token_valid.any()
    assert projection.configuration_field[:, 3:].count_nonzero() == 0
    path = build_policy(config()).eval().sample(c).path
    assert torch.isfinite(path).all()


def test_optical_z_intrinsics_extrinsics_and_pose_composition():
    c = condition(height=4, width=4)
    c.depth.zero_()
    c.depth[0, 0, 0, 2, 2] = 0.4
    c.observation_valid[:, 1:] = False
    c.observation_to_current[0, 0, :3, 3] = torch.tensor([1.0, 2.0, 0.3])
    c.observation_to_current[0, 0, :3, :3] = torch.tensor(
        [[0.0, -1, 0], [1, 0, 0], [0, 0, 1.0]]
    )
    projection = build_evaluation_projector(config())(c)
    points = projection.points[projection.token_valid]
    # Pixel centre [2.5,2.5] -> optical [.01,.01,2] -> current [1.01,4,.89].
    torch.testing.assert_close(points, torch.tensor([[1.01, 4.0, 0.89]]))
    c.depth *= 0.5
    point = build_evaluation_projector(config())(c).points[projection.token_valid]
    torch.testing.assert_close(point, torch.tensor([[1.005, 3.0, 0.895]]))


def test_projection_matches_independent_homogeneous_reference_with_camera_tilt():
    from scipy.spatial.transform import Rotation

    c = condition(height=4, width=4)
    c.depth.zero_()
    c.depth[0, 0, 0, 1, 3] = 0.4
    world_past, world_current, body_camera = np.tile(np.eye(4), (3, 1, 1))
    world_past[:3, :3] = Rotation.from_euler("z", 0.7).as_matrix()
    world_current[:3, :3] = Rotation.from_euler("z", -0.3).as_matrix()
    body_camera[:3, :3] = Rotation.from_euler("xyz", [0.2, -0.4, 0.6]).as_matrix()
    world_past[:3, 3], world_current[:3, 3] = [2, -1, 0.3], [-1, 3, 0.2]
    body_camera[:3, 3] = [0.3, -0.1, 0.6]
    c.observation_to_current[0, 0] = torch.from_numpy(
        np.linalg.inv(world_current) @ world_past
    )
    c.camera_to_body[0, 0] = torch.from_numpy(body_camera)
    optical = 2 * np.linalg.solve(c.camera_intrinsics[0, 0].numpy(), [3.5, 1.5, 1])
    expected = (
        np.linalg.inv(world_current) @ world_past @ body_camera @ np.r_[optical, 1]
    )[:3]
    projection = build_evaluation_projector(config())(c)
    np.testing.assert_allclose(
        projection.points[projection.token_valid], expected[None], atol=1e-6
    )


def test_rigid_inverse_rejects_scale_and_reflection():
    t = np.eye(4)
    t[:3, :3] = [[0, -1, 0], [1, 0, 0], [0, 0, 1]]
    t[:3, 3] = [1, 2, 3]
    validate_transform(t)
    np.testing.assert_allclose(inverse_rigid(t) @ t, np.eye(4))
    for bad in [np.diag([2.0, 1, 1, 1]), np.diag([-1.0, 1, 1, 1])]:
        with pytest.raises(ValueError):
            validate_transform(bad)


def test_history_sliding_rotation_timestamps_and_runtime_parity():
    cfg = config()
    history = ObservationHistory()
    buffer = DepthContextBuffer(cfg.data)
    buffer.reset(1)
    c = condition()
    depth = np.ones((1, 126, 224, 1), np.float32)
    pose = np.eye(4, dtype=np.float32)
    for i in range(200):
        angle = i * 0.03
        pose[:2, :2] = [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
        indices, relative, age, valid = history.update(i, pose, i * 0.1)
        result = buffer.update(
            depth,
            pose[None],
            c.camera_intrinsics[:, 0].numpy(),
            c.camera_to_body[:, 0].numpy(),
            np.array([i * 0.1]),
        )
        np.testing.assert_allclose(
            result["observation_to_current"][0], relative, atol=1e-6
        )
        np.testing.assert_array_equal(result["observation_valid"][0], valid)
        assert len(buffer.frames[0]) <= 16
    assert valid.sum() == 4
    np.testing.assert_array_equal(indices, [183, 190, 198, 199])
    np.testing.assert_allclose(age, [1.6, 0.9, 0.1, 0], atol=1e-6)
    _, _, age, valid = history.update(201, pose, 30.0)
    assert valid.sum() == 4 and age.max() > 8
    with pytest.raises(ValueError):
        history.update(202, pose, 30.0)


def test_bev_bin_centres_match_convolution_and_pooling():
    enc = build_policy(config()).condition_encoder.configuration_encoder
    axis = enc.metric_position[0, :16, 0]
    expected = (torch.arange(16) * 4 + 1.5) * 7.2 / 63 - 3.6
    torch.testing.assert_close(axis, expected)
    # Two kernel-2/stride-2 receptive fields aggregate indices 4j..4j+3.
    assert enc.field_encoder[0].kernel_size == (2, 2)
    assert enc.field_encoder[2].kernel_size == (2, 2)
    tokens = torch.zeros(1, 1, 32)
    tokens[0, 0, 0] = 1
    enc.visual_projection = torch.nn.Identity()
    splat = enc._splat_visual(
        tokens, enc.metric_position[:, 37:38], torch.ones(1, 1, dtype=torch.bool)
    )
    assert splat[0, 37, 0] > 0.99


def test_depth_flow_backward_and_strict_state_roundtrip(tmp_path):
    torch.manual_seed(42)
    cfg = config()
    p = build_policy(cfg)
    c = condition(4)
    target = TrajectoryTarget(p.curve_codec.values_from_coordinates(torch.randn(4, 14)))
    loss = p(c, target, torch.randn(4, 14), torch.arange(4, dtype=torch.uint8)).loss
    assert torch.isfinite(loss)
    loss.backward()
    assert p.depth_encoder.backbone[0].weight.grad[:, :3].abs().sum() > 0
    assert all(
        torch.isfinite(x.grad).all() for x in p.parameters() if x.grad is not None
    )
    p.eval()
    a = p.sample(c).path
    torch.save(p.state_dict(), tmp_path / "state.pt")
    q = build_policy(cfg).eval()
    q.load_state_dict(torch.load(tmp_path / "state.pt", weights_only=True), strict=True)
    torch.testing.assert_close(a, q.sample(c).path, rtol=0, atol=0)
    changed = replace(c, depth=c.depth * .75)
    assert not torch.equal(
        p.encode_condition(c).tokens, p.encode_condition(changed).tokens
    )


def test_new_free_space_removes_contradicted_history_but_unknown_does_not():
    c = condition()
    c.depth[:, 0] = 0.2
    projector = build_evaluation_projector(config())
    result = projector(c)
    assert not result.token_valid[:, 0].any()
    c.depth[:, -1] = 0
    assert not projector(c).token_valid[:, 0].any()
    c.depth[:, 1:] = 0
    assert projector(c).token_valid[:, 0].all()


def test_world_coordinate_gauge_does_not_change_relative_memory():
    cfg = config()
    a = ObservationHistory()
    b = ObservationHistory()
    gauge = np.eye(4)
    gauge[:2, :2] = [[0, -1], [1, 0]]
    gauge[:3, 3] = [4, -7, 2]
    for i in range(4):
        pose = np.eye(4)
        pose[:3, 3] = [i * 0.5, 0, 0]
        ia, ta, agea, va = a.update(i, pose, i * 0.5)
        ib, tb, ageb, vb = b.update(i, gauge @ pose, i * 0.5)
        np.testing.assert_array_equal(ia, ib)
        np.testing.assert_allclose(ta, tb, atol=1e-6)
        np.testing.assert_array_equal(agea, ageb)
        np.testing.assert_array_equal(va, vb)


def test_depth_encoder_compiles_as_one_cpu_autograd_graph():
    encoder = build_policy(config()).depth_encoder
    c = condition()
    eager = encoder(c)
    compiled = torch.compile(encoder, backend="aot_eager", fullgraph=True)
    actual = compiled(c)
    torch.testing.assert_close(actual.tokens, eager.tokens)
    actual.tokens.square().mean().backward()
    assert encoder.backbone[0].weight.grad is not None
    assert torch.isfinite(encoder.backbone[0].weight.grad).all()


def test_continuous_clearance_is_a_lipschitz_lower_bound():
    from curvenav.configuration_space import query_configuration_field

    # Obstacle at cell centre: averaging the four node distances would invent
    # positive clearance exactly at the obstacle.
    field = torch.zeros(1, 5, 2, 2)
    field[:, 0] = 2**0.5
    field[:, 3] = 1
    torch.manual_seed(91)
    points = torch.cat(
        (torch.zeros(1, 1, 2), torch.rand(1, 100, 2) * 2 - 1), dim=1
    ).requires_grad_()
    query = query_configuration_field(field, points, 1.0)
    truth = points.norm(dim=-1)
    assert (query.signed_clearance_m <= truth + 1e-6).all()
    assert abs(query.signed_clearance_m[0, 0]) < 1e-6
    query.signed_clearance_m.sum().backward()
    assert torch.isfinite(points.grad).all()


def test_horizontal_platform_in_robot_body_band_remains_an_obstacle():
    c = condition(height=16, width=24)
    c.observation_valid[:, :-1] = False
    # Optical +Z points downward onto a horizontal platform 8 cm above
    # the base link. Its zero slope does not make it traversable.
    c.camera_to_body[:, -1, :3, :3] = torch.diag(torch.tensor([1.0, -1.0, -1.0]))
    c.camera_to_body[:, -1, :3, 3] = torch.tensor([1.0, 0.0, 0.6])
    c.depth[:] = (0.6 - 0.08) / 5
    projection = build_evaluation_projector(config())(c)
    assert projection.obstacle_valid[:, -1].any()


def test_depth_pixel_feature_sampling_uses_backbone_stride_not_pool_bins():
    from curvenav.encoders.depth import DepthObservationEncoder
    y, x = torch.meshgrid(torch.arange(8.), torch.arange(14.), indexing="ij")
    features = torch.stack((16*x, 16*y))[None].requires_grad_()
    # Includes fractional feature locations and the documented border extension.
    pixels = torch.tensor([[[0, 3*224+7, 63*224+112, 125*224+223]]])
    actual = DepthObservationEncoder.sample_pixel_features(features, pixels, 224)
    expected = torch.stack(((pixels % 224).clamp_max(208),
                            (pixels // 224).clamp_max(112)), dim=-1).float()
    torch.testing.assert_close(actual, expected, atol=2e-5, rtol=0)
    actual.sum().backward()
    assert features.grad is not None and torch.isfinite(features.grad).all()


def test_intermediate_free_space_prevents_old_obstacle_returning():
    c = condition()
    c.depth[:, 0] = .2
    c.depth[:, -1] = 0
    projector = build_evaluation_projector(config())
    result = projector(c)
    assert not result.obstacle_valid[:, 0].any()
    without_old = replace(c, observation_valid=c.observation_valid.clone())
    without_old.observation_valid[:, 0] = False
    torch.testing.assert_close(result.configuration_field, projector(without_old).configuration_field)
    # A masked newer frame cannot clear history.
    c.observation_valid[:, 1:3] = False
    assert projector(c).obstacle_valid[:, 0].any()


def test_selected_pixel_ray_and_metric_point_are_the_same_measurement():
    from curvenav.encoders.geometry import MetricDepthProjector
    c = condition()
    torch.manual_seed(82)
    c.depth = .1 + .6 * torch.rand_like(c.depth)
    c.observation_valid[:, :-1] = False
    result = MetricDepthProjector(8, 12, 5, 3.6)(c)
    indices = result.pixel_indices[:, -1]
    pixel = torch.stack(((indices % 224).float()+.5,
                         (indices // 224).float()+.5, torch.ones_like(indices).float()), dim=-1)
    depth = c.depth[:, -1].flatten(1).gather(1, indices) * 5
    camera = (torch.linalg.inv(c.camera_intrinsics[:, -1]) @ pixel.transpose(1,2)).transpose(1,2) * depth[...,None]
    homogeneous = torch.cat((camera, torch.ones_like(camera[...,:1])),dim=-1)
    body = (c.camera_to_body[:, -1] @ homogeneous.transpose(1,2)).transpose(1,2)[...,:3]
    torch.testing.assert_close(result.points[:, -1],body,atol=1e-6,rtol=1e-6)


def test_neural_regions_leave_metric_outputs_and_master_parameters_float32():
    policy = build_policy(config())
    c = condition()
    features = policy.depth_encoder(c)
    assert features.tokens.dtype == torch.bfloat16
    assert features.configuration_field.dtype == torch.float32
    prediction = policy.sample(c)
    assert prediction.path.dtype == torch.float32
    assert all(p.dtype == torch.float32 for p in policy.parameters())


def test_obstacle_outside_bev_still_inflates_into_boundary():
    projector = build_evaluation_projector(config())
    point = torch.tensor([3.65, 0., .06])
    field = projector._configuration_field(
        point.reshape(1, 1, 1, 1, 3), torch.ones(1, 1, 1, 1, dtype=torch.bool),
        point.reshape(1, 1, 1, 3), torch.ones(1, 1, 1, dtype=torch.bool),
        torch.eye(4).reshape(1, 1, 4, 4), torch.ones(1, 1, dtype=torch.bool),
    )
    assert field.shape == (1, 5, 64, 64)
    assert field[0, 0, 31:33, -1].max() < 0
    assert field[0, 3:, 31:33, -1].all()
