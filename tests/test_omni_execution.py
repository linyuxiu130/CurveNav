from types import SimpleNamespace
import numpy as np
import pytest
from curvenav.evaluation.bspline_execution import advance_path


def environment(boxes=None):
    env=SimpleNamespace(view=np.zeros(3),path=0.,clock=0.,case=dict(boxes=[[10.,10.,11.,11.]] if boxes is None else boxes))
    def write(view):env.view=np.asarray(view).copy();env.clock+=.5
    env.write=write
    return env


def test_omni_lateral_reverse_and_corner_motion_hold_yaw_without_curvature_limit():
    for curve,expected in [([[0,0],[0,1]],[0,.25]),([[0,0],[-1,0]],[-.25,0]),([[0,0],[0,.1],[.2,.1]],[.15,.1])]:
        env=environment();result=advance_path(env,np.array(curve))
        assert result['status']=='running'
        np.testing.assert_allclose(env.view[:2],expected,atol=1e-12)
        assert env.view[2]==0 and env.path==pytest.approx(.25) and env.clock==.5


def test_recentered_remaining_reference_progresses_instead_of_restarting():
    env=environment();path=np.array([[0.,0.],[0.,1.]])
    advance_path(env,path)
    advance_path(env,path-env.view[:2])
    np.testing.assert_allclose(env.view,[0.,.5,0.]);assert env.path==.5
    advance_path(env,np.zeros((64,2)))
    np.testing.assert_allclose(env.view,[0.,.5,0.])


def test_collision_is_reported_without_filtering_or_repairing_the_candidate():
    env=environment([[.3,-1.,.4,1.]])
    path=np.array([[0.,0.],[1.,0.]]);original=path.copy()
    result=advance_path(env,path)
    assert result['status']=='collision' and result['margin_m']<0
    np.testing.assert_array_equal(env.view,np.zeros(3));np.testing.assert_array_equal(path,original)
