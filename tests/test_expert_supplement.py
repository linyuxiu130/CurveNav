from curvenav.data.prepare_fixed40 import selected_states


def test_episode_partition_keeps_the_endpoint_and_every_captured_state():
    for partition in ('train','validation'):
        assert selected_states(23,partition)==[(i,partition) for i in range(23)]


def test_invalid_episode_partition_fails_before_creating_capture(tmp_path):
    import pytest
    from curvenav.data.prepare_fixed40 import collect_case
    folder=tmp_path/'capture'
    with pytest.raises(ValueError,match='partition'):
        collect_case(None,None,folder,partition='automatic')
    assert not folder.exists()


def test_real_near_stationary_planner_trial_keeps_original_quadrature_tolerance(monkeypatch):
    import numpy as np
    from types import SimpleNamespace
    from curvenav.data_generation import geometry as g
    controls=np.array([[14.192580176926473,8.156824221841372],
        [14.175051486291318,8.061812330450326],[14.172965671056508,7.959369075630475],
        [14.08870327981764,7.552581703988837],[13.667610724285334,7.174695118333863],
        [12.837393761651626,7.170635278180467],[13.136867188926926,7.172101563358507],
        [12.786161780924676,7.170378934417441]])
    path=np.array([controls[0],[13.97,7.3],controls[-1]])
    assert max(4,int(np.ceil(g.path_length(path)/(2*g.ROBOT_FOOTPRINT_RADIUS_M)))+1)==8
    handle=controls[1]-controls[0];yaw=np.arctan2(handle[1],handle[0])
    values=np.r_[np.log(np.linalg.norm(handle)),controls[2:-1].ravel()]
    def inspect_trial(objective,initial,**kwargs):
        assert initial.shape==values.shape
        value,gradient=objective(values)
        assert np.isfinite(value) and np.isfinite(gradient).all()
        return SimpleNamespace(success=True,x=values)
    monkeypatch.setattr(g,'minimize',inspect_trial)
    grid=g.Grid(np.ones((40,40),bool),np.full((40,40),.5),np.array([10.,5.]),.2,minimum_clearance_m=0.)
    curve,points=g._smooth(grid,path,yaw)
    np.testing.assert_allclose(curve([0,1]),controls[[0,-1]],atol=1e-12)
    assert grid.safe(points)
