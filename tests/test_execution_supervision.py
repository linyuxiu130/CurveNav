import torch
import numpy as np
from types import SimpleNamespace
from curvenav.data.observed_bspline_routes import ObservedBSplineRoutes
from curvenav.data.fixed40_targets import task_failure_mask,task_state_from_history
from test_critic import oracle,source_batch
from curvenav.training.critic import RouteUtilityTeacher


def test_no_goal_utility_uses_causal_coverage_not_hidden_exit():
    batch=source_batch(1,'cpu');batch['goal_present'][:]=False
    t=torch.linspace(0,2,64);paths=torch.stack((torch.stack((t*0,t),-1),torch.stack((t*0,-t),-1)))[None]
    teacher=RouteUtilityTeacher(oracle(),3.6)
    batch['observed_field'][:,3,32:]=0
    scores=teacher(paths,batch).score
    assert scores[0,0]>scores[0,1]
    batch['observed_field'][:,3]=1-batch['observed_field'][:,3]
    assert teacher(paths,batch).score[0,0]<teacher(paths,batch).score[0,1]




def test_outside_coverage_lattice_does_not_create_a_novelty_bonus():
    batch=source_batch(1,'cpu');batch['goal_present'][:]=False
    t=torch.linspace(0,6,64);paths=torch.stack((t*0,t),-1)[None,None]
    labels=RouteUtilityTeacher(oracle(),3.6)(paths,batch)
    # All available coverage is known. Padding outside the local label field is not unseen evidence.
    torch.testing.assert_close(labels.score,(labels.progress_m-.5*torch.tensor([[6.]]))/3.6)
    assert labels.progress_m.item()<6.1


def test_exploration_teacher_does_not_reselect_its_already_reached_frontier_approach():
    free=np.zeros((60,60),bool);free[20:40,10:50]=True
    blocked=np.ones_like(free);blocked[20:40]=False
    teacher=SimpleNamespace(grid=.05,radius=.167584539,margin=.1,waypoint_tolerance=.005,forecast_guard=.0075,
        _raster=lambda view:(free,blocked,np.zeros(2,dtype=int)),footprint_observed=lambda points:True)
    start=np.array([(45+.5)*.05,(30+.5)*.05,0.])
    source=SimpleNamespace(source=SimpleNamespace(grids=[SimpleNamespace(origin_xy=np.zeros(2),cell_size_m=.05)]),
        query=lambda index,origin,cells:np.linalg.norm(cells-origin,axis=-1)*.05)
    target,detail=ObservedBSplineRoutes.source_frontier_target(teacher,start,source)
    assert detail['reason']=='source_connected_frontier'
    assert target[0]<start[0]-1.


def test_q_cannot_reward_reentry_into_a_causally_confirmed_closed_branch():
    batch=source_batch(1,'cpu');batch['goal_present'][:]=False
    batch['task_states']=[dict(panel='K',case=dict(start=[0.,0.,0.],hall=1.,
        known_complete_branch='closed',branch_routes={'closed':[[[0.,0.],[0.,-3.]]] }))]
    paths=torch.tensor([[[[0.,0.],[0.,-3.]],[[0.,0.],[0.,3.]]]])
    labels=RouteUtilityTeacher(oracle(),3.6)(paths,batch)
    assert labels.score[0,0]<labels.score[0,1]
    assert labels.progress_m[0,0]<1.


def test_task_labels_allow_return_before_fork_but_reject_subsequent_reentry():
    points=np.array([[[[0.,2.],[0.,1.],[0.,0.],[0.,-1.],[0.,0.],[0.,1.]]]])
    case=dict(hall=1.,deadend_arms=[dict(fork=[0.,0.],route=[[0.,0.],[0.,3.]])])
    d2=dict(panel='D2',case=case,arm=0,confirmed_arms=[0],fork_seen=False)
    expected=np.array([[[False,False,False,False,False,True]]])
    np.testing.assert_array_equal(task_failure_mask(points,[d2]),expected)
    leg=dict(panel='L',case=dict(hall=1.,known_direction=[0.,1.],other_direction=[1.,0.]),fork_seen=False)
    np.testing.assert_array_equal(task_failure_mask(points,[leg]),expected)
    assert not task_failure_mask(points,[dict(panel='J',case={})]).any()


def test_known_branch_q_labels_match_existing_evaluation_geometry():
    from curvenav.evaluation.bspline40 import in_closed_branch,branch_at_d2
    points=np.random.default_rng(7).uniform(-3,3,(1,2,100,2))
    case=dict(start=[0.,0.],hall=1.,known_complete_branch='closed',
        branch_routes={'closed':[[[0.,0.],[0.,-2.]],[[0.,-2.],[2.,-2.]]]})
    actual=task_failure_mask(points,[dict(panel='K',case=case)])[0]
    expected=np.array([[in_closed_branch(p,case) for p in path] for path in points[0]])
    np.testing.assert_array_equal(actual,expected)
    case=dict(hall=1.,deadend_arms=[dict(fork=[0.,0.],route=[[0.,0.],[0.,-2.]])])
    actual=task_failure_mask(points,[dict(panel='D2',case=case,arm=0,confirmed_arms=[0],fork_seen=True)])[0]
    expected=np.array([[branch_at_d2(p,case,[0]) is not None for p in path] for path in points[0]])
    np.testing.assert_array_equal(actual,expected)


def test_task_state_uses_no_future_execution_to_unlock_fork_state():
    raw=dict(task=dict(panel='D2',arm=0,stage='at_end'),case=dict(prefix_views=[[0.,2.,0.]],
        deadend_arms=[dict(fork=[0.,0.])]),task_evidence=dict(initially_confirmed_arms=[0]),
        acquired_views=np.array([[0.,2.,0.],[0.,1.,0.],[0.,0.,0.],[0.,2.,0.]]))
    assert not task_state_from_history(raw,2)['fork_seen']
    assert task_state_from_history(raw,4)['fork_seen']
