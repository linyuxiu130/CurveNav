"""Omnidirectional geometric B-spline teachers; no heading or MPC-speed gate."""
import numpy as np
import torch
from curvenav.data import bspline_replay as replay
from curvenav.data.goal_distance import GoalDistanceQuery
from curvenav.data_generation.geometry import source_route,PlanningError
from curvenav.physical import EXTRA_CLEARANCE_M


def source_geometry_teacher(actor,view,pose,target_world,tree,error,boxes,source_query,grid,required_point=None,required_distance_m=None):
    try:plan=source_route(grid,view[:2],target_world,None)
    except PlanningError as problem:return None,dict(reason='source_route',detail=str(problem))
    route=plan.path_xy.astype(np.float64)
    if required_point is not None and np.linalg.norm(route-required_point,axis=1).min()>required_distance_m:
        return None,dict(reason='task_route_misses_required_waypoint')
    reference=replay.resample_route(route,horizon=actor.policy.planning_horizon_m)
    if reference is None:return None,dict(reason='zero_length_route')
    local=(reference-pose[:2,3])@pose[:2,:2];local[0]=0
    device=next(actor.policy.parameters()).device
    values,_=replay.fit_reference_tangent(actor.policy.curve_codec,torch.tensor(local,dtype=torch.float32,device=device)[None])
    dense,pad=replay.dense_curve(values);pad=float(pad[0]);dense=dense[0].cpu().numpy()
    world=dense@pose[:2,:2].T+pose[:2,3]
    observed=float(tree.query(dense,workers=1)[0].min()-error-actor.head.radius)
    physical=float(replay.point_box_clearance(world,np.asarray(boxes),actor.head.radius).min())
    horizon=max(actor.policy.planning_horizon_m,float(np.linalg.norm(np.diff(dense,axis=0),axis=-1).sum()))
    source=source_query.query(torch.from_numpy(dense)[None],torch.zeros(1,dtype=torch.long),
        torch.tensor(view[:2],dtype=torch.float64)[None],torch.tensor([view[2]],dtype=torch.float64),horizon)
    source_minimum=float(source.minimum_clearance_m[0])
    checks=dict(observed_hit_clear=observed>=pad,physical_margin=physical>=EXTRA_CLEARANCE_M+pad,
        source_margin=source_minimum>=EXTRA_CLEARANCE_M+pad)
    certificate=dict(kind='continuous_spline_geometry',qualified=all(checks.values()),observed_margin_m=observed,
        physical_margin_m=physical,source_margin_m=source_minimum,sampling_pad_m=pad)
    if not certificate['qualified']:return None,dict(reason='geometric_curve_qualification',checks=checks,certificate=certificate)
    return dict(teacher_values=values[0].cpu(),teacher_checks=checks,teacher_certificate=certificate),None


def source_utility_availability(goal_distance: GoalDistanceQuery, view, point_goal):
    """Use exactly RouteUtilityTeacher's fixed source cells before declaring Q available."""
    source_query=goal_distance.source
    goal=torch.as_tensor(point_goal,dtype=torch.float32)[None]
    cells=source_query.point_cells(torch.stack((torch.zeros_like(goal),goal),1),
        torch.zeros(1,dtype=torch.long),torch.as_tensor(view[:2],dtype=torch.float64)[None],
        torch.tensor([view[2]],dtype=torch.float64))[0].cpu().numpy()
    distances=goal_distance.query(0,cells[1],cells)
    available=bool(np.isfinite(distances).all())
    return available,dict(defined=available,start_cell=cells[0].tolist(),goal_cell=cells[1].tolist(),
        fixed_distances_m=[float(value) if np.isfinite(value) else None for value in distances],
        reason=None if available else 'source_start_or_goal_disconnected',
        semantics='RouteUtilityTeacher point_cells plus goal_distance.query; no snapping or source graph modification')
