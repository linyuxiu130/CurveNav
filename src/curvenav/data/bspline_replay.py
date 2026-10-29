"""Observed-route B-spline labels and native joint losses; never timed-control labels."""
from curvenav.experiment_io import sha
from functools import lru_cache

import numpy as np
import torch

from curvenav.trajectory.basis import bspline_basis_and_first_derivative, _open_uniform_knots
from curvenav.data.privileged import SourceConfigurationSpaceQuery
from curvenav.data.observed_geometry import clearance

SAMPLES = 257


@lru_cache(maxsize=1)
def dense_basis():
    basis, _ = bspline_basis_and_first_derivative(8, 3, SAMPLES)
    knots = _open_uniform_knots(8, 3)
    return basis, 3/(knots[4:11]-knots[1:8])


def dense_curve(values):
    """Exact fixed cubic basis; nearest-sample distance bounded by sup|C'|/(2N)."""
    controls = torch.cat((values.new_zeros(len(values),1,2), values.float().reshape(-1,7,2)), 1)
    basis, factors = dense_basis()
    path = torch.einsum('pc,bcd->bpd', basis.to(values.device), controls)
    derivative_controls = torch.diff(controls,dim=1)*factors.to(values.device)[None,:,None]
    pad = derivative_controls.norm(dim=-1).amax(-1)/(2*(SAMPLES-1))
    return path, pad


from curvenav.trajectory.control_points import project_reference_tangent as fit_reference_tangent


def resample_route(route, count=64, horizon=3.6):
    lengths = np.linalg.norm(np.diff(route,axis=0),axis=1)
    route = route[np.r_[True,lengths>1e-8]]
    if len(route)<2:return None
    arc = np.r_[0.,np.linalg.norm(np.diff(route,axis=0),axis=1).cumsum()]
    at = np.linspace(0,min(horizon,arc[-1]),count)
    return np.column_stack([np.interp(at,arc,route[:,i]) for i in range(2)])


def point_box_clearance(points, boxes, radius):
    points = np.asarray(points,dtype=np.float64)
    flat = points.reshape(-1,2);out=[]
    for start in range(0,len(flat),4096):
        delta = np.maximum(np.maximum(boxes[None,:,:2]-flat[start:start+4096,None],
            flat[start:start+4096,None]-boxes[None,:,2:]),0)
        out.append(np.linalg.norm(delta,axis=-1).min(-1)-radius)
    return np.concatenate(out).reshape(points.shape[:-1])


def source_grid(case, destination, radius):
    """Source oracle in explicit canonical world XY, matching saved pose/view geometry."""
    boxes=np.asarray(case['boxes'],dtype=np.float64)
    bounds=np.asarray(case.get('floor_bounds',np.r_[boxes[:,:2].min(0),boxes[:,2:].max(0)]))
    spacing=.05;origin=bounds[:2]-.2;shape=np.ceil((bounds[2:]+.2-origin)/spacing).astype(int)
    x,y=np.indices(shape);points=origin+np.stack((x+.5,y+.5),-1)*spacing
    clearance=point_box_clearance(points,boxes,radius);free=clearance>=0
    np.savez_compressed(destination,free=free,clearance_m=np.where(free,clearance,0).astype(np.float32),
        origin_xy=origin,cell_size_m=spacing,coordinate_frame="canonical_world_xy")
    return SourceConfigurationSpaceQuery.from_paths((destination,))


def observed_risk(values, tree, error, radius):
    paths,pad=dense_curve(values)
    flat=paths.reshape(-1,2).double()
    _,nearest=tree.query(flat.detach().cpu().numpy(),workers=8)
    sites=torch.as_tensor(tree.data[nearest],device=flat.device)
    margin=(flat-sites).norm(dim=-1).reshape(len(values),SAMPLES)-error-radius
    deficit=((pad[:,None]-margin)/radius).clamp_min(0)
    risk=4*torch.where(deficit<=1,deficit.square(),2*deficit-1).amax(-1)
    return risk.mean(),dict(per_source=risk,minimum=margin.amin(-1),pad=pad)


@torch.no_grad()
def require_native_geometry(device):
    if torch.device(device).type != 'cuda':
        raise ValueError('Geometry preparation requires CUDA native projection; CPU is not runtime-equivalent')
    torch.use_deterministic_algorithms(True, warn_only=True)
