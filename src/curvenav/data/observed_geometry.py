"""Offline geometry targets from causal depth-only ObservedTeacher state.

The caller updates the teacher with each actual depth and executed=False before
reading a target. These labels are supervision only, never policy inputs.
"""

from itertools import product
import math

import numpy as np
import torch
from torch.nn import functional as F

from curvenav.data.obstacle_memory import VOXEL_M

GEOMETRY_LABEL_CONTRACT = 'observed_hit_clearance_raw_spatial_gradient_v2'


def field_from_observed_hits(tree, point_error_m, observed_free, horizon, radius):
    """Use the same body-frame hits and conservative error subtraction as GF."""
    resolution=2*horizon/63
    axis=np.arange(64)*resolution-horizon
    x,y=np.meshgrid(axis,axis,indexing='xy')
    body=np.stack((x,y),-1)
    distance=tree.query(body.reshape(-1,2),workers=1)[0].reshape(64,64)
    clearance=torch.from_numpy((distance-point_error_m-radius).astype(np.float32))[None,None]
    padded=F.pad(clearance,(1,1,1,1),mode='replicate')
    dx=(padded[:,:,1:-1,2:]-padded[:,:,1:-1,:-2])/(2*resolution)
    dy=(padded[:,:,2:,1:-1]-padded[:,:,:-2,1:-1])/(2*resolution)
    overlap=(clearance<=0).float()
    free=torch.as_tensor(observed_free,dtype=torch.float32).reshape(1,1,64,64)*(1-overlap)
    return torch.cat((clearance,dx,dy,torch.maximum(free,overlap),overlap,free),1)[0]


def footprint_observed_per_point(teacher, points, radius):
    """Vectorize the teacher's exact disk/fully-observed-column predicate."""
    points=np.asarray(points,dtype=float).reshape(-1,2)
    reach=int(np.ceil(radius/teacher.grid))+1
    offsets=np.asarray(list(product(range(-reach,reach+1),repeat=2)))
    keys=np.floor(points[:,None]/teacher.grid).astype(int)+offsets
    distance=np.maximum(abs((keys+.5)*teacher.grid-points[:,None])-teacher.grid/2,0)
    overlap=np.linalg.norm(distance,axis=-1)<=radius
    needed,inverse=np.unique(keys[overlap],axis=0,return_inverse=True)
    known=np.fromiter((teacher.cells.get(tuple(key),0)==255 for key in needed),
        dtype=bool,count=len(needed))
    observed=np.ones(overlap.shape,dtype=bool)
    observed[overlap]=known[inverse]
    return observed.all(axis=1)


def target_from_observations(teacher, pose, k, camera, projector, tree, point_error_m):
    """Return the CPU supervision field[6,64,64], without mutation.

    ``pose`` is the current 4x4 body-to-world matrix. The teacher must contain
    this acquisition already. Unknown cells have support/free zero; their
    represented distance is not an observed-free-space certificate.
    """
    horizon=projector.planning_horizon_m
    radius=projector.radius
    geometry=(radius,projector.min_z,projector.max_z)
    if (not np.isclose(horizon,3.6,rtol=0,atol=1e-12)
            or not np.isclose(radius,.167584539,rtol=0,atol=1e-9)
            or not np.isclose(radius,teacher.radius,rtol=0,atol=1e-9)
            or projector.max_depth_m!=teacher.maximum
            or VOXEL_M!=.025
            or projector.configuration_grid_size!=64
            or projector.obstacle_padding!=4):
        raise ValueError('targets require the current 3.6m/64+8/body-height teacher calibration')
    if not np.array_equal(np.asarray(k),teacher.k) or not np.array_equal(np.asarray(camera),teacher.camera):
        raise ValueError('target calibration differs from the observed teacher history')
    pose=np.asarray(pose,dtype=float)
    if pose.shape!=(4,4) or not np.isfinite(pose).all():
        raise ValueError('pose must be a finite 4x4 body-to-world matrix')
    if not np.allclose(pose[2,:3],(0,0,1),rtol=0,atol=1e-12):
        raise ValueError('the teacher and projector body-height bands require planar ground poses')
    # Teacher columns use ground/world Z; the native projector uses base-link Z.
    # Their historical upper-bound constants differ only at FP32 precision.
    world_band=np.asarray(geometry[1:])+pose[2,3]
    band_rounding=np.finfo(np.float32).eps*max(abs(np.asarray(teacher.band)))
    if not np.allclose(world_band,teacher.band,rtol=0,atol=band_rounding):
        raise ValueError('teacher world-height columns differ from the native body-height band')
    if teacher.view is None or not np.array_equal(
            pose[:2,3].astype(np.float32),teacher.view[:2].astype(np.float32)):
        raise ValueError('teacher must include the current acquired pose before target construction')
    # The native writer computes sin/cos in Python, then stores the matrix FP32.
    c,s=math.cos(teacher.view[2]),math.sin(teacher.view[2])
    rotation=np.asarray(((c,-s),(s,c)),dtype=np.float32)
    if not np.array_equal(pose[:2,:2].astype(np.float32),rotation):
        raise ValueError('teacher heading differs from the current pose')

    resolution=2*horizon/63
    axis=np.arange(64)*resolution-horizon
    x,y=np.meshgrid(axis,axis,indexing='xy')
    body=np.stack((x,y),-1)
    world=body@pose[:2,:2].T+pose[:2,3]
    free=footprint_observed_per_point(teacher,world,radius)
    field=field_from_observed_hits(tree,point_error_m,free,horizon,radius)
    return field


def _minimum_squared(paths,delta,denominator,boxes):
    output=np.empty(len(paths),np.float64)
    for candidate in range(len(paths)):
        best=np.inf
        for segment in range(paths.shape[1]-1):
            ax,ay=paths[candidate,segment];bx,by=paths[candidate,segment+1]
            dx,dy=delta[candidate,segment]
            for box in range(len(boxes)):
                lx,ly,hx,hy=boxes[box];enter=0.;leave=1.;intersects=True
                if dx==0:
                    if ax<lx or ax>hx:intersects=False
                else:
                    a=(lx-ax)/dx;b=(hx-ax)/dx
                    enter=max(enter,min(a,b));leave=min(leave,max(a,b))
                if dy==0:
                    if ay<ly or ay>hy:intersects=False
                else:
                    a=(ly-ay)/dy;b=(hy-ay)/dy
                    enter=max(enter,min(a,b));leave=min(leave,max(a,b))
                if intersects and enter<=leave:
                    best=0.;break
                ex=max(lx-ax,ax-hx,0.);ey=max(ly-ay,ay-hy,0.)
                best=min(best,ex*ex+ey*ey)
                ex=max(lx-bx,bx-hx,0.);ey=max(ly-by,by-hy,0.)
                best=min(best,ex*ex+ey*ey)
                for corner in range(4):
                    cx=lx if corner==0 or corner==3 else hx
                    cy=ly if corner<2 else hy
                    fraction=((cx-ax)*dx+(cy-ay)*dy)/denominator[candidate,segment]
                    fraction=min(1.,max(0.,fraction))
                    ex=cx-(ax+fraction*dx);ey=cy-(ay+fraction*dy)
                    best=min(best,ex*ex+ey*ey)
            if best==0:break
        output[candidate]=best
    return output


def clearance(paths,boxes,*,radius):
    paths=np.asarray(paths);boxes=np.asarray(boxes)
    if paths.ndim!=3 or paths.shape[-1]!=2 or paths.shape[1]<2 or len(paths)==0:raise ValueError('paths must be nonempty [C,N>=2,2]')
    if boxes.ndim!=2 or boxes.shape[-1]!=4 or len(boxes)==0 or boxes.dtype!=np.float64:raise ValueError('training map boxes must be nonempty float64 [M,4]')
    # Keep the original broad-phase bound and path-dtype subtraction/reduction.
    center=paths[0,0];reach=np.linalg.norm(paths-center,axis=-1).max()
    upper=np.linalg.norm(np.maximum(np.maximum(boxes[None,:,:2]-paths[:,0,None],paths[:,0,None]-boxes[None,:,2:]),0),axis=-1).min(-1).max()
    distance=np.linalg.norm(np.maximum(np.maximum(boxes[:,:2]-center,center-boxes[:,2:]),0),axis=-1)
    nearby=boxes[distance<=reach+upper+1e-7]
    if len(nearby)==0:raise ValueError('invalid geometry produced an empty broad phase')
    delta=np.diff(paths,axis=1);denominator=np.maximum((delta**2).sum(-1),1e-20)
    return np.sqrt(_minimum_squared(paths,delta,denominator,nearby))-radius
