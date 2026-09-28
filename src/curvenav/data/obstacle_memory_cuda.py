"""CUDA replay of the causal voxel memory used by the dataset compiler."""
import math
import numpy as np
import torch
from numba import cuda
from curvenav.data.obstacle_memory import ObstacleMemory, VOXEL_M

_SHIFT = 1048576
_BASE = 2097152


@cuda.jit(device=True)
def _decode(key):
    z = key % _BASE - _SHIFT
    key //= _BASE
    y = key % _BASE - _SHIFT
    x = key // _BASE - _SHIFT
    return x, y, z


@cuda.jit(device=True)
def _encode(x, y, z):
    return ((x + _SHIFT) * _BASE + y + _SHIFT) * _BASE + z + _SHIFT


@cuda.jit(cache=True)
def _retain(keys, out, depth, body, camera, intrinsic, extent, maximum, voxel):
    i = cuda.grid(1)
    if i >= keys.size:
        return
    vx, vy, vz = _decode(keys[i])
    lx0, ly0, lx1, ly1 = math.inf, math.inf, -math.inf, -math.inf
    u0, v0, u1, v1, zmax = math.inf, math.inf, -math.inf, -math.inf, -math.inf
    inside = True
    h, w = depth.shape
    for c in range(8):
        wx = (vx + (c // 4)) * voxel
        wy = (vy + ((c // 2) % 2)) * voxel
        wz = (vz + (c % 2)) * voxel
        dx, dy, dz = wx-body[0,3], wy-body[1,3], wz-body[2,3]
        lx = dx*body[0,0] + dy*body[1,0] + dz*body[2,0]
        ly = dx*body[0,1] + dy*body[1,1] + dz*body[2,1]
        lx0, ly0, lx1, ly1 = min(lx0,lx), min(ly0,ly), max(lx1,lx), max(ly1,ly)
        dx, dy, dz = wx-camera[0,3], wy-camera[1,3], wz-camera[2,3]
        x = dx*camera[0,0] + dy*camera[1,0] + dz*camera[2,0]
        y = dx*camera[0,1] + dy*camera[1,1] + dz*camera[2,1]
        z = dx*camera[0,2] + dy*camera[1,2] + dz*camera[2,2]
        positive = z if z > 0 else 1.0
        u = intrinsic[0,0]*x/positive + intrinsic[0,2] - .5
        v = intrinsic[1,1]*y/positive + intrinsic[1,2] - .5
        inside = inside and z > 0 and u >= 0 and u <= w-1 and v >= 0 and v <= h-1
        u0, v0, u1, v1, zmax = min(u0,u), min(v0,v), max(u1,u), max(v1,v), max(zmax,z)
    keep = lx0 <= extent and ly0 <= extent and lx1 >= -extent and ly1 >= -extent
    if keep and inside:
        x0, y0 = int(math.floor(u0)), int(math.floor(v0))
        x1, y1 = min(int(math.floor(u1))+1,w-1), min(int(math.floor(v1))+1,h-1)
        measured = np.float32(math.inf)
        for yy in range(y0,y1+1):
            for xx in range(x0,x1+1):
                measured = min(measured,depth[yy,xx])
        metric = np.float32(measured * np.float32(maximum))
        if metric > zmax + maximum * 0.0009765625:
            keep = False
    out[i] = keys[i] if keep else -1


@cuda.jit(cache=True)
def _new_keys(depth, intrinsic, camera_body, body, out, maximum, extent, min_z, max_z, voxel, invalid):
    i = cuda.grid(1)
    if i >= depth.size:
        return
    y, x = i // depth.shape[1], i % depth.shape[1]
    d = depth[y,x]
    out[i] = -1
    if d <= 0 or d >= 1:
        return
    metric = np.float32(d * np.float32(maximum))
    rx = (x+.5-intrinsic[0,2])/intrinsic[0,0]*metric
    ry = (y+.5-intrinsic[1,2])/intrinsic[1,1]*metric
    rz = np.float64(metric)
    px = rx*camera_body[0,0]+ry*camera_body[0,1]+rz*camera_body[0,2]+camera_body[0,3]
    py = rx*camera_body[1,0]+ry*camera_body[1,1]+rz*camera_body[1,2]+camera_body[1,3]
    pz = rx*camera_body[2,0]+ry*camera_body[2,1]+rz*camera_body[2,2]+camera_body[2,3]
    if pz < min_z or pz > max_z or abs(px) > extent or abs(py) > extent:
        return
    wx = px*body[0,0]+py*body[0,1]+pz*body[0,2]+body[0,3]
    wy = px*body[1,0]+py*body[1,1]+pz*body[1,2]+body[1,3]
    wz = px*body[2,0]+py*body[2,1]+pz*body[2,2]+body[2,3]
    vx, vy, vz = int(math.floor(wx/voxel)), int(math.floor(wy/voxel)), int(math.floor(wz/voxel))
    if min(vx,vy,vz) < -_SHIFT or max(vx,vy,vz) >= _SHIFT:
        invalid[0] = 1
        return
    out[i] = _encode(vx,vy,vz)


@cuda.jit(cache=True)
def _raster(keys, body, raster, horizon, resolution, padding, min_z, max_z, voxel):
    i = cuda.grid(1)
    if i >= keys.size:
        return
    vx, vy, vz = _decode(keys[i])
    x0,y0,z0,x1,y1,z1 = math.inf,math.inf,math.inf,-math.inf,-math.inf,-math.inf
    for c in range(8):
        dx = (vx+c//4)*voxel-body[0,3]
        dy = (vy+(c//2)%2)*voxel-body[1,3]
        dz = (vz+c%2)*voxel-body[2,3]
        x = dx*body[0,0]+dy*body[1,0]+dz*body[2,0]
        y = dx*body[0,1]+dy*body[1,1]+dz*body[2,1]
        z = dx*body[0,2]+dy*body[1,2]+dz*body[2,2]
        x0,y0,z0,x1,y1,z1 = min(x0,x),min(y0,y),min(z0,z),max(x1,x),max(y1,y),max(z1,z)
    if z0 > max_z or z1 < min_z:
        return
    lo_x = max(0,int(math.floor((x0+horizon)/resolution+.5))+padding)
    lo_y = max(0,int(math.floor((y0+horizon)/resolution+.5))+padding)
    hi_x = min(raster.shape[1]-1,int(math.floor((x1+horizon)/resolution+.5))+padding)
    hi_y = min(raster.shape[0]-1,int(math.floor((y1+horizon)/resolution+.5))+padding)
    for yy in range(lo_y,hi_y+1):
        for xx in range(lo_x,hi_x+1):
            cuda.atomic.max(raster,(yy,xx),1)


class _CudaView:
    def __init__(self, tensor):
        self.tensor = tensor
        dtype = {torch.float32:'<f4',torch.float64:'<f8',torch.int64:'<i8',torch.int32:'<i4'}[tensor.dtype]
        self.__cuda_array_interface__ = dict(shape=tuple(tensor.shape), strides=None,
            typestr=dtype, data=(tensor.data_ptr(),False), version=3)


def _view(tensor):
    return cuda.as_cuda_array(_CudaView(tensor), sync=False)


class CudaObstacleMemory(ObstacleMemory):
    def __init__(self, horizon_m, max_depth_m, geometry, device=0):
        super().__init__(horizon_m,max_depth_m,geometry)
        self.device = torch.device('cuda',device)
        cuda.select_device(device)
        self.keys = torch.empty(0,dtype=torch.int64,device=self.device)
        self.invalid = torch.zeros(1,dtype=torch.int32,device=self.device)

    def update(self, depth, intrinsic, camera_to_body, body_to_world):
        # Preserve the reference CPU matrix product and its input dtype.
        camera_world = body_to_world @ camera_to_body
        arrays = [torch.as_tensor(np.array(a, copy=True, order="C"),device=self.device,dtype=t)
                  for a,t in [(depth,torch.float32),(intrinsic,torch.float64),
                              (camera_to_body,torch.float64),(body_to_world,torch.float64),
                              (camera_world,torch.float64)]]
        d,k,cb,b,cw = arrays
        stream = cuda.external_stream(torch.cuda.current_stream(self.device).cuda_stream)
        merged = torch.empty(len(self.keys)+d.numel(),dtype=torch.int64,device=self.device)
        if len(self.keys):
            _retain[(len(self.keys)+127)//128,128,stream](_view(self.keys),_view(merged),_view(d),_view(b),_view(cw),_view(k),self.extent,self.maximum,VOXEL_M)
        _new_keys[(d.numel()+127)//128,128,stream](_view(d),_view(k),_view(cb),_view(b),_view(merged[len(self.keys):]),self.maximum,self.extent,self.min_z,self.max_z,VOXEL_M,_view(self.invalid))
        keys = torch.unique(merged,sorted=True)
        self.keys = keys[keys >= 0]
        raster = torch.zeros((self.size,self.size),dtype=torch.int32,device=self.device)
        if len(self.keys):
            _raster[(len(self.keys)+127)//128,128,stream](_view(self.keys),_view(b),_view(raster),self.horizon,self.resolution,self.padding,self.min_z,self.max_z,VOXEL_M)
        result = raster.cpu().numpy().astype(bool)
        if self.invalid.item():
            raise ValueError('Voxel coordinate exceeds CUDA integer key range')
        return result
