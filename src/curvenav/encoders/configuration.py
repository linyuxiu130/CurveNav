"""Calibrated memory splatting and metric position encoding."""

import math

import torch
from torch import Tensor


def memory_splat_stencil(points,valid,planning_horizon_m,grid_size,padding):
    """Compute immutable calibrated destinations, weights and mass once."""
    batch=points.shape[0]
    normalized = points[..., :2].float() / planning_horizon_m
    grid = grid_size
    stride = 64 / grid
    size = grid+2*padding
    coordinate = ((normalized + 1.0) * 0.5 * 63 - (stride-1)/2) / stride+padding
    coordinate = coordinate.clamp(0, size - 1)
    inside = valid & (normalized.abs() <= 1.0+padding*stride*2/63).all(dim=-1)
    x0 = coordinate[..., 0].floor().clamp(0, size - 1).long()
    y0 = coordinate[..., 1].floor().clamp(0, size - 1).long()
    x1 = (x0 + 1).clamp_max(size - 1)
    y1 = (y0 + 1).clamp_max(size - 1)
    wx, wy = coordinate[..., 0] - x0, coordinate[..., 1] - y0
    entries=[]
    mass=coordinate.new_zeros(batch,size**2,1)
    for x, y, weight in (
        (x0, y0, (1 - wx) * (1 - wy)),
        (x1, y0, wx * (1 - wy)),
        (x0, y1, (1 - wx) * wy),
        (x1, y1, wx * wy),
    ):
        weight = (weight * inside).float()[..., None]
        index = (y * size + x)[..., None]
        mass.scatter_add_(1, index, weight)
        entries.append((index,weight))
    return entries,mass,size


def apply_memory_splat(tokens,stencil):
    """Only learned features are recomputed; accumulation remains FP32."""
    entries,mass,size=stencil
    batch,_,model_dim=tokens.shape
    projected=tokens.float();bev=projected.new_zeros(batch,size**2,model_dim)
    for index,weight in entries:
        bev.scatter_add_(1,index.expand(-1,-1,model_dim),projected*weight)
    return bev/mass.clamp_min(1e-6),mass


def splat_memory(tokens,points,valid,planning_horizon_m,grid_size,padding):
    batch,count,_=tokens.shape
    if points.shape!=(batch,count,3):raise ValueError("visual points must have shape [B,N,3]")
    if valid.shape!=(batch,count) or valid.dtype!=torch.bool:raise ValueError("visual validity must be boolean [B,N]")
    return apply_memory_splat(tokens,memory_splat_stencil(points,valid,planning_horizon_m,grid_size,padding))


def metric_grid(
    model_dim: int,
    grid_size: int,
    planning_horizon_m: float,
) -> tuple[Tensor, Tensor]:
    axis = ((torch.arange(grid_size) * 4 + 1.5) / 63 * 2 - 1) * planning_horizon_m
    y, x = torch.meshgrid(axis, axis, indexing="ij")
    position = torch.stack((x, y, torch.zeros_like(x)), dim=-1).reshape(
        1, grid_size**2, 3
    )
    quarter = model_dim // 4
    frequency = torch.exp(
        torch.arange(quarter, dtype=torch.float32)
        * (-math.log(10_000.0) / max(quarter - 1, 1))
    )
    x_phase = math.pi * x[..., None] / planning_horizon_m * frequency
    y_phase = math.pi * y[..., None] / planning_horizon_m * frequency
    encoding = torch.cat(
        (x_phase.sin(), x_phase.cos(), y_phase.sin(), y_phase.cos()), dim=-1
    ).reshape(1, grid_size**2, model_dim)
    return position, encoding
