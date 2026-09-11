"""Shared HTTP tensor protocol for benchmark policy servers."""

from contextlib import contextmanager
import io
import json
import logging
import random

import numpy as np
from flask import Response, jsonify, request

logging.getLogger("werkzeug").setLevel(logging.ERROR)

def seed_process(seed):
    """Seed policy-side stochastic inference at each scene boundary."""
    value = int(seed)
    random.seed(value)
    np.random.seed(value)
    import torch
    torch.manual_seed(value)
    torch.cuda.manual_seed_all(value)


def read_raw(name):
    shape = tuple(json.loads(request.form[f"{name}_shape"]))
    dtype = np.dtype(request.form[f"{name}_dtype"])
    return np.frombuffer(request.files[name].read(), dtype=dtype).reshape(shape)


def read_rgb(name, batch_size):
    """Read contiguous uint8 RGB [B,H,W,3] from the sole request format."""
    image = read_raw(name)
    if image.dtype != np.uint8:
        raise ValueError(f"{name} dtype must be uint8, got {image.dtype}")
    if image.ndim != 4 or image.shape[0] != batch_size or image.shape[-1] != 3:
        raise ValueError(f"invalid {name} shape {image.shape} for batch {batch_size}")
    return np.ascontiguousarray(image)


def read_depth(batch_size):
    """Read the single depth wire format: contiguous float32 meters [B,H,W,1]."""
    depth = read_raw("depth")
    if depth.dtype != np.float32:
        raise ValueError(f"depth dtype must be float32 meters, got {depth.dtype}")
    if depth.ndim != 4 or depth.shape[0] != batch_size or depth.shape[-1] != 1:
        raise ValueError(f"invalid depth shape {depth.shape} for batch {batch_size}")
    return np.ascontiguousarray(depth)


def as_numpy(value):
    if value is None:
        return None
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=np.float32)


def policy_response(trajectory):
    """Encode only the trajectory consumed by the released evaluator."""
    trajectory = as_numpy(trajectory)
    if trajectory is None:
        raise ValueError("policy returned no execution trajectory")

    output = io.BytesIO()
    np.savez(output, trajectory=trajectory)
    return Response(output.getvalue(), mimetype="application/x-npz")


def health_payload():
    return jsonify({"ok": True})


@contextmanager
def inference_context():
    """Shared fp32 inference context for the official evaluation."""
    import torch

    with torch.inference_mode():
        yield


@contextmanager
def gradient_inference_context():
    """Gradient-enabled fp32 context for X-NavDP temporal guidance."""
    import torch

    with torch.enable_grad():
        yield
