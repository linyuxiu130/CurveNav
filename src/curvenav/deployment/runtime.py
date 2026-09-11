"""Strict registered depth inference with bounded metric scene history."""

from dataclasses import dataclass
import time
import numpy as np
import torch

from curvenav.config_io import load_config
from curvenav.data.observation import DEPTH_CONTEXT_FIELDS
from curvenav.data.depth import PinholeIntrinsics
from curvenav.data.history import validate_transform
from curvenav.factory import build_policy
from curvenav.training.checkpoint import validate_policy_contract
from curvenav.training.ema import ExponentialMovingAverage
from curvenav.types import PolicyCondition


def load_policy(checkpoint_path, config_path, device: str):
    config = load_config(config_path)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    validate_policy_contract(checkpoint, config)
    policy = build_policy(config)
    policy.load_state_dict(checkpoint["model"], strict=True)
    ema = ExponentialMovingAverage(policy, decay=config.training.ema_decay)
    ema.load_state_dict(checkpoint["ema"])
    ema.copy_to(policy)
    policy.eval().to(device)
    return config, policy


@dataclass(frozen=True)
class RuntimePrediction:
    path: np.ndarray


class CurveNavRuntime:
    def __init__(self, config, policy, device: str):
        self.config, self.policy = config, policy
        self.device = torch.device(device)
        self.batch_size = 0
        self.requests = 0
        self.request_seconds = 0.0

    def reset(self, batch_size: int):
        self.batch_size = batch_size

    def step(self, point_goals, context):
        started = time.perf_counter()
        goals = np.asarray(point_goals, dtype=np.float32)
        if goals.shape != (self.batch_size, 2) or not np.isfinite(goals).all():
            raise ValueError("point_goals must be finite [B,2]")
        if set(context) != DEPTH_CONTEXT_FIELDS:
            raise ValueError(
                "inference requires one complete calibrated depth snapshot"
            )
        if context["depth"].dtype != np.float16:
            raise TypeError("snapshot depth must be float16")
        for name in DEPTH_CONTEXT_FIELDS - {"depth", "observation_valid"}:
            if context[name].dtype != np.float32 or not np.isfinite(context[name]).all():
                raise ValueError(f"snapshot {name} must be finite float32")
        depth = context["depth"]
        if not np.isfinite(depth).all() or np.any(depth < 0) or np.any(depth > 1):
            raise ValueError("snapshot depth must be finite and normalized to [0,1]")
        data = self.config.data
        if depth.shape != (self.batch_size, data.observation_frames, 1, data.image_height, data.image_width):
            raise ValueError("snapshot dimensions do not match the policy contract")
        validate_transform(context["camera_to_body"])
        validate_transform(context["observation_to_current"])
        for intrinsic in context["camera_intrinsics"].reshape(-1, 3, 3):
            PinholeIntrinsics.from_matrix(intrinsic, width=data.image_width, height=data.image_height)
        if np.any(context["observation_age_s"] < 0):
            raise ValueError("snapshot observations cannot come from the future")
        context = dict(context)
        context["depth"] = context["depth"].astype(np.float32)
        condition = PolicyCondition(
            point_goal=torch.tensor(goals, device=self.device),
            **{
                name: torch.tensor(value, device=self.device)
                for name, value in context.items()
            },
        )
        condition.validate()
        with torch.inference_mode():
            prediction = self.policy.sample(condition)
        xy = prediction.path[:, 1:].float().cpu().numpy()
        path = np.concatenate(
            (xy, np.zeros((*xy.shape[:-1], 1), dtype=np.float32)), axis=-1
        )
        self.request_seconds += time.perf_counter() - started
        self.requests += 1
        return RuntimePrediction(path)
