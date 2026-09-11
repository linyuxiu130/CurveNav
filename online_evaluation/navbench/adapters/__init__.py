"""Per-model launch adapters sharing transport, simulator and metrics."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .common import ModelAdapter, ModelArtifact, ServerSpec
from . import iplanner, viplanner, navdp, sandplanner, x_navdp, curvenav


ADAPTERS = {
    "iplanner": ModelAdapter(
        "iplanner/plannernet.pt", iplanner.build_server,
    ),
    "viplanner": ModelAdapter(
        "viplanner/model.pt", viplanner.build_server,
    ),
    "navdp": ModelAdapter(
        "navdp/navdp_pretrain.ckpt", navdp.build_server, supports_policy_shards=True,
    ),
    "sandplanner": ModelAdapter(
        "sandplanner/NoMax.pth", sandplanner.build_server,
    ),
    "x-navdp": ModelAdapter(
        "x-navdp/x-navdp_posttrain.ckpt", x_navdp.build_server, policy_gpu_slots=2,
    ),
    "curvenav": ModelAdapter(
        None, curvenav.build_server,
    ),
}


def get_adapter(name: str) -> ModelAdapter:
    return ADAPTERS[name]


def build_server(
    args: Any, port: int, root: Path, artifact: ModelArtifact,
) -> ServerSpec:
    return get_adapter(args.model).build_server(args, port, root, artifact)
