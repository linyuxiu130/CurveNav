from __future__ import annotations

from pathlib import Path
from typing import Any

import os

from .common import ModelArtifact, ServerSpec, required_paths


def build_server(args: Any, port: int, root: Path, artifact: ModelArtifact) -> ServerSpec:
    checkpoint = artifact.checkpoint
    server = root / "baselines/sandplanner/sand_planner/server/simple_server.py"
    command = [
        args.server_python,
        str(server),
        "--port", str(port), "--device", "cuda:0", "--checkpoint", str(checkpoint),
    ]
    env = {"PYTHONPATH": os.pathsep.join((str(root / "baselines/sandplanner"), str(root)))}
    return ServerSpec(command, root, env, required_paths(args, checkpoint, server, root / "baselines/sandplanner/trajectory_stats.json"))

