from __future__ import annotations

from pathlib import Path
from typing import Any

import os

from .common import ModelArtifact, ServerSpec, required_paths


def build_server(args: Any, port: int, root: Path, artifact: ModelArtifact) -> ServerSpec:
    checkpoint = artifact.checkpoint
    cwd = root / "baselines/x-navdp"
    server = cwd / "eval/src/policy_server.py"
    command = [
        args.server_python, "-m", "eval.src.policy_server", "--port", str(port),
        "--device", "cuda:0", "--embodiment", "wheeled",
        "--checkpoint", str(checkpoint),
    ]
    env = {"PYTHONPATH": os.pathsep.join((str(cwd), str(root)))}
    return ServerSpec(command, cwd, env, required_paths(args, checkpoint, server))

