from __future__ import annotations

from pathlib import Path
from typing import Any


from .common import ModelArtifact, ServerSpec, required_paths


def build_server(args: Any, port: int, root: Path, artifact: ModelArtifact) -> ServerSpec:
    checkpoint = artifact.checkpoint
    server = root / "baselines/iplanner/iplanner_server.py"
    config = root / "baselines/iplanner/configs/iplanner.yaml"
    command = [
        args.server_python, str(server),
        "--port", str(port), "--device", "cuda:0", "--checkpoint", str(checkpoint),
        "--config", str(config),
    ]
    return ServerSpec(command, root, {}, required_paths(args, checkpoint, server, config))

