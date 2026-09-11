from __future__ import annotations

from pathlib import Path
from typing import Any


from .common import ModelArtifact, ServerSpec, required_paths


def build_server(args: Any, port: int, root: Path, artifact: ModelArtifact) -> ServerSpec:
    checkpoint = artifact.checkpoint
    server = root / "baselines/navdp/navdp_server.py"
    command = [
        args.server_python, str(server),
        "--port", str(port), "--device", "cuda:0", "--checkpoint", str(checkpoint),
    ]
    return ServerSpec(command, root, {}, required_paths(args, checkpoint, server))

