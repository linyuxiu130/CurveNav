from __future__ import annotations

from pathlib import Path
from typing import Any

import os

from .common import ModelArtifact, ServerSpec, required_paths


def build_server(args: Any, port: int, root: Path, artifact: ModelArtifact) -> ServerSpec:
    checkpoint = artifact.checkpoint
    if artifact.model_config is None:
        raise ValueError("CurveNav requires an explicit checkpoint and config")
    config = artifact.model_config.expanduser().resolve()
    # The benchmark owns only this protocol shim.  Resolve the model package
    # from the explicit config bundle so a stale copied source tree cannot be
    # loaded accidentally when checkpoints/configs are staged independently.
    source_root = config.parent.parent
    server = root / "baselines/curvenav/curvenav_server.py"
    command = [
        args.server_python, str(server),
        "--port", str(port), "--device", "cuda:0",
        "--checkpoint", str(checkpoint), "--config", str(config),
    ]
    env = {"PYTHONPATH": os.pathsep.join((str(source_root / "src"), str(root)))}
    required = required_paths(
        args, checkpoint, server, config, source_root / "src/curvenav"
    )
    return ServerSpec(command, root, env, required)

