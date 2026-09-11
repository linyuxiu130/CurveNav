from __future__ import annotations

from pathlib import Path
from typing import Any

import os
import subprocess

from .common import ModelArtifact, ServerSpec, required_paths


def _discover_m2f_config(args: Any) -> Path:
    if args.m2f_config:
        return args.m2f_config.expanduser().resolve()
    configured = os.environ.get("NAVBENCH_M2F_CONFIG")
    if configured:
        return Path(configured).expanduser().resolve()
    code = (
        "import pathlib,mmdet; print(pathlib.Path(mmdet.__file__).parent/"
        "'.mim/configs/mask2former/mask2former_r50_8xb2-lsj-50e_coco-panoptic.py')"
    )
    result = subprocess.run(
        [args.server_python, "-c", code], check=True, text=True,
        capture_output=True, timeout=20,
    )
    path = Path(result.stdout.strip())
    if not path.is_file():
        raise FileNotFoundError(f"Mask2Former config was not found: {path}")
    return path



def build_server(args: Any, port: int, root: Path, artifact: ModelArtifact) -> ServerSpec:
    checkpoint = artifact.checkpoint
    server = root / "baselines/viplanner/viplanner_server.py"
    config = root / "baselines/viplanner/configs/viplanner.yaml"
    m2f_config = _discover_m2f_config(args)
    m2f_checkpoint = (
        args.weight_root
        / "viplanner/mask2former_r50_8xb2-lsj-50e_coco-panoptic_20230118_125535-54df384a.pth"
    )
    command = [
        args.server_python, str(server),
        "--port", str(port), "--device", "cuda:0", "--checkpoint", str(checkpoint),
        "--config", str(config),
        "--m2f_config", str(m2f_config), "--m2f_checkpoint", str(m2f_checkpoint),
    ]
    required = required_paths(
        args, checkpoint, server, config, m2f_config, m2f_checkpoint
    )
    return ServerSpec(command, root, {}, required)

