"""Policy-server adapter registry.

An adapter only describes how to start a policy server. All models share the
same simulator, tasks, episode sharding, transport, and metrics.
"""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import subprocess
from typing import Any, Callable


@dataclass(frozen=True)
class ServerSpec:
    command: list[str]
    cwd: Path
    env: dict[str, str]
    required_paths: tuple[Path, ...] = ()


@dataclass(frozen=True)
class ModelArtifact:
    checkpoint: Path
    model_config: Path | None = None


@dataclass(frozen=True)
class ModelAdapter:
    checkpoint: str | None
    build_server: Callable[[Any, int, Path, ModelArtifact], ServerSpec]
    supports_policy_shards: bool = False
    policy_gpu_slots: int = 1


def _base_paths(args: Any, checkpoint: Path, *extra: Path) -> tuple[Path, ...]:
    paths = [Path(args.server_python), checkpoint]
    paths.extend(extra)
    return tuple(paths)


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


def _iplanner(args: Any, port: int, root: Path, artifact: ModelArtifact) -> ServerSpec:
    checkpoint = artifact.checkpoint
    server = root / "baselines/iplanner/iplanner_server.py"
    config = root / "baselines/iplanner/configs/iplanner.yaml"
    command = [
        args.server_python, str(server),
        "--port", str(port), "--device", "cuda:0", "--checkpoint", str(checkpoint),
        "--config", str(config),
    ]
    return ServerSpec(command, root, {}, _base_paths(args, checkpoint, server, config))


def _viplanner(args: Any, port: int, root: Path, artifact: ModelArtifact) -> ServerSpec:
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
    required = _base_paths(
        args, checkpoint, server, config, m2f_config, m2f_checkpoint
    )
    return ServerSpec(command, root, {}, required)


def _navdp(args: Any, port: int, root: Path, artifact: ModelArtifact) -> ServerSpec:
    checkpoint = artifact.checkpoint
    server = root / "baselines/navdp/navdp_server.py"
    command = [
        args.server_python, str(server),
        "--port", str(port), "--device", "cuda:0", "--checkpoint", str(checkpoint),
    ]
    return ServerSpec(command, root, {}, _base_paths(args, checkpoint, server))


def _sandplanner(args: Any, port: int, root: Path, artifact: ModelArtifact) -> ServerSpec:
    checkpoint = artifact.checkpoint
    server = root / "baselines/sandplanner/sand_planner/server/simple_server.py"
    command = [
        args.server_python,
        str(server),
        "--port", str(port), "--device", "cuda:0", "--checkpoint", str(checkpoint),
    ]
    env = {"PYTHONPATH": os.pathsep.join((str(root / "baselines/sandplanner"), str(root)))}
    return ServerSpec(command, root, env, _base_paths(args, checkpoint, server))


def _x_navdp(args: Any, port: int, root: Path, artifact: ModelArtifact) -> ServerSpec:
    checkpoint = artifact.checkpoint
    cwd = root / "baselines/x-navdp"
    server = cwd / "eval/src/policy_server.py"
    command = [
        args.server_python, "-m", "eval.src.policy_server", "--port", str(port),
        "--device", "cuda:0", "--embodiment", "wheeled",
        "--checkpoint", str(checkpoint),
    ]
    env = {"PYTHONPATH": os.pathsep.join((str(cwd), str(root)))}
    return ServerSpec(command, cwd, env, _base_paths(args, checkpoint, server))


def _curvenav(args: Any, port: int, root: Path, artifact: ModelArtifact) -> ServerSpec:
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
    required = _base_paths(
        args, checkpoint, server, config, source_root / "src/curvenav"
    )
    return ServerSpec(command, root, env, required)


ADAPTERS = {
    "iplanner": ModelAdapter(
        "iplanner/plannernet.pt", _iplanner,
    ),
    "viplanner": ModelAdapter(
        "viplanner/model.pt", _viplanner,
    ),
    "navdp": ModelAdapter(
        "navdp/navdp_pretrain.ckpt", _navdp, supports_policy_shards=True,
    ),
    "sandplanner": ModelAdapter(
        "sandplanner/NoMax.pth", _sandplanner,
    ),
    "x-navdp": ModelAdapter(
        "x-navdp/x-navdp_posttrain.ckpt", _x_navdp, policy_gpu_slots=2,
    ),
    "curvenav": ModelAdapter(
        None, _curvenav,
    ),
}


def get_adapter(name: str) -> ModelAdapter:
    return ADAPTERS[name]


def build_server(
    args: Any, port: int, root: Path, artifact: ModelArtifact,
) -> ServerSpec:
    return get_adapter(args.model).build_server(args, port, root, artifact)
