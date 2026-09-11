from __future__ import annotations

from pathlib import Path
from typing import Any

from dataclasses import dataclass
from typing import Callable


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



def required_paths(args: Any, checkpoint: Path, *extra: Path) -> tuple[Path, ...]:
    paths = [Path(args.server_python), checkpoint]
    paths.extend(extra)
    return tuple(paths)

