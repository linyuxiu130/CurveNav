#!/usr/bin/env python3
"""Multi-GPU launcher for the PointGoal benchmark suite."""

from __future__ import annotations

import argparse
import concurrent.futures
import contextlib
import csv
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import queue
import shlex
import shutil
import signal
import subprocess
import sys
import threading
import time
import urllib.request

from .adapters import ADAPTERS, ModelArtifact, ServerSpec, build_server, get_adapter
from .evaluator_process import EvaluatorProcess
from .episodes import sha256_file
from .suite import (
    SceneJob,
    invalid_episode_assets,
    invalid_navigation_assets,
    load_suite,
    missing_assets,
    suite_definition_sha256,
)


ROOT = Path(__file__).resolve().parents[1]
OFFICIAL_NUM_ENVS = 16
GPU_MIN_FREE_MB = 8500
PAPER_SEED = 1234
STOP = threading.Event()
ACTIVE: set[subprocess.Popen] = set()
ACTIVE_LOCK = threading.Lock()
INPUT_LAYOUT_VERSION = "mdl-overlay-v1"
UE4_MDL_MODULES = (
    "OmniUe4Base",
    "OmniUe4Function",
    "OmniUe4Translucent",
)


@dataclass(frozen=True)
class CheckpointTarget:
    model: str
    key: str
    artifact: ModelArtifact
    checkpoint_sha256: str
    model_config_sha256: str | None
    source_sha256: str | None
    artifact_sha256: str
    run_root: Path

    @property
    def checkpoint(self) -> Path:
        return self.artifact.checkpoint

    @property
    def model_config(self) -> Path | None:
        return self.artifact.model_config


@dataclass(frozen=True)
class ResidentMessage:
    kind: str
    scene_key: str
    target_key: str | None = None
    error: BaseException | None = None


class PolicyGpuPool:
    """Allocate explicit memory slots across policy processes sharing a GPU."""

    def __init__(self, gpus: list[int], slots_per_gpu: int) -> None:
        self.capacity = {gpu: slots_per_gpu for gpu in set(gpus)}
        self.available = dict(self.capacity)
        self.condition = threading.Condition()

    @contextlib.contextmanager
    def reserve(self, gpus: list[int], slots_per_server: int):
        requested: dict[int, int] = {}
        for gpu in gpus:
            requested[gpu] = requested.get(gpu, 0) + slots_per_server
        invalid = {
            gpu: slots for gpu, slots in requested.items()
            if slots > self.capacity[gpu]
        }
        if invalid:
            raise RuntimeError(
                f"policy GPU slot request exceeds capacity: {invalid}"
            )
        with self.condition:
            while not all(
                self.available[gpu] >= slots
                for gpu, slots in requested.items()
            ):
                if STOP.is_set():
                    raise RuntimeError("policy GPU allocation cancelled")
                self.condition.wait(timeout=1.0)
            for gpu, slots in requested.items():
                self.available[gpu] -= slots
        try:
            yield
        finally:
            with self.condition:
                for gpu, slots in requested.items():
                    self.available[gpu] += slots
                self.condition.notify_all()


def positive_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("value must be a positive integer") from exc
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def nonnegative_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("value must be a non-negative integer") from exc
    if parsed < 0:
        raise argparse.ArgumentTypeError("value must be non-negative")
    return parsed


def load_local_env(path: Path) -> None:
    """Load a small KEY=VALUE file without evaluating shell code."""
    if not path.is_file():
        return
    for raw_line in path.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip("'\"")
        if key and key not in os.environ:
            os.environ[key] = os.path.expandvars(os.path.expanduser(value))


def existing_python(*candidates: str | Path | None) -> str:
    for candidate in candidates:
        if not candidate:
            continue
        path = Path(candidate).expanduser()
        if path.is_file() and os.access(path, os.X_OK):
            # Keep the venv/conda entry point itself. Resolving its ``python``
            # symlink can bypass pyvenv.cfg and silently select base packages.
            return str(path.absolute())
    raise FileNotFoundError("no configured executable Python was found")


def parse_gpus(value: str) -> list[int]:
    try:
        gpus = [int(item.strip()) for item in value.split(",") if item.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("GPU IDs must be comma-separated integers") from exc
    if not gpus:
        raise argparse.ArgumentTypeError("provide at least one GPU")
    if len(set(gpus)) != len(gpus) or min(gpus) < 0:
        raise argparse.ArgumentTypeError("GPU IDs must be unique non-negative integers")
    return gpus


def parse_gpu_mapping(value: str) -> list[int]:
    """Parse an ordered server assignment; physical GPU IDs may repeat."""
    try:
        gpus = [int(item.strip()) for item in value.split(",") if item.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("GPU IDs must be comma-separated integers") from exc
    if not gpus or min(gpus) < 0:
        raise argparse.ArgumentTypeError("provide non-negative GPU IDs")
    return gpus


def parse_shard_weights(value: str) -> list[int]:
    try:
        weights = [int(item.strip()) for item in value.split(",") if item.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "shard weights must be comma-separated integers"
        ) from exc
    if not weights or min(weights) <= 0:
        raise argparse.ArgumentTypeError("shard weights must be positive")
    return weights


def weighted_scene_shard(
    jobs: list[SceneJob], shard_index: int, weights: list[int],
) -> list[SceneJob]:
    """Assign ordered scenes with deterministic smooth weighted round-robin."""
    current = [0] * len(weights)
    total = sum(weights)
    selected: list[SceneJob] = []
    for job in jobs:
        current = [value + weight for value, weight in zip(current, weights)]
        owner = max(range(len(weights)), key=current.__getitem__)
        current[owner] -= total
        if owner == shard_index:
            selected.append(job)
    return selected


def discover_gpu_ids() -> list[int]:
    """Return every visible physical GPU; evaluation is GPU-only."""
    result = subprocess.run(
        [
            "nvidia-smi", "--query-gpu=index", "--format=csv,noheader,nounits",
        ],
        check=True, text=True, capture_output=True, timeout=5,
    )
    ids = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    if not ids:
        raise ValueError("nvidia-smi reported no GPUs")
    return parse_gpus(",".join(ids))


def parse_args() -> argparse.Namespace:
    load_local_env(ROOT / "config" / "local.env")
    parser = argparse.ArgumentParser(
        description="Evaluate PointGoal policies on the official X-NavDP suite."
    )
    parser.add_argument("--model", required=True, choices=tuple(ADAPTERS))
    parser.add_argument(
        "--suite-manifest", type=Path, default=ROOT / "suites/pointgoal-v2.json",
    )
    parser.add_argument(
        "--scene-root", type=Path,
        help="root containing the official held-out InternScenes assets",
    )
    parser.add_argument(
        "--episodes-per-scene", type=int,
        help="explicit upstream num_episodes diagnostic; omit for the 100-episode paper run",
    )
    parser.add_argument(
        "--num-envs", type=positive_int,
        default=positive_int(os.environ.get(
            "NAVBENCH_NUM_ENVS", str(OFFICIAL_NUM_ENVS)
        )),
        help=("Isaac vector environment count; the pointgoal-v2 benchmark "
              f"contract uses {OFFICIAL_NUM_ENVS}"),
    )
    parser.add_argument(
        "--scenes",
        help="comma-separated scene names or split/name keys for diagnostics",
    )
    parser.add_argument(
        "--gpus",
        type=parse_gpus,
        default=None,
        help="available GPU IDs; omitted uses all nvidia-smi-visible GPUs",
    )
    parser.add_argument(
        "--server-gpus", type=parse_gpu_mapping,
        help=("single-scene policy shard GPU pool, or one policy GPU per "
              "evaluator GPU for multi-scene runs; omitted keeps one server "
              "on each evaluator GPU"),
    )
    parser.add_argument(
        "--shard-index", type=nonnegative_int,
        default=nonnegative_int(os.environ.get("NAVBENCH_SHARD_INDEX", "0")),
        help="zero-based scene shard for distributing one frozen suite across hosts",
    )
    parser.add_argument(
        "--shard-count", type=positive_int,
        default=positive_int(os.environ.get("NAVBENCH_SHARD_COUNT", "1")),
        help="number of deterministic scene shards participating in the run",
    )
    parser.add_argument(
        "--shard-weights", type=parse_shard_weights,
        default=None,
        help=("comma-separated relative host capacities; all hosts must use "
              "the same list (default: equal-capacity shards)"),
    )
    parser.add_argument(
        "--checkpoint", dest="checkpoints", action="append", type=Path,
        metavar="PATH",
        help="model checkpoint; repeat to evaluate several checkpoints per scene load",
    )
    parser.add_argument(
        "--artifact-bundle", dest="artifact_bundles", action="append", type=Path,
        metavar="DIR",
        help=("immutable model artifact bundle to evaluate after --model; repeat "
              "to reuse every scene across several model adapters"),
    )
    parser.add_argument(
        "--checkpoint-queue", type=Path, metavar="DIR",
        help=("keep selected scenes resident and evaluate each complete "
              "release bundle atomically renamed to DIR/*.ready"),
    )
    parser.add_argument(
        "--model-config", type=Path,
        help="model-specific inference config (required for CurveNav)",
    )
    parser.add_argument("--weight-root", type=Path)
    parser.add_argument("--output-root", type=Path, default=ROOT / "runs")
    parser.add_argument(
        "--resume-root", type=Path,
        help="resume a prior run root, skipping scenes with complete metric.csv files",
    )
    parser.add_argument("--port-base", type=int, default=18880)
    parser.add_argument(
        "--launch-stagger", type=float,
        default=float(os.environ.get("NAVBENCH_LAUNCH_STAGGER", "2.0")),
        help="seconds between worker starts to avoid Isaac Kit startup contention",
    )
    parser.add_argument(
        "--max-workers", type=positive_int,
        default=(
            positive_int(os.environ["NAVBENCH_MAX_WORKERS"])
            if os.environ.get("NAVBENCH_MAX_WORKERS") else None
        ),
        help="maximum concurrently resident simulator scenes",
    )
    parser.add_argument(
        "--cpu-threads-per-worker", type=positive_int,
        default=positive_int(os.environ.get("NAVBENCH_CPU_THREADS_PER_WORKER", "4")),
        help="OMP/MKL/OpenBLAS threads assigned to each simulator worker",
    )
    parser.add_argument(
        "--gpu-poll-seconds", type=positive_int,
        default=positive_int(os.environ.get("NAVBENCH_GPU_POLL_SECONDS", "60")),
        help="seconds between GPU memory gate checks",
    )
    parser.add_argument(
        "--policy-gpu-slots", type=positive_int,
        default=positive_int(os.environ.get("NAVBENCH_POLICY_GPU_SLOTS", "2")),
        help=("memory capacity units per policy GPU; X-NavDP consumes two, "
              "other adapters consume one"),
    )
    parser.add_argument("--server-python")
    parser.add_argument("--eval-python")
    parser.add_argument(
        "--xnavdp-root", type=Path,
        help="pinned upstream checkout's baselines/x-navdp directory",
    )
    parser.add_argument("--m2f-config", type=Path)
    parser.add_argument("--dry-run", action="store_true", help="print the resolved plan without launching anything")
    parser.add_argument(
        "--check-assets", action="store_true",
        help="validate all selected scene and episode inputs, then exit",
    )
    args = parser.parse_args()

    if args.gpus is None:
        configured_gpus = os.environ.get("NAVBENCH_GPUS")
        try:
            args.gpus = (
                parse_gpus(configured_gpus)
                if configured_gpus else discover_gpu_ids()
            )
        except (
            argparse.ArgumentTypeError,
            OSError,
            ValueError,
            subprocess.SubprocessError,
        ) as exc:
            parser.error(str(exc))
    if args.server_gpus is None:
        configured_server_gpus = os.environ.get("NAVBENCH_SERVER_GPUS")
        if configured_server_gpus:
            try:
                args.server_gpus = parse_gpu_mapping(configured_server_gpus)
            except argparse.ArgumentTypeError as exc:
                parser.error(str(exc))
    if args.max_workers is None:
        args.max_workers = len(args.gpus)
    if args.max_workers > len(args.gpus):
        parser.error("--max-workers cannot exceed the number of --gpus")
    if args.server_gpus is not None and len(args.server_gpus) > args.num_envs:
        parser.error("--server-gpus cannot contain more entries than --num-envs")

    if args.shard_index >= args.shard_count:
        parser.error("--shard-index must be smaller than --shard-count")
    if args.shard_weights is None:
        configured_weights = os.environ.get("NAVBENCH_SHARD_WEIGHTS")
        args.shard_weights = (
            parse_shard_weights(configured_weights)
            if configured_weights else [1] * args.shard_count
        )
    if len(args.shard_weights) != args.shard_count:
        parser.error("--shard-weights must contain exactly --shard-count values")
    args.precision = "fp32"
    args.seed = PAPER_SEED
    if args.episodes_per_scene is not None and args.episodes_per_scene <= 0:
        parser.error("--episodes-per-scene must be positive")
    if args.num_envs is not None and args.episodes_per_scene is not None:
        if args.episodes_per_scene < args.num_envs:
            parser.error("--num-envs cannot exceed --episodes-per-scene")
    if args.launch_stagger < 0:
        parser.error("--launch-stagger must be non-negative")
    if args.checkpoint_queue is not None and args.resume_root is not None:
        parser.error("--checkpoint-queue cannot be combined with --resume-root")
    return args


def resolve_paths(args: argparse.Namespace) -> None:
    home = Path.home()
    args.eval_python = existing_python(
        args.eval_python,
        os.environ.get("NAVBENCH_EVAL_PYTHON"),
        home / "miniconda3/envs/isaaclab/bin/python",
    )
    if args.model == "viplanner":
        args.server_python = existing_python(
            args.server_python,
            os.environ.get("NAVBENCH_VIPLANNER_PYTHON"),
            home / "miniconda3/envs/viplanner/bin/python",
        )
    else:
        args.server_python = existing_python(
            args.server_python,
            os.environ.get("NAVBENCH_SERVER_PYTHON"),
            ROOT / ".venv/bin/python",
        )

    weight_root = args.weight_root or Path(os.environ.get("NAVBENCH_WEIGHT_ROOT", ROOT / "weights"))
    args.weight_root = weight_root.expanduser().resolve()
    adapter = get_adapter(args.model)
    if args.checkpoints:
        args.checkpoints = [path.expanduser().resolve() for path in args.checkpoints]
    elif adapter.checkpoint:
        args.checkpoints = [args.weight_root / adapter.checkpoint]
    else:
        raise SystemExit(f"{args.model} requires at least one --checkpoint")
    if len(set(args.checkpoints)) != len(args.checkpoints):
        raise SystemExit("--checkpoint paths must be unique")
    if args.model_config is not None:
        args.model_config = args.model_config.expanduser().resolve()
    if args.checkpoint_queue is not None:
        args.checkpoint_queue = args.checkpoint_queue.expanduser().resolve()
    args.artifact_bundles = [
        path.expanduser().resolve() for path in (args.artifact_bundles or [])
    ]
    args.additional_artifacts = [
        queued_artifact(bundle) for bundle in args.artifact_bundles
    ]

    args.suite_manifest = args.suite_manifest.expanduser().resolve()
    configured_runtime = args.xnavdp_root or os.environ.get("NAVBENCH_XNAVDP_ROOT")
    args.xnavdp_root = (
        Path(configured_runtime).expanduser().resolve() if configured_runtime else None
    )
    configured_scene_root = args.scene_root or os.environ.get("NAVBENCH_SCENE_ROOT")
    args.scene_root = Path(configured_scene_root or ROOT / "assets/scenes").expanduser().resolve()
    selected = (
        {value.strip() for value in args.scenes.split(",") if value.strip()}
        if args.scenes else None
    )
    try:
        args.suite, all_jobs = load_suite(
            args.suite_manifest, args.scene_root, ROOT,
            args.episodes_per_scene, selected,
        )
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise SystemExit(f"Invalid suite: {exc}") from exc
    args.jobs = weighted_scene_shard(
        all_jobs, args.shard_index, args.shard_weights,
    )
    if not args.jobs:
        raise SystemExit(
            f"scene shard {args.shard_index}/{args.shard_count} contains no selected scenes"
        )
    if (
        args.server_gpus is not None
        and len(args.jobs) > 1
        and len(args.server_gpus) != len(args.gpus)
    ):
        raise SystemExit(
            "multi-scene --server-gpus must contain exactly one GPU for each "
            "--gpus entry"
        )
    if (
        args.server_gpus is not None
        and len(args.jobs) == 1
        and len(set(args.server_gpus)) != len(args.server_gpus)
    ):
        raise SystemExit("single-scene policy shards require distinct server GPUs")
    if (
        args.server_gpus is not None
        and len(args.jobs) == 1
        and len(args.server_gpus) > 1
        and any(
            not get_adapter(model).supports_policy_shards
            for model in [args.model, *(model for model, _ in args.additional_artifacts)]
        )
    ):
        raise SystemExit(
            "every model in a single-scene run must have an audited "
            "multi-GPU policy-sharding contract"
        )
    if args.checkpoint_queue is not None and len(args.jobs) > len(args.gpus):
        raise SystemExit(
            "--checkpoint-queue requires one GPU per selected resident scene; "
            f"selected {len(args.jobs)} scenes for {len(args.gpus)} GPUs"
        )
    models = [args.model, *(model for model, _ in args.additional_artifacts)]
    oversized = [
        model for model in models
        if get_adapter(model).policy_gpu_slots > args.policy_gpu_slots
    ]
    if oversized:
        raise SystemExit(
            f"--policy-gpu-slots is too small for adapters: {oversized}"
        )
    args.all_jobs = all_jobs
    args.output_root = args.output_root.expanduser().resolve()
    if args.resume_root is not None:
        args.resume_root = args.resume_root.expanduser().resolve()


def python_cuda_library_path(python: str) -> str | None:
    code = (
        "from pathlib import Path; import torch; "
        "root=Path(torch.__file__).resolve().parent; "
        "nvidia=root.parent/'nvidia'; "
        "print(':'.join(map(str,[root/'lib',*sorted(nvidia.glob('*/lib'))])))"
    )
    result = subprocess.run(
        [python, "-c", code], check=True, text=True, capture_output=True, timeout=20
    )
    return result.stdout.strip() or None


def build_checkpoint_targets(
    model: str,
    checkpoints: list[Path],
    model_config: Path | None,
    session_root: Path,
) -> list[CheckpointTarget]:
    targets = []
    digests: set[str] = set()
    for index, checkpoint in enumerate(checkpoints):
        target = build_checkpoint_target(
            model, ModelArtifact(checkpoint, model_config), session_root, index
        )
        digest = target.artifact_sha256
        if digest in digests:
            raise SystemExit("--checkpoint artifacts must have distinct contents")
        digests.add(digest)
        targets.append(target)
    return targets


def build_initial_targets(
    args: argparse.Namespace, session_root: Path,
) -> list[CheckpointTarget]:
    """Bind one ordered, immutable model set to each scene lifetime."""
    artifacts = [
        *((args.model, ModelArtifact(checkpoint, args.model_config))
          for checkpoint in args.checkpoints),
        *args.additional_artifacts,
    ]
    targets: list[CheckpointTarget] = []
    identities: set[str] = set()
    for index, (model, artifact) in enumerate(artifacts):
        target = build_checkpoint_target(model, artifact, session_root, index)
        if target.artifact_sha256 in identities:
            raise SystemExit("model artifacts must have distinct contents")
        identities.add(target.artifact_sha256)
        targets.append(target)
    return targets


def sha256_source_tree(root: Path) -> str:
    """Hash CurveNav inference source by relative path and file content."""
    files = sorted(path for path in root.rglob("*.py") if path.is_file())
    if not files:
        raise FileNotFoundError(f"CurveNav source tree has no Python files: {root}")
    digest = hashlib.sha256()
    for path in files:
        digest.update(path.relative_to(root).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def build_checkpoint_target(
    model: str,
    artifact: ModelArtifact,
    session_root: Path,
    index: int,
) -> CheckpointTarget:
    checkpoint_digest = sha256_file(artifact.checkpoint)
    config_digest = (
        sha256_file(artifact.model_config) if artifact.model_config else None
    )
    source_digest = None
    if artifact.model_config is not None:
        source_digest = sha256_source_tree(
            artifact.model_config.parent.parent / "src" / "curvenav"
        )
    identity = hashlib.sha256("\0".join(filter(None, (
        model, checkpoint_digest, config_digest, source_digest,
    ))).encode("ascii")).hexdigest()
    key = f"{index:02d}-{model}-{identity[:12]}"
    return CheckpointTarget(
        model=model,
        key=key,
        artifact=artifact,
        checkpoint_sha256=checkpoint_digest,
        model_config_sha256=config_digest,
        source_sha256=source_digest,
        artifact_sha256=identity,
        run_root=session_root / "models" / model / key,
    )


def target_identity(target: CheckpointTarget) -> dict[str, str | None]:
    return {
        "model": target.model,
        "key": target.key,
        "artifact_sha256": target.artifact_sha256,
        "checkpoint_sha256": target.checkpoint_sha256,
        "model_config_sha256": target.model_config_sha256,
        "source_sha256": target.source_sha256,
    }


def args_for_target(
    args: argparse.Namespace, target: CheckpointTarget,
) -> argparse.Namespace:
    """Bind one immutable model adapter to a resident-scene evaluation."""
    target_args = argparse.Namespace(**vars(args))
    target_args.model = target.model
    return target_args


def evaluator_command(
    args: argparse.Namespace, port: int, job: SceneJob, config: Path,
    gpu: int | None = None,
) -> list[str]:
    device = "cuda:0"
    portable_root = cache_root() / "kit" / f"gpu_{gpu or 0}"
    command = [
        args.eval_python, "-m", "navbench.scene_evaluator",
        "--config-file", str(config),
        "--scene-index", str(args.suite["splits"][job.split]["scenes"].index(job.name)),
        "--server-port", str(port), "--device", device,
        f"--/renderer/activeGpu={gpu if gpu is not None else 0}",
        (
            "--/plugins/carb.tasking.plugin/threadCount="
            f"{args.cpu_threads_per_worker}"
        ),
        "--portable-root", str(portable_root),
    ]
    command.extend(shlex.split(os.environ.get("NAVBENCH_EVAL_KIT_ARGS", "")))
    return command


def write_upstream_config(
    args: argparse.Namespace, job: SceneJob, scene_root: Path, input_root: Path,
) -> Path:
    source = (
        args.xnavdp_root / "eval/config/eval_pointgoal"
        / f"wheeled_internscene_{job.split}.yaml"
    )
    text = source.read_text()
    replacements = {
        "run_root_dir: outputs/evaluation":
            f"run_root_dir: {json.dumps(str(scene_root / 'upstream-output'))}",
        "  scene_dir: data/scenes":
            f"  scene_dir: {json.dumps(str(input_root))}",
        f"  dataset_dir: data/scenes/navigation_metadata/internscenes_{job.split}":
            "  dataset_dir: " + json.dumps(str(
                input_root / "navigation_metadata" / f"internscenes_{job.split}"
            )),
    }
    for old, new in replacements.items():
        if text.count(old) != 1:
            raise RuntimeError(f"pinned upstream config line differs: {old}")
        text = text.replace(old, new)
    if args.num_envs is not None:
        old = "  num_envs: 1"
        new = f"  num_envs: {args.num_envs}"
        if text.count(old) != 1:
            raise RuntimeError("pinned upstream config num_envs line differs")
        text = text.replace(old, new)
    target = scene_root / "upstream-config.yaml"
    target.write_text(text)
    return target


def cache_root() -> Path:
    return Path(
        os.environ.get("NAVBENCH_CACHE_ROOT", Path.home() / ".cache" / "navbench")
    ).expanduser()


def isaac_ue4_mdl_root(eval_python: str) -> Path:
    """Locate the three Isaac modules referenced relatively by scene MDLs."""
    env_root = Path(eval_python).expanduser().resolve().parents[1]
    candidates = sorted(
        env_root.glob("lib/python*/site-packages/omni/mdl/core/Ue4")
    )
    valid = sorted({
        path.resolve() for path in candidates
        if all((path / f"{name}.mdl").is_file() for name in UE4_MDL_MODULES)
    })
    if len(valid) != 1:
        raise RuntimeError(
            f"expected one Isaac UE4 MDL directory under {env_root}, found {valid}"
        )
    return valid[0]


def _symlink_asset(source: str, target: str) -> str:
    Path(target).symlink_to(Path(source).absolute())
    return target


def _ignore_isaac_mdl_dependencies(_directory: str, names: list[str]) -> set[str]:
    """Keep the runtime's canonical UE4 MDL modules out of scene overlays."""
    return {f"{name}.mdl" for name in UE4_MDL_MODULES}.intersection(names)


def materialize_scene_overlay(
    source: Path, target: Path, mdl_root: Path,
) -> int:
    """Mirror immutable assets by symlink and close their relative MDL imports."""
    shutil.copytree(
        source,
        target,
        symlinks=True,
        copy_function=_symlink_asset,
        ignore=_ignore_isaac_mdl_dependencies,
    )
    linked = 0
    mdl_directories = sorted({path.parent for path in target.rglob("*.mdl")})
    for directory in mdl_directories:
        text = "\n".join(
            path.read_text(errors="ignore") for path in directory.glob("*.mdl")
        )
        for name in UE4_MDL_MODULES:
            if f".::{name}" not in text:
                continue
            destination = directory / f"{name}.mdl"
            if destination.exists() or destination.is_symlink():
                raise RuntimeError(
                    f"scene asset shadows Isaac MDL dependency: {destination}"
                )
            destination.symlink_to(mdl_root / f"{name}.mdl")
            linked += 1
    return linked


def prepare_upstream_inputs(args: argparse.Namespace, run_root: Path) -> Path:
    """Create one stable, immutable evaluator input tree and link it into the run."""
    suite_hash = suite_definition_sha256(args.suite)
    selected_splits = sorted({job.split for job in args.jobs})
    scene_root_hash = hashlib.sha256(
        ("\0".join((
            str(args.scene_root.resolve()),
            INPUT_LAYOUT_VERSION,
            ",".join(selected_splits),
        ))).encode("utf-8")
    ).hexdigest()
    inputs_root = cache_root() / "official-inputs"
    inputs_root.mkdir(parents=True, exist_ok=True)
    suite_root = inputs_root / f"{suite_hash[:16]}-{scene_root_hash[:16]}"
    target = suite_root / "data/scenes"
    if target.is_dir():
        runtime_cwd = run_root / "upstream-runtime"
        runtime_cwd.mkdir()
        (runtime_cwd / "data").symlink_to(suite_root / "data", target_is_directory=True)
        return target
    if suite_root.exists():
        raise RuntimeError(f"incomplete stable input cache: {suite_root}")

    incoming = inputs_root / f"{suite_root.name}.incoming.{os.getpid()}"
    if incoming.exists():
        raise RuntimeError(f"stable input staging path exists: {incoming}")
    target = incoming / "data/scenes"
    target.mkdir(parents=True)
    robot_root = incoming / "data/robots"
    robot_root.mkdir()
    shutil.copy2(ROOT / "assets/robots/dingo.usd", robot_root / "dingo.usd")
    mdl_root = isaac_ue4_mdl_root(args.eval_python)
    mdl_links: dict[str, int] = {}
    for name in ("Materials", "SkyTexture"):
        (target / name).symlink_to(args.scene_root / name, target_is_directory=True)
    shutil.copy2(ROOT / "assets/scenes/scene_split.json", target / "scene_split.json")
    for split in selected_splits:
        source_domain = args.scene_root / f"internscenes_{split}"
        domain_name = f"internscenes_{split}"
        mdl_links[domain_name] = materialize_scene_overlay(
            source_domain, target / domain_name, mdl_root,
        )
        metadata = target / "navigation_metadata" / f"internscenes_{split}"
        metadata.mkdir(parents=True)
        (metadata / "esdf").symlink_to(
            args.scene_root / "navigation_metadata" / f"internscenes_{split}" / "esdf",
            target_is_directory=True,
        )
        for scene in args.suite["splits"][split]["scenes"]:
            shutil.copytree(
                ROOT / "assets/scenes" / f"internscenes_{split}" / scene,
                metadata / "pointgoal_start_pair" / scene,
            )
    (incoming / "data/mdl-closure.json").write_text(json.dumps({
        "layout": INPUT_LAYOUT_VERSION,
        "modules": list(UE4_MDL_MODULES),
        "links_by_domain": mdl_links,
    }, indent=2) + "\n")
    try:
        incoming.rename(suite_root)
    except FileExistsError:
        shutil.rmtree(incoming)
        if not (suite_root / "data/scenes").is_dir():
            raise
    runtime_cwd = run_root / "upstream-runtime"
    runtime_cwd.mkdir()
    (runtime_cwd / "data").symlink_to(suite_root / "data", target_is_directory=True)
    target = suite_root / "data/scenes"
    return target


def prepare_scene_roots(
    args: argparse.Namespace, session_root: Path, input_root: Path,
) -> dict[str, tuple[Path, Path]]:
    """Materialize one evaluator config for each resident scene."""
    prepared: dict[str, tuple[Path, Path]] = {}
    for job in args.jobs:
        scene_root = session_root / "scene-sessions" / job.split / job.name
        scene_root.mkdir(parents=True, exist_ok=False)
        eval_config = write_upstream_config(args, job, scene_root, input_root)
        prepared[job.key] = (scene_root, eval_config)
    return prepared


def load_prepared_scene_roots(
    jobs: list[SceneJob], session_root: Path,
) -> dict[str, tuple[Path, Path]]:
    """Reuse immutable resident-scene configs when a session is resumed."""
    prepared: dict[str, tuple[Path, Path]] = {}
    missing: list[Path] = []
    for job in jobs:
        scene_root = session_root / "scene-sessions" / job.split / job.name
        eval_config = scene_root / "upstream-config.yaml"
        if not scene_root.is_dir() or not eval_config.is_file():
            missing.append(eval_config)
        else:
            prepared[job.key] = (scene_root, eval_config)
    if missing:
        raise SystemExit(
            "Resume root is missing prepared scene configs:\n  "
            + "\n  ".join(str(path) for path in missing)
        )
    return prepared


def prepare_checkpoint_roots(
    args: argparse.Namespace, targets: list[CheckpointTarget], *, resume: bool,
) -> None:
    for target in targets:
        if not resume:
            target.run_root.mkdir(parents=True, exist_ok=False)
        for job in args.jobs:
            scene_root = target.run_root / "scenes" / job.split / job.name
            if resume:
                if not scene_root.is_dir():
                    raise SystemExit(f"resume root is missing scene output: {scene_root}")
            else:
                scene_root.mkdir(parents=True, exist_ok=False)


def metric_is_complete(path: Path, expected: int) -> bool:
    """Return whether one scene has the complete official episode prefix."""
    if not path.is_file():
        return False
    try:
        with path.open(newline="") as handle:
            reader = csv.DictReader(handle)
            required = {"success", "spl", "distance", "episode_idx"}
            if not reader.fieldnames or not required.issubset(reader.fieldnames):
                return False
            rows = list(reader)
        ids = [int(row["episode_idx"]) for row in rows]
    except (OSError, TypeError, ValueError, csv.Error):
        return False
    return len(rows) == expected and ids == list(range(expected))


def validate_resume_root(
    args: argparse.Namespace,
    session_root: Path,
    targets: list[CheckpointTarget],
) -> dict:
    """Require an exact session identity before reusing prior results."""
    metadata_path = session_root / "run.json"
    if not metadata_path.is_file():
        raise SystemExit(f"Resume root has no run.json: {session_root}")
    try:
        metadata = json.loads(metadata_path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise SystemExit(f"Invalid resume metadata: {metadata_path}") from exc
    expected = {
        "suite_definition_sha256": suite_definition_sha256(args.suite),
        "suite": args.suite["name"],
        "model": args.model,
        "seed": args.seed,
        "num_envs": args.num_envs or 1,
        "runtime_revision": args.suite["source"]["code_revision"],
        "shard_index": args.shard_index,
        "shard_count": args.shard_count,
        "shard_weights": args.shard_weights,
        "scene_keys": [job.key for job in args.jobs],
        "episode_count": sum(job.episodes for job in args.jobs),
        "gpus": args.gpus,
        "server_gpus": args.server_gpus,
        "checkpoints": [
            target_identity(target) for target in targets
        ],
        "model_config_sha256": (
            sha256_file(args.model_config)
            if args.model_config and args.model_config.is_file() else None
        ),
    }
    mismatches = [
        f"{key}: expected {value!r}, found {metadata.get(key)!r}"
        for key, value in expected.items()
        if metadata.get(key) != value
    ]
    if mismatches:
        raise SystemExit(
            "Resume root identity mismatch:\n  " + "\n  ".join(mismatches)
        )
    return metadata


def common_asset_errors(args: argparse.Namespace) -> list[str]:
    errors: list[str] = []
    for path in (args.scene_root / "Materials", args.scene_root / "SkyTexture"):
        if not path.is_dir():
            errors.append(str(path))
    robot = ROOT / "assets/robots/dingo.usd"
    if not robot.is_file():
        errors.append(str(robot))
    elif sha256_file(robot) != args.suite["simulator_contract"]["robot_asset_sha256"]:
        errors.append("Dingo USD SHA-256 differs from X-NavDP")
    split = json.loads((ROOT / "assets/scenes/scene_split.json").read_text())
    for domain, config in args.suite["splits"].items():
        if split.get(f"{domain}_eval") != config["scenes"]:
            errors.append(f"{domain} scene split differs from X-NavDP")
    return errors


def validate(args: argparse.Namespace) -> None:
    if args.xnavdp_root is None:
        raise SystemExit(
            "NAVBENCH_XNAVDP_ROOT is required; run scripts/prepare_xnavdp_runtime.sh"
        )
    artifacts = [
        *((args.model, ModelArtifact(checkpoint, args.model_config))
          for checkpoint in args.checkpoints),
        *args.additional_artifacts,
    ]
    specs = []
    for model, artifact in artifacts:
        target_args = argparse.Namespace(**vars(args))
        target_args.model = model
        specs.append(build_server(target_args, args.port_base, ROOT, artifact))
    required = (
        Path(args.eval_python), args.xnavdp_root, args.suite_manifest, args.scene_root,
        ROOT / "navbench/scene_evaluator.py",
        *(path for spec in specs for path in (spec.cwd, *spec.required_paths)),
    )
    missing = [str(path) for path in required if not path.exists()]
    missing.extend(str(path) for path in missing_assets(args.jobs))
    missing.extend(common_asset_errors(args))
    missing.extend(invalid_episode_assets(args.jobs))
    missing.extend(invalid_navigation_assets(args.suite, args.jobs))
    if missing:
        raise SystemExit("Missing required paths:\n  " + "\n  ".join(missing))
    for spec in specs:
        executable = spec.command[0] if spec.command else ""
        executable_path = Path(executable)
        if executable_path.is_absolute():
            resolved_executable = executable_path
        elif executable_path.parent != Path("."):
            resolved_executable = spec.cwd / executable_path
        else:
            resolved_executable = Path(shutil.which(executable) or "")
        if not resolved_executable.is_file() or not os.access(resolved_executable, os.X_OK):
            executable = executable or "<empty command>"
            raise SystemExit(f"Policy server executable was not found: {executable}")
    if shutil.which("nvidia-smi") is None:
        raise SystemExit("nvidia-smi was not found")
    subprocess.run([
        sys.executable, str(ROOT / "scripts/check_xnavdp_runtime.py"),
        str(args.xnavdp_root), "--python", args.eval_python,
    ], check=True)


def gpu_free_mb(gpu: int) -> int:
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits", "-i", str(gpu)],
        check=True, text=True, capture_output=True,
    )
    return int(result.stdout.strip())


def wait_for_gpu(gpu: int, poll_seconds: int) -> None:
    last_message = 0.0
    while not STOP.is_set():
        free = gpu_free_mb(gpu)
        if free >= GPU_MIN_FREE_MB:
            return
        now = time.monotonic()
        if now - last_message >= 30:
            print(
                f"[wait] GPU {gpu}: free {free} MiB; need {GPU_MIN_FREE_MB} MiB",
                flush=True,
            )
            last_message = now
        STOP.wait(poll_seconds)
    raise RuntimeError("launch cancelled")


def register(process: subprocess.Popen) -> None:
    with ACTIVE_LOCK:
        ACTIVE.add(process)


def unregister(process: subprocess.Popen) -> None:
    with ACTIVE_LOCK:
        ACTIVE.discard(process)


def stop_process(process: subprocess.Popen) -> None:
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait()


def wait_health(port: int, server: subprocess.Popen, timeout: float = 120.0) -> None:
    deadline = time.monotonic() + timeout
    url = f"http://127.0.0.1:{port}/health"
    while time.monotonic() < deadline and not STOP.is_set():
        code = server.poll()
        if code is not None:
            raise RuntimeError(f"policy server exited early with code {code}")
        try:
            with urllib.request.urlopen(url, timeout=1) as response:
                if response.status == 200:
                    return
        except OSError:
            time.sleep(0.1)
    raise RuntimeError("policy server readiness timed out")


def server_environment(
    args: argparse.Namespace,
    spec: ServerSpec,
    server_gpu: int,
    cuda_library_path: str | None,
) -> dict[str, str]:
    env = os.environ.copy()
    env.update(spec.env)
    python_paths = []
    for value in (spec.env.get("PYTHONPATH"), str(ROOT), os.environ.get("PYTHONPATH")):
        if value:
            for path in value.split(os.pathsep):
                if path and path not in python_paths:
                    python_paths.append(path)
    env["PYTHONPATH"] = os.pathsep.join(python_paths)
    env.update({
        "CUDA_VISIBLE_DEVICES": str(server_gpu),
        "OMP_NUM_THREADS": str(args.cpu_threads_per_worker),
        "MKL_NUM_THREADS": str(args.cpu_threads_per_worker),
        "OPENBLAS_NUM_THREADS": str(args.cpu_threads_per_worker),
        "PXR_WORK_THREAD_LIMIT": str(args.cpu_threads_per_worker),
    })
    if cuda_library_path:
        env["LD_LIBRARY_PATH"] = os.pathsep.join(filter(None, (
            cuda_library_path, os.environ.get("LD_LIBRARY_PATH")
        )))
    return env


def policy_server_gpus(
    args: argparse.Namespace, slot: int, evaluator_gpu: int,
) -> list[int]:
    if args.server_gpus is None:
        return [evaluator_gpu]
    if len(args.jobs) == 1:
        return list(args.server_gpus)
    return [args.server_gpus[slot]]


def evaluator_environment(
    args: argparse.Namespace,
    gpu: int,
    optix_cache: Path,
) -> dict[str, str]:
    """Build the single Isaac evaluator environment for one resident scene."""
    env = os.environ.copy()
    # CUDA uses the one visible device (ordinal 0); Vulkan retains the physical
    # ordinal supplied separately in evaluator_command. Do not initialize other GPUs.
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    (cache_root() / "kit" / f"gpu_{gpu}").mkdir(parents=True, exist_ok=True)
    env.update({
        "PYTHONUNBUFFERED": "1",
        "OPTIX_CACHE_PATH": str(optix_cache),
        "PYTHONPATH": os.pathsep.join(filter(None, (
            str(args.xnavdp_root), str(ROOT), os.environ.get("PYTHONPATH"),
        ))),
        "OMP_NUM_THREADS": str(args.cpu_threads_per_worker),
        "MKL_NUM_THREADS": str(args.cpu_threads_per_worker),
        "OPENBLAS_NUM_THREADS": str(args.cpu_threads_per_worker),
    })
    acados_lib = Path(os.environ["ACADOS_SOURCE_DIR"]) / "lib"
    env["LD_LIBRARY_PATH"] = os.pathsep.join(filter(None, (
        str(acados_lib), os.environ.get("LD_LIBRARY_PATH"),
    )))
    return env


def evaluate_checkpoint(
    args: argparse.Namespace,
    target: CheckpointTarget,
    slot: int,
    gpu: int,
    port: int,
    job: SceneJob,
    eval_cmd: list[str],
    resident_scene_root: Path,
    evaluator: EvaluatorProcess,
    cuda_library_path: str | None,
    policy_gpu_pool: PolicyGpuPool,
) -> None:
    """Evaluate one checkpoint through an already-loaded scene process."""
    scene_root = target.run_root / "scenes" / job.split / job.name
    metric_path = scene_root / "metric.csv"
    if metric_is_complete(metric_path, job.episodes):
        return

    target_args = args_for_target(args, target)
    server_gpus = policy_server_gpus(args, slot, gpu)
    server_ports = [
        port + index * len(args.gpus) for index in range(len(server_gpus))
    ]
    specs = [
        build_server(target_args, server_port, ROOT, target.artifact)
        for server_port in server_ports
    ]
    run_id = f"{target.key}:{job.key}"
    scene_metadata = {
        "suite": args.suite["name"], "model": target.model,
        "scene": job.name, "split": job.split,
        "episodes": job.episodes, "scene_scale": job.scene_scale,
        "height_offset_m": job.height_offset_m,
        "rgb_num_samples": job.rgb_num_samples,
        "seed": args.seed,
        "num_envs": args.num_envs,
        "scene_dir": str(job.scene_dir),
        "navigation_file": str(job.navigation_file),
        "navigation_file_sha256": job.navigation_sha256,
        "episode_file": str(job.episode_file),
        "episode_file_sha256": job.episode_sha256,
        "checkpoint": str(target.checkpoint),
        "server_gpus": server_gpus,
        "server_ports": server_ports,
        "checkpoint_sha256": target.checkpoint_sha256,
        "model_config": (
            str(target.model_config) if target.model_config else None
        ),
        "model_config_sha256": target.model_config_sha256,
        "source_sha256": target.source_sha256,
        "artifact_sha256": target.artifact_sha256,
        "worker": slot,
        "evaluator_command": eval_cmd,
        "resident_scene_log": str(resident_scene_root / "eval.log"),
        "server_commands": [spec.command for spec in specs],
    }
    (scene_root / "run.json").write_text(
        json.dumps(scene_metadata, indent=2) + "\n"
    )
    print(
        f"[run] worker={slot} scene={job.key} model={target.model} "
        f"checkpoint={target.key}",
        flush=True,
    )
    adapter = get_adapter(target.model)
    with policy_gpu_pool.reserve(server_gpus, adapter.policy_gpu_slots):
        servers: list[subprocess.Popen] = []
        try:
            with contextlib.ExitStack() as stack:
                for index, (spec, server_gpu) in enumerate(zip(specs, server_gpus)):
                    server_log = stack.enter_context(
                        (scene_root / f"server-{index:02d}.log").open("w")
                    )
                    server = subprocess.Popen(
                        spec.command,
                        cwd=spec.cwd,
                        env=server_environment(
                            target_args, spec, server_gpu, cuda_library_path
                        ),
                        stdout=server_log,
                        stderr=subprocess.STDOUT,
                        start_new_session=True,
                    )
                    servers.append(server)
                    register(server)
                for server_port, server in zip(server_ports, servers):
                    wait_health(server_port, server)
                evaluator.send({
                    "command": "run",
                    "run_id": run_id,
                    "episodes": job.episodes,
                    "metric_path": str(metric_path),
                    "server_ports": server_ports,
                })
                done = evaluator.wait_for("done")
                if done.get("run_id") != run_id:
                    raise RuntimeError(
                        f"evaluator completed unexpected run {done!r}"
                    )
                if not metric_is_complete(metric_path, job.episodes):
                    raise RuntimeError(
                        f"evaluator produced an incomplete metric: {metric_path}"
                    )
        finally:
            for server in reversed(servers):
                stop_process(server)
                unregister(server)
    print(
        f"[done] worker={slot} scene={job.key} model={target.model} "
        f"checkpoint={target.key}",
        flush=True,
    )


def run_worker(
    args: argparse.Namespace,
    targets: list[CheckpointTarget],
    slot: int,
    gpu: int,
    session_root: Path,
    input_root: Path,
    prepared_scenes: dict[str, tuple[Path, Path]],
    jobs: queue.Queue[SceneJob],
    cuda_library_path: str | None,
    policy_gpu_pool: PolicyGpuPool,
) -> None:
    if args.launch_stagger and STOP.wait(slot * args.launch_stagger):
        raise RuntimeError("launch cancelled")
    port = args.port_base + slot
    worker_root = session_root / f"worker_{slot}"
    worker_root.mkdir(parents=True, exist_ok=True)
    worker_cache_root = cache_root()
    optix_cache = worker_cache_root / "optix" / f"gpu_{gpu}"
    optix_cache.mkdir(parents=True, exist_ok=True)
    metadata = {
        "model": args.model, "task": "pointgoal", "worker": slot,
        "precision": args.precision,
        "gpu": gpu,
        "server_gpus": policy_server_gpus(args, slot, gpu),
        "num_envs": args.num_envs or 1,
        "cpu_threads_per_worker": args.cpu_threads_per_worker,
        "evaluator_world_size": len(args.gpus),
        "gpu_poll_seconds": args.gpu_poll_seconds,
        "policy_gpu_slots": args.policy_gpu_slots,
        "optix_cache": str(optix_cache),
        "checkpoints": [
            target_identity(target) for target in targets
        ],
    }
    (worker_root / "run.json").write_text(json.dumps(metadata, indent=2) + "\n")

    evaluator: EvaluatorProcess | None = None
    try:
        eval_env = evaluator_environment(args, gpu, optix_cache)
        while not STOP.is_set():
            try:
                job = jobs.get_nowait()
            except queue.Empty:
                break
            resident_scene_root, eval_config = prepared_scenes[job.key]
            eval_cmd = evaluator_command(args, port, job, eval_config, gpu)
            pending_targets = [
                target for target in targets
                if not metric_is_complete(
                    target.run_root / "scenes" / job.split / job.name / "metric.csv",
                    job.episodes,
                )
            ]
            if not pending_targets:
                jobs.task_done()
                continue
            print(
                f"[load] worker={slot} scene={job.key} "
                f"checkpoints={len(pending_targets)}",
                flush=True,
            )
            try:
                wait_for_gpu(gpu, args.gpu_poll_seconds)
                evaluator = EvaluatorProcess(
                    eval_cmd,
                    cwd=input_root.parents[1],
                    env=eval_env,
                    log_path=resident_scene_root / "eval.log",
                    cancelled=STOP.is_set,
                )
                register(evaluator.process)
                ready = evaluator.wait_for("ready")
                if ready.get("scene") != job.name:
                    raise RuntimeError(
                        f"evaluator loaded {ready.get('scene')!r}, expected {job.name!r}"
                    )
                print(
                    f"[ready] worker={slot} scene={job.key} "
                    f"startup_seconds={ready['startup_seconds']:.3f}",
                    flush=True,
                )

                for target in pending_targets:
                    evaluate_checkpoint(
                        args, target, slot, gpu, port, job, eval_cmd,
                        resident_scene_root, evaluator, cuda_library_path,
                        policy_gpu_pool,
                    )

                evaluator.close()
                unregister(evaluator.process)
                evaluator = None
            finally:
                if evaluator is not None:
                    stop_process(evaluator.process)
                    unregister(evaluator.process)
                    evaluator = None
                jobs.task_done()
    finally:
        if evaluator is not None:
            stop_process(evaluator.process)
            unregister(evaluator.process)


def run_resident_worker(
    args: argparse.Namespace,
    slot: int,
    gpu: int,
    session_root: Path,
    input_root: Path,
    prepared_scenes: dict[str, tuple[Path, Path]],
    job: SceneJob,
    targets: queue.Queue[CheckpointTarget | None],
    messages: queue.Queue[ResidentMessage],
    cuda_library_path: str | None,
    policy_gpu_pool: PolicyGpuPool,
) -> None:
    """Own one scene for the lifetime of a checkpoint-queue service."""
    evaluator: EvaluatorProcess | None = None
    failed = False
    try:
        if args.launch_stagger and STOP.wait(slot * args.launch_stagger):
            return
        port = args.port_base + slot
        worker_root = session_root / f"worker_{slot}"
        worker_root.mkdir(parents=True, exist_ok=True)
        optix_cache = cache_root() / "optix" / f"gpu_{gpu}"
        optix_cache.mkdir(parents=True, exist_ok=True)
        resident_scene_root, eval_config = prepared_scenes[job.key]
        eval_cmd = evaluator_command(args, port, job, eval_config, gpu)
        (worker_root / "run.json").write_text(json.dumps({
            "service": "resident-scene",
            "task": "pointgoal",
            "worker": slot,
            "gpu": gpu,
            "server_gpus": policy_server_gpus(args, slot, gpu),
            "scene": job.key,
            "num_envs": args.num_envs,
            "checkpoint_queue": str(args.checkpoint_queue),
            "optix_cache": str(optix_cache),
        }, indent=2) + "\n")
        print(f"[load] worker={slot} scene={job.key} resident=true", flush=True)
        wait_for_gpu(gpu, args.gpu_poll_seconds)
        evaluator = EvaluatorProcess(
            eval_cmd,
            cwd=input_root.parents[1],
            env=evaluator_environment(args, gpu, optix_cache),
            log_path=resident_scene_root / "eval.log",
            cancelled=STOP.is_set,
        )
        register(evaluator.process)
        ready = evaluator.wait_for("ready")
        if ready.get("scene") != job.name:
            raise RuntimeError(
                f"evaluator loaded {ready.get('scene')!r}, expected {job.name!r}"
            )
        print(
            f"[ready] worker={slot} scene={job.key} "
            f"startup_seconds={ready['startup_seconds']:.3f}",
            flush=True,
        )
        messages.put(ResidentMessage("ready", job.key))

        while not STOP.is_set():
            try:
                target = targets.get(timeout=1.0)
            except queue.Empty:
                continue
            if target is None:
                break
            try:
                evaluate_checkpoint(
                    args, target, slot, gpu, port, job, eval_cmd,
                    resident_scene_root, evaluator, cuda_library_path,
                    policy_gpu_pool,
                )
                messages.put(ResidentMessage("done", job.key, target.key))
            finally:
                targets.task_done()
    except BaseException as error:
        failed = True
        messages.put(ResidentMessage("error", job.key, error=error))
        STOP.set()
    finally:
        if evaluator is not None:
            try:
                evaluator.close()
            except BaseException as error:
                if not failed and not STOP.is_set():
                    messages.put(ResidentMessage("error", job.key, error=error))
                    STOP.set()
            finally:
                unregister(evaluator.process)


def wait_resident_messages(
    messages: queue.Queue[ResidentMessage],
    kind: str,
    count: int,
    target_key: str | None = None,
) -> None:
    received: set[str] = set()
    while len(received) < count:
        try:
            message = messages.get(timeout=1.0)
        except queue.Empty:
            if STOP.is_set():
                raise RuntimeError("resident evaluation cancelled")
            continue
        if message.kind == "error":
            assert message.error is not None
            raise RuntimeError(
                f"resident scene {message.scene_key} failed"
            ) from message.error
        if message.kind != kind or message.target_key != target_key:
            raise RuntimeError(f"unexpected resident evaluator message: {message}")
        if message.scene_key in received:
            raise RuntimeError(
                f"duplicate resident evaluator message from {message.scene_key}"
            )
        received.add(message.scene_key)


def prepare_queued_target(
    args: argparse.Namespace,
    target: CheckpointTarget,
    session_root: Path,
    input_root: Path,
    selected_jobs: list[SceneJob],
) -> CheckpointTarget:
    target_args = args_for_target(args, target)
    spec = build_server(target_args, args.port_base, ROOT, target.artifact)
    missing = [
        str(path)
        for path in (spec.cwd, *spec.required_paths)
        if not path.exists()
    ]
    if missing:
        raise RuntimeError(
            f"ready model bundle cannot start {target.model}:\n  "
            + "\n  ".join(missing)
        )
    target.run_root.mkdir(parents=True, exist_ok=False)
    for job in selected_jobs:
        (target.run_root / "scenes" / job.split / job.name).mkdir(
            parents=True, exist_ok=False
        )
    (target.run_root / "run.json").write_text(
        json.dumps(
            checkpoint_run_metadata(args, target, input_root, selected_jobs),
            indent=2,
        ) + "\n"
    )
    metadata_path = session_root / "run.json"
    metadata = json.loads(metadata_path.read_text())
    metadata["checkpoints"].append(target_identity(target))
    metadata["checkpoint_paths"].append(str(target.checkpoint))
    metadata["models"] = sorted({
        identity["model"] for identity in metadata["checkpoints"]
    })
    incoming = metadata_path.with_name(f"run.json.incoming.{os.getpid()}")
    incoming.write_text(json.dumps(metadata, indent=2) + "\n")
    incoming.replace(metadata_path)
    return target


def queued_artifact(bundle: Path) -> tuple[str, ModelArtifact]:
    manifest_path = bundle / "artifact.json"
    try:
        manifest = json.loads(manifest_path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(
            f"ready model bundle has no valid artifact.json: {bundle}"
        ) from exc
    if set(manifest) != {"schema", "model"}:
        raise RuntimeError(
            "artifact.json must contain exactly 'schema' and 'model'"
        )
    if manifest["schema"] != "navbench-model-artifact-v1":
        raise RuntimeError(f"unsupported artifact schema: {manifest['schema']!r}")
    model = manifest["model"]
    if not isinstance(model, str) or model not in ADAPTERS:
        raise RuntimeError(f"unknown model adapter in artifact.json: {model!r}")
    checkpoint = bundle / "checkpoint.pt"
    model_config = (
        bundle / "configs" / "base.yaml" if model == "curvenav" else None
    )
    required = [checkpoint]
    if model_config is not None:
        required.extend((model_config, bundle / "src" / "curvenav"))
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise RuntimeError(
            f"ready checkpoint bundle is incomplete: {bundle}\n  "
            + "\n  ".join(missing)
        )
    artifact = ModelArtifact(
        checkpoint.resolve(), model_config.resolve() if model_config else None
    )
    return model, artifact


def run_resident_service(
    args: argparse.Namespace,
    initial_targets: list[CheckpointTarget],
    session_root: Path,
    input_root: Path,
    prepared_scenes: dict[str, tuple[Path, Path]],
    selected_jobs: list[SceneJob],
    cuda_library_path: str | None,
    policy_gpu_pool: PolicyGpuPool,
) -> None:
    """Keep selected scenes loaded and broadcast immutable checkpoints to them."""
    assert args.checkpoint_queue is not None
    args.checkpoint_queue.mkdir(parents=True, exist_ok=True)
    messages: queue.Queue[ResidentMessage] = queue.Queue()
    target_queues: list[queue.Queue[CheckpointTarget | None]] = [
        queue.Queue() for _ in selected_jobs
    ]
    futures: list[concurrent.futures.Future[None]] = []
    with concurrent.futures.ThreadPoolExecutor(
        max_workers=len(selected_jobs)
    ) as pool:
        for slot, job in enumerate(selected_jobs):
            futures.append(pool.submit(
                run_resident_worker,
                args,
                slot,
                args.gpus[slot],
                session_root,
                input_root,
                prepared_scenes,
                job,
                target_queues[slot],
                messages,
                cuda_library_path,
                policy_gpu_pool,
            ))
        try:
            wait_resident_messages(messages, "ready", len(selected_jobs))
            print(
                f"[resident] {len(selected_jobs)} scenes ready; "
                f"watching {args.checkpoint_queue}",
                flush=True,
            )
            known_digests = {
                target.artifact_sha256 for target in initial_targets
            }
            processed_bundles: set[Path] = set()
            completed_targets: list[CheckpointTarget] = []
            next_index = len(initial_targets)

            def evaluate_target(target: CheckpointTarget) -> None:
                for target_queue in target_queues:
                    target_queue.put(target)
                wait_resident_messages(
                    messages, "done", len(selected_jobs), target.key
                )
                merge_metrics(
                    target.run_root,
                    sum(job.episodes for job in selected_jobs),
                    args.eval_python,
                )
                completed_targets.append(target)
                merge_trajectory_diagnostics(
                    [item.run_root for item in completed_targets],
                    session_root,
                    args.eval_python,
                )
                print(f"[complete] {target.run_root / 'summary.csv'}", flush=True)

            for target in initial_targets:
                evaluate_target(target)

            while not STOP.is_set():
                discovered = False
                for bundle in sorted(args.checkpoint_queue.glob("*.ready")):
                    if not bundle.is_dir() or bundle in processed_bundles:
                        continue
                    model, artifact = queued_artifact(bundle)
                    target = build_checkpoint_target(
                        model, artifact, session_root, next_index
                    )
                    processed_bundles.add(bundle)
                    if target.artifact_sha256 in known_digests:
                        continue
                    target = prepare_queued_target(
                        args, target, session_root, input_root, selected_jobs,
                    )
                    known_digests.add(target.artifact_sha256)
                    next_index += 1
                    discovered = True
                    evaluate_target(target)
                if not discovered:
                    STOP.wait(2.0)
        finally:
            for target_queue in target_queues:
                target_queue.put(None)
            for future in futures:
                try:
                    future.result()
                except BaseException:
                    if not STOP.is_set():
                        raise


def execution_profile(num_envs: int) -> str:
    return (
        "pointgoal-v2-official-b16"
        if num_envs == OFFICIAL_NUM_ENVS else "pointgoal-v2-nonstandard-batch"
    )


def dry_run_plan(args: argparse.Namespace) -> None:
    workers = min(args.max_workers, len(args.gpus), len(args.jobs))
    plan = {
        "model": args.model,
        "models": [
            args.model, *(model for model, _ in args.additional_artifacts)
        ],
        "task": "pointgoal",
        "suite": args.suite["name"],
        "episodes_per_scene": args.jobs[0].episodes,
        "seed": args.seed,
        "precision": args.precision,
        "num_envs": args.num_envs,
        "gpus": args.gpus,
        "max_workers": args.max_workers,
        "server_gpus": args.server_gpus,
        "cpu_threads_per_worker": args.cpu_threads_per_worker,
        "gpu_poll_seconds": args.gpu_poll_seconds,
        "policy_gpu_slots": args.policy_gpu_slots,
        "profile": execution_profile(args.num_envs),
        "runtime_root": str(args.xnavdp_root) if args.xnavdp_root else None,
        "scene_root": str(args.scene_root),
        "scene_count": len(args.jobs),
        "episode_count": sum(job.episodes for job in args.jobs),
        "full_suite_scene_count": len(args.all_jobs),
        "shard_index": args.shard_index,
        "shard_count": args.shard_count,
        "shard_weights": args.shard_weights,
        "checkpoints": [
            *(str(path) for path in args.checkpoints),
            *(str(artifact.checkpoint)
              for _, artifact in args.additional_artifacts),
        ],
        "checkpoint_queue": (
            str(args.checkpoint_queue) if args.checkpoint_queue else None
        ),
        "scenes": [
            {
                "key": job.key, "episodes": job.episodes,
                "scene_dir": str(job.scene_dir),
                "episode_file": str(job.episode_file),
                "scene_scale": job.scene_scale,
                "height_offset_m": job.height_offset_m,
                "rgb_num_samples": job.rgb_num_samples,
            }
            for job in args.jobs
        ],
        "workers": [],
    }
    for slot in range(workers):
        server_gpus = policy_server_gpus(args, slot, args.gpus[slot])
        servers = []
        artifacts = [
            *((args.model, ModelArtifact(checkpoint, args.model_config))
              for checkpoint in args.checkpoints),
            *args.additional_artifacts,
        ]
        for model, artifact in artifacts:
            for shard, server_gpu in enumerate(server_gpus):
                server_port = args.port_base + slot + shard * len(args.gpus)
                target_args = argparse.Namespace(**vars(args))
                target_args.model = model
                spec = build_server(
                    target_args, server_port, ROOT, artifact,
                )
                servers.append({
                    "model": model,
                    "checkpoint": str(artifact.checkpoint),
                    "policy_shard": shard,
                    "gpu": server_gpu,
                    "port": server_port,
                    "server_cwd": str(spec.cwd),
                    "server_env": spec.env,
                    "server_command": spec.command,
                })
        plan["workers"].append({
            "worker": slot,
            "gpu": args.gpus[slot],
            "server_gpus": server_gpus,
            "servers": servers,
            "evaluator_example": evaluator_command(
                args, args.port_base + slot, args.jobs[slot % len(args.jobs)],
                Path("<RUN_DIR>") / "upstream-config.yaml", args.gpus[slot],
            ),
        })
    print(json.dumps(plan, indent=2))


def merge_metrics(
    run_root: Path,
    expected: int,
    python: str,
) -> None:
    subprocess.run([
        python, "-m", "navbench.metrics",
        "--root", str(run_root), "--output", str(run_root / "episodes.csv"),
        "--summary", str(run_root / "summary.csv"), "--expected", str(expected),
    ], check=True, cwd=ROOT)


def merge_trajectory_diagnostics(
    run_roots: list[Path],
    output_root: Path,
    python: str,
) -> None:
    command = [python, "-m", "navbench.trajectory_metrics"]
    for run_root in run_roots:
        command.extend(("--root", str(run_root)))
    command.extend(("--output-dir", str(output_root)))
    subprocess.run(command, check=True, cwd=ROOT)


def terminate_all(*_unused: object) -> None:
    STOP.set()
    with ACTIVE_LOCK:
        processes = list(ACTIVE)
    for process in processes:
        stop_process(process)


def checkpoint_run_metadata(
    args: argparse.Namespace,
    target: CheckpointTarget,
    input_root: Path,
    selected_jobs: list[SceneJob],
) -> dict[str, object]:
    return {
        "suite": args.suite["name"],
        "episodes_per_scene": selected_jobs[0].episodes,
        "suite_manifest": str(args.suite_manifest),
        "suite_definition_sha256": suite_definition_sha256(args.suite),
        "model": target.model,
        "precision": args.precision,
        "cpu_threads_per_worker": args.cpu_threads_per_worker,
        "gpu_poll_seconds": args.gpu_poll_seconds,
        "num_envs": args.num_envs,
        "execution_profile": execution_profile(args.num_envs),
        "runtime": "x-navdp-upstream-with-raw-transport",
        "runtime_root": str(args.xnavdp_root),
        "runtime_revision": args.suite["source"]["code_revision"],
        "input_root": str(input_root),
        "checkpoint": str(target.checkpoint),
        "checkpoint_sha256": target.checkpoint_sha256,
        "model_config": (
            str(target.model_config) if target.model_config else None
        ),
        "model_config_sha256": target.model_config_sha256,
        "source_sha256": target.source_sha256,
        "artifact_sha256": target.artifact_sha256,
        "seed": args.seed,
        "scene_count": len(selected_jobs),
        "full_suite_scene_count": len(args.all_jobs),
        "episode_count": sum(job.episodes for job in selected_jobs),
        "shard_index": args.shard_index,
        "shard_count": args.shard_count,
        "shard_weights": args.shard_weights,
        "scene_keys": [job.key for job in selected_jobs],
        "gpus": args.gpus,
        "server_gpus": args.server_gpus,
    }


def main() -> None:
    args = parse_args()
    resolve_paths(args)
    if args.check_assets:
        missing = missing_assets(args.jobs)
        invalid = [
            *common_asset_errors(args),
            *invalid_episode_assets(args.jobs),
            *invalid_navigation_assets(args.suite, args.jobs),
        ]
        if missing or invalid:
            raise SystemExit(
                f"Invalid or missing {len(missing) + len(invalid)} suite assets:\n  "
                + "\n  ".join([*(str(path) for path in missing), *invalid])
            )
        print(
            f"[ok] {len(args.jobs)} scenes and "
            f"{sum(job.episodes for job in args.jobs)} episodes"
        )
        return
    if args.dry_run:
        dry_run_plan(args)
        return
    if args.suite.get("status") != "frozen":
        raise SystemExit(
            "Suite is not frozen: the official X-NavDP manifest is required"
        )
    validate(args)

    signal.signal(signal.SIGINT, terminate_all)
    signal.signal(signal.SIGTERM, terminate_all)
    selected_jobs = list(args.jobs)
    if args.resume_root is None:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        if args.shard_count > 1:
            timestamp += f"_shard{args.shard_index}of{args.shard_count}"
        service_name = (
            "resident" if args.checkpoint_queue is not None
            else "model-set" if args.additional_artifacts
            else args.model
        )
        session_root = args.output_root / args.suite["name"] / service_name / timestamp
        session_root.mkdir(parents=True, exist_ok=False)
    else:
        session_root = args.resume_root

    targets = build_initial_targets(args, session_root)
    if args.resume_root is not None:
        validate_resume_root(args, session_root, targets)
        input_view = session_root / "upstream-runtime" / "data" / "scenes"
        if not input_view.is_dir():
            raise SystemExit(f"Resume root has no upstream input view: {input_view}")
        input_root = input_view.resolve()
        prepared_scenes = load_prepared_scene_roots(selected_jobs, session_root)
        prepare_checkpoint_roots(args, targets, resume=True)
    else:
        input_root = prepare_upstream_inputs(args, session_root)
        prepared_scenes = prepare_scene_roots(args, session_root, input_root)
        prepare_checkpoint_roots(args, targets, resume=False)

    pending_jobs = [
        job for job in selected_jobs
        if any(
            not metric_is_complete(
                target.run_root / "scenes" / job.split / job.name / "metric.csv",
                job.episodes,
            )
            for target in targets
        )
    ]

    if args.resume_root is not None:
        (session_root / "resume.json").write_text(json.dumps({
            "resumed_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "pending_scenes": [job.key for job in pending_jobs],
            "active_gpus": args.gpus[:min(
                args.max_workers, len(args.gpus), len(pending_jobs)
            )],
            "max_workers": args.max_workers,
        }, indent=2) + "\n")
    else:
        (session_root / "run.json").write_text(json.dumps({
            "suite": args.suite["name"],
            "suite_definition_sha256": suite_definition_sha256(args.suite),
            "service": (
                "resident-scene" if args.checkpoint_queue is not None
                else "fixed-model-set" if args.additional_artifacts
                else "single-model"
            ),
            "model": args.model if args.checkpoint_queue is None else None,
            "models": sorted({target.model for target in targets}),
            "seed": args.seed,
            "num_envs": args.num_envs,
            "runtime_revision": args.suite["source"]["code_revision"],
            "shard_index": args.shard_index,
            "shard_count": args.shard_count,
            "shard_weights": args.shard_weights,
            "scene_keys": [job.key for job in selected_jobs],
            "episode_count": sum(job.episodes for job in selected_jobs),
            "gpus": args.gpus,
            "max_workers": args.max_workers,
            "server_gpus": args.server_gpus,
            "model_config_sha256": (
                sha256_file(args.model_config) if args.model_config else None
            ),
            "checkpoints": [
                target_identity(target) for target in targets
            ],
            "checkpoint_paths": [str(target.checkpoint) for target in targets],
            "checkpoint_queue": (
                str(args.checkpoint_queue) if args.checkpoint_queue else None
            ),
            "resident_scene_evaluator": "navbench.scene_evaluator",
        }, indent=2) + "\n")
        for target in targets:
            (target.run_root / "run.json").write_text(
                json.dumps(
                    checkpoint_run_metadata(args, target, input_root, selected_jobs),
                    indent=2,
                ) + "\n"
            )

    if not pending_jobs and args.checkpoint_queue is None:
        for target in targets:
            merge_metrics(
                target.run_root,
                sum(job.episodes for job in selected_jobs),
                args.eval_python,
            )
            print(f"[complete] {target.run_root / 'summary.csv'}")
        return

    cuda_library_path = python_cuda_library_path(args.server_python)
    policy_gpus = args.server_gpus or args.gpus
    policy_gpu_pool = PolicyGpuPool(policy_gpus, args.policy_gpu_slots)
    if args.checkpoint_queue is not None:
        run_resident_service(
            args,
            targets,
            session_root,
            input_root,
            prepared_scenes,
            selected_jobs,
            cuda_library_path,
            policy_gpu_pool,
        )
        return

    workers = min(args.max_workers, len(args.gpus), len(pending_jobs))
    job_queue: queue.Queue[SceneJob] = queue.Queue()
    for job in pending_jobs:
        job_queue.put(job)

    futures = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        for slot in range(workers):
            futures.append(pool.submit(
                run_worker, args, targets, slot,
                args.gpus[slot], session_root, input_root, prepared_scenes,
                job_queue, cuda_library_path, policy_gpu_pool,
            ))
        try:
            for future in concurrent.futures.as_completed(futures):
                future.result()
        except Exception:
            STOP.set()
            terminate_all()
            raise

    for target in targets:
        merge_metrics(
            target.run_root,
            sum(job.episodes for job in selected_jobs),
            args.eval_python,
        )
        print(f"[complete] {target.run_root / 'summary.csv'}")
    merge_trajectory_diagnostics(
        [target.run_root for target in targets],
        session_root,
        args.eval_python,
    )


if __name__ == "__main__":
    main()
