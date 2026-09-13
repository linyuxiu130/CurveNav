#!/usr/bin/env python3
"""Check the pinned X-NavDP metric runtime and raw transport bridge."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]


def load_versions() -> dict[str, str]:
    values = {}
    for line in (ROOT / "config/evaluator-versions.env").read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            key, value = line.split("=", 1)
            values[key] = value
    return values


VERSIONS = load_versions()
REVISION = VERSIONS["XNAVDP_REVISION"]
PINNED = {
    "eval/config/eval_pointgoal/wheeled_internscene_home.yaml":
        "cc4b5c36218cf61333f0a2836f5b2768a3b1c62923f7d84240af1832da9044ef",
    "eval/config/eval_pointgoal/wheeled_internscene_commercial.yaml":
        "3e4a298f0e650738aee79a287db42a321522d7a55c4660849d7e858ed387cf2a",
    "eval/scripts/evaluate_pointgoal.py":
        "38c942ecfd2fe5c7c397ca8c71c03e6655d764fc3752a6a3eff7d87a24561bb4",
    "src/environment/robots/dingo_config.py":
        "3b7da016006eb6c4f536387e1b020825fafcdca3dd7c45fbcdf21cf11359bf27",
    "src/environment/tasks/event_utils.py":
        "6805d20eb09358fd5f22d0192c4f3931d8fbacd21df0900b613360dd1d98a904",
    "src/environment/tasks/observation_utils.py":
        "1fd53663aba07618d5a0ae909850b062bc661c7f21e9ce1d7dcbe55028d708f6",
    "src/environment/tasks/reward_utils.py":
        "413233f58eb6ca2360a03c0f70fa796f44919c00b6c59bafa34edc024fbd6350",
    "src/environment/tasks/terminal_utils.py":
        "c6ab4f4cc392e339b83eceb8efbe631a6a4b32228557d7f8c14782ddda896044",
    "src/environment/env_wrapper.py":
        "903666368ccbf3c3fa4b5f972df5ea7d084262188a6d3535a7f34cf76c462f50",
    "src/environment/scenes/default_scene.py":
        "1fe53196df705bc996d5ff087c03d61159ab58bcf54b7897008298969b8d1cf6",
    "src/environment/wheeled_tasks.py":
        "ced3f3bbb196277cef12fddc3f4216a15abcd2909e42fef601849c4d9df937f6",
    "src/utils/mpc_tracking.py":
        "1c99cb781efc2241f6d8df5c66d4ae070927236b29d20c00340bb890171c33f1",
}


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("runtime_root", type=Path)
    parser.add_argument("--python", default=sys.executable)
    args = parser.parse_args()
    runtime = args.runtime_root.resolve()
    checkout = runtime.parents[1]
    actual_revision = subprocess.run(
        ("git", "-C", str(checkout), "rev-parse", "HEAD"),
        check=True, text=True, capture_output=True,
    ).stdout.strip()
    if actual_revision != REVISION:
        raise SystemExit(f"runtime revision {actual_revision}, expected {REVISION}")
    for relative, expected in PINNED.items():
        path = runtime / relative
        if not path.is_file() or digest(path) != expected:
            raise SystemExit(f"upstream runtime file differs: {path}")
    for name in ("client_utils.py", "policy_agent.py", "policy_server.py", "policy_backbone.py", "policy_network_embodiment.py"):
        bridge = runtime / "eval/src" / name
        expected = ROOT / "baselines/x-navdp/eval/src" / name
        if not bridge.is_file() or bridge.read_bytes() != expected.read_bytes():
            raise SystemExit(f"runtime policy bridge differs: {name}")
    cuda_tag = VERSIONS["PYTORCH_CUDA_TAG"]
    cuda_version = f"{cuda_tag[2:-1]}.{cuda_tag[-1]}"
    dependency_code = (
        "from importlib.metadata import version; import torch; "
        f"expected={{'isaacsim':'{VERSIONS['ISAACSIM_DIST_VERSION']}',"
        f"'setuptools':'{VERSIONS['SETUPTOOLS_VERSION']}',"
        f"'click':'{VERSIONS['ISAACSIM_CLICK_VERSION']}',"
        f"'typing_extensions':'{VERSIONS['ISAACSIM_TYPING_EXTENSIONS_VERSION']}',"
        f"'isaaclab':'{VERSIONS['ISAACLAB_VERSION']}',"
        f"'isaaclab-rl':'{VERSIONS['ISAACLAB_RL_VERSION']}',"
        f"'rsl-rl-lib':'{VERSIONS['RSL_RL_VERSION']}',"
        f"'tensordict':'{VERSIONS['TENSORDICT_VERSION']}',"
        f"'warp-lang':'{VERSIONS['WARP_VERSION']}',"
        f"'acados-template':'{VERSIONS['ACADOS_TEMPLATE_VERSION']}',"
        f"'torch':'{VERSIONS['TORCH_VERSION']}',"
        f"'torchvision':'{VERSIONS['TORCHVISION_VERSION']}'}}; "
        "actual={name:version(name).split('+', 1)[0] for name in expected}; "
        f"assert actual==expected, (actual, expected); assert torch.version.cuda=='{cuda_version}', torch.version.cuda; "
        "import isaaclab, acados_template"
    )
    dependency_check = subprocess.run(
        (args.python, "-c", dependency_code),
        text=True, capture_output=True,
    )
    if dependency_check.returncode:
        detail = dependency_check.stderr.strip().splitlines()[-1]
        raise SystemExit(f"upstream runtime dependency check failed: {detail}")
    print(f"[ok] X-NavDP {REVISION} runtime and raw-depth bridge")


if __name__ == "__main__":
    main()
