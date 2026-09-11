#!/usr/bin/env python3
"""Install the immutable X-NavDP navigation meshes for the 40 eval scenes."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import sys

from huggingface_hub import HfApi, hf_hub_download


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from navbench.episodes import sha256_file  # noqa: E402


REPOSITORY = "InternRobotics/X-NavDP"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--suite-manifest", type=Path, default=ROOT / "suites/pointgoal-v2.json"
    )
    parser.add_argument("--staging-root", type=Path, required=True)
    parser.add_argument("--install-root", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    suite = json.loads(args.suite_manifest.read_text())
    source = suite["source"]
    if source.get("asset_repository") != REPOSITORY:
        raise ValueError("suite navigation repository differs from the downloader")
    revision = source["asset_revision"]
    info = HfApi().model_info(
        REPOSITORY, revision=revision, files_metadata=True
    )
    if info.sha != revision:
        raise ValueError("Hugging Face did not resolve the pinned navigation revision")
    expected_sizes = {
        sibling.rfilename: sibling.size for sibling in info.siblings
    }
    source_split = Path(hf_hub_download(REPOSITORY, "scene_split.json", revision=revision))
    split = json.loads(source_split.read_text())

    target = args.install_root / "navigation_metadata" if args.install_root else None
    if target is not None and target.is_dir():
        provenance_path = target / "PROVENANCE.json"
        provenance = json.loads(provenance_path.read_text())
        if (
            provenance.get("repository") != REPOSITORY
            or provenance.get("revision") != revision
            or provenance.get("suite") != suite["name"]
            or provenance.get("scene_count") != 40
            or len(provenance.get("files", [])) != 40
        ):
            raise ValueError(f"installed metadata at {target} has invalid provenance")
        expected_installed = {provenance_path.resolve()}
        for row in provenance["files"]:
            installed = args.install_root / row["path"]
            if not installed.is_file() or sha256_file(installed) != row["sha256"]:
                raise ValueError(f"installed metadata differs: {installed}")
            expected_installed.add(installed.resolve())
        actual_installed = {
            path.resolve() for path in target.rglob("*") if path.is_file()
        }
        if actual_installed != expected_installed:
            raise ValueError(f"installed metadata at {target} contains extra files")
        print(f"[cached] {target.resolve()}")
        return

    requested = []
    for domain, config in suite["splits"].items():
        expected_scenes = split[f"{domain}_eval"]
        if config["scenes"] != expected_scenes:
            raise ValueError(
                f"{domain} suite scenes differ from canonical X-NavDP eval split"
            )
        for scene_name in config["scenes"]:
            base = f"navigation_metadata/internscenes_{domain}"
            requested.append(
                f"{base}/esdf/{scene_name}/navigable.ply",
            )

    missing = [path for path in requested if path not in expected_sizes]
    if missing:
        raise FileNotFoundError(
            "canonical X-NavDP revision is missing: " + ", ".join(missing)
        )
    stage = args.staging_root / f"x-navdp-eval-navigation-{revision[:12]}"
    stage.mkdir(parents=True, exist_ok=True)
    rows = []
    for index, relative in enumerate(requested, start=1):
        source = Path(hf_hub_download(REPOSITORY, relative, revision=revision))
        destination = stage / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        actual_size = destination.stat().st_size
        if actual_size != expected_sizes[relative]:
            raise ValueError(
                f"{relative}: expected {expected_sizes[relative]} bytes, "
                f"found {actual_size}"
            )
        rows.append({
            "path": relative,
            "bytes": actual_size,
            "sha256": sha256_file(destination),
        })
        print(f"[{index}/{len(requested)}] {relative}", flush=True)

    provenance = {
        "schema": "x-navdp-eval-navigation-metadata-v2",
        "repository": REPOSITORY,
        "revision": revision,
        "suite": suite["name"],
        "scene_count": sum(len(config["scenes"]) for config in suite["splits"].values()),
        "scene_split_sha256": sha256_file(source_split),
        "episode_bundle_sha256": suite["episode_contract"]["episode_bundle_sha256"],
        "files": rows,
    }
    provenance_path = stage / "navigation_metadata" / "PROVENANCE.json"
    provenance_path.write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n"
    )
    print(f"[ready] {stage.resolve()}")
    if target is not None:
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            raise FileExistsError(f"refusing to replace existing {target}")
        os.replace(stage / "navigation_metadata", target)
        print(f"[installed] {target.resolve()}")


if __name__ == "__main__":
    main()
