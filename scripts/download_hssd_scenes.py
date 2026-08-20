#!/usr/bin/env python3
"""Download only the HSSD assets referenced by selected static scenes."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import time

from huggingface_hub import HfApi, hf_hub_download


REPOSITORY = "hssd/hssd-hab"
REVISION = "4369cb9876214c7fbebcf552eb532380e4d287e4"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_root", type=Path)
    parser.add_argument("scene_ids", nargs="+")
    return parser


def _repository_files(output_root: Path) -> list[str]:
    inventory_path = output_root / "repository_files.json"
    if inventory_path.exists():
        inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
        if inventory.get("revision") == REVISION:
            return inventory["files"]
    files = [
        item.rfilename
        for item in HfApi().dataset_info(
            REPOSITORY, revision=REVISION, files_metadata=False
        ).siblings
    ]
    inventory_path.parent.mkdir(parents=True, exist_ok=True)
    inventory_path.write_text(
        json.dumps({"revision": REVISION, "files": files}), encoding="utf-8"
    )
    return files


def main() -> None:
    args = _parser().parse_args()
    output_root = args.output_root.expanduser().resolve()
    repository_files = _repository_files(output_root)
    paths_by_name: dict[str, list[str]] = {}
    for path in repository_files:
        paths_by_name.setdefault(Path(path).name, []).append(path)

    selected = {
        "hssd-hab.scene_dataset_config.json",
        "semantics/hssd-hab_semantic_lexicon.json",
    }
    for scene_id in args.scene_ids:
        scene_path = f"scenes/{scene_id}.scene_instance.json"
        local_scene = Path(
            hf_hub_download(
                REPOSITORY,
                scene_path,
                repo_type="dataset",
                revision=REVISION,
                local_dir=output_root,
            )
        )
        scene = json.loads(local_scene.read_text(encoding="utf-8"))
        selected.update(
            {
                scene_path,
                f"stages/{scene_id}.glb",
                f"stages/{scene_id}.stage_config.json",
                f"semantics/scenes/{scene_id}.semantic_config.json",
            }
        )
        handles = {item["template_name"] for item in scene["object_instances"]}
        for handle in handles:
            for suffix in (".object_config.json", ".glb", ".collider.glb"):
                matches = paths_by_name.get(f"{handle}{suffix}", [])
                if not matches and suffix != ".collider.glb":
                    raise RuntimeError(f"HSSD asset is missing: {handle}{suffix}")
                if matches:
                    selected.add(matches[0])

    def download(path: str) -> str:
        for attempt in range(3):
            try:
                return hf_hub_download(
                    REPOSITORY,
                    path,
                    repo_type="dataset",
                    revision=REVISION,
                    local_dir=output_root,
                )
            except Exception:
                if attempt == 2:
                    raise
                time.sleep(2**attempt)
        raise AssertionError("unreachable")

    with ThreadPoolExecutor(max_workers=8) as executor:
        list(executor.map(download, sorted(selected)))
    manifest_path = output_root / "download_manifest.json"
    downloaded_scenes = set(args.scene_ids)
    if manifest_path.exists():
        previous = json.loads(manifest_path.read_text(encoding="utf-8"))
        if previous.get("revision") != REVISION:
            raise RuntimeError("existing HSSD assets use a different revision")
        downloaded_scenes.update(previous.get("scenes", []))
    manifest = {
        "repository": REPOSITORY,
        "revision": REVISION,
        "scenes": sorted(downloaded_scenes),
        "selected_files_last_request": len(selected),
        "output_root": str(output_root),
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
