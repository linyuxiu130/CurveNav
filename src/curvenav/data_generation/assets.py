"""Download the exact HSSD assets required by CurveNav."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
from pathlib import Path
import shutil
import struct
import time
from typing import Any
from urllib.request import Request, urlopen
from urllib.error import HTTPError, URLError


HSSD_REPOSITORY = "hssd/hssd-hab"
HSSD_COMMIT = "4369cb9876214c7fbebcf552eb532380e4d287e4"
HF_ENDPOINT = os.environ.get("HF_ENDPOINT", "https://huggingface.co").rstrip("/")
# Keep enough connections that a stalled large GLB does not block the queue.
DOWNLOAD_WORKERS = 32


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _repository_paths() -> list[str]:
    url = f"{HF_ENDPOINT}/api/datasets/{HSSD_REPOSITORY}/revision/{HSSD_COMMIT}"
    with urlopen(Request(url, headers={"User-Agent": "CurveNav/1.0"}), timeout=60) as response:
        repository = json.load(response)
    return [item["rfilename"] for item in repository["siblings"]]


def _download(root: Path, path: str) -> None:
    destination = root / path
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".partial")
    url = (
        f"{HF_ENDPOINT}/datasets/{HSSD_REPOSITORY}/resolve/"
        f"{HSSD_COMMIT}/{path}"
    )
    for attempt in range(4):
        try:
            with urlopen(Request(url, headers={"User-Agent": "CurveNav/1.0"}), timeout=60) as response, temporary.open("wb") as output:
                shutil.copyfileobj(response, output)
            os.replace(temporary, destination)
            return
        except (URLError, TimeoutError, ConnectionError) as error:
            if attempt == 3 or (isinstance(error, HTTPError) and error.code not in {429, 500, 502, 503, 504}):
                raise
            time.sleep(2 ** attempt)


def _download_paths(root: Path, paths: set[str]) -> None:
    pending = [path for path in sorted(paths) if not (root / path).is_file()]
    print(f"Assets: {len(paths) - len(pending)}/{len(paths)} present; downloading {len(pending)} with {DOWNLOAD_WORKERS} workers", flush=True)
    with ThreadPoolExecutor(max_workers=DOWNLOAD_WORKERS) as executor:
        futures = [executor.submit(_download, root, path) for path in pending]
        for count, future in enumerate(as_completed(futures), 1):
            future.result()
            if count % 100 == 0 or count == len(pending):
                print(f"Downloaded {count}/{len(pending)} pending assets", flush=True)


def selected_asset_paths(
    root: Path,
    repository_paths: list[str],
    scene_ids: list[str],
) -> set[str]:
    paths_by_name: dict[str, list[str]] = {}
    for path in repository_paths:
        paths_by_name.setdefault(Path(path).name, []).append(path)
    selected = {
        "hssd-hab.scene_dataset_config.json",
        "semantics/hssd-hab_semantic_lexicon.json",
    }
    for scene_id in scene_ids:
        selected.update(
            {
                f"scenes/{scene_id}.scene_instance.json",
                f"stages/{scene_id}.glb",
                f"stages/{scene_id}.stage_config.json",
                f"semantics/scenes/{scene_id}.semantic_config.json",
            }
        )
        scene = _read_json(root / f"scenes/{scene_id}.scene_instance.json")
        for instance in scene["object_instances"]:
            handle = instance["template_name"]
            configurations = paths_by_name.get(f"{handle}.object_config.json", [])
            meshes = paths_by_name.get(f"{handle}.glb", [])
            if len(configurations) != 1 or len(meshes) != 1:
                raise ValueError(f"HSSD object assets are ambiguous or missing: {handle}")
            selected.update((configurations[0], meshes[0]))
            colliders = paths_by_name.get(f"{handle}.collider.glb", [])
            if len(colliders) > 1:
                raise ValueError(f"HSSD collider asset is ambiguous: {handle}")
            selected.update(colliders)
    return selected


def _validate_glb(path: Path) -> None:
    with path.open("rb") as stream:
        header = stream.read(12)
    if len(header) != 12 or header[:4] != b"glTF":
        raise ValueError(f"invalid GLB header: {path}")
    _, file_length = struct.unpack_from("<II", header, 4)
    if file_length != path.stat().st_size:
        raise ValueError(f"truncated GLB: {path}")


def validate_assets(root: Path, scene_ids: list[str]) -> int:
    """Verify every frozen asset referenced by the selected HSSD scenes."""
    repository = _read_json(root / "repository_files.json")
    if repository.get("revision") != HSSD_COMMIT:
        raise ValueError("HSSD repository index does not match the frozen commit")
    selected = selected_asset_paths(root, repository["files"], scene_ids)
    missing = sorted(path for path in selected if not (root / path).is_file())
    if missing:
        raise FileNotFoundError(f"HSSD selected assets are missing: {missing}")
    for path in sorted(selected):
        asset = root / path
        if path.endswith(".json"):
            _read_json(asset)
        elif path.endswith(".glb"):
            _validate_glb(asset)
    return len(selected)


def download_assets(config_path: Path) -> dict[str, Any]:
    config_path = config_path.resolve()
    config = _read_json(config_path)
    project_root = config_path.parent.parent
    root = (project_root / config["asset_root"]).resolve()
    building = root.with_name(root.name + ".building")
    if root.exists():
        raise FileExistsError(root)
    building.mkdir(parents=True, exist_ok=True)
    scene_ids = [scene["scene_id"] for scene in config["selected_scenes"]]
    index = building / "repository_files.json"
    if index.exists():
        repository = _read_json(index)
        if repository["revision"] != HSSD_COMMIT:
            raise ValueError("HSSD repository index does not match the frozen commit")
        repository_paths = repository["files"]
    else:
        repository_paths = _repository_paths()
    (building / "repository_files.json").write_text(
        json.dumps(
            {"revision": HSSD_COMMIT, "files": repository_paths},
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    scene_paths = {f"scenes/{scene_id}.scene_instance.json" for scene_id in scene_ids}
    _download_paths(building, scene_paths)
    selected = selected_asset_paths(building, repository_paths, scene_ids)
    _download_paths(building, selected)
    validate_assets(building, scene_ids)
    manifest = {
        "repository": HSSD_REPOSITORY,
        "commit": HSSD_COMMIT,
        "scenes": sorted(scene_ids),
        "files": len(selected),
    }
    (building / "download_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(building, root)
    return manifest


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    args = parser.parse_args(argv)
    print(json.dumps(download_assets(args.config), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
