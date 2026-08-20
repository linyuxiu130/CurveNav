"""Download only the small X-NavDP metadata needed for offline route planning."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
from pathlib import Path
import time
from urllib.parse import quote
from urllib.request import Request, urlopen

from curvenav.data_generation.scene_catalog import KNOWN_BAD_SCENES


DEFAULT_REPOSITORY = "InternRobotics/X-NavDP"
_USER_AGENT = "CurveNav-expert-metadata/1.0"


def _get_bytes(url: str, *, timeout_s: float = 120.0) -> bytes:
    request = Request(url, headers={"User-Agent": _USER_AGENT})
    with urlopen(request, timeout=timeout_s) as response:
        return response.read()


def _download_one(url: str, destination: Path, expected_size: int) -> str:
    if destination.is_file() and destination.stat().st_size == expected_size:
        return "cached"
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.part")
    for attempt in range(3):
        try:
            payload = _get_bytes(url)
            if len(payload) != expected_size:
                raise IOError(
                    f"size mismatch for {url}: expected {expected_size}, got {len(payload)}"
                )
            temporary.write_bytes(payload)
            os.replace(temporary, destination)
            return "downloaded"
        except Exception:
            temporary.unlink(missing_ok=True)
            if attempt == 2:
                raise
            time.sleep(1.0 * (attempt + 1))
    raise AssertionError("unreachable")


def download_training_metadata(
    output_root: str | Path,
    *,
    repository: str = DEFAULT_REPOSITORY,
    workers: int = 8,
) -> dict[str, object]:
    """Fetch official-train navigable maps and safe point-goal pairs only."""

    if workers <= 0:
        raise ValueError("workers must be positive")
    root = Path(output_root).expanduser().resolve()
    split_url = f"https://huggingface.co/{repository}/raw/main/scene_split.json"
    split_payload = _get_bytes(split_url)
    split = json.loads(split_payload)
    split_path = root / "scene_split.json"
    split_path.parent.mkdir(parents=True, exist_ok=True)
    split_path.write_bytes(split_payload)

    api_url = (
        f"https://huggingface.co/api/models/{repository}/tree/main/"
        "navigation_metadata?recursive=true&expand=false&limit=1000"
    )
    tree = json.loads(_get_bytes(api_url))
    file_sizes = {
        item["path"]: int(item["size"])
        for item in tree
        if item.get("type") == "file"
    }
    jobs: list[tuple[str, Path, int]] = []
    scene_count = 0
    for scene_type, subset in (
        ("home", "internscenes_home"),
        ("commercial", "internscenes_commercial"),
    ):
        for scene_id in split[f"{scene_type}_train"]:
            if scene_id in KNOWN_BAD_SCENES:
                continue
            scene_count += 1
            for relative in (
                f"navigation_metadata/{subset}/esdf/{scene_id}/navigable.ply",
                f"navigation_metadata/{subset}/pointgoal_start_pair/{scene_id}/"
                "pointgoal_start_pair_samples_safe.npy",
            ):
                if relative not in file_sizes:
                    raise FileNotFoundError(f"remote metadata tree is missing {relative}")
                url = f"https://huggingface.co/{repository}/resolve/main/{quote(relative)}"
                jobs.append((url, root / relative, file_sizes[relative]))

    counts = {"downloaded": 0, "cached": 0}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(_download_one, url, destination, size): destination
            for url, destination, size in jobs
        }
        for future in as_completed(futures):
            status = future.result()
            counts[status] += 1

    source = {
        "repository": repository,
        "revision": "main",
        "official_split": "train",
        "scene_count": scene_count,
        "file_count": len(jobs),
        "downloaded_file_count": counts["downloaded"],
        "cached_file_count": counts["cached"],
        "total_bytes": sum(size for _, _, size in jobs),
        "excluded_known_bad_scenes": sorted(KNOWN_BAD_SCENES),
    }
    source_path = root / "source.json"
    temporary = source_path.with_name(f".{source_path.name}.{os.getpid()}.part")
    temporary.write_text(
        json.dumps(source, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, source_path)
    return source


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Download official-train X-NavDP navigation metadata without large USD assets"
    )
    parser.add_argument("output_root")
    parser.add_argument("--repository", default=DEFAULT_REPOSITORY)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args(argv)
    result = download_training_metadata(
        args.output_root,
        repository=args.repository,
        workers=args.workers,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
