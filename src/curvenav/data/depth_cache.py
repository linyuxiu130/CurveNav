"""One-time packed depth preparation for the fixed CurveNav SanD route."""

import json
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import shutil

import cv2
import numpy as np

from curvenav.data.depth import preprocess_metric_depth


CACHE_SCHEMA_VERSION = 2
HSSD_CACHE_SCHEMA_VERSION = 1


def depth_cache_root(dataset_root: str | Path, height: int, width: int) -> Path:
    return Path(dataset_root) / f"curvenav_depth_{height}x{width}_float16"


def hssd_depth_cache_root(dataset_root: str | Path, height: int, width: int) -> Path:
    return Path(dataset_root) / f"curvenav_hssd_depth_{height}x{width}_float16"


def _prepare_run(
    task: tuple[str, tuple[str, ...], str, int, int, float, float],
) -> tuple[str, int]:
    (
        run_key,
        source_paths,
        destination_string,
        height,
        width,
        depth_units_per_m,
        max_depth_m,
    ) = task
    cv2.setNumThreads(0)
    destination = Path(destination_string)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".tmp.npy")
    packed = np.lib.format.open_memmap(
        temporary,
        mode="w+",
        dtype=np.float16,
        shape=(len(source_paths), height, width),
    )
    for frame_index, source_path in enumerate(source_paths):
        image = cv2.imread(source_path, cv2.IMREAD_UNCHANGED)
        if image is None or image.ndim != 2 or image.dtype != np.uint16:
            raise ValueError(f"invalid uint16 depth image: {source_path}")
        if image.shape != (height, width):
            image = cv2.resize(
                image,
                (width, height),
                interpolation=cv2.INTER_NEAREST,
            )
        normalized = image.astype(np.float32) / depth_units_per_m
        normalized = np.clip(normalized, 0, max_depth_m) / max_depth_m
        packed[frame_index] = normalized.astype(np.float16)
    packed.flush()
    del packed
    os.replace(temporary, destination)
    return run_key, len(source_paths)


def prepare_depth_cache(
    dataset_root: str | Path,
    height: int,
    width: int,
    depth_units_per_m: float,
    max_depth_m: float,
    workers: int,
) -> dict[str, object]:
    """Pack every SanD run into the exact FP16 tensor precision seen by AMP."""
    if workers < 1:
        raise ValueError("workers must be positive")
    source_root = Path(dataset_root).resolve()
    destination_root = depth_cache_root(source_root, height, width)
    tasks = []
    for dataset_directory in sorted(source_root.glob("dataset_*")):
        if not dataset_directory.is_dir():
            continue
        for run_directory in sorted(dataset_directory.glob("run_*")):
            depth_directory = run_directory / "depth"
            source_paths = tuple(sorted(depth_directory.glob("depth_*.png")))
            if not source_paths:
                continue
            expected_names = tuple(
                f"depth_{frame_index:04d}.png"
                for frame_index in range(len(source_paths))
            )
            if tuple(path.name for path in source_paths) != expected_names:
                raise ValueError(f"non-contiguous depth sequence: {depth_directory}")
            run_key = f"{dataset_directory.name}/{run_directory.name}"
            destination = (
                destination_root / dataset_directory.name / f"{run_directory.name}.npy"
            )
            tasks.append(
                (
                    run_key,
                    tuple(str(path) for path in source_paths),
                    str(destination),
                    height,
                    width,
                    depth_units_per_m,
                    max_depth_m,
                )
            )
    if not tasks:
        raise ValueError(f"no SanD depth sequences found under {source_root}")

    with ProcessPoolExecutor(max_workers=workers) as executor:
        run_counts = dict(executor.map(_prepare_run, tasks))
    manifest: dict[str, object] = {
        "schema_version": CACHE_SCHEMA_VERSION,
        "height": height,
        "width": width,
        "dtype": "float16",
        "depth_units_per_m": depth_units_per_m,
        "max_depth_m": max_depth_m,
        "total_frames": sum(run_counts.values()),
        "runs": dict(sorted(run_counts.items())),
    }
    destination_root.mkdir(parents=True, exist_ok=True)
    temporary_manifest = destination_root / "manifest.tmp.json"
    temporary_manifest.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary_manifest, destination_root / "manifest.json")
    return manifest


def _prepare_hssd_episode(
    task: tuple[str, str, str, int, int, float, int, str],
) -> tuple[str, dict[str, object]]:
    (
        run_key,
        source_string,
        destination_string,
        height,
        width,
        max_depth_m,
        expected_frames,
        source_sha256,
    ) = task
    cv2.setNumThreads(0)
    source = np.load(source_string, mmap_mode="r")
    if source.dtype != np.float32 or source.ndim != 3 or len(source) != expected_frames:
        raise ValueError(f"invalid HSSD physical depth array: {source_string}")
    destination = Path(destination_string)
    destination.parent.mkdir(parents=True, exist_ok=True)
    packed = np.lib.format.open_memmap(
        destination,
        mode="w+",
        dtype=np.float16,
        shape=(expected_frames, height, width),
    )
    for frame_index in range(expected_frames):
        packed[frame_index] = preprocess_metric_depth(
            source[frame_index],
            height=height,
            width=width,
            maximum_m=max_depth_m,
        ).astype(np.float16)
    packed.flush()
    del packed
    return run_key, {
        "frames": expected_frames,
        "source_depth_sha256": source_sha256,
    }


def prepare_hssd_v2_depth_cache(
    dataset_root: str | Path,
    *,
    height: int,
    width: int,
    max_depth_m: float,
    workers: int,
) -> dict[str, object]:
    """Build the only training/inference frame bank for physical HSSD v2 depth."""
    if workers < 1:
        raise ValueError("workers must be positive")
    source_root = Path(dataset_root).resolve()
    destination_root = hssd_depth_cache_root(source_root, height, width)
    building_root = destination_root.with_name(destination_root.name + ".building")
    if destination_root.exists() or building_root.exists():
        raise FileExistsError(f"HSSD packed depth cache already exists: {destination_root}")
    tasks = []
    for split in ("train", "validation"):
        for episode_dir in sorted((source_root / split).glob("dataset_hssd_*/run_*")):
            metadata = json.loads((episode_dir / "metadata.json").read_text())
            relative = episode_dir.relative_to(source_root)
            run_key = relative.as_posix()
            tasks.append(
                (
                    run_key,
                    str(episode_dir / "depth_m.npy"),
                    str(building_root / relative.parent / f"{relative.name}.npy"),
                    height,
                    width,
                    max_depth_m,
                    int(metadata["frames"]),
                    str(metadata["depth_sha256"]),
                )
            )
    if not tasks:
        raise ValueError(f"no HSSD v2 episodes found under {source_root}")
    building_root.mkdir(parents=True)
    try:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            runs = dict(executor.map(_prepare_hssd_episode, tasks))
        manifest: dict[str, object] = {
            "schema_version": HSSD_CACHE_SCHEMA_VERSION,
            "dataset_schema_version": "curvenav_hssd_v2.0",
            "height": height,
            "width": width,
            "dtype": "float16",
            "max_depth_m": max_depth_m,
            "total_frames": sum(int(value["frames"]) for value in runs.values()),
            "runs": dict(sorted(runs.items())),
        }
        (building_root / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(building_root, destination_root)
        return manifest
    except BaseException:
        shutil.rmtree(building_root, ignore_errors=True)
        raise
