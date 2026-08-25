"""One-time calibrated depth preparation for CurveNav route datasets."""

import json
import os
import shutil
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np

from curvenav.config import DataConfig
from curvenav.data.depth import (
    CANONICAL_INTRINSICS,
    SAND_INTRINSICS,
    depth_camera_contract,
    preprocess_metric_depth,
)


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
        if (height, width) != (CANONICAL_INTRINSICS.height, CANONICAL_INTRINSICS.width):
            raise ValueError("packed depth must use the canonical camera")
        normalized = preprocess_metric_depth(
            image.astype(np.float32) / depth_units_per_m,
            source_intrinsics=SAND_INTRINSICS,
            maximum_m=max_depth_m,
        )
        packed[frame_index] = normalized.astype(np.float16)
    packed.flush()
    del packed
    os.replace(temporary, destination)
    return run_key, len(source_paths)


def _prepare_hssd_run(
    task: tuple[str, str, str, int, int, float],
) -> tuple[str, dict[str, object]]:
    route_id, source_string, destination_string, height, width, max_depth_m = task
    cv2.setNumThreads(0)
    source = Path(source_string)
    destination = Path(destination_string)
    destination.parent.mkdir(parents=True, exist_ok=True)
    depth = np.load(source, mmap_mode="r")
    if (
        depth.dtype != np.float32
        or depth.ndim != 3
        or depth.shape[1:] != (height, width)
        or len(depth) < 3
    ):
        raise ValueError(f"invalid HSSD depth tensor: {source}")
    temporary = destination.with_suffix(".tmp.npy")
    packed = np.lib.format.open_memmap(
        temporary,
        mode="w+",
        dtype=np.float16,
        shape=depth.shape,
    )
    for frame_index in range(len(depth)):
        packed[frame_index] = preprocess_metric_depth(
            np.asarray(depth[frame_index]),
            source_intrinsics=CANONICAL_INTRINSICS,
            maximum_m=max_depth_m,
        ).astype(np.float16)
    packed.flush()
    del packed
    os.replace(temporary, destination)
    return route_id, {
        "file": destination.as_posix(),
        "frames": len(depth),
    }


def prepare_depth_cache(
    dataset_root: str | Path,
    *,
    data: DataConfig,
    depth_units_per_m: float,
    workers: int,
) -> dict[str, object]:
    """Pack every SanD run into the exact FP16 tensor precision seen by AMP."""
    if workers < 1:
        raise ValueError("workers must be positive")
    height, width = data.image_height, data.image_width
    max_depth_m = data.max_depth_m
    source_root = Path(dataset_root).resolve()
    destination_root = depth_cache_root(source_root, height, width)
    building_root = destination_root.with_name(destination_root.name + ".building")
    if destination_root.exists() or building_root.exists():
        raise FileExistsError(f"SanD depth cache already exists: {destination_root}")
    tasks = []
    excluded_runs: dict[str, str] = {}
    for dataset_directory in sorted(source_root.glob("dataset_*")):
        if not dataset_directory.is_dir():
            raise ValueError(
                f"SanD dataset entry is not a directory: {dataset_directory}"
            )
        for run_directory in sorted(dataset_directory.glob("run_*")):
            run_key = f"{dataset_directory.name}/{run_directory.name}"
            depth_directory = run_directory / "depth"
            source_paths = tuple(sorted(depth_directory.glob("depth_*.png")))
            if not source_paths:
                excluded_runs[run_key] = "no_depth_frames"
                continue
            expected_names = tuple(
                f"depth_{frame_index:04d}.png"
                for frame_index in range(len(source_paths))
            )
            if tuple(path.name for path in source_paths) != expected_names:
                raise ValueError(f"non-contiguous depth sequence: {depth_directory}")
            destination = (
                building_root / dataset_directory.name / f"{run_directory.name}.npy"
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

    building_root.mkdir(parents=True)
    try:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            run_counts = dict(executor.map(_prepare_run, tasks))
        manifest: dict[str, object] = {
            "height": height,
            "width": width,
            "dtype": "float16",
            "depth_units_per_m": depth_units_per_m,
            "max_depth_m": max_depth_m,
            "source_camera": {
                "image_height": SAND_INTRINSICS.height,
                "image_width": SAND_INTRINSICS.width,
                "focal_x_px": SAND_INTRINSICS.fx,
                "focal_y_px": SAND_INTRINSICS.fy,
                "principal_x_px": SAND_INTRINSICS.cx,
                "principal_y_px": SAND_INTRINSICS.cy,
            },
            "target_camera": depth_camera_contract(data),
            "total_frames": sum(run_counts.values()),
            "runs": dict(sorted(run_counts.items())),
            "excluded_runs": dict(sorted(excluded_runs.items())),
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


def prepare_hssd_depth_cache(
    dataset_root: str | Path,
    *,
    data: DataConfig,
    workers: int,
) -> dict[str, object]:
    """Normalize each generated continuous HSSD route into FP16 depth."""
    if workers < 1:
        raise ValueError("workers must be positive")
    height, width = data.image_height, data.image_width
    max_depth_m = data.max_depth_m
    if (height, width) != (CANONICAL_INTRINSICS.height, CANONICAL_INTRINSICS.width):
        raise ValueError("packed depth must use the canonical camera")
    source_root = Path(dataset_root).resolve()
    destination_root = hssd_depth_cache_root(source_root, height, width)
    building_root = destination_root.with_name(destination_root.name + ".building")
    if destination_root.exists() or building_root.exists():
        raise FileExistsError(f"HSSD depth cache already exists: {destination_root}")
    dataset_manifest = json.loads(
        (source_root / "dataset_manifest.json").read_text(encoding="utf-8")
    )
    if dataset_manifest.get("schema") != "curvenav_hssd_expert_routes":
        raise ValueError("HSSD dataset schema does not match CurveNav")
    records = [
        json.loads(line)
        for line in (source_root / "routes.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    if len(records) != int(dataset_manifest.get("routes", -1)):
        raise ValueError("HSSD manifest and route index disagree")
    tasks = []
    for index, record in enumerate(records):
        route_id = str(record["route_id"])
        route_directory = source_root / str(record["route_directory"])
        relative_destination = Path(record["split"]) / f"{index:05d}.npy"
        destination = building_root / relative_destination
        tasks.append(
            (
                route_id,
                str(route_directory / "depth_m.npy"),
                str(destination),
                height,
                width,
                max_depth_m,
            )
        )

    building_root.mkdir(parents=True)
    try:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            runs = dict(executor.map(_prepare_hssd_run, tasks))
        for value in runs.values():
            value["file"] = str(Path(value["file"]).relative_to(building_root))
        manifest: dict[str, object] = {
            "height": height,
            "width": width,
            "dtype": "float16",
            "source_dtype": "float32_metric_m",
            "max_depth_m": max_depth_m,
            "target_camera": depth_camera_contract(data),
            "total_frames": sum(int(item["frames"]) for item in runs.values()),
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
