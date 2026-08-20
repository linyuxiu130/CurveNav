#!/usr/bin/env python3
"""Map Habitat no-return depth pixels to CurveNav's 8 m depth limit."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path

import cv2


DEPTH_LIMIT_MM = 8000


def _repair_frame(path: Path) -> tuple[int, int]:
    depth = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    missing = depth == 0
    replaced = int(missing.sum())
    if replaced:
        depth[missing] = DEPTH_LIMIT_MM
        if not cv2.imwrite(str(path), depth):
            raise RuntimeError(f"failed to write repaired depth frame: {path}")
    return replaced, int((depth >= DEPTH_LIMIT_MM).sum())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset_root", type=Path)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    root = args.dataset_root.expanduser().resolve()
    repaired_runs = 0
    repaired_pixels = 0
    repaired_frames = 0
    for metadata_path in sorted(root.glob("*/dataset_hssd_*/run_*/metadata.json")):
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        frames = sorted((metadata_path.parent / "depth").glob("*.png"))
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            counts = list(executor.map(_repair_frame, frames))
        replaced = sum(item[0] for item in counts)
        if replaced:
            repaired_runs += 1
            repaired_frames += sum(item[0] > 0 for item in counts)
            repaired_pixels += replaced
        depth = metadata["depth"]
        depth["invalid_depth_fraction_max"] = max(
            depth["invalid_depth_fraction_max"],
            depth.pop("raw_invalid_depth_fraction_max", 0.0),
        )
        pixels_per_frame = 360 * 640
        depth["depth_at_limit_fraction_mean"] = sum(item[1] for item in counts) / (
            len(counts) * pixels_per_frame
        )
        metadata_path.write_text(
            json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8"
        )
    print(
        json.dumps(
            {
                "repaired_runs": repaired_runs,
                "repaired_frames": repaired_frames,
                "repaired_pixels": repaired_pixels,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
