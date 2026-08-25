#!/usr/bin/env python3
"""Prepare CurveNav's required packed SanD depth asset."""

import argparse
import json
from pathlib import Path

from curvenav.data.depth_cache import depth_cache_root, prepare_depth_cache
from curvenav.config_io import load_config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    data = load_config(args.config).data
    manifest = prepare_depth_cache(
        args.dataset,
        height=data.image_height,
        width=data.image_width,
        depth_units_per_m=1000.0,
        max_depth_m=data.max_depth_m,
        workers=args.workers,
    )
    print(
        json.dumps(
            {
                "cache": str(
                    depth_cache_root(
                        args.dataset, data.image_height, data.image_width
                    ).resolve()
                ),
                "runs": len(manifest["runs"]),
                "frames": manifest["total_frames"],
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
