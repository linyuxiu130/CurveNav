#!/usr/bin/env python3
"""Prepare CurveNav's normalized HSSD expert-route depth cache."""

import argparse
import json
from pathlib import Path

from curvenav.config_io import load_config
from curvenav.data.depth_cache import hssd_depth_cache_root, prepare_hssd_depth_cache


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    data = load_config(args.config).data
    manifest = prepare_hssd_depth_cache(
        args.dataset,
        height=data.image_height,
        width=data.image_width,
        max_depth_m=data.max_depth_m,
        workers=args.workers,
    )
    print(
        json.dumps(
            {
                "cache": str(
                    hssd_depth_cache_root(
                        args.dataset, data.image_height, data.image_width
                    ).resolve()
                ),
                "routes": len(manifest["runs"]),
                "frames": manifest["total_frames"],
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
