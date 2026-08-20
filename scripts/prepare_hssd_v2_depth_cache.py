#!/usr/bin/env python3
"""Prepare CurveNav's required packed HSSD v2 depth frame bank."""

import argparse
import json
from pathlib import Path

from curvenav.config_io import load_config
from curvenav.data.depth_cache import (
    hssd_depth_cache_root,
    prepare_hssd_v2_depth_cache,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    config = load_config(args.config)
    manifest = prepare_hssd_v2_depth_cache(
        args.dataset,
        height=config.data.image_height,
        width=config.data.image_width,
        max_depth_m=config.data.max_depth_m,
        workers=args.workers,
    )
    print(
        json.dumps(
            {
                "cache": str(
                    hssd_depth_cache_root(
                        args.dataset,
                        config.data.image_height,
                        config.data.image_width,
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
