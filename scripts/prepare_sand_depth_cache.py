#!/usr/bin/env python3
"""Prepare CurveNav's required packed SanD depth asset."""

import argparse
import json
from pathlib import Path

from curvenav.config_io import load_config
from curvenav.data.depth_cache import depth_cache_root, prepare_depth_cache


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    config = load_config(args.config)
    roots = dict.fromkeys(
        source.root
        for source in (
            *config.data.training_sources,
            *config.data.validation_sources,
        )
    )
    for root in roots:
        manifest = prepare_depth_cache(
            root,
            height=config.data.image_height,
            width=config.data.image_width,
            depth_units_per_m=config.data.depth_units_per_m,
            max_depth_m=config.data.max_depth_m,
            workers=args.workers,
        )
        print(
            json.dumps(
                {
                    "cache": str(
                        depth_cache_root(
                            root,
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
