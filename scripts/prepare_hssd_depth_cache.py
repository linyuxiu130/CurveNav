#!/usr/bin/env python3
"""Pack both HSSD dataset splits for direct CurveNav training."""

import argparse
import json
from pathlib import Path

from curvenav.config import DataConfig
from curvenav.data.depth_cache import depth_cache_root, prepare_depth_cache


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset_root", type=Path)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    contract = DataConfig()
    summary: dict[str, object] = {"schema_version": 1, "splits": {}}
    splits = summary["splits"]
    assert isinstance(splits, dict)

    for split in ("train", "validation"):
        split_root = args.dataset_root / split
        manifest = prepare_depth_cache(
            split_root,
            height=contract.image_height,
            width=contract.image_width,
            depth_units_per_m=contract.depth_units_per_m,
            max_depth_m=contract.max_depth_m,
            workers=args.workers,
        )
        splits[split] = {
            "cache": str(
                depth_cache_root(
                    split_root,
                    contract.image_height,
                    contract.image_width,
                ).resolve()
            ),
            "runs": len(manifest["runs"]),
            "frames": manifest["total_frames"],
        }

    output_path = args.dataset_root / "depth_cache_summary.json"
    output_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
