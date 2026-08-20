"""Independent quality gate for serialized expert episode manifests."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
import math
from pathlib import Path

import numpy as np

from curvenav.data_generation.occupancy import build_navigation_grid


def _path_length(path: np.ndarray) -> float:
    return float(np.linalg.norm(np.diff(path, axis=0), axis=1).sum())


def audit_manifest_bundle(
    manifest_dir: str | Path,
    *,
    manifest_name: str = "manifest.jsonl",
) -> dict[str, object]:
    root = Path(manifest_dir).expanduser().resolve()
    summary = json.loads((root / "summary.json").read_text(encoding="utf-8"))
    rows = [
        json.loads(line)
        for line in (root / manifest_name).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    planner = summary["planner"]
    grid_cell_size = float(summary["manifest"]["grid_cell_size_m"])
    grouped: defaultdict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["navigable_ply"])].append(row)

    failures: list[dict[str, object]] = []
    seen_ids: set[str] = set()
    for navigable_ply, episodes in grouped.items():
        grid = build_navigation_grid(navigable_ply, cell_size_m=grid_cell_size)
        for episode in episodes:
            episode_id = str(episode["episode_id"])
            reasons: list[str] = []
            if episode_id in seen_ids:
                reasons.append("duplicate episode_id")
            seen_ids.add(episode_id)
            if episode.get("official_split") != "train":
                reasons.append("non-training official split")
            path = np.asarray(episode["path_xy"], dtype=np.float64)
            start = np.asarray(episode["start_xy"], dtype=np.float64)
            goal = np.asarray(episode["goal_xy"], dtype=np.float64)
            if path.ndim != 2 or path.shape[1] != 2 or len(path) < 2:
                reasons.append("invalid path shape")
            else:
                if not np.allclose(path[0], start, atol=2e-5):
                    reasons.append("path start mismatch")
                if not np.allclose(path[-1], goal, atol=2e-5):
                    reasons.append("path goal mismatch")
                if not grid.path_is_safe(
                    path,
                    minimum_clearance_m=float(planner["minimum_clearance_m"]),
                    sample_step_m=float(planner["validation_step_m"]),
                ):
                    reasons.append("serialized path fails safety check")
                stored_length = float(episode["metrics"]["length_m"])
                if not math.isclose(_path_length(path), stored_length, abs_tol=2e-3):
                    reasons.append("serialized path length mismatch")
            metric_values = [float(value) for value in episode["metrics"].values()]
            if not all(math.isfinite(value) for value in metric_values):
                reasons.append("non-finite route metric")
            if (
                float(episode["metrics"]["reference_length_ratio"])
                > float(planner["maximum_safe_detour_ratio"]) + 1e-6
            ):
                reasons.append("detour budget exceeded")
            if float(episode["metrics"]["maximum_curvature"]) > float(
                planner["maximum_curvature"]
            ) + 1e-6:
                reasons.append("maximum curvature exceeded")
            if float(episode["metrics"]["curvature_p95"]) > float(
                planner["maximum_curvature_p95"]
            ) + 1e-6:
                reasons.append("p95 curvature exceeded")
            if reasons:
                failures.append({"episode_id": episode_id, "reasons": reasons})

    report: dict[str, object] = {
        "manifest": str((root / manifest_name).resolve()),
        "episode_count": len(rows),
        "scene_count": len(grouped),
        "failure_count": len(failures),
        "passed": not failures,
        "by_scene_type": dict(sorted(Counter(str(row["scene_type"]) for row in rows).items())),
        "by_difficulty": dict(sorted(Counter(str(row["difficulty"]) for row in rows).items())),
        "failures": failures,
    }
    return report


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Audit every serialized CurveNav expert route")
    parser.add_argument("manifest_dir")
    parser.add_argument("--manifest-name", default="manifest.jsonl")
    args = parser.parse_args(argv)
    report = audit_manifest_bundle(args.manifest_dir, manifest_name=args.manifest_name)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
