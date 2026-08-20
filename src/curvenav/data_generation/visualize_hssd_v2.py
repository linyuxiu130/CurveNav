"""Render BEV previews from the episodes actually present in a v2 dataset."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
from typing import Any

import numpy as np


DIFFICULTY_COLORS = {
    "open": "#2b8cbe",
    "turning": "#7b3294",
    "detour": "#e66101",
    "narrow": "#d73027",
}


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]


def render_bev(dataset_root: Path) -> list[dict[str, Any]]:
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    dataset_root = dataset_root.resolve()
    episodes = _read_jsonl(dataset_root / "episodes.jsonl")
    by_scene: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for episode in episodes:
        by_scene[(episode["split"], episode["scene_id"])].append(episode)

    records = []
    for (split, scene_id), scene_episodes in sorted(by_scene.items()):
        scene_dir = dataset_root / split / f"dataset_hssd_{scene_id}"
        with np.load(scene_dir / "navigation_grid.npz") as values:
            free = values["free"]
            origin = values["origin_xy"].astype(np.float64)
            cell_size = float(values["cell_size_m"])
        x0, y0 = origin + cell_size / 2.0
        x1 = x0 + free.shape[0] * cell_size
        y1 = y0 + free.shape[1] * cell_size
        figure, axis = plt.subplots(figsize=(10, 8), constrained_layout=True)
        axis.imshow(
            (~free).T,
            origin="lower",
            extent=[x0, x1, y0, y1],
            cmap="gray_r",
            interpolation="nearest",
            alpha=0.9,
        )
        present_difficulties = set()
        for episode in sorted(scene_episodes, key=lambda item: item["run_id"]):
            with np.load(scene_dir / episode["run_id"] / "route.npz") as route:
                world_xy = route["world_xy"]
            difficulty = episode["difficulty"]
            present_difficulties.add(difficulty)
            color = DIFFICULTY_COLORS[difficulty]
            axis.plot(world_xy[:, 0], world_xy[:, 1], color=color, linewidth=2.0, alpha=0.9)
            axis.scatter(*world_xy[0], color="#1a9850", s=26, zorder=3)
            axis.scatter(*world_xy[-1], color="#d73027", marker="*", s=48, zorder=3)
        legend = [
            Line2D([0], [0], color=DIFFICULTY_COLORS[name], lw=2.5, label=name)
            for name in DIFFICULTY_COLORS
            if name in present_difficulties
        ]
        legend.extend(
            [
                Line2D([0], [0], marker="o", color="w", markerfacecolor="#1a9850", label="start", markersize=7),
                Line2D([0], [0], marker="*", color="w", markerfacecolor="#d73027", label="final task goal", markersize=10),
            ]
        )
        axis.legend(handles=legend, loc="upper right", fontsize=8)
        axis.set_title(
            f"CurveNav v2 {split} / HSSD {scene_id}: {len(scene_episodes)} actual episodes\n"
            "black = obstacle or non-navigable robot-center region"
        )
        axis.set_xlabel("world x [m]")
        axis.set_ylabel("world z [m]")
        axis.set_aspect("equal")
        output_path = scene_dir / "bev_routes.png"
        figure.savefig(output_path, dpi=170)
        plt.close(figure)
        records.append(
            {
                "split": split,
                "scene_id": scene_id,
                "episodes": len(scene_episodes),
                "path": str(output_path),
            }
        )
    audit_dir = dataset_root / "audit"
    audit_dir.mkdir(exist_ok=True)
    (audit_dir / "bev_index.json").write_text(
        json.dumps(records, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return records


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset_root", type=Path)
    args = parser.parse_args(argv)
    records = render_bev(args.dataset_root)
    print(json.dumps({"scenes": len(records), "episodes": sum(record["episodes"] for record in records)}, sort_keys=True))


if __name__ == "__main__":
    main()
