"""Minimal, deterministic index for the public SanD trajectory dataset."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np


class SandRunIndex:
    """Index SanD runs without importing the upstream diffusion training stack."""

    def __init__(
        self,
        dataset_root: str,
        *,
        sequence_length: int,
        frame_skip: int,
        min_gap: int,
        max_gap: int,
        samples_per_epoch: int,
        image_size: tuple[int, int],
        depth_scale: float,
        max_depth: float,
        split_mode: str,
        train_ratio: float,
        random_seed: int,
    ) -> None:
        root = Path(dataset_root)
        if not root.is_dir():
            raise ValueError(f"dataset root not found: {root}")
        if split_mode not in {"all", "train", "val"}:
            raise ValueError("split_mode must be 'all', 'train', or 'val'")
        if not 0.0 < train_ratio < 1.0:
            raise ValueError("train_ratio must be between zero and one")

        self.dataset_root = str(root)
        self.sequence_length = sequence_length
        self.frame_skip = frame_skip
        self.min_gap = min_gap
        self.max_gap = max_gap
        self.samples_per_epoch = samples_per_epoch
        self.image_size = image_size
        self.depth_scale = depth_scale
        self.max_depth = max_depth
        self.split_mode = split_mode
        self.train_ratio = train_ratio
        self.random_seed = random_seed

        self.run_info: dict[str, dict[str, object]] = {}
        self.scene_runs: dict[str, list[str]] = {}
        self._index_runs(root)
        if split_mode != "all":
            self._apply_split()
        if not self.run_info:
            raise ValueError(f"SanD {split_mode} split contains no valid runs: {root}")
        total_runs = sum(len(runs) for runs in self.scene_runs.values())
        self.scene_weights = {
            scene: len(runs) / total_runs for scene, runs in self.scene_runs.items()
        }

    def _index_runs(self, root: Path) -> None:
        scenes = sorted(
            path for path in root.iterdir() if path.is_dir() and path.name.startswith("dataset_")
        )
        if not scenes:
            raise ValueError(f"no SanD dataset_* directories found in {root}")
        for scene in scenes:
            run_ids: list[str] = []
            for run_dir in sorted(scene.glob("run_*")):
                if not run_dir.is_dir():
                    continue
                paths = {
                    "traj_xyz": run_dir / "traj_xyz.npy",
                    "traj_yaw": run_dir / "traj_yaw.npy",
                    "traj_pitch": run_dir / "traj_pitch.npy",
                }
                depth_dir = run_dir / "depth"
                if not depth_dir.is_dir() or not all(path.is_file() for path in paths.values()):
                    continue
                trajectory = {
                    name: np.load(path) for name, path in paths.items()
                }
                length = len(trajectory["traj_xyz"])
                depth_count = sum(1 for _ in depth_dir.glob("*.png"))
                if (
                    length < self.min_gap + 1
                    or len(trajectory["traj_yaw"]) != length
                    or len(trajectory["traj_pitch"]) != length
                    or depth_count != length
                ):
                    continue
                run_id = f"{scene.name}_{run_dir.name}"
                self.run_info[run_id] = {
                    "length": length,
                    **trajectory,
                    "dataset_name": scene.name,
                    "run_dir": run_dir.name,
                }
                run_ids.append(run_id)
            if run_ids:
                self.scene_runs[scene.name] = run_ids

    def _apply_split(self) -> None:
        selected: set[str] = set()
        threshold = self.train_ratio * 10_000
        for run_id in self.run_info:
            digest = hashlib.md5(
                f"{run_id}_{self.random_seed}".encode(), usedforsecurity=False
            ).hexdigest()
            is_train = int(digest, 16) % 10_000 < threshold
            if (self.split_mode == "train" and is_train) or (
                self.split_mode == "val" and not is_train
            ):
                selected.add(run_id)
        self.run_info = {
            run_id: value for run_id, value in self.run_info.items() if run_id in selected
        }
        self.scene_runs = {
            scene: [run_id for run_id in run_ids if run_id in selected]
            for scene, run_ids in self.scene_runs.items()
            if any(run_id in selected for run_id in run_ids)
        }

    def __len__(self) -> int:
        return self.samples_per_epoch

    def _sample_scene(self) -> str:
        scenes = tuple(self.scene_weights)
        probabilities = tuple(self.scene_weights.values())
        return str(np.random.choice(scenes, p=probabilities))
