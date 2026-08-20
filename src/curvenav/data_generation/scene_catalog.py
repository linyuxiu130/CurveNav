"""Leakage-safe discovery and internal splitting of X-NavDP scene metadata."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import hashlib
import json
from pathlib import Path
from typing import Iterable


_SCENE_LAYOUTS = {
    "home": ("internscenes_home", "scenes_home"),
    "commercial": ("internscenes_commercial", "scenes_commercial"),
}

# Inherited from the X-NavDP release loader: these scenes have known asset or
# navigation metadata problems and should not enter a stable collection run.
KNOWN_BAD_SCENES = {
    "MWHLEPQKTIFZIAABAAAAAAA8_usd",
    "MWAX5JYKTKJZ2AABAAAAAAQ8_usd",
}


@dataclass(frozen=True)
class SceneRecord:
    """Files and split metadata required to plan one scene."""

    scene_id: str
    scene_type: str
    official_split: str
    internal_split: str
    family_id: str
    navigable_ply: str
    pointgoal_npy: str
    usd_path: str | None = None

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def scene_family_id(scene_id: str) -> str:
    """Return the scan/building family used for group-disjoint validation.

    InternScenes identifiers have a 16-character scan prefix followed by an
    8-character sub-scene identifier.  Unknown names conservatively become
    their own family instead of being merged accidentally.
    """

    stem = scene_id.removesuffix("_usd")
    return stem[:-8] if len(stem) == 24 else stem


def _load_official_split(path: Path) -> dict[str, list[str]]:
    if not path.is_file():
        raise FileNotFoundError(
            f"official scene split is required to prevent benchmark leakage: {path}"
        )
    data = json.loads(path.read_text(encoding="utf-8"))
    required = {"home_train", "home_eval", "commercial_train", "commercial_eval"}
    missing = sorted(required.difference(data))
    if missing:
        raise ValueError(f"scene split is missing keys: {missing}")
    return {key: list(value) for key, value in data.items()}


def _find_usd(data_root: Path, subset: str, scenes_dir: str, scene_id: str) -> str | None:
    scene_dir = data_root / subset / scenes_dir / scene_id
    for name in (
        "start_result_navigation.usd",
        "start_result_raw.usd",
        "start_result_interaction.usd",
    ):
        candidate = scene_dir / name
        if candidate.is_file():
            return str(candidate.resolve())
    return None


def discover_training_scenes(
    data_root: str | Path,
    *,
    scene_split_file: str | Path | None = None,
) -> list[SceneRecord]:
    """Discover only official training scenes with complete navigation metadata.

    Cluttered-easy/hard scenes are intentionally absent here: in Scene-N1 they
    are benchmark scenes.  Training on their geometry would invalidate the
    intended held-out NavDP evaluation.
    """

    root = Path(data_root).expanduser().resolve()
    split_path = (
        Path(scene_split_file).expanduser().resolve()
        if scene_split_file is not None
        else root / "scene_split.json"
    )
    split = _load_official_split(split_path)
    metadata_root = root / "navigation_metadata"
    records: list[SceneRecord] = []

    for scene_type, (subset, scenes_dir) in _SCENE_LAYOUTS.items():
        train_ids = set(split[f"{scene_type}_train"])
        eval_ids = set(split[f"{scene_type}_eval"])
        overlap = train_ids.intersection(eval_ids)
        if overlap:
            raise ValueError(
                f"official {scene_type} train/eval scene overlap: {sorted(overlap)[:5]}"
            )
        esdf_root = metadata_root / subset / "esdf"
        pointgoal_root = metadata_root / subset / "pointgoal_start_pair"
        for scene_id in sorted(train_ids):
            if scene_id in KNOWN_BAD_SCENES:
                continue
            navigable = esdf_root / scene_id / "navigable.ply"
            pointgoal = pointgoal_root / scene_id / "pointgoal_start_pair_samples_safe.npy"
            if not navigable.is_file() or not pointgoal.is_file():
                continue
            records.append(
                SceneRecord(
                    scene_id=scene_id,
                    scene_type=scene_type,
                    official_split="train",
                    internal_split="train",
                    family_id=scene_family_id(scene_id),
                    navigable_ply=str(navigable.resolve()),
                    pointgoal_npy=str(pointgoal.resolve()),
                    usd_path=_find_usd(root, subset, scenes_dir, scene_id),
                )
            )

    if not records:
        raise FileNotFoundError(
            f"no official training scenes with navigable.ply and safe point-goals under {root}"
        )
    return records


def assign_group_disjoint_validation(
    scenes: Iterable[SceneRecord],
    *,
    validation_fraction: float,
    seed: int,
) -> list[SceneRecord]:
    """Assign whole scan families to an internal validation split."""

    if not 0.0 <= validation_fraction < 1.0:
        raise ValueError("validation_fraction must be in [0, 1)")
    scenes = list(scenes)
    families = sorted(
        {scene.family_id for scene in scenes},
        key=lambda family: hashlib.sha256(f"{seed}:{family}".encode("utf-8")).hexdigest(),
    )
    validation_families: set[str] = set()
    if validation_fraction > 0.0 and len(families) >= 2:
        count = max(1, round(len(families) * validation_fraction))
        count = min(count, len(families) - 1)
        validation_families.update(families[:count])

    return [
        replace(
            scene,
            internal_split=(
                "validation"
                if scene.family_id in validation_families
                else "train"
            ),
        )
        for scene in scenes
    ]
