"""Canonical USD scene-file discovery without simulator imports."""

from __future__ import annotations

from pathlib import Path


USD_SUFFIXES = frozenset({".usd", ".usda", ".usdc"})
PREFERRED_SCENE_FILES = (
    "start_result_navigation.usd",
    "scene.usda",
    "scene.usd",
    "scene.usdc",
)


def is_scene_usd(path: Path) -> bool:
    return (
        path.is_file()
        and path.suffix.lower() in USD_SUFFIXES
        and "noMDL" not in path.name
    )


def find_canonical_scene_usd(directory: Path) -> Path:
    """Return one explicit runtime scene layer or reject ambiguity."""
    candidates = sorted(path for path in directory.iterdir() if is_scene_usd(path))
    by_name = {path.name: path for path in candidates}
    for name in PREFERRED_SCENE_FILES:
        if name in by_name:
            return by_name[name]
    if len(candidates) == 1:
        return candidates[0]
    if not candidates:
        raise FileNotFoundError(f"No USD scene found under {directory}")
    raise RuntimeError(
        f"Ambiguous USD scene under {directory}: "
        + ", ".join(path.name for path in candidates)
    )
