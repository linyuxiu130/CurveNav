"""Load and validate the official X-NavDP PointGoal evaluation suite."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

from .episodes import sha256_file, sha256_json, validate_official_episode_file
from .usd import find_canonical_scene_usd


@dataclass(frozen=True)
class SceneJob:
    """One official X-NavDP scene and its immutable episode file."""

    key: str
    split: str
    name: str
    scene_dir: Path
    navigation_file: Path
    navigation_source_path: str
    navigation_sha256: str
    episode_file: Path
    episode_source_path: str
    episode_sha256: str
    scene_scale: float
    height_offset_m: float
    rgb_num_samples: int
    episodes: int
    episode_contract: dict


def suite_definition_sha256(suite: dict) -> str:
    """Hash all semantics that define a measured result."""
    return sha256_json({
        key: suite[key]
        for key in (
            "name",
            "source",
            "episodes_per_scene",
            "episode_contract",
            "simulator_contract",
            "splits",
        )
    })


def load_suite(
    manifest: Path,
    scene_root: Path,
    repository_root: Path,
    episodes_per_scene: int | None = None,
    selected_scenes: set[str] | None = None,
) -> tuple[dict, list[SceneJob]]:
    """Expand the pinned 20-Home/20-Commercial X-NavDP split."""
    data = json.loads(manifest.read_text())
    if data.get("name") != "pointgoal-v2":
        raise ValueError("only the official X-NavDP PointGoal suite is executable")
    if data.get("status") != "frozen":
        raise ValueError("the X-NavDP suite manifest must be frozen")
    episode_count = (
        int(episodes_per_scene)
        if episodes_per_scene is not None
        else int(data["episodes_per_scene"])
    )
    available = int(data["episode_contract"]["episode_count"])
    if episode_count <= 0 or episode_count > available:
        raise ValueError(f"episodes per scene must be in [1, {available}]")

    jobs: list[SceneJob] = []
    for split, config in data["splits"].items():
        expected_hashes = config["episode_sha256"]
        expected_navigation_hashes = config["navigation_sha256"]
        if set(expected_hashes) != set(config["scenes"]):
            raise ValueError(f"{split}: episode SHA map differs from the scene list")
        if set(expected_navigation_hashes) != set(config["scenes"]):
            raise ValueError(f"{split}: navigation SHA map differs from the scene list")
        for scene_name in config["scenes"]:
            key = f"{split}/{scene_name}"
            if selected_scenes and key not in selected_scenes and scene_name not in selected_scenes:
                continue
            episode_source_path = (
                f"navigation_metadata/internscenes_{split}/pointgoal_start_pair/"
                f"{scene_name}/{data['episode_contract']['episode_filename']}"
            )
            navigation_source_path = (
                f"navigation_metadata/internscenes_{split}/esdf/"
                f"{scene_name}/navigable.ply"
            )
            jobs.append(SceneJob(
                key=key,
                split=split,
                name=scene_name,
                scene_dir=scene_root / config["scene_root"] / scene_name,
                navigation_file=(
                    scene_root / config["navigation_root"] / scene_name / "navigable.ply"
                ),
                navigation_source_path=navigation_source_path,
                navigation_sha256=expected_navigation_hashes[scene_name],
                episode_file=(
                    repository_root / config["episode_root"] / scene_name
                    / data["episode_contract"]["episode_filename"]
                ),
                episode_source_path=episode_source_path,
                episode_sha256=expected_hashes[scene_name],
                scene_scale=float(config["scene_scale"]),
                height_offset_m=float(config["height_offset_m"]),
                rgb_num_samples=int(config["rgb_num_samples"]),
                episodes=episode_count,
                episode_contract=data["episode_contract"],
            ))

    if selected_scenes:
        matched = {job.key for job in jobs} | {job.name for job in jobs}
        missing = sorted(selected_scenes - matched)
        if missing:
            raise ValueError(f"unknown scene selection: {', '.join(missing)}")
    if not jobs:
        raise ValueError("suite selection contains no scenes")
    return data, jobs


def missing_assets(jobs: list[SceneJob]) -> list[Path]:
    """Return every missing official simulator or episode input."""
    missing: list[Path] = []
    for job in jobs:
        if not job.scene_dir.is_dir():
            missing.append(job.scene_dir)
        else:
            for name in ("models", "Materials"):
                if not (job.scene_dir / name).is_dir():
                    missing.append(job.scene_dir / name)
            try:
                find_canonical_scene_usd(job.scene_dir)
            except (FileNotFoundError, RuntimeError):
                missing.append(job.scene_dir / "<canonical-scene-layer>")
        if not job.navigation_file.is_file():
            missing.append(job.navigation_file)
        if not job.episode_file.is_file():
            missing.append(job.episode_file)
    return missing


def invalid_navigation_assets(suite: dict, jobs: list[SceneJob]) -> list[str]:
    """Reject navigation meshes that differ from the pinned X-NavDP release."""
    errors: list[str] = []
    actual_hashes: dict[str, str] = {}
    for job in jobs:
        if job.navigation_file.is_file():
            actual = sha256_file(job.navigation_file)
            actual_hashes[job.key] = actual
            if actual != job.navigation_sha256:
                errors.append(
                    f"{job.key}: navigable.ply SHA-256 differs from X-NavDP"
                )
    if len(jobs) == 40 and all(job.navigation_file.is_file() for job in jobs):
        expected = suite["source"]["navigation_bundle_sha256"]
        digest = hashlib.sha256()
        for job in jobs:
            digest.update(job.navigation_source_path.encode("utf-8"))
            digest.update(b"\0")
            digest.update(actual_hashes[job.key].encode("ascii"))
            digest.update(b"\n")
        if digest.hexdigest() != expected:
            errors.append("the 40-scene navigation bundle differs from X-NavDP")
    return errors


def invalid_episode_assets(jobs: list[SceneJob]) -> list[str]:
    """Reject any episode file that differs from the pinned X-NavDP release."""
    errors: list[str] = []
    actual_hashes: dict[str, str] = {}
    for job in jobs:
        if not job.episode_file.is_file():
            continue
        actual_hash = sha256_file(job.episode_file)
        actual_hashes[job.key] = actual_hash
        for error in validate_official_episode_file(
            job.episode_file,
            expected_count=int(job.episode_contract["episode_count"]),
            expected_dtype=job.episode_contract["storage_dtype"],
            expected_sha256=job.episode_sha256,
            actual_sha256=actual_hash,
        ):
            errors.append(f"{job.key}: {error}")
    if len(jobs) == 40 and all(job.episode_file.is_file() for job in jobs):
        expected = jobs[0].episode_contract["episode_bundle_sha256"]
        digest = hashlib.sha256()
        for job in jobs:
            digest.update(job.episode_source_path.encode("utf-8"))
            digest.update(b"\0")
            digest.update(actual_hashes[job.key].encode("ascii"))
            digest.update(b"\n")
        actual = digest.hexdigest()
        if actual != expected:
            errors.append(
                "the 40-scene episode bundle differs from the pinned X-NavDP release"
            )
    return errors
