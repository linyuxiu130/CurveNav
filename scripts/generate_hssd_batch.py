#!/usr/bin/env python3
"""Run a resumable download, generate and audit HSSD expert-data batch."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
MAXIMUM_INVALID_DEPTH_FRACTION = 0.95
DEFAULT_HABITAT_PYTHON = (
    REPOSITORY_ROOT.parent / ".venvs" / "habitat-hssd" / "bin" / "python"
)


def _run(
    command: list[str],
    log_path: Path,
    env: dict[str, str],
    *,
    check: bool = True,
) -> float:
    start = time.monotonic()
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as log:
        log.write(f"\n$ {' '.join(command)}\n")
        log.flush()
        result = subprocess.run(
            command,
            cwd=REPOSITORY_ROOT,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=False,
        )
        if check:
            result.check_returncode()
    return time.monotonic() - start


def _scene_seed(base_seed: int, scene_id: str) -> int:
    digest = hashlib.sha256(scene_id.encode("utf-8")).hexdigest()
    return base_seed + int(digest[:8], 16) % 1_000_000


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")


def _completed_scenes(output_root: Path, routes_per_scene: int) -> dict[str, list[str]]:
    completed = {"train": [], "validation": []}
    for split in completed:
        for dataset in sorted((output_root / split).glob("dataset_hssd_*")):
            if len(list(dataset.glob("run_*"))) != routes_per_scene:
                continue
            if not (dataset / "codec_audit.json").exists():
                continue
            completed[split].append(dataset.name.removeprefix("dataset_hssd_"))
    return completed


def _annotate_dataset(dataset: Path, split: str, revision: str) -> None:
    for metadata_path in sorted(dataset.glob("run_*/metadata.json")):
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        metadata["official_split"] = "val" if split == "validation" else "train"
        metadata["source_revision"] = revision
        _write_json(metadata_path, metadata)


def _keep_safe_runs(work_root: Path, scene_id: str, target_runs: int) -> int:
    report = json.loads((work_root / "codec_audit.json").read_text(encoding="utf-8"))
    dataset = work_root / f"dataset_hssd_{scene_id}"
    valid_depth = {
        path.parent.name
        for path in dataset.glob("run_*/metadata.json")
        if float(
            json.loads(path.read_text(encoding="utf-8"))["depth"][
                "invalid_depth_fraction_max"
            ]
        )
        <= MAXIMUM_INVALID_DEPTH_FRACTION
    }
    safe_runs = [
        run["run_id"]
        for run in report["runs"]
        if int(run["unsafe_windows"]) == 0 and run["run_id"] in valid_depth
    ]
    if len(safe_runs) < target_runs:
        raise RuntimeError(
            f"only {len(safe_runs)} codec-safe runs remain, need {target_runs}"
        )
    keep = set(safe_runs[:target_runs])
    all_runs = sorted(dataset.glob("run_*"))
    for run_dir in all_runs:
        if run_dir.name not in keep:
            shutil.rmtree(run_dir)
    rejected = len(all_runs) - len(keep)
    summary_path = work_root / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["candidate_runs"] = summary["runs"]
    summary["rejected_candidate_runs"] = rejected
    summary["runs"] = len(keep)
    summary["frames"] = sum(
        int(json.loads(path.read_text(encoding="utf-8"))["frames"])
        for path in dataset.glob("run_*/metadata.json")
    )
    _write_json(summary_path, summary)
    return rejected


def _write_batch_summary(output_root: Path, target: dict[str, int]) -> None:
    records = []
    scene_audits = []
    split_runs = Counter()
    split_scenes = Counter()
    for split in ("train", "validation"):
        for dataset in sorted((output_root / split).glob("dataset_hssd_*")):
            split_scenes[split] += 1
            audit_path = dataset / "codec_audit.json"
            if audit_path.exists():
                scene_audits.append(json.loads(audit_path.read_text(encoding="utf-8")))
            for metadata_path in sorted(dataset.glob("run_*/metadata.json")):
                record = json.loads(metadata_path.read_text(encoding="utf-8"))
                records.append(record)
                split_runs[split] += 1
    manifest_lines = [json.dumps(record, sort_keys=True) + "\n" for record in records]
    (output_root / "manifest.jsonl").write_text("".join(manifest_lines), encoding="utf-8")
    difficulty = Counter(str(record["difficulty"]) for record in records)
    audits = [run for report in scene_audits for run in report["runs"]]
    summary = {
        "target": target,
        "runs": len(records),
        "frames": sum(int(record["frames"]) for record in records),
        "scenes": dict(split_scenes),
        "split_runs": dict(split_runs),
        "difficulty": dict(sorted(difficulty.items())),
        "unsafe_codec_windows": sum(int(run["unsafe_windows"]) for run in audits),
        "minimum_decoded_clearance_m": min(
            (float(run["minimum_decoded_clearance_m"]) for run in audits),
            default=None,
        ),
        "maximum_invalid_depth_fraction": max(
            (
                float(record["depth"]["invalid_depth_fraction_max"])
                for record in records
            ),
            default=None,
        ),
    }
    _write_json(output_root / "summary.json", summary)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("asset_root", type=Path)
    parser.add_argument("output_root", type=Path)
    parser.add_argument("scene_plan", type=Path)
    parser.add_argument("--habitat-python", type=Path, default=DEFAULT_HABITAT_PYTHON)
    parser.add_argument("--cuda-visible-device", default="1")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    asset_root = args.asset_root.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    plan = json.loads(args.scene_plan.read_text(encoding="utf-8"))
    target = plan["target"]
    routes_per_scene = int(target["routes_per_scene"])
    target_scenes = {
        "train": int(target["train_scenes"]),
        "validation": int(target["validation_scenes"]),
    }
    output_root.mkdir(parents=True, exist_ok=True)
    for split in target_scenes:
        (output_root / split).mkdir(exist_ok=True)
    progress_path = output_root / "progress.json"
    progress = (
        json.loads(progress_path.read_text(encoding="utf-8"))
        if progress_path.exists()
        else {"failures": [], "successful_scenes": []}
    )
    completed = _completed_scenes(output_root, routes_per_scene)
    progress["failures"] = [
        item
        for item in progress["failures"]
        if "download_hssd_scenes.py" not in item["error"]
    ]
    failed = {item["scene_id"] for item in progress["failures"]}

    base_env = os.environ.copy()
    base_env["HF_HUB_DISABLE_XET"] = "1"
    base_env["PYTHONPATH"] = str(REPOSITORY_ROOT / "src")
    revision = json.loads(
        (asset_root / "download_manifest.json").read_text(encoding="utf-8")
    )["revision"]

    for split, needed in target_scenes.items():
        for scene_id in plan["queues"][split]:
            if len(completed[split]) >= needed:
                break
            if scene_id in completed[split] or scene_id in failed:
                continue
            work_root = output_root / ".staging" / f"{split}_{scene_id}"
            log_path = output_root / "logs" / f"{split}_{scene_id}.log"
            if work_root.exists():
                shutil.rmtree(work_root)
            work_root.mkdir(parents=True)
            durations = {}
            try:
                durations["download_s"] = _run(
                    [
                        sys.executable,
                        "scripts/download_hssd_scenes.py",
                        str(asset_root),
                        scene_id,
                    ],
                    log_path,
                    base_env,
                )
                generation_env = base_env.copy()
                generation_env["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_device
                durations["generation_s"] = _run(
                    [
                        str(args.habitat_python),
                        "-m",
                        "curvenav.data_generation.hssd_pilot",
                        str(asset_root),
                        str(work_root),
                        "--scene-id",
                        scene_id,
                        "--routes-per-scene",
                        str(routes_per_scene + 2),
                        "--gpu-device",
                        "0",
                        "--seed",
                        str(_scene_seed(args.seed, scene_id)),
                    ],
                    log_path,
                    generation_env,
                )
                durations["screening_audit_s"] = _run(
                    [
                        sys.executable,
                        "-m",
                        "curvenav.data_generation.audit_hssd_runs",
                        str(work_root),
                    ],
                    log_path,
                    base_env,
                    check=False,
                )
                rejected_runs = _keep_safe_runs(
                    work_root, scene_id, routes_per_scene
                )
                durations["final_audit_s"] = _run(
                    [
                        sys.executable,
                        "-m",
                        "curvenav.data_generation.audit_hssd_runs",
                        str(work_root),
                    ],
                    log_path,
                    base_env,
                )
                dataset = work_root / f"dataset_hssd_{scene_id}"
                _annotate_dataset(dataset, split, revision)
                shutil.move(str(work_root / "codec_audit.json"), dataset / "codec_audit.json")
                shutil.move(str(work_root / "summary.json"), dataset / "generation_summary.json")
                destination = output_root / split / dataset.name
                dataset.rename(destination)
                completed[split].append(scene_id)
                progress["successful_scenes"].append(
                    {
                        "scene_id": scene_id,
                        "split": split,
                        "durations": durations,
                        "rejected_candidate_runs": rejected_runs,
                    }
                )
            except Exception as error:
                progress["failures"].append(
                    {
                        "scene_id": scene_id,
                        "split": split,
                        "error": f"{type(error).__name__}: {error}",
                        "log": str(log_path),
                    }
                )
                failed.add(scene_id)
            finally:
                if work_root.exists():
                    shutil.rmtree(work_root)
                _write_json(progress_path, progress)
                _write_batch_summary(output_root, target)

    missing = {
        split: target_scenes[split] - len(completed[split]) for split in target_scenes
    }
    if any(missing.values()):
        raise RuntimeError(f"HSSD scene queues exhausted with missing scenes: {missing}")
    print((output_root / "summary.json").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
