"""Audit v2 path curvature against the currently deployed benchmark controls.

This supplement reads only route and descriptor files.  It deliberately does
not interpret a desired-speed heuristic as a hard robot turning-radius limit.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


BENCHMARK_V_MAX_MPS = 0.5
BENCHMARK_W_MAX_RADPS = 0.5
BENCHMARK_DT_S = 0.1
BENCHMARK_PLANNER_INTERVAL_S = 0.2
DINGO_WHEEL_RADIUS_M = 0.0591
DINGO_WHEEL_BASE_M = 0.22616
DINGO_WHEEL_VELOCITY_LIMIT_RADPS = 100.0


EVIDENCE = [
    {
        "scope": "active quick100 benchmark",
        "path": "/mnt/data1/huangshibo/H/general-navigation-benchmark/scripts/run_quick100.sh",
        "lines": "36-48",
        "sha256": "992837f1309f3aba8055b2f99c9596f9abefbe55b9abf7669a419223f6ec9063",
        "fact": "quick100 leaves speed at the CLI default and sets planner_interval=0.2 s",
    },
    {
        "scope": "active quick100 benchmark",
        "path": "/mnt/data1/huangshibo/H/general-navigation-benchmark/navbench/cli.py",
        "lines": "148, 228-239, 334-350",
        "sha256": "c9ac912004cd00ebe16efccf27806648b0c702c614961432118a5112653986a4",
        "fact": "speed defaults to 0.5 and is forwarded to the evaluator",
    },
    {
        "scope": "active quick100 benchmark",
        "path": "/mnt/data1/huangshibo/H/general-navigation-benchmark/navbench/isaac/evaluators/pointgoal.py",
        "lines": "149-156, 239-241, 399-435",
        "sha256": "cd8c6af1f900b6912f4d15af391b5b0b1f2cbd13b14814ccba17152b9b962079",
        "fact": "the same speed sets desired_v, v_max and w_max; MPC output drives the differential controller",
    },
    {
        "scope": "active quick100 benchmark",
        "path": "/mnt/data1/huangshibo/H/general-navigation-benchmark/navbench/isaac/tracking.py",
        "lines": "49-55, 71-93, 169-180",
        "sha256": "9b95c05c1e9e6ac088571f14f247da8b5ef7b45b56600a63ddbc28b9a85f9f2c",
        "fact": "unicycle MPC uses dt=0.1 s with hard bounds 0<=v<=0.5 m/s and |w|<=0.5 rad/s; v=0 is allowed",
    },
    {
        "scope": "active Dingo simulation",
        "path": "/mnt/data1/huangshibo/H/general-navigation-benchmark/navbench/isaac/configs/robots/dingo_config.py",
        "lines": "22-35",
        "sha256": "19c1f5f507ef5e7442d95122edb809a068dacda6af80b457a33a0f164ef73c8f",
        "fact": "wheel velocity limit=100 rad/s, wheel radius=0.0591 m and wheel base=0.22616 m",
    },
    {
        "scope": "active Dingo simulation",
        "path": "/mnt/data1/huangshibo/H/general-navigation-benchmark/navbench/isaac/controllers/differential_controller.py",
        "lines": "71-85, 97-112",
        "sha256": "a8ff3a1919feb08ef55367ec2d65cf95106b161a4fbfc47b31b7b9b2e7a80b6e",
        "fact": "wheel commands implement the standard differential-drive mapping and permit opposite wheel directions",
    },
    {
        "scope": "downloaded X-NavDP secondary reference",
        "path": "/mnt/data/huangshibo/H/navigation_three_projects/open_source_full/x_navdp/baselines/x-navdp/config/x-navdp_config.yaml",
        "lines": "27-34",
        "sha256": "e1271059d69ed6cbdaf68f278f97ef143bfd910346a52f89f5fdbe12d312a6d9",
        "fact": "configured desired_v=v_max=w_max=0.5, v_min_high_curvature=0.05 and control dt=0.1 s",
    },
    {
        "scope": "downloaded X-NavDP secondary reference",
        "path": "/mnt/data/huangshibo/H/navigation_three_projects/open_source_full/x_navdp/baselines/x-navdp/src/utils/mpc_tracking.py",
        "lines": "55-66, 106-109, 278-286, 332-347",
        "sha256": "e9144624296331b973d7a030b735fd615561102672ab6adf41cf6c7ca1361f03",
        "fact": "0.05 m/s is a minimum desired-speed heuristic; it is not a lower hard bound on v",
    },
    {
        "scope": "downloaded NavDP real-world secondary reference",
        "path": "/mnt/data/huangshibo/H/navigation_three_projects/open_source_full/navdp/scripts/realworld/controllers.py",
        "lines": "14-26, 69-71",
        "sha256": "251522af6da05daee5279e07371b5f9ca319326e7097c0c7e5ae68e76217e408",
        "fact": "NavDP real-world MPC defaults to desired_v=0.3, v_max=w_max=0.4 and also permits v=0",
    },
    {
        "scope": "downloaded SanD source",
        "path": "/mnt/data/huangshibo/H/navigation_three_projects/open_source_full/sand/sand_planner/config.py",
        "lines": "69-80",
        "sha256": "1b8c114b484e58f0711920e7526a2de809a2fdfb8cd49ab054730295f5a2bc8e",
        "fact": "SanD states an ESDF obstacle-inflation distance but no linear/angular control or turning-radius constraint",
    },
]


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]


def geometric_curvature(path_xy: np.ndarray) -> np.ndarray:
    """Return three-point Menger curvature at interior path vertices."""

    path = np.asarray(path_xy, dtype=np.float64)
    if len(path) < 3:
        return np.zeros(0, dtype=np.float64)
    first = path[1:-1] - path[:-2]
    second = path[2:] - path[1:-1]
    chord = path[2:] - path[:-2]
    a = np.linalg.norm(first, axis=1)
    b = np.linalg.norm(second, axis=1)
    c = np.linalg.norm(chord, axis=1)
    twice_area = np.abs(first[:, 0] * chord[:, 1] - first[:, 1] * chord[:, 0])
    denominator = a * b * c
    return np.divide(
        2.0 * twice_area,
        denominator,
        out=np.zeros_like(twice_area),
        where=denominator > 1e-9,
    )


def _heading_curvature(path_xy: np.ndarray, yaw: np.ndarray) -> np.ndarray:
    segment = np.linalg.norm(np.diff(path_xy, axis=0), axis=1)
    if len(segment) < 2:
        return np.zeros(0, dtype=np.float64)
    delta_yaw = np.abs(
        np.arctan2(np.sin(np.diff(yaw[:-1])), np.cos(np.diff(yaw[:-1])))
    )
    local_arc = 0.5 * (segment[:-1] + segment[1:])
    return np.divide(
        delta_yaw,
        local_arc,
        out=np.zeros_like(delta_yaw),
        where=local_arc > 1e-9,
    )


def _distribution(values: list[float] | np.ndarray) -> dict[str, float | int | None]:
    array = np.asarray(values, dtype=np.float64)
    if not len(array):
        return {
            "count": 0,
            "min": None,
            "p01": None,
            "p05": None,
            "p50": None,
            "p95": None,
            "p99": None,
            "max": None,
            "mean": None,
        }
    percentiles = np.percentile(array, [1, 5, 50, 95, 99])
    return {
        "count": len(array),
        "min": float(array.min()),
        "p01": float(percentiles[0]),
        "p05": float(percentiles[1]),
        "p50": float(percentiles[2]),
        "p95": float(percentiles[3]),
        "p99": float(percentiles[4]),
        "max": float(array.max()),
        "mean": float(array.mean()),
    }


def _speed_cap_for_curvature(curvature: float) -> float:
    if curvature <= BENCHMARK_W_MAX_RADPS / BENCHMARK_V_MAX_MPS:
        return BENCHMARK_V_MAX_MPS
    return BENCHMARK_W_MAX_RADPS / curvature


def _wheel_speeds(linear: float, angular: float) -> tuple[float, float]:
    half_turn = 0.5 * angular * DINGO_WHEEL_BASE_M
    return (
        (linear - half_turn) / DINGO_WHEEL_RADIUS_M,
        (linear + half_turn) / DINGO_WHEEL_RADIUS_M,
    )


def _threshold_counts(values: list[float], thresholds: list[float]) -> dict[str, Any]:
    return {
        str(threshold): {
            "count_above": int(np.sum(np.asarray(values) > threshold + 1e-9)),
            "fraction_above": float(np.mean(np.asarray(values) > threshold + 1e-9)),
            "constant_speed_mps": BENCHMARK_W_MAX_RADPS / threshold,
        }
        for threshold in thresholds
    }


def audit_dynamics(dataset_root: Path) -> dict[str, Any]:
    dataset_root = dataset_root.resolve()
    episodes = _read_jsonl(dataset_root / "episodes.jsonl")
    samples = _read_jsonl(dataset_root / "samples.jsonl")
    routes: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    episode_maximum = []
    episode_internal_p95 = []
    pointwise = []
    heading_pointwise = []
    by_difficulty: dict[str, list[float]] = defaultdict(list)
    by_split: dict[str, list[float]] = defaultdict(list)
    episode_metrics = []

    for episode in episodes:
        episode_dir = (
            dataset_root
            / episode["split"]
            / f"dataset_hssd_{episode['scene_id']}"
            / episode["run_id"]
        )
        with np.load(episode_dir / "route.npz") as route:
            world_xy = route["world_xy"].astype(np.float64)
            yaw = route["yaw_rad"].astype(np.float64)
        curvature = geometric_curvature(world_xy)
        heading = _heading_curvature(world_xy, yaw)
        maximum = float(curvature.max(initial=0.0))
        internal_p95 = float(np.percentile(curvature, 95)) if len(curvature) else 0.0
        speed_cap = _speed_cap_for_curvature(maximum)
        routes[episode["episode_id"]] = (world_xy, curvature)
        episode_maximum.append(maximum)
        episode_internal_p95.append(internal_p95)
        pointwise.extend(curvature.tolist())
        heading_pointwise.extend(heading.tolist())
        by_difficulty[episode["difficulty"]].append(maximum)
        by_split[episode["split"]].append(maximum)
        episode_metrics.append(
            {
                "episode_id": episode["episode_id"],
                "split": episode["split"],
                "difficulty": episode["difficulty"],
                "maximum_geometric_curvature_per_m": maximum,
                "internal_p95_geometric_curvature_per_m": internal_p95,
                "constant_speed_cap_for_wmax_mps": speed_cap,
                "requires_slowdown_from_0.5_mps": maximum > 1.0 + 1e-9,
            }
        )

    sample_maximum = []
    sample_internal_p95 = []
    sample_first_12_maximum = []
    pooled_sample_points = []
    for sample in samples:
        world_xy, _ = routes[sample["episode_id"]]
        target = world_xy[
            int(sample["target_start_index"]) : int(sample["target_end_index"]) + 1
        ]
        curvature = geometric_curvature(target)
        sample_maximum.append(float(curvature.max(initial=0.0)))
        sample_internal_p95.append(
            float(np.percentile(curvature, 95)) if len(curvature) else 0.0
        )
        sample_first_12_maximum.append(float(curvature[:12].max(initial=0.0)))
        pooled_sample_points.extend(curvature.tolist())

    thresholds = [1.0, 2.0, 5.0, 10.0]
    episode_speed_caps = [_speed_cap_for_curvature(value) for value in episode_maximum]
    sample_speed_caps = [_speed_cap_for_curvature(value) for value in sample_maximum]
    tightest_curvature = max(episode_maximum)
    tightest_speed = _speed_cap_for_curvature(tightest_curvature)
    tightest_wheels = _wheel_speeds(tightest_speed, BENCHMARK_W_MAX_RADPS)
    bound_wheels = _wheel_speeds(BENCHMARK_V_MAX_MPS, BENCHMARK_W_MAX_RADPS)
    in_place_wheels = _wheel_speeds(0.0, BENCHMARK_W_MAX_RADPS)

    summary = {
        "dataset_root": str(dataset_root),
        "read_only_inputs": ["episodes.jsonl", "samples.jsonl", "route.npz"],
        "episodes": len(episodes),
        "samples": len(samples),
        "control_facts": {
            "active_benchmark": {
                "model": "unicycle kinematics",
                "nominal_reference_speed_mps": BENCHMARK_V_MAX_MPS,
                "hard_v_bounds_mps": [0.0, BENCHMARK_V_MAX_MPS],
                "hard_w_bounds_radps": [-BENCHMARK_W_MAX_RADPS, BENCHMARK_W_MAX_RADPS],
                "control_dt_s": BENCHMARK_DT_S,
                "planner_interval_s": BENCHMARK_PLANNER_INTERVAL_S,
                "configured_minimum_nonzero_speed_mps": None,
                "configured_minimum_turning_radius_m": None,
                "derived_kinematic_turning_radius_infimum_m": 0.0,
                "in_place_rotation_representable": True,
            },
            "dingo": {
                "wheel_radius_m": DINGO_WHEEL_RADIUS_M,
                "wheel_base_m": DINGO_WHEEL_BASE_M,
                "simulated_wheel_velocity_limit_radps": DINGO_WHEEL_VELOCITY_LIMIT_RADPS,
                "wheel_rates_at_vmax_wmax_radps": list(bound_wheels),
                "wheel_rates_for_in_place_wmax_radps": list(in_place_wheels),
            },
            "not_defined_by_current_sources": [
                "minimum stable nonzero forward speed",
                "linear acceleration limit",
                "angular acceleration limit",
                "jerk limit",
                "lateral acceleration or wheel-slip limit",
                "hardware trackability error versus geometric curvature",
            ],
        },
        "curvature_definition": {
            "primary": "three-point Menger curvature on raw approximately 0.15 m-spaced expert vertices",
            "cross_check": "wrapped heading change divided by local arc length",
            "units": "m^-1",
            "speed_relation": "for nonzero forward motion, |omega| = v * |kappa|",
            "warning": "geometric curvature is not itself a hard feasibility label when v may reach zero",
        },
        "distributions": {
            "episode_maximum_curvature_per_m": _distribution(episode_maximum),
            "episode_internal_p95_curvature_per_m": _distribution(episode_internal_p95),
            "unique_route_point_curvature_per_m": _distribution(pointwise),
            "heading_change_curvature_cross_check_per_m": _distribution(heading_pointwise),
            "sample_target_maximum_curvature_per_m": _distribution(sample_maximum),
            "sample_target_internal_p95_curvature_per_m": _distribution(sample_internal_p95),
            "sample_target_first_12_maximum_curvature_per_m": _distribution(sample_first_12_maximum),
            "pooled_sample_target_point_curvature_per_m": _distribution(pooled_sample_points),
            "episode_speed_cap_for_wmax_mps": _distribution(episode_speed_caps),
            "sample_speed_cap_for_wmax_mps": _distribution(sample_speed_caps),
        },
        "constant_speed_bands": {
            "episode_maximum": _threshold_counts(episode_maximum, thresholds),
            "sample_target_maximum": _threshold_counts(sample_maximum, thresholds),
            "unique_route_points": _threshold_counts(pointwise, thresholds),
            "pooled_sample_target_points": _threshold_counts(pooled_sample_points, thresholds),
        },
        "by_difficulty_episode_maximum": {
            difficulty: _distribution(values)
            for difficulty, values in sorted(by_difficulty.items())
        },
        "by_split_episode_maximum": {
            split: _distribution(values) for split, values in sorted(by_split.items())
        },
        "tightest_observed_turn": {
            "curvature_per_m": tightest_curvature,
            "geometric_radius_m": 1.0 / tightest_curvature,
            "constant_speed_cap_for_wmax_mps": tightest_speed,
            "wheel_rates_at_speed_cap_and_wmax_radps": list(tightest_wheels),
            "wheel_velocity_limit_exceeded": max(map(abs, tightest_wheels)) > DINGO_WHEEL_VELOCITY_LIMIT_RADPS,
        },
        "interpretation": {
            "episodes_requiring_slowdown_from_nominal_0.5_mps": int(np.sum(np.asarray(episode_maximum) > 1.0 + 1e-9)),
            "samples_requiring_slowdown_from_nominal_0.5_mps": int(np.sum(np.asarray(sample_maximum) > 1.0 + 1e-9)),
            "episodes_above_secondary_0.05_mps_reference_band": int(np.sum(np.asarray(episode_maximum) > 10.0 + 1e-9)),
            "samples_above_secondary_0.05_mps_reference_band": int(np.sum(np.asarray(sample_maximum) > 10.0 + 1e-9)),
            "hard_control_infeasible_count": None,
            "hard_control_infeasible_reason": "cannot be derived: current MPC permits v=0 and sources define no positive minimum speed or tracking-error limit",
        },
        "gate_decision": {
            "new_source_grounded_hard_curvature_gate": False,
            "reject_at_kappa_greater_than_1": False,
            "reason": "kappa=1 m^-1 is only the full-speed boundary at v=0.5 m/s, not a minimum-radius boundary",
            "recommended_action": "retain curvature distributions as a regression audit; do not relabel 1<kappa<=10 samples as non-executable",
            "existing_4_and_10_thresholds": "quality heuristics in CurveNav generation, not physical Dingo limits; this audit does not change them",
            "future_hard_gate_requirement": "first freeze a positive v_min or an empirical closed-loop tracking-error envelope, then derive kappa from that contract",
        },
        "evidence": EVIDENCE,
    }

    audit_dir = dataset_root / "audit"
    audit_dir.mkdir(exist_ok=True)
    (audit_dir / "dynamics_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (audit_dir / "dynamics_episode_metrics.json").write_text(
        json.dumps(episode_metrics, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    episode_dist = summary["distributions"]["episode_maximum_curvature_per_m"]
    target_dist = summary["distributions"]["sample_target_internal_p95_curvature_per_m"]
    pooled_dist = summary["distributions"]["pooled_sample_target_point_curvature_per_m"]
    evidence_lines = "\n".join(
        f"- `{item['path']}:{item['lines']}` (SHA-256 `{item['sha256']}`): "
        f"{item['fact']}."
        for item in EVIDENCE
    )
    difficulty_lines = "\n".join(
        f"- {name}: n={distribution['count']}, p50={distribution['p50']:.3f}, "
        f"p95={distribution['p95']:.3f}, max={distribution['max']:.3f} m^-1."
        for name, distribution in summary["by_difficulty_episode_maximum"].items()
    )
    episode_bands = summary["constant_speed_bands"]["episode_maximum"]
    sample_bands = summary["constant_speed_bands"]["sample_target_maximum"]
    report = f"""# CurveNav v2 pilot dynamics supplement

## Decision

Do **not** add a source-grounded hard geometric-curvature rejection gate. The active benchmark permits `v=0`, so the configured Dingo differential drive can rotate in place and no positive minimum turning radius is defined. Keep curvature as a distribution/regression audit.

## Pilot 200

- Per-episode maximum curvature p50 / p95 / max: **{episode_dist['p50']:.3f} / {episode_dist['p95']:.3f} / {episode_dist['max']:.3f} m^-1**.
- Per-sample internal-p95 curvature distribution p95: **{target_dist['p95']:.3f} m^-1**.
- Pooled target-point curvature p95 / p99: **{pooled_dist['p95']:.3f} / {pooled_dist['p99']:.3f} m^-1**.
- Episodes requiring slowdown from constant 0.5 m/s: **{summary['interpretation']['episodes_requiring_slowdown_from_nominal_0.5_mps']} / {len(episodes)}**.
- Episodes exceeding the secondary 0.05 m/s, 0.5 rad/s reference band: **{summary['interpretation']['episodes_above_secondary_0.05_mps_reference_band']}**.
- Tightest observed radius / required constant-speed cap: **{summary['tightest_observed_turn']['geometric_radius_m']:.3f} m / {summary['tightest_observed_turn']['constant_speed_cap_for_wmax_mps']:.3f} m/s**.

`kappa=1 m^-1` is the full-speed boundary at 0.5 m/s, not a physical minimum-radius boundary. Acceleration, jerk, slip and a positive minimum stable speed are absent from the inspected benchmark and source configurations, so a harder threshold cannot be inferred.

## Constant-speed interpretation

For nonzero forward motion, `|omega| = v * |kappa|`. The counts below say how many paths need to slow below the stated constant speed; they do not label those paths infeasible.

| Curvature band | Constant speed at `|omega|=0.5` | Episodes above | Sample targets above |
| ---: | ---: | ---: | ---: |
| 1 m^-1 | 0.5 m/s | {episode_bands['1.0']['count_above']} / {len(episodes)} | {sample_bands['1.0']['count_above']} / {len(samples)} |
| 2 m^-1 | 0.25 m/s | {episode_bands['2.0']['count_above']} / {len(episodes)} | {sample_bands['2.0']['count_above']} / {len(samples)} |
| 5 m^-1 | 0.1 m/s | {episode_bands['5.0']['count_above']} / {len(episodes)} | {sample_bands['5.0']['count_above']} / {len(samples)} |
| 10 m^-1 | 0.05 m/s | {episode_bands['10.0']['count_above']} / {len(episodes)} | {sample_bands['10.0']['count_above']} / {len(samples)} |

## Episode maximum by difficulty

{difficulty_lines}

## Source facts

{evidence_lines}

## Missing constraints

No inspected source defines a positive minimum stable speed, acceleration or jerk limits, a lateral-acceleration or slip limit, or a closed-loop tracking-error envelope. X-NavDP's `0.05 m/s` value is a desired-speed heuristic, not a hard lower control bound. SanD defines obstacle inflation but no velocity or turning-radius contract.

The existing CurveNav route-generation thresholds at 4 and 10 m^-1 remain quality heuristics, not physical Dingo limits. A future hard gate needs either a frozen positive `v_min` (then `kappa <= w_max / v_min`) or an empirical closed-loop tracking envelope under the active controller.
"""
    (audit_dir / "dynamics_report.md").write_text(report, encoding="utf-8")
    return summary


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset_root", type=Path)
    args = parser.parse_args(argv)
    summary = audit_dynamics(args.dataset_root)
    print(
        json.dumps(
            {
                "episodes": summary["episodes"],
                "samples": summary["samples"],
                "new_hard_gate": summary["gate_decision"]["new_source_grounded_hard_curvature_gate"],
                "episode_max_p95": summary["distributions"]["episode_maximum_curvature_per_m"]["p95"],
                "target_internal_p95_distribution_p95": summary["distributions"]["sample_target_internal_p95_curvature_per_m"]["p95"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
