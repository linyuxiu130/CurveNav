"""Compact artifacts for inspecting strict offline evaluation cases."""

from __future__ import annotations

import html
import json
import math
from pathlib import Path

import numpy as np
import torch
from torch import Tensor

from curvenav.data.privileged import SourceConfigurationSpaceQuery
from curvenav.evaluation.protocol import evaluation_strata


def _representative_index(mask: Tensor, score: Tensor, quantile: float) -> int | None:
    indices = mask.nonzero(as_tuple=False).flatten()
    if not len(indices):
        return None
    ordered = indices[score[indices].argsort()]
    position = round(quantile * (len(ordered) - 1))
    return int(ordered[position].item())


def select_cases(metrics: dict[str, Tensor]) -> list[tuple[str, int]]:
    """Choose deterministic behavior and collision-diagnosis cases."""
    strata = evaluation_strata(metrics)
    requested = (
        ("forward_direct_median", "forward_direct", 0.5),
        ("forward_detour_median", "forward_detour", 0.5),
        ("forward_detour_hard", "forward_detour", 1.0),
        ("rear_goal_median", "rear_goal", 0.5),
        ("expert_moves_away_median", "expert_moves_away_from_goal", 0.5),
    )
    selected = []
    for label, stratum, quantile in requested:
        index = _representative_index(
            strata[stratum], metrics["fixed_horizon_ade_m"], quantile
        )
        if index is not None and index not in {item[1] for item in selected}:
            selected.append((label, index))
    diagnostic_cases = (
        (
            "current_visible_collision",
            "first_collision_current_depth_visible",
            "distance_to_first_collision_m",
        ),
        (
            "history_only_visible_collision",
            "first_collision_history_only_depth_visible",
            "distance_to_first_collision_m",
        ),
        (
            "four_frame_unrecognized_collision",
            "first_collision_unrecognized_by_full_depth",
            "distance_to_first_collision_m",
        ),
        (
            "endpoint_collision",
            "path_endpoint_collision",
            "min_clearance_m",
        ),
        (
            "execution_prefix_collision_earliest",
            "execution_prefix_1p0m_collision",
            "distance_to_first_collision_m",
        ),
    )
    for label, mask_name, score_name in diagnostic_cases:
        index = _representative_index(
            metrics[mask_name].bool(), metrics[score_name], 0.0
        )
        if index is not None and index not in {item[1] for item in selected}:
            selected.append((label, index))
    return selected


def _write_case_visualization(records: list[dict[str, object]], path: Path) -> None:
    """Render selected cases as one clean, metric local-navigation figure."""
    panel_width = 440
    panel_height = 426
    plot_size = 344
    plot_left = 48
    plot_top = 48
    legend_height = 54
    columns = 2
    rows = math.ceil(len(records) / columns)
    parts = [
        f'<svg id="curvenav-collision-bev" xmlns="http://www.w3.org/2000/svg" '
        f'width="{columns * panel_width}" '
        f'height="{legend_height + rows * panel_height}" viewBox="0 0 '
        f'{columns * panel_width} {legend_height + rows * panel_height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<style>'
        'text{font-family:Inter,ui-sans-serif,system-ui,-apple-system,sans-serif;fill:#334155}'
        '.title{font-size:13px;font-weight:650;fill:#0f172a}'
        '.metric{font-size:10px;fill:#64748b}'
        '.axis{font-size:9px;fill:#94a3b8}'
        '</style>',
        '<g font-size="10">',
        '<rect x="14" y="12" width="11" height="11" rx="2" fill="#fecaca"/>'
        '<text x="30" y="21">source collision</text>',
        '<rect x="132" y="12" width="11" height="11" rx="2" fill="#fed7aa"/>'
        '<text x="148" y="21">0.10 m margin</text>',
        '<rect x="257" y="12" width="11" height="11" rx="2" fill="#0891b2"/>'
        '<text x="273" y="21">current obstacle</text>',
        '<rect x="385" y="12" width="11" height="11" rx="2" fill="#7c3aed"/>'
        '<text x="401" y="21">history obstacle</text>',
        '<line x1="515" y1="18" x2="533" y2="18" stroke="#16a34a" stroke-width="3"/>'
        '<text x="538" y="21">expert</text>',
        '<line x1="588" y1="18" x2="606" y2="18" stroke="#2563eb" stroke-width="3"/>'
        '<text x="611" y="21">4-frame</text>',
        '<line x1="674" y1="18" x2="692" y2="18" stroke="#f59e0b" stroke-width="3"/>'
        '<text x="697" y="21">current-only</text>',
        '<circle cx="790" cy="18" r="4" fill="#dc2626"/>'
        '<text x="799" y="21">visible hit</text>',
        '<path d="M 14 39 l 5 -5 l 5 5 l -5 5 z" fill="#7c3aed"/>'
        '<text x="30" y="42">history hit</text>',
        '<path d="M 118 34 l 10 10 M 128 34 l -10 10" stroke="#111827" stroke-width="2"/>'
        '<text x="134" y="42">unseen hit</text>',
        '</g>',
    ]
    for panel, record in enumerate(records):
        panel_x = (panel % columns) * panel_width
        panel_y = legend_height + (panel // columns) * panel_height
        left = panel_x + plot_left
        top = panel_y + plot_top
        extent = float(record["configuration_extent_m"])
        coverage = np.asarray(record["configuration_ray_coverage"], dtype=np.bool_)
        current_coverage = np.asarray(
            record["current_configuration_ray_coverage"], dtype=np.bool_
        )
        history_coverage = coverage & ~current_coverage
        full_forbidden = np.asarray(record["configuration_forbidden"], dtype=np.bool_)
        current_forbidden = np.asarray(
            record["current_configuration_forbidden"], dtype=np.bool_
        )
        history_forbidden = full_forbidden & ~current_forbidden
        source_clearance = np.asarray(record["source_clearance_m"], dtype=np.float64)
        height, width = coverage.shape
        cell_x = plot_size / width
        cell_y = plot_size / height
        title = str(record["label"]).replace("_", " ")
        parts.append(
            f'<text class="title" x="{panel_x + 12}" y="{panel_y + 22}">'
            f'{html.escape(title)}</text>'
        )
        parts.append(
            f'<clipPath id="case-{panel}"><rect x="{left}" y="{top}" '
            f'width="{plot_size}" height="{plot_size}"/></clipPath>'
        )
        parts.append(
            f'<g clip-path="url(#case-{panel})"><rect x="{left}" y="{top}" '
            f'width="{plot_size}" height="{plot_size}" fill="#f8fafc"/>'
        )
        state = np.where(
            source_clearance < 0.0,
            2,
            np.where(source_clearance < 0.10, 1, 0),
        )
        colors = ("#f8fafc", "#ffedd5", "#fee2e2")
        for row in range(height):
            start = 0
            while start < width:
                value = int(state[row, start])
                end = start + 1
                while end < width and int(state[row, end]) == value:
                    end += 1
                x = left + start * cell_x
                y = top + (height - 1 - row) * cell_y
                parts.append(
                    f'<rect x="{x:.2f}" y="{y:.2f}" '
                    f'width="{(end-start)*cell_x + 0.05:.2f}" '
                    f'height="{cell_y + 0.05:.2f}" fill="{colors[value]}"/>'
                )
                start = end

        def overlay(mask: np.ndarray, color: str, opacity: float) -> None:
            for row in range(height):
                columns = np.flatnonzero(mask[row])
                if not len(columns):
                    continue
                starts = columns[np.r_[True, np.diff(columns) > 1]]
                ends = columns[np.r_[np.diff(columns) > 1, True]] + 1
                for start, end in zip(starts, ends, strict=True):
                    x = left + start * cell_x
                    y = top + (height - 1 - row) * cell_y
                    parts.append(
                        f'<rect x="{x:.2f}" y="{y:.2f}" '
                        f'width="{(end - start) * cell_x + .05:.2f}" '
                        f'height="{cell_y + .05:.2f}" fill="{color}" '
                        f'opacity="{opacity:.2f}"/>'
                    )

        # Sensor support remains subtle; saturated cells alone denote measured
        # C-space obstacles. This keeps coverage from looking like occupancy.
        overlay(current_coverage, "#0ea5e9", 0.045)
        overlay(history_coverage, "#8b5cf6", 0.055)
        overlay(current_forbidden, "#0891b2", 0.72)
        overlay(history_forbidden, "#7c3aed", 0.64)

        origin_axis_x = left + plot_size / 2.0
        origin_axis_y = top + plot_size / 2.0
        parts.extend(
            (
                f'<line x1="{left}" y1="{origin_axis_y:.2f}" x2="{left + plot_size}" '
                f'y2="{origin_axis_y:.2f}" stroke="#cbd5e1" stroke-width="0.7"/>',
                f'<line x1="{origin_axis_x:.2f}" y1="{top}" x2="{origin_axis_x:.2f}" '
                f'y2="{top + plot_size}" stroke="#cbd5e1" stroke-width="0.7"/>',
            )
        )

        def map_point(point: list[float]) -> tuple[float, float]:
            x, y = point
            return (
                left + (x + extent) / (2.0 * extent) * plot_size,
                top + (extent - y) / (2.0 * extent) * plot_size,
            )

        def polyline(name: str, color: str, width_px: float) -> None:
            points = " ".join(
                f"{x:.2f},{y:.2f}"
                for x, y in map(map_point, record[name])
            )
            parts.append(
                f'<polyline points="{points}" fill="none" stroke="{color}" '
                f'stroke-width="{width_px}" stroke-linecap="round" '
                'stroke-linejoin="round"/>'
            )

        polyline("reference_path", "#16a34a", 2.5)
        polyline("current_frame_predicted_path", "#f59e0b", 2.0)
        polyline("predicted_path", "#2563eb", 3.0)
        collision_markers = (
            ("collision_points_current_visible", "#dc2626", "circle"),
            ("collision_points_history_only_visible", "#7c3aed", "diamond"),
            ("collision_points_unrecognized", "#111827", "cross"),
        )
        for name, color, marker in collision_markers:
            for point in record[name]:
                marker_x, marker_y = map_point(point)
                if marker == "circle":
                    parts.append(
                        f'<circle cx="{marker_x:.2f}" cy="{marker_y:.2f}" r="3.0" '
                        f'fill="{color}" stroke="#ffffff" stroke-width="0.8"/>'
                    )
                elif marker == "diamond":
                    parts.append(
                        f'<path d="M {marker_x:.2f} {marker_y - 3.5:.2f} '
                        f'L {marker_x + 3.5:.2f} {marker_y:.2f} '
                        f'L {marker_x:.2f} {marker_y + 3.5:.2f} '
                        f'L {marker_x - 3.5:.2f} {marker_y:.2f} Z" '
                        f'fill="{color}" stroke="#ffffff" stroke-width="0.8"/>'
                    )
                else:
                    parts.append(
                        f'<path d="M {marker_x - 3:.2f} {marker_y - 3:.2f} '
                        f'L {marker_x + 3:.2f} {marker_y + 3:.2f} '
                        f'M {marker_x + 3:.2f} {marker_y - 3:.2f} '
                        f'L {marker_x - 3:.2f} {marker_y + 3:.2f}" '
                        f'stroke="{color}" stroke-width="1.8"/>'
                    )
        goal = np.asarray(record["point_goal"], dtype=np.float64)
        scale = min(1.0, 0.96 * extent / max(float(np.abs(goal).max()), 1e-9))
        goal_x, goal_y = map_point((goal * scale).tolist())
        origin_x, origin_y = map_point([0.0, 0.0])
        parts.extend((
            f'<line x1="{origin_x:.2f}" y1="{origin_y:.2f}" '
            f'x2="{goal_x:.2f}" y2="{goal_y:.2f}" stroke="#64748b" '
            'stroke-width="1" stroke-dasharray="4 4"/>',
            f'<rect x="{goal_x - 3.5:.2f}" y="{goal_y - 3.5:.2f}" width="7" '
            f'height="7" fill="#0f172a" stroke="#ffffff" '
            f'transform="rotate(45 {goal_x:.2f} {goal_y:.2f})"/>',
            f'<circle cx="{origin_x:.2f}" cy="{origin_y:.2f}" r="3" '
            'fill="#ffffff" stroke="#0f172a" stroke-width="1.4"/>',
            '</g>',
            f'<rect x="{left}" y="{top}" width="{plot_size}" height="{plot_size}" '
            'fill="none" stroke="#cbd5e1" stroke-width="1"/>',
            f'<text class="axis" x="{left}" y="{top + plot_size + 13}">-{extent:.1f}</text>',
            f'<text class="axis" x="{left + plot_size / 2 - 3:.2f}" '
            f'y="{top + plot_size + 13}">0</text>',
            f'<text class="axis" x="{left + plot_size - 18:.2f}" '
            f'y="{top + plot_size + 13}">+{extent:.1f} m</text>',
            f'<text class="axis" x="{left + plot_size - 18:.2f}" y="{top + 12}">y-left</text>',
        ))
        first_hit = (
            f'{float(record["distance_to_first_collision_m"]):.2f}m'
            if bool(record["footprint_collision"])
            else "none"
        )
        parts.append(
            f'<text class="metric" x="{panel_x + 12}" y="{panel_y + 410}">'
            f'ADE {float(record["fixed_horizon_ade_m"]):.3f}m | '
            f'min c* {float(record["predicted_min_clearance_m"]):.3f}m | '
            f'first hit {first_hit}'
            '</text>'
        )
        parts.append(
            f'<text class="metric" x="{panel_x + 12}" y="{panel_y + 423}">'
            f'hits current/history/unseen: {int(record["current_visible_collision_points"])} / '
            f'{int(record["history_only_visible_collision_points"])} / '
            f'{int(record["unrecognized_collision_points"])} | '
            f'endpoint {str(bool(record["path_endpoint_collision"])).lower()}'
            '</text>'
        )
    parts.append('</svg>')
    path.write_text("\n".join(parts) + "\n", encoding="utf-8")


def write_case_report(
    output_dir: Path,
    metrics: dict[str, Tensor],
    sample_data: dict[str, Tensor],
    source_query: SourceConfigurationSpaceQuery,
) -> Path:
    """Write small, renderer-independent JSON containing representative geometry."""
    output_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for label, index in select_cases(metrics):
        goal = sample_data["point_goal"][index]
        field = sample_data["configuration_field"][index].float()
        extent = float(sample_data["configuration_extent_m"][index])
        height, width = field.shape[-2:]
        x = np.linspace(-extent, extent, width, dtype=np.float64)
        y = np.linspace(-extent, extent, height, dtype=np.float64)
        local_x, local_y = np.meshgrid(x, y, indexing="xy")
        origin = sample_data["source_origin_xy"][index].double().numpy()
        yaw = float(sample_data["source_yaw_rad"][index])
        cosine, sine = math.cos(yaw), math.sin(yaw)
        world = np.stack(
            (
                origin[0] + cosine * local_x + sine * local_y,
                origin[1] + sine * local_x - cosine * local_y,
            ),
            axis=-1,
        )
        grid_index = int(sample_data["source_grid_index"][index])
        source_clearance = source_query.grids[grid_index].query_world(
            world.reshape(-1, 2)
        ).reshape(height, width)
        source_clearance = np.nan_to_num(
            source_clearance, neginf=-extent, posinf=extent
        )
        source_path = sample_data["source_collision_path_points"][index]
        current_visible = sample_data["source_collision_current_visible"][index]
        history_only_visible = sample_data[
            "source_collision_history_only_visible"
        ][index]
        unrecognized = sample_data["source_collision_unrecognized"][index]
        records.append(
            {
                "label": label,
                "sample_index": index,
                "fixed_horizon_ade_m": float(
                    metrics["fixed_horizon_ade_m"][index]
                ),
                "fixed_horizon_fde_m": float(
                    metrics["fixed_horizon_fde_m"][index]
                ),
                "horizon_coverage_fraction": float(
                    metrics["horizon_coverage_fraction"][index]
                ),
                "goal_distance_m": float(metrics["point_goal_distance_m"][index]),
                "goal_bearing_deg": float(
                    np.degrees(np.arctan2(float(goal[1]), float(goal[0])))
                ),
                "predicted_min_clearance_m": float(
                    metrics["min_clearance_m"][index]
                ),
                "footprint_collision": bool(
                    metrics["footprint_collision"][index]
                ),
                "reference_min_clearance_m": float(
                    metrics["reference_min_clearance_m"][index]
                ),
                "depth_min_clearance_m": float(
                    metrics["depth_min_clearance_m"][index]
                ),
                "distance_to_first_collision_m": float(
                    metrics["distance_to_first_collision_m"][index]
                ),
                "distance_to_first_margin_violation_m": float(
                    metrics["distance_to_first_margin_violation_m"][index]
                ),
                "execution_prefix_0p5m_collision": bool(
                    metrics["execution_prefix_0p5m_collision"][index]
                ),
                "execution_prefix_1p0m_collision": bool(
                    metrics["execution_prefix_1p0m_collision"][index]
                ),
                "path_endpoint_collision": bool(
                    metrics["path_endpoint_collision"][index]
                ),
                "terminal_0p25m_collision": bool(
                    metrics["terminal_0p25m_collision"][index]
                ),
                "mpc_desired_speed_mps": float(
                    metrics["mpc_desired_speed_mps"][index]
                ),
                "mpc_max_curvature_first12_inv_m": float(
                    metrics["mpc_max_curvature_first12_inv_m"][index]
                ),
                "configuration_extent_m": extent,
                "point_goal": goal.tolist(),
                "predicted_path": sample_data["predicted_path"][index].tolist(),
                "current_frame_predicted_path": sample_data[
                    "current_frame_predicted_path"
                ][index].tolist(),
                "reference_path": sample_data["reference_path"][index].tolist(),
                "source_clearance_m": source_clearance.tolist(),
                "configuration_clearance_m": field[0].tolist(),
                "configuration_ray_coverage": (field[1] > 0.5).tolist(),
                "current_configuration_ray_coverage": sample_data[
                    "current_configuration_ray_coverage"
                ][index].tolist(),
                "configuration_forbidden": (field[2] > 0.5).tolist(),
                "current_configuration_forbidden": sample_data[
                    "current_configuration_forbidden"
                ][index].tolist(),
                "collision_points_current_visible": source_path[
                    current_visible
                ].tolist(),
                "collision_points_history_only_visible": source_path[
                    history_only_visible
                ].tolist(),
                "collision_points_unrecognized": source_path[
                    unrecognized
                ].tolist(),
                "current_visible_collision_points": int(current_visible.sum()),
                "history_only_visible_collision_points": int(
                    history_only_visible.sum()
                ),
                "unrecognized_collision_points": int(unrecognized.sum()),
            }
        )
    path = output_dir / "offline-cases.json"
    path.write_text(json.dumps(records, indent=2) + "\n", encoding="utf-8")
    _write_case_visualization(records, output_dir / "offline-cases.svg")
    return path
