from __future__ import annotations

import argparse
import csv
import gzip
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.ticker import FormatStrFormatter
import numpy as np


DEFAULT_DATE_STAMP = "2026-04-30"
PUBLISHED_V2 = Path("experiment/artifacts/published/world_model_official_baselines_v2")


@dataclass(frozen=True)
class SummaryRow:
    label: str
    eval_protocol: str
    rmse_pu: float
    mae_pu: float
    status: str
    source: str
    rmse_pu_std: float | None = None
    mae_pu_std: float | None = None
    seed_count: int = 1


LABELS = {
    "baseline_last_value_persistence_v2": "Official v2: Last value",
    "baseline_seasonal_persistence_v2": "Official v2: Seasonal",
    "baseline_ridge_residual_persistence_b0_v2": "Official v2: Ridge residual B0",
    "chronos2_official_zero_shot_b2_v2": "Official v2: Chronos-2 B2",
    "itransformer_official_target_only_residual_b0_v2": "Official v2: iTransformer residual B0",
    "timexer_official_target_only_residual_b0_v2": "Official v2: TimeXer residual B0",
    "mtgnn_official_core_calendar_residual_b1_v2": "Official v2: MTGNN residual B1",
    "dgcrn_official_core_residual_b2_v2": "Official v2: DGCRN rescue B2",
    "tft_pf_residual_b2_v2": "Official v2: TFT-PF B2",
    "world_model_persistence_last_value_v1_farm_sync": "Baseline v1: Persistence",
    "world_model_shared_weight_tft_no_graph_v1_farm_sync": "Baseline v1: TFT",
    "world_model_shared_weight_timexer_no_graph_v1_farm_sync": "Baseline v1: TimeXer",
    "world_model_dgcrn_v1_farm_sync": "Baseline v1: DGCRN",
    "world_model_chronos_2_zero_shot_v1_farm_sync": "Baseline v1: Chronos-2",
    "world_model_itransformer_no_graph_v1_farm_sync": "Baseline v1: iTransformer",
    "world_model_mtgnn_calendar_graph_v1_farm_sync": "Baseline v1: MTGNN",
    "world_model_state_space_v1_farm_sync": "State-space: Official canonical",
}

SCRATCH_SOURCES = {
    "experiment/artifacts/scratch/world_model_state_space_v1/kelmarsh_ablation_batch.csv": {
        "world_model_state_space_v1_no_farm_aux_farm_sync": "State-space: Best ablation (no_farm_aux)",
    },
    "experiment/artifacts/scratch/world_model_state_space_v1/residual_head_plan_20260419_farm0p05/train_outputs/repro_residual_farm0p05_val_rmse_pu.csv": {
        None: "State-space: Residual scratch (farm=0.05)",
    },
    "experiment/artifacts/scratch/world_model_state_space_v1/residual_head_plan_20260419_farm0p05/train_outputs/dual_residual_farm0p00_val_rmse_pu.csv": {
        None: "State-space: Residual scratch (farm=0.0)",
    },
    "experiment/artifacts/scratch/world_model_state_space_v1/global_local_ablation_20260422_farm0p00_val_rmse/train_outputs/global_local_residual_farm0p00_val_rmse_pu.csv": {
        None: "State-space: Global-local residual",
    },
    "experiment/artifacts/scratch/world_model_state_space_v1/global_local_ablation_20260422_farm0p00_val_rmse/train_outputs/global_local_increment_farm0p00_val_rmse_pu.csv": {
        None: "State-space: Global-local increment",
    },
    "experiment/artifacts/scratch/world_model_state_space_v1/ramp_v2_ablation_20260422_residual_farm0p00_val_rmse/train_outputs/residual_farm0p00_ramp0p00_val_rmse_pu.csv": {
        None: "State-space: Ramp v2 residual (0.00)",
    },
    "experiment/artifacts/scratch/world_model_state_space_v1/ramp_v2_ablation_20260422_residual_farm0p00_val_rmse/train_outputs/residual_farm0p00_ramp0p02_val_rmse_pu.csv": {
        None: "State-space: Ramp v2 residual (0.02)",
    },
    "experiment/artifacts/scratch/world_model_state_space_v1/ramp_v2_ablation_20260422_residual_farm0p00_val_rmse/train_outputs/residual_farm0p00_ramp0p05_val_rmse_pu.csv": {
        None: "State-space: Ramp v2 residual (0.05)",
    },
    "experiment/artifacts/scratch/world_model_state_space_v1/edge_gating_2x2_20260423_residual_farm0p00_val_rmse/train_outputs/residual_baseline_farm0p00_val_rmse_pu.csv": {
        None: "State-space: Edge residual control",
    },
    "experiment/artifacts/scratch/world_model_state_space_v1/edge_gating_2x2_20260423_residual_farm0p00_val_rmse/train_outputs/gated_sum_farm0p00_val_rmse_pu.csv": {
        None: "State-space: Edge gated sum",
    },
    "experiment/artifacts/scratch/world_model_state_space_v1/edge_gating_2x2_20260423_residual_farm0p00_val_rmse/train_outputs/rotor_units_wake_farm0p00_val_rmse_pu.csv": {
        None: "State-space: Edge rotor wake",
    },
    "experiment/artifacts/scratch/world_model_state_space_v1/edge_gating_2x2_20260423_residual_farm0p00_val_rmse/train_outputs/gated_sum_rotor_units_wake_farm0p00_val_rmse_pu.csv": {
        None: "State-space: Edge gated + rotor wake",
    },
}

SERIES_ORDER = [
    "Official v2: Last value",
    "Official v2: Seasonal",
    "Official v2: Ridge residual B0",
    "Official v2: Chronos-2 B2",
    "Official v2: iTransformer residual B0",
    "Official v2: TimeXer residual B0",
    "Official v2: MTGNN residual B1",
    "Official v2: DGCRN rescue B2",
    "Official v2: TFT-PF B2",
    "State-space: Official canonical",
    "State-space: Best ablation (no_farm_aux)",
    "State-space: Residual scratch (farm=0.05)",
    "State-space: Residual scratch (farm=0.0)",
    "State-space: Global-local residual",
    "State-space: Global-local increment",
    "State-space: Ramp v2 residual (0.00)",
    "State-space: Ramp v2 residual (0.02)",
    "State-space: Ramp v2 residual (0.05)",
    "State-space: Edge residual control",
    "State-space: Edge gated sum",
    "State-space: Edge rotor wake",
    "State-space: Edge gated + rotor wake",
]

COLORS = {
    "Official v2: Last value": "#2E5EAA",
    "Official v2: Seasonal": "#87919E",
    "Official v2: Ridge residual B0": "#344E41",
    "Official v2: Chronos-2 B2": "#C84C5A",
    "Official v2: iTransformer residual B0": "#D99C1F",
    "Official v2: TimeXer residual B0": "#A35C2C",
    "Official v2: MTGNN residual B1": "#2F8F83",
    "Official v2: DGCRN rescue B2": "#8F6BB3",
    "Official v2: TFT-PF B2": "#4C8C4A",
    "State-space: Official canonical": "#7A3E00",
    "State-space: Best ablation (no_farm_aux)": "#A0446A",
    "State-space: Residual scratch (farm=0.05)": "#B3265D",
    "State-space: Residual scratch (farm=0.0)": "#E06C2F",
    "State-space: Global-local residual": "#476C9B",
    "State-space: Global-local increment": "#6C5B7B",
    "State-space: Ramp v2 residual (0.00)": "#5A7D2B",
    "State-space: Ramp v2 residual (0.02)": "#3B8C88",
    "State-space: Ramp v2 residual (0.05)": "#0EAD69",
    "State-space: Edge residual control": "#4D5566",
    "State-space: Edge gated sum": "#D1663F",
    "State-space: Edge rotor wake": "#2A9D8F",
    "State-space: Edge gated + rotor wake": "#355C7D",
}


def read_csv(path: Path) -> list[dict[str, str]]:
    resolved = path if path.exists() else Path(f"{path}.gz")
    if not resolved.exists():
        return []
    opener = gzip.open if resolved.suffix == ".gz" else Path.open
    with opener(resolved, "rt", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def as_float(row: dict[str, str], *names: str) -> float | None:
    for name in names:
        raw = row.get(name)
        if raw not in (None, ""):
            return float(raw)
    return None


def as_int(row: dict[str, str], *names: str, default: int = 1) -> int:
    for name in names:
        raw = row.get(name)
        if raw not in (None, ""):
            return int(float(raw))
    return default


def row_label(row: dict[str, str], source_label_map: dict[str | None, str] | None = None) -> str | None:
    variant = row.get("model_variant", "")
    if source_label_map:
        return source_label_map.get(variant) or source_label_map.get(None)
    return LABELS.get(variant)


def collect_csv_rows(
    root: Path,
    relpath: str,
    *,
    source_label_map: dict[str | None, str] | None = None,
    status: str = "main",
) -> list[SummaryRow]:
    out: list[SummaryRow] = []
    for row in read_csv(root / relpath):
        if row.get("split_name") != "test":
            continue
        if row.get("metric_scope") != "overall":
            continue
        if row.get("eval_protocol") not in ("rolling_origin_no_refit", "non_overlap"):
            continue
        label = row_label(row, source_label_map)
        rmse = as_float(row, "rmse_pu_mean", "rmse_pu")
        mae = as_float(row, "mae_pu_mean", "mae_pu")
        rmse_std = as_float(row, "rmse_pu_std")
        mae_std = as_float(row, "mae_pu_std")
        if label is None or rmse is None or mae is None:
            continue
        row_status = row.get("main_table_status") or status
        out.append(
            SummaryRow(
                label,
                row["eval_protocol"],
                rmse,
                mae,
                row_status,
                relpath,
                rmse_pu_std=rmse_std,
                mae_pu_std=mae_std,
                seed_count=as_int(row, "seed_count", default=1),
            )
        )
    return out


def collect_tft_diagnostic(root: Path) -> list[SummaryRow]:
    aggregate = root / (
        "experiment/artifacts/scratch/world_model_official_baselines_v2/"
        "tft_pf_test_rolling_shards_full_20260425/"
        "tft_pf_residual_kelmarsh_test_rolling_origin_no_refit_aggregate.json"
    )
    if not aggregate.exists():
        return []
    payload = json.loads(aggregate.read_text(encoding="utf-8"))
    metrics = payload.get("metrics", payload)
    return [
        SummaryRow(
            "Official v2: TFT-PF B2",
            "rolling_origin_no_refit",
            float(metrics["rmse_pu"]),
            float(metrics["mae_pu"]),
            "appendix_diagnostic",
            str(aggregate.relative_to(root)),
            seed_count=1,
        )
    ]


def collect_rows(root: Path, include_state_space: bool) -> list[SummaryRow]:
    rows: list[SummaryRow] = []
    rows.extend(collect_csv_rows(root, str(PUBLISHED_V2 / "20260425-paper-grade-final-table.csv")))
    rows.extend(
        collect_csv_rows(
            root,
            str(PUBLISHED_V2 / "20260425-paper-grade-dgcrn-rescue-statistics-summary.csv"),
            status="appendix_diagnostic",
        )
    )
    rows.extend(collect_tft_diagnostic(root))

    if include_state_space:
        rows.extend(collect_csv_rows(root, "experiment/artifacts/published/world_model_state_space_v1/latest.csv"))
        for relpath, label_map in SCRATCH_SOURCES.items():
            rows.extend(collect_csv_rows(root, relpath, source_label_map=label_map, status="diagnostic"))

    # Prefer the latest official v2 label when duplicate labels/protocols appear.
    dedup: dict[tuple[str, str], SummaryRow] = {}
    for row in rows:
        dedup[(row.label, row.eval_protocol)] = row
    return list(dedup.values())


def grouped_for_plot(rows: Iterable[SummaryRow]) -> list[dict[str, object]]:
    grouped: dict[tuple[str, str], SummaryRow] = {(r.label, r.eval_protocol): r for r in rows}
    plot_rows: list[dict[str, object]] = []
    seen = set(SERIES_ORDER)
    ordered_labels = [label for label in SERIES_ORDER if any(r.label == label for r in rows)]
    ordered_labels.extend(sorted({r.label for r in rows if r.label not in seen}))
    for label in ordered_labels:
        rolling = grouped.get((label, "rolling_origin_no_refit"))
        non_overlap = grouped.get((label, "non_overlap"))
        if rolling is None and non_overlap is None:
            continue
        plot_rows.append(
            {
                "label": label,
                "rolling_rmse": rolling.rmse_pu if rolling else np.nan,
                "rolling_mae": rolling.mae_pu if rolling else np.nan,
                "non_overlap_rmse": non_overlap.rmse_pu if non_overlap else np.nan,
                "non_overlap_mae": non_overlap.mae_pu if non_overlap else np.nan,
                "rolling_rmse_std": rolling.rmse_pu_std if rolling and rolling.rmse_pu_std is not None else np.nan,
                "rolling_mae_std": rolling.mae_pu_std if rolling and rolling.mae_pu_std is not None else np.nan,
                "non_overlap_rmse_std": (
                    non_overlap.rmse_pu_std if non_overlap and non_overlap.rmse_pu_std is not None else np.nan
                ),
                "non_overlap_mae_std": (
                    non_overlap.mae_pu_std if non_overlap and non_overlap.mae_pu_std is not None else np.nan
                ),
                "rolling_seed_count": rolling.seed_count if rolling else 0,
                "non_overlap_seed_count": non_overlap.seed_count if non_overlap else 0,
                "status": (rolling or non_overlap).status,
                "color": COLORS.get(label, "#6B7280"),
            }
        )
    return plot_rows


def write_summary_csv(plot_rows: list[dict[str, object]], output_path: Path) -> None:
    fieldnames = [
        "label",
        "status",
        "rolling_rmse",
        "rolling_rmse_std",
        "rolling_mae",
        "rolling_mae_std",
        "rolling_seed_count",
        "non_overlap_rmse",
        "non_overlap_rmse_std",
        "non_overlap_mae",
        "non_overlap_mae_std",
        "non_overlap_seed_count",
    ]
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in plot_rows:
            writer.writerow({name: row.get(name, "") for name in fieldnames})


def finite(values: Iterable[float]) -> list[float]:
    return [float(v) for v in values if np.isfinite(float(v))]


def axis_limits(values: list[float]) -> tuple[float, float]:
    vmin = min(values)
    vmax = max(values)
    pad = max((vmax - vmin) * 0.08, 0.0015)
    return max(0.0, vmin - pad), vmax + pad


def annotate(ax, values: list[float], y_positions: np.ndarray, offset: float) -> None:
    for value, y in zip(values, y_positions):
        if not np.isfinite(value):
            continue
        ax.text(value + offset, y, f"{value:.6f}", va="center", ha="left", fontsize=8.2, color="#222222")


def plot(plot_rows: list[dict[str, object]], output_png: Path, output_svg: Path, *, log_x: bool = False) -> None:
    labels = [str(row["label"]) for row in plot_rows]
    y = np.arange(len(plot_rows))
    height = 0.36
    fig_height = max(10.0, 0.54 * len(plot_rows) + 3.7)
    fig, axes = plt.subplots(1, 2, figsize=(18.6, fig_height), sharey=True)
    fig.patch.set_facecolor("#F7F3EA")

    rmse_values = finite(
        [float(row["rolling_rmse"]) for row in plot_rows] + [float(row["non_overlap_rmse"]) for row in plot_rows]
    )
    mae_values = finite(
        [float(row["rolling_mae"]) for row in plot_rows] + [float(row["non_overlap_mae"]) for row in plot_rows]
    )
    panels = [
        ("RMSE", "rolling_rmse", "non_overlap_rmse", axes[0], *axis_limits(rmse_values)),
        ("MAE", "rolling_mae", "non_overlap_mae", axes[1], *axis_limits(mae_values)),
    ]

    for title, rolling_key, non_overlap_key, ax, xmin, xmax in panels:
        ax.set_facecolor("#FFFDFC")
        colors = [str(row["color"]) for row in plot_rows]
        rolling = [float(row[rolling_key]) for row in plot_rows]
        non_overlap = [float(row[non_overlap_key]) for row in plot_rows]

        ax.barh(
            y - height / 2,
            rolling,
            height=height,
            color=colors,
            alpha=0.92,
            edgecolor="#3A312A",
            linewidth=0.6,
            label="Rolling",
        )
        ax.barh(
            y + height / 2,
            non_overlap,
            height=height,
            color=colors,
            alpha=0.46,
            edgecolor="#3A312A",
            linewidth=0.6,
            hatch="///",
            label="Non-overlap",
        )

        annotate(ax, rolling, y - height / 2, 0.00015)
        annotate(ax, non_overlap, y + height / 2, 0.00015)
        ax.set_title(title, fontsize=16, weight="bold", color="#221D18")
        ax.set_xlim(xmin, xmax)
        if log_x:
            ax.set_xscale("log")
            ax.xaxis.set_major_formatter(FormatStrFormatter("%.3f"))
        ax.grid(axis="x", linestyle="--", linewidth=0.7, alpha=0.35)
        ax.set_axisbelow(True)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color("#6D655D")
        ax.spines["bottom"].set_color("#6D655D")
        ax.tick_params(axis="x", labelsize=10, colors="#2A251F")
        ax.tick_params(axis="y", labelsize=9.4, colors="#2A251F")

    axes[0].set_yticks(y)
    axes[0].set_yticklabels(labels)
    axes[1].tick_params(axis="y", left=False, labelleft=False)
    axes[0].invert_yaxis()
    handles, legend_labels = axes[0].get_legend_handles_labels()
    fig.legend(handles[:2], legend_labels[:2], loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 0.956))
    title_suffix = " (Log X)" if log_x else ""
    fig.suptitle(
        f"Kelmarsh Full-Test Summary: Official Baselines v2, DGCRN Rescue, and State-Space Variants{title_suffix}",
        fontsize=18,
        weight="bold",
        color="#1D1814",
        y=0.988,
    )
    fig.text(
        0.5,
        0.018,
        "Hatched bars are non-overlap. DGCRN/TFT rescue rows are diagnostic; main-table rows use validation-only selection and frozen test-once evidence."
        + (" X axis uses log scale." if log_x else ""),
        ha="center",
        fontsize=10,
        color="#4F463E",
    )
    fig.tight_layout(rect=(0.0, 0.04, 1.0, 0.952))
    fig.savefig(output_png, dpi=220, bbox_inches="tight")
    fig.savefig(output_svg, bbox_inches="tight")


OURS_SOURCE_LABEL = "State-space: Ramp v2 residual (0.05)"
OURS_DISPLAY_LABEL = "★ Ours: State-space ramp v2 (0.05)"
OFFICIAL_COMPARISON_ORDER = [
    "Official v2: Ridge residual B0",
    "Official v2: TimeXer residual B0",
    "Official v2: iTransformer residual B0",
    "Official v2: DGCRN rescue B2",
    "Official v2: MTGNN residual B1",
    "Official v2: Chronos-2 B2",
    "Official v2: Last value",
    "Official v2: Seasonal",
]
OFFICIAL_DISPLAY_LABELS = {
    "Official v2: Ridge residual B0": "Ridge residual B0",
    "Official v2: TimeXer residual B0": "TimeXer residual B0",
    "Official v2: iTransformer residual B0": "iTransformer residual B0",
    "Official v2: DGCRN rescue B2": "DGCRN rescue B2 (diag.)",
    "Official v2: MTGNN residual B1": "MTGNN residual B1",
    "Official v2: Chronos-2 B2": "Chronos-2 B2",
    "Official v2: Last value": "Last-value persistence",
    "Official v2: Seasonal": "Seasonal persistence",
}


def _row_by_label(plot_rows: list[dict[str, object]]) -> dict[str, dict[str, object]]:
    return {str(row["label"]): row for row in plot_rows}


def build_ours_vs_official_rows(plot_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    by_label = _row_by_label(plot_rows)
    if OURS_SOURCE_LABEL not in by_label:
        return []
    rows: list[dict[str, object]] = []
    ours = dict(by_label[OURS_SOURCE_LABEL])
    ours["display_label"] = OURS_DISPLAY_LABEL
    ours["is_ours"] = True
    rows.append(ours)
    for label in OFFICIAL_COMPARISON_ORDER:
        row = by_label.get(label)
        if row is None:
            continue
        if not np.isfinite(float(row["rolling_rmse"])) or not np.isfinite(float(row["non_overlap_rmse"])):
            continue
        enriched = dict(row)
        enriched["display_label"] = OFFICIAL_DISPLAY_LABELS.get(label, label)
        enriched["is_ours"] = False
        rows.append(enriched)
    return rows


def _break_limits(
    values: list[float], seasonal_values: list[float] | None
) -> tuple[tuple[float, float], tuple[float, float] | None]:
    finite_values = sorted(v for v in values if np.isfinite(v))
    low_min = min(finite_values) * 0.96
    finite_seasonal = [v for v in (seasonal_values or []) if np.isfinite(v)]
    if not finite_seasonal:
        return (low_min, max(finite_values) * 1.04), None
    seasonal_min = min(finite_seasonal)
    seasonal_max = max(finite_seasonal)
    low_values = [v for v in finite_values if v < seasonal_min * 0.75]
    low_max = max(low_values) * 1.045
    high_min = seasonal_min * 0.975
    high_max = seasonal_max * 1.012
    return (low_min, low_max), (high_min, high_max)


def _draw_break_marks(left_ax, right_ax) -> None:
    d = 0.007
    kwargs = {"color": "#2A251F", "clip_on": False, "linewidth": 0.85, "alpha": 0.82}
    left_ax.plot((1 - d, 1 + d), (-d, +d), transform=left_ax.transAxes, **kwargs)
    right_ax.plot((-d, +d), (-d, +d), transform=right_ax.transAxes, **kwargs)


def _metric_std(row: dict[str, object], key: str) -> float:
    std_key = f"{key}_std"
    seed_key = "rolling_seed_count" if key.startswith("rolling") else "non_overlap_seed_count"
    seed_count = int(row.get(seed_key, 0) or 0)
    if seed_count <= 1:
        return 0.0
    raw = row.get(std_key, np.nan)
    value = float(raw)
    return value if np.isfinite(value) else 0.0


def _annotate_broken_bars(
    left_ax,
    right_ax,
    values: list[float],
    y_positions: np.ndarray,
    ours_value: float,
    left_xlim: tuple[float, float],
    high_xlim: tuple[float, float] | None,
) -> None:
    for value, y in zip(values, y_positions):
        pct = (value / ours_value - 1.0) * 100.0
        label = f"{value:.6f}" if abs(pct) < 1e-9 else f"{value:.6f}  (+{pct:.1f}%)"
        if high_xlim is not None and value >= high_xlim[0]:
            ax = right_ax
            offset = (high_xlim[1] - high_xlim[0]) * 0.018
            ax.text(value - offset, y, label, va="center", ha="right", fontsize=8.5, color="#222222")
        else:
            ax = left_ax
            offset = (left_xlim[1] - left_xlim[0]) * 0.012
            ax.text(value + offset, y, label, va="center", ha="left", fontsize=8.5, color="#222222")


def _plot_metric_group(
    fig,
    gridspec,
    grid_offset: int,
    rows: list[dict[str, object]],
    *,
    title: str,
    rolling_key: str,
    non_overlap_key: str,
    labels: list[str],
    colors: list[str],
) -> tuple[object, object]:
    left_ax = fig.add_subplot(gridspec[0, grid_offset])
    right_ax = fig.add_subplot(gridspec[0, grid_offset + 1], sharey=left_ax)
    y = np.arange(len(rows))
    height = 0.34
    rolling_y = y - height / 2
    non_overlap_y = y + height / 2
    rolling_values = [float(row[rolling_key]) for row in rows]
    non_overlap_values = [float(row[non_overlap_key]) for row in rows]
    rolling_std = [_metric_std(row, rolling_key) for row in rows]
    non_overlap_std = [_metric_std(row, non_overlap_key) for row in rows]
    seasonal_values = [
        float(row[key])
        for row in rows
        if str(row["display_label"]) == "Seasonal persistence"
        for key in (rolling_key, non_overlap_key)
    ]
    left_xlim, right_xlim = _break_limits(rolling_values + non_overlap_values, seasonal_values)

    for ax in (left_ax, right_ax):
        ax.set_facecolor("#FFFDFC")
        ax.barh(
            rolling_y,
            rolling_values,
            height=height,
            color=colors,
            alpha=0.91,
            edgecolor="#3A312A",
            linewidth=0.6,
            xerr=rolling_std,
            error_kw={"elinewidth": 0.9, "ecolor": "#2A251F", "capsize": 2.5, "capthick": 0.8},
            label="Rolling",
        )
        ax.barh(
            non_overlap_y,
            non_overlap_values,
            height=height,
            color=colors,
            alpha=0.54,
            edgecolor="#3A312A",
            linewidth=0.6,
            hatch="///",
            xerr=non_overlap_std,
            error_kw={"elinewidth": 0.9, "ecolor": "#2A251F", "capsize": 2.5, "capthick": 0.8},
            label="Non-overlap",
        )
        ax.grid(axis="x", linestyle="--", linewidth=0.7, alpha=0.35)
        ax.set_axisbelow(True)
        ax.spines["top"].set_visible(False)
        ax.spines["bottom"].set_color("#6D655D")
        ax.tick_params(axis="x", labelsize=9, colors="#2A251F")
        ax.tick_params(axis="y", labelsize=9.8, colors="#2A251F")

    left_ax.set_xlim(*left_xlim)
    if right_xlim is not None:
        right_ax.set_xlim(*right_xlim)
        right_ax.set_xticks([max(seasonal_values)])
        right_ax.xaxis.set_major_formatter(FormatStrFormatter("%.3f"))
    right_ax.tick_params(axis="y", left=False, labelleft=False)
    left_ax.spines["right"].set_visible(False)
    right_ax.spines["left"].set_visible(False)
    _draw_break_marks(left_ax, right_ax)

    left_ax.set_title(title, fontsize=15, weight="bold", color="#221D18")
    _annotate_broken_bars(
        left_ax,
        right_ax,
        rolling_values,
        rolling_y,
        float(rows[0][rolling_key]),
        left_xlim,
        right_xlim,
    )
    _annotate_broken_bars(
        left_ax,
        right_ax,
        non_overlap_values,
        non_overlap_y,
        float(rows[0][non_overlap_key]),
        left_xlim,
        right_xlim,
    )
    left_ax.set_yticks(y)
    left_ax.set_yticklabels(labels)
    left_ax.invert_yaxis()
    return left_ax, right_ax


def plot_ours_vs_official_broken(
    rows: list[dict[str, object]],
    output_png: Path,
    output_svg: Path,
) -> None:
    if not rows:
        return
    labels = [str(row["display_label"]) for row in rows]
    colors = ["#B3265D" if bool(row["is_ours"]) else str(row["color"]) for row in rows]
    fig = plt.figure(figsize=(20.8, 8.6), facecolor="#F7F3EA")
    gs = fig.add_gridspec(1, 4, width_ratios=[4.9, 1.05, 4.9, 1.05], wspace=0.07)
    rmse_left, _ = _plot_metric_group(
        fig,
        gs,
        0,
        rows,
        title="RMSE",
        rolling_key="rolling_rmse",
        non_overlap_key="non_overlap_rmse",
        labels=labels,
        colors=colors,
    )
    mae_left, _ = _plot_metric_group(
        fig,
        gs,
        2,
        rows,
        title="MAE",
        rolling_key="rolling_mae",
        non_overlap_key="non_overlap_mae",
        labels=labels,
        colors=colors,
    )
    mae_left.tick_params(axis="y", left=False, labelleft=False)

    fig.suptitle(
        "Kelmarsh: ★ Ours vs Official Baselines v2",
        fontsize=19,
        weight="bold",
        color="#1D1814",
        y=0.982,
    )
    legend_handles = [
        Patch(facecolor="#777777", edgecolor="#3A312A", alpha=0.91, label="Rolling"),
        Patch(facecolor="#777777", edgecolor="#3A312A", alpha=0.54, hatch="///", label="Non-overlap"),
    ]
    fig.legend(legend_handles, ["Rolling", "Non-overlap"], loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 0.925))
    fig.text(
        0.5,
        0.042,
        "Lower is better. Solid bars are rolling; hatched bars are non-overlap. Labels show absolute metric and relative increase vs ★ Ours for the same protocol. Error bars show ±1 seed std where seed_count > 1.",
        ha="center",
        fontsize=10.5,
        color="#4F463E",
    )
    fig.subplots_adjust(left=0.20, right=0.985, top=0.84, bottom=0.13)
    fig.savefig(output_png, dpi=220, bbox_inches="tight")
    fig.savefig(output_svg, bbox_inches="tight")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Make RMSE/MAE summary bars from local Kelmarsh experiment artifacts.")
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--date-stamp", default=DEFAULT_DATE_STAMP)
    parser.add_argument("--official-only", action="store_true", help="Only plot official-baseline v2 rows.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    output_dir = (args.output_dir or root / "figures").resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_rows = grouped_for_plot(collect_rows(root, include_state_space=not args.official_only))
    if not plot_rows:
        raise SystemExit("No report rows found.")

    csv_path = output_dir / f"{args.date_stamp}-experiment-summary-bars.csv"
    png_path = output_dir / f"{args.date_stamp}-experiment-summary-bars.png"
    svg_path = output_dir / f"{args.date_stamp}-experiment-summary-bars.svg"
    log_png_path = output_dir / f"{args.date_stamp}-experiment-summary-bars-logx.png"
    log_svg_path = output_dir / f"{args.date_stamp}-experiment-summary-bars-logx.svg"
    ours_png_path = output_dir / f"{args.date_stamp}-ours-vs-official-v2-summary-bars-broken.png"
    ours_svg_path = output_dir / f"{args.date_stamp}-ours-vs-official-v2-summary-bars-broken.svg"
    write_summary_csv(plot_rows, csv_path)
    plot(plot_rows, png_path, svg_path)
    plot(plot_rows, log_png_path, log_svg_path, log_x=True)
    plot_ours_vs_official_broken(build_ours_vs_official_rows(plot_rows), ours_png_path, ours_svg_path)
    print(csv_path)
    print(png_path)
    print(svg_path)
    print(log_png_path)
    print(log_svg_path)
    print(ours_png_path)
    print(ours_svg_path)


if __name__ == "__main__":
    main()
