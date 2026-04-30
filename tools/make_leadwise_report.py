from __future__ import annotations

import argparse
import csv
import gzip
import json
from dataclasses import dataclass
from statistics import fmean
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter


DEFAULT_DATE_STAMP = "2026-04-30"
PUBLISHED_V2 = Path("experiment/artifacts/published/world_model_official_baselines_v2")
OFFICIAL_V2_LEADWISE_RECOVERY = Path(
    "experiment/artifacts/scratch/world_model_official_baselines_v2/leadwise_recovery_20260430"
)


@dataclass(frozen=True)
class LeadRow:
    label: str
    eval_protocol: str
    lead_step: int
    rmse_pu: float
    mae_pu: float
    source: str


@dataclass(frozen=True)
class BucketRow:
    label: str
    eval_protocol: str
    lead1_rmse: float
    short_rmse: float
    mid_rmse: float
    long_rmse: float
    status: str
    source: str


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
    "tft_pf_per_turbine_residual_b2_v2": "Official v2: TFT-PF B2",
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
    "experiment/artifacts/scratch/world_model_state_space_v1/residual_head_plan_20260419_farm0p05/train_outputs/repro_residual_farm0p05_val_rmse_pu.csv": "State-space: Residual scratch (farm=0.05)",
    "experiment/artifacts/scratch/world_model_state_space_v1/residual_head_plan_20260419_farm0p05/train_outputs/dual_residual_farm0p00_val_rmse_pu.csv": "State-space: Residual scratch (farm=0.0)",
    "experiment/artifacts/scratch/world_model_state_space_v1/global_local_ablation_20260422_farm0p00_val_rmse/train_outputs/global_local_residual_farm0p00_val_rmse_pu.csv": "State-space: Global-local residual",
    "experiment/artifacts/scratch/world_model_state_space_v1/global_local_ablation_20260422_farm0p00_val_rmse/train_outputs/global_local_increment_farm0p00_val_rmse_pu.csv": "State-space: Global-local increment",
    "experiment/artifacts/scratch/world_model_state_space_v1/ramp_v2_ablation_20260422_residual_farm0p00_val_rmse/train_outputs/residual_farm0p00_ramp0p00_val_rmse_pu.csv": "State-space: Ramp v2 residual (0.00)",
    "experiment/artifacts/scratch/world_model_state_space_v1/ramp_v2_ablation_20260422_residual_farm0p00_val_rmse/train_outputs/residual_farm0p00_ramp0p02_val_rmse_pu.csv": "State-space: Ramp v2 residual (0.02)",
    "experiment/artifacts/scratch/world_model_state_space_v1/ramp_v2_ablation_20260422_residual_farm0p00_val_rmse/train_outputs/residual_farm0p00_ramp0p05_val_rmse_pu.csv": "State-space: Ramp v2 residual (0.05)",
    "experiment/artifacts/scratch/world_model_state_space_v1/edge_gating_2x2_20260423_residual_farm0p00_val_rmse/train_outputs/residual_baseline_farm0p00_val_rmse_pu.csv": "State-space: Edge residual control",
    "experiment/artifacts/scratch/world_model_state_space_v1/edge_gating_2x2_20260423_residual_farm0p00_val_rmse/train_outputs/gated_sum_farm0p00_val_rmse_pu.csv": "State-space: Edge gated sum",
    "experiment/artifacts/scratch/world_model_state_space_v1/edge_gating_2x2_20260423_residual_farm0p00_val_rmse/train_outputs/rotor_units_wake_farm0p00_val_rmse_pu.csv": "State-space: Edge rotor wake",
    "experiment/artifacts/scratch/world_model_state_space_v1/edge_gating_2x2_20260423_residual_farm0p00_val_rmse/train_outputs/gated_sum_rotor_units_wake_farm0p00_val_rmse_pu.csv": "State-space: Edge gated + rotor wake",
}

SERIES_ORDER = [
    "Official v2: Last value",
    "Official v2: Ridge residual B0",
    "Official v2: Chronos-2 B2",
    "Official v2: iTransformer residual B0",
    "Official v2: TimeXer residual B0",
    "Official v2: DGCRN rescue B2",
    "Official v2: MTGNN residual B1",
    "Official v2: TFT-PF B2",
    "Official v2: Seasonal",
    "Baseline v1: Persistence",
    "Baseline v1: TFT",
    "Baseline v1: TimeXer",
    "Baseline v1: DGCRN",
    "Baseline v1: Chronos-2",
    "Baseline v1: iTransformer",
    "Baseline v1: MTGNN",
    "State-space: Official canonical",
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
    "Baseline v1: Persistence": "#7C99C7",
    "Baseline v1: TFT": "#8CB989",
    "Baseline v1: TimeXer": "#C88A5B",
    "Baseline v1: DGCRN": "#B09AD1",
    "Baseline v1: Chronos-2": "#DB7A84",
    "Baseline v1: iTransformer": "#E4B64F",
    "Baseline v1: MTGNN": "#72B6AD",
    "State-space: Official canonical": "#7A3E00",
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


def collect_lead_rows(root: Path, *, split_name: str = "test") -> list[LeadRow]:
    rows: list[LeadRow] = []
    sources: list[tuple[str, str | None]] = [
        ("experiment/artifacts/published/world_model_baselines_v1/latest.csv", None),
        ("experiment/artifacts/published/world_model_state_space_v1/latest.csv", None),
    ]
    sources.extend((rel, label) for rel, label in SCRATCH_SOURCES.items())

    for relpath, forced_label in sources:
        for row in read_csv(root / relpath):
            if row.get("split_name") != split_name:
                continue
            if row.get("metric_scope") != "horizon":
                continue
            if row.get("eval_protocol") not in ("rolling_origin_no_refit", "non_overlap"):
                continue
            label = forced_label or LABELS.get(row.get("model_variant", ""))
            rmse = as_float(row, "rmse_pu")
            mae = as_float(row, "mae_pu")
            lead_raw = row.get("lead_step")
            if label is None or rmse is None or mae is None or not lead_raw:
                continue
            rows.append(LeadRow(label, row["eval_protocol"], int(lead_raw), rmse, mae, relpath))
    return rows


def _lead_rows_from_horizon_csv(root: Path, relpath: str, *, split_name: str = "val") -> list[LeadRow]:
    rows: list[LeadRow] = []
    for row in read_csv(root / relpath):
        if row.get("split_name") != split_name or row.get("metric_scope") != "horizon":
            continue
        if row.get("eval_protocol") not in ("rolling_origin_no_refit", "non_overlap"):
            continue
        label = LABELS.get(row.get("model_variant", ""))
        rmse = as_float(row, "rmse_pu")
        mae = as_float(row, "mae_pu")
        lead_raw = row.get("lead_step")
        if label is None or rmse is None or mae is None or not lead_raw:
            continue
        rows.append(LeadRow(label, row["eval_protocol"], int(lead_raw), rmse, mae, relpath))
    return rows


def _lead_rows_from_aggregate_json(root: Path, relpath: str, *, label: str) -> list[LeadRow]:
    path = root / relpath
    if not path.exists():
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    components = payload.get("components") or {}
    lead_valid = components.get("lead_valid") or []
    lead_abs = components.get("lead_abs") or []
    lead_sq = components.get("lead_sq") or []
    if not (len(lead_valid) == len(lead_abs) == len(lead_sq) == 36):
        return []
    rows: list[LeadRow] = []
    for index, (valid, abs_sum, sq_sum) in enumerate(zip(lead_valid, lead_abs, lead_sq, strict=True), start=1):
        valid_f = float(valid)
        if valid_f <= 0:
            continue
        rows.append(
            LeadRow(
                label=label,
                eval_protocol=str(payload["eval_protocol"]),
                lead_step=index,
                rmse_pu=(float(sq_sum) / valid_f) ** 0.5,
                mae_pu=float(abs_sum) / valid_f,
                source=relpath,
            )
        )
    return rows


def collect_official_v2_validation_lead_rows(root: Path) -> list[LeadRow]:
    relpaths = [
        OFFICIAL_V2_LEADWISE_RECOVERY / "official_v2_controls_val_leadwise.csv",
        OFFICIAL_V2_LEADWISE_RECOVERY / "official_v2_chronos2_val_nonoverlap_leadwise.csv",
        OFFICIAL_V2_LEADWISE_RECOVERY / "official_v2_itransformer_val_leadwise.csv",
        OFFICIAL_V2_LEADWISE_RECOVERY / "official_v2_timexer_val_leadwise.csv",
        OFFICIAL_V2_LEADWISE_RECOVERY / "official_v2_mtgnn_val_leadwise.csv",
        OFFICIAL_V2_LEADWISE_RECOVERY
        / "dgcrn_val_shards"
        / "dgcrn_kelmarsh_dgcrn_official_core_residual_b2_v2_val_rolling_origin_no_refit_seed3407_aggregate.csv",
        OFFICIAL_V2_LEADWISE_RECOVERY
        / "dgcrn_val_shards"
        / "dgcrn_kelmarsh_dgcrn_official_core_residual_b2_v2_val_non_overlap_seed3407_aggregate.csv",
    ]
    rows: list[LeadRow] = []
    for relpath in relpaths:
        rows.extend(_lead_rows_from_horizon_csv(root, str(relpath), split_name="val"))
    rows.extend(
        _lead_rows_from_aggregate_json(
            root,
            "experiment/artifacts/scratch/world_model_official_baselines_v2/"
            "long_run_20260425_paper_grade/phase1_controls/chronos2_rolling_shards/"
            "chronos2_kelmarsh_val_rolling_origin_no_refit_aggregate.json",
            label="Official v2: Chronos-2 B2",
        )
    )
    rows.extend(
        _lead_rows_from_aggregate_json(
            root,
            "experiment/artifacts/scratch/world_model_official_baselines_v2/"
            "tft_pf_val_rolling_shards_full_20260425/"
            "tft_pf_residual_kelmarsh_val_rolling_origin_no_refit_aggregate.json",
            label="Official v2: TFT-PF B2",
        )
    )
    dedup: dict[tuple[str, str, int], LeadRow] = {}
    for row in rows:
        dedup[(row.label, row.eval_protocol, row.lead_step)] = row
    return list(dedup.values())


def bucket_from_row(row: dict[str, str], label: str, source: str, status: str) -> BucketRow | None:
    values = {
        "lead1_rmse": as_float(row, "lead1_rmse_pu_mean", "lead1_rmse_pu"),
        "short_rmse": as_float(row, "short_rmse_pu_mean", "short_rmse_pu"),
        "mid_rmse": as_float(row, "mid_rmse_pu_mean", "mid_rmse_pu"),
        "long_rmse": as_float(row, "long_rmse_pu_mean", "long_rmse_pu"),
    }
    if any(value is None for value in values.values()):
        return None
    return BucketRow(
        label=label,
        eval_protocol=row["eval_protocol"],
        lead1_rmse=float(values["lead1_rmse"]),
        short_rmse=float(values["short_rmse"]),
        mid_rmse=float(values["mid_rmse"]),
        long_rmse=float(values["long_rmse"]),
        status=row.get("main_table_status") or status,
        source=source,
    )


def collect_bucket_rows(root: Path) -> list[BucketRow]:
    rows: list[BucketRow] = []
    relpaths = [
        str(PUBLISHED_V2 / "20260425-paper-grade-controls-and-chronos-test-rows.csv"),
        str(PUBLISHED_V2 / "20260425-paper-grade-trainable-statistics-summary.csv"),
        str(PUBLISHED_V2 / "20260425-paper-grade-dgcrn-rescue-statistics-summary.csv"),
    ]
    for relpath in relpaths:
        for row in read_csv(root / relpath):
            if row.get("split_name") != "test":
                continue
            if row.get("metric_scope") != "overall":
                continue
            if row.get("eval_protocol") not in ("rolling_origin_no_refit", "non_overlap"):
                continue
            label = LABELS.get(row.get("model_variant", ""))
            if label is None:
                continue
            bucket = bucket_from_row(row, label, relpath, row.get("main_table_status") or "main")
            if bucket is not None:
                rows.append(bucket)

    tft_aggregate = root / (
        "experiment/artifacts/scratch/world_model_official_baselines_v2/"
        "tft_pf_test_rolling_shards_full_20260425/"
        "tft_pf_residual_kelmarsh_test_rolling_origin_no_refit_aggregate.json"
    )
    if tft_aggregate.exists():
        metrics = json.loads(tft_aggregate.read_text(encoding="utf-8")).get("metrics", {})
        row = {
            "eval_protocol": "rolling_origin_no_refit",
            "lead1_rmse_pu": str(metrics["lead1_rmse_pu"]),
            "short_rmse_pu": str(metrics["short_rmse_pu"]),
            "mid_rmse_pu": str(metrics["mid_rmse_pu"]),
            "long_rmse_pu": str(metrics["long_rmse_pu"]),
            "main_table_status": "appendix_diagnostic",
        }
        bucket = bucket_from_row(row, "Official v2: TFT-PF B2", str(tft_aggregate.relative_to(root)), "appendix_diagnostic")
        if bucket is not None:
            rows.append(bucket)

    dedup: dict[tuple[str, str], BucketRow] = {}
    for row in rows:
        dedup[(row.label, row.eval_protocol)] = row
    return list(dedup.values())


def bucket_rows_from_lead_rows(lead_rows: list[LeadRow]) -> list[BucketRow]:
    grouped: dict[tuple[str, str], list[LeadRow]] = {}
    for row in lead_rows:
        if not row.label.startswith("State-space:"):
            continue
        grouped.setdefault((row.label, row.eval_protocol), []).append(row)

    rows: list[BucketRow] = []
    for (label, protocol), seq in grouped.items():
        by_lead = {row.lead_step: row.rmse_pu for row in seq}
        if not all(lead in by_lead for lead in range(1, 37)):
            continue
        rows.append(
            BucketRow(
                label=label,
                eval_protocol=protocol,
                lead1_rmse=by_lead[1],
                short_rmse=fmean(by_lead[lead] for lead in range(1, 7)),
                mid_rmse=fmean(by_lead[lead] for lead in range(7, 19)),
                long_rmse=fmean(by_lead[lead] for lead in range(19, 37)),
                status="diagnostic" if label != "State-space: Official canonical" else "main",
                source="derived_from_leadwise_rows",
            )
        )
    return rows


def ordered_labels(labels: Iterable[str]) -> list[str]:
    present = set(labels)
    ordered = [label for label in SERIES_ORDER if label in present]
    ordered.extend(sorted(present.difference(SERIES_ORDER)))
    return ordered


def best_label_by_mean_rmse(rows: list[LeadRow], *, prefix: str, exclude: set[str] | None = None) -> str | None:
    exclude = exclude or set()
    grouped: dict[str, list[float]] = {}
    for row in rows:
        if row.label.startswith(prefix) and row.label not in exclude:
            grouped.setdefault(row.label, []).append(row.rmse_pu)
    if not grouped:
        return None
    return min(grouped, key=lambda label: fmean(grouped[label]))


def line_style(label: str) -> tuple[float, str | tuple[int, tuple[int, ...]]]:
    if label.startswith("Official v2"):
        return 2.6, "-"
    if label.startswith("State-space"):
        if "Global-local" in label:
            return 2.2, "--"
        if "Ramp v2" in label:
            return 2.2, "-."
        if "Edge" in label:
            return 2.2, (0, (3, 1, 1, 1))
        return 2.8, ":"
    return 1.6, "-"


def write_lead_csv(rows: list[LeadRow], output_path: Path) -> None:
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["label", "eval_protocol", "lead_step", "lead_minutes", "rmse_pu", "mae_pu", "source"],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "label": row.label,
                    "eval_protocol": row.eval_protocol,
                    "lead_step": row.lead_step,
                    "lead_minutes": row.lead_step * 10,
                    "rmse_pu": row.rmse_pu,
                    "mae_pu": row.mae_pu,
                    "source": row.source,
                }
            )


def write_bucket_csv(rows: list[BucketRow], output_path: Path) -> None:
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "label",
                "eval_protocol",
                "status",
                "lead1_rmse",
                "short_rmse",
                "mid_rmse",
                "long_rmse",
                "source",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row.__dict__)


def plot_leadwise(
    rows: list[LeadRow],
    output_png: Path,
    output_svg: Path,
    *,
    title: str = "Kelmarsh Mainline Variants: 36-Step Leadwise RMSE",
    note: str = (
        "True per-lead rows are available for the mainline state-space variants. "
        "Official v2 paper-grade rows publish horizon buckets, not 36 separate lead rows."
    ),
    log_y: bool = False,
) -> None:
    grouped: dict[tuple[str, str], list[LeadRow]] = {}
    for row in rows:
        grouped.setdefault((row.label, row.eval_protocol), []).append(row)

    fig, axes = plt.subplots(1, 2, figsize=(21.5, 10.2), sharey=True)
    fig.patch.set_facecolor("#F7F3EA")
    panels = [("rolling_origin_no_refit", "Rolling Origin No Refit"), ("non_overlap", "Non-overlap")]
    figure_title = title
    for ax, (protocol, panel_title) in zip(axes, panels):
        ax.set_facecolor("#FFFDFC")
        for label in ordered_labels(label for label, proto in grouped if proto == protocol):
            seq = sorted(grouped[(label, protocol)], key=lambda row: row.lead_step)
            width, linestyle = line_style(label)
            ax.plot(
                [row.lead_step for row in seq],
                [row.rmse_pu for row in seq],
                label=label,
                color=COLORS.get(label, "#6B7280"),
                linewidth=width,
                alpha=0.92,
                linestyle=linestyle,
            )
        ax.set_title(panel_title, fontsize=15, weight="bold", color="#221D18", pad=10)
        ax.set_xlabel("Lead step (10 min each)", fontsize=11, color="#2A251F")
        ax.set_xlim(1, 36)
        if log_y:
            ax.set_yscale("log")
            ax.yaxis.set_major_formatter(FormatStrFormatter("%.3f"))
        ax.grid(True, linestyle="--", linewidth=0.7, alpha=0.35)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color("#6D655D")
        ax.spines["bottom"].set_color("#6D655D")
        ax.tick_params(axis="both", labelsize=10, colors="#2A251F")
    axes[0].set_ylabel("RMSE (pu)", fontsize=11, color="#2A251F")
    handle_by_label = {}
    for ax in axes:
        handles, legend_labels = ax.get_legend_handles_labels()
        for handle, label in zip(handles, legend_labels, strict=False):
            handle_by_label.setdefault(label, handle)
    legend_labels = ordered_labels(handle_by_label)
    handles = [handle_by_label[label] for label in legend_labels]
    fig.legend(
        handles,
        legend_labels,
        loc="center left",
        bbox_to_anchor=(0.735, 0.50),
        frameon=False,
        fontsize=8.6,
        labelspacing=0.55,
    )
    title_suffix = " (Log Y)" if log_y and "Log Y" not in figure_title else ""
    fig.suptitle(f"{figure_title}{title_suffix}", fontsize=16, weight="bold", color="#1D1814", y=0.965)
    fig.text(
        0.5,
        0.035,
        note + (" Y axis uses log scale." if log_y else ""),
        ha="center",
        fontsize=9.5,
        color="#4F463E",
    )
    fig.subplots_adjust(left=0.065, right=0.72, top=0.88, bottom=0.12, wspace=0.055)
    fig.savefig(output_png, dpi=220, bbox_inches="tight")
    fig.savefig(output_svg, bbox_inches="tight")


def plot_leadwise_relative(
    rows: list[LeadRow],
    output_png: Path,
    output_svg: Path,
    *,
    reference_label: str = "State-space: Official canonical",
    title: str,
    note: str,
    ylim: tuple[float, float] | None = None,
    solid_lines: bool = False,
) -> None:
    reference = {
        (row.eval_protocol, row.lead_step): row.rmse_pu
        for row in rows
        if row.label == reference_label and row.rmse_pu > 0
    }
    transformed: list[LeadRow] = []
    for row in rows:
        ref = reference.get((row.eval_protocol, row.lead_step))
        if ref is None:
            continue
        transformed.append(
            LeadRow(
                label=row.label,
                eval_protocol=row.eval_protocol,
                lead_step=row.lead_step,
                rmse_pu=100.0 * (row.rmse_pu / ref - 1.0),
                mae_pu=100.0 * (row.mae_pu / ref - 1.0),
                source=row.source,
            )
        )

    grouped: dict[tuple[str, str], list[LeadRow]] = {}
    for row in transformed:
        grouped.setdefault((row.label, row.eval_protocol), []).append(row)

    fig, axes = plt.subplots(1, 2, figsize=(21.5, 10.2), sharey=True)
    fig.patch.set_facecolor("#F7F3EA")
    panels = [("rolling_origin_no_refit", "Rolling Origin No Refit"), ("non_overlap", "Non-overlap")]
    for ax, (protocol, panel_title) in zip(axes, panels):
        ax.set_facecolor("#FFFDFC")
        ax.axhline(0.0, color="#5E564F", linewidth=1.0, alpha=0.65)
        for label in ordered_labels(label for label, proto in grouped if proto == protocol):
            seq = sorted(grouped[(label, protocol)], key=lambda row: row.lead_step)
            width, linestyle = line_style(label)
            if solid_lines:
                linestyle = "-"
            ax.plot(
                [row.lead_step for row in seq],
                [row.rmse_pu for row in seq],
                label=label,
                color=COLORS.get(label, "#6B7280"),
                linewidth=width,
                alpha=0.92,
                linestyle=linestyle,
            )
        ax.set_title(panel_title, fontsize=15, weight="bold", color="#221D18", pad=10)
        ax.set_xlabel("Lead step (10 min each)", fontsize=11, color="#2A251F")
        ax.set_xlim(1, 36)
        if ylim is not None:
            ax.set_ylim(*ylim)
        ax.grid(True, linestyle="--", linewidth=0.7, alpha=0.35)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color("#6D655D")
        ax.spines["bottom"].set_color("#6D655D")
        ax.tick_params(axis="both", labelsize=10, colors="#2A251F")
    axes[0].set_ylabel(f"RMSE delta vs {reference_label} (%)", fontsize=11, color="#2A251F")
    handle_by_label = {}
    for ax in axes:
        handles, legend_labels = ax.get_legend_handles_labels()
        for handle, label in zip(handles, legend_labels, strict=False):
            handle_by_label.setdefault(label, handle)
    legend_labels = ordered_labels(handle_by_label)
    handles = [handle_by_label[label] for label in legend_labels]
    fig.legend(
        handles,
        legend_labels,
        loc="center left",
        bbox_to_anchor=(0.735, 0.50),
        frameon=False,
        fontsize=8.6,
        labelspacing=0.55,
    )
    fig.suptitle(title, fontsize=16, weight="bold", color="#1D1814", y=0.965)
    fig.text(0.5, 0.035, note, ha="center", fontsize=9.5, color="#4F463E")
    fig.subplots_adjust(left=0.065, right=0.72, top=0.88, bottom=0.12, wspace=0.055)
    fig.savefig(output_png, dpi=220, bbox_inches="tight")
    fig.savefig(output_svg, bbox_inches="tight")


def plot_buckets(rows: list[BucketRow], output_png: Path, output_svg: Path, *, log_y: bool = False) -> None:
    grouped: dict[tuple[str, str], BucketRow] = {(row.label, row.eval_protocol): row for row in rows}
    fig, axes = plt.subplots(1, 2, figsize=(18.6, 9.0), sharey=True)
    fig.patch.set_facecolor("#F7F3EA")
    x_labels = ["Lead 1", "Short 1-6", "Mid 7-18", "Long 19-36"]
    x = [1, 2, 3, 4]
    panels = [("rolling_origin_no_refit", "Rolling Origin No Refit"), ("non_overlap", "Non-overlap")]
    for ax, (protocol, title) in zip(axes, panels):
        ax.set_facecolor("#FFFDFC")
        for label in ordered_labels(label for label, proto in grouped if proto == protocol):
            row = grouped[(label, protocol)]
            width, linestyle = line_style(label)
            ax.plot(
                x,
                [row.lead1_rmse, row.short_rmse, row.mid_rmse, row.long_rmse],
                marker="o",
                markersize=5.0,
                label=label,
                color=COLORS.get(label, "#6B7280"),
                linewidth=width,
                alpha=0.92,
                linestyle=linestyle,
            )
        ax.set_title(title, fontsize=15, weight="bold", color="#221D18")
        ax.set_xticks(x)
        ax.set_xticklabels(x_labels)
        if log_y:
            ax.set_yscale("log")
            ax.yaxis.set_major_formatter(FormatStrFormatter("%.3f"))
        ax.grid(True, linestyle="--", linewidth=0.7, alpha=0.35)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color("#6D655D")
        ax.spines["bottom"].set_color("#6D655D")
        ax.tick_params(axis="both", labelsize=10, colors="#2A251F")
    axes[0].set_ylabel("RMSE (pu)", fontsize=11, color="#2A251F")
    handles, legend_labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        legend_labels,
        loc="center left",
        bbox_to_anchor=(0.855, 0.5),
        frameon=False,
        fontsize=9.0,
        labelspacing=0.55,
    )
    title_suffix = " (Log Y)" if log_y else ""
    fig.suptitle(
        f"Kelmarsh Official Baselines v2 + Mainline Variants: Horizon-Bucket RMSE{title_suffix}",
        fontsize=18,
        weight="bold",
        color="#1D1814",
        y=0.98,
    )
    fig.text(
        0.5,
        0.04,
        "Buckets are lead1, mean short-horizon RMSE (1-6), mean mid-horizon RMSE (7-18), and mean long-horizon RMSE (19-36). DGCRN/TFT and scratch mainline variants are diagnostic."
        + (" Y axis uses log scale." if log_y else ""),
        ha="center",
        fontsize=10,
        color="#4F463E",
    )
    fig.tight_layout(rect=(0.03, 0.07, 0.84, 0.93))
    fig.savefig(output_png, dpi=220, bbox_inches="tight")
    fig.savefig(output_svg, bbox_inches="tight")


def plot_bucket_subset(
    rows: list[BucketRow],
    output_png: Path,
    output_svg: Path,
    *,
    title: str,
    note: str,
    log_y: bool = False,
) -> None:
    plot_buckets_with_title(rows, output_png, output_svg, title=title, note=note, log_y=log_y)


def plot_buckets_with_title(
    rows: list[BucketRow],
    output_png: Path,
    output_svg: Path,
    *,
    title: str,
    note: str,
    log_y: bool = False,
) -> None:
    grouped: dict[tuple[str, str], BucketRow] = {(row.label, row.eval_protocol): row for row in rows}
    fig, axes = plt.subplots(1, 2, figsize=(18.6, 9.0), sharey=True)
    fig.patch.set_facecolor("#F7F3EA")
    x_labels = ["Lead 1", "Short 1-6", "Mid 7-18", "Long 19-36"]
    x = [1, 2, 3, 4]
    panels = [("rolling_origin_no_refit", "Rolling Origin No Refit"), ("non_overlap", "Non-overlap")]
    for ax, (protocol, panel_title) in zip(axes, panels):
        ax.set_facecolor("#FFFDFC")
        for label in ordered_labels(label for label, proto in grouped if proto == protocol):
            row = grouped[(label, protocol)]
            width, linestyle = line_style(label)
            ax.plot(
                x,
                [row.lead1_rmse, row.short_rmse, row.mid_rmse, row.long_rmse],
                marker="o",
                markersize=5.0,
                label=label,
                color=COLORS.get(label, "#6B7280"),
                linewidth=width,
                alpha=0.92,
                linestyle=linestyle,
            )
        ax.set_title(panel_title, fontsize=15, weight="bold", color="#221D18")
        ax.set_xticks(x)
        ax.set_xticklabels(x_labels)
        if log_y:
            ax.set_yscale("log")
            ax.yaxis.set_major_formatter(FormatStrFormatter("%.3f"))
        ax.grid(True, linestyle="--", linewidth=0.7, alpha=0.35)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color("#6D655D")
        ax.spines["bottom"].set_color("#6D655D")
        ax.tick_params(axis="both", labelsize=10, colors="#2A251F")
    axes[0].set_ylabel("RMSE (pu)", fontsize=11, color="#2A251F")
    handles, legend_labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        legend_labels,
        loc="center left",
        bbox_to_anchor=(0.855, 0.5),
        frameon=False,
        fontsize=9.0,
        labelspacing=0.55,
    )
    fig.suptitle(title, fontsize=18, weight="bold", color="#1D1814", y=0.98)
    fig.text(0.5, 0.04, note, ha="center", fontsize=10, color="#4F463E")
    fig.tight_layout(rect=(0.03, 0.07, 0.84, 0.93))
    fig.savefig(output_png, dpi=220, bbox_inches="tight")
    fig.savefig(output_svg, bbox_inches="tight")


def write_highlights(bucket_rows: list[BucketRow], output_path: Path) -> None:
    lines = [
        "# Horizon Bucket Highlights",
        "",
        "Official baselines v2 RMSE buckets. DGCRN/TFT are diagnostic rows.",
        "",
    ]
    for protocol in ("rolling_origin_no_refit", "non_overlap"):
        title = "Rolling Origin No Refit" if protocol == "rolling_origin_no_refit" else "Non-overlap"
        rows = [row for row in bucket_rows if row.eval_protocol == protocol]
        rows.sort(key=lambda row: row.long_rmse)
        lines.extend([f"## {title}", "", "| Series | Lead 1 | Short | Mid | Long | Status |", "| --- | ---: | ---: | ---: | ---: | --- |"])
        for row in rows:
            lines.append(
                f"| {row.label} | {row.lead1_rmse:.6f} | {row.short_rmse:.6f} | {row.mid_rmse:.6f} | {row.long_rmse:.6f} | {row.status} |"
            )
        lines.append("")
    output_path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Make Kelmarsh leadwise and horizon-bucket reports from local artifacts.")
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--date-stamp", default=DEFAULT_DATE_STAMP)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    output_dir = (args.output_dir or root / "figures").resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    lead_rows = collect_lead_rows(root, split_name="test")
    mainline_lead_rows = [row for row in lead_rows if row.label.startswith("State-space:")]
    validation_lead_rows = collect_lead_rows(root, split_name="val")
    validation_mainline_rows = [row for row in validation_lead_rows if row.label.startswith("State-space:")]
    official_validation_rows = collect_official_v2_validation_lead_rows(root)
    bucket_rows = collect_bucket_rows(root) + bucket_rows_from_lead_rows(lead_rows)
    if not lead_rows and not bucket_rows:
        raise SystemExit("No leadwise or bucket rows found.")

    if mainline_lead_rows:
        lead_csv = output_dir / f"{args.date_stamp}-leadwise-summary.csv"
        lead_png = output_dir / f"{args.date_stamp}-mainline-36step-leadwise-rmse.png"
        lead_svg = output_dir / f"{args.date_stamp}-mainline-36step-leadwise-rmse.svg"
        write_lead_csv(mainline_lead_rows, lead_csv)
        plot_leadwise(mainline_lead_rows, lead_png, lead_svg)
        print(lead_csv)
        print(lead_png)
        print(lead_svg)

    combined_validation_rows = official_validation_rows + validation_mainline_rows
    if combined_validation_rows:
        validation_csv = output_dir / f"{args.date_stamp}-official-v2-plus-mainline-validation-36step-leadwise-rmse.csv"
        validation_png = output_dir / f"{args.date_stamp}-official-v2-plus-mainline-validation-36step-leadwise-rmse.png"
        validation_svg = output_dir / f"{args.date_stamp}-official-v2-plus-mainline-validation-36step-leadwise-rmse.svg"
        write_lead_csv(combined_validation_rows, validation_csv)
        plot_leadwise(
            combined_validation_rows,
            validation_png,
            validation_svg,
            title="Kelmarsh Validation: Official Baselines v2 + Mainline 36-Step Leadwise RMSE",
            note=(
                "Official v2 leadwise rows are recovered from validation-only reruns or checkpoint/shard "
                "validation aggregates; no test metrics are used for config selection."
            ),
        )
        print(validation_csv)
        print(validation_png)
        print(validation_svg)
        validation_no_seasonal_rows = [row for row in combined_validation_rows if "Seasonal" not in row.label]
        validation_no_seasonal_png = (
            output_dir / f"{args.date_stamp}-official-v2-plus-mainline-validation-36step-leadwise-rmse-no-seasonal.png"
        )
        validation_no_seasonal_svg = (
            output_dir / f"{args.date_stamp}-official-v2-plus-mainline-validation-36step-leadwise-rmse-no-seasonal.svg"
        )
        validation_no_seasonal_log_png = (
            output_dir / f"{args.date_stamp}-official-v2-plus-mainline-validation-36step-leadwise-rmse-no-seasonal-logy.png"
        )
        validation_no_seasonal_log_svg = (
            output_dir / f"{args.date_stamp}-official-v2-plus-mainline-validation-36step-leadwise-rmse-no-seasonal-logy.svg"
        )
        validation_no_seasonal_relative_png = (
            output_dir
            / f"{args.date_stamp}-official-v2-plus-mainline-validation-36step-leadwise-rmse-no-seasonal-relative-canonical.png"
        )
        validation_no_seasonal_relative_svg = (
            output_dir
            / f"{args.date_stamp}-official-v2-plus-mainline-validation-36step-leadwise-rmse-no-seasonal-relative-canonical.svg"
        )
        validation_no_seasonal_relative_zoom_png = (
            output_dir
            / f"{args.date_stamp}-official-v2-plus-mainline-validation-36step-leadwise-rmse-no-seasonal-relative-canonical-zoom.png"
        )
        validation_no_seasonal_relative_zoom_svg = (
            output_dir
            / f"{args.date_stamp}-official-v2-plus-mainline-validation-36step-leadwise-rmse-no-seasonal-relative-canonical-zoom.svg"
        )
        validation_ramp_edge_focus_png = (
            output_dir
            / f"{args.date_stamp}-official-v2-plus-mainline-validation-36step-leadwise-rmse-ramp0p05-vs-edge-rotor-wake.png"
        )
        validation_ramp_edge_focus_svg = (
            output_dir
            / f"{args.date_stamp}-official-v2-plus-mainline-validation-36step-leadwise-rmse-ramp0p05-vs-edge-rotor-wake.svg"
        )
        validation_best_mainline_vs_baselines_png = (
            output_dir
            / f"{args.date_stamp}-official-v2-plus-best-mainline-validation-36step-leadwise-rmse-no-seasonal-relative-canonical.png"
        )
        validation_best_mainline_vs_baselines_svg = (
            output_dir
            / f"{args.date_stamp}-official-v2-plus-best-mainline-validation-36step-leadwise-rmse-no-seasonal-relative-canonical.svg"
        )
        validation_mainline_internal_png = (
            output_dir
            / f"{args.date_stamp}-mainline-validation-36step-leadwise-rmse-relative-canonical.png"
        )
        validation_mainline_internal_svg = (
            output_dir
            / f"{args.date_stamp}-mainline-validation-36step-leadwise-rmse-relative-canonical.svg"
        )
        plot_leadwise(
            validation_no_seasonal_rows,
            validation_no_seasonal_png,
            validation_no_seasonal_svg,
            title="Kelmarsh Validation: Official Baselines v2 + Mainline 36-Step RMSE (No Seasonal)",
            note=(
                "Seasonal persistence is omitted to keep the close-performing validation curves readable. "
                "Official v2 rows are validation-only recovery exports."
            ),
        )
        plot_leadwise(
            validation_no_seasonal_rows,
            validation_no_seasonal_log_png,
            validation_no_seasonal_log_svg,
            title="Kelmarsh Validation: Official Baselines v2 + Mainline 36-Step RMSE (No Seasonal)",
            note=(
                "Seasonal persistence is omitted to keep the close-performing validation curves readable. "
                "Official v2 rows are validation-only recovery exports."
            ),
            log_y=True,
        )
        plot_leadwise_relative(
            validation_no_seasonal_rows,
            validation_no_seasonal_relative_png,
            validation_no_seasonal_relative_svg,
            title="Kelmarsh Validation: 36-Step RMSE Delta vs State-Space Canonical",
            note=(
                "Seasonal persistence is omitted. Values are percent RMSE difference at the same lead; "
                "negative means lower RMSE than State-space: Official canonical."
            ),
        )
        plot_leadwise_relative(
            validation_no_seasonal_rows,
            validation_no_seasonal_relative_zoom_png,
            validation_no_seasonal_relative_zoom_svg,
            title="Kelmarsh Validation: Close-Cluster RMSE Delta vs State-Space Canonical",
            note=(
                "Zoomed to -8%..5% to separate close curves; higher official outliers are clipped. "
                "Negative means lower RMSE than State-space: Official canonical."
            ),
            ylim=(-8.0, 5.0),
        )
        focus_labels = {
            "State-space: Official canonical",
            "State-space: Ramp v2 residual (0.05)",
            "State-space: Edge rotor wake",
        }
        plot_leadwise_relative(
            [row for row in validation_no_seasonal_rows if row.label in focus_labels],
            validation_ramp_edge_focus_png,
            validation_ramp_edge_focus_svg,
            title="Kelmarsh Validation: Ramp v2 0.05 vs Edge Rotor Wake",
            note=(
                "Focused diagnostic: percent RMSE difference at the same lead relative to "
                "State-space: Official canonical. Negative means lower RMSE."
            ),
            ylim=(-7.5, 1.0),
        )
        best_mainline_label = best_label_by_mean_rmse(
            validation_no_seasonal_rows,
            prefix="State-space:",
            exclude={"State-space: Official canonical"},
        )
        if best_mainline_label is not None:
            baseline_focus_labels = {
                row.label
                for row in validation_no_seasonal_rows
                if row.label.startswith("Official v2:")
            }
            baseline_focus_labels.update({"State-space: Official canonical", best_mainline_label})
            plot_leadwise_relative(
                [row for row in validation_no_seasonal_rows if row.label in baseline_focus_labels],
                validation_best_mainline_vs_baselines_png,
                validation_best_mainline_vs_baselines_svg,
                title="Kelmarsh Validation: Best Mainline vs Official Baselines v2",
                note=(
                    f"Best mainline is `{best_mainline_label}` by mean validation 36-step RMSE across both protocols. "
                    "Seasonal persistence is omitted; negative means lower RMSE than State-space: Official canonical."
                ),
            )
            print(validation_best_mainline_vs_baselines_png)
            print(validation_best_mainline_vs_baselines_svg)
        mainline_internal_rows = [row for row in validation_no_seasonal_rows if row.label.startswith("State-space:")]
        plot_leadwise_relative(
            mainline_internal_rows,
            validation_mainline_internal_png,
            validation_mainline_internal_svg,
            title="Kelmarsh Validation: Mainline State-Space Variant Comparison",
            note=(
                "Only State-space mainline variants are shown. Values are percent RMSE difference at the same lead; "
                "negative means lower RMSE than State-space: Official canonical."
            ),
            solid_lines=True,
        )
        print(validation_no_seasonal_png)
        print(validation_no_seasonal_svg)
        print(validation_no_seasonal_log_png)
        print(validation_no_seasonal_log_svg)
        print(validation_no_seasonal_relative_png)
        print(validation_no_seasonal_relative_svg)
        print(validation_no_seasonal_relative_zoom_png)
        print(validation_no_seasonal_relative_zoom_svg)
        print(validation_ramp_edge_focus_png)
        print(validation_ramp_edge_focus_svg)
        print(validation_mainline_internal_png)
        print(validation_mainline_internal_svg)

    if bucket_rows:
        bucket_csv = output_dir / f"{args.date_stamp}-official-v2-plus-mainline-horizon-buckets.csv"
        bucket_png = output_dir / f"{args.date_stamp}-official-v2-plus-mainline-horizon-buckets-rmse.png"
        bucket_svg = output_dir / f"{args.date_stamp}-official-v2-plus-mainline-horizon-buckets-rmse.svg"
        bucket_log_png = output_dir / f"{args.date_stamp}-official-v2-plus-mainline-horizon-buckets-rmse-logy.png"
        bucket_log_svg = output_dir / f"{args.date_stamp}-official-v2-plus-mainline-horizon-buckets-rmse-logy.svg"
        bucket_no_seasonal_png = output_dir / f"{args.date_stamp}-official-v2-plus-mainline-horizon-buckets-rmse-no-seasonal.png"
        bucket_no_seasonal_svg = output_dir / f"{args.date_stamp}-official-v2-plus-mainline-horizon-buckets-rmse-no-seasonal.svg"
        bucket_no_seasonal_log_png = output_dir / f"{args.date_stamp}-official-v2-plus-mainline-horizon-buckets-rmse-no-seasonal-logy.png"
        bucket_no_seasonal_log_svg = output_dir / f"{args.date_stamp}-official-v2-plus-mainline-horizon-buckets-rmse-no-seasonal-logy.svg"
        bucket_seasonal_png = output_dir / f"{args.date_stamp}-official-v2-seasonal-horizon-buckets-rmse.png"
        bucket_seasonal_svg = output_dir / f"{args.date_stamp}-official-v2-seasonal-horizon-buckets-rmse.svg"
        highlights = output_dir / f"{args.date_stamp}-official-horizon-bucket-highlights.md"
        write_bucket_csv(bucket_rows, bucket_csv)
        plot_buckets(bucket_rows, bucket_png, bucket_svg)
        plot_buckets(bucket_rows, bucket_log_png, bucket_log_svg, log_y=True)
        no_seasonal = [row for row in bucket_rows if "Seasonal" not in row.label]
        seasonal = [row for row in bucket_rows if "Seasonal" in row.label]
        plot_bucket_subset(
            no_seasonal,
            bucket_no_seasonal_png,
            bucket_no_seasonal_svg,
            title="Kelmarsh Official Baselines v2 + Mainline Variants: Horizon-Bucket RMSE (No Seasonal)",
            note="Seasonal persistence is omitted here because its much larger error compresses the useful comparison range. DGCRN/TFT and scratch mainline variants are diagnostic.",
        )
        plot_bucket_subset(
            no_seasonal,
            bucket_no_seasonal_log_png,
            bucket_no_seasonal_log_svg,
            title="Kelmarsh Official Baselines v2 + Mainline Variants: Horizon-Bucket RMSE (No Seasonal, Log Y)",
            note="Seasonal persistence is omitted and the y axis uses log scale to separate close-performing baselines. DGCRN/TFT and scratch mainline variants are diagnostic.",
            log_y=True,
        )
        plot_bucket_subset(
            seasonal,
            bucket_seasonal_png,
            bucket_seasonal_svg,
            title="Kelmarsh Seasonal Persistence: Horizon-Bucket RMSE",
            note="Seasonal persistence is separated as a scale reference; it is much worse than last-value and learned/residual baselines on this task.",
        )
        write_highlights(bucket_rows, highlights)
        print(bucket_csv)
        print(bucket_png)
        print(bucket_svg)
        print(bucket_log_png)
        print(bucket_log_svg)
        print(bucket_no_seasonal_png)
        print(bucket_no_seasonal_svg)
        print(bucket_no_seasonal_log_png)
        print(bucket_no_seasonal_log_svg)
        print(bucket_seasonal_png)
        print(bucket_seasonal_svg)
        print(highlights)


if __name__ == "__main__":
    main()
