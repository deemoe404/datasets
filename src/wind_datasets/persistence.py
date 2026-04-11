from __future__ import annotations

from dataclasses import dataclass
from math import sqrt
from pathlib import Path

import numpy as np
import polars as pl

from .api import build_gold_base
from .models import DatasetSpec, ResolvedTaskSpec, TaskSpec
from .paths import dataset_cache_paths
from .registry import get_dataset_spec

__all__ = ["PersistenceExperimentResult", "run_persistence_experiment"]

_PERSISTENCE_COLUMNS = [
    "dataset",
    "turbine_id",
    "timestamp",
    "target_kw",
    "is_observed",
    "quality_flags",
]

# Dataset-level rated power is part of the workspace contract in AGENTS.md.
_DATASET_RATED_POWER_KW = {
    "kelmarsh": 2050.0,
    "penmanshiel": 2050.0,
    "hill_of_towie": 2300.0,
    "sdwpf_kddcup": 1500.0,
}


@dataclass(frozen=True)
class PersistenceExperimentResult:
    summary: pl.DataFrame
    per_horizon: pl.DataFrame
    per_turbine: pl.DataFrame


def run_persistence_experiment(
    dataset_id: str,
    *,
    cache_root: str | Path = "cache",
    task_spec: TaskSpec | None = None,
) -> PersistenceExperimentResult:
    """Evaluate a turbine-level persistence baseline from canonical gold-base rows.

    This analysis helper intentionally stays outside the active farm task-bundle
    experiment surface.
    """
    spec = get_dataset_spec(dataset_id)
    resolved_task = (
        task_spec or TaskSpec.next_6h_from_24h(granularity="turbine")
    ).resolve(spec.resolution_minutes)
    rated_power_by_turbine = _resolve_rated_power_by_turbine(spec)
    series = _load_persistence_series(
        spec=spec,
        cache_root=Path(cache_root),
    )
    return _evaluate_persistence_series(
        series,
        spec=spec,
        quality_profile=spec.default_quality_profile,
        task=resolved_task,
        rated_power_by_turbine=rated_power_by_turbine,
    )


def _load_persistence_series(
    *,
    spec: DatasetSpec,
    cache_root: Path,
) -> pl.DataFrame:
    cache_paths = dataset_cache_paths(cache_root, spec.dataset_id)
    gold_base_path = cache_paths.gold_base_series_path
    if not gold_base_path.exists():
        build_gold_base(
            spec.dataset_id,
            cache_root=cache_root,
        )
    return (
        pl.scan_parquet(gold_base_path)
        .select(_PERSISTENCE_COLUMNS)
        .sort(["turbine_id", "timestamp"])
        .collect()
    )


def _evaluate_persistence_series(
    series: pl.DataFrame,
    *,
    spec: DatasetSpec,
    quality_profile: str,
    task: ResolvedTaskSpec,
    rated_power_by_turbine: dict[str, float] | None = None,
) -> PersistenceExperimentResult:
    if task.granularity != "turbine":
        raise ValueError(f"Persistence experiments require turbine granularity, got {task.granularity!r}.")
    if task.history_steps < 1 or task.forecast_steps < 1:
        raise ValueError("Persistence experiments require positive history and forecast steps.")

    missing_columns = [column for column in _PERSISTENCE_COLUMNS if column not in series.columns]
    if missing_columns:
        missing = ", ".join(missing_columns)
        raise ValueError(f"Persistence series is missing required columns: {missing}.")

    base = series.select(_PERSISTENCE_COLUMNS).sort(["turbine_id", "timestamp"])
    resolved_rated_power_by_turbine = rated_power_by_turbine or _resolve_rated_power_by_turbine(spec)
    partitions = (
        {
            partition["turbine_id"][0]: partition
            for partition in base.partition_by("turbine_id", maintain_order=True)
        }
        if not base.is_empty()
        else {}
    )

    horizon_abs_error_sums = np.zeros(task.forecast_steps, dtype=np.float64)
    horizon_squared_error_sums = np.zeros(task.forecast_steps, dtype=np.float64)
    horizon_normalized_abs_error_sums = np.zeros(task.forecast_steps, dtype=np.float64)
    horizon_normalized_squared_error_sums = np.zeros(task.forecast_steps, dtype=np.float64)
    horizon_prediction_counts = np.zeros(task.forecast_steps, dtype=np.int64)
    per_turbine_rows: list[dict[str, object]] = []

    for turbine_id in spec.turbine_ids:
        turbine_frame = partitions.get(turbine_id)
        turbine_result = _evaluate_turbine_frame(
            turbine_id=turbine_id,
            turbine_frame=turbine_frame,
            task=task,
            rated_power_kw=resolved_rated_power_by_turbine[turbine_id],
        )
        per_turbine_rows.append(turbine_result["row"])
        horizon_abs_error_sums += turbine_result["horizon_abs_error_sums"]
        horizon_squared_error_sums += turbine_result["horizon_squared_error_sums"]
        horizon_normalized_abs_error_sums += turbine_result["horizon_normalized_abs_error_sums"]
        horizon_normalized_squared_error_sums += turbine_result["horizon_normalized_squared_error_sums"]
        horizon_prediction_counts += turbine_result["horizon_prediction_counts"]

    total_prediction_count = int(horizon_prediction_counts.sum())
    total_abs_error_sum = float(horizon_abs_error_sums.sum())
    total_squared_error_sum = float(horizon_squared_error_sums.sum())
    total_normalized_abs_error_sum = float(horizon_normalized_abs_error_sums.sum())
    total_normalized_squared_error_sum = float(horizon_normalized_squared_error_sums.sum())
    total_eligible_windows = int(sum(int(row["eligible_windows"]) for row in per_turbine_rows))
    start_timestamp = base["timestamp"].min() if not base.is_empty() else None
    end_timestamp = base["timestamp"].max() if not base.is_empty() else None
    unique_rated_powers = {float(value) for value in resolved_rated_power_by_turbine.values()}
    dataset_rated_power_kw = unique_rated_powers.pop() if len(unique_rated_powers) == 1 else None

    summary = pl.DataFrame(
        [
            {
                "dataset_id": spec.dataset_id,
                "quality_profile": quality_profile,
                "rated_power_kw": dataset_rated_power_kw,
                "turbines": len(spec.turbine_ids),
                "eligible_windows": total_eligible_windows,
                "prediction_count": total_prediction_count,
                "mae_kw": _safe_divide(total_abs_error_sum, total_prediction_count),
                "rmse_kw": _safe_rmse(total_squared_error_sum, total_prediction_count),
                "mae_pu": _safe_divide(total_normalized_abs_error_sum, total_prediction_count),
                "rmse_pu": _safe_rmse(total_normalized_squared_error_sum, total_prediction_count),
                "start_timestamp": start_timestamp,
                "end_timestamp": end_timestamp,
            }
        ]
    )

    per_horizon = pl.DataFrame(
        {
            "horizon_step": np.arange(1, task.forecast_steps + 1, dtype=np.int64),
            "horizon_minutes": np.arange(
                task.resolution_minutes,
                (task.forecast_steps + 1) * task.resolution_minutes,
                task.resolution_minutes,
                dtype=np.int64,
            ),
            "n_predictions": horizon_prediction_counts,
            "mae_kw": [
                _safe_divide(float(abs_error_sum), int(prediction_count))
                for abs_error_sum, prediction_count in zip(
                    horizon_abs_error_sums,
                    horizon_prediction_counts,
                    strict=True,
                )
            ],
            "rmse_kw": [
                _safe_rmse(float(squared_error_sum), int(prediction_count))
                for squared_error_sum, prediction_count in zip(
                    horizon_squared_error_sums,
                    horizon_prediction_counts,
                    strict=True,
                )
            ],
            "mae_pu": [
                _safe_divide(float(abs_error_sum), int(prediction_count))
                for abs_error_sum, prediction_count in zip(
                    horizon_normalized_abs_error_sums,
                    horizon_prediction_counts,
                    strict=True,
                )
            ],
            "rmse_pu": [
                _safe_rmse(float(squared_error_sum), int(prediction_count))
                for squared_error_sum, prediction_count in zip(
                    horizon_normalized_squared_error_sums,
                    horizon_prediction_counts,
                    strict=True,
                )
            ],
        }
    )

    per_turbine = pl.DataFrame(per_turbine_rows)

    return PersistenceExperimentResult(
        summary=summary,
        per_horizon=per_horizon,
        per_turbine=per_turbine,
    )


def _evaluate_turbine_frame(
    *,
    turbine_id: str,
    turbine_frame: pl.DataFrame | None,
    task: ResolvedTaskSpec,
    rated_power_kw: float,
) -> dict[str, object]:
    horizon_abs_error_sums = np.zeros(task.forecast_steps, dtype=np.float64)
    horizon_squared_error_sums = np.zeros(task.forecast_steps, dtype=np.float64)
    horizon_normalized_abs_error_sums = np.zeros(task.forecast_steps, dtype=np.float64)
    horizon_normalized_squared_error_sums = np.zeros(task.forecast_steps, dtype=np.float64)
    horizon_prediction_counts = np.zeros(task.forecast_steps, dtype=np.int64)

    if turbine_frame is None or turbine_frame.is_empty():
        return {
            "row": _build_turbine_row(
                turbine_id=turbine_id,
                rated_power_kw=rated_power_kw,
                eligible_windows=0,
                prediction_count=0,
                abs_error_sum=0.0,
                squared_error_sum=0.0,
                normalized_abs_error_sum=0.0,
                normalized_squared_error_sum=0.0,
            ),
            "horizon_abs_error_sums": horizon_abs_error_sums,
            "horizon_squared_error_sums": horizon_squared_error_sums,
            "horizon_normalized_abs_error_sums": horizon_normalized_abs_error_sums,
            "horizon_normalized_squared_error_sums": horizon_normalized_squared_error_sums,
            "horizon_prediction_counts": horizon_prediction_counts,
        }

    clean_mask = (
        turbine_frame.select(
            (
                pl.col("is_observed").fill_null(False)
                & pl.col("target_kw").is_not_null()
                & (pl.col("quality_flags").fill_null("") == "")
            ).alias("is_clean")
        )["is_clean"]
        .to_numpy()
        .astype(bool, copy=False)
    )
    target_values = turbine_frame["target_kw"].to_numpy()
    total_points = target_values.shape[0]

    max_anchor = total_points - task.forecast_steps
    if max_anchor <= task.history_steps - 1:
        return {
            "row": _build_turbine_row(
                turbine_id=turbine_id,
                rated_power_kw=rated_power_kw,
                eligible_windows=0,
                prediction_count=0,
                abs_error_sum=0.0,
                squared_error_sum=0.0,
                normalized_abs_error_sum=0.0,
                normalized_squared_error_sum=0.0,
            ),
            "horizon_abs_error_sums": horizon_abs_error_sums,
            "horizon_squared_error_sums": horizon_squared_error_sums,
            "horizon_normalized_abs_error_sums": horizon_normalized_abs_error_sums,
            "horizon_normalized_squared_error_sums": horizon_normalized_squared_error_sums,
            "horizon_prediction_counts": horizon_prediction_counts,
        }

    prefix_sums = np.concatenate(([0], np.cumsum(clean_mask.astype(np.int64, copy=False))))
    anchor_indices = np.arange(
        task.history_steps - 1,
        max_anchor,
        task.stride_steps,
        dtype=np.int64,
    )
    history_counts = prefix_sums[anchor_indices + 1] - prefix_sums[anchor_indices + 1 - task.history_steps]
    future_counts = prefix_sums[anchor_indices + 1 + task.forecast_steps] - prefix_sums[anchor_indices + 1]
    eligible_anchor_indices = anchor_indices[
        (history_counts == task.history_steps) & (future_counts == task.forecast_steps)
    ]
    eligible_windows = int(eligible_anchor_indices.size)

    if eligible_windows == 0:
        return {
            "row": _build_turbine_row(
                turbine_id=turbine_id,
                rated_power_kw=rated_power_kw,
                eligible_windows=0,
                prediction_count=0,
                abs_error_sum=0.0,
                squared_error_sum=0.0,
                normalized_abs_error_sum=0.0,
                normalized_squared_error_sum=0.0,
            ),
            "horizon_abs_error_sums": horizon_abs_error_sums,
            "horizon_squared_error_sums": horizon_squared_error_sums,
            "horizon_normalized_abs_error_sums": horizon_normalized_abs_error_sums,
            "horizon_normalized_squared_error_sums": horizon_normalized_squared_error_sums,
            "horizon_prediction_counts": horizon_prediction_counts,
        }

    anchor_values = target_values[eligible_anchor_indices].astype(np.float64, copy=False)

    for horizon_offset in range(1, task.forecast_steps + 1):
        actual_values = target_values[eligible_anchor_indices + horizon_offset].astype(np.float64, copy=False)
        errors = actual_values - anchor_values
        normalized_errors = errors / rated_power_kw
        horizon_index = horizon_offset - 1
        horizon_abs_error_sums[horizon_index] = float(np.abs(errors).sum())
        horizon_squared_error_sums[horizon_index] = float(np.square(errors).sum())
        horizon_normalized_abs_error_sums[horizon_index] = float(np.abs(normalized_errors).sum())
        horizon_normalized_squared_error_sums[horizon_index] = float(np.square(normalized_errors).sum())
        horizon_prediction_counts[horizon_index] = int(errors.size)

    prediction_count = int(horizon_prediction_counts.sum())
    abs_error_sum = float(horizon_abs_error_sums.sum())
    squared_error_sum = float(horizon_squared_error_sums.sum())
    normalized_abs_error_sum = float(horizon_normalized_abs_error_sums.sum())
    normalized_squared_error_sum = float(horizon_normalized_squared_error_sums.sum())

    return {
        "row": _build_turbine_row(
            turbine_id=turbine_id,
            rated_power_kw=rated_power_kw,
            eligible_windows=eligible_windows,
            prediction_count=prediction_count,
            abs_error_sum=abs_error_sum,
            squared_error_sum=squared_error_sum,
            normalized_abs_error_sum=normalized_abs_error_sum,
            normalized_squared_error_sum=normalized_squared_error_sum,
        ),
        "horizon_abs_error_sums": horizon_abs_error_sums,
        "horizon_squared_error_sums": horizon_squared_error_sums,
        "horizon_normalized_abs_error_sums": horizon_normalized_abs_error_sums,
        "horizon_normalized_squared_error_sums": horizon_normalized_squared_error_sums,
        "horizon_prediction_counts": horizon_prediction_counts,
    }


def _build_turbine_row(
    *,
    turbine_id: str,
    rated_power_kw: float,
    eligible_windows: int,
    prediction_count: int,
    abs_error_sum: float,
    squared_error_sum: float,
    normalized_abs_error_sum: float,
    normalized_squared_error_sum: float,
) -> dict[str, object]:
    return {
        "turbine_id": turbine_id,
        "rated_power_kw": rated_power_kw,
        "eligible_windows": eligible_windows,
        "prediction_count": prediction_count,
        "mae_kw": _safe_divide(abs_error_sum, prediction_count),
        "rmse_kw": _safe_rmse(squared_error_sum, prediction_count),
        "mae_pu": _safe_divide(normalized_abs_error_sum, prediction_count),
        "rmse_pu": _safe_rmse(normalized_squared_error_sum, prediction_count),
    }


def _resolve_dataset_rated_power_kw(spec: DatasetSpec) -> float:
    try:
        return _DATASET_RATED_POWER_KW[spec.dataset_id]
    except KeyError as exc:
        raise KeyError(f"No rated power is configured for dataset_id {spec.dataset_id!r}.") from exc


def _resolve_rated_power_by_turbine(spec: DatasetSpec) -> dict[str, float]:
    rated_power_kw = _resolve_dataset_rated_power_kw(spec)
    return {turbine_id: rated_power_kw for turbine_id in spec.turbine_ids}


def _safe_divide(numerator: float, denominator: int) -> float | None:
    if denominator == 0:
        return None
    return float(numerator / denominator)


def _safe_rmse(squared_error_sum: float, denominator: int) -> float | None:
    if denominator == 0:
        return None
    return float(sqrt(squared_error_sum / denominator))
