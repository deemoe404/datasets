from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import UTC, datetime
import json
from math import sqrt
from pathlib import Path
import sys
import time
from typing import Any, Iterable, Sequence

import numpy as np
import polars as pl

MODEL_ID = "amazon/chronos-2"
TASK_ID = "next_6h_from_24h_stride_6h"
TARGET_POLICY = "invalid_to_nan_clip_0_rated"
DEFAULT_DATASETS = ("kelmarsh", "penmanshiel", "hill_of_towie", "sdwpf_kddcup")
DEFAULT_BATCH_SIZE = 256
DEFAULT_NEIGHBOR_COUNT = 6
UNIVARIATE_SUFFIX = "_univariate"
MULTIVARIATE_KNN6_SUFFIX = "_multivariate_knn6"
MODE_CHOICES = ("all", "univariate", "multivariate_knn6")
_REPO_ROOT = Path(__file__).resolve().parents[2]
_CACHE_ROOT = _REPO_ROOT / "cache"
_OUTPUT_PATH = _REPO_ROOT / "experiment" / "chronos-2.csv"
_POWER_ONLY_COLUMNS = ("dataset", "turbine_id", "timestamp", "target_kw", "quality_flags")
_DATASET_RATED_POWER_KW = {
    "kelmarsh": 2050.0,
    "penmanshiel": 2050.0,
    "hill_of_towie": 2300.0,
    "sdwpf_kddcup": 1500.0,
}
_DATASET_ID_ALIASES = {
    "sdwpf_full": "sdwpf_kddcup",
}
_RESULT_COLUMNS = [
    "dataset_id",
    "model_id",
    "task_id",
    "history_steps",
    "forecast_steps",
    "stride_steps",
    "target_policy",
    "window_count",
    "prediction_count",
    "start_timestamp",
    "end_timestamp",
    "mae_kw",
    "rmse_kw",
    "mae_pu",
    "rmse_pu",
    "device",
    "runtime_seconds",
]


@dataclass(frozen=True)
class TurbineSeries:
    timestamps_us: np.ndarray
    target_kw_masked: np.ndarray


@dataclass(frozen=True)
class FarmPanel:
    turbine_ids: tuple[str, ...]
    timestamps_us: np.ndarray
    target_kw_masked: np.ndarray


def _ensure_repo_src_on_path() -> None:
    src_path = _REPO_ROOT / "src"
    if str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))


def build_task_spec():
    _ensure_repo_src_on_path()
    from wind_datasets import TaskSpec

    return TaskSpec(
        history_duration="24h",
        forecast_duration="6h",
        stride_duration="6h",
        task_id=TASK_ID,
        granularity="turbine",
    )


def _ordered_unique(values: Sequence[str]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(values))


def _profile_log(dataset_id: str, phase: str, **fields: object) -> None:
    payload = {
        "dataset_id": resolve_dataset_id(dataset_id),
        "phase": phase,
        **fields,
    }
    print(
        f"[chronos2_power_only] {json.dumps(payload, ensure_ascii=False, default=str, sort_keys=True)}",
        file=sys.stderr,
        flush=True,
    )


def resolve_dataset_id(dataset_id: str) -> str:
    return _DATASET_ID_ALIASES.get(dataset_id, dataset_id)


def resolve_dataset_task(
    dataset_id: str,
    *,
    task_spec=None,
):
    _ensure_repo_src_on_path()
    from wind_datasets import get_dataset_spec

    resolved_dataset_id = resolve_dataset_id(dataset_id)
    spec = get_dataset_spec(resolved_dataset_id)
    resolved_task_spec = task_spec or build_task_spec()
    return spec, resolved_task_spec.resolve(spec.resolution_minutes)


def resolve_power_only_series_path(
    dataset_id: str,
    *,
    cache_root: str | Path = _CACHE_ROOT,
) -> tuple[Any, Path]:
    _ensure_repo_src_on_path()
    from wind_datasets.datasets import get_builder

    spec, _ = resolve_dataset_task(dataset_id)
    cache_root_path = Path(cache_root)
    builder = get_builder(spec, cache_root_path)
    resolved_feature_set = builder.resolve_feature_set(None)
    series_path = builder.cache_paths.gold_base_series_path_for(
        spec.default_quality_profile,
        layout="turbine",
        feature_set=resolved_feature_set,
    )
    if not series_path.exists():
        builder.build_gold_base(
            quality_profile=spec.default_quality_profile,
            layout="turbine",
            feature_set=resolved_feature_set,
        )
    return spec, series_path


def load_power_only_series_frame(
    dataset_id: str,
    *,
    cache_root: str | Path = _CACHE_ROOT,
    turbine_ids: Sequence[str] | None = None,
) -> tuple[Any, Path, pl.DataFrame]:
    spec, series_path = resolve_power_only_series_path(dataset_id, cache_root=cache_root)
    requested_turbine_ids = _ordered_unique(turbine_ids) if turbine_ids is not None else tuple(spec.turbine_ids)
    load_started = time.monotonic()
    series_scan = pl.scan_parquet(series_path).select(list(_POWER_ONLY_COLUMNS))
    if requested_turbine_ids != tuple(spec.turbine_ids):
        series_scan = series_scan.filter(pl.col("turbine_id").is_in(list(requested_turbine_ids)))
    series = series_scan.collect()
    load_duration = time.monotonic() - load_started
    _profile_log(
        spec.dataset_id,
        "load_power_only_series",
        path=str(series_path),
        columns=len(_POWER_ONLY_COLUMNS),
        rows=series.height,
        target_turbines=None if turbine_ids is None else len(_ordered_unique(turbine_ids)),
        input_turbines=len(requested_turbine_ids),
        duration_seconds=round(load_duration, 6),
    )
    return spec, series_path, series


def resolve_rated_power_kw(dataset_id: str) -> float:
    resolved_dataset_id = resolve_dataset_id(dataset_id)
    try:
        return _DATASET_RATED_POWER_KW[resolved_dataset_id]
    except KeyError as exc:
        raise KeyError(f"No rated power is configured for dataset_id {dataset_id!r}.") from exc


def clip_target_values(values: Sequence[float | None], rated_power_kw: float) -> np.ndarray:
    array = np.asarray(values, dtype=np.float32)
    clipped = array.copy()
    valid = ~np.isnan(clipped)
    clipped[valid] = np.clip(clipped[valid], 0.0, rated_power_kw)
    return clipped


def apply_target_policy(
    values: Sequence[float | None],
    invalid_mask: Sequence[bool],
    rated_power_kw: float,
) -> np.ndarray:
    clipped = clip_target_values(values, rated_power_kw)
    invalid = np.asarray(invalid_mask, dtype=bool)
    clipped[invalid] = np.nan
    return clipped


def prepare_power_only_series(series: pl.DataFrame, rated_power_kw: float) -> pl.DataFrame:
    invalid_expr = pl.col("target_kw").is_null() | (pl.col("quality_flags").fill_null("") != "")
    return (
        series.sort(["turbine_id", "timestamp"])
        .with_columns(
            invalid_expr.alias("invalid_target"),
            pl.when(invalid_expr)
            .then(pl.lit(None, dtype=pl.Float64))
            .otherwise(pl.col("target_kw").clip(0.0, rated_power_kw))
            .alias("target_kw_masked"),
        )
        .select(
            [
                "dataset",
                "turbine_id",
                "timestamp",
                "target_kw",
                "quality_flags",
                "invalid_target",
                "target_kw_masked",
            ]
        )
    )


def select_device(torch_module: Any | None = None) -> str:
    if torch_module is None:
        import torch as torch_module

    if bool(torch_module.cuda.is_available()):
        return "cuda"
    mps_backend = getattr(getattr(torch_module, "backends", None), "mps", None)
    if mps_backend is not None and bool(mps_backend.is_available()):
        return "mps"
    return "cpu"


def load_pipeline(*, model_id: str = MODEL_ID, device: str | None = None):
    try:
        from chronos import Chronos2Pipeline
    except ImportError as exc:
        raise ImportError(
            "Chronos2Pipeline is unavailable in the current environment. "
            "Rebuild experiment/chronos-2/.conda with ./create_env.sh."
        ) from exc

    resolved_device = device or select_device()
    return Chronos2Pipeline.from_pretrained(model_id, device_map=resolved_device)


def load_dataset_inputs(
    dataset_id: str,
    *,
    cache_root: str | Path = _CACHE_ROOT,
    task_spec=None,
    turbine_ids: Sequence[str] | None = None,
) -> tuple[Any, Any, pl.DataFrame]:
    spec, resolved_task = resolve_dataset_task(dataset_id, task_spec=task_spec)
    _, _, series = load_power_only_series_frame(
        dataset_id,
        cache_root=cache_root,
        turbine_ids=turbine_ids,
    )
    return spec, resolved_task, series


def load_dataset_turbine_static(
    dataset_id: str,
    *,
    cache_root: str | Path = _CACHE_ROOT,
) -> pl.DataFrame:
    _ensure_repo_src_on_path()
    from wind_datasets import load_turbine_static

    resolved_dataset_id = resolve_dataset_id(dataset_id)
    return load_turbine_static(resolved_dataset_id, cache_root=cache_root)


def build_turbine_series_map(series: pl.DataFrame) -> dict[str, TurbineSeries]:
    turbines: dict[str, TurbineSeries] = {}
    for turbine_frame in series.partition_by("turbine_id", maintain_order=True):
        turbine_id = turbine_frame["turbine_id"][0]
        turbines[turbine_id] = TurbineSeries(
            timestamps_us=turbine_frame["timestamp"].cast(pl.Int64).to_numpy(),
            target_kw_masked=turbine_frame["target_kw_masked"].cast(pl.Float32).to_numpy(),
        )
    return turbines


def build_farm_panel(
    series: pl.DataFrame,
    *,
    turbine_ids: Sequence[str],
    resolution_minutes: int,
) -> FarmPanel:
    if series.is_empty():
        raise ValueError("Cannot build a multivariate farm panel from an empty series.")

    ordered_turbine_ids = tuple(turbine_ids)
    full_index = pl.DataFrame(
        {
            "timestamp": pl.datetime_range(
                start=series["timestamp"].min(),
                end=series["timestamp"].max(),
                interval=f"{resolution_minutes}m",
                eager=True,
            )
        }
    )
    wide = (
        series.select(["timestamp", "turbine_id", "target_kw_masked"])
        .pivot(on="turbine_id", index="timestamp", values="target_kw_masked", aggregate_function="first")
        .sort("timestamp")
    )
    missing_columns = [
        pl.lit(None, dtype=pl.Float32).alias(turbine_id)
        for turbine_id in ordered_turbine_ids
        if turbine_id not in wide.columns
    ]
    if missing_columns:
        wide = wide.with_columns(missing_columns)

    aligned = (
        full_index.join(wide, on="timestamp", how="left")
        .sort("timestamp")
        .with_columns(
            [
                pl.col(turbine_id).cast(pl.Float32).fill_null(float("nan")).alias(turbine_id)
                for turbine_id in ordered_turbine_ids
            ]
        )
        .select(["timestamp", *ordered_turbine_ids])
    )
    return FarmPanel(
        turbine_ids=ordered_turbine_ids,
        timestamps_us=aligned["timestamp"].cast(pl.Int64).to_numpy(),
        target_kw_masked=np.asarray(
            aligned.select(list(ordered_turbine_ids)).to_numpy(),
            dtype=np.float32,
        ),
    )


def _euclidean_distance(point_a: tuple[float, float], point_b: tuple[float, float]) -> float:
    return float(sqrt((point_a[0] - point_b[0]) ** 2 + (point_a[1] - point_b[1]) ** 2))


def _haversine_distance_km(point_a: tuple[float, float], point_b: tuple[float, float]) -> float:
    lat1, lon1 = np.radians(point_a)
    lat2, lon2 = np.radians(point_b)
    delta_lat = lat2 - lat1
    delta_lon = lon2 - lon1
    hav = (
        np.sin(delta_lat / 2.0) ** 2
        + np.cos(lat1) * np.cos(lat2) * (np.sin(delta_lon / 2.0) ** 2)
    )
    return float(2.0 * 6371.0088 * np.arcsin(np.sqrt(hav)))


def build_knn_neighbor_map(
    turbine_static: pl.DataFrame,
    *,
    turbine_ids: Sequence[str],
    max_neighbors: int = DEFAULT_NEIGHBOR_COUNT,
) -> dict[str, tuple[str, ...]]:
    if max_neighbors <= 0:
        raise ValueError("max_neighbors must be positive.")

    ordered_turbine_ids = tuple(turbine_ids)
    rows_by_turbine = {
        row["turbine_id"]: row
        for row in (
            turbine_static
            .filter(pl.col("turbine_id").is_in(ordered_turbine_ids))
            .unique(subset=["turbine_id"], keep="first")
            .iter_rows(named=True)
        )
    }
    missing_turbines = [turbine_id for turbine_id in ordered_turbine_ids if turbine_id not in rows_by_turbine]
    if missing_turbines:
        raise ValueError(f"Missing turbine static rows for: {missing_turbines!r}")

    rows = [rows_by_turbine[turbine_id] for turbine_id in ordered_turbine_ids]
    has_xy = all(row["coord_x"] is not None and row["coord_y"] is not None for row in rows)
    has_latlon = all(row["latitude"] is not None and row["longitude"] is not None for row in rows)
    if has_xy:
        coordinates = {
            turbine_id: (float(row["coord_x"]), float(row["coord_y"]))
            for turbine_id, row in zip(ordered_turbine_ids, rows, strict=True)
        }
        distance_fn = _euclidean_distance
    elif has_latlon:
        coordinates = {
            turbine_id: (float(row["latitude"]), float(row["longitude"]))
            for turbine_id, row in zip(ordered_turbine_ids, rows, strict=True)
        }
        distance_fn = _haversine_distance_km
    else:
        raise ValueError("Turbine static coordinates are incomplete: expected either full coord_x/coord_y or full latitude/longitude.")

    neighborhoods: dict[str, tuple[str, ...]] = {}
    for target_id in ordered_turbine_ids:
        other_turbines = [candidate_id for candidate_id in ordered_turbine_ids if candidate_id != target_id]
        nearest_other_ids = sorted(
            other_turbines,
            key=lambda candidate_id: (
                distance_fn(coordinates[target_id], coordinates[candidate_id]),
                candidate_id,
            ),
        )[: max_neighbors - 1]
        neighborhoods[target_id] = (target_id, *nearest_other_ids)
    return neighborhoods


def build_local_panel(
    turbine_series_map: dict[str, TurbineSeries],
    *,
    turbine_ids: Sequence[str],
    resolution_minutes: int,
) -> FarmPanel:
    ordered_turbine_ids = tuple(turbine_ids)
    if not ordered_turbine_ids:
        raise ValueError("Cannot build a local panel with no turbine ids.")
    missing_turbines = [turbine_id for turbine_id in ordered_turbine_ids if turbine_id not in turbine_series_map]
    if missing_turbines:
        raise ValueError(f"Missing turbine series for: {missing_turbines!r}")

    step_us = resolution_minutes * 60 * 1_000_000
    if step_us <= 0:
        raise ValueError("resolution_minutes must be positive.")

    min_timestamp_us = min(int(turbine_series_map[turbine_id].timestamps_us.min()) for turbine_id in ordered_turbine_ids)
    max_timestamp_us = max(int(turbine_series_map[turbine_id].timestamps_us.max()) for turbine_id in ordered_turbine_ids)
    full_timestamps_us = np.arange(min_timestamp_us, max_timestamp_us + step_us, step_us, dtype=np.int64)

    panel_columns: list[np.ndarray] = []
    for turbine_id in ordered_turbine_ids:
        turbine_series = turbine_series_map[turbine_id]
        offsets = turbine_series.timestamps_us - min_timestamp_us
        if np.any(offsets % step_us != 0):
            raise ValueError(f"Turbine {turbine_id!r} has timestamps that do not align to a {resolution_minutes}-minute grid.")

        positions = (offsets // step_us).astype(np.int64, copy=False)
        aligned_values = np.full(full_timestamps_us.shape, np.nan, dtype=np.float32)
        aligned_values[positions] = turbine_series.target_kw_masked
        panel_columns.append(aligned_values)

    return FarmPanel(
        turbine_ids=ordered_turbine_ids,
        timestamps_us=full_timestamps_us,
        target_kw_masked=np.column_stack(panel_columns).astype(np.float32, copy=False),
    )


def _timestamp_us_to_string(timestamp_us: int | None) -> str | None:
    if timestamp_us is None:
        return None
    return datetime.fromtimestamp(timestamp_us / 1_000_000, tz=UTC).strftime("%Y-%m-%d %H:%M:%S")


def _safe_divide(numerator: float, denominator: int) -> float | None:
    if denominator == 0:
        return None
    return float(numerator / denominator)


def _safe_rmse(squared_error_sum: float, denominator: int) -> float | None:
    if denominator == 0:
        return None
    return float(sqrt(squared_error_sum / denominator))


def _coerce_array(value: Any) -> np.ndarray:
    if isinstance(value, np.ndarray):
        return value
    if hasattr(value, "numpy"):
        return value.numpy()
    return np.asarray(value)


def _extract_median_forecast(prediction: Any) -> np.ndarray:
    array = _coerce_array(prediction)
    if array.ndim != 3 or array.shape[2] != 1:
        raise ValueError(f"Expected quantile forecast with shape (variates, horizon, 1), got {array.shape}.")
    return array[:, :, 0].astype(np.float64, copy=False)


def _iter_univariate_batches(
    *,
    turbine_series: TurbineSeries,
    history_steps: int,
    forecast_steps: int,
    stride_steps: int,
    batch_size: int,
    window_offset: int = 0,
    max_windows_per_dataset: int | None = None,
) -> Iterable[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    context_batch: list[np.ndarray] = []
    future_rows: list[np.ndarray] = []
    future_timestamps: list[np.ndarray] = []
    emitted_windows = 0
    skipped_windows = 0
    max_anchor_index = turbine_series.timestamps_us.shape[0] - forecast_steps

    for anchor_index in range(history_steps - 1, max_anchor_index, stride_steps):
        context = turbine_series.target_kw_masked[anchor_index - history_steps + 1 : anchor_index + 1]
        future = turbine_series.target_kw_masked[anchor_index + 1 : anchor_index + 1 + forecast_steps]
        if context.shape[0] != history_steps or future.shape[0] != forecast_steps:
            continue
        if np.isnan(context).all() or np.isnan(future).all():
            continue
        if skipped_windows < window_offset:
            skipped_windows += 1
            continue

        context_batch.append(context.astype(np.float32, copy=True))
        future_rows.append(future.astype(np.float64, copy=True))
        future_timestamps.append(
            turbine_series.timestamps_us[anchor_index + 1 : anchor_index + 1 + forecast_steps].astype(np.int64, copy=True)
        )
        emitted_windows += 1

        if len(context_batch) >= batch_size:
            yield (
                np.stack(context_batch)[:, None, :],
                np.stack(future_rows),
                np.stack(future_timestamps),
            )
            context_batch = []
            future_rows = []
            future_timestamps = []

        if max_windows_per_dataset is not None and emitted_windows >= max_windows_per_dataset:
            break

    if context_batch:
        yield (
            np.stack(context_batch)[:, None, :],
            np.stack(future_rows),
            np.stack(future_timestamps),
        )


def _iter_multivariate_batches(
    *,
    farm_panel: FarmPanel,
    history_steps: int,
    forecast_steps: int,
    stride_steps: int,
    batch_size: int,
    window_offset: int = 0,
    max_windows_per_dataset: int | None = None,
    scored_row_index: int = 0,
) -> Iterable[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    context_batch: list[np.ndarray] = []
    future_rows: list[np.ndarray] = []
    future_timestamps: list[np.ndarray] = []
    emitted_windows = 0
    skipped_windows = 0
    max_anchor_index = farm_panel.timestamps_us.shape[0] - forecast_steps

    for anchor_index in range(history_steps - 1, max_anchor_index, stride_steps):
        context = farm_panel.target_kw_masked[anchor_index - history_steps + 1 : anchor_index + 1].T
        future = farm_panel.target_kw_masked[anchor_index + 1 : anchor_index + 1 + forecast_steps].T
        if context.shape != (len(farm_panel.turbine_ids), history_steps):
            continue
        if future.shape != (len(farm_panel.turbine_ids), forecast_steps):
            continue
        if np.isnan(context).all() or np.isnan(future).all() or np.isnan(future[scored_row_index]).all():
            continue
        if skipped_windows < window_offset:
            skipped_windows += 1
            continue

        context_batch.append(context.astype(np.float32, copy=True))
        future_rows.append(future.astype(np.float64, copy=True))
        future_timestamps.append(
            farm_panel.timestamps_us[anchor_index + 1 : anchor_index + 1 + forecast_steps].astype(np.int64, copy=True)
        )
        emitted_windows += 1

        if len(context_batch) >= batch_size:
            yield (
                np.stack(context_batch),
                np.stack(future_rows),
                np.stack(future_timestamps),
            )
            context_batch = []
            future_rows = []
            future_timestamps = []

        if max_windows_per_dataset is not None and emitted_windows >= max_windows_per_dataset:
            break

    if context_batch:
        yield (
            np.stack(context_batch),
            np.stack(future_rows),
            np.stack(future_timestamps),
        )


def resolve_pipeline_batch_size(context_batch: np.ndarray) -> int:
    if context_batch.ndim != 3:
        raise ValueError(
            f"Expected context_batch with shape (batch, n_variates, history_length), got {context_batch.shape}."
        )
    batch_windows, n_variates, _ = context_batch.shape
    return max(1, int(batch_windows * n_variates))


def _valid_timestamp_span_from_univariate(
    future_timestamps_batch: np.ndarray,
    valid_mask: np.ndarray,
) -> tuple[int | None, int | None]:
    if not valid_mask.any():
        return None, None
    valid_timestamps = future_timestamps_batch[valid_mask]
    return int(valid_timestamps.min()), int(valid_timestamps.max())


def _valid_timestamp_span_from_multivariate(
    future_timestamps_batch: np.ndarray,
    valid_mask: np.ndarray,
) -> tuple[int | None, int | None]:
    valid_per_horizon = valid_mask.any(axis=1)
    if not valid_per_horizon.any():
        return None, None
    valid_timestamps = future_timestamps_batch[valid_per_horizon]
    return int(valid_timestamps.min()), int(valid_timestamps.max())


def _initialize_metrics() -> dict[str, float | int | None]:
    return {
        "abs_error_sum": 0.0,
        "squared_error_sum": 0.0,
        "normalized_abs_error_sum": 0.0,
        "normalized_squared_error_sum": 0.0,
        "window_count": 0,
        "prediction_count": 0,
        "start_timestamp_us": None,
        "end_timestamp_us": None,
    }


def _update_timestamp_bounds(metrics: dict[str, float | int | None], batch_start_us: int | None, batch_end_us: int | None) -> None:
    if batch_start_us is None or batch_end_us is None:
        return
    current_start = metrics["start_timestamp_us"]
    current_end = metrics["end_timestamp_us"]
    metrics["start_timestamp_us"] = batch_start_us if current_start is None else min(int(current_start), batch_start_us)
    metrics["end_timestamp_us"] = batch_end_us if current_end is None else max(int(current_end), batch_end_us)


def _build_result_row(
    *,
    dataset_id: str,
    suffix: str,
    resolved_task: Any,
    rated_power_kw: float,
    metrics: dict[str, float | int | None],
    runtime_seconds: float,
    device: str,
) -> dict[str, object]:
    prediction_count = int(metrics["prediction_count"])
    return {
        "dataset_id": f"{dataset_id}{suffix}",
        "model_id": MODEL_ID,
        "task_id": TASK_ID,
        "history_steps": resolved_task.history_steps,
        "forecast_steps": resolved_task.forecast_steps,
        "stride_steps": resolved_task.stride_steps,
        "target_policy": TARGET_POLICY,
        "window_count": int(metrics["window_count"]),
        "prediction_count": prediction_count,
        "start_timestamp": _timestamp_us_to_string(metrics["start_timestamp_us"]),
        "end_timestamp": _timestamp_us_to_string(metrics["end_timestamp_us"]),
        "mae_kw": _safe_divide(float(metrics["abs_error_sum"]), prediction_count),
        "rmse_kw": _safe_rmse(float(metrics["squared_error_sum"]), prediction_count),
        "mae_pu": _safe_divide(float(metrics["normalized_abs_error_sum"]), prediction_count),
        "rmse_pu": _safe_rmse(float(metrics["normalized_squared_error_sum"]), prediction_count),
        "device": device,
        "runtime_seconds": round(runtime_seconds, 6),
    }


def evaluate_univariate_dataset(
    dataset_id: str,
    *,
    pipeline: Any,
    cache_root: str | Path = _CACHE_ROOT,
    task_spec=None,
    batch_size: int = DEFAULT_BATCH_SIZE,
    window_offset: int = 0,
    max_windows_per_dataset: int | None = None,
    device: str | None = None,
    turbine_ids: Sequence[str] | None = None,
) -> dict[str, object]:
    dataset_start = time.monotonic()
    spec, resolved_task = resolve_dataset_task(dataset_id, task_spec=task_spec)
    if turbine_ids is None:
        selected_turbine_ids = tuple(spec.turbine_ids)
    else:
        selected_turbine_ids = _ordered_unique(turbine_ids)
        unknown_turbine_ids = [turbine_id for turbine_id in selected_turbine_ids if turbine_id not in spec.turbine_ids]
        if unknown_turbine_ids:
            raise ValueError(f"Unknown turbine ids for dataset {spec.dataset_id!r}: {unknown_turbine_ids!r}")

    spec, resolved_task, series = load_dataset_inputs(
        dataset_id,
        cache_root=cache_root,
        task_spec=task_spec,
        turbine_ids=selected_turbine_ids,
    )
    rated_power_kw = resolve_rated_power_kw(spec.dataset_id)
    prepare_started = time.monotonic()
    prepared_series = prepare_power_only_series(
        series,
        rated_power_kw=rated_power_kw,
    )
    _profile_log(
        spec.dataset_id,
        "prepare_power_only_series",
        rows=prepared_series.height,
        columns=prepared_series.width,
        target_turbines=len(selected_turbine_ids),
        input_turbines=len(selected_turbine_ids),
        duration_seconds=round(time.monotonic() - prepare_started, 6),
    )
    turbine_series_map = build_turbine_series_map(prepared_series)
    metrics = _initialize_metrics()

    for turbine_id in selected_turbine_ids:
        turbine_series = turbine_series_map[turbine_id]
        for context_batch, actual_batch, future_timestamps_batch in _iter_univariate_batches(
            turbine_series=turbine_series,
            history_steps=resolved_task.history_steps,
            forecast_steps=resolved_task.forecast_steps,
            stride_steps=resolved_task.stride_steps,
            batch_size=batch_size,
            window_offset=window_offset,
            max_windows_per_dataset=max_windows_per_dataset,
        ):
            quantiles, _ = pipeline.predict_quantiles(
                inputs=context_batch,
                prediction_length=resolved_task.forecast_steps,
                quantile_levels=[0.5],
                batch_size=resolve_pipeline_batch_size(context_batch),
                limit_prediction_length=False,
            )
            prediction_batch = np.stack([_extract_median_forecast(prediction)[0] for prediction in quantiles])
            valid_mask = ~np.isnan(actual_batch)
            if not valid_mask.any():
                continue

            errors = prediction_batch - actual_batch
            valid_errors = errors[valid_mask]
            normalized_valid_errors = valid_errors / rated_power_kw

            metrics["window_count"] = int(metrics["window_count"]) + prediction_batch.shape[0]
            metrics["prediction_count"] = int(metrics["prediction_count"]) + int(valid_mask.sum())
            metrics["abs_error_sum"] = float(metrics["abs_error_sum"]) + float(np.abs(valid_errors).sum())
            metrics["squared_error_sum"] = float(metrics["squared_error_sum"]) + float(np.square(valid_errors).sum())
            metrics["normalized_abs_error_sum"] = float(metrics["normalized_abs_error_sum"]) + float(np.abs(normalized_valid_errors).sum())
            metrics["normalized_squared_error_sum"] = float(metrics["normalized_squared_error_sum"]) + float(
                np.square(normalized_valid_errors).sum()
            )
            batch_start_us, batch_end_us = _valid_timestamp_span_from_univariate(future_timestamps_batch, valid_mask)
            _update_timestamp_bounds(metrics, batch_start_us, batch_end_us)

    runtime_seconds = time.monotonic() - dataset_start
    _profile_log(
        spec.dataset_id,
        "evaluate_univariate_complete",
        target_turbines=len(selected_turbine_ids),
        input_turbines=len(selected_turbine_ids),
        runtime_seconds=round(runtime_seconds, 6),
        window_count=int(metrics["window_count"]),
        prediction_count=int(metrics["prediction_count"]),
    )
    return _build_result_row(
        dataset_id=spec.dataset_id,
        suffix=UNIVARIATE_SUFFIX,
        resolved_task=resolved_task,
        rated_power_kw=rated_power_kw,
        metrics=metrics,
        runtime_seconds=runtime_seconds,
        device=device or select_device(),
    )


def evaluate_multivariate_knn6_dataset(
    dataset_id: str,
    *,
    pipeline: Any,
    cache_root: str | Path = _CACHE_ROOT,
    task_spec=None,
    batch_size: int = DEFAULT_BATCH_SIZE,
    window_offset: int = 0,
    max_windows_per_dataset: int | None = None,
    device: str | None = None,
    turbine_ids: Sequence[str] | None = None,
) -> dict[str, object]:
    dataset_start = time.monotonic()
    spec, resolved_task = resolve_dataset_task(dataset_id, task_spec=task_spec)
    if turbine_ids is None:
        target_turbine_ids = tuple(spec.turbine_ids)
    else:
        target_turbine_ids = _ordered_unique(turbine_ids)
        unknown_turbine_ids = [turbine_id for turbine_id in target_turbine_ids if turbine_id not in spec.turbine_ids]
        if unknown_turbine_ids:
            raise ValueError(f"Unknown turbine ids for dataset {spec.dataset_id!r}: {unknown_turbine_ids!r}")
    neighborhood_map = build_knn_neighbor_map(
        load_dataset_turbine_static(spec.dataset_id, cache_root=cache_root),
        turbine_ids=spec.turbine_ids,
        max_neighbors=DEFAULT_NEIGHBOR_COUNT,
    )
    required_input_turbine_ids = _ordered_unique(
        turbine_id
        for target_turbine_id in target_turbine_ids
        for turbine_id in neighborhood_map[target_turbine_id]
    )
    _profile_log(
        spec.dataset_id,
        "multivariate_scope",
        target_turbines=len(target_turbine_ids),
        input_turbines=len(required_input_turbine_ids),
    )
    spec, resolved_task, series = load_dataset_inputs(
        dataset_id,
        cache_root=cache_root,
        task_spec=task_spec,
        turbine_ids=required_input_turbine_ids,
    )
    rated_power_kw = resolve_rated_power_kw(spec.dataset_id)
    prepare_started = time.monotonic()
    prepared_series = prepare_power_only_series(series, rated_power_kw=rated_power_kw)
    _profile_log(
        spec.dataset_id,
        "prepare_power_only_series",
        rows=prepared_series.height,
        columns=prepared_series.width,
        target_turbines=len(target_turbine_ids),
        input_turbines=len(required_input_turbine_ids),
        duration_seconds=round(time.monotonic() - prepare_started, 6),
    )
    turbine_series_map = build_turbine_series_map(prepared_series)
    metrics = _initialize_metrics()

    for target_turbine_id in target_turbine_ids:
        panel_started = time.monotonic()
        local_panel = build_local_panel(
            turbine_series_map,
            turbine_ids=neighborhood_map[target_turbine_id],
            resolution_minutes=spec.resolution_minutes,
        )
        _profile_log(
            spec.dataset_id,
            "build_local_panel",
            target_turbine_id=target_turbine_id,
            input_turbines=len(neighborhood_map[target_turbine_id]),
            timestamps=len(local_panel.timestamps_us),
            rows=int(local_panel.target_kw_masked.shape[0]),
            columns=int(local_panel.target_kw_masked.shape[1]),
            duration_seconds=round(time.monotonic() - panel_started, 6),
        )
        for context_batch, actual_batch, future_timestamps_batch in _iter_multivariate_batches(
            farm_panel=local_panel,
            history_steps=resolved_task.history_steps,
            forecast_steps=resolved_task.forecast_steps,
            stride_steps=resolved_task.stride_steps,
            batch_size=batch_size,
            window_offset=window_offset,
            max_windows_per_dataset=max_windows_per_dataset,
            scored_row_index=0,
        ):
            quantiles, _ = pipeline.predict_quantiles(
                inputs=context_batch,
                prediction_length=resolved_task.forecast_steps,
                quantile_levels=[0.5],
                batch_size=resolve_pipeline_batch_size(context_batch),
                limit_prediction_length=False,
                cross_learning=False,
            )
            prediction_batch = np.stack([_extract_median_forecast(prediction) for prediction in quantiles])
            target_prediction_batch = prediction_batch[:, 0, :]
            target_actual_batch = actual_batch[:, 0, :]
            valid_mask = ~np.isnan(target_actual_batch)
            if not valid_mask.any():
                continue
            valid_window_mask = ~np.isnan(target_actual_batch).all(axis=1)

            errors = target_prediction_batch - target_actual_batch
            valid_errors = errors[valid_mask]
            normalized_valid_errors = valid_errors / rated_power_kw

            metrics["window_count"] = int(metrics["window_count"]) + int(valid_window_mask.sum())
            metrics["prediction_count"] = int(metrics["prediction_count"]) + int(valid_mask.sum())
            metrics["abs_error_sum"] = float(metrics["abs_error_sum"]) + float(np.abs(valid_errors).sum())
            metrics["squared_error_sum"] = float(metrics["squared_error_sum"]) + float(np.square(valid_errors).sum())
            metrics["normalized_abs_error_sum"] = float(metrics["normalized_abs_error_sum"]) + float(
                np.abs(normalized_valid_errors).sum()
            )
            metrics["normalized_squared_error_sum"] = float(metrics["normalized_squared_error_sum"]) + float(
                np.square(normalized_valid_errors).sum()
            )
            batch_start_us, batch_end_us = _valid_timestamp_span_from_univariate(
                future_timestamps_batch,
                valid_mask,
            )
            _update_timestamp_bounds(metrics, batch_start_us, batch_end_us)

    runtime_seconds = time.monotonic() - dataset_start
    _profile_log(
        spec.dataset_id,
        "evaluate_multivariate_complete",
        target_turbines=len(target_turbine_ids),
        input_turbines=len(required_input_turbine_ids),
        runtime_seconds=round(runtime_seconds, 6),
        window_count=int(metrics["window_count"]),
        prediction_count=int(metrics["prediction_count"]),
    )
    return _build_result_row(
        dataset_id=spec.dataset_id,
        suffix=MULTIVARIATE_KNN6_SUFFIX,
        resolved_task=resolved_task,
        rated_power_kw=rated_power_kw,
        metrics=metrics,
        runtime_seconds=runtime_seconds,
        device=device or select_device(),
    )


def run_experiment(
    *,
    dataset_ids: Sequence[str] = DEFAULT_DATASETS,
    cache_root: str | Path = _CACHE_ROOT,
    output_path: str | Path = _OUTPUT_PATH,
    batch_size: int = DEFAULT_BATCH_SIZE,
    window_offset: int = 0,
    max_windows_per_dataset: int | None = None,
    device: str | None = None,
    task_spec=None,
    pipeline: Any | None = None,
    mode: str = "all",
    turbine_ids: Sequence[str] | None = None,
) -> pl.DataFrame:
    if mode not in MODE_CHOICES:
        raise ValueError(f"Unsupported mode {mode!r}. Expected one of {MODE_CHOICES}.")

    resolved_task_spec = task_spec or build_task_spec()
    resolved_device = device or select_device()
    resolved_pipeline = pipeline or load_pipeline(device=resolved_device)
    evaluators = []
    if mode in {"all", "univariate"}:
        evaluators.append(evaluate_univariate_dataset)
    if mode in {"all", "multivariate_knn6"}:
        evaluators.append(evaluate_multivariate_knn6_dataset)

    rows = [
        evaluator(
            dataset_id,
            pipeline=resolved_pipeline,
            cache_root=cache_root,
            task_spec=resolved_task_spec,
            batch_size=batch_size,
            window_offset=window_offset,
            max_windows_per_dataset=max_windows_per_dataset,
            device=resolved_device,
            turbine_ids=turbine_ids,
        )
        for evaluator in evaluators
        for dataset_id in dataset_ids
    ]
    results = pl.DataFrame(rows).select(_RESULT_COLUMNS).sort("dataset_id")
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    results.write_csv(output)
    return results


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Chronos-2 power_only zero-shot experiments.")
    parser.add_argument(
        "--dataset",
        action="append",
        choices=sorted(set(DEFAULT_DATASETS) | set(_DATASET_ID_ALIASES)),
        dest="datasets",
        help="Limit execution to one or more datasets. Defaults to all supported datasets.",
    )
    parser.add_argument(
        "--cache-root",
        type=Path,
        default=_CACHE_ROOT,
        help="Cache root containing built dataset artifacts.",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=_OUTPUT_PATH,
        help="Output CSV path. Defaults to experiment/chronos-2.csv in the repo root.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help="Prediction batch size passed to Chronos-2.",
    )
    parser.add_argument(
        "--window-offset",
        type=int,
        default=0,
        help="Skip this many retained evaluation windows before scoring.",
    )
    parser.add_argument(
        "--max-windows-per-dataset",
        type=int,
        default=None,
        help="Optional smoke-test limit on the number of retained windows evaluated per dataset.",
    )
    parser.add_argument(
        "--device",
        choices=["cuda", "mps", "cpu"],
        default=None,
        help="Override automatic device selection.",
    )
    parser.add_argument(
        "--mode",
        choices=list(MODE_CHOICES),
        default="all",
        help="Run all experiments, or only the univariate or multivariate_knn6 subset.",
    )
    parser.add_argument(
        "--turbine-id",
        action="append",
        dest="turbine_ids",
        help="Optional turbine subset for univariate runs. May be passed multiple times.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    results = run_experiment(
        dataset_ids=tuple(args.datasets) if args.datasets else DEFAULT_DATASETS,
        cache_root=args.cache_root,
        output_path=args.output_path,
        batch_size=args.batch_size,
        window_offset=args.window_offset,
        max_windows_per_dataset=args.max_windows_per_dataset,
        device=args.device,
        mode=args.mode,
        turbine_ids=tuple(args.turbine_ids) if args.turbine_ids else None,
    )
    print(results)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
