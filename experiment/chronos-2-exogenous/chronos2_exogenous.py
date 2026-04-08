from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import UTC, datetime
import json
import math
from pathlib import Path
import sys
import time
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import polars as pl

EXPERIMENT_DIR = Path(__file__).resolve().parent
if str(EXPERIMENT_DIR) not in sys.path:
    sys.path.insert(0, str(EXPERIMENT_DIR))

from chronos2_exogenous_manifest import (
    DEFAULT_COVARIATE_STAGES,
    CovariatePackSpec,
    iter_covariate_packs,
    reference_pack_for,
    resolve_covariate_pack,
)


MODEL_ID = "amazon/chronos-2"
TASK_ID = "next_6h_from_24h_stride_6h"
TARGET_POLICY = "invalid_to_nan_clip_0_rated"
DEFAULT_DATASETS = ("kelmarsh", "penmanshiel", "hill_of_towie", "sdwpf_kddcup")
DEFAULT_SERIES_BUDGET = 1024
COVARIATE_POLICY = "dataset_custom_past_only"
LAYOUT = "univariate"
_REPO_ROOT = EXPERIMENT_DIR.parents[1]
_CACHE_ROOT = _REPO_ROOT / "cache"
_OUTPUT_PATH = _REPO_ROOT / "experiment" / "chronos-2-exogenous.csv"
_BASE_COLUMNS = (
    "dataset",
    "turbine_id",
    "timestamp",
    "target_kw",
    "quality_flags",
    "feature_quality_flags",
)
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
    "layout",
    "covariate_stage",
    "covariate_pack",
    "feature_set",
    "covariate_count",
    "covariate_policy",
]
_GROUP_KEY_COLUMNS = [
    "dataset_id",
    "model_id",
    "task_id",
    "history_steps",
    "forecast_steps",
    "stride_steps",
    "target_policy",
    "layout",
    "covariate_stage",
    "covariate_pack",
    "feature_set",
    "covariate_count",
    "covariate_policy",
]
PROFILE_LOG_PREFIX = "[chronos2_exogenous] "


@dataclass(frozen=True)
class TurbineExogenousSeries:
    timestamps_us: np.ndarray
    target_kw_masked: np.ndarray
    past_covariates: dict[str, np.ndarray]


@dataclass(frozen=True)
class StageProgressPlan:
    stage: str
    pack_name: str
    feature_set: str
    covariate_count: int
    window_batch_size: int
    total_windows: int
    total_batches: int


@dataclass(frozen=True)
class DatasetProgressPlan:
    dataset_id: str
    selected_turbine_ids: tuple[str, ...]
    retained_windows_by_turbine: dict[str, int]
    stage_plans: tuple[StageProgressPlan, ...]
    total_batches: int


@dataclass
class ChunkProgressState:
    chunk_total_batches: int
    completed_chunk_batches: int = 0


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


def resolve_selected_turbine_ids(
    spec: Any,
    turbine_ids: Sequence[str] | None,
) -> tuple[str, ...]:
    if turbine_ids is None:
        return tuple(spec.turbine_ids)
    selected_turbine_ids = _ordered_unique(turbine_ids)
    unknown_turbine_ids = [turbine_id for turbine_id in selected_turbine_ids if turbine_id not in spec.turbine_ids]
    if unknown_turbine_ids:
        raise ValueError(f"Unknown turbine ids for dataset {spec.dataset_id!r}: {unknown_turbine_ids!r}")
    return selected_turbine_ids


def _profile_log(dataset_id: str, phase: str, **fields: object) -> None:
    payload = {
        "dataset_id": resolve_dataset_id(dataset_id),
        "phase": phase,
        **fields,
    }
    print(
        f"{PROFILE_LOG_PREFIX}{json.dumps(payload, ensure_ascii=False, default=str, sort_keys=True)}",
        file=sys.stderr,
        flush=True,
    )


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
            "Rebuild experiment/chronos-2-exogenous/.conda with ./create_env.sh."
        ) from exc

    resolved_device = device or select_device()
    return Chronos2Pipeline.from_pretrained(model_id, device_map=resolved_device)


def resolve_turbine_series_path(
    dataset_id: str,
    *,
    feature_set: str,
    cache_root: str | Path = _CACHE_ROOT,
) -> tuple[Any, str, Path]:
    _ensure_repo_src_on_path()
    from wind_datasets.datasets import get_builder

    spec, _ = resolve_dataset_task(dataset_id)
    cache_root_path = Path(cache_root)
    builder = get_builder(spec, cache_root_path)
    resolved_feature_set = builder.resolve_feature_set(feature_set)
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
    return spec, resolved_feature_set, series_path


def load_covariate_series_frame(
    dataset_id: str,
    *,
    pack: CovariatePackSpec,
    cache_root: str | Path = _CACHE_ROOT,
    turbine_ids: Sequence[str] | None = None,
) -> tuple[Any, str, Path, tuple[str, ...], pl.DataFrame]:
    spec, resolved_feature_set, series_path = resolve_turbine_series_path(
        dataset_id,
        feature_set=pack.feature_set,
        cache_root=cache_root,
    )
    requested_turbine_ids = _ordered_unique(turbine_ids) if turbine_ids is not None else tuple(spec.turbine_ids)
    available_columns = set(pl.read_parquet_schema(series_path))
    covariate_columns = pack.selected_covariate_columns(available_columns)
    selected_columns = tuple(_BASE_COLUMNS) + covariate_columns
    missing_columns = [column for column in _BASE_COLUMNS if column not in available_columns]
    if missing_columns:
        raise ValueError(
            f"Series {series_path} for dataset {spec.dataset_id!r} is missing required base columns {missing_columns!r}."
        )

    load_started = time.monotonic()
    series_scan = pl.scan_parquet(series_path).select(list(selected_columns))
    if requested_turbine_ids != tuple(spec.turbine_ids):
        series_scan = series_scan.filter(pl.col("turbine_id").is_in(list(requested_turbine_ids)))
    series = series_scan.collect()
    _profile_log(
        spec.dataset_id,
        "load_covariate_series",
        stage=pack.stage,
        pack=pack.pack_name,
        feature_set=resolved_feature_set,
        path=str(series_path),
        rows=series.height,
        columns=len(selected_columns),
        covariates=len(covariate_columns),
        target_turbines=None if turbine_ids is None else len(requested_turbine_ids),
        input_turbines=len(requested_turbine_ids),
        duration_seconds=round(time.monotonic() - load_started, 6),
    )
    return spec, resolved_feature_set, series_path, covariate_columns, series


def load_target_series_frame(
    dataset_id: str,
    *,
    feature_set: str,
    cache_root: str | Path = _CACHE_ROOT,
    turbine_ids: Sequence[str] | None = None,
) -> tuple[Any, str, Path, pl.DataFrame]:
    spec, resolved_feature_set, series_path = resolve_turbine_series_path(
        dataset_id,
        feature_set=feature_set,
        cache_root=cache_root,
    )
    requested_turbine_ids = _ordered_unique(turbine_ids) if turbine_ids is not None else tuple(spec.turbine_ids)
    load_started = time.monotonic()
    series_scan = pl.scan_parquet(series_path).select(list(_BASE_COLUMNS))
    if requested_turbine_ids != tuple(spec.turbine_ids):
        series_scan = series_scan.filter(pl.col("turbine_id").is_in(list(requested_turbine_ids)))
    series = series_scan.collect()
    _profile_log(
        spec.dataset_id,
        "load_target_series",
        feature_set=resolved_feature_set,
        path=str(series_path),
        rows=series.height,
        columns=len(_BASE_COLUMNS),
        target_turbines=None if turbine_ids is None else len(requested_turbine_ids),
        input_turbines=len(requested_turbine_ids),
        duration_seconds=round(time.monotonic() - load_started, 6),
    )
    return spec, resolved_feature_set, series_path, series


def prepare_exogenous_series(
    series: pl.DataFrame,
    *,
    covariate_columns: Sequence[str],
    rated_power_kw: float,
) -> pl.DataFrame:
    invalid_expr = pl.col("target_kw").is_null() | (pl.col("quality_flags").fill_null("") != "")
    covariate_expressions: list[pl.Expr] = []
    for column in covariate_columns:
        dtype = series.schema[column]
        if dtype == pl.Boolean:
            covariate_expressions.append(
                pl.when(pl.col(column).is_null())
                .then(pl.lit(None, dtype=pl.Float64))
                .otherwise(pl.col(column).cast(pl.UInt8))
                .cast(pl.Float64)
                .alias(column)
            )
        else:
            covariate_expressions.append(
                pl.col(column).cast(pl.Float64, strict=False).alias(column)
            )

    return (
        series.sort(["turbine_id", "timestamp"])
        .with_columns(
            invalid_expr.alias("invalid_target"),
            pl.when(invalid_expr)
            .then(pl.lit(None, dtype=pl.Float64))
            .otherwise(pl.col("target_kw").clip(0.0, rated_power_kw))
            .alias("target_kw_masked"),
            *covariate_expressions,
        )
        .select(
            [
                *_BASE_COLUMNS,
                "invalid_target",
                "target_kw_masked",
                *covariate_columns,
            ]
        )
    )


def build_turbine_exogenous_series_map(
    series: pl.DataFrame,
    *,
    covariate_columns: Sequence[str],
) -> dict[str, TurbineExogenousSeries]:
    turbines: dict[str, TurbineExogenousSeries] = {}
    for turbine_frame in series.partition_by("turbine_id", maintain_order=True):
        turbine_id = turbine_frame["turbine_id"][0]
        turbines[turbine_id] = TurbineExogenousSeries(
            timestamps_us=turbine_frame["timestamp"].cast(pl.Int64).to_numpy(),
            target_kw_masked=turbine_frame["target_kw_masked"].cast(pl.Float32).to_numpy(),
            past_covariates={
                column: turbine_frame[column].cast(pl.Float32).to_numpy()
                for column in covariate_columns
            },
        )
    return turbines


def _count_retained_windows_for_turbine(
    *,
    turbine_series: TurbineExogenousSeries,
    history_steps: int,
    forecast_steps: int,
    stride_steps: int,
    window_offset: int = 0,
    max_windows_per_dataset: int | None = None,
) -> int:
    retained_windows = 0
    skipped_windows = 0
    emitted_windows = 0
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

        retained_windows += 1
        emitted_windows += 1
        if max_windows_per_dataset is not None and emitted_windows >= max_windows_per_dataset:
            break

    return retained_windows


def _selected_packs_for_dataset(
    dataset_id: str,
    *,
    requested_stages: Sequence[str],
    include_power_only_reference: bool,
) -> tuple[CovariatePackSpec, ...]:
    packs: list[CovariatePackSpec] = []
    if include_power_only_reference:
        packs.append(reference_pack_for(dataset_id))
    packs.extend(iter_covariate_packs((dataset_id,), requested_stages))
    return tuple(packs)


def _resolve_dataset_progress_plan(
    dataset_id: str,
    *,
    packs: Sequence[CovariatePackSpec],
    cache_root: str | Path = _CACHE_ROOT,
    task_spec=None,
    series_budget: int = DEFAULT_SERIES_BUDGET,
    window_offset: int = 0,
    max_windows_per_dataset: int | None = None,
    turbine_ids: Sequence[str] | None = None,
) -> DatasetProgressPlan:
    if not packs:
        spec, _ = resolve_dataset_task(dataset_id, task_spec=task_spec)
        selected_turbine_ids = resolve_selected_turbine_ids(spec, turbine_ids)
        return DatasetProgressPlan(
            dataset_id=spec.dataset_id,
            selected_turbine_ids=selected_turbine_ids,
            retained_windows_by_turbine={turbine_id: 0 for turbine_id in selected_turbine_ids},
            stage_plans=(),
            total_batches=0,
        )

    spec, resolved_task = resolve_dataset_task(dataset_id, task_spec=task_spec)
    selected_turbine_ids = resolve_selected_turbine_ids(spec, turbine_ids)
    progress_feature_set = packs[0].feature_set
    spec, _, series_path, target_series = load_target_series_frame(
        dataset_id,
        feature_set=progress_feature_set,
        cache_root=cache_root,
        turbine_ids=selected_turbine_ids,
    )
    rated_power_kw = resolve_rated_power_kw(spec.dataset_id)
    prepared_target_series = prepare_exogenous_series(
        target_series,
        covariate_columns=(),
        rated_power_kw=rated_power_kw,
    )
    turbine_series_map = build_turbine_exogenous_series_map(
        prepared_target_series,
        covariate_columns=(),
    )
    retained_windows_by_turbine = {
        turbine_id: _count_retained_windows_for_turbine(
            turbine_series=turbine_series_map[turbine_id],
            history_steps=resolved_task.history_steps,
            forecast_steps=resolved_task.forecast_steps,
            stride_steps=resolved_task.stride_steps,
            window_offset=window_offset,
            max_windows_per_dataset=max_windows_per_dataset,
        )
        for turbine_id in selected_turbine_ids
    }
    total_windows = sum(retained_windows_by_turbine.values())
    available_columns = set(pl.read_parquet_schema(series_path))
    stage_plans: list[StageProgressPlan] = []
    for pack in packs:
        covariate_count = len(pack.selected_covariate_columns(available_columns))
        window_batch_size = resolve_window_batch_size(
            series_budget=series_budget,
            covariate_count=covariate_count,
        )
        total_batches = sum(
            math.ceil(window_count / window_batch_size)
            for window_count in retained_windows_by_turbine.values()
            if window_count > 0
        )
        stage_plans.append(
            StageProgressPlan(
                stage=pack.stage,
                pack_name=pack.pack_name,
                feature_set=pack.feature_set,
                covariate_count=covariate_count,
                window_batch_size=window_batch_size,
                total_windows=total_windows,
                total_batches=total_batches,
            )
        )
    return DatasetProgressPlan(
        dataset_id=spec.dataset_id,
        selected_turbine_ids=selected_turbine_ids,
        retained_windows_by_turbine=retained_windows_by_turbine,
        stage_plans=tuple(stage_plans),
        total_batches=sum(stage_plan.total_batches for stage_plan in stage_plans),
    )


def _emit_progress_event(
    dataset_id: str,
    *,
    enabled: bool,
    phase: str,
    **fields: object,
) -> None:
    if not enabled:
        return
    _profile_log(dataset_id, phase, **fields)


def _iter_univariate_covariate_batches(
    *,
    turbine_series: TurbineExogenousSeries,
    covariate_columns: Sequence[str],
    history_steps: int,
    forecast_steps: int,
    stride_steps: int,
    window_batch_size: int,
    window_offset: int = 0,
    max_windows_per_dataset: int | None = None,
) -> Iterable[tuple[list[dict[str, object]], np.ndarray, np.ndarray]]:
    input_batch: list[dict[str, object]] = []
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

        batch_item: dict[str, object] = {
            "target": context.astype(np.float32, copy=True),
        }
        if covariate_columns:
            batch_item["past_covariates"] = {
                column: turbine_series.past_covariates[column][anchor_index - history_steps + 1 : anchor_index + 1].astype(
                    np.float32,
                    copy=True,
                )
                for column in covariate_columns
            }
        input_batch.append(batch_item)
        future_rows.append(future.astype(np.float64, copy=True))
        future_timestamps.append(
            turbine_series.timestamps_us[anchor_index + 1 : anchor_index + 1 + forecast_steps].astype(np.int64, copy=True)
        )
        emitted_windows += 1

        if len(input_batch) >= window_batch_size:
            yield (
                input_batch,
                np.stack(future_rows),
                np.stack(future_timestamps),
            )
            input_batch = []
            future_rows = []
            future_timestamps = []

        if max_windows_per_dataset is not None and emitted_windows >= max_windows_per_dataset:
            break

    if input_batch:
        yield (
            input_batch,
            np.stack(future_rows),
            np.stack(future_timestamps),
        )


def _coerce_array(value: Any) -> np.ndarray:
    if isinstance(value, np.ndarray):
        return value
    if hasattr(value, "numpy"):
        return value.numpy()
    return np.asarray(value)


def _extract_median_forecast(prediction: Any) -> np.ndarray:
    array = _coerce_array(prediction)
    if array.ndim != 3 or array.shape[2] != 1:
        raise ValueError(f"Expected quantile forecast with shape (n_variates, horizon, 1), got {array.shape}.")
    return array[:, :, 0].astype(np.float64, copy=False)


def _count_variates(value: Any) -> int:
    array = _coerce_array(value)
    if array.ndim == 1:
        return 1
    if array.ndim == 2:
        return int(array.shape[0])
    raise ValueError(f"Expected a 1d or 2d time series target, got shape {array.shape}.")


def resolve_pipeline_batch_size(inputs: Any) -> int:
    if isinstance(inputs, np.ndarray):
        if inputs.ndim != 3:
            raise ValueError(
                f"Expected context_batch with shape (batch, n_variates, history_length), got {inputs.shape}."
            )
        batch_windows, n_variates, _ = inputs.shape
        return max(1, int(batch_windows * n_variates))

    if isinstance(inputs, Sequence) and not isinstance(inputs, (str, bytes)):
        total_series = 0
        for item in inputs:
            if isinstance(item, Mapping):
                if "target" not in item:
                    raise ValueError("Chronos covariate input dictionaries must include a 'target' key.")
                total_series += _count_variates(item["target"])
                covariate_names = set()
                past_covariates = item.get("past_covariates", {})
                future_covariates = item.get("future_covariates", {})
                if isinstance(past_covariates, Mapping):
                    covariate_names.update(str(name) for name in past_covariates)
                if isinstance(future_covariates, Mapping):
                    covariate_names.update(str(name) for name in future_covariates)
                total_series += len(covariate_names)
            else:
                total_series += _count_variates(item)
        return max(1, total_series)

    raise ValueError(f"Unsupported Chronos input type for batch-size resolution: {type(inputs)!r}.")


def resolve_window_batch_size(
    *,
    series_budget: int,
    covariate_count: int,
    target_variates: int = 1,
) -> int:
    if series_budget <= 0:
        raise ValueError("series_budget must be positive.")
    if target_variates <= 0:
        raise ValueError("target_variates must be positive.")
    per_window_series = target_variates + max(0, covariate_count)
    return max(1, series_budget // per_window_series)


def _valid_timestamp_span_from_univariate(
    future_timestamps_batch: np.ndarray,
    valid_mask: np.ndarray,
) -> tuple[int | None, int | None]:
    if not valid_mask.any():
        return None, None
    valid_timestamps = future_timestamps_batch[valid_mask]
    return int(valid_timestamps.min()), int(valid_timestamps.max())


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
    return float(np.sqrt(squared_error_sum / denominator))


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
    resolved_task: Any,
    metrics: dict[str, float | int | None],
    runtime_seconds: float,
    device: str,
    layout: str,
    covariate_stage: str,
    covariate_pack: str,
    feature_set: str,
    covariate_count: int,
) -> dict[str, object]:
    prediction_count = int(metrics["prediction_count"])
    return {
        "dataset_id": dataset_id,
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
        "layout": layout,
        "covariate_stage": covariate_stage,
        "covariate_pack": covariate_pack,
        "feature_set": feature_set,
        "covariate_count": covariate_count,
        "covariate_policy": COVARIATE_POLICY,
    }


def evaluate_univariate_covariate_pack(
    dataset_id: str,
    *,
    pack: CovariatePackSpec,
    pipeline: Any,
    cache_root: str | Path = _CACHE_ROOT,
    task_spec=None,
    series_budget: int = DEFAULT_SERIES_BUDGET,
    window_offset: int = 0,
    max_windows_per_dataset: int | None = None,
    device: str | None = None,
    turbine_ids: Sequence[str] | None = None,
    emit_progress_events: bool = False,
    progress_state: ChunkProgressState | None = None,
    stage_progress_plan: StageProgressPlan | None = None,
) -> dict[str, object]:
    dataset_start = time.monotonic()
    spec, resolved_task = resolve_dataset_task(dataset_id, task_spec=task_spec)
    selected_turbine_ids = resolve_selected_turbine_ids(spec, turbine_ids)
    spec, resolved_feature_set, _, covariate_columns, series = load_covariate_series_frame(
        dataset_id,
        pack=pack,
        cache_root=cache_root,
        turbine_ids=selected_turbine_ids,
    )
    rated_power_kw = resolve_rated_power_kw(spec.dataset_id)
    prepare_started = time.monotonic()
    prepared_series = prepare_exogenous_series(
        series,
        covariate_columns=covariate_columns,
        rated_power_kw=rated_power_kw,
    )
    _profile_log(
        spec.dataset_id,
        "prepare_exogenous_series",
        stage=pack.stage,
        pack=pack.pack_name,
        rows=prepared_series.height,
        columns=prepared_series.width,
        target_turbines=len(selected_turbine_ids),
        input_turbines=len(selected_turbine_ids),
        covariates=len(covariate_columns),
        duration_seconds=round(time.monotonic() - prepare_started, 6),
    )
    turbine_series_map = build_turbine_exogenous_series_map(
        prepared_series,
        covariate_columns=covariate_columns,
    )
    window_batch_size = resolve_window_batch_size(
        series_budget=series_budget,
        covariate_count=len(covariate_columns),
    )
    metrics = _initialize_metrics()
    if emit_progress_events:
        if progress_state is None or stage_progress_plan is None:
            raise ValueError("emit_progress_events requires progress_state and stage_progress_plan.")
        if stage_progress_plan.pack_name != pack.pack_name:
            raise ValueError(
                f"Stage progress plan pack {stage_progress_plan.pack_name!r} does not match {pack.pack_name!r}."
            )
        _emit_progress_event(
            spec.dataset_id,
            enabled=True,
            phase="progress_stage_start",
            stage=pack.stage,
            pack=pack.pack_name,
            feature_set=resolved_feature_set,
            covariate_count=len(covariate_columns),
            stage_total_batches=stage_progress_plan.total_batches,
            completed_stage_batches=0,
            completed_chunk_batches=progress_state.completed_chunk_batches,
            chunk_total_batches=progress_state.chunk_total_batches,
        )
    stage_completed_batches = 0

    for turbine_id in selected_turbine_ids:
        turbine_series = turbine_series_map[turbine_id]
        for input_batch, actual_batch, future_timestamps_batch in _iter_univariate_covariate_batches(
            turbine_series=turbine_series,
            covariate_columns=covariate_columns,
            history_steps=resolved_task.history_steps,
            forecast_steps=resolved_task.forecast_steps,
            stride_steps=resolved_task.stride_steps,
            window_batch_size=window_batch_size,
            window_offset=window_offset,
            max_windows_per_dataset=max_windows_per_dataset,
        ):
            quantiles, _ = pipeline.predict_quantiles(
                inputs=input_batch,
                prediction_length=resolved_task.forecast_steps,
                quantile_levels=[0.5],
                batch_size=resolve_pipeline_batch_size(input_batch),
                limit_prediction_length=False,
            )
            if emit_progress_events:
                stage_completed_batches += 1
                progress_state.completed_chunk_batches += 1
                _emit_progress_event(
                    spec.dataset_id,
                    enabled=True,
                    phase="progress_batch",
                    stage=pack.stage,
                    pack=pack.pack_name,
                    feature_set=resolved_feature_set,
                    covariate_count=len(covariate_columns),
                    turbine_id=turbine_id,
                    completed_stage_batches=stage_completed_batches,
                    stage_total_batches=stage_progress_plan.total_batches,
                    completed_chunk_batches=progress_state.completed_chunk_batches,
                    chunk_total_batches=progress_state.chunk_total_batches,
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

    if emit_progress_events:
        _emit_progress_event(
            spec.dataset_id,
            enabled=True,
            phase="progress_stage_complete",
            stage=pack.stage,
            pack=pack.pack_name,
            feature_set=resolved_feature_set,
            covariate_count=len(covariate_columns),
            completed_stage_batches=stage_completed_batches,
            stage_total_batches=stage_progress_plan.total_batches,
            completed_chunk_batches=progress_state.completed_chunk_batches,
            chunk_total_batches=progress_state.chunk_total_batches,
        )

    runtime_seconds = time.monotonic() - dataset_start
    _profile_log(
        spec.dataset_id,
        "evaluate_univariate_complete",
        stage=pack.stage,
        pack=pack.pack_name,
        target_turbines=len(selected_turbine_ids),
        input_turbines=len(selected_turbine_ids),
        covariates=len(covariate_columns),
        window_batch_size=window_batch_size,
        runtime_seconds=round(runtime_seconds, 6),
        window_count=int(metrics["window_count"]),
        prediction_count=int(metrics["prediction_count"]),
    )
    return _build_result_row(
        dataset_id=spec.dataset_id,
        resolved_task=resolved_task,
        metrics=metrics,
        runtime_seconds=runtime_seconds,
        device=device or select_device(),
        layout=LAYOUT,
        covariate_stage=pack.stage,
        covariate_pack=pack.pack_name,
        feature_set=resolved_feature_set,
        covariate_count=len(covariate_columns),
    )


def run_experiment(
    *,
    dataset_ids: Sequence[str] = DEFAULT_DATASETS,
    covariate_stages: Sequence[str] = DEFAULT_COVARIATE_STAGES,
    include_power_only_reference: bool = False,
    cache_root: str | Path = _CACHE_ROOT,
    output_path: str | Path = _OUTPUT_PATH,
    series_budget: int = DEFAULT_SERIES_BUDGET,
    window_offset: int = 0,
    max_windows_per_dataset: int | None = None,
    device: str | None = None,
    task_spec=None,
    pipeline: Any | None = None,
    turbine_ids: Sequence[str] | None = None,
    emit_progress_events: bool = False,
) -> pl.DataFrame:
    requested_dataset_ids = tuple(resolve_dataset_id(dataset_id) for dataset_id in dataset_ids)
    requested_stages = tuple(covariate_stages)
    if not requested_stages:
        raise ValueError("At least one covariate stage must be selected.")
    for stage in requested_stages:
        if stage not in DEFAULT_COVARIATE_STAGES:
            raise ValueError(
                f"Unsupported covariate stage {stage!r}. Expected one of {DEFAULT_COVARIATE_STAGES!r}."
            )

    resolved_task_spec = task_spec or build_task_spec()
    resolved_device = device or select_device()
    resolved_pipeline = pipeline or load_pipeline(device=resolved_device)
    dataset_packs = {
        dataset_id: _selected_packs_for_dataset(
            dataset_id,
            requested_stages=requested_stages,
            include_power_only_reference=include_power_only_reference,
        )
        for dataset_id in requested_dataset_ids
    }
    dataset_progress_plans: dict[str, DatasetProgressPlan] = {}
    progress_state: ChunkProgressState | None = None
    if emit_progress_events:
        dataset_progress_plans = {
            dataset_id: _resolve_dataset_progress_plan(
                dataset_id,
                packs=dataset_packs[dataset_id],
                cache_root=cache_root,
                task_spec=resolved_task_spec,
                series_budget=series_budget,
                window_offset=window_offset,
                max_windows_per_dataset=max_windows_per_dataset,
                turbine_ids=turbine_ids,
            )
            for dataset_id in requested_dataset_ids
        }
        progress_state = ChunkProgressState(
            chunk_total_batches=sum(plan.total_batches for plan in dataset_progress_plans.values())
        )
        progress_dataset_id = requested_dataset_ids[0] if requested_dataset_ids else "multi_dataset"
        _emit_progress_event(
            progress_dataset_id,
            enabled=True,
            phase="progress_chunk_plan",
            chunk_total_batches=progress_state.chunk_total_batches,
            dataset_count=len(requested_dataset_ids),
        )
    rows: list[dict[str, object]] = []

    for dataset_id in requested_dataset_ids:
        dataset_progress_plan = dataset_progress_plans.get(dataset_id)
        stage_progress_lookup = {
            stage_plan.pack_name: stage_plan
            for stage_plan in (() if dataset_progress_plan is None else dataset_progress_plan.stage_plans)
        }
        for pack in dataset_packs[dataset_id]:
            rows.append(
                evaluate_univariate_covariate_pack(
                    dataset_id,
                    pack=pack,
                    pipeline=resolved_pipeline,
                    cache_root=cache_root,
                    task_spec=resolved_task_spec,
                    series_budget=series_budget,
                    window_offset=window_offset,
                    max_windows_per_dataset=max_windows_per_dataset,
                    device=resolved_device,
                    turbine_ids=turbine_ids,
                    emit_progress_events=emit_progress_events,
                    progress_state=progress_state,
                    stage_progress_plan=stage_progress_lookup.get(pack.pack_name),
                )
            )

    results = (
        pl.DataFrame(rows, schema={column: pl.Null for column in _RESULT_COLUMNS})
        if not rows
        else pl.DataFrame(rows).select(_RESULT_COLUMNS).sort(["dataset_id", "covariate_stage", "covariate_pack"])
    )
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    results.write_csv(output)
    return results


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Chronos-2 exogenous univariate zero-shot experiments.")
    parser.add_argument(
        "--dataset",
        action="append",
        choices=sorted(set(DEFAULT_DATASETS) | set(_DATASET_ID_ALIASES)),
        dest="datasets",
        help="Limit execution to one or more datasets. Defaults to all supported datasets.",
    )
    parser.add_argument(
        "--covariate-stage",
        action="append",
        choices=list(DEFAULT_COVARIATE_STAGES),
        dest="covariate_stages",
        help="Limit execution to one or more staged covariate packs. Defaults to all stages.",
    )
    parser.add_argument(
        "--include-power-only-reference",
        action="store_true",
        help="Also emit one power-only reference row per dataset in the same output file.",
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
        help="Output CSV path. Defaults to experiment/chronos-2-exogenous.csv in the repo root.",
    )
    parser.add_argument(
        "--series-budget",
        type=int,
        default=DEFAULT_SERIES_BUDGET,
        help="Maximum number of target-plus-covariate series passed to Chronos-2 per batch.",
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
        "--turbine-id",
        action="append",
        dest="turbine_ids",
        help="Optional turbine subset for univariate runs. May be passed multiple times.",
    )
    parser.add_argument(
        "--emit-progress-events",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    results = run_experiment(
        dataset_ids=tuple(args.datasets) if args.datasets else DEFAULT_DATASETS,
        covariate_stages=tuple(args.covariate_stages) if args.covariate_stages else DEFAULT_COVARIATE_STAGES,
        include_power_only_reference=bool(args.include_power_only_reference),
        cache_root=args.cache_root,
        output_path=args.output_path,
        series_budget=args.series_budget,
        window_offset=args.window_offset,
        max_windows_per_dataset=args.max_windows_per_dataset,
        device=args.device,
        turbine_ids=tuple(args.turbine_ids) if args.turbine_ids else None,
        emit_progress_events=bool(args.emit_progress_events),
    )
    print(results)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
