from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import UTC, datetime
from math import sqrt
from pathlib import Path
import sys
import time
from typing import Any, Iterable, Sequence

import numpy as np
import polars as pl

MODEL_ID = "amazon/chronos-2"
TASK_ID = "next_6h_from_24h_stride_6h"
TARGET_POLICY = "clip_0_rated"
DEFAULT_DATASETS = ("kelmarsh", "penmanshiel", "hill_of_towie")
DEFAULT_BATCH_SIZE = 256
MULTIVARIATE_SUFFIX = "_multivariate"
MODE_CHOICES = ("univariate", "multivariate", "append_multivariate")
_REPO_ROOT = Path(__file__).resolve().parents[2]
_CACHE_ROOT = _REPO_ROOT / "cache"
_OUTPUT_PATH = _REPO_ROOT / "experiment" / "chronos-2.csv"
_DATASET_RATED_POWER_KW = {
    "kelmarsh": 2050.0,
    "penmanshiel": 2050.0,
    "hill_of_towie": 2300.0,
    "sdwpf_full": 1500.0,
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
    target_kw_clipped: np.ndarray

    @property
    def first_timestamp_us(self) -> int:
        return int(self.timestamps_us[0])


@dataclass(frozen=True)
class FarmPanel:
    turbine_ids: tuple[str, ...]
    timestamps_us: np.ndarray
    target_kw_clipped: np.ndarray


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


def resolve_rated_power_kw(dataset_id: str) -> float:
    try:
        return _DATASET_RATED_POWER_KW[dataset_id]
    except KeyError as exc:
        raise KeyError(f"No rated power is configured for dataset_id {dataset_id!r}.") from exc


def clip_target_values(values: Sequence[float | None], rated_power_kw: float) -> np.ndarray:
    array = np.asarray(values, dtype=np.float32)
    clipped = array.copy()
    valid = ~np.isnan(clipped)
    clipped[valid] = np.clip(clipped[valid], 0.0, rated_power_kw)
    return clipped


def prepare_power_only_series(series: pl.DataFrame, rated_power_kw: float) -> pl.DataFrame:
    return (
        series.sort(["turbine_id", "timestamp"])
        .with_columns(pl.col("target_kw").clip(0.0, rated_power_kw).alias("target_kw_clipped"))
        .select(["dataset", "turbine_id", "timestamp", "target_kw", "target_kw_clipped"])
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
) -> tuple[Any, Any, pl.DataFrame, pl.DataFrame]:
    _ensure_repo_src_on_path()
    from wind_datasets import get_dataset_spec, load_series, load_window_index

    spec = get_dataset_spec(dataset_id)
    resolved_task_spec = task_spec or build_task_spec()
    resolved_task = resolved_task_spec.resolve(spec.resolution_minutes)
    series = load_series(
        dataset_id,
        cache_root=cache_root,
        quality_profile=spec.default_quality_profile,
        layout="turbine",
    )
    window_index = load_window_index(
        dataset_id,
        task_spec=resolved_task_spec,
        cache_root=cache_root,
        quality_profile=spec.default_quality_profile,
    )
    return spec, resolved_task, series, window_index


def build_turbine_series_map(series: pl.DataFrame) -> dict[str, TurbineSeries]:
    turbines: dict[str, TurbineSeries] = {}
    for turbine_frame in series.partition_by("turbine_id", maintain_order=True):
        turbine_id = turbine_frame["turbine_id"][0]
        turbines[turbine_id] = TurbineSeries(
            timestamps_us=turbine_frame["timestamp"].cast(pl.Int64).to_numpy(),
            target_kw_clipped=turbine_frame["target_kw_clipped"].cast(pl.Float32).to_numpy(),
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
        series.select(["timestamp", "turbine_id", "target_kw_clipped"])
        .pivot(on="turbine_id", index="timestamp", values="target_kw_clipped", aggregate_function="first")
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
        target_kw_clipped=np.asarray(
            aligned.select(list(ordered_turbine_ids)).to_numpy(),
            dtype=np.float32,
        ),
    )


def filter_complete_windows(window_index: pl.DataFrame, max_windows_per_dataset: int | None = None) -> pl.DataFrame:
    filtered = (
        window_index.filter(pl.col("is_complete_input") & pl.col("is_complete_output"))
        .sort(["turbine_id", "input_end_ts"])
    )
    if max_windows_per_dataset is not None:
        filtered = filtered.head(max_windows_per_dataset)
    return filtered


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


def _iter_valid_batches(
    *,
    turbine_series: TurbineSeries,
    anchor_indices: np.ndarray,
    history_steps: int,
    forecast_steps: int,
    batch_size: int,
) -> Iterable[tuple[list[np.ndarray], np.ndarray, np.ndarray, np.ndarray]]:
    context_batch: list[np.ndarray] = []
    future_rows: list[np.ndarray] = []
    future_start_us: list[int] = []
    future_end_us: list[int] = []

    for anchor_index in anchor_indices:
        context = turbine_series.target_kw_clipped[anchor_index - history_steps + 1 : anchor_index + 1]
        future = turbine_series.target_kw_clipped[anchor_index + 1 : anchor_index + 1 + forecast_steps]
        if context.shape[0] != history_steps or future.shape[0] != forecast_steps:
            continue
        if np.isnan(context).any() or np.isnan(future).any():
            continue

        context_batch.append(context.astype(np.float32, copy=True))
        future_rows.append(future.astype(np.float64, copy=True))
        future_start_us.append(int(turbine_series.timestamps_us[anchor_index + 1]))
        future_end_us.append(int(turbine_series.timestamps_us[anchor_index + forecast_steps]))

        if len(context_batch) >= batch_size:
            yield (
                context_batch,
                np.stack(future_rows),
                np.asarray(future_start_us, dtype=np.int64),
                np.asarray(future_end_us, dtype=np.int64),
            )
            context_batch = []
            future_rows = []
            future_start_us = []
            future_end_us = []

    if context_batch:
        yield (
            context_batch,
            np.stack(future_rows),
            np.asarray(future_start_us, dtype=np.int64),
            np.asarray(future_end_us, dtype=np.int64),
        )


def _iter_multivariate_batches(
    *,
    farm_panel: FarmPanel,
    history_steps: int,
    forecast_steps: int,
    stride_steps: int,
    batch_size: int,
    max_windows_per_dataset: int | None = None,
) -> Iterable[tuple[list[np.ndarray], np.ndarray, np.ndarray, np.ndarray]]:
    context_batch: list[np.ndarray] = []
    future_rows: list[np.ndarray] = []
    future_start_us: list[int] = []
    future_end_us: list[int] = []
    emitted_windows = 0
    max_anchor_index = farm_panel.timestamps_us.shape[0] - forecast_steps

    for anchor_index in range(history_steps - 1, max_anchor_index, stride_steps):
        context = farm_panel.target_kw_clipped[anchor_index - history_steps + 1 : anchor_index + 1].T
        future = farm_panel.target_kw_clipped[anchor_index + 1 : anchor_index + 1 + forecast_steps].T
        if context.shape != (len(farm_panel.turbine_ids), history_steps):
            continue
        if future.shape != (len(farm_panel.turbine_ids), forecast_steps):
            continue
        if np.isnan(future).all():
            continue

        context_batch.append(context.astype(np.float32, copy=True))
        future_rows.append(future.astype(np.float64, copy=True))
        future_start_us.append(int(farm_panel.timestamps_us[anchor_index + 1]))
        future_end_us.append(int(farm_panel.timestamps_us[anchor_index + forecast_steps]))
        emitted_windows += 1

        if len(context_batch) >= batch_size:
            yield (
                context_batch,
                np.stack(future_rows),
                np.asarray(future_start_us, dtype=np.int64),
                np.asarray(future_end_us, dtype=np.int64),
            )
            context_batch = []
            future_rows = []
            future_start_us = []
            future_end_us = []

        if max_windows_per_dataset is not None and emitted_windows >= max_windows_per_dataset:
            break

    if context_batch:
        yield (
            context_batch,
            np.stack(future_rows),
            np.asarray(future_start_us, dtype=np.int64),
            np.asarray(future_end_us, dtype=np.int64),
        )


def evaluate_dataset(
    dataset_id: str,
    *,
    pipeline: Any,
    cache_root: str | Path = _CACHE_ROOT,
    task_spec=None,
    batch_size: int = DEFAULT_BATCH_SIZE,
    max_windows_per_dataset: int | None = None,
    device: str | None = None,
) -> dict[str, object]:
    dataset_start = time.monotonic()
    spec, resolved_task, series, window_index = load_dataset_inputs(
        dataset_id,
        cache_root=cache_root,
        task_spec=task_spec,
    )
    rated_power_kw = resolve_rated_power_kw(dataset_id)
    prepared_series = prepare_power_only_series(series, rated_power_kw=rated_power_kw)
    turbine_series_map = build_turbine_series_map(prepared_series)
    candidate_windows = filter_complete_windows(
        window_index,
        max_windows_per_dataset=max_windows_per_dataset,
    )

    abs_error_sum = 0.0
    squared_error_sum = 0.0
    normalized_abs_error_sum = 0.0
    normalized_squared_error_sum = 0.0
    window_count = 0
    prediction_count = 0
    start_timestamp_us: int | None = None
    end_timestamp_us: int | None = None
    resolution_us = spec.resolution_minutes * 60 * 1_000_000

    for turbine_windows in candidate_windows.partition_by("turbine_id", maintain_order=True):
        turbine_id = turbine_windows["turbine_id"][0]
        turbine_series = turbine_series_map[turbine_id]
        anchor_timestamps_us = turbine_windows["input_end_ts"].cast(pl.Int64).to_numpy()
        anchor_indices = ((anchor_timestamps_us - turbine_series.first_timestamp_us) // resolution_us).astype(
            np.int64,
            copy=False,
        )

        for contexts, actual_batch, future_start_batch, future_end_batch in _iter_valid_batches(
            turbine_series=turbine_series,
            anchor_indices=anchor_indices,
            history_steps=resolved_task.history_steps,
            forecast_steps=resolved_task.forecast_steps,
            batch_size=batch_size,
        ):
            quantiles, _ = pipeline.predict_quantiles(
                inputs=contexts,
                prediction_length=resolved_task.forecast_steps,
                quantile_levels=[0.5],
                batch_size=min(batch_size, len(contexts)),
                limit_prediction_length=False,
            )
            prediction_batch = np.stack([_extract_median_forecast(prediction)[0] for prediction in quantiles])
            errors = prediction_batch - actual_batch
            normalized_errors = errors / rated_power_kw

            window_count += prediction_batch.shape[0]
            prediction_count += prediction_batch.size
            abs_error_sum += float(np.abs(errors).sum())
            squared_error_sum += float(np.square(errors).sum())
            normalized_abs_error_sum += float(np.abs(normalized_errors).sum())
            normalized_squared_error_sum += float(np.square(normalized_errors).sum())
            batch_start_us = int(future_start_batch.min())
            batch_end_us = int(future_end_batch.max())
            start_timestamp_us = batch_start_us if start_timestamp_us is None else min(start_timestamp_us, batch_start_us)
            end_timestamp_us = batch_end_us if end_timestamp_us is None else max(end_timestamp_us, batch_end_us)

    runtime_seconds = time.monotonic() - dataset_start
    return {
        "dataset_id": dataset_id,
        "model_id": MODEL_ID,
        "task_id": TASK_ID,
        "history_steps": resolved_task.history_steps,
        "forecast_steps": resolved_task.forecast_steps,
        "stride_steps": resolved_task.stride_steps,
        "target_policy": TARGET_POLICY,
        "window_count": window_count,
        "prediction_count": prediction_count,
        "start_timestamp": _timestamp_us_to_string(start_timestamp_us),
        "end_timestamp": _timestamp_us_to_string(end_timestamp_us),
        "mae_kw": _safe_divide(abs_error_sum, prediction_count),
        "rmse_kw": _safe_rmse(squared_error_sum, prediction_count),
        "mae_pu": _safe_divide(normalized_abs_error_sum, prediction_count),
        "rmse_pu": _safe_rmse(normalized_squared_error_sum, prediction_count),
        "device": device or select_device(),
        "runtime_seconds": round(runtime_seconds, 6),
    }


def evaluate_multivariate_dataset(
    dataset_id: str,
    *,
    pipeline: Any,
    cache_root: str | Path = _CACHE_ROOT,
    task_spec=None,
    batch_size: int = DEFAULT_BATCH_SIZE,
    max_windows_per_dataset: int | None = None,
    device: str | None = None,
) -> dict[str, object]:
    dataset_start = time.monotonic()
    spec, resolved_task, series, _ = load_dataset_inputs(
        dataset_id,
        cache_root=cache_root,
        task_spec=task_spec,
    )
    rated_power_kw = resolve_rated_power_kw(dataset_id)
    prepared_series = prepare_power_only_series(series, rated_power_kw=rated_power_kw)
    farm_panel = build_farm_panel(
        prepared_series,
        turbine_ids=spec.turbine_ids,
        resolution_minutes=spec.resolution_minutes,
    )

    abs_error_sum = 0.0
    squared_error_sum = 0.0
    normalized_abs_error_sum = 0.0
    normalized_squared_error_sum = 0.0
    window_count = 0
    prediction_count = 0
    start_timestamp_us: int | None = None
    end_timestamp_us: int | None = None

    for contexts, actual_batch, future_start_batch, future_end_batch in _iter_multivariate_batches(
        farm_panel=farm_panel,
        history_steps=resolved_task.history_steps,
        forecast_steps=resolved_task.forecast_steps,
        stride_steps=resolved_task.stride_steps,
        batch_size=batch_size,
        max_windows_per_dataset=max_windows_per_dataset,
    ):
        quantiles, _ = pipeline.predict_quantiles(
            inputs=contexts,
            prediction_length=resolved_task.forecast_steps,
            quantile_levels=[0.5],
            batch_size=min(batch_size, len(contexts)),
            limit_prediction_length=False,
            cross_learning=False,
        )
        prediction_batch = np.stack([_extract_median_forecast(prediction) for prediction in quantiles])
        valid_mask = ~np.isnan(actual_batch)
        if not valid_mask.any():
            continue

        errors = prediction_batch - actual_batch
        valid_errors = errors[valid_mask]
        normalized_valid_errors = valid_errors / rated_power_kw

        window_count += prediction_batch.shape[0]
        prediction_count += int(valid_mask.sum())
        abs_error_sum += float(np.abs(valid_errors).sum())
        squared_error_sum += float(np.square(valid_errors).sum())
        normalized_abs_error_sum += float(np.abs(normalized_valid_errors).sum())
        normalized_squared_error_sum += float(np.square(normalized_valid_errors).sum())
        batch_start_us = int(future_start_batch.min())
        batch_end_us = int(future_end_batch.max())
        start_timestamp_us = batch_start_us if start_timestamp_us is None else min(start_timestamp_us, batch_start_us)
        end_timestamp_us = batch_end_us if end_timestamp_us is None else max(end_timestamp_us, batch_end_us)

    runtime_seconds = time.monotonic() - dataset_start
    return {
        "dataset_id": f"{dataset_id}{MULTIVARIATE_SUFFIX}",
        "model_id": MODEL_ID,
        "task_id": TASK_ID,
        "history_steps": resolved_task.history_steps,
        "forecast_steps": resolved_task.forecast_steps,
        "stride_steps": resolved_task.stride_steps,
        "target_policy": TARGET_POLICY,
        "window_count": window_count,
        "prediction_count": prediction_count,
        "start_timestamp": _timestamp_us_to_string(start_timestamp_us),
        "end_timestamp": _timestamp_us_to_string(end_timestamp_us),
        "mae_kw": _safe_divide(abs_error_sum, prediction_count),
        "rmse_kw": _safe_rmse(squared_error_sum, prediction_count),
        "mae_pu": _safe_divide(normalized_abs_error_sum, prediction_count),
        "rmse_pu": _safe_rmse(normalized_squared_error_sum, prediction_count),
        "device": device or select_device(),
        "runtime_seconds": round(runtime_seconds, 6),
    }


def _merge_results(output_path: str | Path, new_results: pl.DataFrame) -> pl.DataFrame:
    output = Path(output_path)
    if output.exists():
        existing = pl.read_csv(output).select(_RESULT_COLUMNS)
        existing = existing.filter(~pl.col("dataset_id").is_in(new_results["dataset_id"].to_list()))
        return pl.concat([existing, new_results], how="vertical_relaxed").sort("dataset_id").select(_RESULT_COLUMNS)
    return new_results.sort("dataset_id").select(_RESULT_COLUMNS)


def run_experiment(
    *,
    dataset_ids: Sequence[str] = DEFAULT_DATASETS,
    cache_root: str | Path = _CACHE_ROOT,
    output_path: str | Path = _OUTPUT_PATH,
    batch_size: int = DEFAULT_BATCH_SIZE,
    max_windows_per_dataset: int | None = None,
    device: str | None = None,
    task_spec=None,
    pipeline: Any | None = None,
    mode: str = "univariate",
) -> pl.DataFrame:
    if mode not in MODE_CHOICES:
        raise ValueError(f"Unsupported mode {mode!r}. Expected one of {MODE_CHOICES}.")
    resolved_task_spec = task_spec or build_task_spec()
    resolved_device = device or select_device()
    resolved_pipeline = pipeline or load_pipeline(device=resolved_device)
    evaluator = evaluate_dataset if mode == "univariate" else evaluate_multivariate_dataset
    rows = [
        evaluator(
            dataset_id,
            pipeline=resolved_pipeline,
            cache_root=cache_root,
            task_spec=resolved_task_spec,
            batch_size=batch_size,
            max_windows_per_dataset=max_windows_per_dataset,
            device=resolved_device,
        )
        for dataset_id in dataset_ids
    ]
    new_results = pl.DataFrame(rows).select(_RESULT_COLUMNS)
    results = _merge_results(output_path, new_results) if mode == "append_multivariate" else new_results
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    results.write_csv(output)
    return results


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Chronos-2 power_only zero-shot experiments.")
    parser.add_argument(
        "--dataset",
        action="append",
        choices=sorted(DEFAULT_DATASETS),
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
        "--max-windows-per-dataset",
        type=int,
        default=None,
        help="Optional smoke-test limit on the number of candidate windows evaluated per dataset.",
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
        default="univariate",
        help="Run univariate, multivariate, or append multivariate rows into an existing CSV.",
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
        max_windows_per_dataset=args.max_windows_per_dataset,
        device=args.device,
        mode=args.mode,
    )
    print(results)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
