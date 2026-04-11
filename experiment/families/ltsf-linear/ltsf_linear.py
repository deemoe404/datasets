from __future__ import annotations

import argparse
import copy
from dataclasses import dataclass
from datetime import UTC, datetime
import json
import math
from pathlib import Path
import random
import sys
import time
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import polars as pl
try:
    from tqdm.auto import tqdm

    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False

    class tqdm:  # type: ignore[no-redef]
        def __init__(self, *args, **kwargs) -> None:
            del args
            self.total = kwargs.get("total")
            self.desc = kwargs.get("desc")
            self.disable = kwargs.get("disable", False)
            self.n = 0

        def update(self, value=1) -> None:
            self.n += int(value)

        def set_postfix_str(self, value: str, refresh: bool = True) -> None:
            del value, refresh

        def close(self) -> None:
            return None

try:
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, Dataset
except ImportError:  # pragma: no cover - exercised in the root env where torch is absent
    torch = None
    nn = None
    DataLoader = None
    Dataset = None


EXPERIMENT_DIR = Path(__file__).resolve().parent
EXPERIMENT_ROOT = EXPERIMENT_DIR.parents[1]
COMMON_DIR = EXPERIMENT_ROOT / "infra" / "common"
if str(COMMON_DIR) not in sys.path:
    sys.path.insert(0, str(COMMON_DIR))

from run_records import record_cli_run, resolve_family_feature_protocol_ids  # noqa: E402
from published_outputs import default_family_output_path  # noqa: E402
from covariate_packs import (  # noqa: E402
    DEFAULT_COVARIATE_STAGES,
    CovariatePackSpec,
    iter_covariate_packs,
    reference_pack_for,
)
from window_protocols import (  # noqa: E402
    DEFAULT_WINDOW_PROTOCOL,
    HORIZON_METRIC_SCOPE,
    NON_OVERLAP_EVAL_PROTOCOL,
    OVERALL_METRIC_SCOPE,
    ROLLING_EVAL_PROTOCOL,
    SPLIT_PROTOCOL,
    SplitBoundary,
    WindowDescriptorIndex,
    build_chrono_split_lookup as shared_build_chrono_split_lookup,
    build_split_boundaries as shared_build_split_boundaries,
    build_window_descriptor_index as shared_build_window_descriptor_index,
    resolve_window_protocol,
    split_window_index as shared_split_window_index,
    thin_non_overlap_window_index as shared_thin_non_overlap_window_index,
)


MODEL_ID = "LTSF-Linear"
FAMILY_ID = "ltsf_linear_local"
WINDOW_PROTOCOL = DEFAULT_WINDOW_PROTOCOL
TASK_ID = resolve_window_protocol(WINDOW_PROTOCOL).task_id
DEFAULT_DATASETS = ("kelmarsh", "penmanshiel", "hill_of_towie", "sdwpf_kddcup")
MODEL_VARIANTS = ("nlinear", "dlinear")
HISTORY_STEPS = 144
FORECAST_STEPS = 36
STRIDE_STEPS = 1
DEFAULT_BATCH_SIZE = 1024
DEFAULT_LEARNING_RATE = 1e-3
DEFAULT_MAX_EPOCHS = 50
DEFAULT_EARLY_STOPPING_PATIENCE = 5
DEFAULT_SEED = 42
DLINEAR_KERNEL_SIZE = 25
QUALITY_PROFILE = "default"
SERIES_LAYOUT = "turbine"
REFERENCE_STAGE = "reference"
REFERENCE_PACK = "power_only"
COVARIATE_POLICY = "past_only_train_zscore_fill0_mask"
REFERENCE_COVARIATE_POLICY = "none"
PROFILE_LOG_PREFIX = "[ltsf_linear] "

_REPO_ROOT = EXPERIMENT_ROOT.parent
_CACHE_ROOT = _REPO_ROOT / "cache"
_OUTPUT_PATH = default_family_output_path(repo_root=_REPO_ROOT, family_id=FAMILY_ID)
_TASK_DIR_NAME = TASK_ID
_TASK_WINDOW_COLUMNS = (
    "dataset",
    "turbine_id",
    "output_start_ts",
    "output_end_ts",
    "is_complete_input",
    "is_complete_output",
    "quality_flags",
)
_SERIES_BASE_COLUMNS = (
    "dataset",
    "turbine_id",
    "timestamp",
    "target_kw",
    "quality_flags",
)
_RESULT_COLUMNS = [
    "dataset_id",
    "model_id",
    "model_variant",
    "task_id",
    "window_protocol",
    "history_steps",
    "forecast_steps",
    "stride_steps",
    "split_protocol",
    "split_name",
    "eval_protocol",
    "metric_scope",
    "lead_step",
    "lead_minutes",
    "covariate_stage",
    "covariate_pack",
    "covariate_count",
    "covariate_policy",
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
    "train_window_count",
    "val_window_count",
    "test_window_count",
    "best_epoch",
    "epochs_ran",
    "best_val_rmse_pu",
    "seed",
    "batch_size",
    "learning_rate",
]
_DATASET_ORDER = {dataset_id: index for index, dataset_id in enumerate(DEFAULT_DATASETS)}
_STAGE_ORDER = {REFERENCE_STAGE: 0, **{stage: index + 1 for index, stage in enumerate(DEFAULT_COVARIATE_STAGES)}}
_MODEL_ORDER = {model_variant: index for index, model_variant in enumerate(MODEL_VARIANTS)}
_SPLIT_ORDER = {"val": 0, "test": 1}
_EVAL_PROTOCOL_ORDER = {ROLLING_EVAL_PROTOCOL: 0, NON_OVERLAP_EVAL_PROTOCOL: 1}
_METRIC_SCOPE_ORDER = {OVERALL_METRIC_SCOPE: 0, HORIZON_METRIC_SCOPE: 1}


@dataclass(frozen=True)
class TaskCachePaths:
    dataset_id: str
    dataset_root: Path
    task_dir: Path
    window_index_path: Path
    task_context_path: Path
    turbine_static_path: Path


@dataclass(frozen=True)
class DatasetMetadata:
    dataset_id: str
    turbine_ids: tuple[str, ...]
    rated_power_kw: float
    task_paths: TaskCachePaths


@dataclass(frozen=True)
class TurbineSeries:
    timestamps_us: np.ndarray
    target_pu: np.ndarray
    past_covariates: dict[str, np.ndarray]


@dataclass(frozen=True)
class PreparedDataset:
    dataset_id: str
    resolution_minutes: int
    rated_power_kw: float
    history_steps: int
    forecast_steps: int
    stride_steps: int
    covariate_stage: str
    covariate_pack: str
    covariate_columns: tuple[str, ...]
    covariate_count: int
    covariate_policy: str
    turbine_ids: tuple[str, ...]
    turbine_series: tuple[TurbineSeries, ...]
    train_windows: WindowDescriptorIndex
    val_rolling_windows: WindowDescriptorIndex
    val_non_overlap_windows: WindowDescriptorIndex
    test_rolling_windows: WindowDescriptorIndex
    test_non_overlap_windows: WindowDescriptorIndex
    covariate_means: np.ndarray
    covariate_stds: np.ndarray


@dataclass(frozen=True)
class TrainingOutcome:
    best_epoch: int
    epochs_ran: int
    best_val_rmse_pu: float
    device: str
    model: Any


@dataclass(frozen=True)
class EvaluationMetrics:
    window_count: int
    prediction_count: int
    mae_kw: float
    rmse_kw: float
    mae_pu: float
    rmse_pu: float
    horizon_window_count: np.ndarray
    horizon_prediction_count: np.ndarray
    horizon_mae_kw: np.ndarray
    horizon_rmse_kw: np.ndarray
    horizon_mae_pu: np.ndarray
    horizon_rmse_pu: np.ndarray


def _profile_log(dataset_id: str, phase: str, **fields: object) -> None:
    payload = {"dataset_id": dataset_id, "phase": phase, **fields}
    print(
        f"{PROFILE_LOG_PREFIX}{json.dumps(payload, ensure_ascii=False, default=str, sort_keys=True)}",
        file=sys.stderr,
        flush=True,
    )


def progress_is_enabled() -> bool:
    return HAS_TQDM and sys.stderr.isatty()


def _create_progress_bar(
    *,
    total: int | None,
    desc: str,
    leave: bool = False,
    enabled: bool | None = None,
):
    return tqdm(
        total=total,
        desc=desc,
        leave=leave,
        disable=not (progress_is_enabled() if enabled is None else enabled),
        dynamic_ncols=True,
    )


def _loader_batch_total(loader: object) -> int | None:
    try:
        return int(len(loader))  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


def _job_progress_label(dataset_id: str, covariate_pack: str, model_variant: str | None = None) -> str:
    parts = [dataset_id, covariate_pack]
    if model_variant is not None:
        parts.append(model_variant)
    return "/".join(parts)


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
        task_id=TASK_ID,
        granularity="turbine",
    )


def build_requested_packs(
    dataset_id: str,
    *,
    covariate_stages: Sequence[str] = DEFAULT_COVARIATE_STAGES,
    include_power_only_reference: bool = True,
) -> tuple[CovariatePackSpec, ...]:
    packs: list[CovariatePackSpec] = []
    if include_power_only_reference:
        packs.append(reference_pack_for(dataset_id))
    if covariate_stages:
        packs.extend(iter_covariate_packs((dataset_id,), tuple(covariate_stages)))
    if not packs:
        raise ValueError("At least one covariate pack must be selected.")
    return tuple(packs)


def require_torch() -> tuple[Any, Any, Any, Any]:
    if torch is None or nn is None or DataLoader is None or Dataset is None:
        raise ImportError(
            "PyTorch is unavailable in the current environment. "
            "Create experiment/families/ltsf-linear/.conda with ./create_env.sh."
        )
    return torch, nn, DataLoader, Dataset


def select_device(torch_module: Any | None = None) -> str:
    resolved_torch = torch_module or torch
    if resolved_torch is None:
        return "cpu"
    if bool(resolved_torch.cuda.is_available()):
        return "cuda"
    mps_backend = getattr(getattr(resolved_torch, "backends", None), "mps", None)
    if mps_backend is not None and bool(mps_backend.is_available()):
        return "mps"
    return "cpu"


def resolve_device(device: str | None = None, torch_module: Any | None = None) -> str:
    if device is None or device == "auto":
        return select_device(torch_module=torch_module)
    return device


def clip_target_values(values: Sequence[float | None], rated_power_kw: float) -> np.ndarray:
    clipped = np.asarray(values, dtype=np.float32).copy()
    valid = ~np.isnan(clipped)
    clipped[valid] = np.clip(clipped[valid], 0.0, rated_power_kw)
    return clipped


def _normalize_target_values(values: Sequence[float | None], rated_power_kw: float) -> np.ndarray:
    clipped = clip_target_values(values, rated_power_kw)
    normalized = clipped.copy()
    valid = ~np.isnan(normalized)
    normalized[valid] = normalized[valid] / rated_power_kw
    return normalized


def resolve_cache_paths(dataset_id: str, *, cache_root: str | Path = _CACHE_ROOT) -> TaskCachePaths:
    dataset_root = Path(cache_root) / dataset_id
    task_dir = dataset_root / "tasks" / QUALITY_PROFILE / SERIES_LAYOUT / _TASK_DIR_NAME
    return TaskCachePaths(
        dataset_id=dataset_id,
        dataset_root=dataset_root,
        task_dir=task_dir,
        window_index_path=task_dir / "window_index.parquet",
        task_context_path=task_dir / "task_context.json",
        turbine_static_path=dataset_root / "silver" / "meta" / "turbine_static.parquet",
    )


def ensure_task_cache(dataset_id: str, *, cache_root: str | Path = _CACHE_ROOT) -> TaskCachePaths:
    paths = resolve_cache_paths(dataset_id, cache_root=cache_root)
    required_paths = (
        paths.window_index_path,
        paths.task_context_path,
        paths.turbine_static_path,
    )
    if all(path.exists() for path in required_paths):
        return paths

    _ensure_repo_src_on_path()
    try:
        from wind_datasets import build_task_cache
    except ImportError as exc:  # pragma: no cover - defensive
        raise RuntimeError("Unable to import wind_datasets for task cache construction.") from exc

    try:
        build_task_cache(dataset_id, build_task_spec(), cache_root=cache_root)
    except Exception as exc:  # pragma: no cover - exercised only when cache is missing
        raise RuntimeError(
            f"Task cache for dataset {dataset_id!r} is missing and could not be rebuilt. "
            "Either prebuild the cache artifacts or configure wind_datasets.local.toml."
        ) from exc

    if not all(path.exists() for path in required_paths):
        raise RuntimeError(f"Dataset cache for {dataset_id!r} is incomplete after rebuild.")
    return paths


def resolve_turbine_series_path(
    dataset_id: str,
    *,
    cache_root: str | Path = _CACHE_ROOT,
) -> Path:
    cache_root_path = Path(cache_root)
    direct_path = (
        cache_root_path
        / dataset_id
        / "gold_base"
        / QUALITY_PROFILE
        / SERIES_LAYOUT
        / "series.parquet"
    )
    if direct_path.exists():
        return direct_path

    _ensure_repo_src_on_path()
    from wind_datasets import get_dataset_spec
    from wind_datasets.datasets import get_builder

    spec = get_dataset_spec(dataset_id)
    builder = get_builder(spec, cache_root_path)
    series_path = builder.cache_paths.gold_base_series_path_for(
        spec.default_quality_profile,
        layout=SERIES_LAYOUT,
    )
    if not series_path.exists():
        builder.build_gold_base(
            quality_profile=spec.default_quality_profile,
            layout=SERIES_LAYOUT,
        )
    return series_path


def load_dataset_metadata(dataset_id: str, *, cache_root: str | Path = _CACHE_ROOT) -> DatasetMetadata:
    paths = ensure_task_cache(dataset_id, cache_root=cache_root)
    task_context = json.loads(paths.task_context_path.read_text(encoding="utf-8"))
    task = task_context.get("task", {})
    if (
        int(task.get("history_steps", HISTORY_STEPS)) != HISTORY_STEPS
        or int(task.get("forecast_steps", FORECAST_STEPS)) != FORECAST_STEPS
        or int(task.get("stride_steps", STRIDE_STEPS)) != STRIDE_STEPS
    ):
        raise ValueError(
            f"Cached task context for dataset {dataset_id!r} does not match the expected "
            f"{HISTORY_STEPS}/{FORECAST_STEPS}/{STRIDE_STEPS} task."
        )

    turbine_static = pl.read_parquet(paths.turbine_static_path).select(["turbine_id", "rated_power_kw"])
    rated_powers = sorted(
        {
            float(value)
            for value in turbine_static["rated_power_kw"].drop_nulls().to_list()
        }
    )
    if len(rated_powers) != 1:
        raise ValueError(f"Dataset {dataset_id!r} must have a single rated_power_kw, found {rated_powers!r}.")
    return DatasetMetadata(
        dataset_id=dataset_id,
        turbine_ids=tuple(task_context["turbine_ids"]),
        rated_power_kw=rated_powers[0],
        task_paths=paths,
    )


def load_series_frame(
    dataset_id: str,
    *,
    pack: CovariatePackSpec,
    cache_root: str | Path = _CACHE_ROOT,
) -> tuple[str, tuple[str, ...], pl.DataFrame]:
    series_path = resolve_turbine_series_path(
        dataset_id,
        cache_root=cache_root,
    )
    available_columns = set(pl.read_parquet_schema(series_path))
    missing_base = [column for column in _SERIES_BASE_COLUMNS if column not in available_columns]
    if missing_base:
        raise ValueError(
            f"Series {series_path} for dataset {dataset_id!r} is missing required base columns {missing_base!r}."
        )
    covariate_columns = pack.selected_covariate_columns(available_columns)
    selected_columns = tuple(_SERIES_BASE_COLUMNS) + covariate_columns
    load_started = time.monotonic()
    frame = (
        pl.scan_parquet(series_path)
        .select(list(selected_columns))
        .sort(["turbine_id", "timestamp"])
        .collect()
    )
    _profile_log(
        dataset_id,
        "load_series",
        covariate_stage=pack.stage,
        covariate_pack=pack.pack_name,
        rows=frame.height,
        columns=len(frame.columns),
        covariates=len(covariate_columns),
        duration_seconds=round(time.monotonic() - load_started, 6),
    )
    return covariate_columns, frame


def load_strict_window_index(dataset_id: str, *, cache_root: str | Path = _CACHE_ROOT) -> pl.DataFrame:
    paths = ensure_task_cache(dataset_id, cache_root=cache_root)
    load_started = time.monotonic()
    frame = (
        pl.scan_parquet(paths.window_index_path)
        .select(list(_TASK_WINDOW_COLUMNS))
        .filter(
            pl.col("is_complete_input")
            & pl.col("is_complete_output")
            & (pl.col("quality_flags").fill_null("") == "")
        )
        .sort(["output_start_ts", "turbine_id"])
        .collect()
    )
    _profile_log(
        dataset_id,
        "load_window_index",
        strict_windows=frame.height,
        duration_seconds=round(time.monotonic() - load_started, 6),
    )
    if frame.is_empty():
        raise ValueError(f"Dataset {dataset_id!r} has no strict windows for {TASK_ID}.")
    return frame


def build_chrono_split_lookup(raw_timestamps: Sequence[datetime]) -> pl.DataFrame:
    return shared_build_chrono_split_lookup(raw_timestamps)


def build_split_boundaries(raw_timestamps: Sequence[datetime]) -> dict[str, SplitBoundary]:
    return shared_build_split_boundaries(raw_timestamps)


def split_window_index(
    window_index: pl.DataFrame,
    *,
    raw_timestamps: Sequence[datetime],
    resolution_minutes: int,
    max_windows_per_split: int | None = None,
) -> dict[str, pl.DataFrame]:
    return shared_split_window_index(
        window_index,
        raw_timestamps=raw_timestamps,
        resolution_minutes=resolution_minutes,
        history_steps=HISTORY_STEPS,
        max_windows_per_split=max_windows_per_split,
    )


def prepare_series_frame(series: pl.DataFrame, *, covariate_columns: Sequence[str]) -> pl.DataFrame:
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
    return series.with_columns(*covariate_expressions).select([*_SERIES_BASE_COLUMNS, *covariate_columns])


def build_turbine_series_map(
    series: pl.DataFrame,
    *,
    rated_power_kw: float,
    covariate_columns: Sequence[str],
) -> dict[str, TurbineSeries]:
    turbines: dict[str, TurbineSeries] = {}
    for turbine_frame in series.partition_by("turbine_id", maintain_order=True):
        turbine_id = turbine_frame["turbine_id"][0]
        turbines[turbine_id] = TurbineSeries(
            timestamps_us=turbine_frame["timestamp"].cast(pl.Int64).to_numpy(),
            target_pu=_normalize_target_values(turbine_frame["target_kw"].to_list(), rated_power_kw),
            past_covariates={
                column: turbine_frame[column].cast(pl.Float32).to_numpy()
                for column in covariate_columns
            },
        )
    return turbines


def resolve_resolution_minutes(series: pl.DataFrame) -> int:
    timestamps = (
        series.select("timestamp")
        .unique()
        .sort("timestamp")
        .head(2)["timestamp"]
        .cast(pl.Int64)
        .to_list()
    )
    if len(timestamps) < 2:
        raise ValueError("Series must include at least two timestamps to infer resolution.")
    step_us = int(timestamps[1]) - int(timestamps[0])
    if step_us <= 0 or step_us % (60 * 1_000_000) != 0:
        raise ValueError(f"Unsupported resolution step {step_us!r}us.")
    return step_us // (60 * 1_000_000)


def build_turbine_series_tuple(
    metadata: DatasetMetadata,
    turbine_series_map: Mapping[str, TurbineSeries],
) -> tuple[TurbineSeries, ...]:
    return tuple(turbine_series_map[turbine_id] for turbine_id in metadata.turbine_ids)


def build_window_descriptor_index(
    window_index: pl.DataFrame,
    *,
    turbine_ids: Sequence[str],
    turbine_series_map: Mapping[str, TurbineSeries],
) -> WindowDescriptorIndex:
    return shared_build_window_descriptor_index(
        window_index,
        turbine_ids=turbine_ids,
        timestamps_by_turbine={
            turbine_id: turbine_series_map[turbine_id].timestamps_us
            for turbine_id in turbine_ids
        },
    )


def thin_non_overlap_window_index(
    windows: WindowDescriptorIndex,
    *,
    turbine_ids: Sequence[str],
    forecast_steps: int,
) -> WindowDescriptorIndex:
    return shared_thin_non_overlap_window_index(
        windows,
        turbine_ids=turbine_ids,
        forecast_steps=forecast_steps,
    )


def _empty_raw_exogenous_batch(batch_size: int, history_steps: int) -> np.ndarray:
    return np.empty((batch_size, history_steps, 0), dtype=np.float32)


def build_raw_exogenous_batch(
    turbine_series: Sequence[TurbineSeries],
    windows: WindowDescriptorIndex,
    *,
    history_steps: int,
    covariate_columns: Sequence[str],
    start: int = 0,
    stop: int | None = None,
) -> np.ndarray:
    stop = len(windows) if stop is None else min(stop, len(windows))
    batch_size = max(0, stop - start)
    covariate_count = len(covariate_columns)
    if batch_size == 0:
        return _empty_raw_exogenous_batch(0, history_steps)
    if covariate_count == 0:
        return _empty_raw_exogenous_batch(batch_size, history_steps)
    batch = np.empty((batch_size, history_steps, covariate_count), dtype=np.float32)
    for batch_index, window_index_position in enumerate(range(start, stop)):
        turbine_index = int(windows.turbine_indices[window_index_position])
        target_index = int(windows.target_indices[window_index_position])
        series = turbine_series[turbine_index]
        for covariate_index, column in enumerate(covariate_columns):
            batch[batch_index, :, covariate_index] = series.past_covariates[column][
                target_index - history_steps : target_index
            ]
    return batch


def fit_covariate_statistics_from_windows(
    turbine_series: Sequence[TurbineSeries],
    windows: WindowDescriptorIndex,
    *,
    history_steps: int,
    covariate_columns: Sequence[str],
    chunk_size: int = 2048,
    progress_label: str | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    covariate_count = len(covariate_columns)
    if covariate_count == 0:
        return (
            np.empty((0,), dtype=np.float32),
            np.empty((0,), dtype=np.float32),
        )
    sums = np.zeros((covariate_count,), dtype=np.float64)
    squared_sums = np.zeros((covariate_count,), dtype=np.float64)
    counts = np.zeros((covariate_count,), dtype=np.int64)
    chunk_starts = range(0, len(windows), chunk_size)
    progress = _create_progress_bar(
        total=math.ceil(len(windows) / chunk_size) if len(windows) else 0,
        desc=f"{progress_label or 'covariates'} stats",
    )
    try:
        for start in chunk_starts:
            raw_batch = build_raw_exogenous_batch(
                turbine_series,
                windows,
                history_steps=history_steps,
                covariate_columns=covariate_columns,
                start=start,
                stop=start + chunk_size,
            )
            flat = raw_batch.reshape(-1, covariate_count).astype(np.float64, copy=False)
            valid = ~np.isnan(flat)
            filled = np.where(valid, flat, 0.0)
            counts += valid.sum(axis=0)
            sums += filled.sum(axis=0)
            squared_sums += np.square(filled).sum(axis=0)
            progress.update(1)
    finally:
        progress.close()
    means = np.divide(sums, counts, out=np.zeros_like(sums), where=counts > 0)
    variances = np.divide(squared_sums, counts, out=np.ones_like(sums), where=counts > 0) - np.square(means)
    variances = np.maximum(variances, 0.0)
    stds = np.sqrt(variances)
    stds[stds < 1e-6] = 1.0
    stds[counts == 0] = 1.0
    return means.astype(np.float32), stds.astype(np.float32)


def normalize_exogenous_window(
    raw_exogenous_inputs: np.ndarray,
    *,
    means: np.ndarray,
    stds: np.ndarray,
) -> np.ndarray:
    covariate_count = raw_exogenous_inputs.shape[1]
    if covariate_count == 0:
        return np.empty((raw_exogenous_inputs.shape[0], 0), dtype=np.float32)
    reshaped_means = means.reshape(1, covariate_count)
    reshaped_stds = stds.reshape(1, covariate_count)
    valid = ~np.isnan(raw_exogenous_inputs)
    centered = np.where(valid, raw_exogenous_inputs, reshaped_means) - reshaped_means
    normalized = (centered / reshaped_stds).astype(np.float32, copy=False)
    normalized[~valid] = 0.0
    missing_mask = (~valid).astype(np.float32)
    return np.concatenate([normalized, missing_mask], axis=1)


if Dataset is not None:

    class WindowTensorDataset(Dataset):
        def __init__(
            self,
            turbine_series: Sequence[TurbineSeries],
            windows: WindowDescriptorIndex,
            *,
            history_steps: int,
            forecast_steps: int,
            covariate_columns: Sequence[str],
            covariate_means: np.ndarray,
            covariate_stds: np.ndarray,
        ) -> None:
            self.turbine_series = tuple(turbine_series)
            self.windows = windows
            self.history_steps = history_steps
            self.forecast_steps = forecast_steps
            self.covariate_columns = tuple(covariate_columns)
            self.covariate_means = covariate_means
            self.covariate_stds = covariate_stds

        def __len__(self) -> int:
            return len(self.windows)

        def __getitem__(self, index: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
            turbine_index = int(self.windows.turbine_indices[index])
            target_index = int(self.windows.target_indices[index])
            series = self.turbine_series[turbine_index]
            target_inputs = series.target_pu[target_index - self.history_steps : target_index].astype(np.float32, copy=True)
            targets = series.target_pu[target_index : target_index + self.forecast_steps].astype(np.float32, copy=True)
            if self.covariate_columns:
                raw_exogenous = np.column_stack(
                    [
                        series.past_covariates[column][target_index - self.history_steps : target_index]
                        for column in self.covariate_columns
                    ]
                ).astype(np.float32, copy=False)
                exogenous_inputs = normalize_exogenous_window(
                    raw_exogenous,
                    means=self.covariate_means,
                    stds=self.covariate_stds,
                )
            else:
                exogenous_inputs = np.empty((self.history_steps, 0), dtype=np.float32)
            return target_inputs, exogenous_inputs, targets

else:

    class WindowTensorDataset:  # pragma: no cover - exercised only when torch is missing
        def __init__(self, *_args, **_kwargs) -> None:
            require_torch()


def fit_covariate_statistics(raw_exogenous_inputs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if raw_exogenous_inputs.ndim != 3:
        raise ValueError("raw_exogenous_inputs must have shape [batch, history, covariates].")
    covariate_count = raw_exogenous_inputs.shape[2]
    if covariate_count == 0:
        return (
            np.empty((0,), dtype=np.float32),
            np.empty((0,), dtype=np.float32),
        )
    flat = raw_exogenous_inputs.reshape(-1, covariate_count).astype(np.float64, copy=False)
    valid = ~np.isnan(flat)
    counts = valid.sum(axis=0)
    sums = np.where(valid, flat, 0.0).sum(axis=0)
    means = np.divide(sums, counts, out=np.zeros_like(sums), where=counts > 0)
    centered = np.where(valid, flat - means, 0.0)
    variances = np.divide(
        np.square(centered).sum(axis=0),
        counts,
        out=np.ones_like(sums),
        where=counts > 0,
    )
    stds = np.sqrt(variances)
    stds[stds < 1e-6] = 1.0
    return means.astype(np.float32), stds.astype(np.float32)


def transform_exogenous_inputs(
    raw_exogenous_inputs: np.ndarray,
    *,
    means: np.ndarray,
    stds: np.ndarray,
) -> np.ndarray:
    if raw_exogenous_inputs.ndim != 3:
        raise ValueError("raw_exogenous_inputs must have shape [batch, history, covariates].")
    covariate_count = raw_exogenous_inputs.shape[2]
    if covariate_count == 0:
        return np.empty((raw_exogenous_inputs.shape[0], raw_exogenous_inputs.shape[1], 0), dtype=np.float32)
    reshaped_means = means.reshape(1, 1, covariate_count)
    reshaped_stds = stds.reshape(1, 1, covariate_count)
    valid = ~np.isnan(raw_exogenous_inputs)
    centered = np.where(valid, raw_exogenous_inputs, reshaped_means) - reshaped_means
    normalized = (centered / reshaped_stds).astype(np.float32, copy=False)
    normalized[~valid] = 0.0
    missing_mask = (~valid).astype(np.float32)
    return np.concatenate([normalized, missing_mask], axis=2)


def prepare_dataset(
    dataset_id: str,
    *,
    pack: CovariatePackSpec,
    cache_root: str | Path = _CACHE_ROOT,
    max_windows_per_split: int | None = None,
) -> PreparedDataset:
    metadata = load_dataset_metadata(dataset_id, cache_root=cache_root)
    strict_window_index = load_strict_window_index(dataset_id, cache_root=cache_root)
    covariate_columns, series = load_series_frame(dataset_id, pack=pack, cache_root=cache_root)
    prepared_series = prepare_series_frame(series, covariate_columns=covariate_columns)
    resolution_minutes = resolve_resolution_minutes(prepared_series)
    split_frames = split_window_index(
        strict_window_index,
        raw_timestamps=prepared_series["timestamp"].to_list(),
        resolution_minutes=resolution_minutes,
        max_windows_per_split=max_windows_per_split,
    )
    turbine_series_map = build_turbine_series_map(
        prepared_series,
        rated_power_kw=metadata.rated_power_kw,
        covariate_columns=covariate_columns,
    )
    turbine_series = build_turbine_series_tuple(metadata, turbine_series_map)
    train_windows = build_window_descriptor_index(
        split_frames["train"],
        turbine_ids=metadata.turbine_ids,
        turbine_series_map=turbine_series_map,
    )
    val_rolling_windows = build_window_descriptor_index(
        split_frames["val"],
        turbine_ids=metadata.turbine_ids,
        turbine_series_map=turbine_series_map,
    )
    test_rolling_windows = build_window_descriptor_index(
        split_frames["test"],
        turbine_ids=metadata.turbine_ids,
        turbine_series_map=turbine_series_map,
    )
    val_non_overlap_windows = thin_non_overlap_window_index(
        val_rolling_windows,
        turbine_ids=metadata.turbine_ids,
        forecast_steps=FORECAST_STEPS,
    )
    test_non_overlap_windows = thin_non_overlap_window_index(
        test_rolling_windows,
        turbine_ids=metadata.turbine_ids,
        forecast_steps=FORECAST_STEPS,
    )
    means, stds = fit_covariate_statistics_from_windows(
        turbine_series,
        train_windows,
        history_steps=HISTORY_STEPS,
        covariate_columns=covariate_columns,
        progress_label=_job_progress_label(dataset_id, pack.pack_name),
    )
    covariate_policy = COVARIATE_POLICY if covariate_columns else REFERENCE_COVARIATE_POLICY
    _profile_log(
        dataset_id,
        "prepare_dataset_complete",
        covariate_stage=pack.stage,
        covariate_pack=pack.pack_name,
        covariate_count=len(covariate_columns),
        train_windows=len(train_windows),
        val_rolling_windows=len(val_rolling_windows),
        val_non_overlap_windows=len(val_non_overlap_windows),
        test_rolling_windows=len(test_rolling_windows),
        test_non_overlap_windows=len(test_non_overlap_windows),
        rated_power_kw=metadata.rated_power_kw,
        resolution_minutes=resolution_minutes,
    )
    return PreparedDataset(
        dataset_id=dataset_id,
        resolution_minutes=resolution_minutes,
        rated_power_kw=metadata.rated_power_kw,
        history_steps=HISTORY_STEPS,
        forecast_steps=FORECAST_STEPS,
        stride_steps=STRIDE_STEPS,
        covariate_stage=pack.stage,
        covariate_pack=pack.pack_name,
        covariate_columns=tuple(covariate_columns),
        covariate_count=len(covariate_columns),
        covariate_policy=covariate_policy,
        turbine_ids=metadata.turbine_ids,
        turbine_series=turbine_series,
        train_windows=train_windows,
        val_rolling_windows=val_rolling_windows,
        val_non_overlap_windows=val_non_overlap_windows,
        test_rolling_windows=test_rolling_windows,
        test_non_overlap_windows=test_non_overlap_windows,
        covariate_means=means,
        covariate_stds=stds,
    )


def _set_random_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    if torch is None:
        return
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    cudnn_backend = getattr(torch.backends, "cudnn", None)
    if cudnn_backend is not None:
        cudnn_backend.deterministic = True
        cudnn_backend.benchmark = False


def _initialize_linear(layer: Any) -> None:
    if layer is None:  # pragma: no cover - defensive
        return
    with torch.no_grad():
        layer.weight.fill_(1.0 / layer.in_features)
        if layer.bias is not None:
            layer.bias.zero_()


def _initialize_zero_linear(layer: Any) -> None:
    if layer is None:  # pragma: no cover - defensive
        return
    with torch.no_grad():
        layer.weight.zero_()
        if layer.bias is not None:
            layer.bias.zero_()


if nn is not None:

    class MovingAverage(nn.Module):
        def __init__(self, kernel_size: int) -> None:
            super().__init__()
            self.kernel_size = kernel_size
            self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=1)

        def forward(self, x):
            padding = (self.kernel_size - 1) // 2
            front = x[:, :1].repeat(1, padding)
            back = x[:, -1:].repeat(1, padding)
            padded = torch.cat([front, x, back], dim=1)
            return self.avg(padded.unsqueeze(1)).squeeze(1)


    class SeriesDecomposition(nn.Module):
        def __init__(self, kernel_size: int) -> None:
            super().__init__()
            self.moving_average = MovingAverage(kernel_size)

        def forward(self, x):
            trend = self.moving_average(x)
            seasonal = x - trend
            return seasonal, trend


    class NLinear(nn.Module):
        def __init__(self, history_steps: int, forecast_steps: int) -> None:
            super().__init__()
            self.linear = nn.Linear(history_steps, forecast_steps)
            _initialize_linear(self.linear)

        def forward(self, x):
            last = x[:, -1:].detach()
            normalized = x - last
            return self.linear(normalized) + last


    class DLinear(nn.Module):
        def __init__(self, history_steps: int, forecast_steps: int, kernel_size: int = DLINEAR_KERNEL_SIZE) -> None:
            super().__init__()
            self.decomposition = SeriesDecomposition(kernel_size)
            self.linear_seasonal = nn.Linear(history_steps, forecast_steps)
            self.linear_trend = nn.Linear(history_steps, forecast_steps)
            _initialize_linear(self.linear_seasonal)
            _initialize_linear(self.linear_trend)

        def forward(self, x):
            seasonal, trend = self.decomposition(x)
            return self.linear_seasonal(seasonal) + self.linear_trend(trend)


    class ExogenousResidualHead(nn.Module):
        def __init__(self, history_steps: int, exogenous_channels: int, forecast_steps: int) -> None:
            super().__init__()
            self.exogenous_channels = exogenous_channels
            if exogenous_channels > 0:
                self.linear = nn.Linear(history_steps * exogenous_channels, forecast_steps)
                _initialize_zero_linear(self.linear)
            else:
                self.linear = None

        def forward(self, exogenous_inputs, *, target_like):
            if self.linear is None or exogenous_inputs.shape[-1] == 0:
                return torch.zeros_like(target_like)
            flattened = exogenous_inputs.reshape(exogenous_inputs.shape[0], -1)
            return self.linear(flattened)


    class NLinearX(nn.Module):
        def __init__(self, history_steps: int, forecast_steps: int, exogenous_channels: int) -> None:
            super().__init__()
            self.target_backbone = NLinear(history_steps, forecast_steps)
            self.exogenous_head = ExogenousResidualHead(history_steps, exogenous_channels, forecast_steps)

        def forward(self, target_inputs, exogenous_inputs):
            base = self.target_backbone(target_inputs)
            return base + self.exogenous_head(exogenous_inputs, target_like=base)


    class DLinearX(nn.Module):
        def __init__(self, history_steps: int, forecast_steps: int, exogenous_channels: int) -> None:
            super().__init__()
            self.target_backbone = DLinear(history_steps, forecast_steps, kernel_size=DLINEAR_KERNEL_SIZE)
            self.exogenous_head = ExogenousResidualHead(history_steps, exogenous_channels, forecast_steps)

        def forward(self, target_inputs, exogenous_inputs):
            base = self.target_backbone(target_inputs)
            return base + self.exogenous_head(exogenous_inputs, target_like=base)

else:

    class MovingAverage:  # pragma: no cover - exercised only when torch is missing
        def __init__(self, *_args, **_kwargs) -> None:
            require_torch()


    class SeriesDecomposition:  # pragma: no cover - exercised only when torch is missing
        def __init__(self, *_args, **_kwargs) -> None:
            require_torch()


    class NLinear:  # pragma: no cover - exercised only when torch is missing
        def __init__(self, *_args, **_kwargs) -> None:
            require_torch()


    class DLinear:  # pragma: no cover - exercised only when torch is missing
        def __init__(self, *_args, **_kwargs) -> None:
            require_torch()


    class ExogenousResidualHead:  # pragma: no cover - exercised only when torch is missing
        def __init__(self, *_args, **_kwargs) -> None:
            require_torch()


    class NLinearX:  # pragma: no cover - exercised only when torch is missing
        def __init__(self, *_args, **_kwargs) -> None:
            require_torch()


    class DLinearX:  # pragma: no cover - exercised only when torch is missing
        def __init__(self, *_args, **_kwargs) -> None:
            require_torch()


def build_model(
    model_variant: str,
    *,
    history_steps: int,
    forecast_steps: int,
    exogenous_channels: int,
):
    require_torch()
    if model_variant == "nlinear":
        return NLinearX(history_steps, forecast_steps, exogenous_channels)
    if model_variant == "dlinear":
        return DLinearX(history_steps, forecast_steps, exogenous_channels)
    raise ValueError(f"Unsupported model_variant {model_variant!r}.")


def _build_dataloader(
    prepared_dataset: PreparedDataset,
    *,
    windows: WindowDescriptorIndex,
    batch_size: int,
    shuffle: bool,
    seed: int,
):
    resolved_torch, _, resolved_loader, _ = require_torch()
    dataset = WindowTensorDataset(
        prepared_dataset.turbine_series,
        windows,
        history_steps=prepared_dataset.history_steps,
        forecast_steps=prepared_dataset.forecast_steps,
        covariate_columns=prepared_dataset.covariate_columns,
        covariate_means=prepared_dataset.covariate_means,
        covariate_stds=prepared_dataset.covariate_stds,
    )
    generator = resolved_torch.Generator()
    generator.manual_seed(seed)
    return resolved_loader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        generator=generator if shuffle else None,
    )


def evaluate_model(
    model,
    loader,
    *,
    device: str,
    rated_power_kw: float,
    forecast_steps: int,
    progress_label: str | None = None,
) -> EvaluationMetrics:
    resolved_torch, _, _, _ = require_torch()
    window_count = 0
    prediction_count = 0
    abs_error_sum_kw = 0.0
    squared_error_sum_kw = 0.0
    abs_error_sum_pu = 0.0
    squared_error_sum_pu = 0.0
    horizon_window_count = np.zeros((forecast_steps,), dtype=np.int64)
    horizon_prediction_count = np.zeros((forecast_steps,), dtype=np.int64)
    horizon_abs_error_sum_kw = np.zeros((forecast_steps,), dtype=np.float64)
    horizon_squared_error_sum_kw = np.zeros((forecast_steps,), dtype=np.float64)
    horizon_abs_error_sum_pu = np.zeros((forecast_steps,), dtype=np.float64)
    horizon_squared_error_sum_pu = np.zeros((forecast_steps,), dtype=np.float64)
    model.eval()
    progress = _create_progress_bar(
        total=_loader_batch_total(loader),
        desc=progress_label or "evaluate",
    )
    try:
        with resolved_torch.no_grad():
            for batch_target_inputs, batch_exogenous_inputs, batch_targets in loader:
                batch_target_inputs = batch_target_inputs.to(device=device, dtype=resolved_torch.float32)
                batch_exogenous_inputs = batch_exogenous_inputs.to(device=device, dtype=resolved_torch.float32)
                batch_targets = batch_targets.to(device=device, dtype=resolved_torch.float32)
                predictions = model(batch_target_inputs, batch_exogenous_inputs)
                errors_pu = predictions - batch_targets
                errors_kw = errors_pu * rated_power_kw
                batch_window_count = int(batch_target_inputs.shape[0])
                window_count += batch_window_count
                prediction_count += int(batch_targets.numel())

                abs_errors_kw = resolved_torch.abs(errors_kw)
                squared_errors_kw = resolved_torch.square(errors_kw)
                abs_errors_pu = resolved_torch.abs(errors_pu)
                squared_errors_pu = resolved_torch.square(errors_pu)

                abs_error_sum_kw += float(abs_errors_kw.sum().item())
                squared_error_sum_kw += float(squared_errors_kw.sum().item())
                abs_error_sum_pu += float(abs_errors_pu.sum().item())
                squared_error_sum_pu += float(squared_errors_pu.sum().item())

                horizon_window_count += batch_window_count
                horizon_prediction_count += batch_window_count
                horizon_abs_error_sum_kw += abs_errors_kw.sum(dim=0).detach().cpu().numpy().astype(np.float64, copy=False)
                horizon_squared_error_sum_kw += (
                    squared_errors_kw.sum(dim=0).detach().cpu().numpy().astype(np.float64, copy=False)
                )
                horizon_abs_error_sum_pu += abs_errors_pu.sum(dim=0).detach().cpu().numpy().astype(np.float64, copy=False)
                horizon_squared_error_sum_pu += (
                    squared_errors_pu.sum(dim=0).detach().cpu().numpy().astype(np.float64, copy=False)
                )
                progress.update(1)
                progress.set_postfix_str(f"windows={window_count}")
    finally:
        progress.close()
    return EvaluationMetrics(
        window_count=window_count,
        prediction_count=prediction_count,
        mae_kw=_safe_divide(abs_error_sum_kw, prediction_count),
        rmse_kw=_safe_rmse(squared_error_sum_kw, prediction_count),
        mae_pu=_safe_divide(abs_error_sum_pu, prediction_count),
        rmse_pu=_safe_rmse(squared_error_sum_pu, prediction_count),
        horizon_window_count=horizon_window_count,
        horizon_prediction_count=horizon_prediction_count,
        horizon_mae_kw=np.asarray(
            [_safe_divide(horizon_abs_error_sum_kw[index], int(horizon_prediction_count[index])) for index in range(forecast_steps)],
            dtype=np.float64,
        ),
        horizon_rmse_kw=np.asarray(
            [_safe_rmse(horizon_squared_error_sum_kw[index], int(horizon_prediction_count[index])) for index in range(forecast_steps)],
            dtype=np.float64,
        ),
        horizon_mae_pu=np.asarray(
            [_safe_divide(horizon_abs_error_sum_pu[index], int(horizon_prediction_count[index])) for index in range(forecast_steps)],
            dtype=np.float64,
        ),
        horizon_rmse_pu=np.asarray(
            [_safe_rmse(horizon_squared_error_sum_pu[index], int(horizon_prediction_count[index])) for index in range(forecast_steps)],
            dtype=np.float64,
        ),
    )


def train_model(
    model_variant: str,
    prepared_dataset: PreparedDataset,
    *,
    device: str,
    seed: int = DEFAULT_SEED,
    batch_size: int = DEFAULT_BATCH_SIZE,
    learning_rate: float = DEFAULT_LEARNING_RATE,
    max_epochs: int = DEFAULT_MAX_EPOCHS,
    early_stopping_patience: int = DEFAULT_EARLY_STOPPING_PATIENCE,
    progress_label: str | None = None,
) -> TrainingOutcome:
    resolved_torch, _, _, _ = require_torch()
    _set_random_seed(seed)
    resolved_device = resolve_device(device)
    model = build_model(
        model_variant,
        history_steps=prepared_dataset.history_steps,
        forecast_steps=prepared_dataset.forecast_steps,
        exogenous_channels=prepared_dataset.covariate_count * 2,
    ).to(device=resolved_device)
    optimizer = resolved_torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = resolved_torch.nn.MSELoss()
    train_loader = _build_dataloader(
        prepared_dataset,
        windows=prepared_dataset.train_windows,
        batch_size=batch_size,
        shuffle=True,
        seed=seed,
    )
    val_loader = _build_dataloader(
        prepared_dataset,
        windows=prepared_dataset.val_rolling_windows,
        batch_size=batch_size,
        shuffle=False,
        seed=seed,
    )

    best_state: dict[str, Any] | None = None
    best_epoch = 0
    best_val_rmse_pu = float("inf")
    epochs_without_improvement = 0
    epochs_ran = 0
    epoch_progress = _create_progress_bar(
        total=max_epochs,
        desc=f"{progress_label or prepared_dataset.dataset_id} epochs",
        leave=True,
    )
    try:
        for epoch_index in range(1, max_epochs + 1):
            model.train()
            batch_progress = _create_progress_bar(
                total=_loader_batch_total(train_loader),
                desc=f"{progress_label or prepared_dataset.dataset_id} train e{epoch_index}",
            )
            last_loss_value: float | None = None
            try:
                for batch_target_inputs, batch_exogenous_inputs, batch_targets in train_loader:
                    batch_target_inputs = batch_target_inputs.to(device=resolved_device, dtype=resolved_torch.float32)
                    batch_exogenous_inputs = batch_exogenous_inputs.to(device=resolved_device, dtype=resolved_torch.float32)
                    batch_targets = batch_targets.to(device=resolved_device, dtype=resolved_torch.float32)
                    optimizer.zero_grad(set_to_none=True)
                    predictions = model(batch_target_inputs, batch_exogenous_inputs)
                    loss = criterion(predictions, batch_targets)
                    loss.backward()
                    optimizer.step()
                    if hasattr(loss, "item"):
                        last_loss_value = float(loss.item())
                        batch_progress.set_postfix_str(f"loss={last_loss_value:.4f}")
                    batch_progress.update(1)
            finally:
                batch_progress.close()

            epochs_ran = epoch_index
            val_metrics = evaluate_model(
                model,
                val_loader,
                device=resolved_device,
                rated_power_kw=prepared_dataset.rated_power_kw,
                forecast_steps=prepared_dataset.forecast_steps,
                progress_label=f"{progress_label or prepared_dataset.dataset_id} val e{epoch_index}",
            )
            val_rmse_pu = float(val_metrics.rmse_pu)
            if val_rmse_pu < best_val_rmse_pu - 1e-12:
                best_val_rmse_pu = val_rmse_pu
                best_epoch = epoch_index
                best_state = copy.deepcopy(model.state_dict())
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
            epoch_progress.update(1)
            postfix_parts = [f"val_rmse={val_rmse_pu:.4f}", f"best={best_val_rmse_pu:.4f}"]
            if last_loss_value is not None:
                postfix_parts.insert(0, f"loss={last_loss_value:.4f}")
            epoch_progress.set_postfix_str(" ".join(postfix_parts))
            if epochs_without_improvement >= early_stopping_patience:
                break
    finally:
        epoch_progress.close()

    if best_state is None:  # pragma: no cover - defensive
        raise RuntimeError("Training completed without a best checkpoint.")

    model.load_state_dict(best_state)
    return TrainingOutcome(
        best_epoch=best_epoch,
        epochs_ran=epochs_ran,
        best_val_rmse_pu=best_val_rmse_pu,
        device=resolved_device,
        model=model,
    )


def _safe_divide(numerator: float, denominator: int) -> float:
    if denominator <= 0:
        return float("nan")
    return numerator / denominator


def _safe_rmse(squared_error_sum: float, denominator: int) -> float:
    if denominator <= 0:
        return float("nan")
    return math.sqrt(squared_error_sum / denominator)


def _timestamp_us_to_string(value: int | None) -> str | None:
    if value is None:
        return None
    return (
        datetime.fromtimestamp(value / 1_000_000, tz=UTC)
        .replace(tzinfo=None)
        .strftime("%Y-%m-%d %H:%M:%S")
    )


def _window_bounds(windows: WindowDescriptorIndex) -> tuple[str | None, str | None]:
    if len(windows) == 0:
        return None, None
    return (
        _timestamp_us_to_string(int(windows.output_start_us.min())),
        _timestamp_us_to_string(int(windows.output_end_us.max())),
    )


def iter_evaluation_specs(
    prepared_dataset: PreparedDataset,
) -> tuple[tuple[str, str, WindowDescriptorIndex], ...]:
    return (
        ("val", ROLLING_EVAL_PROTOCOL, prepared_dataset.val_rolling_windows),
        ("val", NON_OVERLAP_EVAL_PROTOCOL, prepared_dataset.val_non_overlap_windows),
        ("test", ROLLING_EVAL_PROTOCOL, prepared_dataset.test_rolling_windows),
        ("test", NON_OVERLAP_EVAL_PROTOCOL, prepared_dataset.test_non_overlap_windows),
    )


def build_result_rows(
    prepared_dataset: PreparedDataset,
    *,
    model_variant: str,
    training_outcome: TrainingOutcome,
    runtime_seconds: float,
    seed: int,
    batch_size: int,
    learning_rate: float,
    evaluation_results: Sequence[tuple[str, str, WindowDescriptorIndex, EvaluationMetrics]],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    base_row = {
        "dataset_id": prepared_dataset.dataset_id,
        "model_id": MODEL_ID,
        "model_variant": model_variant,
        "task_id": TASK_ID,
        "window_protocol": WINDOW_PROTOCOL,
        "history_steps": prepared_dataset.history_steps,
        "forecast_steps": prepared_dataset.forecast_steps,
        "stride_steps": prepared_dataset.stride_steps,
        "split_protocol": SPLIT_PROTOCOL,
        "covariate_stage": prepared_dataset.covariate_stage,
        "covariate_pack": prepared_dataset.covariate_pack,
        "covariate_count": prepared_dataset.covariate_count,
        "covariate_policy": prepared_dataset.covariate_policy,
        "device": training_outcome.device,
        "runtime_seconds": round(runtime_seconds, 6),
        "train_window_count": len(prepared_dataset.train_windows),
        "val_window_count": len(prepared_dataset.val_rolling_windows),
        "test_window_count": len(prepared_dataset.test_rolling_windows),
        "best_epoch": training_outcome.best_epoch,
        "epochs_ran": training_outcome.epochs_ran,
        "best_val_rmse_pu": training_outcome.best_val_rmse_pu,
        "seed": seed,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
    }
    for split_name, eval_protocol, windows, metrics in evaluation_results:
        start_timestamp, end_timestamp = _window_bounds(windows)
        rows.append(
            {
                **base_row,
                "split_name": split_name,
                "eval_protocol": eval_protocol,
                "metric_scope": OVERALL_METRIC_SCOPE,
                "lead_step": None,
                "lead_minutes": None,
                "window_count": int(metrics.window_count),
                "prediction_count": int(metrics.prediction_count),
                "start_timestamp": start_timestamp,
                "end_timestamp": end_timestamp,
                "mae_kw": float(metrics.mae_kw),
                "rmse_kw": float(metrics.rmse_kw),
                "mae_pu": float(metrics.mae_pu),
                "rmse_pu": float(metrics.rmse_pu),
            }
        )
        for lead_index in range(prepared_dataset.forecast_steps):
            lead_step = lead_index + 1
            rows.append(
                {
                    **base_row,
                    "split_name": split_name,
                    "eval_protocol": eval_protocol,
                    "metric_scope": HORIZON_METRIC_SCOPE,
                    "lead_step": lead_step,
                    "lead_minutes": lead_step * prepared_dataset.resolution_minutes,
                    "window_count": int(metrics.horizon_window_count[lead_index]),
                    "prediction_count": int(metrics.horizon_prediction_count[lead_index]),
                    "start_timestamp": start_timestamp,
                    "end_timestamp": end_timestamp,
                    "mae_kw": float(metrics.horizon_mae_kw[lead_index]),
                    "rmse_kw": float(metrics.horizon_rmse_kw[lead_index]),
                    "mae_pu": float(metrics.horizon_mae_pu[lead_index]),
                    "rmse_pu": float(metrics.horizon_rmse_pu[lead_index]),
                }
            )
    return rows


def execute_training_job(
    prepared_dataset: PreparedDataset,
    *,
    model_variant: str,
    device: str | None = None,
    seed: int = DEFAULT_SEED,
    batch_size: int = DEFAULT_BATCH_SIZE,
    learning_rate: float = DEFAULT_LEARNING_RATE,
    max_epochs: int = DEFAULT_MAX_EPOCHS,
    early_stopping_patience: int = DEFAULT_EARLY_STOPPING_PATIENCE,
) -> list[dict[str, object]]:
    dataset_start = time.monotonic()
    progress_label = _job_progress_label(
        prepared_dataset.dataset_id,
        prepared_dataset.covariate_pack,
        model_variant,
    )
    training_outcome = train_model(
        model_variant,
        prepared_dataset,
        device=resolve_device(device),
        seed=seed,
        batch_size=batch_size,
        learning_rate=learning_rate,
        max_epochs=max_epochs,
        early_stopping_patience=early_stopping_patience,
        progress_label=progress_label,
    )
    evaluation_results: list[tuple[str, str, WindowDescriptorIndex, EvaluationMetrics]] = []
    eval_specs = iter_evaluation_specs(prepared_dataset)
    eval_progress = _create_progress_bar(total=len(eval_specs), desc=f"{progress_label} eval")
    try:
        for split_name, eval_protocol, windows in eval_specs:
            loader = _build_dataloader(
                prepared_dataset,
                windows=windows,
                batch_size=batch_size,
                shuffle=False,
                seed=seed,
            )
            metrics = evaluate_model(
                training_outcome.model,
                loader,
                device=training_outcome.device,
                rated_power_kw=prepared_dataset.rated_power_kw,
                forecast_steps=prepared_dataset.forecast_steps,
                progress_label=f"{progress_label} {split_name}/{eval_protocol}",
            )
            evaluation_results.append((split_name, eval_protocol, windows, metrics))
            eval_progress.update(1)
            eval_progress.set_postfix_str(f"{split_name}/{eval_protocol}")
    finally:
        eval_progress.close()
    runtime_seconds = time.monotonic() - dataset_start
    test_rolling_metrics = next(
        metrics
        for split_name, eval_protocol, _windows, metrics in evaluation_results
        if split_name == "test" and eval_protocol == ROLLING_EVAL_PROTOCOL
    )
    _profile_log(
        prepared_dataset.dataset_id,
        "training_complete",
        covariate_stage=prepared_dataset.covariate_stage,
        covariate_pack=prepared_dataset.covariate_pack,
        model_variant=model_variant,
        best_epoch=training_outcome.best_epoch,
        epochs_ran=training_outcome.epochs_ran,
        best_val_rmse_pu=training_outcome.best_val_rmse_pu,
        test_rolling_rmse_pu=test_rolling_metrics.rmse_pu,
        runtime_seconds=round(runtime_seconds, 6),
    )
    return build_result_rows(
        prepared_dataset,
        model_variant=model_variant,
        training_outcome=training_outcome,
        runtime_seconds=runtime_seconds,
        seed=seed,
        batch_size=batch_size,
        learning_rate=learning_rate,
        evaluation_results=evaluation_results,
    )


def sort_result_frame(frame: pl.DataFrame) -> pl.DataFrame:
    return (
        frame.with_columns(
            pl.col("dataset_id")
            .replace_strict(_DATASET_ORDER, default=len(_DATASET_ORDER))
            .alias("__dataset_order"),
            pl.col("covariate_stage")
            .replace_strict(_STAGE_ORDER, default=len(_STAGE_ORDER))
            .alias("__stage_order"),
            pl.col("model_variant")
            .replace_strict(_MODEL_ORDER, default=len(_MODEL_ORDER))
            .alias("__model_order"),
            pl.col("split_name")
            .replace_strict(_SPLIT_ORDER, default=len(_SPLIT_ORDER))
            .alias("__split_order"),
            pl.col("eval_protocol")
            .replace_strict(_EVAL_PROTOCOL_ORDER, default=len(_EVAL_PROTOCOL_ORDER))
            .alias("__eval_protocol_order"),
            pl.col("metric_scope")
            .replace_strict(_METRIC_SCOPE_ORDER, default=len(_METRIC_SCOPE_ORDER))
            .alias("__metric_scope_order"),
            pl.col("lead_step").fill_null(0).alias("__lead_order"),
        )
        .sort(
            [
                "__dataset_order",
                "__stage_order",
                "covariate_pack",
                "__model_order",
                "__split_order",
                "__eval_protocol_order",
                "__metric_scope_order",
                "__lead_order",
            ]
        )
        .drop(
            [
                "__dataset_order",
                "__stage_order",
                "__model_order",
                "__split_order",
                "__eval_protocol_order",
                "__metric_scope_order",
                "__lead_order",
            ]
        )
    )


def run_experiment(
    *,
    dataset_ids: Sequence[str] = DEFAULT_DATASETS,
    model_variants: Sequence[str] = MODEL_VARIANTS,
    covariate_stages: Sequence[str] = DEFAULT_COVARIATE_STAGES,
    include_power_only_reference: bool = True,
    cache_root: str | Path = _CACHE_ROOT,
    output_path: str | Path = _OUTPUT_PATH,
    device: str | None = None,
    max_epochs: int = DEFAULT_MAX_EPOCHS,
    max_windows_per_split: int | None = None,
    seed: int = DEFAULT_SEED,
    batch_size: int = DEFAULT_BATCH_SIZE,
    learning_rate: float = DEFAULT_LEARNING_RATE,
    early_stopping_patience: int = DEFAULT_EARLY_STOPPING_PATIENCE,
    dataset_loader: Callable[..., PreparedDataset] | None = None,
    job_runner: Callable[..., list[dict[str, object]]] | None = None,
) -> pl.DataFrame:
    unknown_variants = [variant for variant in model_variants if variant not in MODEL_VARIANTS]
    if unknown_variants:
        raise ValueError(f"Unsupported model variants: {unknown_variants!r}")
    loader = dataset_loader or prepare_dataset
    runner = job_runner or execute_training_job
    rows: list[dict[str, object]] = []
    total_jobs = sum(
        len(
            build_requested_packs(
                dataset_id,
                covariate_stages=covariate_stages,
                include_power_only_reference=include_power_only_reference,
            )
        )
        * len(model_variants)
        for dataset_id in dataset_ids
    )
    job_progress = _create_progress_bar(total=total_jobs, desc="ltsf-linear jobs", leave=True)

    try:
        for dataset_id in dataset_ids:
            for pack in build_requested_packs(
                dataset_id,
                covariate_stages=covariate_stages,
                include_power_only_reference=include_power_only_reference,
            ):
                job_progress.set_postfix_str(f"{dataset_id}/{pack.pack_name}")
                prepared = loader(
                    dataset_id,
                    pack=pack,
                    cache_root=cache_root,
                    max_windows_per_split=max_windows_per_split,
                )
                for model_variant in model_variants:
                    job_progress.set_postfix_str(f"{dataset_id}/{pack.pack_name}/{model_variant}")
                    rows.extend(
                        runner(
                            prepared,
                            model_variant=model_variant,
                            device=device,
                            seed=seed,
                            batch_size=batch_size,
                            learning_rate=learning_rate,
                            max_epochs=max_epochs,
                            early_stopping_patience=early_stopping_patience,
                        )
                    )
                    job_progress.update(1)
    finally:
        job_progress.close()

    results = sort_result_frame(pl.DataFrame(rows).select(_RESULT_COLUMNS))
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    results.write_csv(output)
    return results


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run local LTSF-Linear dense-window training with rolling and non-overlap evaluation."
    )
    parser.add_argument(
        "--dataset",
        action="append",
        choices=list(DEFAULT_DATASETS),
        dest="datasets",
        help="Limit execution to one or more datasets. Defaults to all supported datasets.",
    )
    parser.add_argument(
        "--model",
        action="append",
        choices=list(MODEL_VARIANTS),
        dest="models",
        help="Limit execution to one or more model variants. Defaults to both nlinear and dlinear.",
    )
    parser.add_argument(
        "--covariate-stage",
        action="append",
        choices=list(DEFAULT_COVARIATE_STAGES),
        dest="covariate_stages",
        help="Limit execution to one or more exogenous covariate stages. Defaults to all stages.",
    )
    parser.set_defaults(include_power_only_reference=True)
    parser.add_argument(
        "--include-power-only-reference",
        action="store_true",
        dest="include_power_only_reference",
        help="Also emit one power-only reference row per dataset in the same output file.",
    )
    parser.add_argument(
        "--no-power-only-reference",
        action="store_false",
        dest="include_power_only_reference",
        help="Skip the power-only reference row.",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cuda", "mps", "cpu"),
        default="auto",
        help="Training device. Defaults to auto (cuda -> mps -> cpu).",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=DEFAULT_MAX_EPOCHS,
        help="Maximum training epochs.",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=_OUTPUT_PATH,
        help=f"Output CSV path. Defaults to {_OUTPUT_PATH}.",
    )
    parser.add_argument(
        "--max-windows-per-split",
        type=int,
        default=None,
        help=(
            "Optional smoke-test limit applied to the dense base windows in each split "
            "before non-overlap thinning."
        ),
    )
    parser.add_argument(
        "--reference-only",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--run-label",
        type=str,
        default=None,
        help="Optional label suffix for the formal run record under experiment/artifacts/runs/ltsf_linear_local/.",
    )
    parser.add_argument(
        "--no-record-run",
        action="store_true",
        help="Skip writing a formal run record manifest under experiment/artifacts/runs/.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    if args.reference_only:
        covariate_stages: tuple[str, ...] = ()
        include_power_only_reference = True
    else:
        covariate_stages = tuple(args.covariate_stages) if args.covariate_stages else DEFAULT_COVARIATE_STAGES
        include_power_only_reference = bool(args.include_power_only_reference)
    results = run_experiment(
        dataset_ids=tuple(args.datasets) if args.datasets else DEFAULT_DATASETS,
        model_variants=tuple(args.models) if args.models else MODEL_VARIANTS,
        covariate_stages=covariate_stages,
        include_power_only_reference=include_power_only_reference,
        device=args.device,
        max_epochs=args.epochs,
        output_path=args.output_path,
        max_windows_per_split=args.max_windows_per_split,
    )
    if not args.no_record_run:
        implementation_labels = (
            ("reference",)
            if args.reference_only
            else tuple(
                label
                for label in (
                    ("reference",) if include_power_only_reference else ()
                ) + tuple(covariate_stages)
            )
        )
        record_cli_run(
            family_id=FAMILY_ID,
            repo_root=_REPO_ROOT,
            invocation_kind="family_runner",
            entrypoint="experiment/families/ltsf-linear/run_ltsf_linear.py",
            args=vars(args),
            output_path=args.output_path,
            result_row_count=results.height,
            dataset_ids=tuple(args.datasets) if args.datasets else DEFAULT_DATASETS,
            feature_protocol_ids=resolve_family_feature_protocol_ids(
                FAMILY_ID,
                implementation_labels,
                repo_root=_REPO_ROOT,
            ),
            model_variants=tuple(args.models) if args.models else MODEL_VARIANTS,
            eval_protocols=(ROLLING_EVAL_PROTOCOL, NON_OVERLAP_EVAL_PROTOCOL),
            result_splits=("val", "test"),
            run_label=args.run_label,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
