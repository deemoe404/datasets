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
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset
except ImportError:  # pragma: no cover - exercised in the root env where torch is absent
    torch = None
    nn = None
    DataLoader = None
    TensorDataset = None


EXPERIMENT_DIR = Path(__file__).resolve().parent
COMMON_DIR = EXPERIMENT_DIR.parent / "common"
if str(COMMON_DIR) not in sys.path:
    sys.path.insert(0, str(COMMON_DIR))

from covariate_packs import (  # noqa: E402
    DEFAULT_COVARIATE_STAGES,
    CovariatePackSpec,
    iter_covariate_packs,
    reference_pack_for,
)


MODEL_ID = "LTSF-Linear"
TASK_ID = "next_6h_from_24h_stride_6h"
SPLIT_PROTOCOL = "chrono_70_10_20"
DEFAULT_DATASETS = ("kelmarsh", "penmanshiel", "hill_of_towie", "sdwpf_kddcup")
MODEL_VARIANTS = ("nlinear", "dlinear")
HISTORY_STEPS = 144
FORECAST_STEPS = 36
STRIDE_STEPS = 36
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

_REPO_ROOT = Path(__file__).resolve().parents[2]
_CACHE_ROOT = _REPO_ROOT / "cache"
_OUTPUT_PATH = _REPO_ROOT / "experiment" / "ltsf-linear.csv"
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
    "history_steps",
    "forecast_steps",
    "stride_steps",
    "split_protocol",
    "covariate_stage",
    "covariate_pack",
    "feature_set",
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
class RawSplitData:
    target_inputs: np.ndarray
    raw_exogenous_inputs: np.ndarray
    targets: np.ndarray
    output_start_us: np.ndarray
    output_end_us: np.ndarray


@dataclass(frozen=True)
class SplitData:
    target_inputs: np.ndarray
    exogenous_inputs: np.ndarray
    targets: np.ndarray
    output_start_us: np.ndarray
    output_end_us: np.ndarray


@dataclass(frozen=True)
class PreparedDataset:
    dataset_id: str
    rated_power_kw: float
    history_steps: int
    forecast_steps: int
    stride_steps: int
    covariate_stage: str
    covariate_pack: str
    feature_set: str
    covariate_count: int
    covariate_policy: str
    train: SplitData
    val: SplitData
    test: SplitData


@dataclass(frozen=True)
class TrainingOutcome:
    best_epoch: int
    epochs_ran: int
    best_val_rmse_pu: float
    device: str
    metrics: dict[str, float | int]


def _profile_log(dataset_id: str, phase: str, **fields: object) -> None:
    payload = {"dataset_id": dataset_id, "phase": phase, **fields}
    print(
        f"{PROFILE_LOG_PREFIX}{json.dumps(payload, ensure_ascii=False, default=str, sort_keys=True)}",
        file=sys.stderr,
        flush=True,
    )


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
    if torch is None or nn is None or DataLoader is None or TensorDataset is None:
        raise ImportError(
            "PyTorch is unavailable in the current environment. "
            "Create experiment/ltsf-linear/.conda with ./create_env.sh."
        )
    return torch, nn, DataLoader, TensorDataset


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
    feature_set: str,
    cache_root: str | Path = _CACHE_ROOT,
) -> tuple[str, Path]:
    cache_root_path = Path(cache_root)
    direct_path = (
        cache_root_path
        / dataset_id
        / "gold_base"
        / QUALITY_PROFILE
        / SERIES_LAYOUT
        / feature_set
        / "series.parquet"
    )
    if direct_path.exists():
        return feature_set, direct_path

    _ensure_repo_src_on_path()
    from wind_datasets import get_dataset_spec
    from wind_datasets.datasets import get_builder

    spec = get_dataset_spec(dataset_id)
    builder = get_builder(spec, cache_root_path)
    resolved_feature_set = builder.resolve_feature_set(feature_set)
    series_path = builder.cache_paths.gold_base_series_path_for(
        spec.default_quality_profile,
        layout=SERIES_LAYOUT,
        feature_set=resolved_feature_set,
    )
    if not series_path.exists():
        builder.build_gold_base(
            quality_profile=spec.default_quality_profile,
            layout=SERIES_LAYOUT,
            feature_set=resolved_feature_set,
        )
    return resolved_feature_set, series_path


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
    resolved_feature_set, series_path = resolve_turbine_series_path(
        dataset_id,
        feature_set=pack.feature_set,
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
        feature_set=resolved_feature_set,
        rows=frame.height,
        columns=len(frame.columns),
        covariates=len(covariate_columns),
        duration_seconds=round(time.monotonic() - load_started, 6),
    )
    return resolved_feature_set, covariate_columns, frame


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


def build_chrono_split_lookup(output_start_timestamps: Sequence[datetime]) -> pl.DataFrame:
    unique_sorted = sorted(dict.fromkeys(output_start_timestamps))
    total = len(unique_sorted)
    train_count = math.floor(total * 0.7)
    val_count = math.floor(total * 0.1)
    test_count = total - train_count - val_count
    if min(train_count, val_count, test_count) <= 0:
        raise ValueError(
            f"Chronological split {SPLIT_PROTOCOL!r} requires non-empty train/val/test, found "
            f"{train_count}/{val_count}/{test_count}."
        )
    return pl.DataFrame(
        {
            "output_start_ts": unique_sorted,
            "split": (
                ["train"] * train_count
                + ["val"] * val_count
                + ["test"] * test_count
            ),
        }
    )


def split_window_index(
    window_index: pl.DataFrame,
    *,
    max_windows_per_split: int | None = None,
) -> dict[str, pl.DataFrame]:
    split_lookup = build_chrono_split_lookup(window_index["output_start_ts"].to_list())
    frames = (
        window_index.join(split_lookup, on="output_start_ts", how="inner")
        .sort(["output_start_ts", "turbine_id"])
    )
    split_frames: dict[str, pl.DataFrame] = {}
    for split_name in ("train", "val", "test"):
        split_frame = frames.filter(pl.col("split") == split_name)
        if max_windows_per_split is not None:
            split_frame = split_frame.head(max_windows_per_split)
        if split_frame.is_empty():
            raise ValueError(f"Split {split_name!r} is empty after window selection.")
        split_frames[split_name] = split_frame
    return split_frames


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


def build_raw_split_samples(
    window_index: pl.DataFrame,
    turbine_series_map: Mapping[str, TurbineSeries],
    *,
    history_steps: int,
    forecast_steps: int,
    covariate_columns: Sequence[str],
) -> RawSplitData:
    sample_target_inputs: list[np.ndarray] = []
    sample_covariates: list[np.ndarray] = []
    sample_targets: list[np.ndarray] = []
    output_start_values: list[int] = []
    output_end_values: list[int] = []

    working = window_index.with_columns(
        pl.col("output_start_ts").cast(pl.Int64).alias("output_start_us"),
        pl.col("output_end_ts").cast(pl.Int64).alias("output_end_us"),
    )

    for turbine_frame in working.partition_by("turbine_id", maintain_order=True):
        turbine_id = turbine_frame["turbine_id"][0]
        series = turbine_series_map[turbine_id]
        timestamp_index = {int(timestamp): index for index, timestamp in enumerate(series.timestamps_us.tolist())}
        for output_start_us, output_end_us in zip(
            turbine_frame["output_start_us"].to_list(),
            turbine_frame["output_end_us"].to_list(),
            strict=True,
        ):
            target_index = timestamp_index.get(int(output_start_us))
            if target_index is None:
                raise KeyError(
                    f"Output start timestamp {output_start_us!r} is missing for turbine {turbine_id!r}."
                )
            context = series.target_pu[target_index - history_steps : target_index]
            future = series.target_pu[target_index : target_index + forecast_steps]
            if context.shape[0] != history_steps or future.shape[0] != forecast_steps:
                raise ValueError(
                    f"Window for turbine {turbine_id!r} does not match expected history/forecast sizes."
                )
            if np.isnan(context).any() or np.isnan(future).any():
                continue
            sample_target_inputs.append(context.astype(np.float32, copy=True))
            if covariate_columns:
                covariate_history = np.column_stack(
                    [
                        series.past_covariates[column][target_index - history_steps : target_index]
                        for column in covariate_columns
                    ]
                ).astype(np.float32, copy=True)
            else:
                covariate_history = np.empty((history_steps, 0), dtype=np.float32)
            sample_covariates.append(covariate_history)
            sample_targets.append(future.astype(np.float32, copy=True))
            output_start_values.append(int(output_start_us))
            output_end_values.append(int(output_end_us))

    if not sample_target_inputs:
        raise ValueError("No valid samples remain after filtering NaN target windows.")

    return RawSplitData(
        target_inputs=np.stack(sample_target_inputs),
        raw_exogenous_inputs=np.stack(sample_covariates)
        if sample_covariates
        else np.empty((0, history_steps, 0), dtype=np.float32),
        targets=np.stack(sample_targets),
        output_start_us=np.asarray(output_start_values, dtype=np.int64),
        output_end_us=np.asarray(output_end_values, dtype=np.int64),
    )


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


def finalize_split_data(
    raw_split: RawSplitData,
    *,
    means: np.ndarray,
    stds: np.ndarray,
) -> SplitData:
    return SplitData(
        target_inputs=raw_split.target_inputs,
        exogenous_inputs=transform_exogenous_inputs(raw_split.raw_exogenous_inputs, means=means, stds=stds),
        targets=raw_split.targets,
        output_start_us=raw_split.output_start_us,
        output_end_us=raw_split.output_end_us,
    )


def prepare_dataset(
    dataset_id: str,
    *,
    pack: CovariatePackSpec,
    cache_root: str | Path = _CACHE_ROOT,
    max_windows_per_split: int | None = None,
) -> PreparedDataset:
    metadata = load_dataset_metadata(dataset_id, cache_root=cache_root)
    strict_window_index = load_strict_window_index(dataset_id, cache_root=cache_root)
    split_frames = split_window_index(strict_window_index, max_windows_per_split=max_windows_per_split)
    resolved_feature_set, covariate_columns, series = load_series_frame(dataset_id, pack=pack, cache_root=cache_root)
    prepared_series = prepare_series_frame(series, covariate_columns=covariate_columns)
    turbine_series_map = build_turbine_series_map(
        prepared_series,
        rated_power_kw=metadata.rated_power_kw,
        covariate_columns=covariate_columns,
    )
    raw_split_data = {
        split_name: build_raw_split_samples(
            split_frame,
            turbine_series_map,
            history_steps=HISTORY_STEPS,
            forecast_steps=FORECAST_STEPS,
            covariate_columns=covariate_columns,
        )
        for split_name, split_frame in split_frames.items()
    }
    means, stds = fit_covariate_statistics(raw_split_data["train"].raw_exogenous_inputs)
    split_data = {
        split_name: finalize_split_data(raw_split, means=means, stds=stds)
        for split_name, raw_split in raw_split_data.items()
    }
    covariate_policy = COVARIATE_POLICY if covariate_columns else REFERENCE_COVARIATE_POLICY
    _profile_log(
        dataset_id,
        "prepare_dataset_complete",
        covariate_stage=pack.stage,
        covariate_pack=pack.pack_name,
        feature_set=resolved_feature_set,
        covariate_count=len(covariate_columns),
        train_windows=split_data["train"].target_inputs.shape[0],
        val_windows=split_data["val"].target_inputs.shape[0],
        test_windows=split_data["test"].target_inputs.shape[0],
        rated_power_kw=metadata.rated_power_kw,
    )
    return PreparedDataset(
        dataset_id=dataset_id,
        rated_power_kw=metadata.rated_power_kw,
        history_steps=HISTORY_STEPS,
        forecast_steps=FORECAST_STEPS,
        stride_steps=STRIDE_STEPS,
        covariate_stage=pack.stage,
        covariate_pack=pack.pack_name,
        feature_set=resolved_feature_set,
        covariate_count=len(covariate_columns),
        covariate_policy=covariate_policy,
        train=split_data["train"],
        val=split_data["val"],
        test=split_data["test"],
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
    split_data: SplitData,
    *,
    batch_size: int,
    shuffle: bool,
    seed: int,
):
    resolved_torch, _, resolved_loader, resolved_dataset = require_torch()
    dataset = resolved_dataset(
        resolved_torch.from_numpy(split_data.target_inputs),
        resolved_torch.from_numpy(split_data.exogenous_inputs),
        resolved_torch.from_numpy(split_data.targets),
    )
    generator = resolved_torch.Generator()
    generator.manual_seed(seed)
    return resolved_loader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        generator=generator if shuffle else None,
    )


def evaluate_model(model, loader, *, device: str, rated_power_kw: float) -> dict[str, float | int]:
    resolved_torch, _, _, _ = require_torch()
    metrics = {
        "window_count": 0,
        "prediction_count": 0,
        "abs_error_sum": 0.0,
        "squared_error_sum": 0.0,
        "normalized_abs_error_sum": 0.0,
        "normalized_squared_error_sum": 0.0,
    }
    model.eval()
    with resolved_torch.no_grad():
        for batch_target_inputs, batch_exogenous_inputs, batch_targets in loader:
            batch_target_inputs = batch_target_inputs.to(device=device, dtype=resolved_torch.float32)
            batch_exogenous_inputs = batch_exogenous_inputs.to(device=device, dtype=resolved_torch.float32)
            batch_targets = batch_targets.to(device=device, dtype=resolved_torch.float32)
            predictions = model(batch_target_inputs, batch_exogenous_inputs)
            errors_pu = predictions - batch_targets
            errors_kw = errors_pu * rated_power_kw
            metrics["window_count"] = int(metrics["window_count"]) + int(batch_target_inputs.shape[0])
            metrics["prediction_count"] = int(metrics["prediction_count"]) + int(batch_targets.numel())
            metrics["abs_error_sum"] = float(metrics["abs_error_sum"]) + float(resolved_torch.abs(errors_kw).sum().item())
            metrics["squared_error_sum"] = float(metrics["squared_error_sum"]) + float(
                resolved_torch.square(errors_kw).sum().item()
            )
            metrics["normalized_abs_error_sum"] = float(metrics["normalized_abs_error_sum"]) + float(
                resolved_torch.abs(errors_pu).sum().item()
            )
            metrics["normalized_squared_error_sum"] = float(metrics["normalized_squared_error_sum"]) + float(
                resolved_torch.square(errors_pu).sum().item()
            )
    prediction_count = int(metrics["prediction_count"])
    return {
        **metrics,
        "mae_kw": _safe_divide(float(metrics["abs_error_sum"]), prediction_count),
        "rmse_kw": _safe_rmse(float(metrics["squared_error_sum"]), prediction_count),
        "mae_pu": _safe_divide(float(metrics["normalized_abs_error_sum"]), prediction_count),
        "rmse_pu": _safe_rmse(float(metrics["normalized_squared_error_sum"]), prediction_count),
    }


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
) -> TrainingOutcome:
    resolved_torch, _, _, _ = require_torch()
    _set_random_seed(seed)
    resolved_device = resolve_device(device)
    model = build_model(
        model_variant,
        history_steps=prepared_dataset.history_steps,
        forecast_steps=prepared_dataset.forecast_steps,
        exogenous_channels=prepared_dataset.train.exogenous_inputs.shape[2],
    ).to(device=resolved_device)
    optimizer = resolved_torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = resolved_torch.nn.MSELoss()
    train_loader = _build_dataloader(prepared_dataset.train, batch_size=batch_size, shuffle=True, seed=seed)
    val_loader = _build_dataloader(prepared_dataset.val, batch_size=batch_size, shuffle=False, seed=seed)
    test_loader = _build_dataloader(prepared_dataset.test, batch_size=batch_size, shuffle=False, seed=seed)

    best_state: dict[str, Any] | None = None
    best_epoch = 0
    best_val_rmse_pu = float("inf")
    epochs_without_improvement = 0
    epochs_ran = 0

    for epoch_index in range(1, max_epochs + 1):
        model.train()
        for batch_target_inputs, batch_exogenous_inputs, batch_targets in train_loader:
            batch_target_inputs = batch_target_inputs.to(device=resolved_device, dtype=resolved_torch.float32)
            batch_exogenous_inputs = batch_exogenous_inputs.to(device=resolved_device, dtype=resolved_torch.float32)
            batch_targets = batch_targets.to(device=resolved_device, dtype=resolved_torch.float32)
            optimizer.zero_grad(set_to_none=True)
            predictions = model(batch_target_inputs, batch_exogenous_inputs)
            loss = criterion(predictions, batch_targets)
            loss.backward()
            optimizer.step()

        epochs_ran = epoch_index
        val_metrics = evaluate_model(
            model,
            val_loader,
            device=resolved_device,
            rated_power_kw=prepared_dataset.rated_power_kw,
        )
        val_rmse_pu = float(val_metrics["rmse_pu"])
        if val_rmse_pu < best_val_rmse_pu - 1e-12:
            best_val_rmse_pu = val_rmse_pu
            best_epoch = epoch_index
            best_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= early_stopping_patience:
                break

    if best_state is None:  # pragma: no cover - defensive
        raise RuntimeError("Training completed without a best checkpoint.")

    model.load_state_dict(best_state)
    test_metrics = evaluate_model(
        model,
        test_loader,
        device=resolved_device,
        rated_power_kw=prepared_dataset.rated_power_kw,
    )
    return TrainingOutcome(
        best_epoch=best_epoch,
        epochs_ran=epochs_ran,
        best_val_rmse_pu=best_val_rmse_pu,
        device=resolved_device,
        metrics=test_metrics,
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


def build_result_row(
    prepared_dataset: PreparedDataset,
    *,
    model_variant: str,
    training_outcome: TrainingOutcome,
    runtime_seconds: float,
    seed: int,
    batch_size: int,
    learning_rate: float,
) -> dict[str, object]:
    test_split = prepared_dataset.test
    metrics = training_outcome.metrics
    return {
        "dataset_id": prepared_dataset.dataset_id,
        "model_id": MODEL_ID,
        "model_variant": model_variant,
        "task_id": TASK_ID,
        "history_steps": prepared_dataset.history_steps,
        "forecast_steps": prepared_dataset.forecast_steps,
        "stride_steps": prepared_dataset.stride_steps,
        "split_protocol": SPLIT_PROTOCOL,
        "covariate_stage": prepared_dataset.covariate_stage,
        "covariate_pack": prepared_dataset.covariate_pack,
        "feature_set": prepared_dataset.feature_set,
        "covariate_count": prepared_dataset.covariate_count,
        "covariate_policy": prepared_dataset.covariate_policy,
        "window_count": int(metrics["window_count"]),
        "prediction_count": int(metrics["prediction_count"]),
        "start_timestamp": _timestamp_us_to_string(int(test_split.output_start_us.min())),
        "end_timestamp": _timestamp_us_to_string(int(test_split.output_end_us.max())),
        "mae_kw": float(metrics["mae_kw"]),
        "rmse_kw": float(metrics["rmse_kw"]),
        "mae_pu": float(metrics["mae_pu"]),
        "rmse_pu": float(metrics["rmse_pu"]),
        "device": training_outcome.device,
        "runtime_seconds": round(runtime_seconds, 6),
        "train_window_count": int(prepared_dataset.train.target_inputs.shape[0]),
        "val_window_count": int(prepared_dataset.val.target_inputs.shape[0]),
        "test_window_count": int(prepared_dataset.test.target_inputs.shape[0]),
        "best_epoch": training_outcome.best_epoch,
        "epochs_ran": training_outcome.epochs_ran,
        "best_val_rmse_pu": training_outcome.best_val_rmse_pu,
        "seed": seed,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
    }


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
) -> dict[str, object]:
    dataset_start = time.monotonic()
    training_outcome = train_model(
        model_variant,
        prepared_dataset,
        device=resolve_device(device),
        seed=seed,
        batch_size=batch_size,
        learning_rate=learning_rate,
        max_epochs=max_epochs,
        early_stopping_patience=early_stopping_patience,
    )
    runtime_seconds = time.monotonic() - dataset_start
    _profile_log(
        prepared_dataset.dataset_id,
        "training_complete",
        covariate_stage=prepared_dataset.covariate_stage,
        covariate_pack=prepared_dataset.covariate_pack,
        model_variant=model_variant,
        best_epoch=training_outcome.best_epoch,
        epochs_ran=training_outcome.epochs_ran,
        best_val_rmse_pu=training_outcome.best_val_rmse_pu,
        test_rmse_pu=training_outcome.metrics["rmse_pu"],
        runtime_seconds=round(runtime_seconds, 6),
    )
    return build_result_row(
        prepared_dataset,
        model_variant=model_variant,
        training_outcome=training_outcome,
        runtime_seconds=runtime_seconds,
        seed=seed,
        batch_size=batch_size,
        learning_rate=learning_rate,
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
        )
        .sort(["__dataset_order", "__stage_order", "__model_order"])
        .drop(["__dataset_order", "__stage_order", "__model_order"])
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
    job_runner: Callable[..., dict[str, object]] | None = None,
) -> pl.DataFrame:
    unknown_variants = [variant for variant in model_variants if variant not in MODEL_VARIANTS]
    if unknown_variants:
        raise ValueError(f"Unsupported model variants: {unknown_variants!r}")
    loader = dataset_loader or prepare_dataset
    runner = job_runner or execute_training_job
    rows: list[dict[str, object]] = []

    for dataset_id in dataset_ids:
        for pack in build_requested_packs(
            dataset_id,
            covariate_stages=covariate_stages,
            include_power_only_reference=include_power_only_reference,
        ):
            prepared = loader(
                dataset_id,
                pack=pack,
                cache_root=cache_root,
                max_windows_per_split=max_windows_per_split,
            )
            for model_variant in model_variants:
                rows.append(
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

    results = sort_result_frame(pl.DataFrame(rows).select(_RESULT_COLUMNS))
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    results.write_csv(output)
    return results


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run local LTSF-Linear training benchmarks.")
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
        help="Output CSV path. Defaults to experiment/ltsf-linear.csv in the repo root.",
    )
    parser.add_argument(
        "--max-windows-per-split",
        type=int,
        default=None,
        help="Optional smoke-test limit applied independently to train/val/test after split assignment.",
    )
    parser.add_argument(
        "--reference-only",
        action="store_true",
        help=argparse.SUPPRESS,
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
    run_experiment(
        dataset_ids=tuple(args.datasets) if args.datasets else DEFAULT_DATASETS,
        model_variants=tuple(args.models) if args.models else MODEL_VARIANTS,
        covariate_stages=covariate_stages,
        include_power_only_reference=include_power_only_reference,
        device=args.device,
        max_epochs=args.epochs,
        output_path=args.output_path,
        max_windows_per_split=args.max_windows_per_split,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
