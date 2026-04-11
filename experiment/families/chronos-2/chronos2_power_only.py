from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import UTC, datetime
import json
from math import sqrt
from pathlib import Path
import sys
import time
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import polars as pl

EXPERIMENT_ROOT = Path(__file__).resolve().parents[2]
COMMON_DIR = EXPERIMENT_ROOT / "infra" / "common"
if str(COMMON_DIR) not in sys.path:
    sys.path.insert(0, str(COMMON_DIR))

from run_records import record_cli_run  # noqa: E402
from window_protocols import (  # noqa: E402
    DEFAULT_WINDOW_PROTOCOL,
    HORIZON_METRIC_SCOPE,
    NON_OVERLAP_EVAL_PROTOCOL,
    OVERALL_METRIC_SCOPE,
    ROLLING_EVAL_PROTOCOL,
    SPLIT_PROTOCOL,
    WINDOW_PROTOCOL_CHOICES,
    WindowDescriptorIndex,
    build_chrono_split_lookup,
    build_split_boundaries,
    build_task_spec as build_window_protocol_task_spec,
    build_window_descriptor_index as build_shared_window_descriptor_index,
    default_output_path as resolve_default_output_path,
    resolve_window_protocol,
    split_window_index,
    thin_non_overlap_window_index,
)

MODEL_ID = "amazon/chronos-2"
FAMILY_ID = "chronos2_power_only"
EXPERIMENT_NAME = "chronos-2"
TASK_ID = resolve_window_protocol(DEFAULT_WINDOW_PROTOCOL).task_id
TARGET_POLICY = "invalid_to_nan_clip_0_rated"
DEFAULT_DATASETS = ("kelmarsh", "penmanshiel", "hill_of_towie", "sdwpf_kddcup")
DEFAULT_BATCH_SIZE = 256
DEFAULT_NEIGHBOR_COUNT = 6
UNIVARIATE_SUFFIX = "_univariate"
UNIVARIATE_POWER_STATS_SUFFIX = "_univariate_power_stats"
MULTIVARIATE_KNN6_SUFFIX = "_multivariate_knn6"
MULTIVARIATE_KNN6_POWER_STATS_SUFFIX = "_multivariate_knn6_power_stats"
MODE_CHOICES = ("all", "univariate", "multivariate_knn6", "univariate_power_stats")
EVAL_PROTOCOL_CHOICES = (ROLLING_EVAL_PROTOCOL, NON_OVERLAP_EVAL_PROTOCOL)
_REPO_ROOT = EXPERIMENT_ROOT.parent
_CACHE_ROOT = _REPO_ROOT / "cache"
_OUTPUT_PATH = resolve_default_output_path(
    repo_root=_REPO_ROOT,
    experiment_name=EXPERIMENT_NAME,
    window_protocol=DEFAULT_WINDOW_PROTOCOL,
)
_POWER_ONLY_COLUMNS = ("dataset", "turbine_id", "timestamp", "target_kw", "quality_flags")
_TASK_WINDOW_COLUMNS = (
    "dataset",
    "turbine_id",
    "output_start_ts",
    "output_end_ts",
    "is_complete_input",
    "is_complete_output",
    "quality_flags",
)
_POWER_STATS_COVARIATE_SPECS = {
    "kelmarsh": (
        ("Power, Minimum (kW)", "cov00_min"),
        ("Power, Maximum (kW)", "cov01_max"),
        ("Power, Standard deviation (kW)", "cov02_stddev"),
    ),
    "penmanshiel": (
        ("Power, Minimum (kW)", "cov00_min"),
        ("Power, Maximum (kW)", "cov01_max"),
        ("Power, Standard deviation (kW)", "cov02_stddev"),
    ),
    "hill_of_towie": (
        ("wtc_ActPower_min", "cov00_min"),
        ("wtc_ActPower_max", "cov01_max"),
        ("wtc_ActPower_stddev", "cov02_stddev"),
        ("wtc_ActPower_endvalue", "cov03_endvalue"),
    ),
}
_DATASET_RATED_POWER_KW = {
    "kelmarsh": 2050.0,
    "penmanshiel": 2050.0,
    "hill_of_towie": 2300.0,
    "sdwpf_kddcup": 1500.0,
}
_DATASET_ID_ALIASES = {
    "sdwpf_full": "sdwpf_kddcup",
}
_PENMANSHIEL_EPOCH_BOUNDARY = datetime(2024, 1, 1, 0, 0, 0)
_PENMANSHIEL_POST_2023_TURBINE_IDS = tuple(f"Penmanshiel {index:02d}" for index in range(11, 16))
_RESULT_COLUMNS = [
    "dataset_id",
    "model_id",
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
    "train_window_count",
    "val_window_count",
    "test_window_count",
]
PROFILE_LOG_PREFIX = "[chronos2_power_only] "


@dataclass(frozen=True)
class TurbineSeries:
    timestamps_us: np.ndarray
    target_kw_masked: np.ndarray


@dataclass
class ChunkProgressState:
    chunk_total_batches: int = 0
    completed_chunk_batches: int = 0


@dataclass(frozen=True)
class TaskCachePaths:
    dataset_id: str
    dataset_root: Path
    task_dir: Path
    window_index_path: Path
    task_context_path: Path
    turbine_static_path: Path


@dataclass(frozen=True)
class TurbinePowerStatsSeries:
    timestamps_us: np.ndarray
    target_kw_masked: np.ndarray
    past_covariates: dict[str, np.ndarray]


@dataclass(frozen=True)
class FarmPanel:
    turbine_ids: tuple[str, ...]
    timestamps_us: np.ndarray
    target_kw_masked: np.ndarray


@dataclass(frozen=True)
class FarmPowerStatsPanel:
    turbine_ids: tuple[str, ...]
    timestamps_us: np.ndarray
    target_kw_masked: np.ndarray
    past_covariates: dict[str, np.ndarray]


@dataclass(frozen=True)
class MultivariateEpochConfig:
    name: str
    active_turbine_ids: tuple[str, ...]
    target_turbine_ids: tuple[str, ...]
    forecast_start_us_min: int | None = None
    forecast_start_us_max: int | None = None


@dataclass(frozen=True)
class PreparedMultivariateEpoch:
    config: MultivariateEpochConfig
    required_input_turbine_ids: tuple[str, ...]
    neighbor_map: dict[str, tuple[str, ...]]
    turbine_series_map: dict[str, TurbineSeries]


@dataclass(frozen=True)
class PreparedMultivariatePowerStatsEpoch:
    config: MultivariateEpochConfig
    required_input_turbine_ids: tuple[str, ...]
    neighbor_map: dict[str, tuple[str, ...]]
    turbine_series_map: dict[str, TurbinePowerStatsSeries]


@dataclass(frozen=True)
class TargetPanelRun:
    name: str
    farm_panel: FarmPanel
    scored_row_index: int
    forecast_start_us_min: int | None = None
    forecast_start_us_max: int | None = None
    timestamp_index_map: dict[int, int] | None = None


@dataclass(frozen=True)
class TargetPowerStatsPanelRun:
    name: str
    farm_panel: FarmPowerStatsPanel
    scored_row_index: int
    forecast_start_us_min: int | None = None
    forecast_start_us_max: int | None = None
    timestamp_index_map: dict[int, int] | None = None


@dataclass(frozen=True)
class PreparedSplitWindowSet:
    train_window_count: int
    val_window_count: int
    test_window_count: int
    test_rolling_windows: WindowDescriptorIndex
    test_non_overlap_windows: WindowDescriptorIndex


@dataclass(frozen=True)
class EvaluationMetrics:
    window_count: int
    prediction_count: int
    mae_kw: float | None
    rmse_kw: float | None
    mae_pu: float | None
    rmse_pu: float | None
    horizon_window_count: np.ndarray
    horizon_prediction_count: np.ndarray
    horizon_mae_kw: np.ndarray
    horizon_rmse_kw: np.ndarray
    horizon_mae_pu: np.ndarray
    horizon_rmse_pu: np.ndarray


def _ensure_repo_src_on_path() -> None:
    src_path = _REPO_ROOT / "src"
    if str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))


def build_task_spec(*, window_protocol: str = DEFAULT_WINDOW_PROTOCOL):
    return build_window_protocol_task_spec(window_protocol, granularity="turbine")


def default_output_path(*, window_protocol: str = DEFAULT_WINDOW_PROTOCOL) -> Path:
    return resolve_default_output_path(
        repo_root=_REPO_ROOT,
        experiment_name=EXPERIMENT_NAME,
        window_protocol=window_protocol,
    )


def resolve_cache_paths(dataset_id: str, *, cache_root: str | Path = _CACHE_ROOT) -> TaskCachePaths:
    dataset_root = Path(cache_root) / dataset_id
    task_dir = dataset_root / "tasks" / "default" / "turbine" / TASK_ID
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


def load_strict_window_index(
    dataset_id: str,
    *,
    cache_root: str | Path = _CACHE_ROOT,
    turbine_ids: Sequence[str] | None = None,
) -> pl.DataFrame:
    paths = ensure_task_cache(dataset_id, cache_root=cache_root)
    load_started = time.monotonic()
    window_index_scan = (
        pl.scan_parquet(paths.window_index_path)
        .select(list(_TASK_WINDOW_COLUMNS))
        .filter(
            pl.col("is_complete_input")
            & pl.col("is_complete_output")
            & (pl.col("quality_flags").fill_null("") == "")
        )
    )
    if turbine_ids is not None:
        window_index_scan = window_index_scan.filter(pl.col("turbine_id").is_in(list(turbine_ids)))
    frame = window_index_scan.sort(["output_start_ts", "turbine_id"]).collect()
    _profile_log(
        dataset_id,
        "load_window_index",
        strict_windows=frame.height,
        target_turbines=None if turbine_ids is None else len(tuple(dict.fromkeys(turbine_ids))),
        duration_seconds=round(time.monotonic() - load_started, 6),
    )
    if frame.is_empty():
        raise ValueError(f"Dataset {dataset_id!r} has no strict windows for {TASK_ID}.")
    return frame


def _ordered_unique(values: Sequence[str]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(values))


def _ordered_intersection(values: Sequence[str], allowed_values: Sequence[str]) -> tuple[str, ...]:
    allowed = set(allowed_values)
    return tuple(value for value in values if value in allowed)


def _datetime_to_timestamp_us(value: datetime) -> int:
    return int(value.replace(tzinfo=UTC).timestamp() * 1_000_000)


def resolve_multivariate_epoch_configs(
    spec: Any,
    *,
    target_turbine_ids: Sequence[str],
) -> tuple[MultivariateEpochConfig, ...]:
    resolved_target_turbine_ids = tuple(target_turbine_ids)
    if spec.dataset_id != "penmanshiel":
        return (
            MultivariateEpochConfig(
                name="default",
                active_turbine_ids=tuple(spec.turbine_ids),
                target_turbine_ids=resolved_target_turbine_ids,
            ),
        )

    boundary_us = _datetime_to_timestamp_us(_PENMANSHIEL_EPOCH_BOUNDARY)
    epoch_configs: list[MultivariateEpochConfig] = []
    if resolved_target_turbine_ids:
        epoch_configs.append(
            MultivariateEpochConfig(
                name="pre_2024_full_farm",
                active_turbine_ids=tuple(spec.turbine_ids),
                target_turbine_ids=resolved_target_turbine_ids,
                forecast_start_us_max=boundary_us - 1,
            )
        )
    post_2023_targets = _ordered_intersection(
        resolved_target_turbine_ids,
        _PENMANSHIEL_POST_2023_TURBINE_IDS,
    )
    if post_2023_targets:
        epoch_configs.append(
            MultivariateEpochConfig(
                name="post_2023_active_subset",
                active_turbine_ids=_ordered_intersection(
                    spec.turbine_ids,
                    _PENMANSHIEL_POST_2023_TURBINE_IDS,
                ),
                target_turbine_ids=post_2023_targets,
                forecast_start_us_min=boundary_us,
            )
        )
    return tuple(epoch_configs)


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


def supports_power_stats(dataset_id: str) -> bool:
    return resolve_dataset_id(dataset_id) in _POWER_STATS_COVARIATE_SPECS


def supports_univariate_power_stats(dataset_id: str) -> bool:
    return supports_power_stats(dataset_id)


def resolve_power_stats_covariate_columns(dataset_id: str) -> tuple[str, ...]:
    resolved_dataset_id = resolve_dataset_id(dataset_id)
    try:
        return tuple(column for column, _ in _POWER_STATS_COVARIATE_SPECS[resolved_dataset_id])
    except KeyError as exc:
        raise ValueError(
            f"Dataset {resolved_dataset_id!r} does not support power_stats covariates."
        ) from exc


def resolve_power_stats_covariate_specs(dataset_id: str) -> tuple[tuple[str, str], ...]:
    resolved_dataset_id = resolve_dataset_id(dataset_id)
    try:
        return _POWER_STATS_COVARIATE_SPECS[resolved_dataset_id]
    except KeyError as exc:
        raise ValueError(
            f"Dataset {resolved_dataset_id!r} does not support power_stats covariates."
        ) from exc


def resolve_power_only_columns(
    dataset_id: str,
    *,
    include_power_stats: bool = False,
) -> tuple[str, ...]:
    columns = list(_POWER_ONLY_COLUMNS)
    if include_power_stats:
        columns.extend(resolve_power_stats_covariate_columns(dataset_id))
    return tuple(columns)


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
    series_path = builder.cache_paths.gold_base_series_path_for(
        spec.default_quality_profile,
        layout="turbine",
    )
    if not series_path.exists():
        builder.build_gold_base(
            quality_profile=spec.default_quality_profile,
            layout="turbine",
        )
    return spec, series_path


def load_power_only_series_frame(
    dataset_id: str,
    *,
    cache_root: str | Path = _CACHE_ROOT,
    turbine_ids: Sequence[str] | None = None,
    include_power_stats: bool = False,
) -> tuple[Any, Path, pl.DataFrame]:
    spec, series_path = resolve_power_only_series_path(dataset_id, cache_root=cache_root)
    requested_turbine_ids = _ordered_unique(turbine_ids) if turbine_ids is not None else tuple(spec.turbine_ids)
    selected_columns = resolve_power_only_columns(
        spec.dataset_id,
        include_power_stats=include_power_stats,
    )
    available_columns = set(pl.read_parquet_schema(series_path))
    missing_columns = [column for column in selected_columns if column not in available_columns]
    if missing_columns:
        raise ValueError(
            f"Series {series_path} for dataset {spec.dataset_id!r} is missing required columns {missing_columns!r}."
        )
    load_started = time.monotonic()
    series_scan = pl.scan_parquet(series_path).select(list(selected_columns))
    if requested_turbine_ids != tuple(spec.turbine_ids):
        series_scan = series_scan.filter(pl.col("turbine_id").is_in(list(requested_turbine_ids)))
    series = series_scan.collect()
    load_duration = time.monotonic() - load_started
    _profile_log(
        spec.dataset_id,
        "load_power_only_series",
        path=str(series_path),
        columns=len(selected_columns),
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


def prepare_power_stats_series(
    series: pl.DataFrame,
    *,
    dataset_id: str,
    rated_power_kw: float,
) -> pl.DataFrame:
    covariate_columns = resolve_power_stats_covariate_columns(dataset_id)
    invalid_expr = pl.col("target_kw").is_null() | (pl.col("quality_flags").fill_null("") != "")
    return (
        series.sort(["turbine_id", "timestamp"])
        .with_columns(
            invalid_expr.alias("invalid_target"),
            pl.when(invalid_expr)
            .then(pl.lit(None, dtype=pl.Float64))
            .otherwise(pl.col("target_kw").clip(0.0, rated_power_kw))
            .alias("target_kw_masked"),
            *[
                pl.when(invalid_expr)
                .then(pl.lit(None, dtype=pl.Float64))
                .otherwise(pl.col(column).clip(0.0, rated_power_kw))
                .alias(column)
                for column in covariate_columns
            ],
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
                *covariate_columns,
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
            "Rebuild experiment/families/chronos-2/.conda with ./create_env.sh."
        ) from exc

    resolved_device = device or select_device()
    return Chronos2Pipeline.from_pretrained(model_id, device_map=resolved_device)


def load_dataset_inputs(
    dataset_id: str,
    *,
    cache_root: str | Path = _CACHE_ROOT,
    task_spec=None,
    turbine_ids: Sequence[str] | None = None,
    include_power_stats: bool = False,
) -> tuple[Any, Any, pl.DataFrame]:
    spec, resolved_task = resolve_dataset_task(dataset_id, task_spec=task_spec)
    _, _, series = load_power_only_series_frame(
        dataset_id,
        cache_root=cache_root,
        turbine_ids=turbine_ids,
        include_power_stats=include_power_stats,
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


def build_turbine_power_stats_series_map(
    series: pl.DataFrame,
    *,
    covariate_columns: Sequence[str],
) -> dict[str, TurbinePowerStatsSeries]:
    turbines: dict[str, TurbinePowerStatsSeries] = {}
    for turbine_frame in series.partition_by("turbine_id", maintain_order=True):
        turbine_id = turbine_frame["turbine_id"][0]
        turbines[turbine_id] = TurbinePowerStatsSeries(
            timestamps_us=turbine_frame["timestamp"].cast(pl.Int64).to_numpy(),
            target_kw_masked=turbine_frame["target_kw_masked"].cast(pl.Float32).to_numpy(),
            past_covariates={
                column: turbine_frame[column].cast(pl.Float32).to_numpy()
                for column in covariate_columns
            },
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


def build_epoch_knn_neighbor_map(
    turbine_static: pl.DataFrame,
    *,
    active_turbine_ids: Sequence[str],
) -> dict[str, tuple[str, ...]]:
    resolved_active_turbine_ids = tuple(active_turbine_ids)
    return build_knn_neighbor_map(
        turbine_static,
        turbine_ids=resolved_active_turbine_ids,
        max_neighbors=min(DEFAULT_NEIGHBOR_COUNT, len(resolved_active_turbine_ids)),
    )


def build_epoch_required_input_turbine_ids(
    target_turbine_ids: Sequence[str],
    neighbor_map: dict[str, tuple[str, ...]],
) -> tuple[str, ...]:
    return _ordered_unique(
        turbine_id
        for target_turbine_id in target_turbine_ids
        for turbine_id in neighbor_map[target_turbine_id]
    )


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


def build_local_power_stats_panel(
    turbine_series_map: dict[str, TurbinePowerStatsSeries],
    *,
    turbine_ids: Sequence[str],
    resolution_minutes: int,
    covariate_columns: Sequence[str],
) -> FarmPowerStatsPanel:
    ordered_turbine_ids = tuple(turbine_ids)
    if not ordered_turbine_ids:
        raise ValueError("Cannot build a local power_stats panel with no turbine ids.")
    missing_turbines = [turbine_id for turbine_id in ordered_turbine_ids if turbine_id not in turbine_series_map]
    if missing_turbines:
        raise ValueError(f"Missing turbine power_stats series for: {missing_turbines!r}")

    step_us = resolution_minutes * 60 * 1_000_000
    if step_us <= 0:
        raise ValueError("resolution_minutes must be positive.")

    min_timestamp_us = min(int(turbine_series_map[turbine_id].timestamps_us.min()) for turbine_id in ordered_turbine_ids)
    max_timestamp_us = max(int(turbine_series_map[turbine_id].timestamps_us.max()) for turbine_id in ordered_turbine_ids)
    full_timestamps_us = np.arange(min_timestamp_us, max_timestamp_us + step_us, step_us, dtype=np.int64)

    target_columns: list[np.ndarray] = []
    covariate_panels: dict[str, list[np.ndarray]] = {column: [] for column in covariate_columns}
    for turbine_id in ordered_turbine_ids:
        turbine_series = turbine_series_map[turbine_id]
        offsets = turbine_series.timestamps_us - min_timestamp_us
        if np.any(offsets % step_us != 0):
            raise ValueError(f"Turbine {turbine_id!r} has timestamps that do not align to a {resolution_minutes}-minute grid.")

        positions = (offsets // step_us).astype(np.int64, copy=False)
        aligned_target_values = np.full(full_timestamps_us.shape, np.nan, dtype=np.float32)
        aligned_target_values[positions] = turbine_series.target_kw_masked
        target_columns.append(aligned_target_values)

        for column in covariate_columns:
            aligned_covariate_values = np.full(full_timestamps_us.shape, np.nan, dtype=np.float32)
            aligned_covariate_values[positions] = turbine_series.past_covariates[column]
            covariate_panels[column].append(aligned_covariate_values)

    return FarmPowerStatsPanel(
        turbine_ids=ordered_turbine_ids,
        timestamps_us=full_timestamps_us,
        target_kw_masked=np.column_stack(target_columns).astype(np.float32, copy=False),
        past_covariates={
            column: np.column_stack(columns).astype(np.float32, copy=False)
            for column, columns in covariate_panels.items()
        },
    )


def build_flattened_power_stats_covariates(
    farm_panel: FarmPowerStatsPanel,
    *,
    anchor_index: int,
    history_steps: int,
    covariate_specs: Sequence[tuple[str, str]],
) -> dict[str, np.ndarray]:
    start_index = anchor_index - history_steps + 1
    end_index = anchor_index + 1
    flattened_covariates: dict[str, np.ndarray] = {}
    for turbine_index, _ in enumerate(farm_panel.turbine_ids):
        for raw_column, canonical_key in covariate_specs:
            flattened_covariates[f"neighbor_{turbine_index:02d}__{canonical_key}"] = farm_panel.past_covariates[raw_column][
                start_index:end_index,
                turbine_index,
            ].astype(np.float32, copy=True)
    return flattened_covariates


def build_timestamps_by_turbine(
    turbine_series_map: Mapping[str, TurbineSeries | TurbinePowerStatsSeries],
) -> dict[str, np.ndarray]:
    return {
        turbine_id: np.asarray(turbine_series.timestamps_us, dtype=np.int64)
        for turbine_id, turbine_series in turbine_series_map.items()
    }


def slice_window_descriptor(
    windows: WindowDescriptorIndex,
    *,
    window_offset: int = 0,
    max_windows_per_split: int | None = None,
) -> WindowDescriptorIndex:
    start = max(0, int(window_offset))
    if start >= len(windows):
        return WindowDescriptorIndex.empty()
    stop = len(windows) if max_windows_per_split is None else min(len(windows), start + int(max_windows_per_split))
    if stop <= start:
        return WindowDescriptorIndex.empty()
    return WindowDescriptorIndex(
        turbine_indices=windows.turbine_indices[start:stop].copy(),
        target_indices=windows.target_indices[start:stop].copy(),
        output_start_us=windows.output_start_us[start:stop].copy(),
        output_end_us=windows.output_end_us[start:stop].copy(),
    )


def resolve_selected_eval_protocols(
    eval_protocols: Sequence[str] | None,
) -> tuple[str, ...]:
    if eval_protocols is None:
        return EVAL_PROTOCOL_CHOICES
    resolved = tuple(dict.fromkeys(eval_protocols))
    unsupported = [eval_protocol for eval_protocol in resolved if eval_protocol not in EVAL_PROTOCOL_CHOICES]
    if unsupported:
        raise ValueError(f"Unsupported eval protocols: {unsupported!r}")
    return resolved


def prepare_split_window_set(
    dataset_id: str,
    *,
    cache_root: str | Path,
    raw_timestamps: Sequence[datetime],
    resolution_minutes: int,
    history_steps: int,
    forecast_steps: int,
    turbine_ids: Sequence[str],
    timestamps_by_turbine: Mapping[str, Sequence[int] | np.ndarray],
) -> PreparedSplitWindowSet:
    strict_window_index = load_strict_window_index(
        dataset_id,
        cache_root=cache_root,
        turbine_ids=turbine_ids,
    )
    split_frames = split_window_index(
        strict_window_index,
        raw_timestamps=raw_timestamps,
        resolution_minutes=resolution_minutes,
        history_steps=history_steps,
    )
    test_rolling_windows = build_shared_window_descriptor_index(
        split_frames["test"],
        turbine_ids=turbine_ids,
        timestamps_by_turbine=timestamps_by_turbine,
    )
    test_non_overlap_windows = thin_non_overlap_window_index(
        test_rolling_windows,
        turbine_ids=turbine_ids,
        forecast_steps=forecast_steps,
    )
    return PreparedSplitWindowSet(
        train_window_count=split_frames["train"].height,
        val_window_count=split_frames["val"].height,
        test_window_count=split_frames["test"].height,
        test_rolling_windows=test_rolling_windows,
        test_non_overlap_windows=test_non_overlap_windows,
    )


def iter_evaluation_windows(
    split_windows: PreparedSplitWindowSet,
    *,
    eval_protocols: Sequence[str],
    window_offset: int = 0,
    max_windows_per_split: int | None = None,
) -> tuple[tuple[str, WindowDescriptorIndex], ...]:
    specs: list[tuple[str, WindowDescriptorIndex]] = []
    for eval_protocol in eval_protocols:
        base_windows = (
            split_windows.test_rolling_windows
            if eval_protocol == ROLLING_EVAL_PROTOCOL
            else split_windows.test_non_overlap_windows
        )
        specs.append(
            (
                eval_protocol,
                slice_window_descriptor(
                    base_windows,
                    window_offset=window_offset,
                    max_windows_per_split=max_windows_per_split,
                ),
            )
        )
    return tuple(specs)


def _window_bounds(windows: WindowDescriptorIndex) -> tuple[str | None, str | None]:
    if len(windows) == 0:
        return None, None
    return (
        _timestamp_us_to_string(int(windows.output_start_us.min())),
        _timestamp_us_to_string(int(windows.output_end_us.max())),
    )


def _count_batches_from_windows(windows: WindowDescriptorIndex, batch_size: int) -> int:
    if len(windows) == 0:
        return 0
    return (len(windows) + batch_size - 1) // batch_size


def _iter_univariate_descriptor_batches(
    *,
    turbine_series: Sequence[TurbineSeries],
    windows: WindowDescriptorIndex,
    history_steps: int,
    forecast_steps: int,
    batch_size: int,
) -> Iterable[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    context_batch: list[np.ndarray] = []
    future_rows: list[np.ndarray] = []
    future_timestamp_rows: list[np.ndarray] = []

    for window_index_position in range(len(windows)):
        turbine_index = int(windows.turbine_indices[window_index_position])
        target_index = int(windows.target_indices[window_index_position])
        series = turbine_series[turbine_index]
        context = series.target_kw_masked[target_index - history_steps : target_index]
        future = series.target_kw_masked[target_index : target_index + forecast_steps]
        future_timestamp_values = series.timestamps_us[target_index : target_index + forecast_steps]
        if context.shape[0] != history_steps or future.shape[0] != forecast_steps:
            continue
        if np.isnan(context).all() or np.isnan(future).all():
            continue
        context_batch.append(context.astype(np.float32, copy=True))
        future_rows.append(future.astype(np.float64, copy=True))
        future_timestamp_rows.append(future_timestamp_values.astype(np.int64, copy=True))
        if len(context_batch) >= batch_size:
            yield (
                np.stack(context_batch)[:, None, :],
                np.stack(future_rows),
                np.stack(future_timestamp_rows),
            )
            context_batch = []
            future_rows = []
            future_timestamp_rows = []

    if context_batch:
        yield (
            np.stack(context_batch)[:, None, :],
            np.stack(future_rows),
            np.stack(future_timestamp_rows),
        )


def _iter_univariate_power_stats_descriptor_batches(
    *,
    turbine_series: Sequence[TurbinePowerStatsSeries],
    windows: WindowDescriptorIndex,
    history_steps: int,
    forecast_steps: int,
    batch_size: int,
) -> Iterable[tuple[list[dict[str, object]], np.ndarray, np.ndarray]]:
    input_batch: list[dict[str, object]] = []
    future_rows: list[np.ndarray] = []
    future_timestamp_rows: list[np.ndarray] = []

    for window_index_position in range(len(windows)):
        turbine_index = int(windows.turbine_indices[window_index_position])
        target_index = int(windows.target_indices[window_index_position])
        series = turbine_series[turbine_index]
        context = series.target_kw_masked[target_index - history_steps : target_index]
        future = series.target_kw_masked[target_index : target_index + forecast_steps]
        future_timestamp_values = series.timestamps_us[target_index : target_index + forecast_steps]
        if context.shape[0] != history_steps or future.shape[0] != forecast_steps:
            continue
        if np.isnan(context).all() or np.isnan(future).all():
            continue
        input_batch.append(
            {
                "target": context.astype(np.float32, copy=True),
                "past_covariates": {
                    column: values[target_index - history_steps : target_index].astype(np.float32, copy=True)
                    for column, values in series.past_covariates.items()
                },
            }
        )
        future_rows.append(future.astype(np.float64, copy=True))
        future_timestamp_rows.append(future_timestamp_values.astype(np.int64, copy=True))
        if len(input_batch) >= batch_size:
            yield (
                input_batch,
                np.stack(future_rows),
                np.stack(future_timestamp_rows),
            )
            input_batch = []
            future_rows = []
            future_timestamp_rows = []

    if input_batch:
        yield (
            input_batch,
            np.stack(future_rows),
            np.stack(future_timestamp_rows),
        )


def _select_panel_run_for_timestamp(
    panel_runs: Sequence[TargetPanelRun | TargetPowerStatsPanelRun],
    forecast_start_us: int,
) -> TargetPanelRun | TargetPowerStatsPanelRun | None:
    for panel_run in panel_runs:
        if panel_run.forecast_start_us_min is not None and forecast_start_us < panel_run.forecast_start_us_min:
            continue
        if panel_run.forecast_start_us_max is not None and forecast_start_us > panel_run.forecast_start_us_max:
            continue
        timestamp_index_map = panel_run.timestamp_index_map or {}
        if forecast_start_us in timestamp_index_map:
            return panel_run
    return None


def _iter_selected_multivariate_batches(
    *,
    panel_runs_by_target: Mapping[str, Sequence[TargetPanelRun]],
    target_turbine_ids: Sequence[str],
    windows: WindowDescriptorIndex,
    history_steps: int,
    forecast_steps: int,
    batch_size: int,
) -> Iterable[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    context_batch: list[np.ndarray] = []
    future_rows: list[np.ndarray] = []
    future_timestamps: list[np.ndarray] = []

    for window_index_position in range(len(windows)):
        target_turbine_id = target_turbine_ids[int(windows.turbine_indices[window_index_position])]
        forecast_start_us = int(windows.output_start_us[window_index_position])
        panel_run = _select_panel_run_for_timestamp(panel_runs_by_target[target_turbine_id], forecast_start_us)
        if panel_run is None or panel_run.timestamp_index_map is None:
            continue
        target_index = panel_run.timestamp_index_map[forecast_start_us]
        farm_panel = panel_run.farm_panel
        context = farm_panel.target_kw_masked[target_index - history_steps : target_index].T
        future = farm_panel.target_kw_masked[target_index : target_index + forecast_steps].T
        future_timestamp_values = farm_panel.timestamps_us[target_index : target_index + forecast_steps]
        if context.shape != (len(farm_panel.turbine_ids), history_steps):
            continue
        if future.shape != (len(farm_panel.turbine_ids), forecast_steps):
            continue
        if np.isnan(context).all() or np.isnan(future).all() or np.isnan(future[panel_run.scored_row_index]).all():
            continue
        context_batch.append(context.astype(np.float32, copy=True))
        future_rows.append(future.astype(np.float64, copy=True))
        future_timestamps.append(future_timestamp_values.astype(np.int64, copy=True))
        if len(context_batch) >= batch_size:
            yield (
                np.stack(context_batch),
                np.stack(future_rows),
                np.stack(future_timestamps),
            )
            context_batch = []
            future_rows = []
            future_timestamps = []

    if context_batch:
        yield (
            np.stack(context_batch),
            np.stack(future_rows),
            np.stack(future_timestamps),
        )


def _iter_selected_multivariate_power_stats_batches(
    *,
    panel_runs_by_target: Mapping[str, Sequence[TargetPowerStatsPanelRun]],
    target_turbine_ids: Sequence[str],
    windows: WindowDescriptorIndex,
    history_steps: int,
    forecast_steps: int,
    batch_size: int,
    covariate_specs: Sequence[tuple[str, str]],
) -> Iterable[tuple[list[dict[str, object]], np.ndarray, np.ndarray]]:
    input_batch: list[dict[str, object]] = []
    future_rows: list[np.ndarray] = []
    future_timestamps: list[np.ndarray] = []

    for window_index_position in range(len(windows)):
        target_turbine_id = target_turbine_ids[int(windows.turbine_indices[window_index_position])]
        forecast_start_us = int(windows.output_start_us[window_index_position])
        panel_run = _select_panel_run_for_timestamp(panel_runs_by_target[target_turbine_id], forecast_start_us)
        if panel_run is None or panel_run.timestamp_index_map is None:
            continue
        target_index = panel_run.timestamp_index_map[forecast_start_us]
        farm_panel = panel_run.farm_panel
        context = farm_panel.target_kw_masked[target_index - history_steps : target_index].T
        future = farm_panel.target_kw_masked[target_index : target_index + forecast_steps].T
        future_timestamp_values = farm_panel.timestamps_us[target_index : target_index + forecast_steps]
        if context.shape != (len(farm_panel.turbine_ids), history_steps):
            continue
        if future.shape != (len(farm_panel.turbine_ids), forecast_steps):
            continue
        if np.isnan(context).all() or np.isnan(future).all() or np.isnan(future[panel_run.scored_row_index]).all():
            continue
        input_batch.append(
            {
                "target": context.astype(np.float32, copy=True),
                "past_covariates": build_flattened_power_stats_covariates(
                    farm_panel,
                    anchor_index=target_index - 1,
                    history_steps=history_steps,
                    covariate_specs=covariate_specs,
                ),
            }
        )
        future_rows.append(future.astype(np.float64, copy=True))
        future_timestamps.append(future_timestamp_values.astype(np.int64, copy=True))
        if len(input_batch) >= batch_size:
            yield (
                input_batch,
                np.stack(future_rows),
                np.stack(future_timestamps),
            )
            input_batch = []
            future_rows = []
            future_timestamps = []

    if input_batch:
        yield (
            input_batch,
            np.stack(future_rows),
            np.stack(future_timestamps),
        )


def _initialize_metric_state(forecast_steps: int) -> dict[str, object]:
    return {
        "window_count": 0,
        "prediction_count": 0,
        "abs_error_sum": 0.0,
        "squared_error_sum": 0.0,
        "normalized_abs_error_sum": 0.0,
        "normalized_squared_error_sum": 0.0,
        "horizon_window_count": np.zeros((forecast_steps,), dtype=np.int64),
        "horizon_prediction_count": np.zeros((forecast_steps,), dtype=np.int64),
        "horizon_abs_error_sum": np.zeros((forecast_steps,), dtype=np.float64),
        "horizon_squared_error_sum": np.zeros((forecast_steps,), dtype=np.float64),
        "horizon_normalized_abs_error_sum": np.zeros((forecast_steps,), dtype=np.float64),
        "horizon_normalized_squared_error_sum": np.zeros((forecast_steps,), dtype=np.float64),
    }


def _update_metric_state(
    metrics: dict[str, object],
    *,
    prediction_batch: np.ndarray,
    actual_batch: np.ndarray,
    rated_power_kw: float,
) -> None:
    valid_mask = ~np.isnan(actual_batch)
    if not valid_mask.any():
        return
    valid_window_mask = valid_mask.any(axis=1)
    errors = prediction_batch - actual_batch
    absolute_errors = np.where(valid_mask, np.abs(errors), 0.0)
    squared_errors = np.where(valid_mask, np.square(errors), 0.0)
    normalized_errors = errors / rated_power_kw
    normalized_absolute_errors = np.where(valid_mask, np.abs(normalized_errors), 0.0)
    normalized_squared_errors = np.where(valid_mask, np.square(normalized_errors), 0.0)
    horizon_counts = valid_mask.sum(axis=0).astype(np.int64, copy=False)

    metrics["window_count"] = int(metrics["window_count"]) + int(valid_window_mask.sum())
    metrics["prediction_count"] = int(metrics["prediction_count"]) + int(valid_mask.sum())
    metrics["abs_error_sum"] = float(metrics["abs_error_sum"]) + float(absolute_errors.sum())
    metrics["squared_error_sum"] = float(metrics["squared_error_sum"]) + float(squared_errors.sum())
    metrics["normalized_abs_error_sum"] = float(metrics["normalized_abs_error_sum"]) + float(normalized_absolute_errors.sum())
    metrics["normalized_squared_error_sum"] = float(metrics["normalized_squared_error_sum"]) + float(
        normalized_squared_errors.sum()
    )
    metrics["horizon_window_count"] = np.asarray(metrics["horizon_window_count"], dtype=np.int64) + horizon_counts
    metrics["horizon_prediction_count"] = np.asarray(metrics["horizon_prediction_count"], dtype=np.int64) + horizon_counts
    metrics["horizon_abs_error_sum"] = np.asarray(metrics["horizon_abs_error_sum"], dtype=np.float64) + absolute_errors.sum(axis=0)
    metrics["horizon_squared_error_sum"] = np.asarray(metrics["horizon_squared_error_sum"], dtype=np.float64) + squared_errors.sum(axis=0)
    metrics["horizon_normalized_abs_error_sum"] = np.asarray(
        metrics["horizon_normalized_abs_error_sum"],
        dtype=np.float64,
    ) + normalized_absolute_errors.sum(axis=0)
    metrics["horizon_normalized_squared_error_sum"] = np.asarray(
        metrics["horizon_normalized_squared_error_sum"],
        dtype=np.float64,
    ) + normalized_squared_errors.sum(axis=0)


def _safe_array_divide(numerators: np.ndarray, denominators: np.ndarray) -> np.ndarray:
    result = np.full(numerators.shape, np.nan, dtype=np.float64)
    valid = denominators > 0
    result[valid] = numerators[valid] / denominators[valid]
    return result


def _safe_array_rmse(squared_error_sum: np.ndarray, denominators: np.ndarray) -> np.ndarray:
    result = np.full(squared_error_sum.shape, np.nan, dtype=np.float64)
    valid = denominators > 0
    result[valid] = np.sqrt(squared_error_sum[valid] / denominators[valid])
    return result


def finalize_metric_state(metrics: dict[str, object]) -> EvaluationMetrics:
    prediction_count = int(metrics["prediction_count"])
    horizon_prediction_count = np.asarray(metrics["horizon_prediction_count"], dtype=np.int64)
    horizon_window_count = np.asarray(metrics["horizon_window_count"], dtype=np.int64)
    horizon_mae_kw = _safe_array_divide(
        np.asarray(metrics["horizon_abs_error_sum"], dtype=np.float64),
        horizon_prediction_count,
    )
    horizon_rmse_kw = _safe_array_rmse(
        np.asarray(metrics["horizon_squared_error_sum"], dtype=np.float64),
        horizon_prediction_count,
    )
    horizon_mae_pu = _safe_array_divide(
        np.asarray(metrics["horizon_normalized_abs_error_sum"], dtype=np.float64),
        horizon_prediction_count,
    )
    horizon_rmse_pu = _safe_array_rmse(
        np.asarray(metrics["horizon_normalized_squared_error_sum"], dtype=np.float64),
        horizon_prediction_count,
    )
    return EvaluationMetrics(
        window_count=int(metrics["window_count"]),
        prediction_count=prediction_count,
        mae_kw=_safe_divide(float(metrics["abs_error_sum"]), prediction_count),
        rmse_kw=_safe_rmse(float(metrics["squared_error_sum"]), prediction_count),
        mae_pu=_safe_divide(float(metrics["normalized_abs_error_sum"]), prediction_count),
        rmse_pu=_safe_rmse(float(metrics["normalized_squared_error_sum"]), prediction_count),
        horizon_window_count=horizon_window_count,
        horizon_prediction_count=horizon_prediction_count,
        horizon_mae_kw=horizon_mae_kw,
        horizon_rmse_kw=horizon_rmse_kw,
        horizon_mae_pu=horizon_mae_pu,
        horizon_rmse_pu=horizon_rmse_pu,
    )


def _metric_value(value: float | np.floating[Any] | None) -> float | None:
    if value is None:
        return None
    if np.isnan(value):
        return None
    return float(value)


def build_result_rows(
    *,
    dataset_id: str,
    suffix: str,
    resolved_task: Any,
    resolution_minutes: int,
    split_windows: PreparedSplitWindowSet,
    runtime_seconds: float,
    device: str,
    evaluation_results: Sequence[tuple[str, WindowDescriptorIndex, EvaluationMetrics]],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    base_row = {
        "dataset_id": f"{dataset_id}{suffix}",
        "model_id": MODEL_ID,
        "task_id": resolved_task.task_id,
        "window_protocol": DEFAULT_WINDOW_PROTOCOL,
        "history_steps": resolved_task.history_steps,
        "forecast_steps": resolved_task.forecast_steps,
        "stride_steps": resolved_task.stride_steps,
        "split_protocol": SPLIT_PROTOCOL,
        "split_name": "test",
        "target_policy": TARGET_POLICY,
        "device": device,
        "runtime_seconds": round(runtime_seconds, 6),
        "train_window_count": split_windows.train_window_count,
        "val_window_count": split_windows.val_window_count,
        "test_window_count": split_windows.test_window_count,
    }
    for eval_protocol, windows, metrics in evaluation_results:
        start_timestamp, end_timestamp = _window_bounds(windows)
        rows.append(
            {
                **base_row,
                "eval_protocol": eval_protocol,
                "metric_scope": OVERALL_METRIC_SCOPE,
                "lead_step": None,
                "lead_minutes": None,
                "window_count": int(metrics.window_count),
                "prediction_count": int(metrics.prediction_count),
                "start_timestamp": start_timestamp,
                "end_timestamp": end_timestamp,
                "mae_kw": _metric_value(metrics.mae_kw),
                "rmse_kw": _metric_value(metrics.rmse_kw),
                "mae_pu": _metric_value(metrics.mae_pu),
                "rmse_pu": _metric_value(metrics.rmse_pu),
            }
        )
        for lead_index in range(resolved_task.forecast_steps):
            lead_step = lead_index + 1
            rows.append(
                {
                    **base_row,
                    "eval_protocol": eval_protocol,
                    "metric_scope": HORIZON_METRIC_SCOPE,
                    "lead_step": lead_step,
                    "lead_minutes": lead_step * resolution_minutes,
                    "window_count": int(metrics.horizon_window_count[lead_index]),
                    "prediction_count": int(metrics.horizon_prediction_count[lead_index]),
                    "start_timestamp": start_timestamp,
                    "end_timestamp": end_timestamp,
                    "mae_kw": _metric_value(metrics.horizon_mae_kw[lead_index]),
                    "rmse_kw": _metric_value(metrics.horizon_rmse_kw[lead_index]),
                    "mae_pu": _metric_value(metrics.horizon_mae_pu[lead_index]),
                    "rmse_pu": _metric_value(metrics.horizon_rmse_pu[lead_index]),
                }
            )
    return rows


def sort_result_frame(frame: pl.DataFrame) -> pl.DataFrame:
    eval_order = {ROLLING_EVAL_PROTOCOL: 0, NON_OVERLAP_EVAL_PROTOCOL: 1}
    metric_scope_order = {OVERALL_METRIC_SCOPE: 0, HORIZON_METRIC_SCOPE: 1}
    return (
        frame.with_columns(
            pl.col("eval_protocol").replace_strict(eval_order, default=len(eval_order)).alias("__eval_order"),
            pl.col("metric_scope").replace_strict(metric_scope_order, default=len(metric_scope_order)).alias("__metric_scope_order"),
            pl.col("lead_step").fill_null(0).alias("__lead_order"),
        )
        .sort(["dataset_id", "__eval_order", "__metric_scope_order", "__lead_order"])
        .drop(["__eval_order", "__metric_scope_order", "__lead_order"])
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


def _iter_univariate_power_stats_batches(
    *,
    turbine_series: TurbinePowerStatsSeries,
    history_steps: int,
    forecast_steps: int,
    stride_steps: int,
    batch_size: int,
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

        input_batch.append(
            {
                "target": context.astype(np.float32, copy=True),
                "past_covariates": {
                    column: values[anchor_index - history_steps + 1 : anchor_index + 1].astype(np.float32, copy=True)
                    for column, values in turbine_series.past_covariates.items()
                },
            }
        )
        future_rows.append(future.astype(np.float64, copy=True))
        future_timestamps.append(
            turbine_series.timestamps_us[anchor_index + 1 : anchor_index + 1 + forecast_steps].astype(np.int64, copy=True)
        )
        emitted_windows += 1

        if len(input_batch) >= batch_size:
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


def _iter_multivariate_batches_for_runs(
    *,
    panel_runs: Sequence[TargetPanelRun],
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
    max_windows_reached = False

    def flush_batch() -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
        nonlocal context_batch, future_rows, future_timestamps
        if not context_batch:
            return None
        batch = (
            np.stack(context_batch),
            np.stack(future_rows),
            np.stack(future_timestamps),
        )
        context_batch = []
        future_rows = []
        future_timestamps = []
        return batch

    for panel_run in panel_runs:
        farm_panel = panel_run.farm_panel
        max_anchor_index = farm_panel.timestamps_us.shape[0] - forecast_steps
        for anchor_index in range(history_steps - 1, max_anchor_index, stride_steps):
            forecast_start_us = int(farm_panel.timestamps_us[anchor_index + 1])
            if panel_run.forecast_start_us_min is not None and forecast_start_us < panel_run.forecast_start_us_min:
                continue
            if panel_run.forecast_start_us_max is not None and forecast_start_us > panel_run.forecast_start_us_max:
                continue

            context = farm_panel.target_kw_masked[anchor_index - history_steps + 1 : anchor_index + 1].T
            future = farm_panel.target_kw_masked[anchor_index + 1 : anchor_index + 1 + forecast_steps].T
            if context.shape != (len(farm_panel.turbine_ids), history_steps):
                continue
            if future.shape != (len(farm_panel.turbine_ids), forecast_steps):
                continue
            if (
                np.isnan(context).all()
                or np.isnan(future).all()
                or np.isnan(future[panel_run.scored_row_index]).all()
            ):
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
                batch = flush_batch()
                if batch is not None:
                    yield batch

            if max_windows_per_dataset is not None and emitted_windows >= max_windows_per_dataset:
                max_windows_reached = True
                break

        batch = flush_batch()
        if batch is not None:
            yield batch
        if max_windows_reached:
            break


def _iter_multivariate_power_stats_batches_for_runs(
    *,
    panel_runs: Sequence[TargetPowerStatsPanelRun],
    history_steps: int,
    forecast_steps: int,
    stride_steps: int,
    batch_size: int,
    covariate_specs: Sequence[tuple[str, str]],
    window_offset: int = 0,
    max_windows_per_dataset: int | None = None,
) -> Iterable[tuple[list[dict[str, object]], np.ndarray, np.ndarray]]:
    input_batch: list[dict[str, object]] = []
    future_rows: list[np.ndarray] = []
    future_timestamps: list[np.ndarray] = []
    emitted_windows = 0
    skipped_windows = 0
    max_windows_reached = False

    def flush_batch() -> tuple[list[dict[str, object]], np.ndarray, np.ndarray] | None:
        nonlocal input_batch, future_rows, future_timestamps
        if not input_batch:
            return None
        batch = (
            input_batch,
            np.stack(future_rows),
            np.stack(future_timestamps),
        )
        input_batch = []
        future_rows = []
        future_timestamps = []
        return batch

    for panel_run in panel_runs:
        farm_panel = panel_run.farm_panel
        max_anchor_index = farm_panel.timestamps_us.shape[0] - forecast_steps
        for anchor_index in range(history_steps - 1, max_anchor_index, stride_steps):
            forecast_start_us = int(farm_panel.timestamps_us[anchor_index + 1])
            if panel_run.forecast_start_us_min is not None and forecast_start_us < panel_run.forecast_start_us_min:
                continue
            if panel_run.forecast_start_us_max is not None and forecast_start_us > panel_run.forecast_start_us_max:
                continue

            context = farm_panel.target_kw_masked[anchor_index - history_steps + 1 : anchor_index + 1].T
            future = farm_panel.target_kw_masked[anchor_index + 1 : anchor_index + 1 + forecast_steps].T
            if context.shape != (len(farm_panel.turbine_ids), history_steps):
                continue
            if future.shape != (len(farm_panel.turbine_ids), forecast_steps):
                continue
            if (
                np.isnan(context).all()
                or np.isnan(future).all()
                or np.isnan(future[panel_run.scored_row_index]).all()
            ):
                continue
            if skipped_windows < window_offset:
                skipped_windows += 1
                continue

            input_batch.append(
                {
                    "target": context.astype(np.float32, copy=True),
                    "past_covariates": build_flattened_power_stats_covariates(
                        farm_panel,
                        anchor_index=anchor_index,
                        history_steps=history_steps,
                        covariate_specs=covariate_specs,
                    ),
                }
            )
            future_rows.append(future.astype(np.float64, copy=True))
            future_timestamps.append(
                farm_panel.timestamps_us[anchor_index + 1 : anchor_index + 1 + forecast_steps].astype(np.int64, copy=True)
            )
            emitted_windows += 1

            if len(input_batch) >= batch_size:
                batch = flush_batch()
                if batch is not None:
                    yield batch

            if max_windows_per_dataset is not None and emitted_windows >= max_windows_per_dataset:
                max_windows_reached = True
                break

        batch = flush_batch()
        if batch is not None:
            yield batch
        if max_windows_reached:
            break


def _count_retained_univariate_windows(
    *,
    timestamps_us: np.ndarray,
    target_kw_masked: np.ndarray,
    history_steps: int,
    forecast_steps: int,
    stride_steps: int,
    window_offset: int = 0,
    max_windows_per_dataset: int | None = None,
) -> int:
    retained_windows = 0
    skipped_windows = 0
    emitted_windows = 0
    max_anchor_index = timestamps_us.shape[0] - forecast_steps

    for anchor_index in range(history_steps - 1, max_anchor_index, stride_steps):
        context = target_kw_masked[anchor_index - history_steps + 1 : anchor_index + 1]
        future = target_kw_masked[anchor_index + 1 : anchor_index + 1 + forecast_steps]
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


def _count_univariate_batches_for_turbine(
    *,
    turbine_series: TurbineSeries | TurbinePowerStatsSeries,
    history_steps: int,
    forecast_steps: int,
    stride_steps: int,
    batch_size: int,
    window_offset: int = 0,
    max_windows_per_dataset: int | None = None,
) -> int:
    retained_windows = _count_retained_univariate_windows(
        timestamps_us=turbine_series.timestamps_us,
        target_kw_masked=turbine_series.target_kw_masked,
        history_steps=history_steps,
        forecast_steps=forecast_steps,
        stride_steps=stride_steps,
        window_offset=window_offset,
        max_windows_per_dataset=max_windows_per_dataset,
    )
    if retained_windows == 0:
        return 0
    return (retained_windows + batch_size - 1) // batch_size


def _count_multivariate_batches_for_runs(
    *,
    panel_runs: Sequence[TargetPanelRun | TargetPowerStatsPanelRun],
    history_steps: int,
    forecast_steps: int,
    stride_steps: int,
    batch_size: int,
    window_offset: int = 0,
    max_windows_per_dataset: int | None = None,
) -> int:
    batch_count = 0
    buffered_windows = 0
    emitted_windows = 0
    skipped_windows = 0
    max_windows_reached = False

    def flush_batch() -> None:
        nonlocal batch_count, buffered_windows
        if buffered_windows > 0:
            batch_count += 1
            buffered_windows = 0

    for panel_run in panel_runs:
        farm_panel = panel_run.farm_panel
        max_anchor_index = farm_panel.timestamps_us.shape[0] - forecast_steps
        for anchor_index in range(history_steps - 1, max_anchor_index, stride_steps):
            forecast_start_us = int(farm_panel.timestamps_us[anchor_index + 1])
            if panel_run.forecast_start_us_min is not None and forecast_start_us < panel_run.forecast_start_us_min:
                continue
            if panel_run.forecast_start_us_max is not None and forecast_start_us > panel_run.forecast_start_us_max:
                continue

            context = farm_panel.target_kw_masked[anchor_index - history_steps + 1 : anchor_index + 1].T
            future = farm_panel.target_kw_masked[anchor_index + 1 : anchor_index + 1 + forecast_steps].T
            if context.shape != (len(farm_panel.turbine_ids), history_steps):
                continue
            if future.shape != (len(farm_panel.turbine_ids), forecast_steps):
                continue
            if (
                np.isnan(context).all()
                or np.isnan(future).all()
                or np.isnan(future[panel_run.scored_row_index]).all()
            ):
                continue
            if skipped_windows < window_offset:
                skipped_windows += 1
                continue

            buffered_windows += 1
            emitted_windows += 1
            if buffered_windows >= batch_size:
                batch_count += 1
                buffered_windows = 0
            if max_windows_per_dataset is not None and emitted_windows >= max_windows_per_dataset:
                max_windows_reached = True
                break

        flush_batch()
        if max_windows_reached:
            break

    return batch_count


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
    window_protocol: str,
) -> dict[str, object]:
    prediction_count = int(metrics["prediction_count"])
    return {
        "dataset_id": f"{dataset_id}{suffix}",
        "model_id": MODEL_ID,
        "task_id": resolved_task.task_id,
        "window_protocol": window_protocol,
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
    max_windows_per_split: int | None = None,
    device: str | None = None,
    turbine_ids: Sequence[str] | None = None,
    eval_protocols: Sequence[str] | None = None,
    emit_progress_events: bool = False,
    progress_state: ChunkProgressState | None = None,
) -> list[dict[str, object]]:
    dataset_start = time.monotonic()
    spec, resolved_task = resolve_dataset_task(dataset_id, task_spec=task_spec)
    selected_turbine_ids = resolve_selected_turbine_ids(spec, turbine_ids)
    selected_eval_protocols = resolve_selected_eval_protocols(eval_protocols)

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
    split_windows = prepare_split_window_set(
        spec.dataset_id,
        cache_root=cache_root,
        raw_timestamps=prepared_series["timestamp"].to_list(),
        resolution_minutes=spec.resolution_minutes,
        history_steps=resolved_task.history_steps,
        forecast_steps=resolved_task.forecast_steps,
        turbine_ids=selected_turbine_ids,
        timestamps_by_turbine=build_timestamps_by_turbine(turbine_series_map),
    )
    evaluation_window_specs = iter_evaluation_windows(
        split_windows,
        eval_protocols=selected_eval_protocols,
        window_offset=window_offset,
        max_windows_per_split=max_windows_per_split,
    )
    ordered_turbine_series = tuple(turbine_series_map[turbine_id] for turbine_id in selected_turbine_ids)
    stage_total_batches = sum(
        _count_batches_from_windows(windows, batch_size)
        for _, windows in evaluation_window_specs
    )
    evaluation_results: list[tuple[str, WindowDescriptorIndex, EvaluationMetrics]] = []
    if emit_progress_events:
        if progress_state is None:
            raise ValueError("emit_progress_events requires progress_state.")
        progress_state.chunk_total_batches += stage_total_batches
        _emit_progress_event(
            spec.dataset_id,
            enabled=True,
            phase="progress_stage_start",
            stage="univariate",
            stage_total_batches=stage_total_batches,
            completed_stage_batches=0,
            completed_chunk_batches=progress_state.completed_chunk_batches,
            chunk_total_batches=progress_state.chunk_total_batches,
        )
    stage_completed_batches = 0

    for eval_protocol, windows in evaluation_window_specs:
        metrics = _initialize_metric_state(resolved_task.forecast_steps)
        for context_batch, actual_batch, _future_timestamps_batch in _iter_univariate_descriptor_batches(
            turbine_series=ordered_turbine_series,
            windows=windows,
            history_steps=resolved_task.history_steps,
            forecast_steps=resolved_task.forecast_steps,
            batch_size=batch_size,
        ):
            quantiles, _ = pipeline.predict_quantiles(
                inputs=context_batch,
                prediction_length=resolved_task.forecast_steps,
                quantile_levels=[0.5],
                batch_size=resolve_pipeline_batch_size(context_batch),
                limit_prediction_length=False,
            )
            if emit_progress_events:
                stage_completed_batches += 1
                progress_state.completed_chunk_batches += 1
                _emit_progress_event(
                    spec.dataset_id,
                    enabled=True,
                    phase="progress_batch",
                    stage="univariate",
                    eval_protocol=eval_protocol,
                    completed_stage_batches=stage_completed_batches,
                    stage_total_batches=stage_total_batches,
                    completed_chunk_batches=progress_state.completed_chunk_batches,
                    chunk_total_batches=progress_state.chunk_total_batches,
                )
            prediction_batch = np.stack([_extract_median_forecast(prediction)[0] for prediction in quantiles])
            _update_metric_state(
                metrics,
                prediction_batch=prediction_batch,
                actual_batch=actual_batch,
                rated_power_kw=rated_power_kw,
            )
        evaluation_results.append((eval_protocol, windows, finalize_metric_state(metrics)))

    if emit_progress_events:
        _emit_progress_event(
            spec.dataset_id,
            enabled=True,
            phase="progress_stage_complete",
            stage="univariate",
            completed_stage_batches=stage_completed_batches,
            stage_total_batches=stage_total_batches,
            completed_chunk_batches=progress_state.completed_chunk_batches,
            chunk_total_batches=progress_state.chunk_total_batches,
        )

    runtime_seconds = time.monotonic() - dataset_start
    _profile_log(
        spec.dataset_id,
        "evaluate_univariate_complete",
        target_turbines=len(selected_turbine_ids),
        input_turbines=len(selected_turbine_ids),
        runtime_seconds=round(runtime_seconds, 6),
        test_window_count=split_windows.test_window_count,
    )
    return build_result_rows(
        dataset_id=spec.dataset_id,
        suffix=UNIVARIATE_SUFFIX,
        resolved_task=resolved_task,
        resolution_minutes=spec.resolution_minutes,
        split_windows=split_windows,
        runtime_seconds=runtime_seconds,
        device=device or select_device(),
        evaluation_results=evaluation_results,
    )


def evaluate_univariate_power_stats_dataset(
    dataset_id: str,
    *,
    pipeline: Any,
    cache_root: str | Path = _CACHE_ROOT,
    task_spec=None,
    batch_size: int = DEFAULT_BATCH_SIZE,
    window_offset: int = 0,
    max_windows_per_split: int | None = None,
    device: str | None = None,
    turbine_ids: Sequence[str] | None = None,
    eval_protocols: Sequence[str] | None = None,
    emit_progress_events: bool = False,
    progress_state: ChunkProgressState | None = None,
) -> list[dict[str, object]]:
    dataset_start = time.monotonic()
    spec, resolved_task = resolve_dataset_task(dataset_id, task_spec=task_spec)
    covariate_columns = resolve_power_stats_covariate_columns(spec.dataset_id)
    selected_turbine_ids = resolve_selected_turbine_ids(spec, turbine_ids)
    selected_eval_protocols = resolve_selected_eval_protocols(eval_protocols)

    spec, resolved_task, series = load_dataset_inputs(
        dataset_id,
        cache_root=cache_root,
        task_spec=task_spec,
        turbine_ids=selected_turbine_ids,
        include_power_stats=True,
    )
    rated_power_kw = resolve_rated_power_kw(spec.dataset_id)
    prepare_started = time.monotonic()
    prepared_series = prepare_power_stats_series(
        series,
        dataset_id=spec.dataset_id,
        rated_power_kw=rated_power_kw,
    )
    _profile_log(
        spec.dataset_id,
        "prepare_power_stats_series",
        rows=prepared_series.height,
        columns=prepared_series.width,
        target_turbines=len(selected_turbine_ids),
        input_turbines=len(selected_turbine_ids),
        covariates=len(covariate_columns),
        duration_seconds=round(time.monotonic() - prepare_started, 6),
    )
    turbine_series_map = build_turbine_power_stats_series_map(
        prepared_series,
        covariate_columns=covariate_columns,
    )
    split_windows = prepare_split_window_set(
        spec.dataset_id,
        cache_root=cache_root,
        raw_timestamps=prepared_series["timestamp"].to_list(),
        resolution_minutes=spec.resolution_minutes,
        history_steps=resolved_task.history_steps,
        forecast_steps=resolved_task.forecast_steps,
        turbine_ids=selected_turbine_ids,
        timestamps_by_turbine=build_timestamps_by_turbine(turbine_series_map),
    )
    evaluation_window_specs = iter_evaluation_windows(
        split_windows,
        eval_protocols=selected_eval_protocols,
        window_offset=window_offset,
        max_windows_per_split=max_windows_per_split,
    )
    ordered_turbine_series = tuple(turbine_series_map[turbine_id] for turbine_id in selected_turbine_ids)
    stage_total_batches = sum(
        _count_batches_from_windows(windows, batch_size)
        for _, windows in evaluation_window_specs
    )
    evaluation_results: list[tuple[str, WindowDescriptorIndex, EvaluationMetrics]] = []
    if emit_progress_events:
        if progress_state is None:
            raise ValueError("emit_progress_events requires progress_state.")
        progress_state.chunk_total_batches += stage_total_batches
        _emit_progress_event(
            spec.dataset_id,
            enabled=True,
            phase="progress_stage_start",
            stage="univariate_power_stats",
            covariates=len(covariate_columns),
            stage_total_batches=stage_total_batches,
            completed_stage_batches=0,
            completed_chunk_batches=progress_state.completed_chunk_batches,
            chunk_total_batches=progress_state.chunk_total_batches,
        )
    stage_completed_batches = 0

    for eval_protocol, windows in evaluation_window_specs:
        metrics = _initialize_metric_state(resolved_task.forecast_steps)
        for input_batch, actual_batch, _future_timestamps_batch in _iter_univariate_power_stats_descriptor_batches(
            turbine_series=ordered_turbine_series,
            windows=windows,
            history_steps=resolved_task.history_steps,
            forecast_steps=resolved_task.forecast_steps,
            batch_size=batch_size,
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
                    stage="univariate_power_stats",
                    eval_protocol=eval_protocol,
                    covariates=len(covariate_columns),
                    completed_stage_batches=stage_completed_batches,
                    stage_total_batches=stage_total_batches,
                    completed_chunk_batches=progress_state.completed_chunk_batches,
                    chunk_total_batches=progress_state.chunk_total_batches,
                )
            prediction_batch = np.stack([_extract_median_forecast(prediction)[0] for prediction in quantiles])
            _update_metric_state(
                metrics,
                prediction_batch=prediction_batch,
                actual_batch=actual_batch,
                rated_power_kw=rated_power_kw,
            )
        evaluation_results.append((eval_protocol, windows, finalize_metric_state(metrics)))

    if emit_progress_events:
        _emit_progress_event(
            spec.dataset_id,
            enabled=True,
            phase="progress_stage_complete",
            stage="univariate_power_stats",
            covariates=len(covariate_columns),
            completed_stage_batches=stage_completed_batches,
            stage_total_batches=stage_total_batches,
            completed_chunk_batches=progress_state.completed_chunk_batches,
            chunk_total_batches=progress_state.chunk_total_batches,
        )

    runtime_seconds = time.monotonic() - dataset_start
    _profile_log(
        spec.dataset_id,
        "evaluate_univariate_power_stats_complete",
        target_turbines=len(selected_turbine_ids),
        input_turbines=len(selected_turbine_ids),
        runtime_seconds=round(runtime_seconds, 6),
        test_window_count=split_windows.test_window_count,
    )
    return build_result_rows(
        dataset_id=spec.dataset_id,
        suffix=UNIVARIATE_POWER_STATS_SUFFIX,
        resolved_task=resolved_task,
        resolution_minutes=spec.resolution_minutes,
        split_windows=split_windows,
        runtime_seconds=runtime_seconds,
        device=device or select_device(),
        evaluation_results=evaluation_results,
    )


def evaluate_multivariate_knn6_dataset(
    dataset_id: str,
    *,
    pipeline: Any,
    cache_root: str | Path = _CACHE_ROOT,
    task_spec=None,
    batch_size: int = DEFAULT_BATCH_SIZE,
    window_offset: int = 0,
    max_windows_per_split: int | None = None,
    device: str | None = None,
    turbine_ids: Sequence[str] | None = None,
    eval_protocols: Sequence[str] | None = None,
    emit_progress_events: bool = False,
    progress_state: ChunkProgressState | None = None,
) -> list[dict[str, object]]:
    dataset_start = time.monotonic()
    spec, resolved_task = resolve_dataset_task(dataset_id, task_spec=task_spec)
    target_turbine_ids = resolve_selected_turbine_ids(spec, turbine_ids)
    selected_eval_protocols = resolve_selected_eval_protocols(eval_protocols)
    _, _, target_series = load_dataset_inputs(
        dataset_id,
        cache_root=cache_root,
        task_spec=task_spec,
        turbine_ids=target_turbine_ids,
    )
    rated_power_kw = resolve_rated_power_kw(spec.dataset_id)
    prepared_target_series = prepare_power_only_series(target_series, rated_power_kw=rated_power_kw)
    target_turbine_series_map = build_turbine_series_map(prepared_target_series)
    split_windows = prepare_split_window_set(
        spec.dataset_id,
        cache_root=cache_root,
        raw_timestamps=prepared_target_series["timestamp"].to_list(),
        resolution_minutes=spec.resolution_minutes,
        history_steps=resolved_task.history_steps,
        forecast_steps=resolved_task.forecast_steps,
        turbine_ids=target_turbine_ids,
        timestamps_by_turbine=build_timestamps_by_turbine(target_turbine_series_map),
    )
    epoch_configs = resolve_multivariate_epoch_configs(
        spec,
        target_turbine_ids=target_turbine_ids,
    )
    turbine_static = load_dataset_turbine_static(spec.dataset_id, cache_root=cache_root)
    epoch_contexts: list[PreparedMultivariateEpoch] = []
    all_required_input_turbine_ids: list[str] = []

    for epoch_config in epoch_configs:
        neighbor_map = build_epoch_knn_neighbor_map(
            turbine_static,
            active_turbine_ids=epoch_config.active_turbine_ids,
        )
        required_input_turbine_ids = build_epoch_required_input_turbine_ids(
            epoch_config.target_turbine_ids,
            neighbor_map,
        )
        all_required_input_turbine_ids.extend(required_input_turbine_ids)
        _profile_log(
            spec.dataset_id,
            "multivariate_epoch_scope",
            epoch_name=epoch_config.name,
            target_turbines=len(epoch_config.target_turbine_ids),
            input_turbines=len(required_input_turbine_ids),
            forecast_start_us_min=epoch_config.forecast_start_us_min,
            forecast_start_us_max=epoch_config.forecast_start_us_max,
        )
        _, _, series = load_dataset_inputs(
            dataset_id,
            cache_root=cache_root,
            task_spec=task_spec,
            turbine_ids=required_input_turbine_ids,
        )
        prepare_started = time.monotonic()
        prepared_series = prepare_power_only_series(series, rated_power_kw=rated_power_kw)
        _profile_log(
            spec.dataset_id,
            "prepare_power_only_series",
            epoch_name=epoch_config.name,
            rows=prepared_series.height,
            columns=prepared_series.width,
            target_turbines=len(epoch_config.target_turbine_ids),
            input_turbines=len(required_input_turbine_ids),
            duration_seconds=round(time.monotonic() - prepare_started, 6),
        )
        epoch_contexts.append(
            PreparedMultivariateEpoch(
                config=epoch_config,
                required_input_turbine_ids=required_input_turbine_ids,
                neighbor_map=neighbor_map,
                turbine_series_map=build_turbine_series_map(prepared_series),
            )
        )

    panel_runs_by_target: dict[str, list[TargetPanelRun]] = {}
    for target_turbine_id in target_turbine_ids:
        panel_runs: list[TargetPanelRun] = []
        for epoch_context in epoch_contexts:
            if target_turbine_id not in epoch_context.config.target_turbine_ids:
                continue
            panel_started = time.monotonic()
            local_panel = build_local_panel(
                epoch_context.turbine_series_map,
                turbine_ids=epoch_context.neighbor_map[target_turbine_id],
                resolution_minutes=spec.resolution_minutes,
            )
            _profile_log(
                spec.dataset_id,
                "build_local_panel",
                epoch_name=epoch_context.config.name,
                target_turbine_id=target_turbine_id,
                input_turbines=len(epoch_context.neighbor_map[target_turbine_id]),
                timestamps=len(local_panel.timestamps_us),
                rows=int(local_panel.target_kw_masked.shape[0]),
                columns=int(local_panel.target_kw_masked.shape[1]),
                duration_seconds=round(time.monotonic() - panel_started, 6),
            )
            panel_runs.append(
                TargetPanelRun(
                    name=epoch_context.config.name,
                    farm_panel=local_panel,
                    scored_row_index=0,
                    forecast_start_us_min=epoch_context.config.forecast_start_us_min,
                    forecast_start_us_max=epoch_context.config.forecast_start_us_max,
                    timestamp_index_map={
                        int(timestamp): index
                        for index, timestamp in enumerate(local_panel.timestamps_us.tolist())
                    },
                )
            )
        panel_runs_by_target[target_turbine_id] = panel_runs

    evaluation_window_specs = iter_evaluation_windows(
        split_windows,
        eval_protocols=selected_eval_protocols,
        window_offset=window_offset,
        max_windows_per_split=max_windows_per_split,
    )
    stage_total_batches = sum(
        _count_batches_from_windows(windows, batch_size)
        for _, windows in evaluation_window_specs
    )
    evaluation_results: list[tuple[str, WindowDescriptorIndex, EvaluationMetrics]] = []
    if emit_progress_events:
        if progress_state is None:
            raise ValueError("emit_progress_events requires progress_state.")
        progress_state.chunk_total_batches += stage_total_batches
        _emit_progress_event(
            spec.dataset_id,
            enabled=True,
            phase="progress_stage_start",
            stage="multivariate_knn6",
            stage_total_batches=stage_total_batches,
            completed_stage_batches=0,
            completed_chunk_batches=progress_state.completed_chunk_batches,
            chunk_total_batches=progress_state.chunk_total_batches,
        )
    stage_completed_batches = 0

    for eval_protocol, windows in evaluation_window_specs:
        metrics = _initialize_metric_state(resolved_task.forecast_steps)
        for context_batch, actual_batch, _future_timestamps_batch in _iter_selected_multivariate_batches(
            panel_runs_by_target=panel_runs_by_target,
            target_turbine_ids=target_turbine_ids,
            windows=windows,
            history_steps=resolved_task.history_steps,
            forecast_steps=resolved_task.forecast_steps,
            batch_size=batch_size,
        ):
            quantiles, _ = pipeline.predict_quantiles(
                inputs=context_batch,
                prediction_length=resolved_task.forecast_steps,
                quantile_levels=[0.5],
                batch_size=resolve_pipeline_batch_size(context_batch),
                limit_prediction_length=False,
                cross_learning=False,
            )
            if emit_progress_events:
                stage_completed_batches += 1
                progress_state.completed_chunk_batches += 1
                _emit_progress_event(
                    spec.dataset_id,
                    enabled=True,
                    phase="progress_batch",
                    stage="multivariate_knn6",
                    eval_protocol=eval_protocol,
                    completed_stage_batches=stage_completed_batches,
                    stage_total_batches=stage_total_batches,
                    completed_chunk_batches=progress_state.completed_chunk_batches,
                    chunk_total_batches=progress_state.chunk_total_batches,
                )
            prediction_batch = np.stack([_extract_median_forecast(prediction) for prediction in quantiles])
            target_prediction_batch = prediction_batch[:, 0, :]
            target_actual_batch = actual_batch[:, 0, :]
            _update_metric_state(
                metrics,
                prediction_batch=target_prediction_batch,
                actual_batch=target_actual_batch,
                rated_power_kw=rated_power_kw,
            )
        evaluation_results.append((eval_protocol, windows, finalize_metric_state(metrics)))

    if emit_progress_events:
        _emit_progress_event(
            spec.dataset_id,
            enabled=True,
            phase="progress_stage_complete",
            stage="multivariate_knn6",
            completed_stage_batches=stage_completed_batches,
            stage_total_batches=stage_total_batches,
            completed_chunk_batches=progress_state.completed_chunk_batches,
            chunk_total_batches=progress_state.chunk_total_batches,
        )

    runtime_seconds = time.monotonic() - dataset_start
    _profile_log(
        spec.dataset_id,
        "evaluate_multivariate_complete",
        target_turbines=len(target_turbine_ids),
        input_turbines=len(_ordered_unique(all_required_input_turbine_ids)),
        runtime_seconds=round(runtime_seconds, 6),
        test_window_count=split_windows.test_window_count,
    )
    return build_result_rows(
        dataset_id=spec.dataset_id,
        suffix=MULTIVARIATE_KNN6_SUFFIX,
        resolved_task=resolved_task,
        resolution_minutes=spec.resolution_minutes,
        split_windows=split_windows,
        runtime_seconds=runtime_seconds,
        device=device or select_device(),
        evaluation_results=evaluation_results,
    )


def evaluate_multivariate_knn6_power_stats_dataset(
    dataset_id: str,
    *,
    pipeline: Any,
    cache_root: str | Path = _CACHE_ROOT,
    task_spec=None,
    batch_size: int = DEFAULT_BATCH_SIZE,
    window_offset: int = 0,
    max_windows_per_split: int | None = None,
    device: str | None = None,
    turbine_ids: Sequence[str] | None = None,
    eval_protocols: Sequence[str] | None = None,
    emit_progress_events: bool = False,
    progress_state: ChunkProgressState | None = None,
) -> list[dict[str, object]]:
    dataset_start = time.monotonic()
    spec, resolved_task = resolve_dataset_task(dataset_id, task_spec=task_spec)
    covariate_specs = resolve_power_stats_covariate_specs(spec.dataset_id)
    covariate_columns = tuple(column for column, _ in covariate_specs)
    target_turbine_ids = resolve_selected_turbine_ids(spec, turbine_ids)
    selected_eval_protocols = resolve_selected_eval_protocols(eval_protocols)
    _, _, target_series = load_dataset_inputs(
        dataset_id,
        cache_root=cache_root,
        task_spec=task_spec,
        turbine_ids=target_turbine_ids,
    )
    rated_power_kw = resolve_rated_power_kw(spec.dataset_id)
    prepared_target_series = prepare_power_only_series(target_series, rated_power_kw=rated_power_kw)
    target_turbine_series_map = build_turbine_series_map(prepared_target_series)
    split_windows = prepare_split_window_set(
        spec.dataset_id,
        cache_root=cache_root,
        raw_timestamps=prepared_target_series["timestamp"].to_list(),
        resolution_minutes=spec.resolution_minutes,
        history_steps=resolved_task.history_steps,
        forecast_steps=resolved_task.forecast_steps,
        turbine_ids=target_turbine_ids,
        timestamps_by_turbine=build_timestamps_by_turbine(target_turbine_series_map),
    )
    epoch_configs = resolve_multivariate_epoch_configs(
        spec,
        target_turbine_ids=target_turbine_ids,
    )
    turbine_static = load_dataset_turbine_static(spec.dataset_id, cache_root=cache_root)
    epoch_contexts: list[PreparedMultivariatePowerStatsEpoch] = []
    all_required_input_turbine_ids: list[str] = []

    for epoch_config in epoch_configs:
        neighbor_map = build_epoch_knn_neighbor_map(
            turbine_static,
            active_turbine_ids=epoch_config.active_turbine_ids,
        )
        required_input_turbine_ids = build_epoch_required_input_turbine_ids(
            epoch_config.target_turbine_ids,
            neighbor_map,
        )
        all_required_input_turbine_ids.extend(required_input_turbine_ids)
        _profile_log(
            spec.dataset_id,
            "multivariate_power_stats_epoch_scope",
            epoch_name=epoch_config.name,
            target_turbines=len(epoch_config.target_turbine_ids),
            input_turbines=len(required_input_turbine_ids),
            covariates=len(covariate_columns),
            forecast_start_us_min=epoch_config.forecast_start_us_min,
            forecast_start_us_max=epoch_config.forecast_start_us_max,
        )
        _, _, series = load_dataset_inputs(
            dataset_id,
            cache_root=cache_root,
            task_spec=task_spec,
            turbine_ids=required_input_turbine_ids,
            include_power_stats=True,
        )
        prepare_started = time.monotonic()
        prepared_series = prepare_power_stats_series(
            series,
            dataset_id=spec.dataset_id,
            rated_power_kw=rated_power_kw,
        )
        _profile_log(
            spec.dataset_id,
            "prepare_multivariate_power_stats_series",
            epoch_name=epoch_config.name,
            rows=prepared_series.height,
            columns=prepared_series.width,
            target_turbines=len(epoch_config.target_turbine_ids),
            input_turbines=len(required_input_turbine_ids),
            covariates=len(covariate_columns),
            duration_seconds=round(time.monotonic() - prepare_started, 6),
        )
        epoch_contexts.append(
            PreparedMultivariatePowerStatsEpoch(
                config=epoch_config,
                required_input_turbine_ids=required_input_turbine_ids,
                neighbor_map=neighbor_map,
                turbine_series_map=build_turbine_power_stats_series_map(
                    prepared_series,
                    covariate_columns=covariate_columns,
                ),
            )
        )

    panel_runs_by_target: dict[str, list[TargetPowerStatsPanelRun]] = {}
    for target_turbine_id in target_turbine_ids:
        panel_runs: list[TargetPowerStatsPanelRun] = []
        for epoch_context in epoch_contexts:
            if target_turbine_id not in epoch_context.config.target_turbine_ids:
                continue
            panel_started = time.monotonic()
            local_panel = build_local_power_stats_panel(
                epoch_context.turbine_series_map,
                turbine_ids=epoch_context.neighbor_map[target_turbine_id],
                resolution_minutes=spec.resolution_minutes,
                covariate_columns=covariate_columns,
            )
            _profile_log(
                spec.dataset_id,
                "build_local_power_stats_panel",
                epoch_name=epoch_context.config.name,
                target_turbine_id=target_turbine_id,
                input_turbines=len(epoch_context.neighbor_map[target_turbine_id]),
                timestamps=len(local_panel.timestamps_us),
                rows=int(local_panel.target_kw_masked.shape[0]),
                columns=int(local_panel.target_kw_masked.shape[1]),
                covariates=len(covariate_columns) * len(local_panel.turbine_ids),
                duration_seconds=round(time.monotonic() - panel_started, 6),
            )
            panel_runs.append(
                TargetPowerStatsPanelRun(
                    name=epoch_context.config.name,
                    farm_panel=local_panel,
                    scored_row_index=0,
                    forecast_start_us_min=epoch_context.config.forecast_start_us_min,
                    forecast_start_us_max=epoch_context.config.forecast_start_us_max,
                    timestamp_index_map={
                        int(timestamp): index
                        for index, timestamp in enumerate(local_panel.timestamps_us.tolist())
                    },
                )
            )
        panel_runs_by_target[target_turbine_id] = panel_runs

    evaluation_window_specs = iter_evaluation_windows(
        split_windows,
        eval_protocols=selected_eval_protocols,
        window_offset=window_offset,
        max_windows_per_split=max_windows_per_split,
    )
    stage_total_batches = sum(
        _count_batches_from_windows(windows, batch_size)
        for _, windows in evaluation_window_specs
    )
    evaluation_results: list[tuple[str, WindowDescriptorIndex, EvaluationMetrics]] = []
    if emit_progress_events:
        if progress_state is None:
            raise ValueError("emit_progress_events requires progress_state.")
        progress_state.chunk_total_batches += stage_total_batches
        _emit_progress_event(
            spec.dataset_id,
            enabled=True,
            phase="progress_stage_start",
            stage="multivariate_knn6_power_stats",
            covariates=len(covariate_columns),
            stage_total_batches=stage_total_batches,
            completed_stage_batches=0,
            completed_chunk_batches=progress_state.completed_chunk_batches,
            chunk_total_batches=progress_state.chunk_total_batches,
        )
    stage_completed_batches = 0

    for eval_protocol, windows in evaluation_window_specs:
        metrics = _initialize_metric_state(resolved_task.forecast_steps)
        for input_batch, actual_batch, _future_timestamps_batch in _iter_selected_multivariate_power_stats_batches(
            panel_runs_by_target=panel_runs_by_target,
            target_turbine_ids=target_turbine_ids,
            windows=windows,
            history_steps=resolved_task.history_steps,
            forecast_steps=resolved_task.forecast_steps,
            batch_size=batch_size,
            covariate_specs=covariate_specs,
        ):
            quantiles, _ = pipeline.predict_quantiles(
                inputs=input_batch,
                prediction_length=resolved_task.forecast_steps,
                quantile_levels=[0.5],
                batch_size=resolve_pipeline_batch_size(input_batch),
                limit_prediction_length=False,
                cross_learning=False,
            )
            if emit_progress_events:
                stage_completed_batches += 1
                progress_state.completed_chunk_batches += 1
                _emit_progress_event(
                    spec.dataset_id,
                    enabled=True,
                    phase="progress_batch",
                    stage="multivariate_knn6_power_stats",
                    eval_protocol=eval_protocol,
                    covariates=len(covariate_columns),
                    completed_stage_batches=stage_completed_batches,
                    stage_total_batches=stage_total_batches,
                    completed_chunk_batches=progress_state.completed_chunk_batches,
                    chunk_total_batches=progress_state.chunk_total_batches,
                )
            prediction_batch = np.stack([_extract_median_forecast(prediction) for prediction in quantiles])
            target_prediction_batch = prediction_batch[:, 0, :]
            target_actual_batch = actual_batch[:, 0, :]
            _update_metric_state(
                metrics,
                prediction_batch=target_prediction_batch,
                actual_batch=target_actual_batch,
                rated_power_kw=rated_power_kw,
            )
        evaluation_results.append((eval_protocol, windows, finalize_metric_state(metrics)))

    if emit_progress_events:
        _emit_progress_event(
            spec.dataset_id,
            enabled=True,
            phase="progress_stage_complete",
            stage="multivariate_knn6_power_stats",
            covariates=len(covariate_columns),
            completed_stage_batches=stage_completed_batches,
            stage_total_batches=stage_total_batches,
            completed_chunk_batches=progress_state.completed_chunk_batches,
            chunk_total_batches=progress_state.chunk_total_batches,
        )

    runtime_seconds = time.monotonic() - dataset_start
    _profile_log(
        spec.dataset_id,
        "evaluate_multivariate_power_stats_complete",
        target_turbines=len(target_turbine_ids),
        input_turbines=len(_ordered_unique(all_required_input_turbine_ids)),
        covariates=len(covariate_columns),
        runtime_seconds=round(runtime_seconds, 6),
        test_window_count=split_windows.test_window_count,
    )
    return build_result_rows(
        dataset_id=spec.dataset_id,
        suffix=MULTIVARIATE_KNN6_POWER_STATS_SUFFIX,
        resolved_task=resolved_task,
        resolution_minutes=spec.resolution_minutes,
        split_windows=split_windows,
        runtime_seconds=runtime_seconds,
        device=device or select_device(),
        evaluation_results=evaluation_results,
    )


def run_experiment(
    *,
    dataset_ids: Sequence[str] = DEFAULT_DATASETS,
    cache_root: str | Path = _CACHE_ROOT,
    output_path: str | Path | None = None,
    batch_size: int = DEFAULT_BATCH_SIZE,
    window_offset: int = 0,
    max_windows_per_split: int | None = None,
    device: str | None = None,
    task_spec=None,
    pipeline: Any | None = None,
    mode: str = "all",
    turbine_ids: Sequence[str] | None = None,
    eval_protocols: Sequence[str] | None = None,
    emit_progress_events: bool = False,
) -> pl.DataFrame:
    if mode not in MODE_CHOICES:
        raise ValueError(f"Unsupported mode {mode!r}. Expected one of {MODE_CHOICES}.")

    power_stats_dataset_ids = tuple(dataset_id for dataset_id in dataset_ids if supports_power_stats(dataset_id))
    if mode == "univariate_power_stats" and not power_stats_dataset_ids:
        raise ValueError(
            "Mode 'univariate_power_stats' requires at least one dataset with power_stats support."
        )

    selected_eval_protocols = resolve_selected_eval_protocols(eval_protocols)
    resolved_task_spec = task_spec or build_task_spec(window_protocol=DEFAULT_WINDOW_PROTOCOL)
    resolved_device = device or select_device()
    resolved_pipeline = pipeline or load_pipeline(device=resolved_device)
    rows: list[dict[str, object]] = []
    progress_state = ChunkProgressState() if emit_progress_events else None

    def _run_evaluator(evaluator, active_dataset_ids: Sequence[str]) -> None:
        for dataset_id in active_dataset_ids:
            rows.extend(
                evaluator(
                dataset_id,
                pipeline=resolved_pipeline,
                cache_root=cache_root,
                task_spec=resolved_task_spec,
                batch_size=batch_size,
                window_offset=window_offset,
                max_windows_per_split=max_windows_per_split,
                device=resolved_device,
                turbine_ids=turbine_ids,
                eval_protocols=selected_eval_protocols,
                emit_progress_events=emit_progress_events,
                progress_state=progress_state,
            )
            )

    if mode in {"all", "univariate"}:
        _run_evaluator(evaluate_univariate_dataset, dataset_ids)
    if mode in {"all", "univariate", "univariate_power_stats"}:
        _run_evaluator(evaluate_univariate_power_stats_dataset, power_stats_dataset_ids)
    if mode in {"all", "multivariate_knn6"}:
        _run_evaluator(evaluate_multivariate_knn6_dataset, dataset_ids)
        _run_evaluator(evaluate_multivariate_knn6_power_stats_dataset, power_stats_dataset_ids)

    results = sort_result_frame(pl.DataFrame(rows).select(_RESULT_COLUMNS))
    output = Path(output_path) if output_path is not None else default_output_path(window_protocol=DEFAULT_WINDOW_PROTOCOL)
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
        default=None,
        help="Output CSV path. Defaults to a protocol-specific path under experiment/.",
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
        "--max-windows-per-split",
        type=int,
        default=None,
        help="Optional smoke-test limit on the number of retained test windows evaluated per protocol slice.",
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
        help=(
            "Run all experiments, or only the univariate, multivariate_knn6 "
            "(plain + supported power_stats variants), or univariate_power_stats subset."
        ),
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
        help="Emit structured progress events on stderr for full-run orchestration progress bars.",
    )
    parser.add_argument(
        "--eval-protocol",
        action="append",
        choices=list(EVAL_PROTOCOL_CHOICES),
        dest="eval_protocols",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--run-label",
        type=str,
        default=None,
        help="Optional label suffix for the formal run record under experiment/artifacts/runs/chronos2_power_only/.",
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
    selected_dataset_ids = tuple(args.datasets) if args.datasets else DEFAULT_DATASETS
    selected_eval_protocols = tuple(args.eval_protocols) if args.eval_protocols else EVAL_PROTOCOL_CHOICES
    results = run_experiment(
        dataset_ids=selected_dataset_ids,
        cache_root=args.cache_root,
        output_path=args.output_path,
        batch_size=args.batch_size,
        window_offset=args.window_offset,
        max_windows_per_split=args.max_windows_per_split,
        device=args.device,
        mode=args.mode,
        turbine_ids=tuple(args.turbine_ids) if args.turbine_ids else None,
        eval_protocols=selected_eval_protocols,
        emit_progress_events=bool(args.emit_progress_events),
    )
    if not args.no_record_run:
        include_power_stats = False
        if args.mode == "univariate_power_stats":
            include_power_stats = True
        elif args.mode in {"all", "univariate", "multivariate_knn6"}:
            include_power_stats = any(supports_power_stats(dataset_id) for dataset_id in selected_dataset_ids)
        feature_protocol_ids = ("power_only", "power_stats_history") if include_power_stats else ("power_only",)
        record_cli_run(
            family_id=FAMILY_ID,
            repo_root=_REPO_ROOT,
            invocation_kind="family_runner",
            entrypoint="experiment/families/chronos-2/run_power_only.py",
            args=vars(args),
            output_path=args.output_path or default_output_path(window_protocol=DEFAULT_WINDOW_PROTOCOL),
            result_row_count=results.height,
            dataset_ids=selected_dataset_ids,
            feature_protocol_ids=feature_protocol_ids,
            eval_protocols=selected_eval_protocols,
            result_splits=("test",),
            model_variants=(args.mode,),
            run_label=args.run_label,
        )
    print(results)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
