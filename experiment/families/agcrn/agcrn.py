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
from typing import Any, Callable, Sequence

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
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, Dataset
except ImportError:  # pragma: no cover - exercised in the root env where torch is absent
    torch = None
    nn = None
    F = None
    DataLoader = None
    Dataset = None


EXPERIMENT_DIR = Path(__file__).resolve().parent
EXPERIMENT_ROOT = EXPERIMENT_DIR.parents[1]
COMMON_DIR = EXPERIMENT_ROOT / "infra" / "common"
if str(COMMON_DIR) not in sys.path:
    sys.path.insert(0, str(COMMON_DIR))

import window_protocols  # noqa: E402
from run_records import record_cli_run  # noqa: E402
from published_outputs import default_family_output_path  # noqa: E402
from window_protocols import (  # noqa: E402
    DEFAULT_WINDOW_PROTOCOL,
    HORIZON_METRIC_SCOPE,
    NON_OVERLAP_EVAL_PROTOCOL,
    OVERALL_METRIC_SCOPE,
    ROLLING_EVAL_PROTOCOL,
    SPLIT_PROTOCOL,
    WindowProtocolSpec,
    resolve_window_protocol,
)


MODEL_ID = "AGCRN"
FAMILY_ID = "agcrn_official_aligned"
MODEL_VARIANT = "official_aligned_power_only_farm_sync"
WINDOW_PROTOCOL = DEFAULT_WINDOW_PROTOCOL
TASK_PROTOCOL: WindowProtocolSpec = resolve_window_protocol(WINDOW_PROTOCOL)
TASK_ID = TASK_PROTOCOL.task_id
DEFAULT_DATASETS = ("kelmarsh",)
HISTORY_STEPS = 144
FORECAST_STEPS = 36
STRIDE_STEPS = 1
FEATURE_PROTOCOL_ID = "power_only"
GRAPH_MODE = "adaptive"
GRAPH_SOURCE = "learned_node_embeddings"
INPUT_CHANNELS = 1
OUTPUT_CHANNELS = 1
DEFAULT_BATCH_SIZE = 1024
DEFAULT_LEARNING_RATE = 1e-3
DEFAULT_MAX_EPOCHS = 15
DEFAULT_EARLY_STOPPING_PATIENCE = 4
DEFAULT_SEED = 42
DEFAULT_HIDDEN_DIM = 64
DEFAULT_EMBED_DIM = 10
DEFAULT_NUM_LAYERS = 2
DEFAULT_CHEB_K = 2
DEFAULT_GRAD_CLIP_NORM = 5.0
PROFILE_LOG_PREFIX = "[agcrn] "

_REPO_ROOT = EXPERIMENT_ROOT.parent
_CACHE_ROOT = _REPO_ROOT / "cache"
_OUTPUT_PATH = default_family_output_path(repo_root=_REPO_ROOT, family_id=FAMILY_ID)
_TASK_WINDOW_COLUMNS = (
    "dataset",
    "output_start_ts",
    "output_end_ts",
    "is_complete_input",
    "is_complete_output",
    "is_fully_synchronous_input",
    "is_fully_synchronous_output",
    "quality_flags",
)
_SERIES_BASE_COLUMNS = (
    "dataset",
    "turbine_id",
    "timestamp",
    "target_kw",
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
    "window_count",
    "prediction_count",
    "start_timestamp",
    "end_timestamp",
    "mae_kw",
    "rmse_kw",
    "mae_pu",
    "rmse_pu",
    "graph_mode",
    "graph_source",
    "coordinate_mode",
    "node_count",
    "input_channels",
    "hidden_dim",
    "embed_dim",
    "num_layers",
    "cheb_k",
    "grad_clip_norm",
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
_SPLIT_ORDER = {"val": 0, "test": 1}
_EVAL_PROTOCOL_ORDER = {ROLLING_EVAL_PROTOCOL: 0, NON_OVERLAP_EVAL_PROTOCOL: 1}
_METRIC_SCOPE_ORDER = {OVERALL_METRIC_SCOPE: 0, HORIZON_METRIC_SCOPE: 1}


@dataclass(frozen=True)
class TaskCachePaths:
    dataset_id: str
    dataset_root: Path
    task_dir: Path
    series_path: Path
    window_index_path: Path
    task_context_path: Path
    turbine_static_path: Path

    @property
    def dataset_turbine_static_path(self) -> Path:
        return self.dataset_root / "silver" / "meta" / "turbine_static.parquet"


@dataclass(frozen=True)
class DatasetMetadata:
    dataset_id: str
    turbine_ids: tuple[str, ...]
    rated_power_kw: float
    turbine_static: pl.DataFrame
    task_paths: TaskCachePaths


@dataclass(frozen=True)
class FarmWindowDescriptorIndex:
    target_indices: np.ndarray
    output_start_us: np.ndarray
    output_end_us: np.ndarray

    def __len__(self) -> int:
        return int(self.target_indices.shape[0])

    @classmethod
    def empty(cls) -> "FarmWindowDescriptorIndex":
        return cls(
            target_indices=np.empty((0,), dtype=np.int32),
            output_start_us=np.empty((0,), dtype=np.int64),
            output_end_us=np.empty((0,), dtype=np.int64),
        )


@dataclass(frozen=True)
class PreparedDataset:
    dataset_id: str
    resolution_minutes: int
    rated_power_kw: float
    history_steps: int
    forecast_steps: int
    stride_steps: int
    turbine_ids: tuple[str, ...]
    coordinate_mode: str
    node_count: int
    timestamps_us: np.ndarray
    target_pu: np.ndarray
    train_windows: FarmWindowDescriptorIndex
    val_rolling_windows: FarmWindowDescriptorIndex
    val_non_overlap_windows: FarmWindowDescriptorIndex
    test_rolling_windows: FarmWindowDescriptorIndex
    test_non_overlap_windows: FarmWindowDescriptorIndex


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
        granularity="farm",
    )


def require_torch() -> tuple[Any, Any, Any, Any, Any]:
    if torch is None or nn is None or F is None or DataLoader is None or Dataset is None:
        raise ImportError(
            "PyTorch is unavailable in the current environment. "
            "Create experiment/families/agcrn/.conda with ./create_env.sh."
        )
    return torch, nn, F, DataLoader, Dataset


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
    task_dir = dataset_root / "tasks" / TASK_ID / FEATURE_PROTOCOL_ID
    return TaskCachePaths(
        dataset_id=dataset_id,
        dataset_root=dataset_root,
        task_dir=task_dir,
        series_path=task_dir / "series.parquet",
        window_index_path=task_dir / "window_index.parquet",
        task_context_path=task_dir / "task_context.json",
        turbine_static_path=task_dir / "static.parquet",
    )


def _has_complete_static_columns(frame: pl.DataFrame, columns: Sequence[str]) -> bool:
    return set(columns).issubset(frame.columns) and all(frame[column].null_count() == 0 for column in columns)


def _supplement_task_static_metadata(
    task_static: pl.DataFrame,
    *,
    dataset_id: str,
    paths: TaskCachePaths,
) -> pl.DataFrame:
    has_rated_power = _has_complete_static_columns(task_static, ("rated_power_kw",))
    has_xy_coordinates = _has_complete_static_columns(task_static, ("coord_x", "coord_y"))
    has_latlon_coordinates = _has_complete_static_columns(task_static, ("latitude", "longitude"))
    if has_rated_power and (has_xy_coordinates or has_latlon_coordinates):
        return task_static

    dataset_static_path = paths.dataset_turbine_static_path
    if not dataset_static_path.exists():
        missing_parts = []
        if not has_rated_power:
            missing_parts.append("rated_power_kw")
        if not (has_xy_coordinates or has_latlon_coordinates):
            missing_parts.append("coordinates")
        raise ValueError(
            f"Task-local turbine_static for dataset {dataset_id!r} is missing required "
            f"{', '.join(missing_parts)} and dataset-level turbine static is unavailable at {dataset_static_path}."
        )

    dataset_static = pl.read_parquet(dataset_static_path)
    missing_join_keys = [column for column in ("dataset", "turbine_id") if column not in dataset_static.columns]
    if missing_join_keys:
        raise ValueError(
            f"Dataset-level turbine_static for dataset {dataset_id!r} is missing join keys {missing_join_keys!r}."
        )

    metadata_columns = [
        column
        for column in ("rated_power_kw", "coord_x", "coord_y", "latitude", "longitude")
        if column in dataset_static.columns
    ]
    if not metadata_columns:
        return task_static

    joined = task_static.join(
        dataset_static.select(
            [
                "dataset",
                "turbine_id",
                *(pl.col(column).alias(f"{column}__dataset") for column in metadata_columns),
            ]
        ),
        on=["dataset", "turbine_id"],
        how="left",
    )
    supplemented = joined.with_columns(
        [
            (
                pl.coalesce(pl.col(column), pl.col(f"{column}__dataset"))
                if column in task_static.columns
                else pl.col(f"{column}__dataset")
            ).alias(column)
            for column in metadata_columns
        ]
    )
    return supplemented.drop([f"{column}__dataset" for column in metadata_columns]).sort("turbine_index")


def ensure_task_cache(dataset_id: str, *, cache_root: str | Path = _CACHE_ROOT) -> TaskCachePaths:
    paths = resolve_cache_paths(dataset_id, cache_root=cache_root)
    required_paths = (
        paths.series_path,
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
        build_task_cache(
            dataset_id,
            build_task_spec(),
            cache_root=cache_root,
            feature_protocol_id=FEATURE_PROTOCOL_ID,
        )
    except Exception as exc:  # pragma: no cover - exercised only when cache is missing
        raise RuntimeError(
            f"Farm task cache for dataset {dataset_id!r} is missing and could not be rebuilt. "
            "Either prebuild the cache artifacts or configure wind_datasets.local.toml."
        ) from exc

    if not all(path.exists() for path in required_paths):
        raise RuntimeError(f"Dataset cache for {dataset_id!r} is incomplete after rebuild.")
    return paths


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

    turbine_static = pl.read_parquet(paths.turbine_static_path)
    if "turbine_index" not in turbine_static.columns:
        raise ValueError(f"Task-local turbine_static for dataset {dataset_id!r} is missing turbine_index.")
    ordered_static = turbine_static.sort("turbine_index")
    ordered_ids = tuple(ordered_static["turbine_id"].to_list())
    context_ids = tuple(task_context["turbine_ids"])
    if ordered_ids != context_ids:
        raise ValueError(
            f"Task-local turbine_static order for dataset {dataset_id!r} does not match task_context turbine_ids."
        )
    ordered_static = _supplement_task_static_metadata(ordered_static, dataset_id=dataset_id, paths=paths)

    rated_powers = sorted({float(value) for value in ordered_static["rated_power_kw"].drop_nulls().to_list()})
    if len(rated_powers) != 1:
        raise ValueError(f"Dataset {dataset_id!r} must have a single rated_power_kw, found {rated_powers!r}.")

    return DatasetMetadata(
        dataset_id=dataset_id,
        turbine_ids=context_ids,
        rated_power_kw=rated_powers[0],
        turbine_static=ordered_static,
        task_paths=paths,
    )


def load_series_frame(dataset_id: str, *, cache_root: str | Path = _CACHE_ROOT) -> pl.DataFrame:
    paths = ensure_task_cache(dataset_id, cache_root=cache_root)
    available_columns = set(pl.read_parquet_schema(paths.series_path))
    missing_base = [column for column in _SERIES_BASE_COLUMNS if column not in available_columns]
    if missing_base:
        raise ValueError(
            f"Series {paths.series_path} for dataset {dataset_id!r} is missing required base columns {missing_base!r}."
        )
    load_started = time.monotonic()
    frame = (
        pl.scan_parquet(paths.series_path)
        .select(list(_SERIES_BASE_COLUMNS))
        .sort(["turbine_id", "timestamp"])
        .collect()
    )
    _profile_log(
        dataset_id,
        "load_series",
        rows=frame.height,
        columns=len(frame.columns),
        duration_seconds=round(time.monotonic() - load_started, 6),
    )
    return frame


def load_strict_window_index(dataset_id: str, *, cache_root: str | Path = _CACHE_ROOT) -> pl.DataFrame:
    paths = ensure_task_cache(dataset_id, cache_root=cache_root)
    load_started = time.monotonic()
    frame = (
        pl.scan_parquet(paths.window_index_path)
        .select(list(_TASK_WINDOW_COLUMNS))
        .filter(
            pl.col("is_complete_input")
            & pl.col("is_complete_output")
            & pl.col("is_fully_synchronous_input")
            & pl.col("is_fully_synchronous_output")
            & (pl.col("quality_flags").fill_null("") == "")
        )
        .sort("output_start_ts")
        .collect()
    )
    _profile_log(
        dataset_id,
        "load_window_index",
        strict_windows=frame.height,
        duration_seconds=round(time.monotonic() - load_started, 6),
    )
    if frame.is_empty():
        raise ValueError(f"Dataset {dataset_id!r} has no strict full-synchronous windows for {TASK_ID}.")
    return frame


def split_farm_window_index(
    window_index: pl.DataFrame,
    *,
    raw_timestamps: Sequence[datetime],
    resolution_minutes: int,
    max_windows_per_split: int | None = None,
) -> dict[str, pl.DataFrame]:
    sortable_window_index = (
        window_index
        if "turbine_id" in window_index.columns
        else window_index.with_columns(pl.lit("__farm__").alias("turbine_id"))
    )
    split_frames = window_protocols.split_window_index(
        sortable_window_index,
        raw_timestamps=raw_timestamps,
        resolution_minutes=resolution_minutes,
        history_steps=HISTORY_STEPS,
        max_windows_per_split=max_windows_per_split,
    )
    return {
        split_name: (
            frame.drop("turbine_id")
            if "turbine_id" in frame.columns and frame["turbine_id"].n_unique() == 1
            else frame
        )
        for split_name, frame in split_frames.items()
    }


def resolve_static_coordinate_columns(turbine_static: pl.DataFrame) -> tuple[str, str]:
    if {"coord_x", "coord_y"}.issubset(turbine_static.columns):
        subset = turbine_static.select(["coord_x", "coord_y"])
        if subset["coord_x"].null_count() == 0 and subset["coord_y"].null_count() == 0:
            return ("coord_x", "coord_y")
    if {"latitude", "longitude"}.issubset(turbine_static.columns):
        subset = turbine_static.select(["latitude", "longitude"])
        if subset["latitude"].null_count() == 0 and subset["longitude"].null_count() == 0:
            return ("latitude", "longitude")
    raise ValueError(
        "Turbine static coordinates are incomplete: expected either full coord_x/coord_y or full latitude/longitude."
    )


def _euclidean_distance(point_a: tuple[float, float], point_b: tuple[float, float]) -> float:
    return math.hypot(point_a[0] - point_b[0], point_a[1] - point_b[1])


def _haversine_distance_km(point_a: tuple[float, float], point_b: tuple[float, float]) -> float:
    lat1, lon1 = map(math.radians, point_a)
    lat2, lon2 = map(math.radians, point_b)
    delta_lat = lat2 - lat1
    delta_lon = lon2 - lon1
    sin_lat = math.sin(delta_lat / 2.0)
    sin_lon = math.sin(delta_lon / 2.0)
    hav = sin_lat**2 + math.cos(lat1) * math.cos(lat2) * sin_lon**2
    return 6371.0088 * 2.0 * math.asin(math.sqrt(hav))


def build_distance_sanity_frame(
    turbine_static: pl.DataFrame,
    *,
    ordered_turbine_ids: Sequence[str],
) -> tuple[str, pl.DataFrame]:
    rows_by_turbine = {
        str(row["turbine_id"]): row
        for row in turbine_static.filter(pl.col("turbine_id").is_in(list(ordered_turbine_ids)))
        .iter_rows(named=True)
    }
    missing_turbines = [turbine_id for turbine_id in ordered_turbine_ids if turbine_id not in rows_by_turbine]
    if missing_turbines:
        raise ValueError(f"Missing task-local turbine_static rows for {missing_turbines!r}.")

    rows = [rows_by_turbine[turbine_id] for turbine_id in ordered_turbine_ids]
    frame = pl.DataFrame(rows)
    first_column, second_column = resolve_static_coordinate_columns(frame)
    if first_column == "coord_x":
        coordinate_mode = "coord_xy"
        distance_fn = _euclidean_distance
    else:
        coordinate_mode = "latlon"
        distance_fn = _haversine_distance_km

    coordinates = {
        turbine_id: (float(row[first_column]), float(row[second_column]))
        for turbine_id, row in zip(ordered_turbine_ids, rows, strict=True)
    }
    summary_rows: list[dict[str, object]] = []
    for turbine_id in ordered_turbine_ids:
        neighbors = sorted(
            (
                (other_turbine_id, distance_fn(coordinates[turbine_id], coordinates[other_turbine_id]))
                for other_turbine_id in ordered_turbine_ids
                if other_turbine_id != turbine_id
            ),
            key=lambda item: (item[1], item[0]),
        )
        nearest_turbine_id, nearest_distance = neighbors[0] if neighbors else ("", float("nan"))
        summary_rows.append(
            {
                "turbine_id": turbine_id,
                "nearest_turbine_id": nearest_turbine_id,
                "nearest_distance": float(nearest_distance),
            }
        )
    return coordinate_mode, pl.DataFrame(summary_rows)


def _normalize_target_expr(rated_power_kw: float) -> pl.Expr:
    return (
        pl.when(pl.col("target_kw").is_null())
        .then(pl.lit(None, dtype=pl.Float64))
        .otherwise(pl.col("target_kw").cast(pl.Float64).clip(0.0, rated_power_kw) / rated_power_kw)
        .alias("target_pu")
    )


def prepare_panel_frame(series: pl.DataFrame, *, rated_power_kw: float) -> pl.DataFrame:
    return (
        series.with_columns(_normalize_target_expr(rated_power_kw))
        .select(["timestamp", "turbine_id", "target_pu"])
        .pivot(on="turbine_id", index="timestamp", values="target_pu", aggregate_function="first")
        .sort("timestamp")
    )


def resolve_resolution_minutes(panel_frame: pl.DataFrame) -> int:
    timestamps = panel_frame["timestamp"].cast(pl.Int64).head(2).to_list()
    if len(timestamps) < 2:
        raise ValueError("Panel frame must include at least two timestamps to infer resolution.")
    step_us = int(timestamps[1]) - int(timestamps[0])
    if step_us <= 0 or step_us % (60 * 1_000_000) != 0:
        raise ValueError(f"Unsupported resolution step {step_us!r}us.")
    return step_us // (60 * 1_000_000)


def build_panel_series(
    panel_frame: pl.DataFrame,
    *,
    turbine_ids: Sequence[str],
) -> tuple[np.ndarray, np.ndarray]:
    missing_columns = [turbine_id for turbine_id in turbine_ids if turbine_id not in panel_frame.columns]
    if missing_columns:
        raise ValueError(f"Panel frame is missing turbine columns {missing_columns!r}.")
    timestamps_us = panel_frame["timestamp"].cast(pl.Int64).to_numpy()
    target_pu = panel_frame.select(list(turbine_ids)).to_numpy().astype(np.float32, copy=False)
    return timestamps_us, target_pu


def build_window_descriptor_index(
    window_index: pl.DataFrame,
    *,
    timestamps_us: np.ndarray,
) -> FarmWindowDescriptorIndex:
    if window_index.is_empty():
        return FarmWindowDescriptorIndex.empty()

    timestamp_lookup = {
        int(timestamp): index
        for index, timestamp in enumerate(np.asarray(timestamps_us, dtype=np.int64).tolist())
    }
    working = window_index.with_columns(
        pl.col("output_start_ts").cast(pl.Int64).alias("output_start_us"),
        pl.col("output_end_ts").cast(pl.Int64).alias("output_end_us"),
    )

    target_indices: list[int] = []
    output_start_values: list[int] = []
    output_end_values: list[int] = []
    for row in working.iter_rows(named=True):
        output_start_us = int(row["output_start_us"])
        output_end_us = int(row["output_end_us"])
        try:
            target_index = timestamp_lookup[output_start_us]
        except KeyError as exc:
            raise ValueError(f"Window output_start_ts={output_start_us!r} is missing from panel timestamps.") from exc
        target_indices.append(target_index)
        output_start_values.append(output_start_us)
        output_end_values.append(output_end_us)

    return FarmWindowDescriptorIndex(
        target_indices=np.asarray(target_indices, dtype=np.int32),
        output_start_us=np.asarray(output_start_values, dtype=np.int64),
        output_end_us=np.asarray(output_end_values, dtype=np.int64),
    )


def thin_non_overlap_window_index(
    windows: FarmWindowDescriptorIndex,
    *,
    forecast_steps: int,
) -> FarmWindowDescriptorIndex:
    if len(windows) == 0:
        return FarmWindowDescriptorIndex.empty()
    keep_indices = np.arange(0, len(windows), forecast_steps, dtype=np.int64)
    return FarmWindowDescriptorIndex(
        target_indices=windows.target_indices[keep_indices],
        output_start_us=windows.output_start_us[keep_indices],
        output_end_us=windows.output_end_us[keep_indices],
    )


if Dataset is not None:

    class PanelWindowDataset(Dataset):
        def __init__(
            self,
            target_pu: np.ndarray,
            windows: FarmWindowDescriptorIndex,
            *,
            history_steps: int,
            forecast_steps: int,
        ) -> None:
            self.target_pu = np.asarray(target_pu, dtype=np.float32)
            self.windows = windows
            self.history_steps = history_steps
            self.forecast_steps = forecast_steps

        def __len__(self) -> int:
            return len(self.windows)

        def __getitem__(self, index: int) -> tuple[np.ndarray, np.ndarray]:
            target_index = int(self.windows.target_indices[index])
            history = self.target_pu[target_index - self.history_steps : target_index]
            targets = self.target_pu[target_index : target_index + self.forecast_steps]
            if history.shape != (self.history_steps, self.target_pu.shape[1]):
                raise ValueError(f"History slice for index {index} has unexpected shape {history.shape!r}.")
            if targets.shape != (self.forecast_steps, self.target_pu.shape[1]):
                raise ValueError(f"Target slice for index {index} has unexpected shape {targets.shape!r}.")
            if not np.isfinite(history).all() or not np.isfinite(targets).all():
                raise ValueError(f"Window index {index} contains non-finite target values despite strict filtering.")
            return (
                history[:, :, None].astype(np.float32, copy=True),
                targets[:, :, None].astype(np.float32, copy=True),
            )

else:

    class PanelWindowDataset:  # pragma: no cover - exercised only when torch is missing
        def __init__(self, *_args, **_kwargs) -> None:
            require_torch()


def prepare_dataset(
    dataset_id: str,
    *,
    cache_root: str | Path = _CACHE_ROOT,
    max_train_origins: int | None = None,
    max_eval_origins: int | None = None,
) -> PreparedDataset:
    metadata = load_dataset_metadata(dataset_id, cache_root=cache_root)
    strict_window_index = load_strict_window_index(dataset_id, cache_root=cache_root)
    series = load_series_frame(dataset_id, cache_root=cache_root)
    coordinate_mode, distance_sanity = build_distance_sanity_frame(
        metadata.turbine_static,
        ordered_turbine_ids=metadata.turbine_ids,
    )
    panel_frame = prepare_panel_frame(series, rated_power_kw=metadata.rated_power_kw)
    resolution_minutes = resolve_resolution_minutes(panel_frame)
    raw_timestamps = panel_frame["timestamp"].to_list()
    split_frames = split_farm_window_index(
        strict_window_index,
        raw_timestamps=raw_timestamps,
        resolution_minutes=resolution_minutes,
    )
    if max_train_origins is not None:
        split_frames["train"] = split_frames["train"].head(max_train_origins)
    if max_eval_origins is not None:
        split_frames["val"] = split_frames["val"].head(max_eval_origins)
        split_frames["test"] = split_frames["test"].head(max_eval_origins)

    timestamps_us, target_pu = build_panel_series(panel_frame, turbine_ids=metadata.turbine_ids)
    train_windows = build_window_descriptor_index(split_frames["train"], timestamps_us=timestamps_us)
    val_rolling_windows = build_window_descriptor_index(split_frames["val"], timestamps_us=timestamps_us)
    test_rolling_windows = build_window_descriptor_index(split_frames["test"], timestamps_us=timestamps_us)
    val_non_overlap_windows = thin_non_overlap_window_index(val_rolling_windows, forecast_steps=FORECAST_STEPS)
    test_non_overlap_windows = thin_non_overlap_window_index(test_rolling_windows, forecast_steps=FORECAST_STEPS)
    _profile_log(
        dataset_id,
        "prepare_dataset_complete",
        coordinate_mode=coordinate_mode,
        node_count=len(metadata.turbine_ids),
        train_windows=len(train_windows),
        val_rolling_windows=len(val_rolling_windows),
        val_non_overlap_windows=len(val_non_overlap_windows),
        test_rolling_windows=len(test_rolling_windows),
        test_non_overlap_windows=len(test_non_overlap_windows),
        nearest_neighbors=distance_sanity.to_dicts(),
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
        turbine_ids=metadata.turbine_ids,
        coordinate_mode=coordinate_mode,
        node_count=len(metadata.turbine_ids),
        timestamps_us=timestamps_us,
        target_pu=target_pu,
        train_windows=train_windows,
        val_rolling_windows=val_rolling_windows,
        val_non_overlap_windows=val_non_overlap_windows,
        test_rolling_windows=test_rolling_windows,
        test_non_overlap_windows=test_non_overlap_windows,
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


if nn is not None and F is not None:

    class AVWGCN(nn.Module):
        def __init__(self, dim_in: int, dim_out: int, cheb_k: int, embed_dim: int) -> None:
            super().__init__()
            self.cheb_k = cheb_k
            self.weights_pool = nn.Parameter(torch.empty(embed_dim, cheb_k, dim_in, dim_out))
            self.bias_pool = nn.Parameter(torch.empty(embed_dim, dim_out))

        def forward(self, x, node_embeddings):
            node_count = node_embeddings.shape[0]
            supports = F.softmax(F.relu(torch.mm(node_embeddings, node_embeddings.transpose(0, 1))), dim=1)
            support_list = [torch.eye(node_count, device=supports.device, dtype=supports.dtype), supports]
            for _ in range(2, self.cheb_k):
                support_list.append(torch.matmul(2 * supports, support_list[-1]) - support_list[-2])
            supports = torch.stack(support_list, dim=0)
            weights = torch.einsum("nd,dkio->nkio", node_embeddings, self.weights_pool)
            bias = torch.matmul(node_embeddings, self.bias_pool)
            x_g = torch.einsum("knm,bmc->bknc", supports, x)
            x_g = x_g.permute(0, 2, 1, 3)
            return torch.einsum("bnki,nkio->bno", x_g, weights) + bias


    class AGCRNCell(nn.Module):
        def __init__(self, node_num: int, input_dim: int, hidden_dim: int, cheb_k: int, embed_dim: int) -> None:
            super().__init__()
            self.node_num = node_num
            self.hidden_dim = hidden_dim
            self.gate = AVWGCN(input_dim + self.hidden_dim, 2 * hidden_dim, cheb_k, embed_dim)
            self.update = AVWGCN(input_dim + self.hidden_dim, hidden_dim, cheb_k, embed_dim)

        def forward(self, x, state, node_embeddings):
            state = state.to(device=x.device, dtype=x.dtype)
            input_and_state = torch.cat((x, state), dim=-1)
            z_r = torch.sigmoid(self.gate(input_and_state, node_embeddings))
            z, r = torch.split(z_r, self.hidden_dim, dim=-1)
            candidate = torch.cat((x, z * state), dim=-1)
            hc = torch.tanh(self.update(candidate, node_embeddings))
            return r * state + (1.0 - r) * hc

        def init_hidden_state(self, batch_size: int):
            return torch.zeros(batch_size, self.node_num, self.hidden_dim)


    class AVWDCRNN(nn.Module):
        def __init__(
            self,
            node_num: int,
            dim_in: int,
            dim_out: int,
            cheb_k: int,
            embed_dim: int,
            num_layers: int = 1,
        ) -> None:
            super().__init__()
            if num_layers < 1:
                raise ValueError("At least one AGCRN recurrent layer is required.")
            self.node_num = node_num
            self.input_dim = dim_in
            self.num_layers = num_layers
            self.dcrnn_cells = nn.ModuleList()
            self.dcrnn_cells.append(AGCRNCell(node_num, dim_in, dim_out, cheb_k, embed_dim))
            for _ in range(1, num_layers):
                self.dcrnn_cells.append(AGCRNCell(node_num, dim_out, dim_out, cheb_k, embed_dim))

        def forward(self, x, init_state, node_embeddings):
            if x.shape[2] != self.node_num or x.shape[3] != self.input_dim:
                raise ValueError(
                    f"Expected source with shape [batch, time, {self.node_num}, {self.input_dim}], got {x.shape!r}."
                )
            seq_length = x.shape[1]
            current_inputs = x
            output_hidden = []
            for layer_index in range(self.num_layers):
                state = init_state[layer_index]
                inner_states = []
                for time_index in range(seq_length):
                    state = self.dcrnn_cells[layer_index](current_inputs[:, time_index, :, :], state, node_embeddings)
                    inner_states.append(state)
                output_hidden.append(state)
                current_inputs = torch.stack(inner_states, dim=1)
            return current_inputs, torch.stack(output_hidden, dim=0)

        def init_hidden(self, batch_size: int):
            init_states = []
            for layer_index in range(self.num_layers):
                init_states.append(self.dcrnn_cells[layer_index].init_hidden_state(batch_size))
            return torch.stack(init_states, dim=0)


    class AGCRN(nn.Module):
        def __init__(
            self,
            *,
            node_count: int,
            input_channels: int,
            hidden_dim: int,
            forecast_steps: int,
            embed_dim: int,
            num_layers: int,
            cheb_k: int,
            output_channels: int = OUTPUT_CHANNELS,
        ) -> None:
            super().__init__()
            self.num_node = node_count
            self.input_dim = input_channels
            self.hidden_dim = hidden_dim
            self.output_dim = output_channels
            self.horizon = forecast_steps
            self.num_layers = num_layers
            self.node_embeddings = nn.Parameter(torch.randn(self.num_node, embed_dim), requires_grad=True)
            self.encoder = AVWDCRNN(
                self.num_node,
                self.input_dim,
                self.hidden_dim,
                cheb_k,
                embed_dim,
                self.num_layers,
            )
            self.end_conv = nn.Conv2d(
                1,
                self.horizon * self.output_dim,
                kernel_size=(1, self.hidden_dim),
                bias=True,
            )

        def forward(self, source, targets=None, teacher_forcing_ratio: float = 0.5):
            del targets, teacher_forcing_ratio
            if source.ndim != 4:
                raise ValueError(f"Expected source with shape [batch, history, nodes, channels], got {source.shape!r}.")
            if source.shape[2] != self.num_node:
                raise ValueError(f"Expected node_count={self.num_node}, received {source.shape[2]}.")
            if source.shape[3] != self.input_dim:
                raise ValueError(f"Expected input_channels={self.input_dim}, received {source.shape[3]}.")

            init_state = self.encoder.init_hidden(source.shape[0]).to(device=source.device, dtype=source.dtype)
            output, _ = self.encoder(source, init_state, self.node_embeddings)
            output = output[:, -1:, :, :]
            output = self.end_conv(output)
            output = output.squeeze(-1).reshape(-1, self.horizon, self.output_dim, self.num_node)
            return output.permute(0, 1, 3, 2)

else:

    class AVWGCN:  # pragma: no cover - exercised only when torch is missing
        def __init__(self, *_args, **_kwargs) -> None:
            require_torch()


    class AGCRNCell:  # pragma: no cover - exercised only when torch is missing
        def __init__(self, *_args, **_kwargs) -> None:
            require_torch()


    class AVWDCRNN:  # pragma: no cover - exercised only when torch is missing
        def __init__(self, *_args, **_kwargs) -> None:
            require_torch()


    class AGCRN:  # pragma: no cover - exercised only when torch is missing
        def __init__(self, *_args, **_kwargs) -> None:
            require_torch()


def build_model(
    *,
    node_count: int,
    hidden_dim: int,
    embed_dim: int,
    num_layers: int,
    cheb_k: int,
    forecast_steps: int,
):
    require_torch()
    return AGCRN(
        node_count=node_count,
        input_channels=INPUT_CHANNELS,
        hidden_dim=hidden_dim,
        forecast_steps=forecast_steps,
        embed_dim=embed_dim,
        num_layers=num_layers,
        cheb_k=cheb_k,
        output_channels=OUTPUT_CHANNELS,
    )


def initialize_official_aligned_parameters(model) -> None:
    _, resolved_nn, _, _, _ = require_torch()
    for parameter in model.parameters():
        if parameter.dim() > 1:
            resolved_nn.init.xavier_uniform_(parameter)
        else:
            resolved_nn.init.uniform_(parameter)


def _build_dataloader(
    prepared_dataset: PreparedDataset,
    *,
    windows: FarmWindowDescriptorIndex,
    batch_size: int,
    shuffle: bool,
    seed: int,
):
    resolved_torch, _, _, resolved_loader, _ = require_torch()
    dataset = PanelWindowDataset(
        prepared_dataset.target_pu,
        windows,
        history_steps=prepared_dataset.history_steps,
        forecast_steps=prepared_dataset.forecast_steps,
    )
    generator = resolved_torch.Generator()
    generator.manual_seed(seed)
    return resolved_loader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        generator=generator if shuffle else None,
    )


def _safe_divide(numerator: float, denominator: int) -> float:
    if denominator <= 0:
        return float("nan")
    return numerator / denominator


def _safe_rmse(squared_error_sum: float, denominator: int) -> float:
    if denominator <= 0:
        return float("nan")
    return math.sqrt(squared_error_sum / denominator)


def evaluate_model(
    model,
    loader,
    *,
    device: str,
    rated_power_kw: float,
    forecast_steps: int,
    progress_label: str | None = None,
) -> EvaluationMetrics:
    resolved_torch, _, _, _, _ = require_torch()
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
    progress = _create_progress_bar(total=_loader_batch_total(loader), desc=progress_label or "evaluate")
    try:
        with resolved_torch.no_grad():
            for batch_history, batch_targets in loader:
                batch_history = batch_history.to(device=device, dtype=resolved_torch.float32)
                batch_targets = batch_targets.to(device=device, dtype=resolved_torch.float32)
                predictions = model(batch_history, batch_targets, teacher_forcing_ratio=0.0)
                errors_pu = predictions - batch_targets
                errors_kw = errors_pu * rated_power_kw
                batch_window_count = int(batch_history.shape[0])
                predictions_per_horizon = int(batch_targets.shape[2] * batch_targets.shape[3])
                batch_prediction_count = int(batch_targets.numel())
                window_count += batch_window_count
                prediction_count += batch_prediction_count

                abs_errors_kw = resolved_torch.abs(errors_kw)
                squared_errors_kw = resolved_torch.square(errors_kw)
                abs_errors_pu = resolved_torch.abs(errors_pu)
                squared_errors_pu = resolved_torch.square(errors_pu)

                abs_error_sum_kw += float(abs_errors_kw.sum().item())
                squared_error_sum_kw += float(squared_errors_kw.sum().item())
                abs_error_sum_pu += float(abs_errors_pu.sum().item())
                squared_error_sum_pu += float(squared_errors_pu.sum().item())

                horizon_window_count += batch_window_count
                horizon_prediction_count += batch_window_count * predictions_per_horizon
                horizon_abs_error_sum_kw += (
                    abs_errors_kw.sum(dim=(0, 2, 3)).detach().cpu().numpy().astype(np.float64, copy=False)
                )
                horizon_squared_error_sum_kw += (
                    squared_errors_kw.sum(dim=(0, 2, 3)).detach().cpu().numpy().astype(np.float64, copy=False)
                )
                horizon_abs_error_sum_pu += (
                    abs_errors_pu.sum(dim=(0, 2, 3)).detach().cpu().numpy().astype(np.float64, copy=False)
                )
                horizon_squared_error_sum_pu += (
                    squared_errors_pu.sum(dim=(0, 2, 3)).detach().cpu().numpy().astype(np.float64, copy=False)
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
    prepared_dataset: PreparedDataset,
    *,
    device: str,
    seed: int = DEFAULT_SEED,
    batch_size: int = DEFAULT_BATCH_SIZE,
    learning_rate: float = DEFAULT_LEARNING_RATE,
    max_epochs: int = DEFAULT_MAX_EPOCHS,
    early_stopping_patience: int = DEFAULT_EARLY_STOPPING_PATIENCE,
    hidden_dim: int = DEFAULT_HIDDEN_DIM,
    embed_dim: int = DEFAULT_EMBED_DIM,
    num_layers: int = DEFAULT_NUM_LAYERS,
    cheb_k: int = DEFAULT_CHEB_K,
    grad_clip_norm: float = DEFAULT_GRAD_CLIP_NORM,
    progress_label: str | None = None,
) -> TrainingOutcome:
    resolved_torch, _, _, _, _ = require_torch()
    _set_random_seed(seed)
    resolved_device = resolve_device(device)
    model = build_model(
        node_count=prepared_dataset.node_count,
        hidden_dim=hidden_dim,
        embed_dim=embed_dim,
        num_layers=num_layers,
        cheb_k=cheb_k,
        forecast_steps=prepared_dataset.forecast_steps,
    ).to(device=resolved_device)
    initialize_official_aligned_parameters(model)
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
                for batch_history, batch_targets in train_loader:
                    batch_history = batch_history.to(device=resolved_device, dtype=resolved_torch.float32)
                    batch_targets = batch_targets.to(device=resolved_device, dtype=resolved_torch.float32)
                    optimizer.zero_grad(set_to_none=True)
                    predictions = model(batch_history, batch_targets, teacher_forcing_ratio=0.0)
                    loss = criterion(predictions, batch_targets)
                    loss.backward()
                    if grad_clip_norm > 0:
                        resolved_torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
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


def _timestamp_us_to_string(value: int | None) -> str | None:
    if value is None:
        return None
    return (
        datetime.fromtimestamp(value / 1_000_000, tz=UTC)
        .replace(tzinfo=None)
        .strftime("%Y-%m-%d %H:%M:%S")
    )


def _window_bounds(windows: FarmWindowDescriptorIndex) -> tuple[str | None, str | None]:
    if len(windows) == 0:
        return None, None
    return (
        _timestamp_us_to_string(int(windows.output_start_us.min())),
        _timestamp_us_to_string(int(windows.output_end_us.max())),
    )


def iter_evaluation_specs(prepared_dataset: PreparedDataset) -> tuple[tuple[str, str, FarmWindowDescriptorIndex], ...]:
    return (
        ("val", ROLLING_EVAL_PROTOCOL, prepared_dataset.val_rolling_windows),
        ("val", NON_OVERLAP_EVAL_PROTOCOL, prepared_dataset.val_non_overlap_windows),
        ("test", ROLLING_EVAL_PROTOCOL, prepared_dataset.test_rolling_windows),
        ("test", NON_OVERLAP_EVAL_PROTOCOL, prepared_dataset.test_non_overlap_windows),
    )


def build_result_rows(
    prepared_dataset: PreparedDataset,
    *,
    training_outcome: TrainingOutcome,
    runtime_seconds: float,
    seed: int,
    batch_size: int,
    learning_rate: float,
    hidden_dim: int,
    embed_dim: int,
    num_layers: int,
    cheb_k: int,
    grad_clip_norm: float,
    evaluation_results: Sequence[tuple[str, str, FarmWindowDescriptorIndex, EvaluationMetrics]],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    base_row = {
        "dataset_id": prepared_dataset.dataset_id,
        "model_id": MODEL_ID,
        "model_variant": MODEL_VARIANT,
        "task_id": TASK_ID,
        "window_protocol": WINDOW_PROTOCOL,
        "history_steps": prepared_dataset.history_steps,
        "forecast_steps": prepared_dataset.forecast_steps,
        "stride_steps": prepared_dataset.stride_steps,
        "split_protocol": SPLIT_PROTOCOL,
        "graph_mode": GRAPH_MODE,
        "graph_source": GRAPH_SOURCE,
        "coordinate_mode": prepared_dataset.coordinate_mode,
        "node_count": prepared_dataset.node_count,
        "input_channels": INPUT_CHANNELS,
        "hidden_dim": hidden_dim,
        "embed_dim": embed_dim,
        "num_layers": num_layers,
        "cheb_k": cheb_k,
        "grad_clip_norm": grad_clip_norm,
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
    device: str | None = None,
    seed: int = DEFAULT_SEED,
    batch_size: int = DEFAULT_BATCH_SIZE,
    learning_rate: float = DEFAULT_LEARNING_RATE,
    max_epochs: int = DEFAULT_MAX_EPOCHS,
    early_stopping_patience: int = DEFAULT_EARLY_STOPPING_PATIENCE,
    hidden_dim: int = DEFAULT_HIDDEN_DIM,
    embed_dim: int = DEFAULT_EMBED_DIM,
    num_layers: int = DEFAULT_NUM_LAYERS,
    cheb_k: int = DEFAULT_CHEB_K,
    grad_clip_norm: float = DEFAULT_GRAD_CLIP_NORM,
) -> list[dict[str, object]]:
    dataset_start = time.monotonic()
    progress_label = prepared_dataset.dataset_id
    training_outcome = train_model(
        prepared_dataset,
        device=resolve_device(device),
        seed=seed,
        batch_size=batch_size,
        learning_rate=learning_rate,
        max_epochs=max_epochs,
        early_stopping_patience=early_stopping_patience,
        hidden_dim=hidden_dim,
        embed_dim=embed_dim,
        num_layers=num_layers,
        cheb_k=cheb_k,
        grad_clip_norm=grad_clip_norm,
        progress_label=progress_label,
    )
    evaluation_results: list[tuple[str, str, FarmWindowDescriptorIndex, EvaluationMetrics]] = []
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
        best_epoch=training_outcome.best_epoch,
        epochs_ran=training_outcome.epochs_ran,
        best_val_rmse_pu=training_outcome.best_val_rmse_pu,
        test_rolling_rmse_pu=test_rolling_metrics.rmse_pu,
        runtime_seconds=round(runtime_seconds, 6),
    )
    return build_result_rows(
        prepared_dataset,
        training_outcome=training_outcome,
        runtime_seconds=runtime_seconds,
        seed=seed,
        batch_size=batch_size,
        learning_rate=learning_rate,
        hidden_dim=hidden_dim,
        embed_dim=embed_dim,
        num_layers=num_layers,
        cheb_k=cheb_k,
        grad_clip_norm=grad_clip_norm,
        evaluation_results=evaluation_results,
    )


def sort_result_frame(frame: pl.DataFrame) -> pl.DataFrame:
    return (
        frame.with_columns(
            pl.col("dataset_id")
            .replace_strict(_DATASET_ORDER, default=len(_DATASET_ORDER))
            .alias("__dataset_order"),
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
                "__split_order",
                "__eval_protocol_order",
                "__metric_scope_order",
                "__lead_order",
            ]
        )
        .drop(
            [
                "__dataset_order",
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
    cache_root: str | Path = _CACHE_ROOT,
    output_path: str | Path = _OUTPUT_PATH,
    device: str | None = None,
    max_epochs: int = DEFAULT_MAX_EPOCHS,
    max_train_origins: int | None = None,
    max_eval_origins: int | None = None,
    seed: int = DEFAULT_SEED,
    batch_size: int = DEFAULT_BATCH_SIZE,
    learning_rate: float = DEFAULT_LEARNING_RATE,
    early_stopping_patience: int = DEFAULT_EARLY_STOPPING_PATIENCE,
    hidden_dim: int = DEFAULT_HIDDEN_DIM,
    embed_dim: int = DEFAULT_EMBED_DIM,
    num_layers: int = DEFAULT_NUM_LAYERS,
    cheb_k: int = DEFAULT_CHEB_K,
    grad_clip_norm: float = DEFAULT_GRAD_CLIP_NORM,
    dataset_loader: Callable[..., PreparedDataset] | None = None,
    job_runner: Callable[..., list[dict[str, object]]] | None = None,
) -> pl.DataFrame:
    loader = dataset_loader or prepare_dataset
    runner = job_runner or execute_training_job
    rows: list[dict[str, object]] = []
    job_progress = _create_progress_bar(total=len(dataset_ids), desc="agcrn jobs", leave=True)
    try:
        for dataset_id in dataset_ids:
            job_progress.set_postfix_str(dataset_id)
            prepared = loader(
                dataset_id,
                cache_root=cache_root,
                max_train_origins=max_train_origins,
                max_eval_origins=max_eval_origins,
            )
            rows.extend(
                runner(
                    prepared,
                    device=device,
                    seed=seed,
                    batch_size=batch_size,
                    learning_rate=learning_rate,
                    max_epochs=max_epochs,
                    early_stopping_patience=early_stopping_patience,
                    hidden_dim=hidden_dim,
                    embed_dim=embed_dim,
                    num_layers=num_layers,
                    cheb_k=cheb_k,
                    grad_clip_norm=grad_clip_norm,
                )
            )
            job_progress.update(1)
    finally:
        job_progress.close()

    results = sort_result_frame(pl.DataFrame(rows, infer_schema_length=None).select(_RESULT_COLUMNS))
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    results.write_csv(output)
    return results


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the official-aligned Kelmarsh farm-synchronous power-only AGCRN baseline."
    )
    parser.add_argument(
        "--dataset",
        action="append",
        choices=list(DEFAULT_DATASETS),
        dest="datasets",
        help="Limit execution to one or more datasets. Defaults to kelmarsh.",
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
        "--max-train-origins",
        type=int,
        default=None,
        help="Optional smoke-test limit applied to the dense train origins after split selection.",
    )
    parser.add_argument(
        "--max-eval-origins",
        type=int,
        default=None,
        help="Optional smoke-test limit applied to the dense val/test origins before non-overlap thinning.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help="Training and evaluation batch size.",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=DEFAULT_LEARNING_RATE,
        help="Adam learning rate.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help="Random seed.",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=DEFAULT_EARLY_STOPPING_PATIENCE,
        help="Early stopping patience in epochs.",
    )
    parser.add_argument(
        "--hidden-dim",
        type=int,
        default=DEFAULT_HIDDEN_DIM,
        help="Hidden dimension for the recurrent state.",
    )
    parser.add_argument(
        "--embed-dim",
        type=int,
        default=DEFAULT_EMBED_DIM,
        help="Learned node-embedding dimension for the adaptive graph.",
    )
    parser.add_argument(
        "--num-layers",
        type=int,
        default=DEFAULT_NUM_LAYERS,
        help="Number of stacked AGCRN recurrent layers.",
    )
    parser.add_argument(
        "--cheb-k",
        type=int,
        default=DEFAULT_CHEB_K,
        help="Adaptive graph convolution support order.",
    )
    parser.add_argument(
        "--grad-clip-norm",
        type=float,
        default=DEFAULT_GRAD_CLIP_NORM,
        help="Gradient clipping max norm. Use 0 to disable clipping.",
    )
    parser.add_argument(
        "--run-label",
        type=str,
        default=None,
        help="Optional label suffix for the formal run record under experiment/artifacts/runs/agcrn_official_aligned/.",
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
    results = run_experiment(
        dataset_ids=tuple(args.datasets) if args.datasets else DEFAULT_DATASETS,
        device=args.device,
        max_epochs=args.epochs,
        output_path=args.output_path,
        max_train_origins=args.max_train_origins,
        max_eval_origins=args.max_eval_origins,
        seed=args.seed,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        early_stopping_patience=args.patience,
        hidden_dim=args.hidden_dim,
        embed_dim=args.embed_dim,
        num_layers=args.num_layers,
        cheb_k=args.cheb_k,
        grad_clip_norm=args.grad_clip_norm,
    )
    if not args.no_record_run:
        record_cli_run(
            family_id=FAMILY_ID,
            repo_root=_REPO_ROOT,
            invocation_kind="family_runner",
            entrypoint="experiment/families/agcrn/run_agcrn.py",
            args=vars(args),
            output_path=args.output_path,
            result_row_count=results.height,
            dataset_ids=tuple(args.datasets) if args.datasets else DEFAULT_DATASETS,
            feature_protocol_ids=("power_only",),
            model_variants=(MODEL_VARIANT,),
            eval_protocols=(ROLLING_EVAL_PROTOCOL, NON_OVERLAP_EVAL_PROTOCOL),
            result_splits=("val", "test"),
            run_label=args.run_label,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
