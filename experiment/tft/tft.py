from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import UTC, datetime
import gc
import json
import math
import os
from pathlib import Path
import shutil
import sys
import tempfile
import time
from typing import Any, Callable, Mapping, Sequence
import warnings

import numpy as np
import pandas as pd
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
except ImportError:  # pragma: no cover - exercised in the root env where torch is absent
    torch = None

try:
    import lightning.pytorch as lightning
    from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
except ImportError:  # pragma: no cover - exercised in the root env where lightning is absent
    lightning = None
    EarlyStopping = None
    ModelCheckpoint = None

try:
    from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet
    from pytorch_forecasting.metrics import RMSE
except ImportError:  # pragma: no cover - exercised in the root env where pytorch-forecasting is absent
    TemporalFusionTransformer = None
    TimeSeriesDataSet = None
    RMSE = None


def _configure_warning_filters() -> None:
    warning_specs = (
        (
            "ignore",
            r"The given NumPy array is not writable, and PyTorch does not support non-writable tensors\..*",
            UserWarning,
        ),
        (
            "ignore",
            r"Attribute 'loss' is an instance of `nn\.Module` and is already saved during checkpointing\..*",
            UserWarning,
        ),
        (
            "ignore",
            r"Attribute 'logging_metrics' is an instance of `nn\.Module` and is already saved during checkpointing\..*",
            UserWarning,
        ),
        (
            "ignore",
            r"`isinstance\(treespec, LeafSpec\)` is deprecated, use `isinstance\(treespec, TreeSpec\) and treespec\.is_leaf\(\)` instead\..*",
            Warning,
        ),
        (
            "ignore",
            r"upsample_linear1d_backward_out_cuda does not have a deterministic implementation, but you set 'torch\.use_deterministic_algorithms\(True, warn_only=True\)'\..*",
            UserWarning,
        ),
    )
    for action, message, category in warning_specs:
        warnings.filterwarnings(action, message=message, category=category)


_configure_warning_filters()


EXPERIMENT_DIR = Path(__file__).resolve().parent
COMMON_DIR = EXPERIMENT_DIR.parent / "common"
if str(COMMON_DIR) not in sys.path:
    sys.path.insert(0, str(COMMON_DIR))

from covariate_packs import (  # noqa: E402
    CovariatePackSpec,
    reference_pack_for,
    resolve_covariate_pack,
)
from window_protocols import (  # noqa: E402
    DEFAULT_WINDOW_PROTOCOL,
    HORIZON_METRIC_SCOPE,
    NON_OVERLAP_EVAL_PROTOCOL,
    OVERALL_METRIC_SCOPE,
    ROLLING_EVAL_PROTOCOL,
    SPLIT_PROTOCOL,
    WindowDescriptorIndex,
    build_chrono_split_lookup as shared_build_chrono_split_lookup,
    build_window_descriptor_index as shared_build_window_descriptor_index,
    resolve_window_protocol,
    split_window_index as shared_split_window_index,
    thin_non_overlap_window_index as shared_thin_non_overlap_window_index,
)


MODEL_ID = "TFT"
EXPERIMENT_NAME = "tft-pilot"
WINDOW_PROTOCOL = DEFAULT_WINDOW_PROTOCOL
TASK_ID = resolve_window_protocol(WINDOW_PROTOCOL).task_id
QUALITY_PROFILE = "default"
SERIES_LAYOUT = "turbine"
DEFAULT_DATASETS = ("kelmarsh",)
DEFAULT_INPUT_PACKS = (
    "reference",
    "known_static",
    "hist_stage1",
    "hist_stage2",
    "mixed_stage1",
    "mixed_stage2",
)
HISTORY_STEPS = 144
FORECAST_STEPS = 36
STRIDE_STEPS = 1
TRAIN_DOWNSAMPLE_EVERY = 12
DEFAULT_SEED = 42
DEFAULT_MAX_EPOCHS = 30
DEFAULT_EARLY_STOPPING_PATIENCE = 5
DEFAULT_LEARNING_RATE = 1e-3
DEFAULT_HIDDEN_SIZE = 32
DEFAULT_ATTENTION_HEAD_SIZE = 4
DEFAULT_HIDDEN_CONTINUOUS_SIZE = 16
DEFAULT_DROPOUT = 0.1
DEFAULT_GRADIENT_CLIP_VAL = 0.1
DEFAULT_CPU_BATCH_SIZE = 128
DEFAULT_ACCELERATOR_BATCH_SIZE = 256
DEFAULT_PREFETCH_FACTOR = 4
KNOWN_FUTURE_COLUMNS = ("tod_sin", "tod_cos", "dow_sin", "dow_cos")
STATIC_COORD_COLUMNS = ("static_coord_1", "static_coord_2")
TARGET_MISSING_COLUMN = "target_missing"
REFERENCE_COVARIATE_POLICY = "none"
COVARIATE_POLICY = "historical_train_zscore_fill0_mask + static_dataset_zscore + known_calendar"
PROFILE_LOG_PREFIX = "[tft] "
TRAINER_PRECISION_CHOICES = ("auto", "32-true", "16-mixed", "bf16-mixed")
MATMUL_PRECISION_CHOICES = ("auto", "highest", "high", "medium")

_REPO_ROOT = EXPERIMENT_DIR.parents[1]
_CACHE_ROOT = _REPO_ROOT / "cache"
_OUTPUT_PATH = _REPO_ROOT / "experiment" / "tft-pilot.csv"
_WORK_DIR = EXPERIMENT_DIR / ".work"
_JOB_ARTIFACTS_ROOT = _WORK_DIR / "jobs"
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
    "input_pack",
    "historical_covariate_stage",
    "feature_set",
    "uses_static_covariates",
    "uses_known_future_covariates",
    "static_covariate_count",
    "known_covariate_count",
    "historical_covariate_count",
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
    "hidden_size",
    "attention_head_size",
    "hidden_continuous_size",
    "dropout",
    "gradient_clip_val",
    "num_workers",
    "trainer_precision",
    "matmul_precision",
]
_DATASET_ORDER = {dataset_id: index for index, dataset_id in enumerate(DEFAULT_DATASETS)}
_INPUT_PACK_ORDER = {name: index for index, name in enumerate(DEFAULT_INPUT_PACKS)}
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
class JobArtifactPaths:
    dataset_id: str
    input_pack: str
    job_dir: Path
    checkpoint_dir: Path
    training_state_path: Path


@dataclass(frozen=True)
class InputPackSpec:
    dataset_id: str
    input_pack: str
    feature_set: str
    historical_pack: CovariatePackSpec | None
    historical_covariate_stage: str | None
    historical_source_columns: tuple[str, ...]
    uses_static_covariates: bool
    uses_known_future_covariates: bool

    @property
    def historical_covariate_count(self) -> int:
        return len(self.historical_source_columns)


@dataclass(frozen=True)
class PreparedDataset:
    dataset_id: str
    resolution_minutes: int
    rated_power_kw: float
    history_steps: int
    forecast_steps: int
    stride_steps: int
    input_pack: str
    historical_covariate_stage: str | None
    feature_set: str
    uses_static_covariates: bool
    uses_known_future_covariates: bool
    static_covariate_columns: tuple[str, ...]
    known_covariate_columns: tuple[str, ...]
    historical_covariate_columns: tuple[str, ...]
    historical_mask_columns: tuple[str, ...]
    static_covariate_count: int
    known_covariate_count: int
    historical_covariate_count: int
    covariate_policy: str
    turbine_ids: tuple[str, ...]
    model_frame: pd.DataFrame
    train_windows: WindowDescriptorIndex
    val_rolling_windows: WindowDescriptorIndex
    val_non_overlap_windows: WindowDescriptorIndex
    test_rolling_windows: WindowDescriptorIndex
    test_non_overlap_windows: WindowDescriptorIndex
    train_origin_frame: pd.DataFrame
    val_rolling_origin_frame: pd.DataFrame
    val_non_overlap_origin_frame: pd.DataFrame
    test_rolling_origin_frame: pd.DataFrame
    test_non_overlap_origin_frame: pd.DataFrame


@dataclass(frozen=True)
class BuiltTimeSeriesDatasets:
    train_dataset: Any
    val_rolling_dataset: Any
    val_non_overlap_dataset: Any
    test_rolling_dataset: Any
    test_non_overlap_dataset: Any


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


def _create_progress_bar(*, total: int | None, desc: str, leave: bool = False):
    return tqdm(
        total=total,
        desc=desc,
        leave=leave,
        disable=not progress_is_enabled(),
        dynamic_ncols=True,
    )


def _sanitize_path_component(value: str) -> str:
    return "".join(character if character.isalnum() or character in "-_." else "_" for character in value)


def build_job_artifact_paths(dataset_id: str, input_pack: str) -> JobArtifactPaths:
    job_dir = _JOB_ARTIFACTS_ROOT / _sanitize_path_component(dataset_id) / _sanitize_path_component(input_pack)
    return JobArtifactPaths(
        dataset_id=dataset_id,
        input_pack=input_pack,
        job_dir=job_dir,
        checkpoint_dir=job_dir / "checkpoints",
        training_state_path=job_dir / "training_state.json",
    )


def _job_result_filter(dataset_id: str, input_pack: str) -> pl.Expr:
    return (pl.col("dataset_id") == dataset_id) & (pl.col("input_pack") == input_pack)


def _expected_job_result_row_count(*, forecast_steps: int = FORECAST_STEPS) -> int:
    return 4 * (forecast_steps + 1)


def _has_complete_job_results(
    results: pl.DataFrame | None,
    *,
    dataset_id: str,
    input_pack: str,
    forecast_steps: int = FORECAST_STEPS,
) -> bool:
    if results is None or results.height == 0:
        return False
    return (
        results.filter(_job_result_filter(dataset_id, input_pack)).height
        == _expected_job_result_row_count(forecast_steps=forecast_steps)
    )


def _read_results_frame(path: Path) -> pl.DataFrame | None:
    if not path.exists() or path.stat().st_size == 0:
        return None
    return pl.read_csv(path, infer_schema_length=None, try_parse_dates=True)


def _write_json_atomic(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            json.dump(payload, handle, ensure_ascii=False, sort_keys=True)
            handle.write("\n")
            temp_path = Path(handle.name)
        temp_path.replace(path)
    finally:
        if temp_path is not None and temp_path.exists():
            temp_path.unlink(missing_ok=True)


def _read_training_state(path: Path) -> dict[str, object] | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def _write_result_frame(results: pl.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            dir=output_path.parent,
            prefix=f".{output_path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temp_path = Path(handle.name)
        results.write_csv(temp_path)
        temp_path.replace(output_path)
    finally:
        if temp_path is not None and temp_path.exists():
            temp_path.unlink(missing_ok=True)


def _merge_job_results(
    existing_results: pl.DataFrame | None,
    *,
    dataset_id: str,
    input_pack: str,
    job_rows: Sequence[dict[str, object]],
) -> pl.DataFrame:
    job_frame = pl.DataFrame(job_rows, infer_schema_length=None).select(_RESULT_COLUMNS)
    if existing_results is None or existing_results.height == 0:
        return sort_result_frame(job_frame)
    retained = existing_results.filter(~_job_result_filter(dataset_id, input_pack)).select(_RESULT_COLUMNS)
    return sort_result_frame(pl.concat([retained, job_frame], how="vertical_relaxed"))


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
        granularity="turbine",
    )


def require_torch() -> Any:
    if torch is None:
        raise ImportError(
            "PyTorch is unavailable in the current environment. "
            "Create experiment/tft/.conda with ./create_env.sh."
        )
    return torch


def require_forecasting() -> tuple[Any, Any, Any, Any, Any, Any]:
    resolved_torch = require_torch()
    if lightning is None or EarlyStopping is None or ModelCheckpoint is None:
        raise ImportError(
            "lightning is unavailable in the current environment. "
            "Create experiment/tft/.conda with ./create_env.sh."
        )
    if TimeSeriesDataSet is None or TemporalFusionTransformer is None or RMSE is None:
        raise ImportError(
            "pytorch-forecasting is unavailable in the current environment. "
            "Create experiment/tft/.conda with ./create_env.sh."
        )
    return resolved_torch, lightning, EarlyStopping, ModelCheckpoint, TimeSeriesDataSet, TemporalFusionTransformer


def release_process_memory() -> None:
    gc.collect()
    if torch is not None and bool(torch.cuda.is_available()):
        torch.cuda.empty_cache()


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


def resolve_batch_size(device: str) -> int:
    return DEFAULT_ACCELERATOR_BATCH_SIZE if device == "cuda" else DEFAULT_CPU_BATCH_SIZE


def resolve_num_workers(device: str) -> int:
    # The TFT pipeline materializes large in-memory pandas/TimeSeriesDataset
    # objects, so multi-worker DataLoaders add memory pressure faster than they
    # improve throughput on this repo's Kelmarsh pilot.
    del device
    return 0


def _resolve_lightning_accelerator(device: str) -> str:
    if device == "cuda":
        return "gpu"
    if device == "mps":
        return "mps"
    return "cpu"


def _resolve_trainer_deterministic(device: str) -> bool | str:
    # TFT currently hits a CUDA backward op without a deterministic kernel in
    # PyTorch, so strict deterministic mode fails on GPU-backed servers.
    return True if device == "cpu" else "warn"


def resolve_trainer_precision(device: str, precision: str | None = None) -> str:
    if precision is None or precision == "auto":
        return "bf16-mixed" if device == "cuda" else "32-true"
    return precision


def resolve_matmul_precision(device: str, precision: str | None = None) -> str | None:
    if precision is None or precision == "auto":
        return "high" if device == "cuda" else None
    return precision


def configure_torch_runtime(
    torch_module: Any,
    *,
    device: str,
    matmul_precision: str | None = None,
) -> str | None:
    resolved_matmul_precision = resolve_matmul_precision(device, matmul_precision)
    if resolved_matmul_precision is not None and hasattr(torch_module, "set_float32_matmul_precision"):
        torch_module.set_float32_matmul_precision(resolved_matmul_precision)

    if device != "cuda":
        return resolved_matmul_precision

    backends = getattr(torch_module, "backends", None)
    cuda_backend = getattr(backends, "cuda", None)
    matmul_backend = getattr(cuda_backend, "matmul", None)
    allow_tf32 = resolved_matmul_precision in {"high", "medium"}
    if matmul_backend is not None and hasattr(matmul_backend, "allow_tf32"):
        matmul_backend.allow_tf32 = allow_tf32
    cudnn_backend = getattr(backends, "cudnn", None)
    if cudnn_backend is not None and hasattr(cudnn_backend, "allow_tf32"):
        cudnn_backend.allow_tf32 = allow_tf32
    if cudnn_backend is not None and hasattr(cudnn_backend, "benchmark"):
        cudnn_backend.benchmark = True
    return resolved_matmul_precision


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
    task_dir = dataset_root / "tasks" / QUALITY_PROFILE / SERIES_LAYOUT / TASK_ID
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


def resolve_input_pack(dataset_id: str, input_pack: str) -> InputPackSpec:
    if input_pack == "reference":
        return InputPackSpec(
            dataset_id=dataset_id,
            input_pack=input_pack,
            feature_set="default",
            historical_pack=None,
            historical_covariate_stage=None,
            historical_source_columns=(),
            uses_static_covariates=False,
            uses_known_future_covariates=False,
        )
    if input_pack == "known_static":
        return InputPackSpec(
            dataset_id=dataset_id,
            input_pack=input_pack,
            feature_set="default",
            historical_pack=None,
            historical_covariate_stage=None,
            historical_source_columns=(),
            uses_static_covariates=True,
            uses_known_future_covariates=True,
        )
    if input_pack in {"hist_stage1", "mixed_stage1"}:
        historical_pack = resolve_covariate_pack(dataset_id, "stage1_core")
        return InputPackSpec(
            dataset_id=dataset_id,
            input_pack=input_pack,
            feature_set=historical_pack.feature_set,
            historical_pack=historical_pack,
            historical_covariate_stage="stage1_core",
            historical_source_columns=historical_pack.required_columns,
            uses_static_covariates=input_pack.startswith("mixed_"),
            uses_known_future_covariates=input_pack.startswith("mixed_"),
        )
    if input_pack in {"hist_stage2", "mixed_stage2"}:
        historical_pack = resolve_covariate_pack(dataset_id, "stage2_ops")
        return InputPackSpec(
            dataset_id=dataset_id,
            input_pack=input_pack,
            feature_set=historical_pack.feature_set,
            historical_pack=historical_pack,
            historical_covariate_stage="stage2_ops",
            historical_source_columns=historical_pack.required_columns,
            uses_static_covariates=input_pack.startswith("mixed_"),
            uses_known_future_covariates=input_pack.startswith("mixed_"),
        )
    raise ValueError(f"Unsupported input_pack {input_pack!r}. Expected one of {DEFAULT_INPUT_PACKS!r}.")


def build_requested_input_packs(
    dataset_id: str,
    *,
    input_packs: Sequence[str] = DEFAULT_INPUT_PACKS,
) -> tuple[InputPackSpec, ...]:
    return tuple(resolve_input_pack(dataset_id, input_pack) for input_pack in input_packs)


def load_series_frame(
    dataset_id: str,
    *,
    input_pack_spec: InputPackSpec,
    cache_root: str | Path = _CACHE_ROOT,
) -> tuple[str, pl.DataFrame]:
    resolved_feature_set, series_path = resolve_turbine_series_path(
        dataset_id,
        feature_set=input_pack_spec.feature_set,
        cache_root=cache_root,
    )
    available_columns = set(pl.read_parquet_schema(series_path))
    missing_base = [column for column in _SERIES_BASE_COLUMNS if column not in available_columns]
    if missing_base:
        raise ValueError(
            f"Series {series_path} for dataset {dataset_id!r} is missing required base columns {missing_base!r}."
        )
    missing_historical = [
        column
        for column in input_pack_spec.historical_source_columns
        if column not in available_columns
    ]
    if missing_historical:
        raise ValueError(
            f"Series {series_path} for dataset {dataset_id!r} is missing historical covariates {missing_historical!r}."
        )
    selected_columns = tuple(_SERIES_BASE_COLUMNS) + input_pack_spec.historical_source_columns
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
        input_pack=input_pack_spec.input_pack,
        feature_set=resolved_feature_set,
        rows=frame.height,
        columns=len(frame.columns),
        historical_covariates=len(input_pack_spec.historical_source_columns),
        duration_seconds=round(time.monotonic() - load_started, 6),
    )
    return resolved_feature_set, frame


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


def split_window_index(
    window_index: pl.DataFrame,
    *,
    raw_timestamps: Sequence[datetime],
    resolution_minutes: int,
) -> dict[str, pl.DataFrame]:
    return shared_split_window_index(
        window_index,
        raw_timestamps=raw_timestamps,
        resolution_minutes=resolution_minutes,
        history_steps=HISTORY_STEPS,
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


def build_known_future_feature_frame(frame: pl.DataFrame) -> pl.DataFrame:
    minutes_expr = (
        pl.col("timestamp").dt.hour().cast(pl.Int64) * 60
        + pl.col("timestamp").dt.minute().cast(pl.Int64)
    )
    time_angle_expr = minutes_expr.cast(pl.Float64) * (2.0 * math.pi / 1440.0)
    weekday_expr = (pl.col("timestamp").dt.weekday().cast(pl.Float64) - 1.0)
    dow_angle_expr = weekday_expr * (2.0 * math.pi / 7.0)
    return frame.with_columns(
        time_angle_expr.sin().alias("tod_sin"),
        time_angle_expr.cos().alias("tod_cos"),
        dow_angle_expr.sin().alias("dow_sin"),
        dow_angle_expr.cos().alias("dow_cos"),
    )


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


def _standardize_static_column(values: Sequence[float | None]) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    mean = float(np.nanmean(array))
    std = float(np.nanstd(array))
    if not math.isfinite(std) or std < 1e-12:
        std = 1.0
    return ((array - mean) / std).astype(np.float64)


def build_static_feature_frame(turbine_static: pl.DataFrame) -> pl.DataFrame:
    first_column, second_column = resolve_static_coordinate_columns(turbine_static)
    return pl.DataFrame(
        {
            "turbine_id": turbine_static["turbine_id"].to_list(),
            "static_coord_1": _standardize_static_column(turbine_static[first_column].to_list()).tolist(),
            "static_coord_2": _standardize_static_column(turbine_static[second_column].to_list()).tolist(),
        }
    )


def _cast_historical_expr(column: str, schema: Mapping[str, pl.DataType]) -> pl.Expr:
    dtype = schema[column]
    if dtype == pl.Boolean:
        return (
            pl.when(pl.col(column).is_null())
            .then(pl.lit(None, dtype=pl.Float64))
            .otherwise(pl.col(column).cast(pl.UInt8))
            .cast(pl.Float64)
        )
    return pl.col(column).cast(pl.Float64, strict=False)


def _safe_std(array: np.ndarray) -> float:
    std = float(np.nanstd(array))
    if not math.isfinite(std) or std < 1e-12:
        return 1.0
    return std


def fit_historical_statistics(
    frame: pl.DataFrame,
    *,
    historical_columns: Sequence[str],
) -> tuple[np.ndarray, np.ndarray]:
    if not historical_columns:
        return (
            np.empty((0,), dtype=np.float64),
            np.empty((0,), dtype=np.float64),
        )
    means: list[float] = []
    stds: list[float] = []
    for column in historical_columns:
        casted = _cast_historical_expr(column, frame.schema)
        values = frame.select(casted.alias(column))[column].to_numpy()
        means.append(float(np.nanmean(values)))
        stds.append(_safe_std(values))
    return np.asarray(means, dtype=np.float64), np.asarray(stds, dtype=np.float64)


def apply_historical_preprocessing(
    frame: pl.DataFrame,
    *,
    historical_columns: Sequence[str],
    means: np.ndarray,
    stds: np.ndarray,
) -> tuple[pl.DataFrame, tuple[str, ...], tuple[str, ...]]:
    if not historical_columns:
        return frame, (), ()

    expressions: list[pl.Expr] = []
    normalized_columns: list[str] = []
    mask_columns: list[str] = []
    for index, column in enumerate(historical_columns):
        feature_name = f"hist_{index:02d}"
        mask_name = f"{feature_name}_missing"
        casted = _cast_historical_expr(column, frame.schema)
        expressions.extend(
            [
                pl.when(casted.is_null())
                .then(pl.lit(0.0))
                .otherwise((casted - float(means[index])) / float(stds[index]))
                .cast(pl.Float64)
                .alias(feature_name),
                casted.is_null().cast(pl.Float64).alias(mask_name),
            ]
        )
        normalized_columns.append(feature_name)
        mask_columns.append(mask_name)
    return frame.with_columns(*expressions), tuple(normalized_columns), tuple(mask_columns)


def _normalize_target_expr(rated_power_kw: float) -> pl.Expr:
    return (
        pl.when(pl.col("target_kw").is_null())
        .then(pl.lit(None, dtype=pl.Float64))
        .otherwise(pl.col("target_kw").cast(pl.Float64).clip(0.0, rated_power_kw) / rated_power_kw)
        .alias("target_pu")
    )


def _row_index_expr() -> pl.Expr:
    return (pl.col("timestamp").cum_count().over("turbine_id") - 1).cast(pl.Int64).alias("time_idx")


def _build_timestamps_by_turbine(frame: pl.DataFrame) -> dict[str, np.ndarray]:
    mapping: dict[str, np.ndarray] = {}
    for turbine_frame in frame.partition_by("turbine_id", maintain_order=True):
        mapping[str(turbine_frame["turbine_id"][0])] = turbine_frame["timestamp"].cast(pl.Int64).to_numpy()
    return mapping


def build_window_descriptor_index(
    window_index: pl.DataFrame,
    *,
    turbine_ids: Sequence[str],
    timestamps_by_turbine: Mapping[str, Sequence[int] | np.ndarray],
) -> WindowDescriptorIndex:
    return shared_build_window_descriptor_index(
        window_index,
        turbine_ids=turbine_ids,
        timestamps_by_turbine=timestamps_by_turbine,
    )


def downsample_train_window_index(window_index: pl.DataFrame, *, every_n: int) -> pl.DataFrame:
    if every_n <= 1 or window_index.is_empty():
        return window_index
    sampled_parts: list[pl.DataFrame] = []
    for turbine_frame in window_index.partition_by("turbine_id", maintain_order=True):
        sampled_parts.append(
            turbine_frame.with_row_index("__row_index")
            .filter((pl.col("__row_index") % every_n) == 0)
            .drop("__row_index")
        )
    return pl.concat(sampled_parts, how="vertical") if sampled_parts else window_index.head(0)


def window_descriptor_to_origin_frame(
    dataset_id: str,
    turbine_ids: Sequence[str],
    windows: WindowDescriptorIndex,
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "series_id": [f"{dataset_id}::{turbine_ids[int(index)]}" for index in windows.turbine_indices.tolist()],
            "time_idx_first_prediction": windows.target_indices.astype(np.int64, copy=False),
        }
    )


def filter_decoded_origin_index(
    decoded_index: pd.DataFrame,
    allowed_origins: pd.DataFrame,
) -> np.ndarray:
    if decoded_index.empty:
        return np.zeros((0,), dtype=bool)
    marked_allowed = allowed_origins.copy()
    marked_allowed["__keep__"] = True
    merged = decoded_index.merge(
        marked_allowed,
        how="left",
        on=["series_id", "time_idx_first_prediction"],
    )
    return merged["__keep__"].fillna(False).to_numpy(dtype=bool, copy=False)


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


def prepare_dataset(
    dataset_id: str,
    *,
    input_pack_spec: InputPackSpec,
    cache_root: str | Path = _CACHE_ROOT,
    max_train_origins: int | None = None,
) -> PreparedDataset:
    metadata = load_dataset_metadata(dataset_id, cache_root=cache_root)
    strict_window_index = load_strict_window_index(dataset_id, cache_root=cache_root)
    resolved_feature_set, raw_series = load_series_frame(
        dataset_id,
        input_pack_spec=input_pack_spec,
        cache_root=cache_root,
    )
    resolution_minutes = resolve_resolution_minutes(raw_series)
    raw_timestamps = raw_series.select("timestamp").unique().sort("timestamp")["timestamp"].to_list()
    split_lookup = build_chrono_split_lookup(raw_timestamps)
    turbine_static = pl.read_parquet(metadata.task_paths.turbine_static_path)
    model_frame = (
        raw_series.join(split_lookup, on="timestamp", how="left")
        .with_columns(
            _row_index_expr(),
            pl.format("{}::{}", pl.lit(dataset_id), pl.col("turbine_id")).alias("series_id"),
            _normalize_target_expr(metadata.rated_power_kw),
        )
        .sort(["turbine_id", "timestamp"])
    )
    model_frame = model_frame.with_columns(
        pl.col("target_pu").is_null().cast(pl.Float64).alias(TARGET_MISSING_COLUMN),
        pl.col("target_pu").fill_null(0.0).alias("target_pu"),
    )
    if input_pack_spec.uses_known_future_covariates:
        model_frame = build_known_future_feature_frame(model_frame)
    if input_pack_spec.uses_static_covariates:
        model_frame = model_frame.join(
            build_static_feature_frame(turbine_static),
            on="turbine_id",
            how="left",
        )

    historical_means, historical_stds = fit_historical_statistics(
        model_frame.filter(pl.col("split") == "train"),
        historical_columns=input_pack_spec.historical_source_columns,
    )
    model_frame, historical_covariate_columns, historical_mask_columns = apply_historical_preprocessing(
        model_frame,
        historical_columns=input_pack_spec.historical_source_columns,
        means=historical_means,
        stds=historical_stds,
    )

    split_frames = split_window_index(
        strict_window_index,
        raw_timestamps=raw_timestamps,
        resolution_minutes=resolution_minutes,
    )
    train_frame = downsample_train_window_index(split_frames["train"], every_n=TRAIN_DOWNSAMPLE_EVERY)
    if max_train_origins is not None:
        train_frame = train_frame.head(max_train_origins)

    timestamps_by_turbine = _build_timestamps_by_turbine(model_frame.select(["turbine_id", "timestamp"]))
    train_windows = build_window_descriptor_index(
        train_frame,
        turbine_ids=metadata.turbine_ids,
        timestamps_by_turbine=timestamps_by_turbine,
    )
    val_rolling_windows = build_window_descriptor_index(
        split_frames["val"],
        turbine_ids=metadata.turbine_ids,
        timestamps_by_turbine=timestamps_by_turbine,
    )
    test_rolling_windows = build_window_descriptor_index(
        split_frames["test"],
        turbine_ids=metadata.turbine_ids,
        timestamps_by_turbine=timestamps_by_turbine,
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

    known_covariate_count = len(KNOWN_FUTURE_COLUMNS) + (1 if input_pack_spec.uses_known_future_covariates else 0)
    static_covariate_columns = STATIC_COORD_COLUMNS if input_pack_spec.uses_static_covariates else ()
    known_covariate_columns = KNOWN_FUTURE_COLUMNS if input_pack_spec.uses_known_future_covariates else ()
    covariate_policy = (
        COVARIATE_POLICY
        if (input_pack_spec.uses_static_covariates or input_pack_spec.uses_known_future_covariates or historical_covariate_columns)
        else REFERENCE_COVARIATE_POLICY
    )
    prepared = PreparedDataset(
        dataset_id=dataset_id,
        resolution_minutes=resolution_minutes,
        rated_power_kw=metadata.rated_power_kw,
        history_steps=HISTORY_STEPS,
        forecast_steps=FORECAST_STEPS,
        stride_steps=STRIDE_STEPS,
        input_pack=input_pack_spec.input_pack,
        historical_covariate_stage=input_pack_spec.historical_covariate_stage,
        feature_set=resolved_feature_set,
        uses_static_covariates=input_pack_spec.uses_static_covariates,
        uses_known_future_covariates=input_pack_spec.uses_known_future_covariates,
        static_covariate_columns=static_covariate_columns,
        known_covariate_columns=known_covariate_columns,
        historical_covariate_columns=historical_covariate_columns,
        historical_mask_columns=historical_mask_columns,
        static_covariate_count=len(static_covariate_columns),
        known_covariate_count=known_covariate_count if input_pack_spec.uses_known_future_covariates else 0,
        historical_covariate_count=input_pack_spec.historical_covariate_count,
        covariate_policy=covariate_policy,
        turbine_ids=metadata.turbine_ids,
        model_frame=model_frame.to_pandas(use_pyarrow_extension_array=False),
        train_windows=train_windows,
        val_rolling_windows=val_rolling_windows,
        val_non_overlap_windows=val_non_overlap_windows,
        test_rolling_windows=test_rolling_windows,
        test_non_overlap_windows=test_non_overlap_windows,
        train_origin_frame=window_descriptor_to_origin_frame(dataset_id, metadata.turbine_ids, train_windows),
        val_rolling_origin_frame=window_descriptor_to_origin_frame(dataset_id, metadata.turbine_ids, val_rolling_windows),
        val_non_overlap_origin_frame=window_descriptor_to_origin_frame(dataset_id, metadata.turbine_ids, val_non_overlap_windows),
        test_rolling_origin_frame=window_descriptor_to_origin_frame(dataset_id, metadata.turbine_ids, test_rolling_windows),
        test_non_overlap_origin_frame=window_descriptor_to_origin_frame(dataset_id, metadata.turbine_ids, test_non_overlap_windows),
    )
    _profile_log(
        dataset_id,
        "prepare_dataset_complete",
        input_pack=prepared.input_pack,
        feature_set=prepared.feature_set,
        historical_covariate_stage=prepared.historical_covariate_stage,
        static_covariate_count=prepared.static_covariate_count,
        known_covariate_count=prepared.known_covariate_count,
        historical_covariate_count=prepared.historical_covariate_count,
        train_windows=len(prepared.train_windows),
        val_rolling_windows=len(prepared.val_rolling_windows),
        val_non_overlap_windows=len(prepared.val_non_overlap_windows),
        test_rolling_windows=len(prepared.test_rolling_windows),
        test_non_overlap_windows=len(prepared.test_non_overlap_windows),
    )
    return prepared


def _build_timeseries_dataset_kwargs(prepared_dataset: PreparedDataset) -> dict[str, object]:
    known_reals = list(prepared_dataset.known_covariate_columns)
    unknown_reals = [
        "target_pu",
        TARGET_MISSING_COLUMN,
        *prepared_dataset.historical_covariate_columns,
        *prepared_dataset.historical_mask_columns,
    ]
    kwargs: dict[str, object] = {
        "time_idx": "time_idx",
        "target": "target_pu",
        "group_ids": ["series_id"],
        "max_encoder_length": prepared_dataset.history_steps,
        "min_encoder_length": prepared_dataset.history_steps,
        "max_prediction_length": prepared_dataset.forecast_steps,
        "min_prediction_length": prepared_dataset.forecast_steps,
        "time_varying_known_reals": known_reals,
        "time_varying_unknown_reals": unknown_reals,
        "allow_missing_timesteps": False,
        "add_relative_time_idx": prepared_dataset.uses_known_future_covariates,
        "target_normalizer": None,
        "randomize_length": None,
        "add_target_scales": False,
        "add_encoder_length": False,
    }
    if prepared_dataset.static_covariate_columns:
        kwargs["static_reals"] = list(prepared_dataset.static_covariate_columns)
    return kwargs


def build_frame_for_windows(prepared_dataset: PreparedDataset, windows: WindowDescriptorIndex) -> pd.DataFrame:
    if len(windows) == 0:
        return prepared_dataset.model_frame.iloc[0:0].copy()

    frames: list[pd.DataFrame] = []
    for turbine_index in np.unique(windows.turbine_indices):
        turbine_mask = windows.turbine_indices == turbine_index
        target_indices = windows.target_indices[turbine_mask].astype(np.int64, copy=False)
        start_index = int(target_indices.min()) - prepared_dataset.history_steps
        end_index = int(target_indices.max()) + prepared_dataset.forecast_steps - 1
        series_id = f"{prepared_dataset.dataset_id}::{prepared_dataset.turbine_ids[int(turbine_index)]}"
        subset = prepared_dataset.model_frame.loc[
            (prepared_dataset.model_frame["series_id"] == series_id)
            & (prepared_dataset.model_frame["time_idx"] >= start_index)
            & (prepared_dataset.model_frame["time_idx"] <= end_index)
        ].copy()
        frames.append(subset)
    return pd.concat(frames, ignore_index=True)


def filter_timeseries_dataset(dataset: Any, allowed_origins: pd.DataFrame) -> Any:
    filtered = dataset.filter(
        lambda decoded_index: filter_decoded_origin_index(decoded_index, allowed_origins),
        copy=True,
    )
    decoded_index = filtered.decoded_index
    if int(decoded_index.shape[0]) != int(allowed_origins.shape[0]):
        raise ValueError(
            "Filtered TimeSeriesDataSet does not match the expected strict-window origin count: "
            f"{decoded_index.shape[0]} != {allowed_origins.shape[0]}."
        )
    return filtered


def build_timeseries_datasets(prepared_dataset: PreparedDataset) -> BuiltTimeSeriesDatasets:
    _resolved_torch, _lightning, _EarlyStopping, _ModelCheckpoint, ResolvedTimeSeriesDataSet, _TemporalFusionTransformer = require_forecasting()
    kwargs = _build_timeseries_dataset_kwargs(prepared_dataset)
    train_frame = build_frame_for_windows(prepared_dataset, prepared_dataset.train_windows)
    val_rolling_frame = build_frame_for_windows(prepared_dataset, prepared_dataset.val_rolling_windows)
    val_non_overlap_frame = build_frame_for_windows(prepared_dataset, prepared_dataset.val_non_overlap_windows)
    test_rolling_frame = build_frame_for_windows(prepared_dataset, prepared_dataset.test_rolling_windows)
    test_non_overlap_frame = build_frame_for_windows(prepared_dataset, prepared_dataset.test_non_overlap_windows)
    train_dataset = ResolvedTimeSeriesDataSet(train_frame, **kwargs)
    train_dataset = filter_timeseries_dataset(train_dataset, prepared_dataset.train_origin_frame)
    val_rolling_dataset = ResolvedTimeSeriesDataSet.from_dataset(
        train_dataset,
        val_rolling_frame,
        stop_randomization=True,
        predict=False,
    )
    val_non_overlap_dataset = ResolvedTimeSeriesDataSet.from_dataset(
        train_dataset,
        val_non_overlap_frame,
        stop_randomization=True,
        predict=False,
    )
    test_rolling_dataset = ResolvedTimeSeriesDataSet.from_dataset(
        train_dataset,
        test_rolling_frame,
        stop_randomization=True,
        predict=False,
    )
    test_non_overlap_dataset = ResolvedTimeSeriesDataSet.from_dataset(
        train_dataset,
        test_non_overlap_frame,
        stop_randomization=True,
        predict=False,
    )
    return BuiltTimeSeriesDatasets(
        train_dataset=train_dataset,
        val_rolling_dataset=filter_timeseries_dataset(val_rolling_dataset, prepared_dataset.val_rolling_origin_frame),
        val_non_overlap_dataset=filter_timeseries_dataset(val_non_overlap_dataset, prepared_dataset.val_non_overlap_origin_frame),
        test_rolling_dataset=filter_timeseries_dataset(test_rolling_dataset, prepared_dataset.test_rolling_origin_frame),
        test_non_overlap_dataset=filter_timeseries_dataset(test_non_overlap_dataset, prepared_dataset.test_non_overlap_origin_frame),
    )


def _build_dataloader(
    dataset: Any,
    *,
    batch_size: int,
    train: bool,
    device: str,
    num_workers: int | None = None,
) -> Any:
    resolved_num_workers = resolve_num_workers(device) if num_workers is None else int(num_workers)
    dataloader_kwargs: dict[str, object] = {
        "train": train,
        "batch_size": batch_size,
        "num_workers": resolved_num_workers,
    }
    if device == "cuda":
        dataloader_kwargs["pin_memory"] = True
    if resolved_num_workers > 0:
        dataloader_kwargs["persistent_workers"] = True
        dataloader_kwargs["prefetch_factor"] = DEFAULT_PREFETCH_FACTOR
    return dataset.to_dataloader(
        **dataloader_kwargs,
    )


def _build_model(
    train_dataset: Any,
    *,
    learning_rate: float,
    hidden_size: int,
    attention_head_size: int,
    hidden_continuous_size: int,
    dropout: float,
) -> Any:
    _resolved_torch, _lightning, _EarlyStopping, _ModelCheckpoint, _TimeSeriesDataSet, ResolvedTemporalFusionTransformer = require_forecasting()
    return ResolvedTemporalFusionTransformer.from_dataset(
        train_dataset,
        learning_rate=learning_rate,
        hidden_size=hidden_size,
        attention_head_size=attention_head_size,
        hidden_continuous_size=hidden_continuous_size,
        dropout=dropout,
        output_size=1,
        loss=RMSE(),
        log_interval=-1,
        log_val_interval=-1,
        reduce_on_plateau_patience=1000,
    )


def _resolve_best_epoch(best_model_path: str, torch_module: Any) -> int:
    if not best_model_path:
        return 0
    try:
        # Lightning checkpoints are created locally in this run, so loading the
        # full object graph is expected and avoids PyTorch 2.6+ weights_only
        # restrictions when reading callback metadata such as `epoch`.
        checkpoint = torch_module.load(
            best_model_path,
            map_location="cpu",
            weights_only=False,
        )
    except TypeError:
        checkpoint = torch_module.load(best_model_path, map_location="cpu")
    epoch = checkpoint.get("epoch")
    if epoch is None:
        return 0
    return int(epoch) + 1


def _build_training_config(
    *,
    seed: int,
    batch_size: int,
    learning_rate: float,
    max_epochs: int,
    early_stopping_patience: int,
    hidden_size: int,
    attention_head_size: int,
    hidden_continuous_size: int,
    dropout: float,
    gradient_clip_val: float,
    num_workers: int,
    trainer_precision: str,
    matmul_precision: str | None,
) -> dict[str, object]:
    return {
        "seed": int(seed),
        "batch_size": int(batch_size),
        "learning_rate": float(learning_rate),
        "max_epochs": int(max_epochs),
        "early_stopping_patience": int(early_stopping_patience),
        "hidden_size": int(hidden_size),
        "attention_head_size": int(attention_head_size),
        "hidden_continuous_size": int(hidden_continuous_size),
        "dropout": float(dropout),
        "gradient_clip_val": float(gradient_clip_val),
        "num_workers": int(num_workers),
        "trainer_precision": trainer_precision,
        "matmul_precision": matmul_precision,
    }


def _reset_job_artifacts(paths: JobArtifactPaths) -> None:
    if paths.job_dir.exists():
        shutil.rmtree(paths.job_dir)


def _load_completed_training_outcome(
    *,
    paths: JobArtifactPaths,
    training_config: Mapping[str, object],
    model_loader: Any,
) -> TrainingOutcome | None:
    state = _read_training_state(paths.training_state_path)
    if state is None or state.get("status") != "training_complete":
        return None
    if state.get("training_config") != dict(training_config):
        return None
    best_model_path = state.get("best_model_path")
    if not isinstance(best_model_path, str) or not Path(best_model_path).exists():
        return None
    return TrainingOutcome(
        best_epoch=int(state.get("best_epoch", 0)),
        epochs_ran=int(state.get("epochs_ran", 0)),
        best_val_rmse_pu=float(state.get("best_val_rmse_pu", float("nan"))),
        device=str(state.get("device", "cpu")),
        model=model_loader.load_from_checkpoint(best_model_path),
    )


def _resolve_resume_checkpoint(
    *,
    paths: JobArtifactPaths,
    training_config: Mapping[str, object],
) -> str | None:
    state = _read_training_state(paths.training_state_path)
    if state is not None and state.get("training_config") != dict(training_config):
        return None
    last_checkpoint = paths.checkpoint_dir / "last.ckpt"
    if last_checkpoint.exists():
        return str(last_checkpoint)
    return None


def train_model(
    prepared_dataset: PreparedDataset,
    *,
    device: str,
    built_datasets: BuiltTimeSeriesDatasets | None = None,
    seed: int = DEFAULT_SEED,
    batch_size: int | None = None,
    learning_rate: float = DEFAULT_LEARNING_RATE,
    max_epochs: int = DEFAULT_MAX_EPOCHS,
    early_stopping_patience: int = DEFAULT_EARLY_STOPPING_PATIENCE,
    hidden_size: int = DEFAULT_HIDDEN_SIZE,
    attention_head_size: int = DEFAULT_ATTENTION_HEAD_SIZE,
    hidden_continuous_size: int = DEFAULT_HIDDEN_CONTINUOUS_SIZE,
    dropout: float = DEFAULT_DROPOUT,
    gradient_clip_val: float = DEFAULT_GRADIENT_CLIP_VAL,
    num_workers: int | None = None,
    trainer_precision: str | None = None,
    matmul_precision: str | None = None,
    job_artifacts: JobArtifactPaths | None = None,
    resume: bool = False,
) -> TrainingOutcome:
    resolved_torch, resolved_lightning, ResolvedEarlyStopping, ResolvedModelCheckpoint, _ResolvedTimeSeriesDataSet, ResolvedTemporalFusionTransformer = require_forecasting()
    resolved_device = resolve_device(device)
    resolved_batch_size = batch_size or resolve_batch_size(resolved_device)
    resolved_num_workers = resolve_num_workers(resolved_device) if num_workers is None else int(num_workers)
    resolved_trainer_precision = resolve_trainer_precision(resolved_device, trainer_precision)
    resolved_matmul_precision = configure_torch_runtime(
        resolved_torch,
        device=resolved_device,
        matmul_precision=matmul_precision,
    )
    training_config = _build_training_config(
        seed=seed,
        batch_size=resolved_batch_size,
        learning_rate=learning_rate,
        max_epochs=max_epochs,
        early_stopping_patience=early_stopping_patience,
        hidden_size=hidden_size,
        attention_head_size=attention_head_size,
        hidden_continuous_size=hidden_continuous_size,
        dropout=dropout,
        gradient_clip_val=gradient_clip_val,
        num_workers=resolved_num_workers,
        trainer_precision=resolved_trainer_precision,
        matmul_precision=resolved_matmul_precision,
    )
    resume_checkpoint_path: str | None = None
    if job_artifacts is not None:
        if resume:
            cached_outcome = _load_completed_training_outcome(
                paths=job_artifacts,
                training_config=training_config,
                model_loader=ResolvedTemporalFusionTransformer,
            )
            if cached_outcome is not None:
                _profile_log(
                    prepared_dataset.dataset_id,
                    "fit_reuse",
                    input_pack=prepared_dataset.input_pack,
                    checkpoint_path=str(job_artifacts.training_state_path),
                )
                return cached_outcome
            resume_checkpoint_path = _resolve_resume_checkpoint(
                paths=job_artifacts,
                training_config=training_config,
            )
            if resume_checkpoint_path is None:
                state = _read_training_state(job_artifacts.training_state_path)
                if state is not None and state.get("training_config") != training_config:
                    _profile_log(
                        prepared_dataset.dataset_id,
                        "fit_resume_reset",
                        input_pack=prepared_dataset.input_pack,
                        reason="config_changed",
                    )
                    _reset_job_artifacts(job_artifacts)
        else:
            _reset_job_artifacts(job_artifacts)
        job_artifacts.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        _write_json_atomic(
            job_artifacts.training_state_path,
            {
                "dataset_id": prepared_dataset.dataset_id,
                "device": resolved_device,
                "input_pack": prepared_dataset.input_pack,
                "status": "fit_in_progress",
                "training_config": training_config,
            },
        )
    resolved_torch.manual_seed(seed)
    resolved_built_datasets = built_datasets or build_timeseries_datasets(prepared_dataset)
    train_loader = _build_dataloader(
        resolved_built_datasets.train_dataset,
        batch_size=resolved_batch_size,
        train=True,
        device=resolved_device,
        num_workers=resolved_num_workers,
    )
    val_loader = _build_dataloader(
        resolved_built_datasets.val_rolling_dataset,
        batch_size=resolved_batch_size,
        train=False,
        device=resolved_device,
        num_workers=resolved_num_workers,
    )
    model = _build_model(
        resolved_built_datasets.train_dataset,
        learning_rate=learning_rate,
        hidden_size=hidden_size,
        attention_head_size=attention_head_size,
        hidden_continuous_size=hidden_continuous_size,
        dropout=dropout,
    )
    early_stopping = ResolvedEarlyStopping(
        monitor="val_loss",
        patience=early_stopping_patience,
        mode="min",
    )
    checkpoint = ResolvedModelCheckpoint(
        monitor="val_loss",
        mode="min",
        save_top_k=1,
        save_last=True,
        save_weights_only=False,
        dirpath=str(job_artifacts.checkpoint_dir) if job_artifacts is not None else None,
        filename="{epoch:02d}-{val_loss:.6f}",
    )
    _profile_log(
        prepared_dataset.dataset_id,
        "fit_start",
        input_pack=prepared_dataset.input_pack,
        batch_size=resolved_batch_size,
        num_workers=resolved_num_workers,
        trainer_precision=resolved_trainer_precision,
        matmul_precision=resolved_matmul_precision,
        hidden_size=hidden_size,
        attention_head_size=attention_head_size,
        hidden_continuous_size=hidden_continuous_size,
        dropout=dropout,
        gradient_clip_val=gradient_clip_val,
    )
    fit_started = time.monotonic()
    temp_dir_manager = tempfile.TemporaryDirectory(prefix="tft-pilot-") if job_artifacts is None else None
    try:
        trainer_root_dir = Path(temp_dir_manager.name) if temp_dir_manager is not None else job_artifacts.job_dir
        if job_artifacts is not None:
            trainer_root_dir.mkdir(parents=True, exist_ok=True)
        trainer = resolved_lightning.Trainer(
            max_epochs=max_epochs,
            accelerator=_resolve_lightning_accelerator(resolved_device),
            devices=1,
            logger=False,
            enable_model_summary=False,
            enable_progress_bar=progress_is_enabled(),
            gradient_clip_val=gradient_clip_val,
            deterministic=_resolve_trainer_deterministic(resolved_device),
            precision=resolved_trainer_precision,
            callbacks=[early_stopping, checkpoint],
            default_root_dir=str(trainer_root_dir),
        )
        if resume_checkpoint_path is not None:
            _profile_log(
                prepared_dataset.dataset_id,
                "fit_resume",
                input_pack=prepared_dataset.input_pack,
                checkpoint_path=resume_checkpoint_path,
            )
            trainer.fit(
                model,
                train_dataloaders=train_loader,
                val_dataloaders=val_loader,
                ckpt_path=resume_checkpoint_path,
            )
        else:
            trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)
        best_model_path = checkpoint.best_model_path
        best_val_rmse_pu = float(checkpoint.best_model_score.cpu().item()) if checkpoint.best_model_score is not None else float("nan")
        best_epoch = _resolve_best_epoch(best_model_path, resolved_torch)
        epochs_ran = max(int(getattr(trainer, "current_epoch", -1)) + 1, best_epoch, 1)
        if best_model_path:
            model = ResolvedTemporalFusionTransformer.load_from_checkpoint(best_model_path)
    finally:
        if temp_dir_manager is not None:
            temp_dir_manager.cleanup()
    if job_artifacts is not None and best_model_path:
        _write_json_atomic(
            job_artifacts.training_state_path,
            {
                "best_epoch": best_epoch,
                "best_model_path": str(Path(best_model_path).resolve()),
                "best_val_rmse_pu": best_val_rmse_pu,
                "dataset_id": prepared_dataset.dataset_id,
                "device": resolved_device,
                "epochs_ran": epochs_ran,
                "input_pack": prepared_dataset.input_pack,
                "status": "training_complete",
                "training_config": training_config,
            },
        )
    _profile_log(
        prepared_dataset.dataset_id,
        "fit_complete",
        input_pack=prepared_dataset.input_pack,
        batch_size=resolved_batch_size,
        num_workers=resolved_num_workers,
        trainer_precision=resolved_trainer_precision,
        matmul_precision=resolved_matmul_precision,
        best_epoch=best_epoch,
        epochs_ran=epochs_ran,
        best_val_rmse_pu=best_val_rmse_pu,
        duration_seconds=round(time.monotonic() - fit_started, 6),
    )
    return TrainingOutcome(
        best_epoch=best_epoch,
        epochs_ran=epochs_ran,
        best_val_rmse_pu=best_val_rmse_pu,
        device=resolved_device,
        model=model,
    )


def _to_numpy(value: Any) -> np.ndarray:
    if isinstance(value, np.ndarray):
        return value
    if hasattr(value, "detach"):
        return value.detach().cpu().numpy()
    if hasattr(value, "to_numpy"):
        return value.to_numpy()
    return np.asarray(value)


def _unwrap_prediction_output(prediction_output: Any) -> np.ndarray:
    if hasattr(prediction_output, "output"):
        return _unwrap_prediction_output(prediction_output.output)
    if hasattr(prediction_output, "prediction"):
        return _unwrap_prediction_output(prediction_output.prediction)
    if isinstance(prediction_output, Mapping):
        for key in ("prediction", "output"):
            if key in prediction_output:
                return _unwrap_prediction_output(prediction_output[key])
    array = _to_numpy(prediction_output)
    if array.ndim == 3 and array.shape[-1] == 1:
        array = array[..., 0]
    return array.astype(np.float64, copy=False)


def _unwrap_prediction_targets(prediction_output: Any) -> np.ndarray:
    y_value = getattr(prediction_output, "y", prediction_output)
    if isinstance(y_value, tuple):
        y_value = y_value[0]
    if isinstance(y_value, Mapping) and "y" in y_value:
        y_value = y_value["y"]
    array = _to_numpy(y_value)
    if array.ndim == 3 and array.shape[-1] == 1:
        array = array[..., 0]
    return array.astype(np.float64, copy=False)


def evaluate_predictions(
    predictions_pu: np.ndarray,
    actual_pu: np.ndarray,
    *,
    rated_power_kw: float,
) -> EvaluationMetrics:
    if predictions_pu.shape != actual_pu.shape:
        raise ValueError(
            f"Prediction/actual shape mismatch: {predictions_pu.shape!r} != {actual_pu.shape!r}."
        )
    valid_mask = np.isfinite(actual_pu)
    if not valid_mask.any():
        raise ValueError("No finite targets are available for evaluation.")
    errors_pu = predictions_pu - actual_pu
    errors_kw = errors_pu * rated_power_kw
    absolute_errors_pu = np.where(valid_mask, np.abs(errors_pu), 0.0)
    squared_errors_pu = np.where(valid_mask, np.square(errors_pu), 0.0)
    absolute_errors_kw = np.where(valid_mask, np.abs(errors_kw), 0.0)
    squared_errors_kw = np.where(valid_mask, np.square(errors_kw), 0.0)
    horizon_prediction_count = valid_mask.sum(axis=0).astype(np.int64, copy=False)
    horizon_window_count = valid_mask.any(axis=0).astype(np.int64, copy=False) * predictions_pu.shape[0]
    prediction_count = int(valid_mask.sum())
    window_count = int(valid_mask.any(axis=1).sum())
    return EvaluationMetrics(
        window_count=window_count,
        prediction_count=prediction_count,
        mae_kw=_safe_divide(float(absolute_errors_kw.sum()), prediction_count),
        rmse_kw=_safe_rmse(float(squared_errors_kw.sum()), prediction_count),
        mae_pu=_safe_divide(float(absolute_errors_pu.sum()), prediction_count),
        rmse_pu=_safe_rmse(float(squared_errors_pu.sum()), prediction_count),
        horizon_window_count=horizon_window_count,
        horizon_prediction_count=horizon_prediction_count,
        horizon_mae_kw=np.asarray(
            [
                _safe_divide(float(absolute_errors_kw[:, index].sum()), int(horizon_prediction_count[index]))
                for index in range(predictions_pu.shape[1])
            ],
            dtype=np.float64,
        ),
        horizon_rmse_kw=np.asarray(
            [
                _safe_rmse(float(squared_errors_kw[:, index].sum()), int(horizon_prediction_count[index]))
                for index in range(predictions_pu.shape[1])
            ],
            dtype=np.float64,
        ),
        horizon_mae_pu=np.asarray(
            [
                _safe_divide(float(absolute_errors_pu[:, index].sum()), int(horizon_prediction_count[index]))
                for index in range(predictions_pu.shape[1])
            ],
            dtype=np.float64,
        ),
        horizon_rmse_pu=np.asarray(
            [
                _safe_rmse(float(squared_errors_pu[:, index].sum()), int(horizon_prediction_count[index]))
                for index in range(predictions_pu.shape[1])
            ],
            dtype=np.float64,
        ),
    )


def evaluate_model(
    model: Any,
    dataset: Any,
    *,
    device: str,
    batch_size: int,
    num_workers: int | None = None,
    rated_power_kw: float,
) -> EvaluationMetrics:
    resolved_device = resolve_device(device)
    loader = _build_dataloader(
        dataset,
        batch_size=batch_size,
        train=False,
        device=resolved_device,
        num_workers=num_workers,
    )
    trainer_kwargs = {
        "accelerator": _resolve_lightning_accelerator(resolved_device),
        "devices": 1,
        "logger": False,
        "enable_progress_bar": False,
    }
    prediction_output = model.predict(
        loader,
        mode="prediction",
        return_index=False,
        return_y=True,
        trainer_kwargs=trainer_kwargs,
    )
    predictions = _unwrap_prediction_output(prediction_output)
    actual = _unwrap_prediction_targets(prediction_output)
    return evaluate_predictions(predictions, actual, rated_power_kw=rated_power_kw)


def _safe_divide(numerator: float, denominator: int) -> float:
    if denominator <= 0:
        return float("nan")
    return numerator / denominator


def _safe_rmse(squared_error_sum: float, denominator: int) -> float:
    if denominator <= 0:
        return float("nan")
    return math.sqrt(squared_error_sum / denominator)


def iter_evaluation_specs(
    prepared_dataset: PreparedDataset,
    built_datasets: BuiltTimeSeriesDatasets,
) -> tuple[tuple[str, str, WindowDescriptorIndex, Any], ...]:
    return (
        ("val", ROLLING_EVAL_PROTOCOL, prepared_dataset.val_rolling_windows, built_datasets.val_rolling_dataset),
        ("val", NON_OVERLAP_EVAL_PROTOCOL, prepared_dataset.val_non_overlap_windows, built_datasets.val_non_overlap_dataset),
        ("test", ROLLING_EVAL_PROTOCOL, prepared_dataset.test_rolling_windows, built_datasets.test_rolling_dataset),
        ("test", NON_OVERLAP_EVAL_PROTOCOL, prepared_dataset.test_non_overlap_windows, built_datasets.test_non_overlap_dataset),
    )


def build_result_rows(
    prepared_dataset: PreparedDataset,
    *,
    training_outcome: TrainingOutcome,
    runtime_seconds: float,
    seed: int,
    batch_size: int,
    learning_rate: float,
    hidden_size: int,
    attention_head_size: int,
    hidden_continuous_size: int,
    dropout: float,
    gradient_clip_val: float,
    num_workers: int,
    trainer_precision: str,
    matmul_precision: str | None,
    evaluation_results: Sequence[tuple[str, str, WindowDescriptorIndex, EvaluationMetrics]],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    base_row = {
        "dataset_id": prepared_dataset.dataset_id,
        "model_id": MODEL_ID,
        "task_id": TASK_ID,
        "window_protocol": WINDOW_PROTOCOL,
        "history_steps": prepared_dataset.history_steps,
        "forecast_steps": prepared_dataset.forecast_steps,
        "stride_steps": prepared_dataset.stride_steps,
        "split_protocol": SPLIT_PROTOCOL,
        "input_pack": prepared_dataset.input_pack,
        "historical_covariate_stage": prepared_dataset.historical_covariate_stage,
        "feature_set": prepared_dataset.feature_set,
        "uses_static_covariates": prepared_dataset.uses_static_covariates,
        "uses_known_future_covariates": prepared_dataset.uses_known_future_covariates,
        "static_covariate_count": prepared_dataset.static_covariate_count,
        "known_covariate_count": prepared_dataset.known_covariate_count,
        "historical_covariate_count": prepared_dataset.historical_covariate_count,
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
        "hidden_size": hidden_size,
        "attention_head_size": attention_head_size,
        "hidden_continuous_size": hidden_continuous_size,
        "dropout": dropout,
        "gradient_clip_val": gradient_clip_val,
        "num_workers": num_workers,
        "trainer_precision": trainer_precision,
        "matmul_precision": matmul_precision,
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
    batch_size: int | None = None,
    learning_rate: float = DEFAULT_LEARNING_RATE,
    max_epochs: int = DEFAULT_MAX_EPOCHS,
    early_stopping_patience: int = DEFAULT_EARLY_STOPPING_PATIENCE,
    hidden_size: int = DEFAULT_HIDDEN_SIZE,
    attention_head_size: int = DEFAULT_ATTENTION_HEAD_SIZE,
    hidden_continuous_size: int = DEFAULT_HIDDEN_CONTINUOUS_SIZE,
    dropout: float = DEFAULT_DROPOUT,
    gradient_clip_val: float = DEFAULT_GRADIENT_CLIP_VAL,
    num_workers: int | None = None,
    trainer_precision: str | None = None,
    matmul_precision: str | None = None,
    job_artifacts: JobArtifactPaths | None = None,
    resume: bool = False,
) -> list[dict[str, object]]:
    dataset_start = time.monotonic()
    resolved_device = resolve_device(device)
    resolved_batch_size = batch_size or resolve_batch_size(resolved_device)
    resolved_num_workers = resolve_num_workers(resolved_device) if num_workers is None else int(num_workers)
    resolved_trainer_precision = resolve_trainer_precision(resolved_device, trainer_precision)
    resolved_matmul_precision = resolve_matmul_precision(resolved_device, matmul_precision)
    built_datasets = build_timeseries_datasets(prepared_dataset)
    training_outcome = train_model(
        prepared_dataset,
        device=resolved_device,
        built_datasets=built_datasets,
        seed=seed,
        batch_size=resolved_batch_size,
        learning_rate=learning_rate,
        max_epochs=max_epochs,
        early_stopping_patience=early_stopping_patience,
        hidden_size=hidden_size,
        attention_head_size=attention_head_size,
        hidden_continuous_size=hidden_continuous_size,
        dropout=dropout,
        gradient_clip_val=gradient_clip_val,
        num_workers=resolved_num_workers,
        trainer_precision=resolved_trainer_precision,
        matmul_precision=resolved_matmul_precision,
        job_artifacts=job_artifacts,
        resume=resume,
    )
    evaluation_results: list[tuple[str, str, WindowDescriptorIndex, EvaluationMetrics]] = []
    eval_specs = iter_evaluation_specs(prepared_dataset, built_datasets)
    eval_progress = _create_progress_bar(total=len(eval_specs), desc=f"{prepared_dataset.input_pack} eval")
    try:
        for split_name, eval_protocol, windows, dataset in eval_specs:
            metrics = evaluate_model(
                training_outcome.model,
                dataset,
                device=training_outcome.device,
                batch_size=resolved_batch_size,
                num_workers=resolved_num_workers,
                rated_power_kw=prepared_dataset.rated_power_kw,
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
        input_pack=prepared_dataset.input_pack,
        best_epoch=training_outcome.best_epoch,
        epochs_ran=training_outcome.epochs_ran,
        best_val_rmse_pu=training_outcome.best_val_rmse_pu,
        test_rolling_rmse_pu=test_rolling_metrics.rmse_pu,
        runtime_seconds=round(runtime_seconds, 6),
        batch_size=resolved_batch_size,
        num_workers=resolved_num_workers,
        trainer_precision=resolved_trainer_precision,
        matmul_precision=resolved_matmul_precision,
        hidden_size=hidden_size,
        attention_head_size=attention_head_size,
        hidden_continuous_size=hidden_continuous_size,
        dropout=dropout,
    )
    return build_result_rows(
        prepared_dataset,
        training_outcome=training_outcome,
        runtime_seconds=runtime_seconds,
        seed=seed,
        batch_size=resolved_batch_size,
        learning_rate=learning_rate,
        hidden_size=hidden_size,
        attention_head_size=attention_head_size,
        hidden_continuous_size=hidden_continuous_size,
        dropout=dropout,
        gradient_clip_val=gradient_clip_val,
        num_workers=resolved_num_workers,
        trainer_precision=resolved_trainer_precision,
        matmul_precision=resolved_matmul_precision,
        evaluation_results=evaluation_results,
    )


def sort_result_frame(frame: pl.DataFrame) -> pl.DataFrame:
    return (
        frame.with_columns(
            pl.col("dataset_id")
            .replace_strict(_DATASET_ORDER, default=len(_DATASET_ORDER))
            .alias("__dataset_order"),
            pl.col("input_pack")
            .replace_strict(_INPUT_PACK_ORDER, default=len(_INPUT_PACK_ORDER))
            .alias("__input_pack_order"),
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
                "__input_pack_order",
                "__split_order",
                "__eval_protocol_order",
                "__metric_scope_order",
                "__lead_order",
            ]
        )
        .drop(
            [
                "__dataset_order",
                "__input_pack_order",
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
    input_packs: Sequence[str] = DEFAULT_INPUT_PACKS,
    cache_root: str | Path = _CACHE_ROOT,
    output_path: str | Path = _OUTPUT_PATH,
    device: str | None = None,
    max_epochs: int = DEFAULT_MAX_EPOCHS,
    max_train_origins: int | None = None,
    seed: int = DEFAULT_SEED,
    batch_size: int | None = None,
    learning_rate: float = DEFAULT_LEARNING_RATE,
    early_stopping_patience: int = DEFAULT_EARLY_STOPPING_PATIENCE,
    hidden_size: int = DEFAULT_HIDDEN_SIZE,
    attention_head_size: int = DEFAULT_ATTENTION_HEAD_SIZE,
    hidden_continuous_size: int = DEFAULT_HIDDEN_CONTINUOUS_SIZE,
    dropout: float = DEFAULT_DROPOUT,
    gradient_clip_val: float = DEFAULT_GRADIENT_CLIP_VAL,
    num_workers: int | None = None,
    trainer_precision: str | None = None,
    matmul_precision: str | None = None,
    resume: bool = False,
    dataset_loader: Callable[..., PreparedDataset] | None = None,
    job_runner: Callable[..., list[dict[str, object]]] | None = None,
) -> pl.DataFrame:
    unknown_packs = [input_pack for input_pack in input_packs if input_pack not in DEFAULT_INPUT_PACKS]
    if unknown_packs:
        raise ValueError(f"Unsupported input packs: {unknown_packs!r}")
    loader = dataset_loader or prepare_dataset
    runner = job_runner or execute_training_job
    output = Path(output_path)
    results = _read_results_frame(output) if resume else None
    total_jobs = len(dataset_ids) * len(input_packs)
    job_progress = _create_progress_bar(total=total_jobs, desc="tft jobs", leave=True)
    try:
        for dataset_id in dataset_ids:
            for input_pack_spec in build_requested_input_packs(dataset_id, input_packs=input_packs):
                job_progress.set_postfix_str(f"{dataset_id}/{input_pack_spec.input_pack}")
                if resume and _has_complete_job_results(
                    results,
                    dataset_id=dataset_id,
                    input_pack=input_pack_spec.input_pack,
                ):
                    _profile_log(
                        dataset_id,
                        "job_skip_complete",
                        input_pack=input_pack_spec.input_pack,
                        output_path=str(output),
                    )
                    job_progress.update(1)
                    continue
                prepared = loader(
                    dataset_id,
                    input_pack_spec=input_pack_spec,
                    cache_root=cache_root,
                    max_train_origins=max_train_origins,
                )
                job_rows = runner(
                    prepared,
                    device=device,
                    seed=seed,
                    batch_size=batch_size,
                    learning_rate=learning_rate,
                    max_epochs=max_epochs,
                    early_stopping_patience=early_stopping_patience,
                    hidden_size=hidden_size,
                    attention_head_size=attention_head_size,
                    hidden_continuous_size=hidden_continuous_size,
                    dropout=dropout,
                    gradient_clip_val=gradient_clip_val,
                    num_workers=num_workers,
                    trainer_precision=trainer_precision,
                    matmul_precision=matmul_precision,
                    job_artifacts=build_job_artifact_paths(dataset_id, input_pack_spec.input_pack),
                    resume=resume,
                )
                results = _merge_job_results(
                    results,
                    dataset_id=prepared.dataset_id,
                    input_pack=prepared.input_pack,
                    job_rows=job_rows,
                )
                _write_result_frame(results, output)
                del job_rows
                del prepared
                release_process_memory()
                job_progress.update(1)
    finally:
        job_progress.close()
    if results is None:
        return pl.DataFrame(schema={column: pl.String for column in _RESULT_COLUMNS}).select(_RESULT_COLUMNS)
    return sort_result_frame(results.select(_RESULT_COLUMNS))


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the Kelmarsh TFT pilot with strict-window evaluation and train-only origin downsampling.",
    )
    parser.add_argument(
        "--dataset",
        action="append",
        default=[],
        help="Dataset id to run. Defaults to kelmarsh when omitted.",
    )
    parser.add_argument(
        "--input-pack",
        action="append",
        default=[],
        help="Input pack to run. Repeat to run multiple packs, or use --input-pack all. Defaults to all packs.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        choices=("auto", "cuda", "mps", "cpu"),
        help="Execution device. Defaults to automatic selection.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=DEFAULT_MAX_EPOCHS,
        help=f"Maximum training epochs. Defaults to {DEFAULT_MAX_EPOCHS}.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help=f"Random seed. Defaults to {DEFAULT_SEED}.",
    )
    parser.add_argument(
        "--max-train-origins",
        type=int,
        default=None,
        help="Optional cap on the downsampled train origins for smoke runs.",
    )
    parser.add_argument(
        "--output-path",
        default=str(_OUTPUT_PATH),
        help=f"Output CSV path. Defaults to {_OUTPUT_PATH}.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help=(
            "Resume a previously interrupted run: skip input packs already present in "
            "the output CSV and continue unfinished training from ./.work/jobs/.../checkpoints/last.ckpt when available."
        ),
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Mini-batch size. Defaults to device-specific automatic selection.",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=DEFAULT_LEARNING_RATE,
        help=f"Optimizer learning rate. Defaults to {DEFAULT_LEARNING_RATE}.",
    )
    parser.add_argument(
        "--early-stopping-patience",
        type=int,
        default=DEFAULT_EARLY_STOPPING_PATIENCE,
        help=f"Validation early stopping patience. Defaults to {DEFAULT_EARLY_STOPPING_PATIENCE}.",
    )
    parser.add_argument(
        "--hidden-size",
        type=int,
        default=DEFAULT_HIDDEN_SIZE,
        help=f"TFT hidden size. Defaults to {DEFAULT_HIDDEN_SIZE}.",
    )
    parser.add_argument(
        "--attention-head-size",
        type=int,
        default=DEFAULT_ATTENTION_HEAD_SIZE,
        help=f"Multi-head attention head count. Defaults to {DEFAULT_ATTENTION_HEAD_SIZE}.",
    )
    parser.add_argument(
        "--hidden-continuous-size",
        type=int,
        default=DEFAULT_HIDDEN_CONTINUOUS_SIZE,
        help=f"Continuous variable hidden size. Defaults to {DEFAULT_HIDDEN_CONTINUOUS_SIZE}.",
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=DEFAULT_DROPOUT,
        help=f"Dropout rate. Defaults to {DEFAULT_DROPOUT}.",
    )
    parser.add_argument(
        "--gradient-clip-val",
        type=float,
        default=DEFAULT_GRADIENT_CLIP_VAL,
        help=f"Gradient clipping value. Defaults to {DEFAULT_GRADIENT_CLIP_VAL}.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=None,
        help="DataLoader worker count. Defaults to device-specific automatic selection.",
    )
    parser.add_argument(
        "--trainer-precision",
        default="auto",
        choices=TRAINER_PRECISION_CHOICES,
        help="Lightning precision mode. Defaults to CUDA=bf16-mixed, otherwise 32-true.",
    )
    parser.add_argument(
        "--matmul-precision",
        default="auto",
        choices=MATMUL_PRECISION_CHOICES,
        help="torch float32 matmul precision. Defaults to CUDA=high, otherwise unchanged.",
    )
    return parser


def _normalize_cli_input_packs(values: Sequence[str]) -> tuple[str, ...]:
    if not values:
        return DEFAULT_INPUT_PACKS
    normalized: list[str] = []
    for value in values:
        if value == "all":
            normalized.extend(DEFAULT_INPUT_PACKS)
        else:
            normalized.append(value)
    deduped = tuple(dict.fromkeys(normalized))
    return deduped or DEFAULT_INPUT_PACKS


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    dataset_ids = tuple(dict.fromkeys(args.dataset)) if args.dataset else DEFAULT_DATASETS
    input_packs = _normalize_cli_input_packs(args.input_pack)
    run_experiment(
        dataset_ids=dataset_ids,
        input_packs=input_packs,
        output_path=args.output_path,
        device=args.device,
        max_epochs=args.epochs,
        max_train_origins=args.max_train_origins,
        seed=args.seed,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        early_stopping_patience=args.early_stopping_patience,
        hidden_size=args.hidden_size,
        attention_head_size=args.attention_head_size,
        hidden_continuous_size=args.hidden_continuous_size,
        dropout=args.dropout,
        gradient_clip_val=args.gradient_clip_val,
        num_workers=args.num_workers,
        trainer_precision=args.trainer_precision,
        matmul_precision=args.matmul_precision,
        resume=args.resume,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
