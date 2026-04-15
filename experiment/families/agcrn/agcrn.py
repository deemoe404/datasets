from __future__ import annotations

import argparse
import copy
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
import hashlib
import json
import math
from pathlib import Path
import random
import shutil
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
POWER_WS_HIST_MODEL_VARIANT = "official_aligned_power_ws_hist_farm_sync"
POWER_ATEMP_HIST_MODEL_VARIANT = "official_aligned_power_atemp_hist_farm_sync"
POWER_ITEMP_HIST_MODEL_VARIANT = "official_aligned_power_itemp_hist_farm_sync"
POWER_WD_HIST_SINCOS_MODEL_VARIANT = "official_aligned_power_wd_hist_sincos_farm_sync"
POWER_WD_YAW_HIST_SINCOS_MODEL_VARIANT = "official_aligned_power_wd_yaw_hist_sincos_farm_sync"
POWER_WD_YAW_PITCHMEAN_HIST_SINCOS_MODEL_VARIANT = (
    "official_aligned_power_wd_yaw_pitchmean_hist_sincos_farm_sync"
)
POWER_WD_YAW_LRPM_HIST_SINCOS_MODEL_VARIANT = "official_aligned_power_wd_yaw_lrpm_hist_sincos_farm_sync"
POWER_WS_WD_HIST_SINCOS_MODEL_VARIANT = "official_aligned_power_ws_wd_hist_sincos_farm_sync"
WINDOW_PROTOCOL = DEFAULT_WINDOW_PROTOCOL
TASK_PROTOCOL: WindowProtocolSpec = resolve_window_protocol(WINDOW_PROTOCOL)
TASK_ID = TASK_PROTOCOL.task_id
DEFAULT_DATASETS = ("kelmarsh", "penmanshiel")
HISTORY_STEPS = 144
FORECAST_STEPS = 36
STRIDE_STEPS = 1
FEATURE_PROTOCOL_ID = "power_only"
POWER_WS_HIST_FEATURE_PROTOCOL_ID = "power_ws_hist"
POWER_ATEMP_HIST_FEATURE_PROTOCOL_ID = "power_atemp_hist"
POWER_ITEMP_HIST_FEATURE_PROTOCOL_ID = "power_itemp_hist"
POWER_WD_HIST_SINCOS_FEATURE_PROTOCOL_ID = "power_wd_hist_sincos"
POWER_WD_YAW_HIST_SINCOS_FEATURE_PROTOCOL_ID = "power_wd_yaw_hist_sincos"
POWER_WD_YAW_PITCHMEAN_HIST_SINCOS_FEATURE_PROTOCOL_ID = "power_wd_yaw_pitchmean_hist_sincos"
POWER_WD_YAW_LRPM_HIST_SINCOS_FEATURE_PROTOCOL_ID = "power_wd_yaw_lrpm_hist_sincos"
POWER_WS_WD_HIST_SINCOS_FEATURE_PROTOCOL_ID = "power_ws_wd_hist_sincos"
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
_RUN_AGCRN_WORK_ROOT = EXPERIMENT_DIR / ".work" / "run_agcrn"
_RUN_STATE_SCHEMA_VERSION = "agcrn.run_agcrn.resume.v1"
_TRAINING_CHECKPOINT_SCHEMA_VERSION = "agcrn.training_checkpoint.v1"
_TASK_WINDOW_COLUMNS = (
    "dataset",
    "output_start_ts",
    "output_end_ts",
    "is_complete_input",
    "is_complete_output",
    "is_fully_synchronous_input",
    "is_fully_synchronous_output",
    "quality_flags",
    "feature_quality_flags",
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
_TRAINING_HISTORY_COLUMNS = [
    "dataset_id",
    "model_id",
    "model_variant",
    "feature_protocol_id",
    "task_id",
    "window_protocol",
    "epoch",
    "train_loss_mean",
    "train_loss_last",
    "val_rmse_pu",
    "best_val_rmse_pu",
    "is_best_epoch",
    "epochs_without_improvement",
    "train_batch_count",
    "train_window_count",
    "val_window_count",
    "device",
    "seed",
    "batch_size",
    "learning_rate",
]
_DATASET_ORDER = {dataset_id: index for index, dataset_id in enumerate(DEFAULT_DATASETS)}
_MODEL_VARIANT_ORDER = {
    MODEL_VARIANT: 0,
    POWER_WS_HIST_MODEL_VARIANT: 1,
    POWER_ATEMP_HIST_MODEL_VARIANT: 2,
    POWER_ITEMP_HIST_MODEL_VARIANT: 3,
    POWER_WD_HIST_SINCOS_MODEL_VARIANT: 4,
    POWER_WD_YAW_HIST_SINCOS_MODEL_VARIANT: 5,
    POWER_WD_YAW_PITCHMEAN_HIST_SINCOS_MODEL_VARIANT: 6,
    POWER_WD_YAW_LRPM_HIST_SINCOS_MODEL_VARIANT: 7,
    POWER_WS_WD_HIST_SINCOS_MODEL_VARIANT: 8,
}
_SPLIT_NAMES = ("train", "val", "test")
_WINDOW_KEY_COLUMNS = ("output_start_ts", "output_end_ts")
_SPLIT_ORDER = {"val": 0, "test": 1}
_EVAL_PROTOCOL_ORDER = {ROLLING_EVAL_PROTOCOL: 0, NON_OVERLAP_EVAL_PROTOCOL: 1}
_METRIC_SCOPE_ORDER = {OVERALL_METRIC_SCOPE: 0, HORIZON_METRIC_SCOPE: 1}


@dataclass(frozen=True)
class ExperimentVariant:
    model_variant: str
    feature_protocol_id: str


@dataclass(frozen=True)
class HyperparameterProfile:
    batch_size: int
    learning_rate: float
    max_epochs: int
    early_stopping_patience: int
    hidden_dim: int
    embed_dim: int
    num_layers: int
    cheb_k: int
    grad_clip_norm: float


VARIANT_SPECS = (
    ExperimentVariant(model_variant=MODEL_VARIANT, feature_protocol_id=FEATURE_PROTOCOL_ID),
    ExperimentVariant(
        model_variant=POWER_WS_HIST_MODEL_VARIANT,
        feature_protocol_id=POWER_WS_HIST_FEATURE_PROTOCOL_ID,
    ),
    ExperimentVariant(
        model_variant=POWER_ATEMP_HIST_MODEL_VARIANT,
        feature_protocol_id=POWER_ATEMP_HIST_FEATURE_PROTOCOL_ID,
    ),
    ExperimentVariant(
        model_variant=POWER_ITEMP_HIST_MODEL_VARIANT,
        feature_protocol_id=POWER_ITEMP_HIST_FEATURE_PROTOCOL_ID,
    ),
    ExperimentVariant(
        model_variant=POWER_WD_HIST_SINCOS_MODEL_VARIANT,
        feature_protocol_id=POWER_WD_HIST_SINCOS_FEATURE_PROTOCOL_ID,
    ),
    ExperimentVariant(
        model_variant=POWER_WD_YAW_HIST_SINCOS_MODEL_VARIANT,
        feature_protocol_id=POWER_WD_YAW_HIST_SINCOS_FEATURE_PROTOCOL_ID,
    ),
    ExperimentVariant(
        model_variant=POWER_WD_YAW_PITCHMEAN_HIST_SINCOS_MODEL_VARIANT,
        feature_protocol_id=POWER_WD_YAW_PITCHMEAN_HIST_SINCOS_FEATURE_PROTOCOL_ID,
    ),
    ExperimentVariant(
        model_variant=POWER_WD_YAW_LRPM_HIST_SINCOS_MODEL_VARIANT,
        feature_protocol_id=POWER_WD_YAW_LRPM_HIST_SINCOS_FEATURE_PROTOCOL_ID,
    ),
    ExperimentVariant(
        model_variant=POWER_WS_WD_HIST_SINCOS_MODEL_VARIANT,
        feature_protocol_id=POWER_WS_WD_HIST_SINCOS_FEATURE_PROTOCOL_ID,
    ),
)
DEFAULT_VARIANTS = tuple(spec.model_variant for spec in VARIANT_SPECS)
SEARCH_VARIANTS = DEFAULT_VARIANTS
_VARIANT_SPECS_BY_NAME = {
    spec.model_variant: spec
    for spec in VARIANT_SPECS
}
_BASELINE_BS512_LR1E3_HYPERPARAMETERS = HyperparameterProfile(
    batch_size=512,
    learning_rate=1e-3,
    max_epochs=20,
    early_stopping_patience=5,
    hidden_dim=64,
    embed_dim=10,
    num_layers=2,
    cheb_k=2,
    grad_clip_norm=DEFAULT_GRAD_CLIP_NORM,
)
_BASELINE_BS512_LR5E4_HYPERPARAMETERS = HyperparameterProfile(
    batch_size=512,
    learning_rate=5e-4,
    max_epochs=20,
    early_stopping_patience=5,
    hidden_dim=64,
    embed_dim=10,
    num_layers=2,
    cheb_k=2,
    grad_clip_norm=DEFAULT_GRAD_CLIP_NORM,
)
_GRAPH_BS512_LR5E4_HYPERPARAMETERS = HyperparameterProfile(
    batch_size=512,
    learning_rate=5e-4,
    max_epochs=20,
    early_stopping_patience=5,
    hidden_dim=64,
    embed_dim=16,
    num_layers=2,
    cheb_k=3,
    grad_clip_norm=DEFAULT_GRAD_CLIP_NORM,
)
TUNED_DEFAULT_HYPERPARAMETERS_BY_VARIANT = {
    MODEL_VARIANT: _BASELINE_BS512_LR1E3_HYPERPARAMETERS,
    POWER_WS_HIST_MODEL_VARIANT: _GRAPH_BS512_LR5E4_HYPERPARAMETERS,
    POWER_ATEMP_HIST_MODEL_VARIANT: _BASELINE_BS512_LR1E3_HYPERPARAMETERS,
    POWER_ITEMP_HIST_MODEL_VARIANT: _BASELINE_BS512_LR1E3_HYPERPARAMETERS,
    POWER_WD_HIST_SINCOS_MODEL_VARIANT: _BASELINE_BS512_LR5E4_HYPERPARAMETERS,
    POWER_WD_YAW_HIST_SINCOS_MODEL_VARIANT: _BASELINE_BS512_LR5E4_HYPERPARAMETERS,
    POWER_WD_YAW_PITCHMEAN_HIST_SINCOS_MODEL_VARIANT: _BASELINE_BS512_LR1E3_HYPERPARAMETERS,
    POWER_WD_YAW_LRPM_HIST_SINCOS_MODEL_VARIANT: _BASELINE_BS512_LR1E3_HYPERPARAMETERS,
    POWER_WS_WD_HIST_SINCOS_MODEL_VARIANT: _BASELINE_BS512_LR1E3_HYPERPARAMETERS,
}


def resolve_hyperparameter_profile(
    variant_name: str,
    *,
    batch_size: int | None = None,
    learning_rate: float | None = None,
    max_epochs: int | None = None,
    early_stopping_patience: int | None = None,
    hidden_dim: int | None = None,
    embed_dim: int | None = None,
    num_layers: int | None = None,
    cheb_k: int | None = None,
    grad_clip_norm: float | None = None,
) -> HyperparameterProfile:
    try:
        defaults = TUNED_DEFAULT_HYPERPARAMETERS_BY_VARIANT[variant_name]
    except KeyError as exc:
        raise ValueError(f"Unsupported AGCRN variant {variant_name!r}.") from exc
    return HyperparameterProfile(
        batch_size=defaults.batch_size if batch_size is None else batch_size,
        learning_rate=defaults.learning_rate if learning_rate is None else learning_rate,
        max_epochs=defaults.max_epochs if max_epochs is None else max_epochs,
        early_stopping_patience=(
            defaults.early_stopping_patience
            if early_stopping_patience is None
            else early_stopping_patience
        ),
        hidden_dim=defaults.hidden_dim if hidden_dim is None else hidden_dim,
        embed_dim=defaults.embed_dim if embed_dim is None else embed_dim,
        num_layers=defaults.num_layers if num_layers is None else num_layers,
        cheb_k=defaults.cheb_k if cheb_k is None else cheb_k,
        grad_clip_norm=defaults.grad_clip_norm if grad_clip_norm is None else grad_clip_norm,
    )


@dataclass(frozen=True)
class DatasetMetadata:
    dataset_id: str
    turbine_ids: tuple[str, ...]
    rated_power_kw: float
    turbine_static: pl.DataFrame


@dataclass(frozen=True)
class VariantDatasetContext:
    dataset_id: str
    model_variant: str
    feature_protocol_id: str
    metadata: DatasetMetadata
    series: pl.DataFrame
    target_history_mask_columns: tuple[str, ...]
    past_covariate_columns: tuple[str, ...]
    strict_window_index: pl.DataFrame
    raw_timestamps: tuple[datetime, ...]
    resolution_minutes: int
    timestamps_us: np.ndarray
    target_pu: np.ndarray
    coordinate_mode: str
    distance_sanity: pl.DataFrame


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
    model_variant: str
    feature_protocol_id: str
    resolution_minutes: int
    rated_power_kw: float
    history_steps: int
    forecast_steps: int
    stride_steps: int
    turbine_ids: tuple[str, ...]
    coordinate_mode: str
    node_count: int
    timestamps_us: np.ndarray
    input_channel_names: tuple[str, ...]
    source_tensor: np.ndarray
    target_pu: np.ndarray
    train_windows: FarmWindowDescriptorIndex
    val_rolling_windows: FarmWindowDescriptorIndex
    val_non_overlap_windows: FarmWindowDescriptorIndex
    test_rolling_windows: FarmWindowDescriptorIndex
    test_non_overlap_windows: FarmWindowDescriptorIndex

    @property
    def input_channels(self) -> int:
        return int(self.source_tensor.shape[2])


@dataclass(frozen=True)
class TrainingOutcome:
    best_epoch: int
    epochs_ran: int
    best_val_rmse_pu: float
    device: str
    model: Any


@dataclass(frozen=True)
class ResumePaths:
    slot_dir: Path
    state_path: Path
    partial_results_path: Path
    training_history_path: Path
    checkpoints_dir: Path


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


def _empty_result_frame() -> pl.DataFrame:
    return pl.DataFrame(schema=_RESULT_COLUMNS)


def _empty_training_history_frame() -> pl.DataFrame:
    return pl.DataFrame(schema=_TRAINING_HISTORY_COLUMNS)


def _temporary_path(path: Path) -> Path:
    return path.with_name(f".{path.name}.{time.time_ns()}.tmp")


def _atomic_write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = _temporary_path(path)
    temporary_path.write_text(content, encoding="utf-8")
    temporary_path.replace(path)


def _atomic_write_json(path: Path, payload: object) -> None:
    _atomic_write_text(path, json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n")


def _atomic_write_csv(path: Path, frame: pl.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = _temporary_path(path)
    frame.write_csv(temporary_path)
    temporary_path.replace(path)


def _normalize_output_path(output_path: str | Path) -> Path:
    return Path(output_path).expanduser().resolve()


def _resume_paths_for_output(*, output_path: str | Path, work_root: str | Path) -> ResumePaths:
    normalized_output_path = _normalize_output_path(output_path)
    slot_name = hashlib.sha256(str(normalized_output_path).encode("utf-8")).hexdigest()
    slot_dir = Path(work_root) / slot_name
    return ResumePaths(
        slot_dir=slot_dir,
        state_path=slot_dir / "run_state.json",
        partial_results_path=slot_dir / "partial_results.csv",
        training_history_path=slot_dir / "training_history.csv",
        checkpoints_dir=slot_dir / "checkpoints",
    )


def training_history_output_path(output_path: str | Path) -> Path:
    output = _normalize_output_path(output_path)
    suffix = output.suffix or ".csv"
    stem = output.stem if output.suffix else output.name
    return output.with_name(f"{stem}.training_history{suffix}")


def _job_key(dataset_id: str, model_variant: str) -> tuple[str, str]:
    return dataset_id, model_variant


def _job_identity(
    *,
    dataset_id: str,
    model_variant: str,
    feature_protocol_id: str,
) -> dict[str, str]:
    return {
        "dataset_id": dataset_id,
        "model_variant": model_variant,
        "feature_protocol_id": feature_protocol_id,
    }


def _job_identity_for_prepared_dataset(prepared_dataset: PreparedDataset) -> dict[str, str]:
    return _job_identity(
        dataset_id=prepared_dataset.dataset_id,
        model_variant=prepared_dataset.model_variant,
        feature_protocol_id=prepared_dataset.feature_protocol_id,
    )


def _job_checkpoint_path(paths: ResumePaths, *, dataset_id: str, model_variant: str) -> Path:
    return paths.checkpoints_dir / f"{dataset_id}__{model_variant}.pt"


def _build_effective_config(
    *,
    dataset_ids: Sequence[str],
    variant_specs: Sequence[ExperimentVariant],
    seed: int,
    max_train_origins: int | None,
    max_eval_origins: int | None,
    batch_size: int | None,
    learning_rate: float | None,
    max_epochs: int | None,
    early_stopping_patience: int | None,
    hidden_dim: int | None,
    embed_dim: int | None,
    num_layers: int | None,
    cheb_k: int | None,
    grad_clip_norm: float | None,
) -> dict[str, object]:
    return {
        "dataset_ids": list(dataset_ids),
        "variant_names": [spec.model_variant for spec in variant_specs],
        "seed": seed,
        "max_train_origins": max_train_origins,
        "max_eval_origins": max_eval_origins,
        "resolved_variant_hyperparameters": {
            spec.model_variant: {
                "feature_protocol_id": spec.feature_protocol_id,
                **asdict(
                    resolve_hyperparameter_profile(
                        spec.model_variant,
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
                ),
            }
            for spec in variant_specs
        },
    }


def _load_resume_state(paths: ResumePaths) -> dict[str, object]:
    payload = json.loads(paths.state_path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != _RUN_STATE_SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported AGCRN resume state schema at {paths.state_path}: {payload.get('schema_version')!r}."
        )
    status = payload.get("status")
    if status not in {"running", "complete"}:
        raise ValueError(f"Unsupported AGCRN resume state status at {paths.state_path}: {status!r}.")
    return payload


def _load_resume_state_if_exists(paths: ResumePaths) -> dict[str, object] | None:
    if not paths.state_path.exists():
        return None
    return _load_resume_state(paths)


def _write_resume_state(
    paths: ResumePaths,
    *,
    status: str,
    effective_config: dict[str, object],
    active_job: dict[str, str] | None,
) -> None:
    _atomic_write_json(
        paths.state_path,
        {
            "schema_version": _RUN_STATE_SCHEMA_VERSION,
            "status": status,
            "effective_config": effective_config,
            "active_job": active_job,
        },
    )


def _read_partial_results(paths: ResumePaths) -> pl.DataFrame:
    if not paths.partial_results_path.exists():
        return _empty_result_frame()
    frame = pl.read_csv(paths.partial_results_path, infer_schema_length=None)
    if not frame.columns:
        return _empty_result_frame()
    missing_columns = [column for column in _RESULT_COLUMNS if column not in frame.columns]
    if missing_columns:
        raise ValueError(
            f"Partial results at {paths.partial_results_path} are missing expected columns: {missing_columns!r}."
        )
    frame = frame.select(_RESULT_COLUMNS)
    if frame.is_empty():
        return frame
    return sort_result_frame(frame)


def _write_partial_results(paths: ResumePaths, frame: pl.DataFrame) -> None:
    _atomic_write_csv(paths.partial_results_path, frame)


def _read_training_history(path: str | Path) -> pl.DataFrame:
    history_path = Path(path)
    if not history_path.exists():
        return _empty_training_history_frame()
    frame = pl.read_csv(history_path, infer_schema_length=None)
    if not frame.columns:
        return _empty_training_history_frame()
    missing_columns = [column for column in _TRAINING_HISTORY_COLUMNS if column not in frame.columns]
    if missing_columns:
        raise ValueError(
            f"Training history at {history_path} is missing expected columns: {missing_columns!r}."
        )
    frame = frame.select(_TRAINING_HISTORY_COLUMNS)
    if frame.is_empty():
        return frame
    return sort_training_history_frame(frame)


def _write_training_history(path: str | Path, frame: pl.DataFrame) -> None:
    _atomic_write_csv(Path(path), sort_training_history_frame(frame))


def _training_history_job_expr(job_identity: dict[str, str]) -> pl.Expr:
    return (
        (pl.col("dataset_id") == job_identity["dataset_id"])
        & (pl.col("model_variant") == job_identity["model_variant"])
        & (pl.col("feature_protocol_id") == job_identity["feature_protocol_id"])
    )


def _prune_training_history_for_job(
    path: str | Path,
    *,
    job_identity: dict[str, str],
    min_epoch: int,
) -> None:
    history_path = Path(path)
    if not history_path.exists():
        return
    frame = _read_training_history(history_path)
    if frame.is_empty():
        return
    retained = frame.filter(~(_training_history_job_expr(job_identity) & (pl.col("epoch") >= min_epoch)))
    _write_training_history(history_path, retained)


def _append_training_history_row(path: str | Path, row: dict[str, object]) -> None:
    history_path = Path(path)
    frame = _read_training_history(history_path)
    row_frame = pl.DataFrame([row], infer_schema_length=None).select(_TRAINING_HISTORY_COLUMNS)
    job_identity = {
        "dataset_id": str(row["dataset_id"]),
        "model_variant": str(row["model_variant"]),
        "feature_protocol_id": str(row["feature_protocol_id"]),
    }
    if frame.is_empty():
        combined = row_frame
    else:
        retained = frame.filter(
            ~(_training_history_job_expr(job_identity) & (pl.col("epoch") == int(row["epoch"])))
        )
        combined = pl.concat([retained, row_frame], how="diagonal_relaxed")
    _write_training_history(history_path, combined)


def _publish_training_history(paths: ResumePaths, output_path: str | Path) -> Path:
    published_path = training_history_output_path(output_path)
    history = _read_training_history(paths.training_history_path)
    _write_training_history(published_path, history)
    return published_path


def _reset_resume_slot(paths: ResumePaths, *, effective_config: dict[str, object]) -> None:
    if paths.slot_dir.exists():
        shutil.rmtree(paths.slot_dir)
    paths.checkpoints_dir.mkdir(parents=True, exist_ok=True)
    _write_partial_results(paths, _empty_result_frame())
    _write_training_history(paths.training_history_path, _empty_training_history_frame())
    _write_resume_state(paths, status="running", effective_config=effective_config, active_job=None)


def _delete_job_checkpoint(paths: ResumePaths, *, dataset_id: str, model_variant: str) -> None:
    checkpoint_path = _job_checkpoint_path(paths, dataset_id=dataset_id, model_variant=model_variant)
    checkpoint_path.unlink(missing_ok=True)


def _clear_checkpoint_dir(paths: ResumePaths) -> None:
    if paths.checkpoints_dir.exists():
        shutil.rmtree(paths.checkpoints_dir)
    paths.checkpoints_dir.mkdir(parents=True, exist_ok=True)


def _completed_job_keys(frame: pl.DataFrame) -> set[tuple[str, str]]:
    if frame.is_empty():
        return set()
    return {
        _job_key(dataset_id, model_variant)
        for dataset_id, model_variant in frame.select(["dataset_id", "model_variant"]).unique().iter_rows()
    }


def _result_frame_from_rows(rows: Sequence[dict[str, object]]) -> pl.DataFrame:
    if not rows:
        return _empty_result_frame()
    return sort_result_frame(pl.DataFrame(rows, infer_schema_length=None).select(_RESULT_COLUMNS))


def _torch_load_checkpoint(path: Path, *, map_location: str):
    resolved_torch, _, _, _, _ = require_torch()
    try:
        return resolved_torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return resolved_torch.load(path, map_location=map_location)


def _save_training_checkpoint(path: Path, payload: dict[str, object]) -> None:
    resolved_torch, _, _, _, _ = require_torch()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = _temporary_path(path)
    resolved_torch.save(payload, temporary_path)
    temporary_path.replace(path)


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


def resolve_variant_specs(variant_names: Sequence[str] | None = None) -> tuple[ExperimentVariant, ...]:
    requested = tuple(variant_names or DEFAULT_VARIANTS)
    resolved: list[ExperimentVariant] = []
    seen: set[str] = set()
    for variant_name in requested:
        try:
            spec = _VARIANT_SPECS_BY_NAME[variant_name]
        except KeyError as exc:
            supported = ", ".join(DEFAULT_VARIANTS)
            raise ValueError(f"Unknown model variant {variant_name!r}. Expected one of: {supported}.") from exc
        if spec.model_variant in seen:
            continue
        resolved.append(spec)
        seen.add(spec.model_variant)
    return tuple(resolved)


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


def _has_complete_static_columns(frame: pl.DataFrame, columns: Sequence[str]) -> bool:
    return set(columns).issubset(frame.columns) and all(frame[column].null_count() == 0 for column in columns)


def _load_task_bundle(
    dataset_id: str,
    *,
    feature_protocol_id: str = FEATURE_PROTOCOL_ID,
    cache_root: str | Path = _CACHE_ROOT,
) -> Any:
    _ensure_repo_src_on_path()
    try:
        from wind_datasets import load_task_bundle
    except ImportError as exc:  # pragma: no cover - defensive
        raise RuntimeError("Unable to import wind_datasets for task bundle loading.") from exc

    try:
        return load_task_bundle(
            dataset_id,
            build_task_spec(),
            cache_root=cache_root,
            feature_protocol_id=feature_protocol_id,
        )
    except Exception as exc:  # pragma: no cover - exercised only when cache is unavailable
        raise RuntimeError(
            f"Unable to load the farm task bundle for dataset {dataset_id!r} with "
            f"feature_protocol_id={feature_protocol_id!r}. "
            "Either prebuild the cache artifacts or configure wind_datasets.local.toml."
        ) from exc


def load_dataset_metadata(dataset_id: str, bundle: Any) -> DatasetMetadata:
    task_context = bundle.task_context
    task = task_context.get("task", {})
    if (
        int(task.get("history_steps", HISTORY_STEPS)) != HISTORY_STEPS
        or int(task.get("forecast_steps", FORECAST_STEPS)) != FORECAST_STEPS
        or int(task.get("stride_steps", STRIDE_STEPS)) != STRIDE_STEPS
    ):
        raise ValueError(
            f"Task bundle context for dataset {dataset_id!r} does not match the expected "
            f"{HISTORY_STEPS}/{FORECAST_STEPS}/{STRIDE_STEPS} task."
        )

    turbine_static = bundle.static
    if "turbine_index" not in turbine_static.columns:
        raise ValueError(f"Task bundle static sidecar for dataset {dataset_id!r} is missing turbine_index.")
    ordered_static = turbine_static.sort("turbine_index")
    ordered_ids = tuple(ordered_static["turbine_id"].to_list())
    context_ids = tuple(task_context["turbine_ids"])
    if ordered_ids != context_ids:
        raise ValueError(
            f"Task bundle static sidecar order for dataset {dataset_id!r} does not match task_context turbine_ids."
        )

    if not _has_complete_static_columns(ordered_static, ("rated_power_kw",)):
        raise ValueError(f"Task bundle static sidecar for dataset {dataset_id!r} is missing non-null rated_power_kw.")
    try:
        resolve_static_coordinate_columns(ordered_static)
    except ValueError as exc:
        raise ValueError(
            f"Task bundle static sidecar for dataset {dataset_id!r} must include either full "
            "coord_x/coord_y or full latitude/longitude."
        ) from exc

    rated_powers = sorted({float(value) for value in ordered_static["rated_power_kw"].drop_nulls().to_list()})
    if len(rated_powers) != 1:
        raise ValueError(f"Dataset {dataset_id!r} must have a single rated_power_kw, found {rated_powers!r}.")

    return DatasetMetadata(
        dataset_id=dataset_id,
        turbine_ids=context_ids,
        rated_power_kw=rated_powers[0],
        turbine_static=ordered_static,
    )


def resolve_past_covariate_columns(bundle: Any) -> tuple[str, ...]:
    column_groups = bundle.task_context.get("column_groups", {})
    if not isinstance(column_groups, dict):
        raise ValueError("Task bundle task_context is missing column_groups.")
    raw_columns = column_groups.get("past_covariates") or ()
    return tuple(str(column) for column in raw_columns)


def resolve_target_history_mask_columns(bundle: Any) -> tuple[str, ...]:
    column_groups = bundle.task_context.get("column_groups", {})
    if not isinstance(column_groups, dict):
        raise ValueError("Task bundle task_context is missing column_groups.")
    raw_columns = column_groups.get("target_history_masks") or ()
    return tuple(str(column) for column in raw_columns)


def load_series_frame(
    dataset_id: str,
    bundle: Any,
    *,
    target_history_mask_columns: Sequence[str] = (),
    past_covariate_columns: Sequence[str] = (),
) -> pl.DataFrame:
    available_columns = set(bundle.series.columns)
    required_columns = (*_SERIES_BASE_COLUMNS, *target_history_mask_columns, *past_covariate_columns)
    missing_columns = [column for column in required_columns if column not in available_columns]
    if missing_columns:
        raise ValueError(
            f"Task bundle series for dataset {dataset_id!r} is missing required columns {missing_columns!r}."
        )
    load_started = time.monotonic()
    frame = bundle.series.select(list(required_columns)).sort(["turbine_id", "timestamp"])
    _profile_log(
        dataset_id,
        "load_series",
        rows=frame.height,
        columns=len(frame.columns),
        duration_seconds=round(time.monotonic() - load_started, 6),
    )
    return frame


def load_strict_window_index(
    dataset_id: str,
    bundle: Any,
    *,
    require_clean_input_features: bool = False,
) -> pl.DataFrame:
    available_columns = set(bundle.window_index.columns)
    missing_columns = [column for column in _TASK_WINDOW_COLUMNS if column not in available_columns]
    if missing_columns:
        raise ValueError(
            f"Task bundle window_index for dataset {dataset_id!r} is missing required columns {missing_columns!r}."
        )
    load_started = time.monotonic()
    feature_quality_expr = pl.col("feature_quality_flags").fill_null("")
    frame = (
        bundle.window_index
        .select(list(_TASK_WINDOW_COLUMNS))
        .filter(
            pl.col("is_complete_input")
            & pl.col("is_complete_output")
            & pl.col("is_fully_synchronous_input")
            & pl.col("is_fully_synchronous_output")
            & (pl.col("quality_flags").fill_null("") == "")
            & (
                ~feature_quality_expr.str.contains(r"(^|\|)feature_quality_issues_input($|\|)")
                if require_clean_input_features
                else pl.lit(True)
            )
        )
        .sort("output_start_ts")
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


def _load_variant_dataset_context(
    dataset_id: str,
    *,
    variant_spec: ExperimentVariant,
    cache_root: str | Path = _CACHE_ROOT,
) -> VariantDatasetContext:
    bundle = _load_task_bundle(
        dataset_id,
        feature_protocol_id=variant_spec.feature_protocol_id,
        cache_root=cache_root,
    )
    if bundle.task_context.get("feature_protocol_id") != variant_spec.feature_protocol_id:
        raise ValueError(
            f"Task bundle for dataset {dataset_id!r} does not match requested "
            f"feature_protocol_id={variant_spec.feature_protocol_id!r}."
        )
    metadata = load_dataset_metadata(dataset_id, bundle)
    target_history_mask_columns = resolve_target_history_mask_columns(bundle)
    past_covariate_columns = resolve_past_covariate_columns(bundle)
    strict_window_index = load_strict_window_index(
        dataset_id,
        bundle,
        require_clean_input_features=bool(past_covariate_columns),
    )
    series = load_series_frame(
        dataset_id,
        bundle,
        target_history_mask_columns=target_history_mask_columns,
        past_covariate_columns=past_covariate_columns,
    )
    coordinate_mode, distance_sanity = build_distance_sanity_frame(
        metadata.turbine_static,
        ordered_turbine_ids=metadata.turbine_ids,
    )
    panel_frame = prepare_panel_frame(series, rated_power_kw=metadata.rated_power_kw)
    resolution_minutes = resolve_resolution_minutes(panel_frame)
    raw_timestamps = tuple(panel_frame["timestamp"].to_list())
    timestamps_us, target_pu = build_panel_series(panel_frame, turbine_ids=metadata.turbine_ids)
    return VariantDatasetContext(
        dataset_id=dataset_id,
        model_variant=variant_spec.model_variant,
        feature_protocol_id=variant_spec.feature_protocol_id,
        metadata=metadata,
        series=series,
        target_history_mask_columns=target_history_mask_columns,
        past_covariate_columns=past_covariate_columns,
        strict_window_index=strict_window_index,
        raw_timestamps=raw_timestamps,
        resolution_minutes=resolution_minutes,
        timestamps_us=timestamps_us,
        target_pu=target_pu,
        coordinate_mode=coordinate_mode,
        distance_sanity=distance_sanity,
    )


def _build_feature_panel(
    series: pl.DataFrame,
    *,
    feature_column: str,
    turbine_ids: Sequence[str],
    expected_timestamps: Sequence[datetime],
) -> np.ndarray:
    panel = (
        series.select(["timestamp", "turbine_id", feature_column])
        .pivot(on="turbine_id", index="timestamp", values=feature_column, aggregate_function="first")
        .sort("timestamp")
    )
    if panel["timestamp"].to_list() != list(expected_timestamps):
        raise ValueError(f"Feature panel {feature_column!r} does not align to the task bundle timestamp axis.")
    missing_columns = [turbine_id for turbine_id in turbine_ids if turbine_id not in panel.columns]
    if missing_columns:
        raise ValueError(f"Feature panel {feature_column!r} is missing turbine columns {missing_columns!r}.")
    return panel.select(list(turbine_ids)).to_numpy().astype(np.float32, copy=False)


def _history_row_mask(
    *,
    target_indices: np.ndarray,
    history_steps: int,
    total_steps: int,
) -> np.ndarray:
    mask = np.zeros((total_steps,), dtype=bool)
    for target_index in np.unique(np.asarray(target_indices, dtype=np.int64)):
        start_index = int(target_index) - history_steps
        end_index = int(target_index)
        if start_index < 0 or end_index > total_steps:
            raise ValueError(f"History window [{start_index}, {end_index}) falls outside the available panel axis.")
        mask[start_index:end_index] = True
    return mask


def _compute_train_covariate_stats(
    covariate_tensor: np.ndarray,
    *,
    train_windows: FarmWindowDescriptorIndex,
    history_steps: int,
) -> tuple[np.ndarray, np.ndarray]:
    if covariate_tensor.ndim != 3:
        raise ValueError(f"Expected covariate tensor with shape [time, node, channel], got {covariate_tensor.shape!r}.")
    if covariate_tensor.shape[2] == 0:
        return (
            np.zeros((0,), dtype=np.float32),
            np.ones((0,), dtype=np.float32),
        )
    if len(train_windows) == 0:
        raise ValueError("At least one train window is required to compute covariate normalization statistics.")
    history_mask = _history_row_mask(
        target_indices=train_windows.target_indices,
        history_steps=history_steps,
        total_steps=covariate_tensor.shape[0],
    )
    train_history = covariate_tensor[history_mask]
    means = np.zeros((covariate_tensor.shape[2],), dtype=np.float32)
    stds = np.ones((covariate_tensor.shape[2],), dtype=np.float32)
    for channel_index in range(covariate_tensor.shape[2]):
        values = train_history[:, :, channel_index].reshape(-1)
        finite_values = values[np.isfinite(values)]
        if finite_values.size == 0:
            raise ValueError(f"Covariate channel {channel_index} has no finite training values for normalization.")
        mean = float(finite_values.mean())
        std = float(finite_values.std())
        if np.isfinite(std) and std > 0:
            means[channel_index] = mean
            stds[channel_index] = std
    return means, stds


def build_source_tensor(
    series: pl.DataFrame,
    *,
    turbine_ids: Sequence[str],
    raw_timestamps: Sequence[datetime],
    target_pu: np.ndarray,
    target_history_mask_columns: Sequence[str],
    past_covariate_columns: Sequence[str],
    train_windows: FarmWindowDescriptorIndex,
    history_steps: int,
) -> tuple[np.ndarray, tuple[str, ...]]:
    source_channels = [target_pu.astype(np.float32, copy=False)]
    input_channel_names: list[str] = ["target_pu"]
    auxiliary_columns = (*target_history_mask_columns, *past_covariate_columns)
    if auxiliary_columns:
        covariate_channels = np.stack(
            [
                _build_feature_panel(
                    series,
                    feature_column=feature_column,
                    turbine_ids=turbine_ids,
                    expected_timestamps=raw_timestamps,
                )
                for feature_column in auxiliary_columns
            ],
            axis=-1,
        )
        means, stds = _compute_train_covariate_stats(
            covariate_channels,
            train_windows=train_windows,
            history_steps=history_steps,
        )
        normalized_covariates = ((covariate_channels - means.reshape(1, 1, -1)) / stds.reshape(1, 1, -1)).astype(
            np.float32,
            copy=False,
        )
        source_channels.extend(normalized_covariates[:, :, index] for index in range(normalized_covariates.shape[2]))
        input_channel_names.extend(auxiliary_columns)
    source_tensor = np.stack(source_channels, axis=-1).astype(np.float32, copy=False)
    return source_tensor, tuple(input_channel_names)


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


def _split_context_window_index(context: VariantDatasetContext) -> dict[str, pl.DataFrame]:
    return split_farm_window_index(
        context.strict_window_index,
        raw_timestamps=context.raw_timestamps,
        resolution_minutes=context.resolution_minutes,
    )


def _assert_alignment_compatible(contexts: Sequence[VariantDatasetContext]) -> None:
    if not contexts:
        raise ValueError("At least one variant dataset context is required for alignment.")
    reference = contexts[0]
    feature_protocol_ids = [context.feature_protocol_id for context in contexts]
    for context in contexts[1:]:
        if context.dataset_id != reference.dataset_id:
            raise ValueError(
                f"Variant contexts must share a dataset_id, found {reference.dataset_id!r} and {context.dataset_id!r}."
            )
        if context.metadata.turbine_ids != reference.metadata.turbine_ids:
            raise ValueError(
                f"Dataset {reference.dataset_id!r} variants {feature_protocol_ids!r} do not share turbine_ids."
            )
        if context.metadata.rated_power_kw != reference.metadata.rated_power_kw:
            raise ValueError(
                f"Dataset {reference.dataset_id!r} variants {feature_protocol_ids!r} do not share rated_power_kw."
            )
        if context.raw_timestamps != reference.raw_timestamps:
            raise ValueError(
                f"Dataset {reference.dataset_id!r} variants {feature_protocol_ids!r} do not share raw_timestamps."
            )
        if context.resolution_minutes != reference.resolution_minutes:
            raise ValueError(
                f"Dataset {reference.dataset_id!r} variants {feature_protocol_ids!r} do not share resolution_minutes."
            )
        if not np.array_equal(context.timestamps_us, reference.timestamps_us):
            raise ValueError(
                f"Dataset {reference.dataset_id!r} variants {feature_protocol_ids!r} do not share panel timestamp axis."
            )


def _align_split_frames(
    contexts: Sequence[VariantDatasetContext],
) -> dict[str, dict[str, pl.DataFrame]]:
    _assert_alignment_compatible(contexts)
    split_frames_by_variant = {
        context.model_variant: _split_context_window_index(context)
        for context in contexts
    }
    if len(contexts) == 1:
        return split_frames_by_variant

    feature_protocol_ids = [context.feature_protocol_id for context in contexts]
    aligned_frames_by_variant = {
        context.model_variant: {}
        for context in contexts
    }
    for split_name in _SPLIT_NAMES:
        shared_keys: pl.DataFrame | None = None
        for context in contexts:
            frame_keys = (
                split_frames_by_variant[context.model_variant][split_name]
                .select(list(_WINDOW_KEY_COLUMNS))
                .unique()
                .sort(list(_WINDOW_KEY_COLUMNS))
            )
            shared_keys = (
                frame_keys
                if shared_keys is None
                else shared_keys.join(frame_keys, on=list(_WINDOW_KEY_COLUMNS), how="inner")
            )
        if shared_keys is None or shared_keys.is_empty():
            raise ValueError(
                f"Dataset {contexts[0].dataset_id!r} has no shared strict windows for split {split_name!r} "
                f"across feature_protocol_ids={feature_protocol_ids!r}."
            )
        for context in contexts:
            aligned_frames_by_variant[context.model_variant][split_name] = (
                split_frames_by_variant[context.model_variant][split_name]
                .join(shared_keys, on=list(_WINDOW_KEY_COLUMNS), how="inner")
                .sort("output_start_ts")
            )
        _profile_log(
            contexts[0].dataset_id,
            "align_split_frames",
            split_name=split_name,
            feature_protocol_ids=feature_protocol_ids,
            shared_windows=shared_keys.height,
        )
    return aligned_frames_by_variant


def _limit_split_frames(
    split_frames: dict[str, pl.DataFrame],
    *,
    max_train_origins: int | None = None,
    max_eval_origins: int | None = None,
) -> dict[str, pl.DataFrame]:
    limited_frames = {
        split_name: frame
        for split_name, frame in split_frames.items()
    }
    if max_train_origins is not None:
        limited_frames["train"] = limited_frames["train"].head(max_train_origins)
    if max_eval_origins is not None:
        limited_frames["val"] = limited_frames["val"].head(max_eval_origins)
        limited_frames["test"] = limited_frames["test"].head(max_eval_origins)
    return limited_frames


def _finalize_prepared_dataset(
    context: VariantDatasetContext,
    *,
    split_frames: dict[str, pl.DataFrame],
    max_train_origins: int | None = None,
    max_eval_origins: int | None = None,
) -> PreparedDataset:
    limited_split_frames = _limit_split_frames(
        split_frames,
        max_train_origins=max_train_origins,
        max_eval_origins=max_eval_origins,
    )
    train_windows = build_window_descriptor_index(limited_split_frames["train"], timestamps_us=context.timestamps_us)
    val_rolling_windows = build_window_descriptor_index(
        limited_split_frames["val"],
        timestamps_us=context.timestamps_us,
    )
    test_rolling_windows = build_window_descriptor_index(
        limited_split_frames["test"],
        timestamps_us=context.timestamps_us,
    )
    val_non_overlap_windows = thin_non_overlap_window_index(val_rolling_windows, forecast_steps=FORECAST_STEPS)
    test_non_overlap_windows = thin_non_overlap_window_index(test_rolling_windows, forecast_steps=FORECAST_STEPS)
    source_tensor, input_channel_names = build_source_tensor(
        context.series,
        turbine_ids=context.metadata.turbine_ids,
        raw_timestamps=context.raw_timestamps,
        target_pu=context.target_pu,
        target_history_mask_columns=context.target_history_mask_columns,
        past_covariate_columns=context.past_covariate_columns,
        train_windows=train_windows,
        history_steps=HISTORY_STEPS,
    )
    _profile_log(
        context.dataset_id,
        "prepare_dataset_complete",
        coordinate_mode=context.coordinate_mode,
        feature_protocol_id=context.feature_protocol_id,
        input_channels=source_tensor.shape[2],
        model_variant=context.model_variant,
        node_count=len(context.metadata.turbine_ids),
        train_windows=len(train_windows),
        val_rolling_windows=len(val_rolling_windows),
        val_non_overlap_windows=len(val_non_overlap_windows),
        test_rolling_windows=len(test_rolling_windows),
        test_non_overlap_windows=len(test_non_overlap_windows),
        nearest_neighbors=context.distance_sanity.to_dicts(),
        rated_power_kw=context.metadata.rated_power_kw,
        resolution_minutes=context.resolution_minutes,
    )
    return PreparedDataset(
        dataset_id=context.dataset_id,
        model_variant=context.model_variant,
        feature_protocol_id=context.feature_protocol_id,
        resolution_minutes=context.resolution_minutes,
        rated_power_kw=context.metadata.rated_power_kw,
        history_steps=HISTORY_STEPS,
        forecast_steps=FORECAST_STEPS,
        stride_steps=STRIDE_STEPS,
        turbine_ids=context.metadata.turbine_ids,
        coordinate_mode=context.coordinate_mode,
        node_count=len(context.metadata.turbine_ids),
        timestamps_us=context.timestamps_us,
        input_channel_names=input_channel_names,
        source_tensor=source_tensor,
        target_pu=context.target_pu,
        train_windows=train_windows,
        val_rolling_windows=val_rolling_windows,
        val_non_overlap_windows=val_non_overlap_windows,
        test_rolling_windows=test_rolling_windows,
        test_non_overlap_windows=test_non_overlap_windows,
    )


def _prepare_datasets_for_variants(
    dataset_id: str,
    *,
    variant_specs: Sequence[ExperimentVariant],
    cache_root: str | Path = _CACHE_ROOT,
    max_train_origins: int | None = None,
    max_eval_origins: int | None = None,
) -> tuple[PreparedDataset, ...]:
    contexts = tuple(
        _load_variant_dataset_context(
            dataset_id,
            variant_spec=variant_spec,
            cache_root=cache_root,
        )
        for variant_spec in variant_specs
    )
    if not contexts:
        raise ValueError("At least one variant_spec is required to prepare datasets.")
    split_frames_by_variant = (
        _align_split_frames(contexts)
        if len(contexts) > 1
        else {contexts[0].model_variant: _split_context_window_index(contexts[0])}
    )
    return tuple(
        _finalize_prepared_dataset(
            context,
            split_frames=split_frames_by_variant[context.model_variant],
            max_train_origins=max_train_origins,
            max_eval_origins=max_eval_origins,
        )
        for context in contexts
    )


if Dataset is not None:

    class PanelWindowDataset(Dataset):
        def __init__(
            self,
            source_tensor: np.ndarray,
            target_pu: np.ndarray,
            windows: FarmWindowDescriptorIndex,
            *,
            history_steps: int,
            forecast_steps: int,
        ) -> None:
            self.source_tensor = np.asarray(source_tensor, dtype=np.float32)
            self.target_pu = np.asarray(target_pu, dtype=np.float32)
            self.windows = windows
            self.history_steps = history_steps
            self.forecast_steps = forecast_steps

        def __len__(self) -> int:
            return len(self.windows)

        def __getitem__(self, index: int) -> tuple[np.ndarray, np.ndarray]:
            target_index = int(self.windows.target_indices[index])
            history = self.source_tensor[target_index - self.history_steps : target_index]
            targets = self.target_pu[target_index : target_index + self.forecast_steps]
            if history.shape != (self.history_steps, self.source_tensor.shape[1], self.source_tensor.shape[2]):
                raise ValueError(f"History slice for index {index} has unexpected shape {history.shape!r}.")
            if targets.shape != (self.forecast_steps, self.target_pu.shape[1]):
                raise ValueError(f"Target slice for index {index} has unexpected shape {targets.shape!r}.")
            if not np.isfinite(history).all() or not np.isfinite(targets).all():
                raise ValueError(f"Window index {index} contains non-finite values despite strict filtering.")
            return (
                history.astype(np.float32, copy=True),
                targets[:, :, None].astype(np.float32, copy=True),
            )

else:

    class PanelWindowDataset:  # pragma: no cover - exercised only when torch is missing
        def __init__(self, *_args, **_kwargs) -> None:
            require_torch()


def prepare_dataset(
    dataset_id: str,
    *,
    variant_spec: ExperimentVariant | None = None,
    cache_root: str | Path = _CACHE_ROOT,
    max_train_origins: int | None = None,
    max_eval_origins: int | None = None,
) -> PreparedDataset:
    resolved_variant = variant_spec or VARIANT_SPECS[0]
    context = _load_variant_dataset_context(
        dataset_id,
        variant_spec=resolved_variant,
        cache_root=cache_root,
    )
    return _finalize_prepared_dataset(
        context,
        split_frames=_split_context_window_index(context),
        max_train_origins=max_train_origins,
        max_eval_origins=max_eval_origins,
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
    input_channels: int = INPUT_CHANNELS,
    hidden_dim: int,
    embed_dim: int,
    num_layers: int,
    cheb_k: int,
    forecast_steps: int,
):
    require_torch()
    return AGCRN(
        node_count=node_count,
        input_channels=input_channels,
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
        prepared_dataset.source_tensor,
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
    checkpoint_path: str | Path | None = None,
    training_history_path: str | Path | None = None,
    resume_from_checkpoint: bool = False,
    progress_label: str | None = None,
) -> TrainingOutcome:
    resolved_torch, _, _, _, _ = require_torch()
    _set_random_seed(seed)
    resolved_device = resolve_device(device)
    model = build_model(
        node_count=prepared_dataset.node_count,
        input_channels=prepared_dataset.input_channels,
        hidden_dim=hidden_dim,
        embed_dim=embed_dim,
        num_layers=num_layers,
        cheb_k=cheb_k,
        forecast_steps=prepared_dataset.forecast_steps,
    ).to(device=resolved_device)
    initialize_official_aligned_parameters(model)
    optimizer = resolved_torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = resolved_torch.nn.MSELoss()
    val_loader = _build_dataloader(
        prepared_dataset,
        windows=prepared_dataset.val_rolling_windows,
        batch_size=batch_size,
        shuffle=False,
        seed=seed,
    )

    resolved_checkpoint_path = Path(checkpoint_path) if checkpoint_path is not None else None
    resolved_training_history_path = Path(training_history_path) if training_history_path is not None else None
    checkpoint_job_identity = _job_identity_for_prepared_dataset(prepared_dataset)
    best_state: dict[str, Any] | None = None
    best_epoch = 0
    best_val_rmse_pu = float("inf")
    epochs_without_improvement = 0
    epochs_ran = 0
    start_epoch = 1
    if resume_from_checkpoint:
        if resolved_checkpoint_path is None:
            raise ValueError("resume_from_checkpoint=True requires checkpoint_path to be set.")
        if resolved_checkpoint_path.exists():
            try:
                checkpoint_payload = _torch_load_checkpoint(resolved_checkpoint_path, map_location=resolved_device)
            except Exception as exc:
                raise RuntimeError(
                    f"Failed to load AGCRN checkpoint from {resolved_checkpoint_path}: {exc}"
                ) from exc
            if checkpoint_payload.get("schema_version") != _TRAINING_CHECKPOINT_SCHEMA_VERSION:
                raise RuntimeError(
                    f"Unsupported AGCRN checkpoint schema at {resolved_checkpoint_path}: "
                    f"{checkpoint_payload.get('schema_version')!r}."
                )
            if checkpoint_payload.get("job") != checkpoint_job_identity:
                raise RuntimeError(
                    f"AGCRN checkpoint at {resolved_checkpoint_path} does not match "
                    f"{prepared_dataset.dataset_id}/{prepared_dataset.model_variant}."
                )
            if int(checkpoint_payload.get("seed", seed)) != seed:
                raise RuntimeError(
                    f"AGCRN checkpoint at {resolved_checkpoint_path} was created with seed "
                    f"{checkpoint_payload.get('seed')!r}, expected {seed!r}."
                )
            try:
                model.load_state_dict(checkpoint_payload["model_state_dict"])
                optimizer.load_state_dict(checkpoint_payload["optimizer_state_dict"])
            except Exception as exc:
                raise RuntimeError(
                    f"Failed to restore AGCRN model or optimizer state from {resolved_checkpoint_path}: {exc}"
                ) from exc
            best_state = checkpoint_payload.get("best_state_dict")
            best_epoch = int(checkpoint_payload["best_epoch"])
            best_val_rmse_pu = float(checkpoint_payload["best_val_rmse_pu"])
            epochs_without_improvement = int(checkpoint_payload["epochs_without_improvement"])
            epochs_ran = int(checkpoint_payload["epochs_ran"])
            start_epoch = int(checkpoint_payload["next_epoch"])
            if bool(checkpoint_payload.get("training_complete", False)):
                if best_state is None:
                    raise RuntimeError(
                        f"AGCRN checkpoint at {resolved_checkpoint_path} marks training complete but has no best state."
                    )
                model.load_state_dict(best_state)
                return TrainingOutcome(
                    best_epoch=best_epoch,
                    epochs_ran=epochs_ran,
                    best_val_rmse_pu=best_val_rmse_pu,
                    device=resolved_device,
                    model=model,
                )
    if resolved_training_history_path is not None:
        _prune_training_history_for_job(
            resolved_training_history_path,
            job_identity=checkpoint_job_identity,
            min_epoch=start_epoch,
        )
    epoch_progress = _create_progress_bar(
        total=max_epochs,
        desc=f"{progress_label or prepared_dataset.dataset_id} epochs",
        leave=True,
    )
    try:
        if start_epoch > 1:
            epoch_progress.update(start_epoch - 1)
        for epoch_index in range(start_epoch, max_epochs + 1):
            model.train()
            train_loader = _build_dataloader(
                prepared_dataset,
                windows=prepared_dataset.train_windows,
                batch_size=batch_size,
                shuffle=True,
                seed=seed + epoch_index,
            )
            batch_progress = _create_progress_bar(
                total=_loader_batch_total(train_loader),
                desc=f"{progress_label or prepared_dataset.dataset_id} train e{epoch_index}",
            )
            last_loss_value: float | None = None
            train_loss_sum = 0.0
            train_loss_weight = 0
            train_batch_count = 0
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
                        batch_window_count = int(batch_history.shape[0])
                        train_loss_sum += last_loss_value * batch_window_count
                        train_loss_weight += batch_window_count
                        train_batch_count += 1
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
            is_best_epoch = False
            if val_rmse_pu < best_val_rmse_pu - 1e-12:
                best_val_rmse_pu = val_rmse_pu
                best_epoch = epoch_index
                best_state = copy.deepcopy(model.state_dict())
                epochs_without_improvement = 0
                is_best_epoch = True
            else:
                epochs_without_improvement += 1
            train_loss_mean = train_loss_sum / train_loss_weight if train_loss_weight else math.nan
            epoch_progress.update(1)
            postfix_parts = [f"val_rmse={val_rmse_pu:.4f}", f"best={best_val_rmse_pu:.4f}"]
            if last_loss_value is not None:
                postfix_parts.insert(0, f"loss={last_loss_value:.4f}")
            epoch_progress.set_postfix_str(" ".join(postfix_parts))
            should_stop = epochs_without_improvement >= early_stopping_patience or epoch_index >= max_epochs
            if resolved_training_history_path is not None:
                _append_training_history_row(
                    resolved_training_history_path,
                    {
                        "dataset_id": prepared_dataset.dataset_id,
                        "model_id": MODEL_ID,
                        "model_variant": prepared_dataset.model_variant,
                        "feature_protocol_id": prepared_dataset.feature_protocol_id,
                        "task_id": TASK_ID,
                        "window_protocol": WINDOW_PROTOCOL,
                        "epoch": epoch_index,
                        "train_loss_mean": train_loss_mean,
                        "train_loss_last": last_loss_value,
                        "val_rmse_pu": val_rmse_pu,
                        "best_val_rmse_pu": best_val_rmse_pu,
                        "is_best_epoch": is_best_epoch,
                        "epochs_without_improvement": epochs_without_improvement,
                        "train_batch_count": train_batch_count,
                        "train_window_count": len(prepared_dataset.train_windows),
                        "val_window_count": len(prepared_dataset.val_rolling_windows),
                        "device": resolved_device,
                        "seed": seed,
                        "batch_size": batch_size,
                        "learning_rate": learning_rate,
                    },
                )
            if resolved_checkpoint_path is not None:
                _save_training_checkpoint(
                    resolved_checkpoint_path,
                    {
                        "schema_version": _TRAINING_CHECKPOINT_SCHEMA_VERSION,
                        "job": checkpoint_job_identity,
                        "seed": seed,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "best_state_dict": best_state,
                        "best_epoch": best_epoch,
                        "best_val_rmse_pu": best_val_rmse_pu,
                        "epochs_without_improvement": epochs_without_improvement,
                        "epochs_ran": epochs_ran,
                        "next_epoch": epoch_index + 1,
                        "training_complete": should_stop,
                    },
                )
            if should_stop:
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
        "model_variant": prepared_dataset.model_variant,
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
        "input_channels": prepared_dataset.input_channels,
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
    checkpoint_path: str | Path | None = None,
    training_history_path: str | Path | None = None,
    resume_from_checkpoint: bool = False,
) -> list[dict[str, object]]:
    dataset_start = time.monotonic()
    progress_label = f"{prepared_dataset.dataset_id}/{prepared_dataset.model_variant}"
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
        checkpoint_path=checkpoint_path,
        training_history_path=training_history_path,
        resume_from_checkpoint=resume_from_checkpoint,
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
        feature_protocol_id=prepared_dataset.feature_protocol_id,
        input_channels=prepared_dataset.input_channels,
        model_variant=prepared_dataset.model_variant,
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
            pl.col("model_variant")
            .replace_strict(_MODEL_VARIANT_ORDER, default=len(_MODEL_VARIANT_ORDER))
            .alias("__model_variant_order"),
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
                "__model_variant_order",
                "__split_order",
                "__eval_protocol_order",
                "__metric_scope_order",
                "__lead_order",
            ]
        )
        .drop(
            [
                "__dataset_order",
                "__model_variant_order",
                "__split_order",
                "__eval_protocol_order",
                "__metric_scope_order",
                "__lead_order",
            ]
        )
    )


def sort_training_history_frame(frame: pl.DataFrame) -> pl.DataFrame:
    if frame.is_empty():
        return frame.select(_TRAINING_HISTORY_COLUMNS)
    return (
        frame.select(_TRAINING_HISTORY_COLUMNS)
        .with_columns(
            pl.col("dataset_id")
            .replace_strict(_DATASET_ORDER, default=len(_DATASET_ORDER))
            .alias("__dataset_order"),
            pl.col("model_variant")
            .replace_strict(_MODEL_VARIANT_ORDER, default=len(_MODEL_VARIANT_ORDER))
            .alias("__model_variant_order"),
        )
        .sort(["__dataset_order", "__model_variant_order", "epoch"])
        .drop(["__dataset_order", "__model_variant_order"])
    )


def run_experiment(
    *,
    dataset_ids: Sequence[str] = DEFAULT_DATASETS,
    variant_names: Sequence[str] | None = None,
    cache_root: str | Path = _CACHE_ROOT,
    output_path: str | Path = _OUTPUT_PATH,
    device: str | None = None,
    max_epochs: int | None = None,
    max_train_origins: int | None = None,
    max_eval_origins: int | None = None,
    seed: int = DEFAULT_SEED,
    batch_size: int | None = None,
    learning_rate: float | None = None,
    early_stopping_patience: int | None = None,
    hidden_dim: int | None = None,
    embed_dim: int | None = None,
    num_layers: int | None = None,
    cheb_k: int | None = None,
    grad_clip_norm: float | None = None,
    resume: bool = False,
    force_rerun: bool = False,
    work_root: str | Path = _RUN_AGCRN_WORK_ROOT,
    dataset_loader: Callable[..., PreparedDataset] | None = None,
    job_runner: Callable[..., list[dict[str, object]]] | None = None,
) -> pl.DataFrame:
    variant_specs = resolve_variant_specs(variant_names)
    runner = job_runner or execute_training_job
    output = _normalize_output_path(output_path)
    resume_paths = _resume_paths_for_output(output_path=output, work_root=work_root)
    effective_config = _build_effective_config(
        dataset_ids=dataset_ids,
        variant_specs=variant_specs,
        seed=seed,
        max_train_origins=max_train_origins,
        max_eval_origins=max_eval_origins,
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
    existing_state = _load_resume_state_if_exists(resume_paths)
    if resume and force_rerun:
        raise ValueError("AGCRN --resume and --force-rerun are mutually exclusive.")
    if force_rerun:
        _reset_resume_slot(resume_paths, effective_config=effective_config)
        existing_state = None
    elif resume:
        if existing_state is None:
            raise ValueError(
                f"No AGCRN resume state exists for output path {output}. Expected {resume_paths.state_path}."
            )
        if existing_state.get("effective_config") != effective_config:
            raise ValueError(
                f"AGCRN resume state at {resume_paths.state_path} does not match the requested run configuration."
            )
    else:
        if existing_state is None:
            _reset_resume_slot(resume_paths, effective_config=effective_config)
        elif existing_state.get("status") == "complete":
            _reset_resume_slot(resume_paths, effective_config=effective_config)
            existing_state = None
        else:
            raise ValueError(
                f"AGCRN resume state at {resume_paths.state_path} is still marked running. "
                f"Re-run with --resume, --force-rerun, or remove {resume_paths.slot_dir}."
            )
    partial_results = _read_partial_results(resume_paths)
    rows: list[dict[str, object]] = partial_results.to_dicts()
    completed_job_keys = _completed_job_keys(partial_results)
    total_jobs = len(dataset_ids) * len(variant_specs)
    if existing_state is not None and existing_state.get("status") == "complete" and len(completed_job_keys) != total_jobs:
        raise ValueError(
            f"AGCRN resume state at {resume_paths.state_path} is marked complete, but "
            f"only {len(completed_job_keys)} of {total_jobs} jobs are present in partial results."
        )
    if len(completed_job_keys) == total_jobs:
        results = _result_frame_from_rows(rows)
        _atomic_write_csv(output, results)
        _publish_training_history(resume_paths, output)
        _clear_checkpoint_dir(resume_paths)
        _write_resume_state(resume_paths, status="complete", effective_config=effective_config, active_job=None)
        return results

    resume_active_job = None if existing_state is None else existing_state.get("active_job")
    job_progress = _create_progress_bar(total=total_jobs, desc="agcrn jobs", leave=True)
    try:
        for dataset_id in dataset_ids:
            dataset_job_keys = [_job_key(dataset_id, spec.model_variant) for spec in variant_specs]
            if all(job_key in completed_job_keys for job_key in dataset_job_keys):
                for _, model_variant in dataset_job_keys:
                    job_progress.set_postfix_str(f"{dataset_id}/{model_variant}")
                    job_progress.update(1)
                continue
            prepared_datasets = (
                _prepare_datasets_for_variants(
                    dataset_id,
                    variant_specs=variant_specs,
                    cache_root=cache_root,
                    max_train_origins=max_train_origins,
                    max_eval_origins=max_eval_origins,
                )
                if dataset_loader is None
                else tuple(
                    dataset_loader(
                        dataset_id,
                        variant_spec=variant_spec,
                        cache_root=cache_root,
                        max_train_origins=max_train_origins,
                        max_eval_origins=max_eval_origins,
                    )
                    for variant_spec in variant_specs
                )
            )
            for prepared in prepared_datasets:
                current_job_key = _job_key(prepared.dataset_id, prepared.model_variant)
                current_job_identity = _job_identity_for_prepared_dataset(prepared)
                resolved_profile = resolve_hyperparameter_profile(
                    prepared.model_variant,
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
                job_progress.set_postfix_str(f"{prepared.dataset_id}/{prepared.model_variant}")
                if current_job_key in completed_job_keys:
                    _delete_job_checkpoint(
                        resume_paths,
                        dataset_id=prepared.dataset_id,
                        model_variant=prepared.model_variant,
                    )
                    if resume_active_job == current_job_identity:
                        resume_active_job = None
                    job_progress.update(1)
                    continue
                checkpoint_path = _job_checkpoint_path(
                    resume_paths,
                    dataset_id=prepared.dataset_id,
                    model_variant=prepared.model_variant,
                )
                resume_from_checkpoint = resume_active_job == current_job_identity and checkpoint_path.exists()
                _write_resume_state(
                    resume_paths,
                    status="running",
                    effective_config=effective_config,
                    active_job=current_job_identity,
                )
                rows.extend(
                    runner(
                        prepared,
                        device=device,
                        seed=seed,
                        batch_size=resolved_profile.batch_size,
                        learning_rate=resolved_profile.learning_rate,
                        max_epochs=resolved_profile.max_epochs,
                        early_stopping_patience=resolved_profile.early_stopping_patience,
                        hidden_dim=resolved_profile.hidden_dim,
                        embed_dim=resolved_profile.embed_dim,
                        num_layers=resolved_profile.num_layers,
                        cheb_k=resolved_profile.cheb_k,
                        grad_clip_norm=resolved_profile.grad_clip_norm,
                        checkpoint_path=checkpoint_path,
                        training_history_path=resume_paths.training_history_path,
                        resume_from_checkpoint=resume_from_checkpoint,
                    )
                )
                partial_results = _result_frame_from_rows(rows)
                _write_partial_results(resume_paths, partial_results)
                completed_job_keys.add(current_job_key)
                _delete_job_checkpoint(
                    resume_paths,
                    dataset_id=prepared.dataset_id,
                    model_variant=prepared.model_variant,
                )
                _write_resume_state(resume_paths, status="running", effective_config=effective_config, active_job=None)
                resume_active_job = None
                job_progress.update(1)
    finally:
        job_progress.close()

    results = _result_frame_from_rows(rows)
    _atomic_write_csv(output, results)
    _publish_training_history(resume_paths, output)
    _clear_checkpoint_dir(resume_paths)
    _write_resume_state(resume_paths, status="complete", effective_config=effective_config, active_job=None)
    return results


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the official-aligned farm-synchronous AGCRN variants on kelmarsh and penmanshiel."
    )
    parser.add_argument(
        "--dataset",
        action="append",
        choices=list(DEFAULT_DATASETS),
        dest="datasets",
        help="Limit execution to one or more datasets. Defaults to kelmarsh and penmanshiel.",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cuda", "mps", "cpu"),
        default="auto",
        help="Training device. Defaults to auto (cuda -> mps -> cpu).",
    )
    parser.add_argument(
        "--variant",
        action="append",
        choices=list(DEFAULT_VARIANTS),
        dest="variants",
        help="Limit execution to one or more model variants. Defaults to running all nine active variants.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Maximum training epochs. Defaults to the tuned per-variant profile when omitted.",
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
        default=None,
        help="Training and evaluation batch size. Defaults to the tuned per-variant profile when omitted.",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=None,
        help="Adam learning rate. Defaults to the tuned per-variant profile when omitted.",
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
        default=None,
        help="Early stopping patience in epochs. Defaults to the tuned per-variant profile when omitted.",
    )
    parser.add_argument(
        "--hidden-dim",
        type=int,
        default=None,
        help="Hidden dimension for the recurrent state. Defaults to the tuned per-variant profile when omitted.",
    )
    parser.add_argument(
        "--embed-dim",
        type=int,
        default=None,
        help="Learned node-embedding dimension for the adaptive graph. Defaults to the tuned per-variant profile when omitted.",
    )
    parser.add_argument(
        "--num-layers",
        type=int,
        default=None,
        help="Number of stacked AGCRN recurrent layers. Defaults to the tuned per-variant profile when omitted.",
    )
    parser.add_argument(
        "--cheb-k",
        type=int,
        default=None,
        help="Adaptive graph convolution support order. Defaults to the tuned per-variant profile when omitted.",
    )
    parser.add_argument(
        "--grad-clip-norm",
        type=float,
        default=None,
        help="Gradient clipping max norm. Use 0 to disable clipping. Defaults to the tuned per-variant profile when omitted.",
    )
    parser.add_argument(
        "--run-label",
        type=str,
        default=None,
        help="Optional label suffix for the formal run record under experiment/artifacts/runs/agcrn_official_aligned/.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume an interrupted run_agcrn invocation from experiment/families/agcrn/.work/.",
    )
    parser.add_argument(
        "--force-rerun",
        action="store_true",
        help="Discard any existing resume state for the resolved --output-path and start a fresh run.",
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
    variant_specs = resolve_variant_specs(tuple(args.variants) if args.variants else None)
    resolved_variant_hyperparameters = {
        spec.model_variant: {
            "feature_protocol_id": spec.feature_protocol_id,
            **resolve_hyperparameter_profile(
                spec.model_variant,
                batch_size=args.batch_size,
                learning_rate=args.learning_rate,
                max_epochs=args.epochs,
                early_stopping_patience=args.patience,
                hidden_dim=args.hidden_dim,
                embed_dim=args.embed_dim,
                num_layers=args.num_layers,
                cheb_k=args.cheb_k,
                grad_clip_norm=args.grad_clip_norm,
            ).__dict__,
        }
        for spec in variant_specs
    }
    results = run_experiment(
        dataset_ids=tuple(args.datasets) if args.datasets else DEFAULT_DATASETS,
        variant_names=tuple(spec.model_variant for spec in variant_specs),
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
        resume=args.resume,
        force_rerun=args.force_rerun,
    )
    if not args.no_record_run:
        recorded_args = vars(args).copy()
        recorded_args["resolved_variant_hyperparameters"] = resolved_variant_hyperparameters
        record_cli_run(
            family_id=FAMILY_ID,
            repo_root=_REPO_ROOT,
            invocation_kind="family_runner",
            entrypoint="experiment/families/agcrn/run_agcrn.py",
            args=recorded_args,
            output_path=args.output_path,
            result_row_count=results.height,
            dataset_ids=tuple(args.datasets) if args.datasets else DEFAULT_DATASETS,
            feature_protocol_ids=tuple(spec.feature_protocol_id for spec in variant_specs),
            model_variants=tuple(spec.model_variant for spec in variant_specs),
            eval_protocols=(ROLLING_EVAL_PROTOCOL, NON_OVERLAP_EVAL_PROTOCOL),
            result_splits=("val", "test"),
            artifacts={"training_history": training_history_output_path(args.output_path)},
            run_label=args.run_label,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
