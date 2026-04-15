from __future__ import annotations

import argparse
import copy
from dataclasses import asdict, dataclass, replace
from datetime import UTC, datetime
import hashlib
import json
import math
import os
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


EXPERIMENT_DIR = Path(__file__).resolve().parent
EXPERIMENT_ROOT = EXPERIMENT_DIR.parents[1]
COMMON_DIR = EXPERIMENT_ROOT / "infra" / "common"
if str(COMMON_DIR) not in sys.path:
    sys.path.insert(0, str(COMMON_DIR))

import window_protocols  # noqa: E402
from published_outputs import default_family_output_path  # noqa: E402
from run_records import record_cli_run  # noqa: E402
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

WORLD_MODEL_AGCRN_DIR = EXPERIMENT_DIR.parent / "world_model_agcrn_v1"
if str(WORLD_MODEL_AGCRN_DIR) not in sys.path:
    sys.path.insert(0, str(WORLD_MODEL_AGCRN_DIR))

import world_model_agcrn_v1 as world_model_base  # noqa: E402


torch = world_model_base.torch
nn = world_model_base.nn
F = world_model_base.F
DataLoader = world_model_base.DataLoader
Dataset = world_model_base.Dataset


MODEL_ID = "WORLD_MODEL"
FAMILY_ID = "world_model_rollout_v1"
MODEL_VARIANT = "world_model_rollout_v1_farm_sync"
WINDOW_PROTOCOL = DEFAULT_WINDOW_PROTOCOL
TASK_PROTOCOL: WindowProtocolSpec = resolve_window_protocol(WINDOW_PROTOCOL)
TASK_ID = TASK_PROTOCOL.task_id
DEFAULT_DATASETS = ("kelmarsh", "penmanshiel")
HISTORY_STEPS = 144
FORECAST_STEPS = 36
STRIDE_STEPS = 1
FEATURE_PROTOCOL_ID = "world_model_v1"
DEFAULT_BATCH_SIZE = 64
KELMARSH_DEFAULT_BATCH_SIZE = 256
PENMANSHIEL_DEFAULT_BATCH_SIZE = 64
DEFAULT_LEARNING_RATE = 1e-3
DEFAULT_MAX_EPOCHS = 20
DEFAULT_EARLY_STOPPING_PATIENCE = 5
DEFAULT_SEED = 42
DEFAULT_NODE_STATE_DIM = 128
DEFAULT_HIDDEN_STATE_DIM = 96
DEFAULT_LATENT_STATE_DIM = 32
DEFAULT_GLOBAL_STATE_DIM = 64
DEFAULT_MESSAGE_DIM = 64
DEFAULT_EDGE_HIDDEN_DIM = 64
DEFAULT_TAU_EMBED_DIM = 16
DEFAULT_DROPOUT = 0.1
DEFAULT_GRAD_CLIP_NORM = 1.0
DEFAULT_HIST_LOSS_WEIGHT = 0.3
DEFAULT_AUX_TURBINE_LOSS_WEIGHT = 0.1
DEFAULT_AUX_SITE_LOSS_WEIGHT = 0.1
DEFAULT_CONSISTENCY_LOSS_WEIGHT = 0.05
DEFAULT_WEIGHT_DECAY = 0.0
DEFAULT_EVAL_BATCH_SIZE_MULTIPLIER = 2
DEFAULT_MAX_CUDA_EVAL_BATCH_SIZE = 128
DEFAULT_CUDA_NUM_WORKERS = 0
DEFAULT_CUDA_PREFETCH_FACTOR = 4
PROFILE_LOG_PREFIX = "[world_model_rollout_v1] "

EXPECTED_TARGET_HISTORY_MASK_COLUMNS = ("target_kw__mask",)
EXPECTED_LOCAL_OBSERVATION_VALUE_COUNT = 16
EXPECTED_GLOBAL_OBSERVATION_VALUE_COUNT = 9
KNOWN_FUTURE_FEATURE_COLUMNS = (
    "calendar_hour_sin",
    "calendar_hour_cos",
    "calendar_weekday_sin",
    "calendar_weekday_cos",
    "calendar_month_sin",
    "calendar_month_cos",
    "calendar_is_weekend",
)
STATIC_FEATURE_NAMES = (
    "latitude",
    "longitude",
    "elevation_m",
    "rated_power_kw",
    "hub_height_m",
    "rotor_diameter_m",
)
PAIRWISE_FEATURE_NAMES = (
    "delta_x_m",
    "delta_y_m",
    "log_distance_m",
    "bearing_sin",
    "bearing_cos",
    "elevation_diff_m",
    "distance_in_rotor_diameters",
)

_REPO_ROOT = EXPERIMENT_ROOT.parent
_CACHE_ROOT = _REPO_ROOT / "cache"
_OUTPUT_PATH = default_family_output_path(repo_root=_REPO_ROOT, family_id=FAMILY_ID)
_RUN_WORK_ROOT = EXPERIMENT_DIR / ".work" / "run_world_model_rollout_v1"
_RUN_STATE_SCHEMA_VERSION = "world_model_rollout_v1.run.resume.v1"
_TRAINING_CHECKPOINT_SCHEMA_VERSION = "world_model_rollout_v1.training_checkpoint.v1"
_SPLIT_NAMES = ("train", "val", "test")
_WINDOW_KEY_COLUMNS = ("output_start_ts", "output_end_ts")
_SPLIT_ORDER = {"val": 0, "test": 1}
_EVAL_PROTOCOL_ORDER = {ROLLING_EVAL_PROTOCOL: 0, NON_OVERLAP_EVAL_PROTOCOL: 1}
_METRIC_SCOPE_ORDER = {OVERALL_METRIC_SCOPE: 0, HORIZON_METRIC_SCOPE: 1}
_DATASET_ORDER = {dataset_id: index for index, dataset_id in enumerate(DEFAULT_DATASETS)}
_MODEL_VARIANT_ORDER = {MODEL_VARIANT: 0}

_BINARY_OBSERVATION_COLUMNS = {
    "evt_any_active",
    "evt_stop_active",
    "evt_warning_active",
    "evt_informational_active",
    "farm_evt_any_active",
    "farm_evt_stop_active",
    "farm_evt_warning_active",
    "farm_evt_informational_active",
}
_RAW_OBSERVATION_COLUMNS = {
    "wind_direction_sin",
    "wind_direction_cos",
    "yaw_error_sin",
    "yaw_error_cos",
}
_LOG1P_OBSERVATION_COLUMNS = {
    "evt_active_count",
    "evt_total_overlap_seconds",
    "farm_evt_active_count",
    "farm_evt_total_overlap_seconds",
}
_PAIRWISE_RAW_COLUMNS = {"bearing_sin", "bearing_cos"}
_OBS_TARGET_VALUE_INDEX = 0
_OBS_TARGET_MASK_INDEX = 1
_SERIES_BASE_COLUMNS = (
    "dataset",
    "turbine_id",
    "timestamp",
    "target_kw",
    "is_observed",
    "quality_flags",
    "feature_quality_flags",
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
    "coordinate_mode",
    "node_count",
    "local_input_channels",
    "context_channels",
    "static_feature_count",
    "pairwise_feature_count",
    "node_state_dim",
    "hidden_state_dim",
    "latent_state_dim",
    "global_state_dim",
    "message_dim",
    "edge_hidden_dim",
    "tau_embed_dim",
    "dropout",
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
    "hist_loss_weight",
    "aux_turbine_loss_weight",
    "aux_site_loss_weight",
    "consistency_loss_weight",
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
    "train_future_loss_mean",
    "train_future_loss_last",
    "train_hist1_loss_mean",
    "train_hist1_loss_last",
    "train_aux_turbine_loss_mean",
    "train_aux_turbine_loss_last",
    "train_aux_site_loss_mean",
    "train_aux_site_loss_last",
    "train_consistency_loss_mean",
    "train_consistency_loss_last",
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
    node_state_dim: int
    hidden_state_dim: int
    latent_state_dim: int
    global_state_dim: int
    message_dim: int
    edge_hidden_dim: int
    tau_embed_dim: int
    dropout: float
    grad_clip_norm: float
    hist_loss_weight: float
    aux_turbine_loss_weight: float
    aux_site_loss_weight: float
    consistency_loss_weight: float
    weight_decay: float


@dataclass(frozen=True)
class DatasetContext:
    dataset_id: str
    model_variant: str
    feature_protocol_id: str
    metadata: Any
    series: pl.DataFrame
    known_future: pl.DataFrame
    pairwise: pl.DataFrame
    target_history_mask_columns: tuple[str, ...]
    local_observation_value_columns: tuple[str, ...]
    local_observation_mask_columns: tuple[str, ...]
    global_observation_value_columns: tuple[str, ...]
    global_observation_mask_columns: tuple[str, ...]
    known_future_feature_columns: tuple[str, ...]
    window_index: pl.DataFrame
    raw_timestamps: tuple[datetime, ...]
    resolution_minutes: int
    timestamps_us: np.ndarray
    target_pu: np.ndarray
    coordinate_mode: str
    distance_sanity: pl.DataFrame


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
    local_input_feature_names: tuple[str, ...]
    context_feature_names: tuple[str, ...]
    static_feature_names: tuple[str, ...]
    pairwise_feature_names: tuple[str, ...]
    local_history_tensor: np.ndarray
    context_history_tensor: np.ndarray
    context_future_tensor: np.ndarray
    static_tensor: np.ndarray
    pairwise_tensor: np.ndarray
    target_pu: np.ndarray
    target_pu_filled: np.ndarray
    target_valid_mask: np.ndarray
    train_windows: world_model_base.FarmWindowDescriptorIndex
    val_rolling_windows: world_model_base.FarmWindowDescriptorIndex
    val_non_overlap_windows: world_model_base.FarmWindowDescriptorIndex
    test_rolling_windows: world_model_base.FarmWindowDescriptorIndex
    test_non_overlap_windows: world_model_base.FarmWindowDescriptorIndex

    @property
    def local_input_channels(self) -> int:
        return int(self.local_history_tensor.shape[2])

    @property
    def context_channels(self) -> int:
        return int(self.context_history_tensor.shape[1])

    @property
    def static_feature_count(self) -> int:
        return int(self.static_tensor.shape[1])

    @property
    def pairwise_feature_count(self) -> int:
        return int(self.pairwise_tensor.shape[2])


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


@dataclass(frozen=True)
class ModelOutputs:
    future_predictions: Any
    hist_prior_predictions: Any
    aux_turbine_predictions: Any
    aux_site_predictions: Any


VARIANT_SPECS = (
    ExperimentVariant(model_variant=MODEL_VARIANT, feature_protocol_id=FEATURE_PROTOCOL_ID),
)
DEFAULT_VARIANTS = tuple(spec.model_variant for spec in VARIANT_SPECS)
_VARIANT_SPECS_BY_NAME = {spec.model_variant: spec for spec in VARIANT_SPECS}
_DEFAULT_PROFILE = HyperparameterProfile(
    batch_size=DEFAULT_BATCH_SIZE,
    learning_rate=DEFAULT_LEARNING_RATE,
    max_epochs=DEFAULT_MAX_EPOCHS,
    early_stopping_patience=DEFAULT_EARLY_STOPPING_PATIENCE,
    node_state_dim=DEFAULT_NODE_STATE_DIM,
    hidden_state_dim=DEFAULT_HIDDEN_STATE_DIM,
    latent_state_dim=DEFAULT_LATENT_STATE_DIM,
    global_state_dim=DEFAULT_GLOBAL_STATE_DIM,
    message_dim=DEFAULT_MESSAGE_DIM,
    edge_hidden_dim=DEFAULT_EDGE_HIDDEN_DIM,
    tau_embed_dim=DEFAULT_TAU_EMBED_DIM,
    dropout=DEFAULT_DROPOUT,
    grad_clip_norm=DEFAULT_GRAD_CLIP_NORM,
    hist_loss_weight=DEFAULT_HIST_LOSS_WEIGHT,
    aux_turbine_loss_weight=DEFAULT_AUX_TURBINE_LOSS_WEIGHT,
    aux_site_loss_weight=DEFAULT_AUX_SITE_LOSS_WEIGHT,
    consistency_loss_weight=DEFAULT_CONSISTENCY_LOSS_WEIGHT,
    weight_decay=DEFAULT_WEIGHT_DECAY,
)
TUNED_DEFAULT_HYPERPARAMETERS_BY_DATASET_AND_VARIANT = {
    "kelmarsh": {MODEL_VARIANT: replace(_DEFAULT_PROFILE, batch_size=KELMARSH_DEFAULT_BATCH_SIZE)},
    "penmanshiel": {MODEL_VARIANT: replace(_DEFAULT_PROFILE, batch_size=PENMANSHIEL_DEFAULT_BATCH_SIZE)},
}


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


def resolve_hyperparameter_profile(
    variant_name: str,
    *,
    dataset_id: str,
    batch_size: int | None = None,
    learning_rate: float | None = None,
    max_epochs: int | None = None,
    early_stopping_patience: int | None = None,
    node_state_dim: int | None = None,
    hidden_state_dim: int | None = None,
    latent_state_dim: int | None = None,
    global_state_dim: int | None = None,
    message_dim: int | None = None,
    edge_hidden_dim: int | None = None,
    tau_embed_dim: int | None = None,
    dropout: float | None = None,
    grad_clip_norm: float | None = None,
    hist_loss_weight: float | None = None,
    aux_turbine_loss_weight: float | None = None,
    aux_site_loss_weight: float | None = None,
    consistency_loss_weight: float | None = None,
    weight_decay: float | None = None,
) -> HyperparameterProfile:
    try:
        defaults = TUNED_DEFAULT_HYPERPARAMETERS_BY_DATASET_AND_VARIANT[dataset_id][variant_name]
    except KeyError as exc:
        raise ValueError(
            f"Unsupported world-model rollout dataset/variant pair {dataset_id!r}/{variant_name!r}."
        ) from exc
    profile = HyperparameterProfile(
        batch_size=defaults.batch_size if batch_size is None else batch_size,
        learning_rate=defaults.learning_rate if learning_rate is None else learning_rate,
        max_epochs=defaults.max_epochs if max_epochs is None else max_epochs,
        early_stopping_patience=(
            defaults.early_stopping_patience
            if early_stopping_patience is None
            else early_stopping_patience
        ),
        node_state_dim=defaults.node_state_dim if node_state_dim is None else node_state_dim,
        hidden_state_dim=defaults.hidden_state_dim if hidden_state_dim is None else hidden_state_dim,
        latent_state_dim=defaults.latent_state_dim if latent_state_dim is None else latent_state_dim,
        global_state_dim=defaults.global_state_dim if global_state_dim is None else global_state_dim,
        message_dim=defaults.message_dim if message_dim is None else message_dim,
        edge_hidden_dim=defaults.edge_hidden_dim if edge_hidden_dim is None else edge_hidden_dim,
        tau_embed_dim=defaults.tau_embed_dim if tau_embed_dim is None else tau_embed_dim,
        dropout=defaults.dropout if dropout is None else dropout,
        grad_clip_norm=defaults.grad_clip_norm if grad_clip_norm is None else grad_clip_norm,
        hist_loss_weight=defaults.hist_loss_weight if hist_loss_weight is None else hist_loss_weight,
        aux_turbine_loss_weight=(
            defaults.aux_turbine_loss_weight
            if aux_turbine_loss_weight is None
            else aux_turbine_loss_weight
        ),
        aux_site_loss_weight=defaults.aux_site_loss_weight if aux_site_loss_weight is None else aux_site_loss_weight,
        consistency_loss_weight=(
            defaults.consistency_loss_weight
            if consistency_loss_weight is None
            else consistency_loss_weight
        ),
        weight_decay=defaults.weight_decay if weight_decay is None else weight_decay,
    )
    if profile.node_state_dim != profile.hidden_state_dim + profile.latent_state_dim:
        raise ValueError(
            "node_state_dim must equal hidden_state_dim + latent_state_dim, found "
            f"{profile.node_state_dim} != {profile.hidden_state_dim} + {profile.latent_state_dim}."
        )
    if profile.dropout < 0.0 or profile.dropout >= 1.0:
        raise ValueError(f"dropout must be in [0, 1), found {profile.dropout!r}.")
    return profile


def _build_effective_config(
    *,
    dataset_ids: Sequence[str],
    variant_specs: Sequence[ExperimentVariant],
    device: str | None,
    seed: int,
    max_train_origins: int | None,
    max_eval_origins: int | None,
    batch_size: int | None,
    eval_batch_size: int | None,
    learning_rate: float | None,
    max_epochs: int | None,
    early_stopping_patience: int | None,
    node_state_dim: int | None,
    hidden_state_dim: int | None,
    latent_state_dim: int | None,
    global_state_dim: int | None,
    message_dim: int | None,
    edge_hidden_dim: int | None,
    tau_embed_dim: int | None,
    dropout: float | None,
    grad_clip_norm: float | None,
    hist_loss_weight: float | None,
    aux_turbine_loss_weight: float | None,
    aux_site_loss_weight: float | None,
    consistency_loss_weight: float | None,
    weight_decay: float | None,
    num_workers: int | None,
) -> dict[str, object]:
    resolved_device = world_model_base.resolve_device(device)
    return {
        "dataset_ids": list(dataset_ids),
        "variant_names": [spec.model_variant for spec in variant_specs],
        "device": resolved_device,
        "seed": seed,
        "max_train_origins": max_train_origins,
        "max_eval_origins": max_eval_origins,
        "eval_batch_size": eval_batch_size,
        "num_workers": resolve_loader_num_workers(device=resolved_device, num_workers=num_workers),
        "resolved_dataset_variant_hyperparameters": {
            dataset_id: {
                spec.model_variant: {
                    "feature_protocol_id": spec.feature_protocol_id,
                    **asdict(
                        resolve_hyperparameter_profile(
                            spec.model_variant,
                            dataset_id=dataset_id,
                            batch_size=batch_size,
                            learning_rate=learning_rate,
                            max_epochs=max_epochs,
                            early_stopping_patience=early_stopping_patience,
                            node_state_dim=node_state_dim,
                            hidden_state_dim=hidden_state_dim,
                            latent_state_dim=latent_state_dim,
                            global_state_dim=global_state_dim,
                            message_dim=message_dim,
                            edge_hidden_dim=edge_hidden_dim,
                            tau_embed_dim=tau_embed_dim,
                            dropout=dropout,
                            grad_clip_norm=grad_clip_norm,
                            hist_loss_weight=hist_loss_weight,
                            aux_turbine_loss_weight=aux_turbine_loss_weight,
                            aux_site_loss_weight=aux_site_loss_weight,
                            consistency_loss_weight=consistency_loss_weight,
                            weight_decay=weight_decay,
                        )
                    ),
                }
                for spec in variant_specs
            }
            for dataset_id in dataset_ids
        },
    }


def _load_resume_state(paths: ResumePaths) -> dict[str, object]:
    payload = json.loads(paths.state_path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != _RUN_STATE_SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported world-model rollout resume state schema at {paths.state_path}: "
            f"{payload.get('schema_version')!r}."
        )
    status = payload.get("status")
    if status not in {"running", "complete"}:
        raise ValueError(
            f"Unsupported world-model rollout resume state status at {paths.state_path}: {status!r}."
        )
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


def _require_torch() -> tuple[Any, Any, Any, Any, Any]:
    return world_model_base.require_torch()


def resolve_device(device: str | None = None) -> str:
    return world_model_base.resolve_device(device)


def resolve_eval_batch_size(
    train_batch_size: int,
    *,
    device: str,
    eval_batch_size: int | None = None,
) -> int:
    if eval_batch_size is not None:
        if eval_batch_size <= 0:
            raise ValueError(f"eval_batch_size must be positive, found {eval_batch_size!r}.")
        return eval_batch_size
    if train_batch_size <= 0:
        raise ValueError(f"train_batch_size must be positive, found {train_batch_size!r}.")
    if resolve_device(device) == "cuda":
        return max(
            train_batch_size,
            min(train_batch_size * DEFAULT_EVAL_BATCH_SIZE_MULTIPLIER, DEFAULT_MAX_CUDA_EVAL_BATCH_SIZE),
        )
    return train_batch_size


def resolve_loader_num_workers(*, device: str, num_workers: int | None = None) -> int:
    if num_workers is not None:
        if num_workers < 0:
            raise ValueError(f"num_workers must be >= 0, found {num_workers!r}.")
        return num_workers
    if resolve_device(device) != "cuda":
        return 0
    cpu_count = os.cpu_count() or 0
    if cpu_count <= 1:
        return 0
    return min(DEFAULT_CUDA_NUM_WORKERS, cpu_count)


def _loader_batch_total(loader: object) -> int | None:
    return world_model_base._loader_batch_total(loader)


def progress_is_enabled() -> bool:
    return HAS_TQDM and sys.stderr.isatty()


def _create_progress_bar(*, total: int | None, desc: str, leave: bool = False, enabled: bool | None = None):
    return tqdm(
        total=total,
        desc=desc,
        leave=leave,
        disable=not (progress_is_enabled() if enabled is None else enabled),
        dynamic_ncols=True,
    )


def _profile_log(dataset_id: str, phase: str, **fields: object) -> None:
    payload = {"dataset_id": dataset_id, "phase": phase, **fields}
    print(
        f"{PROFILE_LOG_PREFIX}{json.dumps(payload, ensure_ascii=False, default=str, sort_keys=True)}",
        file=sys.stderr,
        flush=True,
    )


def _resolve_observation_columns(bundle: Any) -> tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...], tuple[str, ...]]:
    column_groups = bundle.task_context.get("column_groups", {})
    if not isinstance(column_groups, dict):
        raise ValueError("Task bundle task_context is missing column_groups.")
    local_values = tuple(str(value) for value in (column_groups.get("local_observation_values") or ()))
    local_masks = tuple(str(value) for value in (column_groups.get("local_observation_masks") or ()))
    global_values = tuple(str(value) for value in (column_groups.get("global_observation_values") or ()))
    global_masks = tuple(str(value) for value in (column_groups.get("global_observation_masks") or ()))
    if len(local_values) != EXPECTED_LOCAL_OBSERVATION_VALUE_COUNT:
        raise ValueError(
            f"Expected {EXPECTED_LOCAL_OBSERVATION_VALUE_COUNT} local observation values, found {len(local_values)}."
        )
    if len(global_values) != EXPECTED_GLOBAL_OBSERVATION_VALUE_COUNT:
        raise ValueError(
            f"Expected {EXPECTED_GLOBAL_OBSERVATION_VALUE_COUNT} global observation values, found {len(global_values)}."
        )
    if local_masks != tuple(f"{value}__mask" for value in local_values):
        raise ValueError("Local observation masks do not align with the local observation values.")
    if global_masks != tuple(f"{value}__mask" for value in global_values):
        raise ValueError("Global observation masks do not align with the global observation values.")
    return local_values, local_masks, global_values, global_masks


def _build_variant_context(
    dataset_id: str,
    *,
    variant_spec: ExperimentVariant,
    cache_root: str | Path = _CACHE_ROOT,
) -> DatasetContext:
    bundle = world_model_base._load_task_bundle(
        dataset_id,
        feature_protocol_id=variant_spec.feature_protocol_id,
        cache_root=cache_root,
    )
    if bundle.task_context.get("feature_protocol_id") != variant_spec.feature_protocol_id:
        raise ValueError(
            f"Task bundle feature_protocol_id does not match {variant_spec.feature_protocol_id!r}."
        )
    metadata = world_model_base.load_dataset_metadata(dataset_id, bundle)
    target_history_mask_columns = world_model_base.resolve_target_history_mask_columns(bundle)
    if target_history_mask_columns != EXPECTED_TARGET_HISTORY_MASK_COLUMNS:
        raise ValueError(
            f"Task bundle target history mask contract drifted: {target_history_mask_columns!r}."
        )
    local_values, local_masks, global_values, global_masks = _resolve_observation_columns(bundle)
    known_future_feature_columns = world_model_base.resolve_known_future_feature_columns(bundle)
    past_covariate_columns = world_model_base.resolve_past_covariate_columns(bundle)
    series = _load_series_frame(
        dataset_id,
        bundle,
        target_history_mask_columns=target_history_mask_columns,
        past_covariate_columns=past_covariate_columns,
    )
    window_index = world_model_base.load_window_index(dataset_id, bundle)
    known_future = world_model_base.load_known_future_frame(
        dataset_id,
        bundle,
        known_future_feature_columns=known_future_feature_columns,
    )
    pairwise = world_model_base.load_pairwise_frame(dataset_id, bundle)
    panel_frame = world_model_base.prepare_panel_frame(series, rated_power_kw=metadata.rated_power_kw)
    resolution_minutes = world_model_base.resolve_resolution_minutes(panel_frame)
    raw_timestamps = tuple(panel_frame["timestamp"].to_list())
    timestamps_us, target_pu = world_model_base.build_panel_series(
        panel_frame,
        turbine_ids=metadata.turbine_ids,
    )
    coordinate_mode, distance_sanity = world_model_base.build_distance_sanity_frame(
        metadata.turbine_static,
        ordered_turbine_ids=metadata.turbine_ids,
    )
    return DatasetContext(
        dataset_id=dataset_id,
        model_variant=variant_spec.model_variant,
        feature_protocol_id=variant_spec.feature_protocol_id,
        metadata=metadata,
        series=series,
        known_future=known_future,
        pairwise=pairwise,
        target_history_mask_columns=target_history_mask_columns,
        local_observation_value_columns=local_values,
        local_observation_mask_columns=local_masks,
        global_observation_value_columns=global_values,
        global_observation_mask_columns=global_masks,
        known_future_feature_columns=known_future_feature_columns,
        window_index=window_index,
        raw_timestamps=raw_timestamps,
        resolution_minutes=resolution_minutes,
        timestamps_us=timestamps_us,
        target_pu=target_pu,
        coordinate_mode=coordinate_mode,
        distance_sanity=distance_sanity,
    )


def _load_series_frame(
    dataset_id: str,
    bundle: Any,
    *,
    target_history_mask_columns: Sequence[str],
    past_covariate_columns: Sequence[str],
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


def _build_feature_panel(
    series: pl.DataFrame,
    *,
    feature_column: str,
    turbine_ids: Sequence[str],
    expected_timestamps: Sequence[datetime],
) -> np.ndarray:
    return world_model_base._build_feature_panel(
        series,
        feature_column=feature_column,
        turbine_ids=turbine_ids,
        expected_timestamps=expected_timestamps,
    )


def _build_boolean_panel(
    series: pl.DataFrame,
    *,
    alias: str,
    expression: pl.Expr,
    turbine_ids: Sequence[str],
    expected_timestamps: Sequence[datetime],
) -> np.ndarray:
    augmented = series.with_columns(expression.alias(alias))
    return _build_feature_panel(
        augmented,
        feature_column=alias,
        turbine_ids=turbine_ids,
        expected_timestamps=expected_timestamps,
    )


def _history_row_mask(
    *,
    target_indices: np.ndarray,
    history_steps: int,
    total_steps: int,
) -> np.ndarray:
    return world_model_base._history_row_mask(
        target_indices=target_indices,
        history_steps=history_steps,
        total_steps=total_steps,
    )


def _fill_non_finite(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.float32).copy()
    array[~np.isfinite(array)] = 0.0
    return array.astype(np.float32, copy=False)


def _normalize_dense_history(values: np.ndarray, *, history_mask: np.ndarray, column_name: str) -> np.ndarray:
    return world_model_base._normalize_value_channel(
        values,
        history_mask=history_mask,
        column_name=column_name,
    )


def _prepare_mask_channel(values: np.ndarray, *, column_name: str) -> np.ndarray:
    return world_model_base._prepare_mask_channel(values, column_name=column_name)


def _transform_observation_values(
    values: np.ndarray,
    *,
    column_name: str,
    history_mask: np.ndarray,
    total_rated_power_kw: float,
) -> np.ndarray:
    transformed = np.asarray(values, dtype=np.float32).copy()
    if column_name == "farm_pmu__gms_power_kw":
        transformed = transformed / np.float32(total_rated_power_kw)
    if column_name in _LOG1P_OBSERVATION_COLUMNS:
        finite_mask = np.isfinite(transformed)
        transformed[finite_mask] = np.log1p(np.maximum(transformed[finite_mask], 0.0))
    if column_name in _RAW_OBSERVATION_COLUMNS or column_name in _BINARY_OBSERVATION_COLUMNS:
        return _fill_non_finite(transformed)
    return _normalize_dense_history(transformed, history_mask=history_mask, column_name=column_name)


def _build_global_feature_vector(
    series: pl.DataFrame,
    *,
    feature_column: str,
    turbine_ids: Sequence[str],
    expected_timestamps: Sequence[datetime],
) -> np.ndarray:
    panel = _build_feature_panel(
        series,
        feature_column=feature_column,
        turbine_ids=turbine_ids,
        expected_timestamps=expected_timestamps,
    )
    reference = panel[:, 0]
    for turbine_index in range(1, panel.shape[1]):
        comparison = panel[:, turbine_index]
        if not np.allclose(reference, comparison, equal_nan=True):
            raise ValueError(
                f"Global feature {feature_column!r} is not identical across turbines on the shared farm axis."
            )
    return reference.astype(np.float32, copy=False)


def _build_local_history_tensor(
    context: DatasetContext,
    *,
    history_mask: np.ndarray,
) -> tuple[np.ndarray, tuple[str, ...]]:
    turbine_ids = context.metadata.turbine_ids
    timestamps = context.raw_timestamps
    total_rated_power_kw = context.metadata.rated_power_kw * len(turbine_ids)
    target_filled = np.where(np.isfinite(context.target_pu), context.target_pu, 0.0).astype(np.float32, copy=False)
    target_mask = _prepare_mask_channel(
        _build_feature_panel(
            context.series,
            feature_column=context.target_history_mask_columns[0],
            turbine_ids=turbine_ids,
            expected_timestamps=timestamps,
        ),
        column_name=context.target_history_mask_columns[0],
    )
    channels: list[np.ndarray] = [target_filled, target_mask]
    feature_names: list[str] = ["target_pu", context.target_history_mask_columns[0]]
    for column_name in context.local_observation_value_columns:
        channels.append(
            _transform_observation_values(
                _build_feature_panel(
                    context.series,
                    feature_column=column_name,
                    turbine_ids=turbine_ids,
                    expected_timestamps=timestamps,
                ),
                column_name=column_name,
                history_mask=history_mask,
                total_rated_power_kw=total_rated_power_kw,
            )
        )
        feature_names.append(column_name)
    for column_name in context.local_observation_mask_columns:
        channels.append(
            _prepare_mask_channel(
                _build_feature_panel(
                    context.series,
                    feature_column=column_name,
                    turbine_ids=turbine_ids,
                    expected_timestamps=timestamps,
                ),
                column_name=column_name,
            )
        )
        feature_names.append(column_name)
    channels.append(
        _fill_non_finite(
            _build_feature_panel(
                context.series.with_columns(pl.col("is_observed").cast(pl.Float32)),
                feature_column="is_observed",
                turbine_ids=turbine_ids,
                expected_timestamps=timestamps,
            )
        )
    )
    feature_names.append("is_observed")
    channels.append(
        _fill_non_finite(
            _build_boolean_panel(
                context.series,
                alias="__row_bad",
                expression=pl.col("quality_flags").fill_null("").ne("").cast(pl.Float32),
                turbine_ids=turbine_ids,
                expected_timestamps=timestamps,
            )
        )
    )
    feature_names.append("row_bad")
    channels.append(
        _fill_non_finite(
            _build_boolean_panel(
                context.series,
                alias="__feat_bad",
                expression=pl.col("feature_quality_flags").fill_null("").ne("").cast(pl.Float32),
                turbine_ids=turbine_ids,
                expected_timestamps=timestamps,
            )
        )
    )
    feature_names.append("feat_bad")
    return np.stack(channels, axis=-1).astype(np.float32, copy=False), tuple(feature_names)


def _build_context_tensors(
    context: DatasetContext,
    *,
    history_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, tuple[str, ...]]:
    turbine_ids = context.metadata.turbine_ids
    timestamps = context.raw_timestamps
    total_rated_power_kw = context.metadata.rated_power_kw * len(turbine_ids)
    global_history_features: list[np.ndarray] = []
    global_feature_names: list[str] = []
    for column_name in context.global_observation_value_columns:
        global_history_features.append(
            _transform_observation_values(
                _build_global_feature_vector(
                    context.series,
                    feature_column=column_name,
                    turbine_ids=turbine_ids,
                    expected_timestamps=timestamps,
                ),
                column_name=column_name,
                history_mask=history_mask,
                total_rated_power_kw=total_rated_power_kw,
            )
        )
        global_feature_names.append(column_name)
    global_history_matrix = np.stack(global_history_features, axis=-1).astype(np.float32, copy=False)
    global_mask_features: list[np.ndarray] = []
    for column_name in context.global_observation_mask_columns:
        global_mask_features.append(
            _prepare_mask_channel(
                _build_global_feature_vector(
                    context.series,
                    feature_column=column_name,
                    turbine_ids=turbine_ids,
                    expected_timestamps=timestamps,
                ),
                column_name=column_name,
            )
        )
        global_feature_names.append(column_name)
    global_mask_matrix = np.stack(global_mask_features, axis=-1).astype(np.float32, copy=False)
    known_future_matrix = world_model_base.build_known_future_tensor(
        context.known_future,
        expected_timestamps=context.raw_timestamps,
        known_future_feature_columns=context.known_future_feature_columns,
    ).astype(np.float32, copy=False)
    context_feature_names = (
        *context.global_observation_value_columns,
        *context.global_observation_mask_columns,
        *context.known_future_feature_columns,
    )
    history_context = np.concatenate(
        [global_history_matrix, global_mask_matrix, known_future_matrix],
        axis=-1,
    ).astype(np.float32, copy=False)
    future_context = np.concatenate(
        [
            np.zeros_like(global_history_matrix, dtype=np.float32),
            np.ones_like(global_mask_matrix, dtype=np.float32),
            known_future_matrix,
        ],
        axis=-1,
    ).astype(np.float32, copy=False)
    return history_context, future_context, tuple(context_feature_names)


def _build_static_tensor(context: DatasetContext) -> np.ndarray:
    ordered = context.metadata.turbine_static.sort("turbine_index")
    available_columns = set(ordered.columns)
    missing_columns = [column for column in STATIC_FEATURE_NAMES if column not in available_columns]
    if missing_columns:
        raise ValueError(f"Task bundle static sidecar is missing required columns {missing_columns!r}.")
    frame = ordered.select(list(STATIC_FEATURE_NAMES))
    incomplete = [column for column in STATIC_FEATURE_NAMES if frame[column].null_count() > 0]
    if incomplete:
        raise ValueError(f"Task bundle static sidecar has null values in required columns {incomplete!r}.")
    return world_model_base._normalize_dense_feature_matrix(
        frame.to_numpy().astype(np.float32, copy=False),
        column_names=STATIC_FEATURE_NAMES,
        context_name="rollout static tensor",
    )


def _build_pairwise_tensor(context: DatasetContext) -> np.ndarray:
    turbine_ids = context.metadata.turbine_ids
    node_count = len(turbine_ids)
    turbine_index_by_id = {turbine_id: index for index, turbine_id in enumerate(turbine_ids)}
    pairwise_frame = context.pairwise.with_columns(
        pl.col("distance_m").cast(pl.Float64).log1p().alias("log_distance_m"),
        (pl.col("bearing_deg").cast(pl.Float64) * pl.lit(math.pi / 180.0)).sin().alias("bearing_sin"),
        (pl.col("bearing_deg").cast(pl.Float64) * pl.lit(math.pi / 180.0)).cos().alias("bearing_cos"),
    )
    feature_frame = pairwise_frame.select(list(PAIRWISE_FEATURE_NAMES))
    incomplete = [column for column in PAIRWISE_FEATURE_NAMES if feature_frame[column].null_count() > 0]
    if incomplete:
        raise ValueError(f"Task bundle pairwise sidecar has null values in required columns {incomplete!r}.")
    raw_features = feature_frame.to_numpy().astype(np.float32, copy=False)
    normalized_features = raw_features.copy()
    for column_index, column_name in enumerate(PAIRWISE_FEATURE_NAMES):
        if column_name in _PAIRWISE_RAW_COLUMNS:
            continue
        normalized_features[:, column_index] = world_model_base._normalize_dense_feature_matrix(
            raw_features[:, [column_index]],
            column_names=(column_name,),
            context_name="rollout pairwise tensor",
        )[:, 0]
    tensor = np.zeros((node_count, node_count, len(PAIRWISE_FEATURE_NAMES)), dtype=np.float32)
    coverage: set[tuple[int, int]] = set()
    for row_index, row in enumerate(pairwise_frame.iter_rows(named=True)):
        src_turbine_id = str(row["src_turbine_id"])
        dst_turbine_id = str(row["dst_turbine_id"])
        src_index = int(row["src_turbine_index"])
        dst_index = int(row["dst_turbine_index"])
        expected_src_index = turbine_index_by_id.get(src_turbine_id)
        expected_dst_index = turbine_index_by_id.get(dst_turbine_id)
        if expected_src_index is None or expected_dst_index is None:
            raise ValueError("Pairwise sidecar references turbine ids that are not present in the task-local order.")
        if src_index != expected_src_index or dst_index != expected_dst_index:
            raise ValueError(
                "Pairwise sidecar src/dst turbine ids do not match the declared src/dst turbine indices."
            )
        if src_index == dst_index:
            raise ValueError("Pairwise sidecar must not contain self edges.")
        if (dst_index, src_index) in coverage:
            raise ValueError("Pairwise sidecar contains duplicate directed turbine pairs.")
        tensor[dst_index, src_index, :] = normalized_features[row_index]
        coverage.add((dst_index, src_index))
    expected_coverage = {
        (dst_index, src_index)
        for src_index in range(node_count)
        for dst_index in range(node_count)
        if src_index != dst_index
    }
    if coverage != expected_coverage:
        missing = sorted(expected_coverage.difference(coverage))
        raise ValueError(f"Pairwise sidecar does not cover all directed turbine pairs; missing {missing[:5]!r}.")
    return tensor


def iter_evaluation_specs(
    prepared_dataset: PreparedDataset,
) -> tuple[tuple[str, str, world_model_base.FarmWindowDescriptorIndex], ...]:
    return (
        ("val", ROLLING_EVAL_PROTOCOL, prepared_dataset.val_rolling_windows),
        ("val", NON_OVERLAP_EVAL_PROTOCOL, prepared_dataset.val_non_overlap_windows),
        ("test", ROLLING_EVAL_PROTOCOL, prepared_dataset.test_rolling_windows),
        ("test", NON_OVERLAP_EVAL_PROTOCOL, prepared_dataset.test_non_overlap_windows),
    )


def prepare_dataset(
    dataset_id: str,
    *,
    variant_spec: ExperimentVariant | None = None,
    cache_root: str | Path = _CACHE_ROOT,
    max_train_origins: int | None = None,
    max_eval_origins: int | None = None,
) -> PreparedDataset:
    resolved_variant = variant_spec or VARIANT_SPECS[0]
    context = _build_variant_context(dataset_id, variant_spec=resolved_variant, cache_root=cache_root)
    split_frames = world_model_base.split_farm_window_index(
        context.window_index,
        raw_timestamps=context.raw_timestamps,
        resolution_minutes=context.resolution_minutes,
    )
    target_valid_mask = world_model_base.build_target_valid_mask(context.target_pu)
    full_train_windows = world_model_base.filter_windows_with_valid_targets(
        world_model_base.build_window_descriptor_index(split_frames["train"], timestamps_us=context.timestamps_us),
        target_valid_mask=target_valid_mask,
        forecast_steps=FORECAST_STEPS,
    )
    val_rolling_windows = world_model_base.filter_windows_with_valid_targets(
        world_model_base.build_window_descriptor_index(split_frames["val"], timestamps_us=context.timestamps_us),
        target_valid_mask=target_valid_mask,
        forecast_steps=FORECAST_STEPS,
    )
    test_rolling_windows = world_model_base.filter_windows_with_valid_targets(
        world_model_base.build_window_descriptor_index(split_frames["test"], timestamps_us=context.timestamps_us),
        target_valid_mask=target_valid_mask,
        forecast_steps=FORECAST_STEPS,
    )
    train_windows = world_model_base.limit_windows(full_train_windows, max_windows=max_train_origins)
    val_rolling_windows = world_model_base.limit_windows(val_rolling_windows, max_windows=max_eval_origins)
    test_rolling_windows = world_model_base.limit_windows(test_rolling_windows, max_windows=max_eval_origins)
    if len(full_train_windows) == 0:
        raise ValueError(f"Dataset {dataset_id!r} has no train windows with at least one valid target.")
    if len(val_rolling_windows) == 0:
        raise ValueError(f"Dataset {dataset_id!r} has no val windows with at least one valid target.")
    if len(test_rolling_windows) == 0:
        raise ValueError(f"Dataset {dataset_id!r} has no test windows with at least one valid target.")
    val_non_overlap_windows = world_model_base.thin_non_overlap_window_index(
        val_rolling_windows,
        forecast_steps=FORECAST_STEPS,
    )
    test_non_overlap_windows = world_model_base.thin_non_overlap_window_index(
        test_rolling_windows,
        forecast_steps=FORECAST_STEPS,
    )
    history_mask = _history_row_mask(
        target_indices=full_train_windows.target_indices,
        history_steps=HISTORY_STEPS,
        total_steps=context.target_pu.shape[0],
    )
    local_history_tensor, local_input_feature_names = _build_local_history_tensor(
        context,
        history_mask=history_mask,
    )
    context_history_tensor, context_future_tensor, context_feature_names = _build_context_tensors(
        context,
        history_mask=history_mask,
    )
    static_tensor = _build_static_tensor(context)
    pairwise_tensor = _build_pairwise_tensor(context)
    target_pu_filled = np.where(target_valid_mask, context.target_pu, 0.0).astype(np.float32, copy=False)
    _profile_log(
        dataset_id,
        "prepare_dataset_complete",
        coordinate_mode=context.coordinate_mode,
        feature_protocol_id=context.feature_protocol_id,
        model_variant=context.model_variant,
        node_count=len(context.metadata.turbine_ids),
        local_input_channels=local_history_tensor.shape[2],
        context_channels=context_history_tensor.shape[1],
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
        dataset_id=dataset_id,
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
        local_input_feature_names=local_input_feature_names,
        context_feature_names=context_feature_names,
        static_feature_names=STATIC_FEATURE_NAMES,
        pairwise_feature_names=PAIRWISE_FEATURE_NAMES,
        local_history_tensor=local_history_tensor,
        context_history_tensor=context_history_tensor,
        context_future_tensor=context_future_tensor,
        static_tensor=static_tensor,
        pairwise_tensor=pairwise_tensor,
        target_pu=context.target_pu,
        target_pu_filled=target_pu_filled,
        target_valid_mask=target_valid_mask,
        train_windows=train_windows,
        val_rolling_windows=val_rolling_windows,
        val_non_overlap_windows=val_non_overlap_windows,
        test_rolling_windows=test_rolling_windows,
        test_non_overlap_windows=test_non_overlap_windows,
    )


if Dataset is not None:

    class PanelWindowDataset(Dataset):
        def __init__(
            self,
            local_history_tensor: np.ndarray,
            context_history_tensor: np.ndarray,
            context_future_tensor: np.ndarray,
            target_pu_filled: np.ndarray,
            target_valid_mask: np.ndarray,
            windows: world_model_base.FarmWindowDescriptorIndex,
            *,
            history_steps: int,
            forecast_steps: int,
        ) -> None:
            self.local_history_tensor = np.asarray(local_history_tensor, dtype=np.float32)
            self.context_history_tensor = np.asarray(context_history_tensor, dtype=np.float32)
            self.context_future_tensor = np.asarray(context_future_tensor, dtype=np.float32)
            self.target_pu_filled = np.asarray(target_pu_filled, dtype=np.float32)
            self.target_valid_mask = np.asarray(target_valid_mask, dtype=np.float32)
            self.windows = windows
            self.history_steps = history_steps
            self.forecast_steps = forecast_steps

        def __len__(self) -> int:
            return len(self.windows)

        def __getitem__(self, index: int):
            target_index = int(self.windows.target_indices[index])
            local_history = self.local_history_tensor[target_index - self.history_steps : target_index]
            context_history = self.context_history_tensor[target_index - self.history_steps : target_index]
            context_future = self.context_future_tensor[target_index : target_index + self.forecast_steps]
            targets = self.target_pu_filled[target_index : target_index + self.forecast_steps]
            target_valid_mask = self.target_valid_mask[target_index : target_index + self.forecast_steps]
            if local_history.shape != (
                self.history_steps,
                self.local_history_tensor.shape[1],
                self.local_history_tensor.shape[2],
            ):
                raise ValueError(f"Local-history slice for index {index} has unexpected shape {local_history.shape!r}.")
            if context_history.shape != (self.history_steps, self.context_history_tensor.shape[1]):
                raise ValueError(
                    f"Context-history slice for index {index} has unexpected shape {context_history.shape!r}."
                )
            if context_future.shape != (self.forecast_steps, self.context_future_tensor.shape[1]):
                raise ValueError(
                    f"Context-future slice for index {index} has unexpected shape {context_future.shape!r}."
                )
            if targets.shape != (self.forecast_steps, self.target_pu_filled.shape[1]):
                raise ValueError(f"Target slice for index {index} has unexpected shape {targets.shape!r}.")
            if target_valid_mask.shape != (self.forecast_steps, self.target_valid_mask.shape[1]):
                raise ValueError(
                    f"Target-valid-mask slice for index {index} has unexpected shape {target_valid_mask.shape!r}."
                )
            return (
                local_history.astype(np.float32, copy=True),
                context_history.astype(np.float32, copy=True),
                context_future.astype(np.float32, copy=True),
                targets[:, :, None].astype(np.float32, copy=True),
                target_valid_mask[:, :, None].astype(np.float32, copy=True),
            )

else:

    class PanelWindowDataset:  # pragma: no cover - exercised only when torch is missing
        def __init__(self, *_args, **_kwargs) -> None:
            _require_torch()


def _set_random_seed(seed: int) -> None:
    world_model_base._set_random_seed(seed)


if nn is not None and F is not None:

    class FeedForward(nn.Module):
        def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, *, dropout: float = 0.0) -> None:
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.SiLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, output_dim),
            )

        def forward(self, inputs):
            return self.net(inputs)


    class EdgeNet(nn.Module):
        def __init__(
            self,
            *,
            static_dim: int,
            pairwise_dim: int,
            context_dim: int,
            global_state_dim: int,
            hidden_dim: int,
            dropout: float,
        ) -> None:
            super().__init__()
            self.edge_hidden = FeedForward(
                (static_dim * 2) + pairwise_dim + context_dim + global_state_dim,
                hidden_dim,
                hidden_dim,
                dropout=dropout,
            )
            self.attention = nn.Linear(hidden_dim, 1)

        def forward(self, static_features, pairwise_features, context_features, global_state):
            batch_size, node_count, static_dim = static_features.shape
            context_dim = context_features.shape[1]
            global_dim = global_state.shape[1]
            dst_static = static_features[:, :, None, :].expand(batch_size, node_count, node_count, static_dim)
            src_static = static_features[:, None, :, :].expand(batch_size, node_count, node_count, static_dim)
            pairwise = pairwise_features[None, :, :, :].expand(batch_size, -1, -1, -1)
            context = context_features[:, None, None, :].expand(batch_size, node_count, node_count, context_dim)
            global_context = global_state[:, None, None, :].expand(batch_size, node_count, node_count, global_dim)
            edge_inputs = torch.cat((dst_static, src_static, pairwise, context, global_context), dim=-1)
            edge_hidden = self.edge_hidden(edge_inputs)
            attention_logits = self.attention(edge_hidden).squeeze(-1)
            diagonal_mask = torch.eye(node_count, dtype=torch.bool, device=attention_logits.device)[None, :, :]
            attention_logits = attention_logits.masked_fill(diagonal_mask, float("-inf"))
            attention_weights = torch.softmax(attention_logits, dim=-1)
            attention_weights = attention_weights.masked_fill(diagonal_mask, 0.0)
            return attention_weights, edge_hidden


    class WorldModelRollout(nn.Module):
        def __init__(
            self,
            *,
            node_count: int,
            local_input_channels: int,
            context_channels: int,
            static_tensor: np.ndarray,
            pairwise_tensor: np.ndarray,
            local_observation_value_count: int,
            node_state_dim: int,
            hidden_state_dim: int,
            latent_state_dim: int,
            global_state_dim: int,
            message_dim: int,
            edge_hidden_dim: int,
            tau_embed_dim: int,
            forecast_steps: int,
            dropout: float,
        ) -> None:
            super().__init__()
            if node_state_dim != hidden_state_dim + latent_state_dim:
                raise ValueError("node_state_dim must equal hidden_state_dim + latent_state_dim.")
            self.node_count = node_count
            self.local_input_channels = local_input_channels
            self.context_channels = context_channels
            self.local_observation_value_count = local_observation_value_count
            self.node_state_dim = node_state_dim
            self.hidden_state_dim = hidden_state_dim
            self.latent_state_dim = latent_state_dim
            self.global_state_dim = global_state_dim
            self.message_dim = message_dim
            self.edge_hidden_dim = edge_hidden_dim
            self.forecast_steps = forecast_steps
            self.local_mask_start = 2 + local_observation_value_count
            static_features = np.asarray(static_tensor, dtype=np.float32)
            pairwise_features = np.asarray(pairwise_tensor, dtype=np.float32)
            self.register_buffer("static_features", torch.from_numpy(static_features))
            self.register_buffer("pairwise_features", torch.from_numpy(pairwise_features))
            self.obs_encoder = FeedForward(
                local_input_channels + static_features.shape[1],
                message_dim,
                message_dim,
                dropout=dropout,
            )
            self.edge_net = EdgeNet(
                static_dim=static_features.shape[1],
                pairwise_dim=pairwise_features.shape[2],
                context_dim=context_channels,
                global_state_dim=global_state_dim,
                hidden_dim=edge_hidden_dim,
                dropout=dropout,
            )
            self.obs_message = FeedForward(
                edge_hidden_dim + node_state_dim,
                message_dim,
                message_dim,
                dropout=dropout,
            )
            self.pred_message = FeedForward(
                edge_hidden_dim + node_state_dim,
                message_dim,
                message_dim,
                dropout=dropout,
            )
            self.obs_fusion = FeedForward(message_dim * 2, message_dim, message_dim, dropout=dropout)
            self.node_update = nn.GRUCell(message_dim + global_state_dim, node_state_dim)
            self.global_update = nn.GRUCell(node_state_dim + context_channels, global_state_dim)
            self.node_transition = nn.GRUCell(message_dim + context_channels + global_state_dim, node_state_dim)
            self.global_transition = nn.GRUCell(node_state_dim + context_channels, global_state_dim)
            self.tau_embedding = nn.Embedding(forecast_steps, tau_embed_dim)
            self.decoder = FeedForward(
                node_state_dim + global_state_dim + static_features.shape[1] + tau_embed_dim,
                node_state_dim,
                1,
                dropout=dropout,
            )
            self.aux_turbine_head = FeedForward(
                node_state_dim + global_state_dim + static_features.shape[1] + tau_embed_dim,
                node_state_dim,
                1,
                dropout=dropout,
            )
            self.aux_site_head = FeedForward(
                global_state_dim + tau_embed_dim,
                global_state_dim,
                1,
                dropout=dropout,
            )

        def _static_batch(self, batch_size: int):
            return self.static_features[None, :, :].expand(batch_size, -1, -1)

        def _pool_states(self, node_states, *, node_mask=None):
            if node_mask is None:
                return node_states.mean(dim=1)
            weights = node_mask.to(device=node_states.device, dtype=node_states.dtype)
            denom = weights.sum(dim=1, keepdim=True)
            weighted = (node_states * weights.unsqueeze(-1)).sum(dim=1)
            pooled = weighted / denom.clamp_min(1.0)
            fallback = node_states.mean(dim=1)
            return torch.where((denom > 0).expand_as(pooled), pooled, fallback)

        def _edge_context(self, batch_size: int, context_features, global_state):
            return self.edge_net(self._static_batch(batch_size), self.pairwise_features, context_features, global_state)

        def _has_local_observation(self, local_observations):
            target_available = local_observations[:, :, _OBS_TARGET_MASK_INDEX] < 0.5
            local_masks = local_observations[
                :,
                :,
                self.local_mask_start : self.local_mask_start + self.local_observation_value_count,
            ]
            local_available = (local_masks < 0.5).any(dim=-1)
            return (target_available | local_available).unsqueeze(-1)

        def _decode_heads(self, node_state, global_state, tau_index: int):
            batch_size = node_state.shape[0]
            tau_indices = torch.full(
                (batch_size,),
                int(tau_index),
                device=node_state.device,
                dtype=torch.long,
            )
            tau_embedding = self.tau_embedding(tau_indices)
            tau_nodes = tau_embedding[:, None, :].expand(-1, self.node_count, -1)
            global_nodes = global_state[:, None, :].expand(-1, self.node_count, -1)
            static_nodes = self._static_batch(batch_size)
            decode_inputs = torch.cat((node_state, global_nodes, static_nodes, tau_nodes), dim=-1)
            turbine_prediction = self.decoder(decode_inputs)
            aux_turbine = self.aux_turbine_head(decode_inputs)
            aux_site = self.aux_site_head(torch.cat((global_state, tau_embedding), dim=-1))
            return turbine_prediction, aux_turbine, aux_site[:, None, :]

        def correct_step(self, node_state_prior, global_state_prior, local_observations, context_features):
            batch_size = local_observations.shape[0]
            static_nodes = self._static_batch(batch_size)
            obs_encoding = self.obs_encoder(torch.cat((local_observations, static_nodes), dim=-1))
            attention_weights, edge_hidden = self._edge_context(
                batch_size,
                context_features,
                global_state_prior,
            )
            src_node_states = node_state_prior[:, None, :, :].expand(-1, self.node_count, -1, -1)
            obs_messages = self.obs_message(torch.cat((edge_hidden, src_node_states), dim=-1))
            obs_aggregate = (attention_weights.unsqueeze(-1) * obs_messages).sum(dim=2)
            fused = self.obs_fusion(torch.cat((obs_encoding, obs_aggregate), dim=-1))
            global_nodes = global_state_prior[:, None, :].expand(-1, self.node_count, -1)
            update_inputs = torch.cat((fused, global_nodes), dim=-1)
            updated = self.node_update(
                update_inputs.reshape(batch_size * self.node_count, -1),
                node_state_prior.reshape(batch_size * self.node_count, -1),
            ).reshape(batch_size, self.node_count, self.node_state_dim)
            local_update_mask = self._has_local_observation(local_observations)
            node_state_post = torch.where(local_update_mask, updated, node_state_prior)
            pooled = self._pool_states(node_state_post, node_mask=local_update_mask.squeeze(-1))
            global_input = torch.cat((pooled, context_features), dim=-1)
            global_state_post = self.global_update(global_input, global_state_prior)
            return node_state_post, global_state_post

        def transition_step(self, node_state, global_state, context_features):
            batch_size = node_state.shape[0]
            attention_weights, edge_hidden = self._edge_context(batch_size, context_features, global_state)
            src_node_states = node_state[:, None, :, :].expand(-1, self.node_count, -1, -1)
            pred_messages = self.pred_message(torch.cat((edge_hidden, src_node_states), dim=-1))
            pred_aggregate = (attention_weights.unsqueeze(-1) * pred_messages).sum(dim=2)
            global_nodes = global_state[:, None, :].expand(-1, self.node_count, -1)
            context_nodes = context_features[:, None, :].expand(-1, self.node_count, -1)
            transition_inputs = torch.cat((pred_aggregate, context_nodes, global_nodes), dim=-1)
            next_node_state = self.node_transition(
                transition_inputs.reshape(batch_size * self.node_count, -1),
                node_state.reshape(batch_size * self.node_count, -1),
            ).reshape(batch_size, self.node_count, self.node_state_dim)
            pooled = self._pool_states(next_node_state)
            global_input = torch.cat((pooled, context_features), dim=-1)
            next_global_state = self.global_transition(global_input, global_state)
            return next_node_state, next_global_state

        def forward(self, local_history, context_history, context_future):
            if local_history.ndim != 4:
                raise ValueError(
                    f"Expected local_history with shape [batch, history, nodes, channels], got {local_history.shape!r}."
                )
            if context_history.ndim != 3:
                raise ValueError(
                    f"Expected context_history with shape [batch, history, channels], got {context_history.shape!r}."
                )
            if context_future.ndim != 3:
                raise ValueError(
                    f"Expected context_future with shape [batch, horizon, channels], got {context_future.shape!r}."
                )
            batch_size = local_history.shape[0]
            node_state_prior = local_history.new_zeros((batch_size, self.node_count, self.node_state_dim))
            global_state_prior = local_history.new_zeros((batch_size, self.global_state_dim))
            node_state_post, global_state_post = self.correct_step(
                node_state_prior,
                global_state_prior,
                local_history[:, 0, :, :],
                context_history[:, 0, :],
            )
            hist_prior_predictions: list[Any] = []
            for history_index in range(1, local_history.shape[1]):
                node_state_prior, global_state_prior = self.transition_step(
                    node_state_post,
                    global_state_post,
                    context_history[:, history_index, :],
                )
                prior_prediction, _aux_turbine, _aux_site = self._decode_heads(
                    node_state_prior,
                    global_state_prior,
                    0,
                )
                hist_prior_predictions.append(prior_prediction)
                node_state_post, global_state_post = self.correct_step(
                    node_state_prior,
                    global_state_prior,
                    local_history[:, history_index, :, :],
                    context_history[:, history_index, :],
                )
            future_predictions: list[Any] = []
            aux_turbine_predictions: list[Any] = []
            aux_site_predictions: list[Any] = []
            rollout_node_state = node_state_post
            rollout_global_state = global_state_post
            for horizon_index in range(context_future.shape[1]):
                rollout_node_state, rollout_global_state = self.transition_step(
                    rollout_node_state,
                    rollout_global_state,
                    context_future[:, horizon_index, :],
                )
                future_prediction, aux_turbine_prediction, aux_site_prediction = self._decode_heads(
                    rollout_node_state,
                    rollout_global_state,
                    horizon_index,
                )
                future_predictions.append(future_prediction)
                aux_turbine_predictions.append(aux_turbine_prediction)
                aux_site_predictions.append(aux_site_prediction)
            return ModelOutputs(
                future_predictions=torch.stack(future_predictions, dim=1),
                hist_prior_predictions=torch.stack(hist_prior_predictions, dim=1),
                aux_turbine_predictions=torch.stack(aux_turbine_predictions, dim=1),
                aux_site_predictions=torch.stack(aux_site_predictions, dim=1),
            )

else:

    class FeedForward:  # pragma: no cover - exercised only when torch is missing
        def __init__(self, *_args, **_kwargs) -> None:
            _require_torch()


    class EdgeNet:  # pragma: no cover - exercised only when torch is missing
        def __init__(self, *_args, **_kwargs) -> None:
            _require_torch()


    class WorldModelRollout:  # pragma: no cover - exercised only when torch is missing
        def __init__(self, *_args, **_kwargs) -> None:
            _require_torch()


def build_model(
    *,
    node_count: int,
    local_input_channels: int,
    context_channels: int,
    static_tensor: np.ndarray,
    pairwise_tensor: np.ndarray,
    local_observation_value_count: int,
    node_state_dim: int,
    hidden_state_dim: int,
    latent_state_dim: int,
    global_state_dim: int,
    message_dim: int,
    edge_hidden_dim: int,
    tau_embed_dim: int,
    forecast_steps: int,
    dropout: float,
):
    _require_torch()
    return WorldModelRollout(
        node_count=node_count,
        local_input_channels=local_input_channels,
        context_channels=context_channels,
        static_tensor=static_tensor,
        pairwise_tensor=pairwise_tensor,
        local_observation_value_count=local_observation_value_count,
        node_state_dim=node_state_dim,
        hidden_state_dim=hidden_state_dim,
        latent_state_dim=latent_state_dim,
        global_state_dim=global_state_dim,
        message_dim=message_dim,
        edge_hidden_dim=edge_hidden_dim,
        tau_embed_dim=tau_embed_dim,
        forecast_steps=forecast_steps,
        dropout=dropout,
    )


def initialize_model_parameters(model) -> None:
    _, resolved_nn, _, _, _ = _require_torch()
    for parameter in model.parameters():
        if parameter.dim() > 1:
            resolved_nn.init.xavier_uniform_(parameter)
        else:
            resolved_nn.init.uniform_(parameter, -0.02, 0.02)


def _build_dataloader(
    prepared_dataset: PreparedDataset,
    *,
    windows: world_model_base.FarmWindowDescriptorIndex,
    batch_size: int,
    device: str,
    shuffle: bool,
    seed: int,
    num_workers: int | None = None,
):
    resolved_torch, _, _, resolved_loader, _ = _require_torch()
    resolved_device = resolve_device(device)
    resolved_num_workers = resolve_loader_num_workers(device=resolved_device, num_workers=num_workers)
    dataset = PanelWindowDataset(
        prepared_dataset.local_history_tensor,
        prepared_dataset.context_history_tensor,
        prepared_dataset.context_future_tensor,
        prepared_dataset.target_pu_filled,
        prepared_dataset.target_valid_mask,
        windows,
        history_steps=prepared_dataset.history_steps,
        forecast_steps=prepared_dataset.forecast_steps,
    )
    generator = resolved_torch.Generator()
    generator.manual_seed(seed)
    loader_kwargs: dict[str, object] = {
        "dataset": dataset,
        "batch_size": batch_size,
        "shuffle": shuffle,
        "generator": generator if shuffle else None,
        "pin_memory": resolved_device == "cuda",
        "num_workers": resolved_num_workers,
    }
    if resolved_num_workers > 0:
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = DEFAULT_CUDA_PREFETCH_FACTOR
    return resolved_loader(**loader_kwargs)


def _safe_divide(numerator: float, denominator: int) -> float:
    if denominator <= 0:
        return float("nan")
    return numerator / denominator


def _safe_rmse(squared_error_sum: float, denominator: int) -> float:
    if denominator <= 0:
        return float("nan")
    return math.sqrt(squared_error_sum / denominator)


def masked_huber_loss(predictions, targets, valid_mask, *, torch_module, allow_zero: bool = False):
    resolved_mask = valid_mask.to(device=predictions.device, dtype=predictions.dtype)
    valid_count = resolved_mask.sum()
    if float(valid_count.item()) <= 0:
        if allow_zero:
            return predictions.new_tensor(0.0)
        raise ValueError("masked_huber_loss received a batch with zero valid targets.")
    errors = predictions - targets
    abs_errors = torch_module.abs(errors)
    huber = torch_module.where(
        abs_errors <= 1.0,
        0.5 * torch_module.square(errors),
        abs_errors - 0.5,
    )
    return (huber * resolved_mask).sum() / valid_count


def _compute_training_losses(
    outputs: ModelOutputs,
    batch_local_history,
    batch_targets,
    batch_target_valid_mask,
    *,
    hist_loss_weight: float,
    aux_turbine_loss_weight: float,
    aux_site_loss_weight: float,
    consistency_loss_weight: float,
    torch_module,
) -> tuple[Any, dict[str, float]]:
    future_loss = masked_huber_loss(
        outputs.future_predictions,
        batch_targets,
        batch_target_valid_mask,
        torch_module=torch_module,
    )
    history_targets = batch_local_history[:, 1:, :, _OBS_TARGET_VALUE_INDEX : _OBS_TARGET_VALUE_INDEX + 1]
    history_valid_mask = 1.0 - batch_local_history[:, 1:, :, _OBS_TARGET_MASK_INDEX : _OBS_TARGET_MASK_INDEX + 1]
    hist1_loss = masked_huber_loss(
        outputs.hist_prior_predictions,
        history_targets,
        history_valid_mask,
        torch_module=torch_module,
        allow_zero=True,
    )
    aux_turbine_loss = masked_huber_loss(
        outputs.aux_turbine_predictions,
        batch_targets,
        batch_target_valid_mask,
        torch_module=torch_module,
        allow_zero=True,
    )
    site_target = (batch_targets * batch_target_valid_mask).sum(dim=2)
    site_valid_mask = (batch_target_valid_mask.sum(dim=(2, 3)) > 0).to(batch_targets.dtype).unsqueeze(-1)
    aux_site_loss = masked_huber_loss(
        outputs.aux_site_predictions.squeeze(dim=2),
        site_target,
        site_valid_mask,
        torch_module=torch_module,
        allow_zero=True,
    )
    decode_site_prediction = outputs.future_predictions.sum(dim=2)
    consistency_loss = masked_huber_loss(
        outputs.aux_site_predictions.squeeze(dim=2),
        decode_site_prediction,
        site_valid_mask,
        torch_module=torch_module,
        allow_zero=True,
    )
    total_loss = (
        future_loss
        + (hist_loss_weight * hist1_loss)
        + (aux_turbine_loss_weight * aux_turbine_loss)
        + (aux_site_loss_weight * aux_site_loss)
        + (consistency_loss_weight * consistency_loss)
    )
    return total_loss, {
        "future": float(future_loss.item()),
        "hist1": float(hist1_loss.item()),
        "aux_turbine": float(aux_turbine_loss.item()),
        "aux_site": float(aux_site_loss.item()),
        "consistency": float(consistency_loss.item()),
        "total": float(total_loss.item()),
    }


def evaluate_model(
    model,
    loader,
    *,
    device: str,
    rated_power_kw: float,
    forecast_steps: int,
    progress_label: str | None = None,
) -> EvaluationMetrics:
    resolved_torch, _, _, _, _ = _require_torch()
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
            for batch_local_history, batch_context_history, batch_context_future, batch_targets, batch_target_valid_mask in loader:
                batch_local_history = batch_local_history.to(
                    device=device,
                    dtype=resolved_torch.float32,
                    non_blocking=device == "cuda",
                )
                batch_context_history = batch_context_history.to(
                    device=device,
                    dtype=resolved_torch.float32,
                    non_blocking=device == "cuda",
                )
                batch_context_future = batch_context_future.to(
                    device=device,
                    dtype=resolved_torch.float32,
                    non_blocking=device == "cuda",
                )
                batch_targets = batch_targets.to(
                    device=device,
                    dtype=resolved_torch.float32,
                    non_blocking=device == "cuda",
                )
                batch_target_valid_mask = batch_target_valid_mask.to(
                    device=device,
                    dtype=resolved_torch.float32,
                    non_blocking=device == "cuda",
                )
                outputs = model(batch_local_history, batch_context_history, batch_context_future)
                predictions = outputs.future_predictions
                errors_pu = predictions - batch_targets
                errors_kw = errors_pu * rated_power_kw
                valid_mask = batch_target_valid_mask
                valid_mask_np = valid_mask.detach().cpu().numpy().astype(np.float64, copy=False)
                batch_window_count = int(batch_local_history.shape[0])
                batch_prediction_count = int(valid_mask.sum().item())
                window_count += batch_window_count
                prediction_count += batch_prediction_count
                abs_errors_kw = resolved_torch.abs(errors_kw) * valid_mask
                squared_errors_kw = resolved_torch.square(errors_kw) * valid_mask
                abs_errors_pu = resolved_torch.abs(errors_pu) * valid_mask
                squared_errors_pu = resolved_torch.square(errors_pu) * valid_mask
                abs_error_sum_kw += float(abs_errors_kw.sum().item())
                squared_error_sum_kw += float(squared_errors_kw.sum().item())
                abs_error_sum_pu += float(abs_errors_pu.sum().item())
                squared_error_sum_pu += float(squared_errors_pu.sum().item())
                horizon_window_count += batch_window_count
                horizon_prediction_count += valid_mask_np.sum(axis=(0, 2, 3)).astype(np.int64, copy=False)
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
    eval_batch_size: int | None = None,
    learning_rate: float = DEFAULT_LEARNING_RATE,
    max_epochs: int = DEFAULT_MAX_EPOCHS,
    early_stopping_patience: int = DEFAULT_EARLY_STOPPING_PATIENCE,
    node_state_dim: int = DEFAULT_NODE_STATE_DIM,
    hidden_state_dim: int = DEFAULT_HIDDEN_STATE_DIM,
    latent_state_dim: int = DEFAULT_LATENT_STATE_DIM,
    global_state_dim: int = DEFAULT_GLOBAL_STATE_DIM,
    message_dim: int = DEFAULT_MESSAGE_DIM,
    edge_hidden_dim: int = DEFAULT_EDGE_HIDDEN_DIM,
    tau_embed_dim: int = DEFAULT_TAU_EMBED_DIM,
    dropout: float = DEFAULT_DROPOUT,
    grad_clip_norm: float = DEFAULT_GRAD_CLIP_NORM,
    hist_loss_weight: float = DEFAULT_HIST_LOSS_WEIGHT,
    aux_turbine_loss_weight: float = DEFAULT_AUX_TURBINE_LOSS_WEIGHT,
    aux_site_loss_weight: float = DEFAULT_AUX_SITE_LOSS_WEIGHT,
    consistency_loss_weight: float = DEFAULT_CONSISTENCY_LOSS_WEIGHT,
    weight_decay: float = DEFAULT_WEIGHT_DECAY,
    num_workers: int | None = None,
    checkpoint_path: str | Path | None = None,
    training_history_path: str | Path | None = None,
    resume_from_checkpoint: bool = False,
    progress_label: str | None = None,
) -> TrainingOutcome:
    resolved_torch, _, _, _, _ = _require_torch()
    _set_random_seed(seed)
    resolved_device = resolve_device(device)
    world_model_base._configure_torch_runtime(device=resolved_device, torch_module=resolved_torch)
    resolved_eval_batch_size = resolve_eval_batch_size(
        batch_size,
        device=resolved_device,
        eval_batch_size=eval_batch_size,
    )
    model = build_model(
        node_count=prepared_dataset.node_count,
        local_input_channels=prepared_dataset.local_input_channels,
        context_channels=prepared_dataset.context_channels,
        static_tensor=prepared_dataset.static_tensor,
        pairwise_tensor=prepared_dataset.pairwise_tensor,
        local_observation_value_count=EXPECTED_LOCAL_OBSERVATION_VALUE_COUNT,
        node_state_dim=node_state_dim,
        hidden_state_dim=hidden_state_dim,
        latent_state_dim=latent_state_dim,
        global_state_dim=global_state_dim,
        message_dim=message_dim,
        edge_hidden_dim=edge_hidden_dim,
        tau_embed_dim=tau_embed_dim,
        forecast_steps=prepared_dataset.forecast_steps,
        dropout=dropout,
    ).to(device=resolved_device)
    initialize_model_parameters(model)
    optimizer = resolved_torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    val_loader = _build_dataloader(
        prepared_dataset,
        windows=prepared_dataset.val_rolling_windows,
        batch_size=resolved_eval_batch_size,
        device=resolved_device,
        shuffle=False,
        seed=seed,
        num_workers=num_workers,
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
            checkpoint_payload = world_model_base._torch_load_checkpoint(
                resolved_checkpoint_path,
                map_location=resolved_device,
            )
            if checkpoint_payload.get("schema_version") != _TRAINING_CHECKPOINT_SCHEMA_VERSION:
                raise RuntimeError(
                    f"Unsupported checkpoint schema at {resolved_checkpoint_path}: "
                    f"{checkpoint_payload.get('schema_version')!r}."
                )
            if checkpoint_payload.get("job") != checkpoint_job_identity:
                raise RuntimeError(
                    f"Checkpoint at {resolved_checkpoint_path} does not match "
                    f"{prepared_dataset.dataset_id}/{prepared_dataset.model_variant}."
                )
            model.load_state_dict(checkpoint_payload["model_state_dict"])
            optimizer.load_state_dict(checkpoint_payload["optimizer_state_dict"])
            best_state = checkpoint_payload.get("best_state_dict")
            best_epoch = int(checkpoint_payload["best_epoch"])
            best_val_rmse_pu = float(checkpoint_payload["best_val_rmse_pu"])
            epochs_without_improvement = int(checkpoint_payload["epochs_without_improvement"])
            epochs_ran = int(checkpoint_payload["epochs_ran"])
            start_epoch = int(checkpoint_payload["next_epoch"])
            if bool(checkpoint_payload.get("training_complete", False)):
                if best_state is None:
                    raise RuntimeError(
                        f"Checkpoint at {resolved_checkpoint_path} marks training complete but has no best state."
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
                device=resolved_device,
                shuffle=True,
                seed=seed + epoch_index,
                num_workers=num_workers,
            )
            batch_progress = _create_progress_bar(
                total=_loader_batch_total(train_loader),
                desc=f"{progress_label or prepared_dataset.dataset_id} train e{epoch_index}",
            )
            latest_losses: dict[str, float] = {}
            loss_sums = {
                "total": 0.0,
                "future": 0.0,
                "hist1": 0.0,
                "aux_turbine": 0.0,
                "aux_site": 0.0,
                "consistency": 0.0,
            }
            train_loss_weight = 0
            train_batch_count = 0
            try:
                for batch_local_history, batch_context_history, batch_context_future, batch_targets, batch_target_valid_mask in train_loader:
                    batch_local_history = batch_local_history.to(
                        device=resolved_device,
                        dtype=resolved_torch.float32,
                        non_blocking=resolved_device == "cuda",
                    )
                    batch_context_history = batch_context_history.to(
                        device=resolved_device,
                        dtype=resolved_torch.float32,
                        non_blocking=resolved_device == "cuda",
                    )
                    batch_context_future = batch_context_future.to(
                        device=resolved_device,
                        dtype=resolved_torch.float32,
                        non_blocking=resolved_device == "cuda",
                    )
                    batch_targets = batch_targets.to(
                        device=resolved_device,
                        dtype=resolved_torch.float32,
                        non_blocking=resolved_device == "cuda",
                    )
                    batch_target_valid_mask = batch_target_valid_mask.to(
                        device=resolved_device,
                        dtype=resolved_torch.float32,
                        non_blocking=resolved_device == "cuda",
                    )
                    optimizer.zero_grad(set_to_none=True)
                    outputs = model(batch_local_history, batch_context_history, batch_context_future)
                    loss, latest_losses = _compute_training_losses(
                        outputs,
                        batch_local_history,
                        batch_targets,
                        batch_target_valid_mask,
                        hist_loss_weight=hist_loss_weight,
                        aux_turbine_loss_weight=aux_turbine_loss_weight,
                        aux_site_loss_weight=aux_site_loss_weight,
                        consistency_loss_weight=consistency_loss_weight,
                        torch_module=resolved_torch,
                    )
                    loss.backward()
                    if grad_clip_norm > 0:
                        resolved_torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
                    optimizer.step()
                    batch_window_count = int(batch_local_history.shape[0])
                    for loss_name, loss_value in latest_losses.items():
                        loss_sums[loss_name] += float(loss_value) * batch_window_count
                    train_loss_weight += batch_window_count
                    train_batch_count += 1
                    batch_progress.set_postfix_str(f"loss={latest_losses['total']:.4f}")
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
            loss_means = {
                loss_name: loss_sum / train_loss_weight if train_loss_weight else math.nan
                for loss_name, loss_sum in loss_sums.items()
            }
            epoch_progress.update(1)
            postfix_parts = [f"val_rmse={val_rmse_pu:.4f}", f"best={best_val_rmse_pu:.4f}"]
            if latest_losses:
                postfix_parts.insert(0, f"loss={latest_losses['total']:.4f}")
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
                        "train_loss_mean": loss_means["total"],
                        "train_loss_last": latest_losses.get("total"),
                        "train_future_loss_mean": loss_means["future"],
                        "train_future_loss_last": latest_losses.get("future"),
                        "train_hist1_loss_mean": loss_means["hist1"],
                        "train_hist1_loss_last": latest_losses.get("hist1"),
                        "train_aux_turbine_loss_mean": loss_means["aux_turbine"],
                        "train_aux_turbine_loss_last": latest_losses.get("aux_turbine"),
                        "train_aux_site_loss_mean": loss_means["aux_site"],
                        "train_aux_site_loss_last": latest_losses.get("aux_site"),
                        "train_consistency_loss_mean": loss_means["consistency"],
                        "train_consistency_loss_last": latest_losses.get("consistency"),
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
                world_model_base._save_training_checkpoint(
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
    if best_state is None:
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


def _window_bounds(
    windows: world_model_base.FarmWindowDescriptorIndex,
) -> tuple[str | None, str | None]:
    if len(windows) == 0:
        return None, None
    return (
        _timestamp_us_to_string(int(windows.output_start_us.min())),
        _timestamp_us_to_string(int(windows.output_end_us.max())),
    )


def build_result_rows(
    prepared_dataset: PreparedDataset,
    *,
    training_outcome: TrainingOutcome,
    runtime_seconds: float,
    seed: int,
    profile: HyperparameterProfile,
    evaluation_results: Sequence[tuple[str, str, world_model_base.FarmWindowDescriptorIndex, EvaluationMetrics]],
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
        "coordinate_mode": prepared_dataset.coordinate_mode,
        "node_count": prepared_dataset.node_count,
        "local_input_channels": prepared_dataset.local_input_channels,
        "context_channels": prepared_dataset.context_channels,
        "static_feature_count": prepared_dataset.static_feature_count,
        "pairwise_feature_count": prepared_dataset.pairwise_feature_count,
        "node_state_dim": profile.node_state_dim,
        "hidden_state_dim": profile.hidden_state_dim,
        "latent_state_dim": profile.latent_state_dim,
        "global_state_dim": profile.global_state_dim,
        "message_dim": profile.message_dim,
        "edge_hidden_dim": profile.edge_hidden_dim,
        "tau_embed_dim": profile.tau_embed_dim,
        "dropout": profile.dropout,
        "grad_clip_norm": profile.grad_clip_norm,
        "device": training_outcome.device,
        "runtime_seconds": round(runtime_seconds, 6),
        "train_window_count": len(prepared_dataset.train_windows),
        "val_window_count": len(prepared_dataset.val_rolling_windows),
        "test_window_count": len(prepared_dataset.test_rolling_windows),
        "best_epoch": training_outcome.best_epoch,
        "epochs_ran": training_outcome.epochs_ran,
        "best_val_rmse_pu": training_outcome.best_val_rmse_pu,
        "seed": seed,
        "batch_size": profile.batch_size,
        "learning_rate": profile.learning_rate,
        "hist_loss_weight": profile.hist_loss_weight,
        "aux_turbine_loss_weight": profile.aux_turbine_loss_weight,
        "aux_site_loss_weight": profile.aux_site_loss_weight,
        "consistency_loss_weight": profile.consistency_loss_weight,
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
    eval_batch_size: int | None = None,
    learning_rate: float = DEFAULT_LEARNING_RATE,
    max_epochs: int = DEFAULT_MAX_EPOCHS,
    early_stopping_patience: int = DEFAULT_EARLY_STOPPING_PATIENCE,
    node_state_dim: int = DEFAULT_NODE_STATE_DIM,
    hidden_state_dim: int = DEFAULT_HIDDEN_STATE_DIM,
    latent_state_dim: int = DEFAULT_LATENT_STATE_DIM,
    global_state_dim: int = DEFAULT_GLOBAL_STATE_DIM,
    message_dim: int = DEFAULT_MESSAGE_DIM,
    edge_hidden_dim: int = DEFAULT_EDGE_HIDDEN_DIM,
    tau_embed_dim: int = DEFAULT_TAU_EMBED_DIM,
    dropout: float = DEFAULT_DROPOUT,
    grad_clip_norm: float = DEFAULT_GRAD_CLIP_NORM,
    hist_loss_weight: float = DEFAULT_HIST_LOSS_WEIGHT,
    aux_turbine_loss_weight: float = DEFAULT_AUX_TURBINE_LOSS_WEIGHT,
    aux_site_loss_weight: float = DEFAULT_AUX_SITE_LOSS_WEIGHT,
    consistency_loss_weight: float = DEFAULT_CONSISTENCY_LOSS_WEIGHT,
    weight_decay: float = DEFAULT_WEIGHT_DECAY,
    num_workers: int | None = None,
    checkpoint_path: str | Path | None = None,
    training_history_path: str | Path | None = None,
    resume_from_checkpoint: bool = False,
) -> list[dict[str, object]]:
    dataset_start = time.monotonic()
    progress_label = f"{prepared_dataset.dataset_id}/{prepared_dataset.model_variant}"
    resolved_profile = HyperparameterProfile(
        batch_size=batch_size,
        learning_rate=learning_rate,
        max_epochs=max_epochs,
        early_stopping_patience=early_stopping_patience,
        node_state_dim=node_state_dim,
        hidden_state_dim=hidden_state_dim,
        latent_state_dim=latent_state_dim,
        global_state_dim=global_state_dim,
        message_dim=message_dim,
        edge_hidden_dim=edge_hidden_dim,
        tau_embed_dim=tau_embed_dim,
        dropout=dropout,
        grad_clip_norm=grad_clip_norm,
        hist_loss_weight=hist_loss_weight,
        aux_turbine_loss_weight=aux_turbine_loss_weight,
        aux_site_loss_weight=aux_site_loss_weight,
        consistency_loss_weight=consistency_loss_weight,
        weight_decay=weight_decay,
    )
    resolved_device = resolve_device(device)
    resolved_eval_batch_size = resolve_eval_batch_size(
        batch_size,
        device=resolved_device,
        eval_batch_size=eval_batch_size,
    )
    training_outcome = train_model(
        prepared_dataset,
        device=resolved_device,
        seed=seed,
        batch_size=batch_size,
        eval_batch_size=resolved_eval_batch_size,
        learning_rate=learning_rate,
        max_epochs=max_epochs,
        early_stopping_patience=early_stopping_patience,
        node_state_dim=node_state_dim,
        hidden_state_dim=hidden_state_dim,
        latent_state_dim=latent_state_dim,
        global_state_dim=global_state_dim,
        message_dim=message_dim,
        edge_hidden_dim=edge_hidden_dim,
        tau_embed_dim=tau_embed_dim,
        dropout=dropout,
        grad_clip_norm=grad_clip_norm,
        hist_loss_weight=hist_loss_weight,
        aux_turbine_loss_weight=aux_turbine_loss_weight,
        aux_site_loss_weight=aux_site_loss_weight,
        consistency_loss_weight=consistency_loss_weight,
        weight_decay=weight_decay,
        num_workers=num_workers,
        checkpoint_path=checkpoint_path,
        training_history_path=training_history_path,
        resume_from_checkpoint=resume_from_checkpoint,
        progress_label=progress_label,
    )
    evaluation_results: list[tuple[str, str, world_model_base.FarmWindowDescriptorIndex, EvaluationMetrics]] = []
    eval_specs = iter_evaluation_specs(prepared_dataset)
    eval_progress = _create_progress_bar(total=len(eval_specs), desc=f"{progress_label} eval")
    try:
        for split_name, eval_protocol, windows in eval_specs:
            loader = _build_dataloader(
                prepared_dataset,
                windows=windows,
                batch_size=resolved_eval_batch_size,
                device=training_outcome.device,
                shuffle=False,
                seed=seed,
                num_workers=num_workers,
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
        local_input_channels=prepared_dataset.local_input_channels,
        model_variant=prepared_dataset.model_variant,
        test_rolling_rmse_pu=test_rolling_metrics.rmse_pu,
        runtime_seconds=round(runtime_seconds, 6),
    )
    return build_result_rows(
        prepared_dataset,
        training_outcome=training_outcome,
        runtime_seconds=runtime_seconds,
        seed=seed,
        profile=resolved_profile,
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
    eval_batch_size: int | None = None,
    learning_rate: float | None = None,
    early_stopping_patience: int | None = None,
    node_state_dim: int | None = None,
    hidden_state_dim: int | None = None,
    latent_state_dim: int | None = None,
    global_state_dim: int | None = None,
    message_dim: int | None = None,
    edge_hidden_dim: int | None = None,
    tau_embed_dim: int | None = None,
    dropout: float | None = None,
    grad_clip_norm: float | None = None,
    hist_loss_weight: float | None = None,
    aux_turbine_loss_weight: float | None = None,
    aux_site_loss_weight: float | None = None,
    consistency_loss_weight: float | None = None,
    weight_decay: float | None = None,
    num_workers: int | None = None,
    resume: bool = False,
    force_rerun: bool = False,
    work_root: str | Path = _RUN_WORK_ROOT,
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
        device=device,
        seed=seed,
        max_train_origins=max_train_origins,
        max_eval_origins=max_eval_origins,
        batch_size=batch_size,
        eval_batch_size=eval_batch_size,
        learning_rate=learning_rate,
        max_epochs=max_epochs,
        early_stopping_patience=early_stopping_patience,
        node_state_dim=node_state_dim,
        hidden_state_dim=hidden_state_dim,
        latent_state_dim=latent_state_dim,
        global_state_dim=global_state_dim,
        message_dim=message_dim,
        edge_hidden_dim=edge_hidden_dim,
        tau_embed_dim=tau_embed_dim,
        dropout=dropout,
        grad_clip_norm=grad_clip_norm,
        hist_loss_weight=hist_loss_weight,
        aux_turbine_loss_weight=aux_turbine_loss_weight,
        aux_site_loss_weight=aux_site_loss_weight,
        consistency_loss_weight=consistency_loss_weight,
        weight_decay=weight_decay,
        num_workers=num_workers,
    )
    existing_state = _load_resume_state_if_exists(resume_paths)
    if resume and force_rerun:
        raise ValueError("--resume and --force-rerun are mutually exclusive.")
    if force_rerun:
        _reset_resume_slot(resume_paths, effective_config=effective_config)
        existing_state = None
    elif resume:
        if existing_state is None:
            raise ValueError(
                f"No resume state exists for output path {output}. Expected {resume_paths.state_path}."
            )
        if existing_state.get("effective_config") != effective_config:
            raise ValueError(
                f"Resume state at {resume_paths.state_path} does not match the requested run configuration."
            )
    else:
        if existing_state is None:
            _reset_resume_slot(resume_paths, effective_config=effective_config)
        elif existing_state.get("status") == "complete":
            _reset_resume_slot(resume_paths, effective_config=effective_config)
            existing_state = None
        else:
            raise ValueError(
                f"Resume state at {resume_paths.state_path} is still marked running. "
                f"Re-run with --resume, --force-rerun, or remove {resume_paths.slot_dir}."
            )
    partial_results = _read_partial_results(resume_paths)
    rows: list[dict[str, object]] = partial_results.to_dicts()
    completed_job_keys = _completed_job_keys(partial_results)
    total_jobs = len(dataset_ids) * len(variant_specs)
    if len(completed_job_keys) == total_jobs:
        results = _result_frame_from_rows(rows)
        _atomic_write_csv(output, results)
        _publish_training_history(resume_paths, output)
        _clear_checkpoint_dir(resume_paths)
        _write_resume_state(resume_paths, status="complete", effective_config=effective_config, active_job=None)
        return results
    resume_active_job = None if existing_state is None else existing_state.get("active_job")
    job_progress = _create_progress_bar(total=total_jobs, desc="world_model_rollout_v1 jobs", leave=True)
    try:
        for dataset_id in dataset_ids:
            for variant_spec in variant_specs:
                current_job_key = _job_key(dataset_id, variant_spec.model_variant)
                current_job_identity = _job_identity(
                    dataset_id=dataset_id,
                    model_variant=variant_spec.model_variant,
                    feature_protocol_id=variant_spec.feature_protocol_id,
                )
                if current_job_key in completed_job_keys:
                    job_progress.set_postfix_str(f"{dataset_id}/{variant_spec.model_variant}")
                    job_progress.update(1)
                    continue
                prepared = (
                    prepare_dataset(
                        dataset_id,
                        variant_spec=variant_spec,
                        cache_root=cache_root,
                        max_train_origins=max_train_origins,
                        max_eval_origins=max_eval_origins,
                    )
                    if dataset_loader is None
                    else dataset_loader(
                        dataset_id,
                        variant_spec=variant_spec,
                        cache_root=cache_root,
                        max_train_origins=max_train_origins,
                        max_eval_origins=max_eval_origins,
                    )
                )
                resolved_profile = resolve_hyperparameter_profile(
                    variant_spec.model_variant,
                    dataset_id=dataset_id,
                    batch_size=batch_size,
                    learning_rate=learning_rate,
                    max_epochs=max_epochs,
                    early_stopping_patience=early_stopping_patience,
                    node_state_dim=node_state_dim,
                    hidden_state_dim=hidden_state_dim,
                    latent_state_dim=latent_state_dim,
                    global_state_dim=global_state_dim,
                    message_dim=message_dim,
                    edge_hidden_dim=edge_hidden_dim,
                    tau_embed_dim=tau_embed_dim,
                    dropout=dropout,
                    grad_clip_norm=grad_clip_norm,
                    hist_loss_weight=hist_loss_weight,
                    aux_turbine_loss_weight=aux_turbine_loss_weight,
                    aux_site_loss_weight=aux_site_loss_weight,
                    consistency_loss_weight=consistency_loss_weight,
                    weight_decay=weight_decay,
                )
                checkpoint_path = _job_checkpoint_path(
                    resume_paths,
                    dataset_id=dataset_id,
                    model_variant=variant_spec.model_variant,
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
                        eval_batch_size=eval_batch_size,
                        learning_rate=resolved_profile.learning_rate,
                        max_epochs=resolved_profile.max_epochs,
                        early_stopping_patience=resolved_profile.early_stopping_patience,
                        node_state_dim=resolved_profile.node_state_dim,
                        hidden_state_dim=resolved_profile.hidden_state_dim,
                        latent_state_dim=resolved_profile.latent_state_dim,
                        global_state_dim=resolved_profile.global_state_dim,
                        message_dim=resolved_profile.message_dim,
                        edge_hidden_dim=resolved_profile.edge_hidden_dim,
                        tau_embed_dim=resolved_profile.tau_embed_dim,
                        dropout=resolved_profile.dropout,
                        grad_clip_norm=resolved_profile.grad_clip_norm,
                        hist_loss_weight=resolved_profile.hist_loss_weight,
                        aux_turbine_loss_weight=resolved_profile.aux_turbine_loss_weight,
                        aux_site_loss_weight=resolved_profile.aux_site_loss_weight,
                        consistency_loss_weight=resolved_profile.consistency_loss_weight,
                        weight_decay=resolved_profile.weight_decay,
                        num_workers=num_workers,
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
                    dataset_id=dataset_id,
                    model_variant=variant_spec.model_variant,
                )
                _write_resume_state(resume_paths, status="running", effective_config=effective_config, active_job=None)
                resume_active_job = None
                job_progress.set_postfix_str(f"{dataset_id}/{variant_spec.model_variant}")
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
        description="Run the world_model_rollout_v1 family on kelmarsh and penmanshiel."
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
        help="Limit execution to the active world-model rollout variant.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Maximum training epochs. Defaults to the per-dataset profile when omitted.",
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
        help="Optional smoke-test limit applied to dense train origins after split selection.",
    )
    parser.add_argument(
        "--max-eval-origins",
        type=int,
        default=None,
        help="Optional smoke-test limit applied to dense val/test origins before non-overlap thinning.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Training batch size. Defaults to the per-dataset profile when omitted.",
    )
    parser.add_argument(
        "--eval-batch-size",
        type=int,
        default=None,
        help="Evaluation batch size. Defaults to an auto-scaled batch when omitted.",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=None,
        help="AdamW learning rate. Defaults to the per-dataset profile when omitted.",
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
        help="Early stopping patience in epochs.",
    )
    parser.add_argument("--node-state-dim", type=int, default=None, help="Full node-state dimension.")
    parser.add_argument("--hidden-state-dim", type=int, default=None, help="Node hidden-state dimension.")
    parser.add_argument("--latent-state-dim", type=int, default=None, help="Node latent-state dimension.")
    parser.add_argument("--global-state-dim", type=int, default=None, help="Global-state dimension.")
    parser.add_argument("--message-dim", type=int, default=None, help="Message-passing dimension.")
    parser.add_argument("--edge-hidden-dim", type=int, default=None, help="Edge-network hidden dimension.")
    parser.add_argument("--tau-embed-dim", type=int, default=None, help="Forecast-step embedding dimension.")
    parser.add_argument("--dropout", type=float, default=None, help="Dropout applied inside the MLP blocks.")
    parser.add_argument(
        "--grad-clip-norm",
        type=float,
        default=None,
        help="Gradient clipping max norm. Use 0 to disable clipping.",
    )
    parser.add_argument(
        "--hist-loss-weight",
        type=float,
        default=None,
        help="Weight for the history one-step prior rollout loss.",
    )
    parser.add_argument(
        "--aux-turbine-loss-weight",
        type=float,
        default=None,
        help="Weight for the turbine-level auxiliary rollout loss.",
    )
    parser.add_argument(
        "--aux-site-loss-weight",
        type=float,
        default=None,
        help="Weight for the site-level auxiliary rollout loss.",
    )
    parser.add_argument(
        "--consistency-loss-weight",
        type=float,
        default=None,
        help="Weight for the site-consistency loss between summed turbine forecasts and the site head.",
    )
    parser.add_argument("--weight-decay", type=float, default=None, help="AdamW weight decay.")
    parser.add_argument(
        "--num-workers",
        type=int,
        default=None,
        help="DataLoader worker count. Defaults to an auto CUDA-tuned value when omitted.",
    )
    parser.add_argument(
        "--run-label",
        type=str,
        default=None,
        help="Optional label suffix for the formal run record under experiment/artifacts/runs/world_model_rollout_v1/.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume an interrupted run_world_model_rollout_v1 invocation from experiment/families/world_model_rollout_v1/.work/.",
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
    resolved_dataset_ids = tuple(args.datasets) if args.datasets else DEFAULT_DATASETS
    variant_specs = resolve_variant_specs(tuple(args.variants) if args.variants else None)
    resolved_dataset_variant_hyperparameters = {
        dataset_id: {
            spec.model_variant: {
                "feature_protocol_id": spec.feature_protocol_id,
                **asdict(
                    resolve_hyperparameter_profile(
                        spec.model_variant,
                        dataset_id=dataset_id,
                        batch_size=args.batch_size,
                        learning_rate=args.learning_rate,
                        max_epochs=args.epochs,
                        early_stopping_patience=args.patience,
                        node_state_dim=args.node_state_dim,
                        hidden_state_dim=args.hidden_state_dim,
                        latent_state_dim=args.latent_state_dim,
                        global_state_dim=args.global_state_dim,
                        message_dim=args.message_dim,
                        edge_hidden_dim=args.edge_hidden_dim,
                        tau_embed_dim=args.tau_embed_dim,
                        dropout=args.dropout,
                        grad_clip_norm=args.grad_clip_norm,
                        hist_loss_weight=args.hist_loss_weight,
                        aux_turbine_loss_weight=args.aux_turbine_loss_weight,
                        aux_site_loss_weight=args.aux_site_loss_weight,
                        consistency_loss_weight=args.consistency_loss_weight,
                        weight_decay=args.weight_decay,
                    )
                ),
            }
            for spec in variant_specs
        }
        for dataset_id in resolved_dataset_ids
    }
    results = run_experiment(
        dataset_ids=resolved_dataset_ids,
        variant_names=tuple(spec.model_variant for spec in variant_specs),
        device=args.device,
        max_epochs=args.epochs,
        output_path=args.output_path,
        max_train_origins=args.max_train_origins,
        max_eval_origins=args.max_eval_origins,
        seed=args.seed,
        batch_size=args.batch_size,
        eval_batch_size=args.eval_batch_size,
        learning_rate=args.learning_rate,
        early_stopping_patience=args.patience,
        node_state_dim=args.node_state_dim,
        hidden_state_dim=args.hidden_state_dim,
        latent_state_dim=args.latent_state_dim,
        global_state_dim=args.global_state_dim,
        message_dim=args.message_dim,
        edge_hidden_dim=args.edge_hidden_dim,
        tau_embed_dim=args.tau_embed_dim,
        dropout=args.dropout,
        grad_clip_norm=args.grad_clip_norm,
        hist_loss_weight=args.hist_loss_weight,
        aux_turbine_loss_weight=args.aux_turbine_loss_weight,
        aux_site_loss_weight=args.aux_site_loss_weight,
        consistency_loss_weight=args.consistency_loss_weight,
        weight_decay=args.weight_decay,
        num_workers=args.num_workers,
        resume=args.resume,
        force_rerun=args.force_rerun,
    )
    if not args.no_record_run:
        recorded_args = vars(args).copy()
        recorded_args["resolved_dataset_variant_hyperparameters"] = resolved_dataset_variant_hyperparameters
        record_cli_run(
            family_id=FAMILY_ID,
            repo_root=_REPO_ROOT,
            invocation_kind="family_runner",
            entrypoint="experiment/families/world_model_rollout_v1/run_world_model_rollout_v1.py",
            args=recorded_args,
            output_path=args.output_path,
            result_row_count=results.height,
            dataset_ids=resolved_dataset_ids,
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
