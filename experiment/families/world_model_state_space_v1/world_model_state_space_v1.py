from __future__ import annotations

import argparse
from contextlib import nullcontext
import copy
from dataclasses import asdict, dataclass
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
    from torch.utils.tensorboard import SummaryWriter
except ImportError:  # pragma: no cover - optional dependency
    SummaryWriter = None

try:
    from tqdm.auto import tqdm

    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False

    class tqdm:  # type: ignore[no-redef]
        def __init__(self, *args, **kwargs) -> None:
            del args
            self.total = kwargs.get("total")
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
WORLD_MODEL_ROLLOUT_DIR = EXPERIMENT_DIR.parent / "world_model_rollout_v1"
for path in (WORLD_MODEL_AGCRN_DIR, WORLD_MODEL_ROLLOUT_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import world_model_agcrn_v1 as world_model_base  # noqa: E402
import world_model_rollout_v1 as rollout_base  # noqa: E402


torch = world_model_base.torch
nn = world_model_base.nn
F = world_model_base.F
DataLoader = world_model_base.DataLoader
Dataset = world_model_base.Dataset


MODEL_ID = "WORLD_MODEL"
FAMILY_ID = "world_model_state_space_v1"
MODEL_VARIANT = "world_model_state_space_v1_farm_sync"
WINDOW_PROTOCOL = DEFAULT_WINDOW_PROTOCOL
TASK_PROTOCOL: WindowProtocolSpec = resolve_window_protocol(WINDOW_PROTOCOL)
TASK_ID = TASK_PROTOCOL.task_id
DEFAULT_DATASETS = ("kelmarsh",)
HISTORY_STEPS = 144
FORECAST_STEPS = 36
STRIDE_STEPS = 1
FEATURE_PROTOCOL_ID = "world_model_v1"

DEFAULT_BATCH_SIZE = 432
DEFAULT_LEARNING_RATE = 2e-4
DEFAULT_MAX_EPOCHS = 30
DEFAULT_EARLY_STOPPING_PATIENCE = 10
DEFAULT_SEED = 3407
DEFAULT_Z_DIM = 32
DEFAULT_H_DIM = 64
DEFAULT_GLOBAL_STATE_DIM = 48
DEFAULT_OBS_ENCODING_DIM = 64
DEFAULT_INNOVATION_DIM = 32
DEFAULT_SOURCE_SUMMARY_DIM = 32
DEFAULT_EDGE_MESSAGE_DIM = 64
DEFAULT_EDGE_HIDDEN_DIM = 128
DEFAULT_TAU_EMBED_DIM = 16
DEFAULT_MET_SUMMARY_DIM = 6
DEFAULT_TURBINE_EMBED_DIM = 8
DEFAULT_DROPOUT = 0.05
DEFAULT_GRAD_CLIP_NORM = 1.0
DEFAULT_HIST_RECON_LOSS_WEIGHT = 0.2
DEFAULT_FARM_LOSS_WEIGHT = 0.1
DEFAULT_MET_LOSS_WEIGHT = 0.05
DEFAULT_INNOVATION_LOSS_WEIGHT = 1e-4
DEFAULT_WEIGHT_DECAY = 1e-4
DEFAULT_WAKE_LAMBDA_X = 6.0
DEFAULT_WAKE_LAMBDA_Y = 2.0
DEFAULT_WAKE_KAPPA = 2.0
DEFAULT_BOUNDED_OUTPUT_EPSILON = 0.05
DEFAULT_DELTA_CAP_STEPS = 72
DEFAULT_HISTORY_RECON_BURN_IN = 108
DEFAULT_CUDA_NUM_WORKERS = 4
DEFAULT_CUDA_PREFETCH_FACTOR = 4
DEFAULT_EVAL_BATCH_SIZE_MULTIPLIER = 2
DEFAULT_MAX_CUDA_EVAL_BATCH_SIZE = 256
PROFILE_LOG_PREFIX = "[world_model_state_space_v1] "

LOCAL_VALUE_COUNT = 17
LOCAL_MASK_COUNT = 17
LOCAL_DELTA_COUNT = 17
SITE_SUMMARY_FEATURE_NAMES = (
    "site_ws_mean",
    "site_wd_cos_mean",
    "site_wd_sin_mean",
    "site_wd_resultant_length",
    "site_ws_avail_rate",
    "site_wd_avail_rate",
)
KNOWN_FUTURE_FEATURE_COLUMNS = rollout_base.KNOWN_FUTURE_FEATURE_COLUMNS
STATIC_FEATURE_NAMES = rollout_base.STATIC_FEATURE_NAMES
PAIRWISE_FEATURE_NAMES = rollout_base.PAIRWISE_FEATURE_NAMES
WAKE_GEOMETRY_FEATURE_NAMES = ("delta_x_like", "delta_y_like", "distance_in_rotor_diameters_like")
_RAW_WAKE_GEOMETRY_COLUMNS = (
    "delta_x_m",
    "delta_y_m",
    "distance_in_rotor_diameters",
)

_REPO_ROOT = EXPERIMENT_ROOT.parent
_CACHE_ROOT = _REPO_ROOT / "cache"
_OUTPUT_PATH = default_family_output_path(repo_root=_REPO_ROOT, family_id=FAMILY_ID)
_RUN_WORK_ROOT = EXPERIMENT_DIR / ".work" / "run_world_model_state_space_v1"
_RUN_STATE_SCHEMA_VERSION = "world_model_state_space_v1.run.resume.v1"
_TRAINING_CHECKPOINT_SCHEMA_VERSION = "world_model_state_space_v1.training_checkpoint.v1"
_DATASET_ORDER = {"kelmarsh": 0}
_MODEL_VARIANT_ORDER = {MODEL_VARIANT: 0}
_SPLIT_ORDER = {"val": 0, "test": 1}
_EVAL_PROTOCOL_ORDER = {ROLLING_EVAL_PROTOCOL: 0, NON_OVERLAP_EVAL_PROTOCOL: 1}
_METRIC_SCOPE_ORDER = {OVERALL_METRIC_SCOPE: 0, HORIZON_METRIC_SCOPE: 1}

_LOCAL_VALUE_START = 0
_LOCAL_MASK_START = LOCAL_VALUE_COUNT
_LOCAL_DELTA_START = _LOCAL_MASK_START + LOCAL_MASK_COUNT
_LOCAL_AUDIT_START = _LOCAL_DELTA_START + LOCAL_DELTA_COUNT
_CONTEXT_GLOBAL_VALUE_START = 0
_CONTEXT_GLOBAL_MASK_START = 9
_CONTEXT_GLOBAL_DELTA_START = 18
_CONTEXT_SITE_SUMMARY_START = 27
_CONTEXT_CALENDAR_START = 33

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
    "context_history_channels",
    "context_future_channels",
    "static_feature_count",
    "pairwise_feature_count",
    "z_dim",
    "h_dim",
    "global_state_dim",
    "source_summary_dim",
    "met_summary_dim",
    "wake_lambda_x",
    "wake_lambda_y",
    "wake_kappa",
    "bounded_output_epsilon",
    "hist_recon_loss_weight",
    "farm_loss_weight",
    "met_loss_weight",
    "innovation_loss_weight",
    "weight_decay",
    "amp_enabled",
    "device",
    "runtime_seconds",
    "train_window_count",
    "val_window_count",
    "test_window_count",
    "best_epoch",
    "epochs_ran",
    "best_val_rmse_pu",
    "best_val_mae_pu",
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
    "train_forecast_loss_mean",
    "train_forecast_loss_last",
    "train_hist_recon_loss_mean",
    "train_hist_recon_loss_last",
    "train_farm_loss_mean",
    "train_farm_loss_last",
    "train_met_loss_mean",
    "train_met_loss_last",
    "train_innovation_loss_mean",
    "train_innovation_loss_last",
    "val_mae_pu",
    "val_rmse_pu",
    "best_val_mae_pu",
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
    "amp_enabled",
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
    z_dim: int
    h_dim: int
    global_state_dim: int
    obs_encoding_dim: int
    innovation_dim: int
    source_summary_dim: int
    edge_message_dim: int
    edge_hidden_dim: int
    tau_embed_dim: int
    met_summary_dim: int
    turbine_embed_dim: int
    dropout: float
    grad_clip_norm: float
    hist_recon_loss_weight: float
    farm_loss_weight: float
    met_loss_weight: float
    innovation_loss_weight: float
    weight_decay: float
    wake_lambda_x: float
    wake_lambda_y: float
    wake_kappa: float
    bounded_output_epsilon: float


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
    turbine_indices: np.ndarray
    coordinate_mode: str
    node_count: int
    timestamps_us: np.ndarray
    local_input_feature_names: tuple[str, ...]
    context_history_feature_names: tuple[str, ...]
    context_future_feature_names: tuple[str, ...]
    future_met_feature_names: tuple[str, ...]
    static_feature_names: tuple[str, ...]
    pairwise_feature_names: tuple[str, ...]
    wake_geometry_feature_names: tuple[str, ...]
    local_history_tensor: np.ndarray
    context_history_tensor: np.ndarray
    context_future_tensor: np.ndarray
    future_met_tensor: np.ndarray
    future_met_valid_mask: np.ndarray
    future_farm_target_pu: np.ndarray
    future_farm_valid_mask: np.ndarray
    static_tensor: np.ndarray
    pairwise_tensor: np.ndarray
    wake_geometry_tensor: np.ndarray
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
    def context_history_channels(self) -> int:
        return int(self.context_history_tensor.shape[1])

    @property
    def context_future_channels(self) -> int:
        return int(self.context_future_tensor.shape[1])

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
    best_val_mae_pu: float
    device: str
    amp_enabled: bool
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


@dataclass(frozen=True)
class ModelOutputs:
    future_predictions: Any
    hist_prior_observations: Any
    hist_post_observations: Any
    future_farm_predictions: Any
    future_met_predictions: Any
    innovation_regularization: Any


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
    z_dim=DEFAULT_Z_DIM,
    h_dim=DEFAULT_H_DIM,
    global_state_dim=DEFAULT_GLOBAL_STATE_DIM,
    obs_encoding_dim=DEFAULT_OBS_ENCODING_DIM,
    innovation_dim=DEFAULT_INNOVATION_DIM,
    source_summary_dim=DEFAULT_SOURCE_SUMMARY_DIM,
    edge_message_dim=DEFAULT_EDGE_MESSAGE_DIM,
    edge_hidden_dim=DEFAULT_EDGE_HIDDEN_DIM,
    tau_embed_dim=DEFAULT_TAU_EMBED_DIM,
    met_summary_dim=DEFAULT_MET_SUMMARY_DIM,
    turbine_embed_dim=DEFAULT_TURBINE_EMBED_DIM,
    dropout=DEFAULT_DROPOUT,
    grad_clip_norm=DEFAULT_GRAD_CLIP_NORM,
    hist_recon_loss_weight=DEFAULT_HIST_RECON_LOSS_WEIGHT,
    farm_loss_weight=DEFAULT_FARM_LOSS_WEIGHT,
    met_loss_weight=DEFAULT_MET_LOSS_WEIGHT,
    innovation_loss_weight=DEFAULT_INNOVATION_LOSS_WEIGHT,
    weight_decay=DEFAULT_WEIGHT_DECAY,
    wake_lambda_x=DEFAULT_WAKE_LAMBDA_X,
    wake_lambda_y=DEFAULT_WAKE_LAMBDA_Y,
    wake_kappa=DEFAULT_WAKE_KAPPA,
    bounded_output_epsilon=DEFAULT_BOUNDED_OUTPUT_EPSILON,
)
TUNED_DEFAULT_HYPERPARAMETERS_BY_DATASET_AND_VARIANT = {
    "kelmarsh": {MODEL_VARIANT: _DEFAULT_PROFILE},
}


def _require_torch() -> tuple[Any, Any, Any, Any, Any]:
    return world_model_base.require_torch()


def resolve_device(device: str | None = None) -> str:
    return world_model_base.resolve_device(device)


def _validate_dataset_ids(dataset_ids: Sequence[str]) -> tuple[str, ...]:
    resolved = tuple(dict.fromkeys(dataset_ids))
    unsupported = sorted(set(resolved) - set(DEFAULT_DATASETS))
    if unsupported:
        raise ValueError(f"{FAMILY_ID} only supports {DEFAULT_DATASETS!r}; received {unsupported!r}.")
    return resolved


def resolve_variant_specs(variant_names: Sequence[str] | None = None) -> tuple[ExperimentVariant, ...]:
    requested = tuple(variant_names or DEFAULT_VARIANTS)
    resolved: list[ExperimentVariant] = []
    seen: set[str] = set()
    for variant_name in requested:
        try:
            spec = _VARIANT_SPECS_BY_NAME[variant_name]
        except KeyError as exc:
            raise ValueError(f"Unknown model variant {variant_name!r}.") from exc
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
    z_dim: int | None = None,
    h_dim: int | None = None,
    global_state_dim: int | None = None,
    obs_encoding_dim: int | None = None,
    innovation_dim: int | None = None,
    source_summary_dim: int | None = None,
    edge_message_dim: int | None = None,
    edge_hidden_dim: int | None = None,
    tau_embed_dim: int | None = None,
    met_summary_dim: int | None = None,
    turbine_embed_dim: int | None = None,
    dropout: float | None = None,
    grad_clip_norm: float | None = None,
    hist_recon_loss_weight: float | None = None,
    farm_loss_weight: float | None = None,
    met_loss_weight: float | None = None,
    innovation_loss_weight: float | None = None,
    weight_decay: float | None = None,
    wake_lambda_x: float | None = None,
    wake_lambda_y: float | None = None,
    wake_kappa: float | None = None,
    bounded_output_epsilon: float | None = None,
) -> HyperparameterProfile:
    try:
        defaults = TUNED_DEFAULT_HYPERPARAMETERS_BY_DATASET_AND_VARIANT[dataset_id][variant_name]
    except KeyError as exc:
        raise ValueError(f"Unsupported dataset/variant pair {dataset_id!r}/{variant_name!r}.") from exc
    profile = HyperparameterProfile(
        batch_size=defaults.batch_size if batch_size is None else batch_size,
        learning_rate=defaults.learning_rate if learning_rate is None else learning_rate,
        max_epochs=defaults.max_epochs if max_epochs is None else max_epochs,
        early_stopping_patience=(
            defaults.early_stopping_patience if early_stopping_patience is None else early_stopping_patience
        ),
        z_dim=defaults.z_dim if z_dim is None else z_dim,
        h_dim=defaults.h_dim if h_dim is None else h_dim,
        global_state_dim=defaults.global_state_dim if global_state_dim is None else global_state_dim,
        obs_encoding_dim=defaults.obs_encoding_dim if obs_encoding_dim is None else obs_encoding_dim,
        innovation_dim=defaults.innovation_dim if innovation_dim is None else innovation_dim,
        source_summary_dim=defaults.source_summary_dim if source_summary_dim is None else source_summary_dim,
        edge_message_dim=defaults.edge_message_dim if edge_message_dim is None else edge_message_dim,
        edge_hidden_dim=defaults.edge_hidden_dim if edge_hidden_dim is None else edge_hidden_dim,
        tau_embed_dim=defaults.tau_embed_dim if tau_embed_dim is None else tau_embed_dim,
        met_summary_dim=defaults.met_summary_dim if met_summary_dim is None else met_summary_dim,
        turbine_embed_dim=defaults.turbine_embed_dim if turbine_embed_dim is None else turbine_embed_dim,
        dropout=defaults.dropout if dropout is None else dropout,
        grad_clip_norm=defaults.grad_clip_norm if grad_clip_norm is None else grad_clip_norm,
        hist_recon_loss_weight=(
            defaults.hist_recon_loss_weight if hist_recon_loss_weight is None else hist_recon_loss_weight
        ),
        farm_loss_weight=defaults.farm_loss_weight if farm_loss_weight is None else farm_loss_weight,
        met_loss_weight=defaults.met_loss_weight if met_loss_weight is None else met_loss_weight,
        innovation_loss_weight=(
            defaults.innovation_loss_weight if innovation_loss_weight is None else innovation_loss_weight
        ),
        weight_decay=defaults.weight_decay if weight_decay is None else weight_decay,
        wake_lambda_x=defaults.wake_lambda_x if wake_lambda_x is None else wake_lambda_x,
        wake_lambda_y=defaults.wake_lambda_y if wake_lambda_y is None else wake_lambda_y,
        wake_kappa=defaults.wake_kappa if wake_kappa is None else wake_kappa,
        bounded_output_epsilon=(
            defaults.bounded_output_epsilon if bounded_output_epsilon is None else bounded_output_epsilon
        ),
    )
    if profile.dropout < 0.0 or profile.dropout >= 1.0:
        raise ValueError(f"dropout must be in [0, 1), found {profile.dropout!r}.")
    return profile


def progress_is_enabled() -> bool:
    return HAS_TQDM and sys.stderr.isatty()


def _create_progress_bar(*, total: int | None, desc: str, leave: bool = False):
    return tqdm(total=total, desc=desc, leave=leave, disable=not progress_is_enabled(), dynamic_ncols=True)


def _loader_batch_total(loader: object) -> int | None:
    return world_model_base._loader_batch_total(loader)


def resolve_loader_num_workers(*, device: str, num_workers: int | None = None) -> int:
    if num_workers is not None:
        if num_workers < 0:
            raise ValueError(f"num_workers must be >= 0, found {num_workers!r}.")
        return num_workers
    if resolve_device(device) != "cuda":
        return 0
    return min(DEFAULT_CUDA_NUM_WORKERS, os.cpu_count() or 0)


def resolve_eval_batch_size(train_batch_size: int, *, device: str, eval_batch_size: int | None = None) -> int:
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


def _profile_log(dataset_id: str, phase: str, **fields: object) -> None:
    payload = {"dataset_id": dataset_id, "phase": phase, **fields}
    print(f"{PROFILE_LOG_PREFIX}{json.dumps(payload, ensure_ascii=False, default=str, sort_keys=True)}", file=sys.stderr)


def _tensorboard_job_log_dir(
    tensorboard_root: str | Path,
    *,
    dataset_id: str,
    model_variant: str,
) -> Path:
    return Path(tensorboard_root) / dataset_id / model_variant


def _open_tensorboard_writer(
    tensorboard_log_dir: str | Path | None,
    *,
    dataset_id: str,
    model_variant: str,
):
    if tensorboard_log_dir is None:
        return None
    resolved_dir = Path(tensorboard_log_dir).expanduser().resolve()
    if SummaryWriter is None:
        _profile_log(
            dataset_id,
            "tensorboard_unavailable",
            model_variant=model_variant,
            log_dir=str(resolved_dir),
            reason="tensorboard package is not installed in the family environment",
        )
        return None
    resolved_dir.mkdir(parents=True, exist_ok=True)
    _profile_log(
        dataset_id,
        "tensorboard_enabled",
        model_variant=model_variant,
        log_dir=str(resolved_dir),
    )
    return SummaryWriter(log_dir=str(resolved_dir), flush_secs=10)


def _close_tensorboard_writer(writer) -> None:
    if writer is None:
        return
    writer.flush()
    writer.close()


def _tensorboard_add_scalar(writer, tag: str, value: object, step: int) -> None:
    if writer is None or value is None:
        return
    numeric_value = float(value)
    if not math.isfinite(numeric_value):
        return
    writer.add_scalar(tag, numeric_value, step)


def _tensorboard_add_text(writer, tag: str, text: str, *, step: int = 0) -> None:
    if writer is None:
        return
    writer.add_text(tag, text, global_step=step)


def _tensorboard_add_metrics(writer, prefix: str, metrics: EvaluationMetrics, *, step: int) -> None:
    if writer is None:
        return
    _tensorboard_add_scalar(writer, f"{prefix}/overall/mae_pu", metrics.mae_pu, step)
    _tensorboard_add_scalar(writer, f"{prefix}/overall/rmse_pu", metrics.rmse_pu, step)
    _tensorboard_add_scalar(writer, f"{prefix}/overall/mae_kw", metrics.mae_kw, step)
    _tensorboard_add_scalar(writer, f"{prefix}/overall/rmse_kw", metrics.rmse_kw, step)
    _tensorboard_add_scalar(writer, f"{prefix}/overall/window_count", metrics.window_count, step)
    _tensorboard_add_scalar(writer, f"{prefix}/overall/prediction_count", metrics.prediction_count, step)
    for lead_index in range(len(metrics.horizon_mae_pu)):
        lead_step = lead_index + 1
        lead_tag = f"lead_{lead_step:02d}"
        _tensorboard_add_scalar(writer, f"{prefix}/horizon/mae_pu/{lead_tag}", metrics.horizon_mae_pu[lead_index], step)
        _tensorboard_add_scalar(writer, f"{prefix}/horizon/rmse_pu/{lead_tag}", metrics.horizon_rmse_pu[lead_index], step)
        _tensorboard_add_scalar(writer, f"{prefix}/horizon/mae_kw/{lead_tag}", metrics.horizon_mae_kw[lead_index], step)
        _tensorboard_add_scalar(writer, f"{prefix}/horizon/rmse_kw/{lead_tag}", metrics.horizon_rmse_kw[lead_index], step)


def _tensorboard_log_run_config(
    writer,
    *,
    prepared_dataset: PreparedDataset,
    profile: HyperparameterProfile,
    seed: int,
    device: str,
    eval_batch_size: int | None,
    num_workers: int | None,
) -> None:
    if writer is None:
        return
    config = {
        "dataset_id": prepared_dataset.dataset_id,
        "model_variant": prepared_dataset.model_variant,
        "feature_protocol_id": prepared_dataset.feature_protocol_id,
        "task_id": TASK_ID,
        "history_steps": prepared_dataset.history_steps,
        "forecast_steps": prepared_dataset.forecast_steps,
        "node_count": prepared_dataset.node_count,
        "resolution_minutes": prepared_dataset.resolution_minutes,
        "rated_power_kw": prepared_dataset.rated_power_kw,
        "local_input_channels": prepared_dataset.local_input_channels,
        "context_history_channels": prepared_dataset.context_history_channels,
        "context_future_channels": prepared_dataset.context_future_channels,
        "static_feature_count": prepared_dataset.static_feature_count,
        "pairwise_feature_count": prepared_dataset.pairwise_feature_count,
        "train_window_count": len(prepared_dataset.train_windows),
        "val_window_count": len(prepared_dataset.val_rolling_windows),
        "test_window_count": len(prepared_dataset.test_rolling_windows),
        "device": device,
        "seed": seed,
        "eval_batch_size": eval_batch_size,
        "num_workers": num_workers,
        **asdict(profile),
    }
    _tensorboard_add_text(writer, "run/config_json", f"```json\n{json.dumps(config, ensure_ascii=False, indent=2, sort_keys=True)}\n```")
    _tensorboard_add_scalar(writer, "data/train_window_count", len(prepared_dataset.train_windows), 0)
    _tensorboard_add_scalar(writer, "data/val_window_count", len(prepared_dataset.val_rolling_windows), 0)
    _tensorboard_add_scalar(writer, "data/test_window_count", len(prepared_dataset.test_rolling_windows), 0)


def _tensorboard_log_final_evaluations(
    writer,
    evaluation_results: Sequence[tuple[str, str, world_model_base.FarmWindowDescriptorIndex, EvaluationMetrics]],
    *,
    step: int,
) -> None:
    if writer is None:
        return
    summary_rows: list[dict[str, object]] = []
    for split_name, eval_protocol, _windows, metrics in evaluation_results:
        prefix = f"final/{split_name}/{eval_protocol}"
        _tensorboard_add_metrics(writer, prefix, metrics, step=step)
        summary_rows.append(
            {
                "split_name": split_name,
                "eval_protocol": eval_protocol,
                "mae_pu": float(metrics.mae_pu),
                "rmse_pu": float(metrics.rmse_pu),
                "mae_kw": float(metrics.mae_kw),
                "rmse_kw": float(metrics.rmse_kw),
                "window_count": int(metrics.window_count),
                "prediction_count": int(metrics.prediction_count),
            }
        )
    _tensorboard_add_text(
        writer,
        "final/summary_json",
        f"```json\n{json.dumps(summary_rows, ensure_ascii=False, indent=2, sort_keys=True)}\n```",
        step=step,
    )


def _delta_since_available(mask_unavailable: np.ndarray, *, cap_steps: int = DEFAULT_DELTA_CAP_STEPS) -> np.ndarray:
    masks = np.asarray(mask_unavailable, dtype=np.float32)
    flat = masks.reshape(masks.shape[0], -1)
    result = np.zeros_like(flat, dtype=np.float32)
    missing_run = np.full((flat.shape[1],), cap_steps, dtype=np.float32)
    seen_valid = np.zeros((flat.shape[1],), dtype=bool)
    for time_index in range(flat.shape[0]):
        available = flat[time_index] < 0.5
        missing_run = np.where(available, 0.0, np.minimum(missing_run + 1.0, float(cap_steps)))
        seen_valid = seen_valid | available
        result[time_index] = np.where(seen_valid, missing_run, float(cap_steps))
    return (np.minimum(result, float(cap_steps)) / float(cap_steps)).reshape(masks.shape)


def _build_raw_wake_geometry_tensor(
    *,
    dataset_id: str,
    feature_protocol_id: str,
    cache_root: str | Path = _CACHE_ROOT,
) -> np.ndarray:
    bundle = world_model_base._load_task_bundle(
        dataset_id,
        feature_protocol_id=feature_protocol_id,
        cache_root=cache_root,
    )
    metadata = world_model_base.load_dataset_metadata(dataset_id, bundle)
    pairwise_frame = world_model_base.load_pairwise_frame(dataset_id, bundle)
    feature_frame = pairwise_frame.select(list(_RAW_WAKE_GEOMETRY_COLUMNS))
    incomplete = [column for column in _RAW_WAKE_GEOMETRY_COLUMNS if feature_frame[column].null_count() > 0]
    if incomplete:
        raise ValueError(f"Task bundle pairwise sidecar has null values in wake geometry columns {incomplete!r}.")

    node_count = len(metadata.turbine_ids)
    turbine_index_by_id = {turbine_id: index for index, turbine_id in enumerate(metadata.turbine_ids)}
    raw_features = feature_frame.to_numpy().astype(np.float32, copy=False)
    tensor = np.zeros((node_count, node_count, len(_RAW_WAKE_GEOMETRY_COLUMNS)), dtype=np.float32)
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
        tensor[dst_index, src_index, :] = raw_features[row_index]
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


def _build_state_space_tensors(base: rollout_base.PreparedDataset, *, raw_wake_geometry: np.ndarray) -> dict[str, object]:
    local = base.local_history_tensor
    value_indices = [0, *range(2, 18)]
    mask_indices = [1, *range(18, 34)]
    audit_indices = list(range(34, 37))
    values = local[:, :, value_indices]
    masks = local[:, :, mask_indices]
    deltas = _delta_since_available(masks)
    local_tensor = np.concatenate([values, masks, deltas, local[:, :, audit_indices]], axis=2).astype(np.float32)
    local_names = tuple(
        [
            "target_pu",
            *base.local_input_feature_names[2:18],
            "target_kw__mask",
            *base.local_input_feature_names[18:34],
            "target_pu__delta_since_valid",
            *(f"{name}__delta_since_valid" for name in base.local_input_feature_names[2:18]),
            *base.local_input_feature_names[34:37],
        ]
    )

    context = base.context_history_tensor
    global_values = context[:, :9]
    global_masks = context[:, 9:18]
    global_deltas = _delta_since_available(global_masks)
    calendar = context[:, 18:]

    wind_speed = values[:, :, 1]
    wind_speed_available = 1.0 - masks[:, :, 1]
    wind_sin = values[:, :, 2]
    wind_cos = values[:, :, 3]
    wind_direction_available = (1.0 - masks[:, :, 2]) * (1.0 - masks[:, :, 3])
    ws_count = wind_speed_available.sum(axis=1)
    wd_count = wind_direction_available.sum(axis=1)
    ws_mean = (wind_speed * wind_speed_available).sum(axis=1) / np.maximum(ws_count, 1.0)
    cos_mean = (wind_cos * wind_direction_available).sum(axis=1) / np.maximum(wd_count, 1.0)
    sin_mean = (wind_sin * wind_direction_available).sum(axis=1) / np.maximum(wd_count, 1.0)
    resultant = np.sqrt(np.square(cos_mean) + np.square(sin_mean))
    ws_rate = ws_count / float(base.node_count)
    wd_rate = wd_count / float(base.node_count)
    site_summary = np.stack([ws_mean, cos_mean, sin_mean, resultant, ws_rate, wd_rate], axis=-1).astype(np.float32)
    site_summary[~np.isfinite(site_summary)] = 0.0
    site_summary_valid = np.stack(
        [
            ws_count > 0,
            wd_count > 0,
            wd_count > 0,
            wd_count > 0,
            np.ones_like(ws_count, dtype=bool),
            np.ones_like(wd_count, dtype=bool),
        ],
        axis=-1,
    ).astype(np.float32)
    history_context = np.concatenate([global_values, global_masks, global_deltas, site_summary, calendar], axis=1).astype(
        np.float32
    )
    future_context = calendar.astype(np.float32)
    farm_target = global_values[:, 1:2].astype(np.float32)
    farm_valid = (1.0 - global_masks[:, 1:2]).astype(np.float32)

    return {
        "local_tensor": local_tensor,
        "local_names": local_names,
        "history_context": history_context,
        "future_context": future_context,
        "history_names": (
            *base.context_feature_names[:9],
            *base.context_feature_names[9:18],
            *(f"{name.removesuffix('__mask')}__delta_since_valid" for name in base.context_feature_names[9:18]),
            *SITE_SUMMARY_FEATURE_NAMES,
            *base.context_feature_names[18:],
        ),
        "future_names": base.context_feature_names[18:],
        "site_summary": site_summary,
        "site_summary_valid": site_summary_valid,
        "farm_target": farm_target,
        "farm_valid": farm_valid,
        "wake_geometry": raw_wake_geometry.astype(np.float32, copy=False),
    }


def prepare_dataset(
    dataset_id: str,
    *,
    variant_spec: ExperimentVariant | None = None,
    cache_root: str | Path = _CACHE_ROOT,
    max_train_origins: int | None = None,
    max_eval_origins: int | None = None,
) -> PreparedDataset:
    _validate_dataset_ids((dataset_id,))
    resolved_variant = variant_spec or VARIANT_SPECS[0]
    base = rollout_base.prepare_dataset(
        dataset_id,
        variant_spec=rollout_base.ExperimentVariant(
            model_variant=resolved_variant.model_variant,
            feature_protocol_id=resolved_variant.feature_protocol_id,
        ),
        cache_root=cache_root,
        max_train_origins=max_train_origins,
        max_eval_origins=max_eval_origins,
    )
    raw_wake_geometry = _build_raw_wake_geometry_tensor(
        dataset_id=dataset_id,
        feature_protocol_id=resolved_variant.feature_protocol_id,
        cache_root=cache_root,
    )
    tensors = _build_state_space_tensors(base, raw_wake_geometry=raw_wake_geometry)
    prepared = PreparedDataset(
        dataset_id=base.dataset_id,
        model_variant=resolved_variant.model_variant,
        feature_protocol_id=base.feature_protocol_id,
        resolution_minutes=base.resolution_minutes,
        rated_power_kw=base.rated_power_kw,
        history_steps=base.history_steps,
        forecast_steps=base.forecast_steps,
        stride_steps=base.stride_steps,
        turbine_ids=base.turbine_ids,
        turbine_indices=np.arange(base.node_count, dtype=np.int64),
        coordinate_mode=base.coordinate_mode,
        node_count=base.node_count,
        timestamps_us=base.timestamps_us,
        local_input_feature_names=tensors["local_names"],  # type: ignore[arg-type]
        context_history_feature_names=tensors["history_names"],  # type: ignore[arg-type]
        context_future_feature_names=tensors["future_names"],  # type: ignore[arg-type]
        future_met_feature_names=SITE_SUMMARY_FEATURE_NAMES,
        static_feature_names=base.static_feature_names,
        pairwise_feature_names=base.pairwise_feature_names,
        wake_geometry_feature_names=WAKE_GEOMETRY_FEATURE_NAMES,
        local_history_tensor=tensors["local_tensor"],  # type: ignore[arg-type]
        context_history_tensor=tensors["history_context"],  # type: ignore[arg-type]
        context_future_tensor=tensors["future_context"],  # type: ignore[arg-type]
        future_met_tensor=tensors["site_summary"],  # type: ignore[arg-type]
        future_met_valid_mask=tensors["site_summary_valid"],  # type: ignore[arg-type]
        future_farm_target_pu=tensors["farm_target"],  # type: ignore[arg-type]
        future_farm_valid_mask=tensors["farm_valid"],  # type: ignore[arg-type]
        static_tensor=base.static_tensor,
        pairwise_tensor=base.pairwise_tensor,
        wake_geometry_tensor=tensors["wake_geometry"],  # type: ignore[arg-type]
        target_pu_filled=base.target_pu_filled,
        target_valid_mask=base.target_valid_mask,
        train_windows=base.train_windows,
        val_rolling_windows=base.val_rolling_windows,
        val_non_overlap_windows=base.val_non_overlap_windows,
        test_rolling_windows=base.test_rolling_windows,
        test_non_overlap_windows=base.test_non_overlap_windows,
    )
    _profile_log(
        dataset_id,
        "prepare_dataset_complete",
        local_input_channels=prepared.local_input_channels,
        context_history_channels=prepared.context_history_channels,
        context_future_channels=prepared.context_future_channels,
        train_windows=len(prepared.train_windows),
    )
    return prepared


def iter_evaluation_specs(
    prepared_dataset: PreparedDataset,
) -> tuple[tuple[str, str, world_model_base.FarmWindowDescriptorIndex], ...]:
    return (
        ("val", ROLLING_EVAL_PROTOCOL, prepared_dataset.val_rolling_windows),
        ("val", NON_OVERLAP_EVAL_PROTOCOL, prepared_dataset.val_non_overlap_windows),
        ("test", ROLLING_EVAL_PROTOCOL, prepared_dataset.test_rolling_windows),
        ("test", NON_OVERLAP_EVAL_PROTOCOL, prepared_dataset.test_non_overlap_windows),
    )


if Dataset is not None:

    class PanelWindowDataset(Dataset):
        def __init__(
            self,
            prepared_dataset: PreparedDataset,
            windows: world_model_base.FarmWindowDescriptorIndex,
        ) -> None:
            self.prepared_dataset = prepared_dataset
            self.windows = windows

        def __len__(self) -> int:
            return len(self.windows)

        def __getitem__(self, index: int):
            prepared = self.prepared_dataset
            target_index = int(self.windows.target_indices[index])
            history_slice = slice(target_index - prepared.history_steps, target_index)
            future_slice = slice(target_index, target_index + prepared.forecast_steps)
            return (
                prepared.local_history_tensor[history_slice].astype(np.float32, copy=True),
                prepared.context_history_tensor[history_slice].astype(np.float32, copy=True),
                prepared.context_future_tensor[future_slice].astype(np.float32, copy=True),
                prepared.target_pu_filled[future_slice, :, None].astype(np.float32, copy=True),
                prepared.target_valid_mask[future_slice, :, None].astype(np.float32, copy=True),
                prepared.future_met_tensor[future_slice].astype(np.float32, copy=True),
                prepared.future_met_valid_mask[future_slice].astype(np.float32, copy=True),
                prepared.future_farm_target_pu[future_slice].astype(np.float32, copy=True),
                prepared.future_farm_valid_mask[future_slice].astype(np.float32, copy=True),
            )

else:

    class PanelWindowDataset:  # pragma: no cover
        def __init__(self, *_args, **_kwargs) -> None:
            _require_torch()


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
    generator = resolved_torch.Generator()
    generator.manual_seed(seed)
    loader_kwargs: dict[str, object] = {
        "dataset": PanelWindowDataset(prepared_dataset, windows),
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


if nn is not None and F is not None:

    class FeedForward(nn.Module):
        def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, *, dropout: float) -> None:
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.SiLU(),
                nn.LayerNorm(hidden_dim),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, output_dim),
            )

        def forward(self, inputs):
            return self.net(inputs)


    class WorldModelStateSpace(nn.Module):
        def __init__(
            self,
            *,
            node_count: int,
            local_input_channels: int,
            context_history_channels: int,
            context_future_channels: int,
            static_tensor: np.ndarray,
            turbine_indices: np.ndarray,
            pairwise_tensor: np.ndarray,
            wake_geometry_tensor: np.ndarray,
            z_dim: int,
            h_dim: int,
            global_state_dim: int,
            obs_encoding_dim: int,
            innovation_dim: int,
            source_summary_dim: int,
            edge_message_dim: int,
            edge_hidden_dim: int,
            tau_embed_dim: int,
            met_summary_dim: int,
            turbine_embed_dim: int,
            forecast_steps: int,
            dropout: float,
            wake_lambda_x: float,
            wake_lambda_y: float,
            wake_kappa: float,
            bounded_output_epsilon: float,
        ) -> None:
            super().__init__()
            self.node_count = node_count
            self.z_dim = z_dim
            self.h_dim = h_dim
            self.node_state_dim = z_dim + h_dim
            self.global_state_dim = global_state_dim
            self.met_summary_dim = met_summary_dim
            self.forecast_steps = forecast_steps
            self.wake_lambda_x = wake_lambda_x
            self.wake_lambda_y = wake_lambda_y
            self.wake_kappa = wake_kappa
            self.bounded_output_epsilon = bounded_output_epsilon
            self.register_buffer("static_features", torch.from_numpy(np.asarray(static_tensor, dtype=np.float32)))
            self.register_buffer("turbine_indices", torch.from_numpy(np.asarray(turbine_indices, dtype=np.int64)))
            self.register_buffer("pairwise_features", torch.from_numpy(np.asarray(pairwise_tensor, dtype=np.float32)))
            self.register_buffer("wake_geometry", torch.from_numpy(np.asarray(wake_geometry_tensor, dtype=np.float32)))
            self.register_buffer("farm_weights", torch.full((node_count,), 1.0 / float(node_count), dtype=torch.float32))
            self.register_buffer(
                "edge_diagonal_mask",
                torch.eye(node_count, dtype=torch.bool)[None, :, :, None],
            )
            self.turbine_embedding = nn.Embedding(max(int(np.max(turbine_indices)) + 1, node_count), turbine_embed_dim)
            static_dim = 32
            static_raw_dim = static_tensor.shape[1] + turbine_embed_dim
            self.static_encoder = FeedForward(static_raw_dim, max(static_dim, static_raw_dim * 2), static_dim, dropout=dropout)
            self.init_z = nn.Linear(static_dim, z_dim)
            self.init_h = nn.Linear(static_dim, h_dim)
            self.init_global = FeedForward(static_dim, max(static_dim, global_state_dim), global_state_dim, dropout=dropout)
            self.obs_encoder = FeedForward(
                local_input_channels + static_dim,
                obs_encoding_dim,
                obs_encoding_dim,
                dropout=dropout,
            )
            self.prior_obs_decoder = FeedForward(self.node_state_dim + global_state_dim + static_dim, self.node_state_dim, LOCAL_VALUE_COUNT, dropout=dropout)
            self.post_obs_decoder = FeedForward(self.node_state_dim + global_state_dim + static_dim, self.node_state_dim, LOCAL_VALUE_COUNT, dropout=dropout)
            self.innovation_encoder = FeedForward((LOCAL_VALUE_COUNT * 4) + obs_encoding_dim, max(obs_encoding_dim, innovation_dim), innovation_dim, dropout=dropout)
            self.update_source = FeedForward(local_input_channels + self.node_state_dim + static_dim, max(obs_encoding_dim, source_summary_dim), source_summary_dim, dropout=dropout)
            self.transition_source = FeedForward(self.node_state_dim + static_dim + global_state_dim, max(edge_hidden_dim, source_summary_dim), source_summary_dim, dropout=dropout)
            edge_input_dim = (static_dim * 2) + pairwise_tensor.shape[2] + 3 + source_summary_dim + global_state_dim + met_summary_dim + context_future_channels
            self.update_edge = FeedForward(edge_input_dim, edge_hidden_dim, edge_message_dim, dropout=dropout)
            self.update_gate = FeedForward(edge_input_dim, edge_hidden_dim, 1, dropout=dropout)
            self.transition_edge = FeedForward(edge_input_dim, edge_hidden_dim, edge_message_dim, dropout=dropout)
            self.transition_gate = FeedForward(edge_input_dim, edge_hidden_dim, 1, dropout=dropout)
            self.update_z = nn.GRUCell(innovation_dim + edge_message_dim + global_state_dim + context_history_channels + static_dim, z_dim)
            self.update_h = nn.GRUCell(z_dim + innovation_dim + edge_message_dim + global_state_dim + context_history_channels + static_dim, h_dim)
            self.transition_z = nn.GRUCell(edge_message_dim + global_state_dim + met_summary_dim + context_future_channels + static_dim + h_dim, z_dim)
            self.transition_h = nn.GRUCell(z_dim + edge_message_dim + global_state_dim + met_summary_dim + context_future_channels + static_dim, h_dim)
            self.global_update = nn.GRUCell(self.node_state_dim + context_history_channels, global_state_dim)
            self.global_transition = nn.GRUCell(self.node_state_dim + met_summary_dim + context_future_channels, global_state_dim)
            self.met_head = FeedForward(global_state_dim + context_future_channels, max(global_state_dim, met_summary_dim), met_summary_dim, dropout=dropout)
            self.tau_embedding = nn.Embedding(forecast_steps, tau_embed_dim)
            self.forecast_decoder = FeedForward(self.node_state_dim + global_state_dim + static_dim + tau_embed_dim, self.node_state_dim, 1, dropout=dropout)
            self.farm_bias = nn.Parameter(torch.zeros((), dtype=torch.float32))
            self.farm_scale = nn.Parameter(torch.ones((), dtype=torch.float32))

        def _static(self, batch_size: int | None = None):
            encoded = self.static_encoder(torch.cat((self.static_features, self.turbine_embedding(self.turbine_indices)), dim=-1))
            if batch_size is None:
                return encoded
            return encoded[None, :, :].expand(batch_size, -1, -1)

        def _initial_states(self, batch_size: int, dtype, device, static_base=None):
            static = self._static() if static_base is None else static_base
            z = self.init_z(static).to(device=device, dtype=dtype)[None, :, :].expand(batch_size, -1, -1)
            h = self.init_h(static).to(device=device, dtype=dtype)[None, :, :].expand(batch_size, -1, -1)
            g = self.init_global(static).mean(dim=0).to(device=device, dtype=dtype)[None, :].expand(batch_size, -1)
            return z, h, g

        def _node_state(self, z, h):
            return torch.cat((z, h), dim=-1)

        def _decode_obs(self, z, h, g, *, posterior: bool, static=None):
            batch_size = z.shape[0]
            static_nodes = self._static(batch_size) if static is None else static
            global_nodes = g[:, None, :].expand(-1, self.node_count, -1)
            inputs = torch.cat((self._node_state(z, h), global_nodes, static_nodes), dim=-1)
            return (self.post_obs_decoder if posterior else self.prior_obs_decoder)(inputs)

        def _met(self, g, calendar):
            raw = self.met_head(torch.cat((g, calendar), dim=-1))
            return torch.cat((raw[:, 0:1], torch.tanh(raw[:, 1:3]), torch.sigmoid(raw[:, 3:6])), dim=-1)

        def _wake(self, met):
            cos_value = met[:, 1]
            sin_value = met[:, 2]
            dx = self.wake_geometry[:, :, 0][None, :, :].to(device=met.device, dtype=met.dtype)
            dy = self.wake_geometry[:, :, 1][None, :, :].to(device=met.device, dtype=met.dtype)
            distance = self.wake_geometry[:, :, 2][None, :, :].abs().clamp_min(1e-6).to(device=met.device, dtype=met.dtype)
            d_parallel = (dx * cos_value[:, None, None] + dy * sin_value[:, None, None]) / distance
            d_cross = (-dx * sin_value[:, None, None] + dy * cos_value[:, None, None]) / distance
            gate = torch.sigmoid(self.wake_kappa * d_parallel) * torch.exp(-F.relu(d_parallel) / self.wake_lambda_x) * torch.exp(-torch.square(d_cross / self.wake_lambda_y))
            diagonal = self.edge_diagonal_mask[..., 0].to(device=met.device)
            gate = gate.masked_fill(diagonal, 0.0)
            return torch.stack((d_parallel, d_cross, gate), dim=-1)

        def _aggregate(self, source_summary, g, met, calendar, edge_net, gate_net, *, static=None):
            batch_size = source_summary.shape[0]
            static_nodes = self._static(batch_size) if static is None else static
            dst_static = static_nodes[:, :, None, :].expand(batch_size, self.node_count, self.node_count, -1)
            src_static = static_nodes[:, None, :, :].expand(batch_size, self.node_count, self.node_count, -1)
            pairwise = self.pairwise_features[None].to(device=source_summary.device, dtype=source_summary.dtype).expand(batch_size, -1, -1, -1)
            dynamic = self._wake(met)
            src_summary = source_summary[:, None, :, :].expand(batch_size, self.node_count, -1, -1)
            global_context = g[:, None, None, :].expand(batch_size, self.node_count, self.node_count, -1)
            met_context = met[:, None, None, :].expand(batch_size, self.node_count, self.node_count, -1)
            cal_context = calendar[:, None, None, :].expand(batch_size, self.node_count, self.node_count, -1)
            edge_inputs = torch.cat((dst_static, src_static, pairwise, dynamic, src_summary, global_context, met_context, cal_context), dim=-1)
            messages = edge_net(edge_inputs)
            gates = torch.sigmoid(gate_net(edge_inputs))
            diagonal = self.edge_diagonal_mask.to(device=messages.device)
            messages = messages.masked_fill(diagonal, 0.0)
            gates = gates.masked_fill(diagonal, 0.0)
            return (messages * gates).sum(dim=2) / gates.sum(dim=2).clamp_min(1e-6)

        def _correct(self, z, h, g, local_obs, context, *, static=None):
            batch_size = local_obs.shape[0]
            static_nodes = self._static(batch_size) if static is None else static
            values = local_obs[:, :, _LOCAL_VALUE_START : _LOCAL_VALUE_START + LOCAL_VALUE_COUNT]
            masks = local_obs[:, :, _LOCAL_MASK_START : _LOCAL_MASK_START + LOCAL_MASK_COUNT]
            deltas = local_obs[:, :, _LOCAL_DELTA_START : _LOCAL_DELTA_START + LOCAL_DELTA_COUNT]
            available = 1.0 - masks.clamp(0.0, 1.0)
            node_available = (available.sum(dim=-1) > 0).to(local_obs.dtype)
            prior_obs = self._decode_obs(z, h, g, posterior=False, static=static_nodes)
            obs_encoding = self.obs_encoder(torch.cat((local_obs, static_nodes), dim=-1))
            innovation = self.innovation_encoder(torch.cat((prior_obs, available * (values - prior_obs), available, deltas, obs_encoding), dim=-1))
            source = self.update_source(torch.cat((local_obs, self._node_state(z, h), static_nodes), dim=-1))
            met = context[:, _CONTEXT_SITE_SUMMARY_START : _CONTEXT_SITE_SUMMARY_START + len(SITE_SUMMARY_FEATURE_NAMES)]
            calendar = context[:, _CONTEXT_CALENDAR_START:]
            edge_message = self._aggregate(source, g, met, calendar, self.update_edge, self.update_gate, static=static_nodes)
            g_nodes = g[:, None, :].expand(-1, self.node_count, -1)
            context_nodes = context[:, None, :].expand(-1, self.node_count, -1)
            z_next = self.update_z(torch.cat((innovation, edge_message, g_nodes, context_nodes, static_nodes), dim=-1).reshape(batch_size * self.node_count, -1), z.reshape(batch_size * self.node_count, -1)).reshape(batch_size, self.node_count, self.z_dim)
            h_next = self.update_h(torch.cat((z_next, innovation, edge_message, g_nodes, context_nodes, static_nodes), dim=-1).reshape(batch_size * self.node_count, -1), h.reshape(batch_size * self.node_count, -1)).reshape(batch_size, self.node_count, self.h_dim)
            update_mask = node_available.unsqueeze(-1)
            z_post = torch.where(update_mask > 0, z_next, z)
            h_post = torch.where(update_mask > 0, h_next, h)
            pooled = (self._node_state(z_post, h_post) * update_mask).sum(dim=1) / update_mask.sum(dim=1).clamp_min(1.0)
            fallback = self._node_state(z_post, h_post).mean(dim=1)
            pooled = torch.where((update_mask.sum(dim=1) > 0).expand_as(pooled), pooled, fallback)
            g_post = self.global_update(torch.cat((pooled, context), dim=-1), g)
            post_obs = self._decode_obs(z_post, h_post, g_post, posterior=True, static=static_nodes)
            innovation_penalty = torch.square(z_post - z).mean() + 0.5 * torch.square(h_post - h).mean() + 0.25 * torch.square(g_post - g).mean()
            return z_post, h_post, g_post, prior_obs, post_obs, innovation_penalty

        def _transition(self, z, h, g, calendar, *, static=None):
            batch_size = z.shape[0]
            static_nodes = self._static(batch_size) if static is None else static
            met = self._met(g, calendar)
            g_nodes = g[:, None, :].expand(-1, self.node_count, -1)
            source = self.transition_source(torch.cat((self._node_state(z, h), static_nodes, g_nodes), dim=-1))
            edge_message = self._aggregate(source, g, met, calendar, self.transition_edge, self.transition_gate, static=static_nodes)
            met_nodes = met[:, None, :].expand(-1, self.node_count, -1)
            calendar_nodes = calendar[:, None, :].expand(-1, self.node_count, -1)
            z_next = self.transition_z(torch.cat((edge_message, g_nodes, met_nodes, calendar_nodes, static_nodes, h), dim=-1).reshape(batch_size * self.node_count, -1), z.reshape(batch_size * self.node_count, -1)).reshape(batch_size, self.node_count, self.z_dim)
            h_next = self.transition_h(torch.cat((z_next, edge_message, g_nodes, met_nodes, calendar_nodes, static_nodes), dim=-1).reshape(batch_size * self.node_count, -1), h.reshape(batch_size * self.node_count, -1)).reshape(batch_size, self.node_count, self.h_dim)
            pooled = self._node_state(z_next, h_next).mean(dim=1)
            g_next = self.global_transition(torch.cat((pooled, met, calendar), dim=-1), g)
            return z_next, h_next, g_next, met

        def _forecast(self, z, h, g, tau_index: int, *, static=None):
            batch_size = z.shape[0]
            static_nodes = self._static(batch_size) if static is None else static
            tau = self.tau_embedding(torch.full((batch_size,), tau_index, dtype=torch.long, device=z.device))
            raw = self.forecast_decoder(torch.cat((self._node_state(z, h), g[:, None, :].expand(-1, self.node_count, -1), static_nodes, tau[:, None, :].expand(-1, self.node_count, -1)), dim=-1))
            return (1.0 + self.bounded_output_epsilon) * torch.sigmoid(raw)

        def _farm_prediction(self, forecast):
            return self.farm_bias + self.farm_scale * (forecast.squeeze(-1) * self.farm_weights[None]).sum(dim=1, keepdim=True)

        def forward(self, local_history, context_history, context_future):
            batch_size = local_history.shape[0]
            static = self._static(batch_size)
            z, h, g = self._initial_states(
                batch_size,
                dtype=local_history.dtype,
                device=local_history.device,
                static_base=static[0],
            )
            hist_prior = []
            hist_post = []
            innovation_terms = []
            for history_index in range(local_history.shape[1]):
                if history_index > 0:
                    z, h, g, _met = self._transition(
                        z,
                        h,
                        g,
                        context_history[:, history_index, _CONTEXT_CALENDAR_START:],
                        static=static,
                    )
                z, h, g, prior_obs, post_obs, innovation = self._correct(
                    z,
                    h,
                    g,
                    local_history[:, history_index],
                    context_history[:, history_index],
                    static=static,
                )
                hist_prior.append(prior_obs)
                hist_post.append(post_obs)
                innovation_terms.append(innovation)
            forecasts = []
            farms = []
            mets = []
            for horizon_index in range(context_future.shape[1]):
                z, h, g, met = self._transition(z, h, g, context_future[:, horizon_index], static=static)
                forecast = self._forecast(z, h, g, horizon_index, static=static)
                forecasts.append(forecast)
                farms.append(self._farm_prediction(forecast))
                mets.append(met)
            return ModelOutputs(
                future_predictions=torch.stack(forecasts, dim=1),
                hist_prior_observations=torch.stack(hist_prior, dim=1),
                hist_post_observations=torch.stack(hist_post, dim=1),
                future_farm_predictions=torch.stack(farms, dim=1),
                future_met_predictions=torch.stack(mets, dim=1),
                innovation_regularization=torch.stack([term.reshape(()) for term in innovation_terms]).mean(),
            )

else:

    class FeedForward:  # pragma: no cover
        def __init__(self, *_args, **_kwargs) -> None:
            _require_torch()

    class WorldModelStateSpace:  # pragma: no cover
        def __init__(self, *_args, **_kwargs) -> None:
            _require_torch()


def build_model(
    *,
    node_count: int,
    local_input_channels: int,
    context_history_channels: int,
    context_future_channels: int,
    static_tensor: np.ndarray,
    turbine_indices: np.ndarray,
    pairwise_tensor: np.ndarray,
    wake_geometry_tensor: np.ndarray,
    z_dim: int,
    h_dim: int,
    global_state_dim: int,
    obs_encoding_dim: int,
    innovation_dim: int,
    source_summary_dim: int,
    edge_message_dim: int,
    edge_hidden_dim: int,
    tau_embed_dim: int,
    met_summary_dim: int,
    turbine_embed_dim: int,
    forecast_steps: int,
    dropout: float,
    wake_lambda_x: float,
    wake_lambda_y: float,
    wake_kappa: float,
    bounded_output_epsilon: float,
):
    _require_torch()
    return WorldModelStateSpace(
        node_count=node_count,
        local_input_channels=local_input_channels,
        context_history_channels=context_history_channels,
        context_future_channels=context_future_channels,
        static_tensor=static_tensor,
        turbine_indices=turbine_indices,
        pairwise_tensor=pairwise_tensor,
        wake_geometry_tensor=wake_geometry_tensor,
        z_dim=z_dim,
        h_dim=h_dim,
        global_state_dim=global_state_dim,
        obs_encoding_dim=obs_encoding_dim,
        innovation_dim=innovation_dim,
        source_summary_dim=source_summary_dim,
        edge_message_dim=edge_message_dim,
        edge_hidden_dim=edge_hidden_dim,
        tau_embed_dim=tau_embed_dim,
        met_summary_dim=met_summary_dim,
        turbine_embed_dim=turbine_embed_dim,
        forecast_steps=forecast_steps,
        dropout=dropout,
        wake_lambda_x=wake_lambda_x,
        wake_lambda_y=wake_lambda_y,
        wake_kappa=wake_kappa,
        bounded_output_epsilon=bounded_output_epsilon,
    )


def initialize_model_parameters(model) -> None:
    _, resolved_nn, _, _, _ = _require_torch()
    for name, parameter in model.named_parameters():
        if name.endswith("farm_bias") or name.endswith("farm_scale"):
            continue
        if parameter.dim() > 1:
            resolved_nn.init.xavier_uniform_(parameter)
        else:
            resolved_nn.init.uniform_(parameter, -0.02, 0.02)
    for module in model.modules():
        if isinstance(module, resolved_nn.LayerNorm):
            resolved_nn.init.ones_(module.weight)
            resolved_nn.init.zeros_(module.bias)


def _set_random_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    resolved_torch, _, _, _, _ = _require_torch()
    resolved_torch.manual_seed(seed)
    if resolved_torch.cuda.is_available():
        resolved_torch.cuda.manual_seed_all(seed)


def _amp_context(*, torch_module, device: str, enabled: bool):
    if enabled and device == "cuda":
        return torch_module.autocast(device_type="cuda")
    return nullcontext()


def masked_huber_loss(predictions, targets, valid_mask, *, torch_module, delta: float, allow_zero: bool = False):
    mask = valid_mask.to(device=predictions.device, dtype=predictions.dtype)
    count = mask.sum()
    if float(count.item()) <= 0:
        if allow_zero:
            return predictions.new_tensor(0.0)
        raise ValueError("masked_huber_loss received a batch with zero valid targets.")
    errors = predictions - targets
    abs_errors = torch_module.abs(errors)
    huber = torch_module.where(abs_errors <= delta, 0.5 * torch_module.square(errors) / delta, abs_errors - 0.5 * delta)
    return (huber * mask).sum() / count


def _compute_training_losses(outputs: ModelOutputs, batch_local_history, batch_targets, batch_target_valid_mask, batch_met_targets, batch_met_valid_mask, batch_farm_targets, batch_farm_valid_mask, *, hist_recon_loss_weight: float, farm_loss_weight: float, met_loss_weight: float, innovation_loss_weight: float, torch_module) -> tuple[Any, dict[str, float]]:
    forecast_loss = masked_huber_loss(outputs.future_predictions, batch_targets, batch_target_valid_mask, torch_module=torch_module, delta=0.03)
    history_targets = batch_local_history[:, DEFAULT_HISTORY_RECON_BURN_IN:, :, _LOCAL_VALUE_START : _LOCAL_VALUE_START + LOCAL_VALUE_COUNT]
    history_valid = 1.0 - batch_local_history[:, DEFAULT_HISTORY_RECON_BURN_IN:, :, _LOCAL_MASK_START : _LOCAL_MASK_START + LOCAL_MASK_COUNT]
    prior_loss = masked_huber_loss(outputs.hist_prior_observations[:, DEFAULT_HISTORY_RECON_BURN_IN:], history_targets, history_valid, torch_module=torch_module, delta=0.05, allow_zero=True)
    post_loss = masked_huber_loss(outputs.hist_post_observations[:, DEFAULT_HISTORY_RECON_BURN_IN:], history_targets, history_valid, torch_module=torch_module, delta=0.05, allow_zero=True)
    hist_loss = 0.25 * prior_loss + 0.75 * post_loss
    farm_loss = masked_huber_loss(outputs.future_farm_predictions, batch_farm_targets, batch_farm_valid_mask, torch_module=torch_module, delta=0.03, allow_zero=True)
    met_loss = masked_huber_loss(outputs.future_met_predictions, batch_met_targets, batch_met_valid_mask, torch_module=torch_module, delta=0.05, allow_zero=True)
    innovation_loss = outputs.innovation_regularization
    total_loss = forecast_loss + hist_recon_loss_weight * hist_loss + farm_loss_weight * farm_loss + met_loss_weight * met_loss + innovation_loss_weight * innovation_loss
    return total_loss, {
        "total": float(total_loss.item()),
        "forecast": float(forecast_loss.item()),
        "hist_recon": float(hist_loss.item()),
        "farm": float(farm_loss.item()),
        "met": float(met_loss.item()),
        "innovation": float(innovation_loss.item()),
    }


def _safe_divide(numerator: float, denominator: int) -> float:
    return float("nan") if denominator <= 0 else numerator / denominator


def _safe_rmse(squared_error_sum: float, denominator: int) -> float:
    return float("nan") if denominator <= 0 else math.sqrt(squared_error_sum / denominator)


def evaluate_model(model, loader, *, device: str, rated_power_kw: float, forecast_steps: int, amp_enabled: bool = False, progress_label: str | None = None) -> EvaluationMetrics:
    resolved_torch, _, _, _, _ = _require_torch()
    window_count = 0
    prediction_count = 0
    abs_error_sum_kw = squared_error_sum_kw = abs_error_sum_pu = squared_error_sum_pu = 0.0
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
            for batch in loader:
                batch_local_history, batch_context_history, batch_context_future, batch_targets, batch_target_valid_mask = batch[:5]
                batch_local_history = batch_local_history.to(device=device, dtype=resolved_torch.float32, non_blocking=device == "cuda")
                batch_context_history = batch_context_history.to(device=device, dtype=resolved_torch.float32, non_blocking=device == "cuda")
                batch_context_future = batch_context_future.to(device=device, dtype=resolved_torch.float32, non_blocking=device == "cuda")
                batch_targets = batch_targets.to(device=device, dtype=resolved_torch.float32, non_blocking=device == "cuda")
                batch_target_valid_mask = batch_target_valid_mask.to(device=device, dtype=resolved_torch.float32, non_blocking=device == "cuda")
                with _amp_context(torch_module=resolved_torch, device=device, enabled=amp_enabled):
                    predictions = model(batch_local_history, batch_context_history, batch_context_future).future_predictions.float()
                errors_pu = predictions - batch_targets
                errors_kw = errors_pu * rated_power_kw
                valid = batch_target_valid_mask
                valid_np = valid.detach().cpu().numpy().astype(np.float64, copy=False)
                batch_window_count = int(batch_local_history.shape[0])
                batch_prediction_count = int(valid.sum().item())
                window_count += batch_window_count
                prediction_count += batch_prediction_count
                abs_kw = resolved_torch.abs(errors_kw) * valid
                sq_kw = resolved_torch.square(errors_kw) * valid
                abs_pu = resolved_torch.abs(errors_pu) * valid
                sq_pu = resolved_torch.square(errors_pu) * valid
                abs_error_sum_kw += float(abs_kw.sum().item())
                squared_error_sum_kw += float(sq_kw.sum().item())
                abs_error_sum_pu += float(abs_pu.sum().item())
                squared_error_sum_pu += float(sq_pu.sum().item())
                horizon_window_count += batch_window_count
                horizon_prediction_count += valid_np.sum(axis=(0, 2, 3)).astype(np.int64, copy=False)
                horizon_abs_error_sum_kw += abs_kw.sum(dim=(0, 2, 3)).detach().cpu().numpy().astype(np.float64, copy=False)
                horizon_squared_error_sum_kw += sq_kw.sum(dim=(0, 2, 3)).detach().cpu().numpy().astype(np.float64, copy=False)
                horizon_abs_error_sum_pu += abs_pu.sum(dim=(0, 2, 3)).detach().cpu().numpy().astype(np.float64, copy=False)
                horizon_squared_error_sum_pu += sq_pu.sum(dim=(0, 2, 3)).detach().cpu().numpy().astype(np.float64, copy=False)
                progress.update(1)
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
        horizon_mae_kw=np.asarray([_safe_divide(horizon_abs_error_sum_kw[i], int(horizon_prediction_count[i])) for i in range(forecast_steps)]),
        horizon_rmse_kw=np.asarray([_safe_rmse(horizon_squared_error_sum_kw[i], int(horizon_prediction_count[i])) for i in range(forecast_steps)]),
        horizon_mae_pu=np.asarray([_safe_divide(horizon_abs_error_sum_pu[i], int(horizon_prediction_count[i])) for i in range(forecast_steps)]),
        horizon_rmse_pu=np.asarray([_safe_rmse(horizon_squared_error_sum_pu[i], int(horizon_prediction_count[i])) for i in range(forecast_steps)]),
    )


@dataclass(frozen=True)
class ResumePaths:
    slot_dir: Path
    state_path: Path
    partial_results_path: Path
    training_history_path: Path
    checkpoints_dir: Path


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


def _job_identity(*, dataset_id: str, model_variant: str, feature_protocol_id: str) -> dict[str, str]:
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


def sort_result_frame(frame: pl.DataFrame) -> pl.DataFrame:
    if frame.is_empty():
        return frame.select(_RESULT_COLUMNS)
    return (
        frame.select(_RESULT_COLUMNS)
        .with_columns(
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


def _load_resume_state(paths: ResumePaths) -> dict[str, object]:
    payload = json.loads(paths.state_path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != _RUN_STATE_SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported {FAMILY_ID} resume state schema at {paths.state_path}: "
            f"{payload.get('schema_version')!r}."
        )
    status = payload.get("status")
    if status not in {"running", "complete"}:
        raise ValueError(f"Unsupported {FAMILY_ID} resume state status at {paths.state_path}: {status!r}.")
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
    return sort_result_frame(frame.select(_RESULT_COLUMNS))


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
    return sort_training_history_frame(frame.select(_TRAINING_HISTORY_COLUMNS))


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
    _job_checkpoint_path(paths, dataset_id=dataset_id, model_variant=model_variant).unlink(missing_ok=True)


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
    z_dim: int | None,
    h_dim: int | None,
    global_state_dim: int | None,
    obs_encoding_dim: int | None,
    innovation_dim: int | None,
    source_summary_dim: int | None,
    edge_message_dim: int | None,
    edge_hidden_dim: int | None,
    tau_embed_dim: int | None,
    met_summary_dim: int | None,
    turbine_embed_dim: int | None,
    dropout: float | None,
    grad_clip_norm: float | None,
    hist_recon_loss_weight: float | None,
    farm_loss_weight: float | None,
    met_loss_weight: float | None,
    innovation_loss_weight: float | None,
    weight_decay: float | None,
    wake_lambda_x: float | None,
    wake_lambda_y: float | None,
    wake_kappa: float | None,
    bounded_output_epsilon: float | None,
    num_workers: int | None,
    tensorboard_root: str | Path | None,
    disable_tensorboard: bool,
) -> dict[str, object]:
    resolved_device = resolve_device(device)
    return {
        "dataset_ids": list(dataset_ids),
        "variant_names": [spec.model_variant for spec in variant_specs],
        "device": resolved_device,
        "seed": seed,
        "max_train_origins": max_train_origins,
        "max_eval_origins": max_eval_origins,
        "eval_batch_size": eval_batch_size,
        "num_workers": resolve_loader_num_workers(device=resolved_device, num_workers=num_workers),
        "tensorboard_root": None if tensorboard_root is None else str(Path(tensorboard_root).expanduser().resolve()),
        "tensorboard_enabled": not disable_tensorboard,
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
                            z_dim=z_dim,
                            h_dim=h_dim,
                            global_state_dim=global_state_dim,
                            obs_encoding_dim=obs_encoding_dim,
                            innovation_dim=innovation_dim,
                            source_summary_dim=source_summary_dim,
                            edge_message_dim=edge_message_dim,
                            edge_hidden_dim=edge_hidden_dim,
                            tau_embed_dim=tau_embed_dim,
                            met_summary_dim=met_summary_dim,
                            turbine_embed_dim=turbine_embed_dim,
                            dropout=dropout,
                            grad_clip_norm=grad_clip_norm,
                            hist_recon_loss_weight=hist_recon_loss_weight,
                            farm_loss_weight=farm_loss_weight,
                            met_loss_weight=met_loss_weight,
                            innovation_loss_weight=innovation_loss_weight,
                            weight_decay=weight_decay,
                            wake_lambda_x=wake_lambda_x,
                            wake_lambda_y=wake_lambda_y,
                            wake_kappa=wake_kappa,
                            bounded_output_epsilon=bounded_output_epsilon,
                        )
                    ),
                }
                for spec in variant_specs
            }
            for dataset_id in dataset_ids
        },
    }


def _make_grad_scaler(torch_module, *, enabled: bool):
    if hasattr(torch_module, "amp") and hasattr(torch_module.amp, "GradScaler"):
        try:
            return torch_module.amp.GradScaler("cuda", enabled=enabled)
        except TypeError:
            return torch_module.amp.GradScaler(enabled=enabled)
    return torch_module.cuda.amp.GradScaler(enabled=enabled)


def _move_batch_to_device(batch, *, torch_module, device: str):
    return tuple(
        item.to(device=device, dtype=torch_module.float32, non_blocking=device == "cuda")
        for item in batch
    )


def _forecast_valid_fraction(*, prediction_count: int, window_count: int, forecast_steps: int, node_count: int) -> float:
    denominator = int(window_count) * int(forecast_steps) * int(node_count)
    if denominator <= 0:
        return math.nan
    return float(prediction_count) / float(denominator)


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
    z_dim: int = DEFAULT_Z_DIM,
    h_dim: int = DEFAULT_H_DIM,
    global_state_dim: int = DEFAULT_GLOBAL_STATE_DIM,
    obs_encoding_dim: int = DEFAULT_OBS_ENCODING_DIM,
    innovation_dim: int = DEFAULT_INNOVATION_DIM,
    source_summary_dim: int = DEFAULT_SOURCE_SUMMARY_DIM,
    edge_message_dim: int = DEFAULT_EDGE_MESSAGE_DIM,
    edge_hidden_dim: int = DEFAULT_EDGE_HIDDEN_DIM,
    tau_embed_dim: int = DEFAULT_TAU_EMBED_DIM,
    met_summary_dim: int = DEFAULT_MET_SUMMARY_DIM,
    turbine_embed_dim: int = DEFAULT_TURBINE_EMBED_DIM,
    dropout: float = DEFAULT_DROPOUT,
    grad_clip_norm: float = DEFAULT_GRAD_CLIP_NORM,
    hist_recon_loss_weight: float = DEFAULT_HIST_RECON_LOSS_WEIGHT,
    farm_loss_weight: float = DEFAULT_FARM_LOSS_WEIGHT,
    met_loss_weight: float = DEFAULT_MET_LOSS_WEIGHT,
    innovation_loss_weight: float = DEFAULT_INNOVATION_LOSS_WEIGHT,
    weight_decay: float = DEFAULT_WEIGHT_DECAY,
    wake_lambda_x: float = DEFAULT_WAKE_LAMBDA_X,
    wake_lambda_y: float = DEFAULT_WAKE_LAMBDA_Y,
    wake_kappa: float = DEFAULT_WAKE_KAPPA,
    bounded_output_epsilon: float = DEFAULT_BOUNDED_OUTPUT_EPSILON,
    num_workers: int | None = None,
    checkpoint_path: str | Path | None = None,
    training_history_path: str | Path | None = None,
    resume_from_checkpoint: bool = False,
    progress_label: str | None = None,
    tensorboard_writer=None,
) -> TrainingOutcome:
    resolved_torch, _, _, _, _ = _require_torch()
    _set_random_seed(seed)
    resolved_device = resolve_device(device)
    world_model_base._configure_torch_runtime(device=resolved_device, torch_module=resolved_torch)
    amp_enabled = resolved_device == "cuda"
    resolved_eval_batch_size = resolve_eval_batch_size(
        batch_size,
        device=resolved_device,
        eval_batch_size=eval_batch_size,
    )
    model = build_model(
        node_count=prepared_dataset.node_count,
        local_input_channels=prepared_dataset.local_input_channels,
        context_history_channels=prepared_dataset.context_history_channels,
        context_future_channels=prepared_dataset.context_future_channels,
        static_tensor=prepared_dataset.static_tensor,
        turbine_indices=prepared_dataset.turbine_indices,
        pairwise_tensor=prepared_dataset.pairwise_tensor,
        wake_geometry_tensor=prepared_dataset.wake_geometry_tensor,
        z_dim=z_dim,
        h_dim=h_dim,
        global_state_dim=global_state_dim,
        obs_encoding_dim=obs_encoding_dim,
        innovation_dim=innovation_dim,
        source_summary_dim=source_summary_dim,
        edge_message_dim=edge_message_dim,
        edge_hidden_dim=edge_hidden_dim,
        tau_embed_dim=tau_embed_dim,
        met_summary_dim=met_summary_dim,
        turbine_embed_dim=turbine_embed_dim,
        forecast_steps=prepared_dataset.forecast_steps,
        dropout=dropout,
        wake_lambda_x=wake_lambda_x,
        wake_lambda_y=wake_lambda_y,
        wake_kappa=wake_kappa,
        bounded_output_epsilon=bounded_output_epsilon,
    ).to(device=resolved_device)
    initialize_model_parameters(model)
    optimizer = resolved_torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scaler = _make_grad_scaler(resolved_torch, enabled=amp_enabled)
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
    best_val_mae_pu = float("inf")
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
            if "scaler_state_dict" in checkpoint_payload:
                scaler.load_state_dict(checkpoint_payload["scaler_state_dict"])
            best_state = checkpoint_payload.get("best_state_dict")
            best_epoch = int(checkpoint_payload["best_epoch"])
            best_val_rmse_pu = float(checkpoint_payload["best_val_rmse_pu"])
            best_val_mae_pu = float(checkpoint_payload.get("best_val_mae_pu", float("inf")))
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
                    best_val_mae_pu=best_val_mae_pu,
                    device=resolved_device,
                    amp_enabled=amp_enabled,
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
                "forecast": 0.0,
                "hist_recon": 0.0,
                "farm": 0.0,
                "met": 0.0,
                "innovation": 0.0,
            }
            train_loss_weight = 0
            train_target_valid_count = 0.0
            train_target_total_count = 0
            train_batch_count = 0
            try:
                for raw_batch in train_loader:
                    (
                        batch_local_history,
                        batch_context_history,
                        batch_context_future,
                        batch_targets,
                        batch_target_valid_mask,
                        batch_met_targets,
                        batch_met_valid_mask,
                        batch_farm_targets,
                        batch_farm_valid_mask,
                    ) = _move_batch_to_device(raw_batch, torch_module=resolved_torch, device=resolved_device)
                    optimizer.zero_grad(set_to_none=True)
                    with _amp_context(torch_module=resolved_torch, device=resolved_device, enabled=amp_enabled):
                        outputs = model(batch_local_history, batch_context_history, batch_context_future)
                        loss, latest_losses = _compute_training_losses(
                            outputs,
                            batch_local_history,
                            batch_targets,
                            batch_target_valid_mask,
                            batch_met_targets,
                            batch_met_valid_mask,
                            batch_farm_targets,
                            batch_farm_valid_mask,
                            hist_recon_loss_weight=hist_recon_loss_weight,
                            farm_loss_weight=farm_loss_weight,
                            met_loss_weight=met_loss_weight,
                            innovation_loss_weight=innovation_loss_weight,
                            torch_module=resolved_torch,
                        )
                    if amp_enabled:
                        scaler.scale(loss).backward()
                        if grad_clip_norm > 0:
                            scaler.unscale_(optimizer)
                            resolved_torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        loss.backward()
                        if grad_clip_norm > 0:
                            resolved_torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
                        optimizer.step()
                    batch_window_count = int(batch_local_history.shape[0])
                    for loss_name, loss_value in latest_losses.items():
                        loss_sums[loss_name] += float(loss_value) * batch_window_count
                    train_loss_weight += batch_window_count
                    train_target_valid_count += float(batch_target_valid_mask.sum().item())
                    train_target_total_count += int(batch_target_valid_mask.numel())
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
                amp_enabled=amp_enabled,
                progress_label=f"{progress_label or prepared_dataset.dataset_id} val e{epoch_index}",
            )
            val_rmse_pu = float(val_metrics.rmse_pu)
            val_mae_pu = float(val_metrics.mae_pu)
            is_best_epoch = False
            if val_rmse_pu < best_val_rmse_pu - 1e-12:
                best_val_rmse_pu = val_rmse_pu
                best_val_mae_pu = val_mae_pu
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
            train_valid_fraction = (
                train_target_valid_count / float(train_target_total_count)
                if train_target_total_count > 0
                else math.nan
            )
            val_valid_fraction = _forecast_valid_fraction(
                prediction_count=val_metrics.prediction_count,
                window_count=val_metrics.window_count,
                forecast_steps=prepared_dataset.forecast_steps,
                node_count=prepared_dataset.node_count,
            )
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
                        "train_forecast_loss_mean": loss_means["forecast"],
                        "train_forecast_loss_last": latest_losses.get("forecast"),
                        "train_hist_recon_loss_mean": loss_means["hist_recon"],
                        "train_hist_recon_loss_last": latest_losses.get("hist_recon"),
                        "train_farm_loss_mean": loss_means["farm"],
                        "train_farm_loss_last": latest_losses.get("farm"),
                        "train_met_loss_mean": loss_means["met"],
                        "train_met_loss_last": latest_losses.get("met"),
                        "train_innovation_loss_mean": loss_means["innovation"],
                        "train_innovation_loss_last": latest_losses.get("innovation"),
                        "val_mae_pu": val_mae_pu,
                        "val_rmse_pu": val_rmse_pu,
                        "best_val_mae_pu": best_val_mae_pu,
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
                        "amp_enabled": amp_enabled,
                    },
                )
            _tensorboard_add_scalar(tensorboard_writer, "train/loss_mean", loss_means["total"], epoch_index)
            _tensorboard_add_scalar(tensorboard_writer, "train/loss_last", latest_losses.get("total"), epoch_index)
            _tensorboard_add_scalar(tensorboard_writer, "train/forecast_loss_mean", loss_means["forecast"], epoch_index)
            _tensorboard_add_scalar(tensorboard_writer, "train/forecast_loss_last", latest_losses.get("forecast"), epoch_index)
            _tensorboard_add_scalar(tensorboard_writer, "train/hist_recon_loss_mean", loss_means["hist_recon"], epoch_index)
            _tensorboard_add_scalar(tensorboard_writer, "train/hist_recon_loss_last", latest_losses.get("hist_recon"), epoch_index)
            _tensorboard_add_scalar(tensorboard_writer, "train/farm_loss_mean", loss_means["farm"], epoch_index)
            _tensorboard_add_scalar(tensorboard_writer, "train/farm_loss_last", latest_losses.get("farm"), epoch_index)
            _tensorboard_add_scalar(tensorboard_writer, "train/met_loss_mean", loss_means["met"], epoch_index)
            _tensorboard_add_scalar(tensorboard_writer, "train/met_loss_last", latest_losses.get("met"), epoch_index)
            _tensorboard_add_scalar(tensorboard_writer, "train/innovation_loss_mean", loss_means["innovation"], epoch_index)
            _tensorboard_add_scalar(tensorboard_writer, "train/innovation_loss_last", latest_losses.get("innovation"), epoch_index)
            _tensorboard_add_scalar(tensorboard_writer, "train/valid_fraction", train_valid_fraction, epoch_index)
            _tensorboard_add_scalar(tensorboard_writer, "train/batch_count", train_batch_count, epoch_index)
            _tensorboard_add_scalar(tensorboard_writer, "train/learning_rate", optimizer.param_groups[0]["lr"], epoch_index)
            _tensorboard_add_scalar(tensorboard_writer, "val/valid_fraction", val_valid_fraction, epoch_index)
            _tensorboard_add_scalar(tensorboard_writer, "best/val_mae_pu", best_val_mae_pu, epoch_index)
            _tensorboard_add_scalar(tensorboard_writer, "best/val_rmse_pu", best_val_rmse_pu, epoch_index)
            _tensorboard_add_scalar(
                tensorboard_writer,
                "early_stopping/epochs_without_improvement",
                epochs_without_improvement,
                epoch_index,
            )
            _tensorboard_add_scalar(
                tensorboard_writer,
                "early_stopping/patience_limit",
                early_stopping_patience,
                epoch_index,
            )
            _tensorboard_add_metrics(tensorboard_writer, "val", val_metrics, step=epoch_index)
            if resolved_checkpoint_path is not None:
                world_model_base._save_training_checkpoint(
                    resolved_checkpoint_path,
                    {
                        "schema_version": _TRAINING_CHECKPOINT_SCHEMA_VERSION,
                        "job": checkpoint_job_identity,
                        "seed": seed,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "scaler_state_dict": scaler.state_dict(),
                        "best_state_dict": best_state,
                        "best_epoch": best_epoch,
                        "best_val_rmse_pu": best_val_rmse_pu,
                        "best_val_mae_pu": best_val_mae_pu,
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
        best_val_mae_pu=best_val_mae_pu,
        device=resolved_device,
        amp_enabled=amp_enabled,
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


def _window_bounds(windows: world_model_base.FarmWindowDescriptorIndex) -> tuple[str | None, str | None]:
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
        "context_history_channels": prepared_dataset.context_history_channels,
        "context_future_channels": prepared_dataset.context_future_channels,
        "static_feature_count": prepared_dataset.static_feature_count,
        "pairwise_feature_count": prepared_dataset.pairwise_feature_count,
        "z_dim": profile.z_dim,
        "h_dim": profile.h_dim,
        "global_state_dim": profile.global_state_dim,
        "source_summary_dim": profile.source_summary_dim,
        "met_summary_dim": profile.met_summary_dim,
        "wake_lambda_x": profile.wake_lambda_x,
        "wake_lambda_y": profile.wake_lambda_y,
        "wake_kappa": profile.wake_kappa,
        "bounded_output_epsilon": profile.bounded_output_epsilon,
        "hist_recon_loss_weight": profile.hist_recon_loss_weight,
        "farm_loss_weight": profile.farm_loss_weight,
        "met_loss_weight": profile.met_loss_weight,
        "innovation_loss_weight": profile.innovation_loss_weight,
        "weight_decay": profile.weight_decay,
        "amp_enabled": training_outcome.amp_enabled,
        "device": training_outcome.device,
        "runtime_seconds": round(runtime_seconds, 6),
        "train_window_count": len(prepared_dataset.train_windows),
        "val_window_count": len(prepared_dataset.val_rolling_windows),
        "test_window_count": len(prepared_dataset.test_rolling_windows),
        "best_epoch": training_outcome.best_epoch,
        "epochs_ran": training_outcome.epochs_ran,
        "best_val_rmse_pu": training_outcome.best_val_rmse_pu,
        "best_val_mae_pu": training_outcome.best_val_mae_pu,
        "seed": seed,
        "batch_size": profile.batch_size,
        "learning_rate": profile.learning_rate,
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
    z_dim: int = DEFAULT_Z_DIM,
    h_dim: int = DEFAULT_H_DIM,
    global_state_dim: int = DEFAULT_GLOBAL_STATE_DIM,
    obs_encoding_dim: int = DEFAULT_OBS_ENCODING_DIM,
    innovation_dim: int = DEFAULT_INNOVATION_DIM,
    source_summary_dim: int = DEFAULT_SOURCE_SUMMARY_DIM,
    edge_message_dim: int = DEFAULT_EDGE_MESSAGE_DIM,
    edge_hidden_dim: int = DEFAULT_EDGE_HIDDEN_DIM,
    tau_embed_dim: int = DEFAULT_TAU_EMBED_DIM,
    met_summary_dim: int = DEFAULT_MET_SUMMARY_DIM,
    turbine_embed_dim: int = DEFAULT_TURBINE_EMBED_DIM,
    dropout: float = DEFAULT_DROPOUT,
    grad_clip_norm: float = DEFAULT_GRAD_CLIP_NORM,
    hist_recon_loss_weight: float = DEFAULT_HIST_RECON_LOSS_WEIGHT,
    farm_loss_weight: float = DEFAULT_FARM_LOSS_WEIGHT,
    met_loss_weight: float = DEFAULT_MET_LOSS_WEIGHT,
    innovation_loss_weight: float = DEFAULT_INNOVATION_LOSS_WEIGHT,
    weight_decay: float = DEFAULT_WEIGHT_DECAY,
    wake_lambda_x: float = DEFAULT_WAKE_LAMBDA_X,
    wake_lambda_y: float = DEFAULT_WAKE_LAMBDA_Y,
    wake_kappa: float = DEFAULT_WAKE_KAPPA,
    bounded_output_epsilon: float = DEFAULT_BOUNDED_OUTPUT_EPSILON,
    num_workers: int | None = None,
    checkpoint_path: str | Path | None = None,
    training_history_path: str | Path | None = None,
    resume_from_checkpoint: bool = False,
    tensorboard_log_dir: str | Path | None = None,
) -> list[dict[str, object]]:
    dataset_start = time.monotonic()
    progress_label = f"{prepared_dataset.dataset_id}/{prepared_dataset.model_variant}"
    resolved_profile = HyperparameterProfile(
        batch_size=batch_size,
        learning_rate=learning_rate,
        max_epochs=max_epochs,
        early_stopping_patience=early_stopping_patience,
        z_dim=z_dim,
        h_dim=h_dim,
        global_state_dim=global_state_dim,
        obs_encoding_dim=obs_encoding_dim,
        innovation_dim=innovation_dim,
        source_summary_dim=source_summary_dim,
        edge_message_dim=edge_message_dim,
        edge_hidden_dim=edge_hidden_dim,
        tau_embed_dim=tau_embed_dim,
        met_summary_dim=met_summary_dim,
        turbine_embed_dim=turbine_embed_dim,
        dropout=dropout,
        grad_clip_norm=grad_clip_norm,
        hist_recon_loss_weight=hist_recon_loss_weight,
        farm_loss_weight=farm_loss_weight,
        met_loss_weight=met_loss_weight,
        innovation_loss_weight=innovation_loss_weight,
        weight_decay=weight_decay,
        wake_lambda_x=wake_lambda_x,
        wake_lambda_y=wake_lambda_y,
        wake_kappa=wake_kappa,
        bounded_output_epsilon=bounded_output_epsilon,
    )
    resolved_device = resolve_device(device)
    resolved_eval_batch_size = resolve_eval_batch_size(
        batch_size,
        device=resolved_device,
        eval_batch_size=eval_batch_size,
    )
    writer = _open_tensorboard_writer(
        tensorboard_log_dir,
        dataset_id=prepared_dataset.dataset_id,
        model_variant=prepared_dataset.model_variant,
    )
    _tensorboard_log_run_config(
        writer,
        prepared_dataset=prepared_dataset,
        profile=resolved_profile,
        seed=seed,
        device=resolved_device,
        eval_batch_size=resolved_eval_batch_size,
        num_workers=resolve_loader_num_workers(device=resolved_device, num_workers=num_workers),
    )
    try:
        training_outcome = train_model(
            prepared_dataset,
            device=resolved_device,
            seed=seed,
            batch_size=batch_size,
            eval_batch_size=resolved_eval_batch_size,
            learning_rate=learning_rate,
            max_epochs=max_epochs,
            early_stopping_patience=early_stopping_patience,
            z_dim=z_dim,
            h_dim=h_dim,
            global_state_dim=global_state_dim,
            obs_encoding_dim=obs_encoding_dim,
            innovation_dim=innovation_dim,
            source_summary_dim=source_summary_dim,
            edge_message_dim=edge_message_dim,
            edge_hidden_dim=edge_hidden_dim,
            tau_embed_dim=tau_embed_dim,
            met_summary_dim=met_summary_dim,
            turbine_embed_dim=turbine_embed_dim,
            dropout=dropout,
            grad_clip_norm=grad_clip_norm,
            hist_recon_loss_weight=hist_recon_loss_weight,
            farm_loss_weight=farm_loss_weight,
            met_loss_weight=met_loss_weight,
            innovation_loss_weight=innovation_loss_weight,
            weight_decay=weight_decay,
            wake_lambda_x=wake_lambda_x,
            wake_lambda_y=wake_lambda_y,
            wake_kappa=wake_kappa,
            bounded_output_epsilon=bounded_output_epsilon,
            num_workers=num_workers,
            checkpoint_path=checkpoint_path,
            training_history_path=training_history_path,
            resume_from_checkpoint=resume_from_checkpoint,
            progress_label=progress_label,
            tensorboard_writer=writer,
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
                    amp_enabled=training_outcome.amp_enabled,
                    progress_label=f"{progress_label} {split_name}/{eval_protocol}",
                )
                evaluation_results.append((split_name, eval_protocol, windows, metrics))
                eval_progress.update(1)
                eval_progress.set_postfix_str(f"{split_name}/{eval_protocol}")
        finally:
            eval_progress.close()
        _tensorboard_log_final_evaluations(writer, evaluation_results, step=training_outcome.epochs_ran)
    finally:
        _close_tensorboard_writer(writer)
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
    z_dim: int | None = None,
    h_dim: int | None = None,
    global_state_dim: int | None = None,
    obs_encoding_dim: int | None = None,
    innovation_dim: int | None = None,
    source_summary_dim: int | None = None,
    edge_message_dim: int | None = None,
    edge_hidden_dim: int | None = None,
    tau_embed_dim: int | None = None,
    met_summary_dim: int | None = None,
    turbine_embed_dim: int | None = None,
    dropout: float | None = None,
    grad_clip_norm: float | None = None,
    hist_recon_loss_weight: float | None = None,
    farm_loss_weight: float | None = None,
    met_loss_weight: float | None = None,
    innovation_loss_weight: float | None = None,
    weight_decay: float | None = None,
    wake_lambda_x: float | None = None,
    wake_lambda_y: float | None = None,
    wake_kappa: float | None = None,
    bounded_output_epsilon: float | None = None,
    num_workers: int | None = None,
    tensorboard_log_dir: str | Path | None = None,
    disable_tensorboard: bool = False,
    resume: bool = False,
    force_rerun: bool = False,
    work_root: str | Path = _RUN_WORK_ROOT,
    dataset_loader: Callable[..., PreparedDataset] | None = None,
    job_runner: Callable[..., list[dict[str, object]]] | None = None,
) -> pl.DataFrame:
    resolved_dataset_ids = _validate_dataset_ids(dataset_ids)
    variant_specs = resolve_variant_specs(variant_names)
    runner = job_runner or execute_training_job
    output = _normalize_output_path(output_path)
    resume_paths = _resume_paths_for_output(output_path=output, work_root=work_root)
    resolved_tensorboard_root = resolve_tensorboard_root(
        output_path=output,
        work_root=work_root,
        tensorboard_log_dir=tensorboard_log_dir,
        disable_tensorboard=disable_tensorboard,
    )
    effective_config = _build_effective_config(
        dataset_ids=resolved_dataset_ids,
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
        z_dim=z_dim,
        h_dim=h_dim,
        global_state_dim=global_state_dim,
        obs_encoding_dim=obs_encoding_dim,
        innovation_dim=innovation_dim,
        source_summary_dim=source_summary_dim,
        edge_message_dim=edge_message_dim,
        edge_hidden_dim=edge_hidden_dim,
        tau_embed_dim=tau_embed_dim,
        met_summary_dim=met_summary_dim,
        turbine_embed_dim=turbine_embed_dim,
        dropout=dropout,
        grad_clip_norm=grad_clip_norm,
        hist_recon_loss_weight=hist_recon_loss_weight,
        farm_loss_weight=farm_loss_weight,
        met_loss_weight=met_loss_weight,
        innovation_loss_weight=innovation_loss_weight,
        weight_decay=weight_decay,
        wake_lambda_x=wake_lambda_x,
        wake_lambda_y=wake_lambda_y,
        wake_kappa=wake_kappa,
        bounded_output_epsilon=bounded_output_epsilon,
        num_workers=num_workers,
        tensorboard_root=resolved_tensorboard_root,
        disable_tensorboard=disable_tensorboard,
    )
    existing_state = _load_resume_state_if_exists(resume_paths)
    if resume and force_rerun:
        raise ValueError("--resume and --force-rerun are mutually exclusive.")
    if force_rerun:
        _reset_resume_slot(resume_paths, effective_config=effective_config)
        existing_state = None
    elif resume:
        if existing_state is None:
            raise ValueError(f"No resume state exists for output path {output}. Expected {resume_paths.state_path}.")
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
    total_jobs = len(resolved_dataset_ids) * len(variant_specs)
    if len(completed_job_keys) == total_jobs:
        results = _result_frame_from_rows(rows)
        _atomic_write_csv(output, results)
        _publish_training_history(resume_paths, output)
        _clear_checkpoint_dir(resume_paths)
        _write_resume_state(resume_paths, status="complete", effective_config=effective_config, active_job=None)
        return results
    resume_active_job = None if existing_state is None else existing_state.get("active_job")
    job_progress = _create_progress_bar(total=total_jobs, desc=f"{FAMILY_ID} jobs", leave=True)
    try:
        for dataset_id in resolved_dataset_ids:
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
                    z_dim=z_dim,
                    h_dim=h_dim,
                    global_state_dim=global_state_dim,
                    obs_encoding_dim=obs_encoding_dim,
                    innovation_dim=innovation_dim,
                    source_summary_dim=source_summary_dim,
                    edge_message_dim=edge_message_dim,
                    edge_hidden_dim=edge_hidden_dim,
                    tau_embed_dim=tau_embed_dim,
                    met_summary_dim=met_summary_dim,
                    turbine_embed_dim=turbine_embed_dim,
                    dropout=dropout,
                    grad_clip_norm=grad_clip_norm,
                    hist_recon_loss_weight=hist_recon_loss_weight,
                    farm_loss_weight=farm_loss_weight,
                    met_loss_weight=met_loss_weight,
                    innovation_loss_weight=innovation_loss_weight,
                    weight_decay=weight_decay,
                    wake_lambda_x=wake_lambda_x,
                    wake_lambda_y=wake_lambda_y,
                    wake_kappa=wake_kappa,
                    bounded_output_epsilon=bounded_output_epsilon,
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
                        z_dim=resolved_profile.z_dim,
                        h_dim=resolved_profile.h_dim,
                        global_state_dim=resolved_profile.global_state_dim,
                        obs_encoding_dim=resolved_profile.obs_encoding_dim,
                        innovation_dim=resolved_profile.innovation_dim,
                        source_summary_dim=resolved_profile.source_summary_dim,
                        edge_message_dim=resolved_profile.edge_message_dim,
                        edge_hidden_dim=resolved_profile.edge_hidden_dim,
                        tau_embed_dim=resolved_profile.tau_embed_dim,
                        met_summary_dim=resolved_profile.met_summary_dim,
                        turbine_embed_dim=resolved_profile.turbine_embed_dim,
                        dropout=resolved_profile.dropout,
                        grad_clip_norm=resolved_profile.grad_clip_norm,
                        hist_recon_loss_weight=resolved_profile.hist_recon_loss_weight,
                        farm_loss_weight=resolved_profile.farm_loss_weight,
                        met_loss_weight=resolved_profile.met_loss_weight,
                        innovation_loss_weight=resolved_profile.innovation_loss_weight,
                        weight_decay=resolved_profile.weight_decay,
                        wake_lambda_x=resolved_profile.wake_lambda_x,
                        wake_lambda_y=resolved_profile.wake_lambda_y,
                        wake_kappa=resolved_profile.wake_kappa,
                        bounded_output_epsilon=resolved_profile.bounded_output_epsilon,
                        num_workers=num_workers,
                        checkpoint_path=checkpoint_path,
                        training_history_path=resume_paths.training_history_path,
                        resume_from_checkpoint=resume_from_checkpoint,
                        tensorboard_log_dir=(
                            None
                            if resolved_tensorboard_root is None
                            else _tensorboard_job_log_dir(
                                resolved_tensorboard_root,
                                dataset_id=dataset_id,
                                model_variant=variant_spec.model_variant,
                            )
                        ),
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
        description="Run the world_model_state_space_v1 family on Kelmarsh."
    )
    parser.add_argument(
        "--dataset",
        action="append",
        choices=list(DEFAULT_DATASETS),
        dest="datasets",
        help="Limit execution to kelmarsh. Defaults to kelmarsh.",
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
        help="Limit execution to the active state-space world-model variant.",
    )
    parser.add_argument("--epochs", type=int, default=None, help="Maximum training epochs.")
    parser.add_argument(
        "--output-path",
        type=Path,
        default=_OUTPUT_PATH,
        help=f"Output CSV path. Defaults to {_OUTPUT_PATH}.",
    )
    parser.add_argument("--max-train-origins", type=int, default=None, help="Optional train-origin smoke limit.")
    parser.add_argument("--max-eval-origins", type=int, default=None, help="Optional val/test-origin smoke limit.")
    parser.add_argument("--batch-size", type=int, default=None, help="Training batch size.")
    parser.add_argument("--eval-batch-size", type=int, default=None, help="Evaluation batch size.")
    parser.add_argument("--learning-rate", type=float, default=None, help="AdamW learning rate.")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Random seed.")
    parser.add_argument("--patience", type=int, default=None, help="Early stopping patience in epochs.")
    parser.add_argument("--z-dim", type=int, default=None, help="Node latent z dimension.")
    parser.add_argument("--h-dim", type=int, default=None, help="Node recurrent h dimension.")
    parser.add_argument("--global-state-dim", type=int, default=None, help="Global state dimension.")
    parser.add_argument("--obs-encoding-dim", type=int, default=None, help="Observation encoding dimension.")
    parser.add_argument("--innovation-dim", type=int, default=None, help="Innovation encoding dimension.")
    parser.add_argument("--source-summary-dim", type=int, default=None, help="Source summary dimension.")
    parser.add_argument("--edge-message-dim", type=int, default=None, help="Edge message dimension.")
    parser.add_argument("--edge-hidden-dim", type=int, default=None, help="Edge MLP hidden dimension.")
    parser.add_argument("--tau-embed-dim", type=int, default=None, help="Forecast-step embedding dimension.")
    parser.add_argument("--met-summary-dim", type=int, default=None, help="Meteorology summary dimension.")
    parser.add_argument("--turbine-embed-dim", type=int, default=None, help="Turbine-index embedding dimension.")
    parser.add_argument("--dropout", type=float, default=None, help="Dropout applied inside MLP blocks.")
    parser.add_argument("--grad-clip-norm", type=float, default=None, help="Gradient clipping max norm; 0 disables.")
    parser.add_argument("--hist-recon-loss-weight", type=float, default=None, help="History reconstruction loss weight.")
    parser.add_argument("--farm-loss-weight", type=float, default=None, help="Farm consistency loss weight.")
    parser.add_argument("--met-loss-weight", type=float, default=None, help="Site-summary auxiliary loss weight.")
    parser.add_argument("--innovation-loss-weight", type=float, default=None, help="Innovation regularization weight.")
    parser.add_argument("--weight-decay", type=float, default=None, help="AdamW weight decay.")
    parser.add_argument("--wake-lambda-x", type=float, default=None, help="Wake prior downstream length scale.")
    parser.add_argument("--wake-lambda-y", type=float, default=None, help="Wake prior crosswind length scale.")
    parser.add_argument("--wake-kappa", type=float, default=None, help="Wake prior upstream/downstream gate scale.")
    parser.add_argument("--bounded-output-epsilon", type=float, default=None, help="Forecast upper bound slack.")
    parser.add_argument("--num-workers", type=int, default=None, help="DataLoader worker count.")
    parser.add_argument("--tensorboard-log-dir", type=Path, default=None, help="Optional TensorBoard log root.")
    parser.add_argument("--disable-tensorboard", action="store_true", help="Disable TensorBoard logging.")
    parser.add_argument(
        "--run-label",
        type=str,
        default=None,
        help=f"Optional label suffix for the formal run record under experiment/artifacts/runs/{FAMILY_ID}/.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help=f"Resume an interrupted {FAMILY_ID} invocation from the family .work directory.",
    )
    parser.add_argument(
        "--force-rerun",
        action="store_true",
        help="Discard any existing resume state for the resolved --output-path and start fresh.",
    )
    parser.add_argument(
        "--no-record-run",
        action="store_true",
        help="Skip writing a formal run record manifest under experiment/artifacts/runs/.",
    )
    return parser


def _resolved_record_hyperparameters(
    dataset_ids: Sequence[str],
    variant_specs: Sequence[ExperimentVariant],
    args: argparse.Namespace,
) -> dict[str, object]:
    return {
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
                        z_dim=args.z_dim,
                        h_dim=args.h_dim,
                        global_state_dim=args.global_state_dim,
                        obs_encoding_dim=args.obs_encoding_dim,
                        innovation_dim=args.innovation_dim,
                        source_summary_dim=args.source_summary_dim,
                        edge_message_dim=args.edge_message_dim,
                        edge_hidden_dim=args.edge_hidden_dim,
                        tau_embed_dim=args.tau_embed_dim,
                        met_summary_dim=args.met_summary_dim,
                        turbine_embed_dim=args.turbine_embed_dim,
                        dropout=args.dropout,
                        grad_clip_norm=args.grad_clip_norm,
                        hist_recon_loss_weight=args.hist_recon_loss_weight,
                        farm_loss_weight=args.farm_loss_weight,
                        met_loss_weight=args.met_loss_weight,
                        innovation_loss_weight=args.innovation_loss_weight,
                        weight_decay=args.weight_decay,
                        wake_lambda_x=args.wake_lambda_x,
                        wake_lambda_y=args.wake_lambda_y,
                        wake_kappa=args.wake_kappa,
                        bounded_output_epsilon=args.bounded_output_epsilon,
                    )
                ),
            }
            for spec in variant_specs
        }
        for dataset_id in dataset_ids
    }


def tensorboard_root_path(
    output_path: str | Path,
    *,
    work_root: str | Path = _RUN_WORK_ROOT,
) -> Path:
    return _resume_paths_for_output(output_path=output_path, work_root=work_root).slot_dir / "tensorboard"


def resolve_tensorboard_root(
    *,
    output_path: str | Path,
    work_root: str | Path = _RUN_WORK_ROOT,
    tensorboard_log_dir: str | Path | None = None,
    disable_tensorboard: bool = False,
) -> Path | None:
    if disable_tensorboard:
        return None
    if tensorboard_log_dir is not None:
        return Path(tensorboard_log_dir).expanduser().resolve()
    return tensorboard_root_path(output_path, work_root=work_root)


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    resolved_dataset_ids = _validate_dataset_ids(tuple(args.datasets) if args.datasets else DEFAULT_DATASETS)
    variant_specs = resolve_variant_specs(tuple(args.variants) if args.variants else None)
    resolved_tensorboard_root = resolve_tensorboard_root(
        output_path=args.output_path,
        work_root=_RUN_WORK_ROOT,
        tensorboard_log_dir=args.tensorboard_log_dir,
        disable_tensorboard=args.disable_tensorboard,
    )
    resolved_dataset_variant_hyperparameters = _resolved_record_hyperparameters(
        resolved_dataset_ids,
        variant_specs,
        args,
    )
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
        z_dim=args.z_dim,
        h_dim=args.h_dim,
        global_state_dim=args.global_state_dim,
        obs_encoding_dim=args.obs_encoding_dim,
        innovation_dim=args.innovation_dim,
        source_summary_dim=args.source_summary_dim,
        edge_message_dim=args.edge_message_dim,
        edge_hidden_dim=args.edge_hidden_dim,
        tau_embed_dim=args.tau_embed_dim,
        met_summary_dim=args.met_summary_dim,
        turbine_embed_dim=args.turbine_embed_dim,
        dropout=args.dropout,
        grad_clip_norm=args.grad_clip_norm,
        hist_recon_loss_weight=args.hist_recon_loss_weight,
        farm_loss_weight=args.farm_loss_weight,
        met_loss_weight=args.met_loss_weight,
        innovation_loss_weight=args.innovation_loss_weight,
        weight_decay=args.weight_decay,
        wake_lambda_x=args.wake_lambda_x,
        wake_lambda_y=args.wake_lambda_y,
        wake_kappa=args.wake_kappa,
        bounded_output_epsilon=args.bounded_output_epsilon,
        num_workers=args.num_workers,
        tensorboard_log_dir=args.tensorboard_log_dir,
        disable_tensorboard=args.disable_tensorboard,
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
            entrypoint=f"experiment/families/{FAMILY_ID}/run_world_model_state_space_v1.py",
            args=recorded_args,
            output_path=args.output_path,
            result_row_count=results.height,
            dataset_ids=resolved_dataset_ids,
            feature_protocol_ids=tuple(spec.feature_protocol_id for spec in variant_specs),
            model_variants=tuple(spec.model_variant for spec in variant_specs),
            eval_protocols=(ROLLING_EVAL_PROTOCOL, NON_OVERLAP_EVAL_PROTOCOL),
            result_splits=("val", "test"),
            artifacts={
                "training_history": training_history_output_path(args.output_path),
                **({} if resolved_tensorboard_root is None else {"tensorboard_root": resolved_tensorboard_root}),
            },
            run_label=args.run_label,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
