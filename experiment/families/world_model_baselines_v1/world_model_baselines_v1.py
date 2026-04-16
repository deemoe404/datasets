from __future__ import annotations

import argparse
from contextlib import nullcontext
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

WORLD_MODEL_STATE_SPACE_DIR = EXPERIMENT_DIR.parent / "world_model_state_space_v1"
if str(WORLD_MODEL_STATE_SPACE_DIR) not in sys.path:
    sys.path.insert(0, str(WORLD_MODEL_STATE_SPACE_DIR))

import world_model_state_space_v1 as state_base  # noqa: E402


world_model_base = state_base.world_model_base
torch = state_base.torch
nn = state_base.nn
F = state_base.F
DataLoader = state_base.DataLoader
Dataset = state_base.Dataset
try:
    from torch.utils.tensorboard import SummaryWriter
except Exception:  # pragma: no cover - optional dependency at runtime
    SummaryWriter = None


MODEL_ID = "WORLD_MODEL_BASELINE"
FAMILY_ID = "world_model_baselines_v1"
PERSISTENCE_VARIANT = "world_model_persistence_last_value_v1_farm_sync"
TFT_VARIANT = "world_model_shared_weight_tft_no_graph_v1_farm_sync"
TIMEXER_VARIANT = "world_model_shared_weight_timexer_no_graph_v1_farm_sync"
WINDOW_PROTOCOL = DEFAULT_WINDOW_PROTOCOL
TASK_PROTOCOL: WindowProtocolSpec = resolve_window_protocol(WINDOW_PROTOCOL)
TASK_ID = TASK_PROTOCOL.task_id
DEFAULT_DATASETS = ("kelmarsh",)
HISTORY_STEPS = state_base.HISTORY_STEPS
FORECAST_STEPS = state_base.FORECAST_STEPS
STRIDE_STEPS = state_base.STRIDE_STEPS
FEATURE_PROTOCOL_ID = state_base.FEATURE_PROTOCOL_ID

DEFAULT_BATCH_SIZE = 256
DEFAULT_LEARNING_RATE = 1e-4
DEFAULT_MAX_EPOCHS = 30
DEFAULT_EARLY_STOPPING_PATIENCE = 10
DEFAULT_SEED = 3407
DEFAULT_D_MODEL = 64
DEFAULT_LSTM_HIDDEN_DIM = 64
DEFAULT_ATTENTION_HEADS = 4
DEFAULT_DROPOUT = 0.2
DEFAULT_GRAD_CLIP_NORM = 1.0
DEFAULT_WEIGHT_DECAY = 1e-3
DEFAULT_BOUNDED_OUTPUT_EPSILON = 0.05
DEFAULT_TIMEXER_PATCH_LEN = 24
DEFAULT_TIMEXER_ENCODER_LAYERS = 2
DEFAULT_TIMEXER_FF_HIDDEN_DIM = 256
DEFAULT_CUDA_NUM_WORKERS = 2
DEFAULT_CUDA_PREFETCH_FACTOR = 4
DEFAULT_EVAL_BATCH_SIZE_MULTIPLIER = 2
DEFAULT_MAX_CUDA_EVAL_BATCH_SIZE = 1024
PROFILE_LOG_PREFIX = "[world_model_baselines_v1] "

_REPO_ROOT = EXPERIMENT_ROOT.parent
_CACHE_ROOT = _REPO_ROOT / "cache"
_OUTPUT_PATH = default_family_output_path(repo_root=_REPO_ROOT, family_id=FAMILY_ID)
_RUN_WORK_ROOT = EXPERIMENT_DIR / ".work" / "run_world_model_baselines_v1"
_RUN_STATE_SCHEMA_VERSION = "world_model_baselines_v1.run.resume.v1"
_TRAINING_CHECKPOINT_SCHEMA_VERSION = "world_model_baselines_v1.training_checkpoint.v1"
_DATASET_ORDER = {"kelmarsh": 0}
_MODEL_VARIANT_ORDER = {PERSISTENCE_VARIANT: 0, TFT_VARIANT: 1, TIMEXER_VARIANT: 2}
_SPLIT_ORDER = {"val": 0, "test": 1}
_EVAL_PROTOCOL_ORDER = {ROLLING_EVAL_PROTOCOL: 0, NON_OVERLAP_EVAL_PROTOCOL: 1}
_METRIC_SCOPE_ORDER = {OVERALL_METRIC_SCOPE: 0, HORIZON_METRIC_SCOPE: 1}

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
    "baseline_type",
    "uses_graph",
    "uses_pairwise",
    "uses_global_latent",
    "uses_future_observations",
    "bounded_output_epsilon",
    "d_model",
    "lstm_hidden_dim",
    "attention_heads",
    "patch_len",
    "encoder_layers",
    "ff_hidden_dim",
    "dropout",
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
    "baseline_type",
    "patch_len",
    "encoder_layers",
    "ff_hidden_dim",
    "epoch",
    "train_loss_mean",
    "train_loss_last",
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
    baseline_type: str


@dataclass(frozen=True)
class HyperparameterProfile:
    batch_size: int
    learning_rate: float
    max_epochs: int
    early_stopping_patience: int
    d_model: int
    lstm_hidden_dim: int
    attention_heads: int
    patch_len: int
    encoder_layers: int
    ff_hidden_dim: int
    dropout: float
    grad_clip_norm: float
    weight_decay: float
    bounded_output_epsilon: float


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
class TrainingOutcome:
    best_epoch: int
    epochs_ran: int
    best_val_rmse_pu: float
    best_val_mae_pu: float
    device: str
    amp_enabled: bool
    model: Any


@dataclass(frozen=True)
class ResumePaths:
    slot_dir: Path
    state_path: Path
    partial_results_path: Path
    training_history_path: Path
    checkpoints_dir: Path


@dataclass(frozen=True)
class TimeXerFeatureLayout:
    endogenous_history_names: tuple[str, ...]
    exogenous_history_names: tuple[str, ...]
    history_mark_names: tuple[str, ...]


VARIANT_SPECS = (
    ExperimentVariant(
        model_variant=PERSISTENCE_VARIANT,
        feature_protocol_id=FEATURE_PROTOCOL_ID,
        baseline_type="persistence_last_value",
    ),
    ExperimentVariant(
        model_variant=TFT_VARIANT,
        feature_protocol_id=FEATURE_PROTOCOL_ID,
        baseline_type="shared_weight_tft_no_graph",
    ),
    ExperimentVariant(
        model_variant=TIMEXER_VARIANT,
        feature_protocol_id=FEATURE_PROTOCOL_ID,
        baseline_type="shared_weight_timexer_no_graph",
    ),
)
DEFAULT_VARIANTS = tuple(spec.model_variant for spec in VARIANT_SPECS)
_VARIANT_SPECS_BY_NAME = {spec.model_variant: spec for spec in VARIANT_SPECS}
_DEFAULT_PROFILE = HyperparameterProfile(
    batch_size=DEFAULT_BATCH_SIZE,
    learning_rate=DEFAULT_LEARNING_RATE,
    max_epochs=DEFAULT_MAX_EPOCHS,
    early_stopping_patience=DEFAULT_EARLY_STOPPING_PATIENCE,
    d_model=DEFAULT_D_MODEL,
    lstm_hidden_dim=DEFAULT_LSTM_HIDDEN_DIM,
    attention_heads=DEFAULT_ATTENTION_HEADS,
    patch_len=DEFAULT_TIMEXER_PATCH_LEN,
    encoder_layers=DEFAULT_TIMEXER_ENCODER_LAYERS,
    ff_hidden_dim=DEFAULT_TIMEXER_FF_HIDDEN_DIM,
    dropout=DEFAULT_DROPOUT,
    grad_clip_norm=DEFAULT_GRAD_CLIP_NORM,
    weight_decay=DEFAULT_WEIGHT_DECAY,
    bounded_output_epsilon=DEFAULT_BOUNDED_OUTPUT_EPSILON,
)
TUNED_DEFAULT_HYPERPARAMETERS_BY_DATASET_AND_VARIANT = {
    "kelmarsh": {
        PERSISTENCE_VARIANT: _DEFAULT_PROFILE,
        TFT_VARIANT: _DEFAULT_PROFILE,
        TIMEXER_VARIANT: _DEFAULT_PROFILE,
    },
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
    d_model: int | None = None,
    lstm_hidden_dim: int | None = None,
    attention_heads: int | None = None,
    patch_len: int | None = None,
    encoder_layers: int | None = None,
    ff_hidden_dim: int | None = None,
    dropout: float | None = None,
    grad_clip_norm: float | None = None,
    weight_decay: float | None = None,
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
        d_model=defaults.d_model if d_model is None else d_model,
        lstm_hidden_dim=defaults.lstm_hidden_dim if lstm_hidden_dim is None else lstm_hidden_dim,
        attention_heads=defaults.attention_heads if attention_heads is None else attention_heads,
        patch_len=defaults.patch_len if patch_len is None else patch_len,
        encoder_layers=defaults.encoder_layers if encoder_layers is None else encoder_layers,
        ff_hidden_dim=defaults.ff_hidden_dim if ff_hidden_dim is None else ff_hidden_dim,
        dropout=defaults.dropout if dropout is None else dropout,
        grad_clip_norm=defaults.grad_clip_norm if grad_clip_norm is None else grad_clip_norm,
        weight_decay=defaults.weight_decay if weight_decay is None else weight_decay,
        bounded_output_epsilon=(
            defaults.bounded_output_epsilon if bounded_output_epsilon is None else bounded_output_epsilon
        ),
    )
    if profile.dropout < 0.0 or profile.dropout >= 1.0:
        raise ValueError(f"dropout must be in [0, 1), found {profile.dropout!r}.")
    if profile.d_model % profile.attention_heads != 0:
        raise ValueError("d_model must be divisible by attention_heads.")
    if variant_name == TIMEXER_VARIANT:
        if profile.patch_len <= 0:
            raise ValueError(f"patch_len must be positive, found {profile.patch_len!r}.")
        if HISTORY_STEPS % profile.patch_len != 0:
            raise ValueError(f"patch_len must evenly divide history_steps={HISTORY_STEPS}, found {profile.patch_len!r}.")
        if profile.encoder_layers <= 0:
            raise ValueError(f"encoder_layers must be positive, found {profile.encoder_layers!r}.")
        if profile.ff_hidden_dim <= 0:
            raise ValueError(f"ff_hidden_dim must be positive, found {profile.ff_hidden_dim!r}.")
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
    prepared_dataset: state_base.PreparedDataset,
    variant_spec: ExperimentVariant,
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
        "model_variant": variant_spec.model_variant,
        "baseline_type": variant_spec.baseline_type,
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


def prepare_dataset(
    dataset_id: str,
    *,
    variant_spec: ExperimentVariant | None = None,
    cache_root: str | Path = _CACHE_ROOT,
    max_train_origins: int | None = None,
    max_eval_origins: int | None = None,
) -> state_base.PreparedDataset:
    _validate_dataset_ids((dataset_id,))
    resolved_variant = variant_spec or VARIANT_SPECS[0]
    prepared = state_base.prepare_dataset(
        dataset_id,
        cache_root=cache_root,
        max_train_origins=max_train_origins,
        max_eval_origins=max_eval_origins,
    )
    return replace(prepared, model_variant=resolved_variant.model_variant)


def resolve_timexer_feature_layout(prepared_dataset: state_base.PreparedDataset) -> TimeXerFeatureLayout:
    history_mark_count = prepared_dataset.context_future_channels
    if history_mark_count <= 0:
        raise ValueError("TimeXer requires at least one history mark feature.")
    if prepared_dataset.local_input_feature_names[0] != "target_pu":
        raise ValueError("TimeXer expects target_pu to be the first local-history feature.")
    history_mark_names = prepared_dataset.context_history_feature_names[-history_mark_count:]
    return TimeXerFeatureLayout(
        endogenous_history_names=("target_pu",),
        exogenous_history_names=(
            *prepared_dataset.local_input_feature_names[1:],
            *prepared_dataset.context_history_feature_names[:-history_mark_count],
        ),
        history_mark_names=history_mark_names,
    )


def _timexer_history_inputs(
    prepared_dataset: state_base.PreparedDataset,
    *,
    history_slice: slice,
    node_index: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    layout = resolve_timexer_feature_layout(prepared_dataset)
    local_history = prepared_dataset.local_history_tensor[history_slice, node_index, :]
    context_history = prepared_dataset.context_history_tensor[history_slice, :]
    history_mark_count = len(layout.history_mark_names)
    endogenous_history = local_history[:, :1].astype(np.float32, copy=True)
    exogenous_history = np.concatenate(
        (local_history[:, 1:], context_history[:, :-history_mark_count]),
        axis=-1,
    ).astype(np.float32, copy=True)
    history_marks = context_history[:, -history_mark_count:].astype(np.float32, copy=True)
    return endogenous_history, exogenous_history, history_marks


def iter_evaluation_specs(
    prepared_dataset: state_base.PreparedDataset,
) -> tuple[tuple[str, str, world_model_base.FarmWindowDescriptorIndex], ...]:
    return state_base.iter_evaluation_specs(prepared_dataset)


def _safe_divide(numerator: float, denominator: int) -> float:
    return float("nan") if denominator <= 0 else numerator / denominator


def _safe_rmse(squared_error_sum: float, denominator: int) -> float:
    return float("nan") if denominator <= 0 else math.sqrt(squared_error_sum / denominator)


def _metrics_from_arrays(
    predictions: np.ndarray,
    targets: np.ndarray,
    valid_mask: np.ndarray,
    *,
    rated_power_kw: float,
) -> EvaluationMetrics:
    valid = np.asarray(valid_mask, dtype=np.float64)
    errors_pu = (np.asarray(predictions, dtype=np.float64) - np.asarray(targets, dtype=np.float64)) * valid
    errors_kw = errors_pu * float(rated_power_kw)
    prediction_count = int(valid.sum())
    forecast_steps = predictions.shape[1]
    horizon_prediction_count = valid.sum(axis=(0, 2, 3)).astype(np.int64, copy=False)
    horizon_abs_pu = np.abs(errors_pu).sum(axis=(0, 2, 3))
    horizon_sq_pu = np.square(errors_pu).sum(axis=(0, 2, 3))
    horizon_abs_kw = np.abs(errors_kw).sum(axis=(0, 2, 3))
    horizon_sq_kw = np.square(errors_kw).sum(axis=(0, 2, 3))
    return EvaluationMetrics(
        window_count=int(predictions.shape[0]),
        prediction_count=prediction_count,
        mae_kw=_safe_divide(float(np.abs(errors_kw).sum()), prediction_count),
        rmse_kw=_safe_rmse(float(np.square(errors_kw).sum()), prediction_count),
        mae_pu=_safe_divide(float(np.abs(errors_pu).sum()), prediction_count),
        rmse_pu=_safe_rmse(float(np.square(errors_pu).sum()), prediction_count),
        horizon_window_count=np.full((forecast_steps,), int(predictions.shape[0]), dtype=np.int64),
        horizon_prediction_count=horizon_prediction_count,
        horizon_mae_kw=np.asarray(
            [_safe_divide(float(horizon_abs_kw[i]), int(horizon_prediction_count[i])) for i in range(forecast_steps)]
        ),
        horizon_rmse_kw=np.asarray(
            [_safe_rmse(float(horizon_sq_kw[i]), int(horizon_prediction_count[i])) for i in range(forecast_steps)]
        ),
        horizon_mae_pu=np.asarray(
            [_safe_divide(float(horizon_abs_pu[i]), int(horizon_prediction_count[i])) for i in range(forecast_steps)]
        ),
        horizon_rmse_pu=np.asarray(
            [_safe_rmse(float(horizon_sq_pu[i]), int(horizon_prediction_count[i])) for i in range(forecast_steps)]
        ),
    )


def _future_targets_for_windows(
    prepared_dataset: state_base.PreparedDataset,
    windows: world_model_base.FarmWindowDescriptorIndex,
) -> tuple[np.ndarray, np.ndarray]:
    targets = np.zeros((len(windows), prepared_dataset.forecast_steps, prepared_dataset.node_count, 1), dtype=np.float32)
    valid = np.zeros_like(targets, dtype=np.float32)
    for window_pos, target_index in enumerate(windows.target_indices):
        future_slice = slice(int(target_index), int(target_index) + prepared_dataset.forecast_steps)
        targets[window_pos, :, :, 0] = prepared_dataset.target_pu_filled[future_slice]
        valid[window_pos, :, :, 0] = prepared_dataset.target_valid_mask[future_slice].astype(np.float32, copy=False)
    return targets, valid


def _train_history_target_mean(prepared_dataset: state_base.PreparedDataset) -> np.ndarray:
    sums = np.zeros((prepared_dataset.node_count,), dtype=np.float64)
    counts = np.zeros((prepared_dataset.node_count,), dtype=np.float64)
    for target_index in prepared_dataset.train_windows.target_indices:
        history_slice = slice(int(target_index) - prepared_dataset.history_steps, int(target_index))
        values = prepared_dataset.local_history_tensor[history_slice, :, state_base._LOCAL_VALUE_START]
        unavailable = prepared_dataset.local_history_tensor[history_slice, :, state_base._LOCAL_MASK_START]
        available = unavailable < 0.5
        sums += np.where(available, values, 0.0).sum(axis=0)
        counts += available.sum(axis=0)
    return np.where(counts > 0, sums / np.maximum(counts, 1.0), 0.0).astype(np.float32)


def persistence_predictions(
    prepared_dataset: state_base.PreparedDataset,
    windows: world_model_base.FarmWindowDescriptorIndex,
    *,
    train_fallback: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    fallback = _train_history_target_mean(prepared_dataset) if train_fallback is None else train_fallback
    predictions = np.zeros((len(windows), prepared_dataset.forecast_steps, prepared_dataset.node_count, 1), dtype=np.float32)
    targets, valid = _future_targets_for_windows(prepared_dataset, windows)
    for window_pos, target_index in enumerate(windows.target_indices):
        history_slice = slice(int(target_index) - prepared_dataset.history_steps, int(target_index))
        values = prepared_dataset.local_history_tensor[history_slice, :, state_base._LOCAL_VALUE_START]
        unavailable = prepared_dataset.local_history_tensor[history_slice, :, state_base._LOCAL_MASK_START]
        available = unavailable < 0.5
        last_values = fallback.copy()
        for node_index in range(prepared_dataset.node_count):
            valid_positions = np.flatnonzero(available[:, node_index])
            if valid_positions.size:
                last_values[node_index] = values[int(valid_positions[-1]), node_index]
        predictions[window_pos, :, :, 0] = last_values[None, :]
    return predictions, targets, valid


def evaluate_persistence(
    prepared_dataset: state_base.PreparedDataset,
    windows: world_model_base.FarmWindowDescriptorIndex,
    *,
    train_fallback: np.ndarray | None = None,
) -> EvaluationMetrics:
    predictions, targets, valid = persistence_predictions(
        prepared_dataset,
        windows,
        train_fallback=train_fallback,
    )
    return _metrics_from_arrays(
        predictions,
        targets,
        valid,
        rated_power_kw=prepared_dataset.rated_power_kw,
    )


if Dataset is not None:

    class TurbineWindowDataset(Dataset):
        def __init__(
            self,
            prepared_dataset: state_base.PreparedDataset,
            windows: world_model_base.FarmWindowDescriptorIndex,
            *,
            include_indices: bool = False,
        ) -> None:
            self.prepared_dataset = prepared_dataset
            self.windows = windows
            self.include_indices = include_indices

        def __len__(self) -> int:
            return len(self.windows) * self.prepared_dataset.node_count

        def __getitem__(self, index: int):
            prepared = self.prepared_dataset
            window_pos = int(index) // prepared.node_count
            node_index = int(index) % prepared.node_count
            target_index = int(self.windows.target_indices[window_pos])
            history_slice = slice(target_index - prepared.history_steps, target_index)
            future_slice = slice(target_index, target_index + prepared.forecast_steps)
            item = (
                prepared.local_history_tensor[history_slice, node_index, :].astype(np.float32, copy=True),
                prepared.context_history_tensor[history_slice, :].astype(np.float32, copy=True),
                prepared.context_future_tensor[future_slice, :].astype(np.float32, copy=True),
                prepared.static_tensor[node_index, :].astype(np.float32, copy=True),
                prepared.target_pu_filled[future_slice, node_index, None].astype(np.float32, copy=True),
                prepared.target_valid_mask[future_slice, node_index, None].astype(np.float32, copy=True),
            )
            if not self.include_indices:
                return item
            return (*item, np.int64(window_pos), np.int64(node_index))


    class TimeXerWindowDataset(Dataset):
        def __init__(
            self,
            prepared_dataset: state_base.PreparedDataset,
            windows: world_model_base.FarmWindowDescriptorIndex,
            *,
            include_indices: bool = False,
        ) -> None:
            self.prepared_dataset = prepared_dataset
            self.windows = windows
            self.include_indices = include_indices

        def __len__(self) -> int:
            return len(self.windows) * self.prepared_dataset.node_count

        def __getitem__(self, index: int):
            prepared = self.prepared_dataset
            window_pos = int(index) // prepared.node_count
            node_index = int(index) % prepared.node_count
            target_index = int(self.windows.target_indices[window_pos])
            history_slice = slice(target_index - prepared.history_steps, target_index)
            future_slice = slice(target_index, target_index + prepared.forecast_steps)
            endogenous_history, exogenous_history, history_marks = _timexer_history_inputs(
                prepared,
                history_slice=history_slice,
                node_index=node_index,
            )
            item = (
                endogenous_history,
                exogenous_history,
                history_marks,
                prepared.target_pu_filled[future_slice, node_index, None].astype(np.float32, copy=True),
                prepared.target_valid_mask[future_slice, node_index, None].astype(np.float32, copy=True),
            )
            if not self.include_indices:
                return item
            return (*item, np.int64(window_pos), np.int64(node_index))

else:

    class TurbineWindowDataset:  # pragma: no cover
        def __init__(self, *_args, **_kwargs) -> None:
            _require_torch()


    class TimeXerWindowDataset:  # pragma: no cover
        def __init__(self, *_args, **_kwargs) -> None:
            _require_torch()


def _build_turbine_dataloader(
    prepared_dataset: state_base.PreparedDataset,
    *,
    windows: world_model_base.FarmWindowDescriptorIndex,
    batch_size: int,
    device: str,
    shuffle: bool,
    seed: int,
    num_workers: int | None = None,
    include_indices: bool = False,
):
    resolved_torch, _, _, resolved_loader, _ = _require_torch()
    resolved_device = resolve_device(device)
    resolved_num_workers = resolve_loader_num_workers(device=resolved_device, num_workers=num_workers)
    generator = resolved_torch.Generator()
    generator.manual_seed(seed)
    loader_kwargs: dict[str, object] = {
        "dataset": TurbineWindowDataset(prepared_dataset, windows, include_indices=include_indices),
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


def _build_timexer_dataloader(
    prepared_dataset: state_base.PreparedDataset,
    *,
    windows: world_model_base.FarmWindowDescriptorIndex,
    batch_size: int,
    device: str,
    shuffle: bool,
    seed: int,
    num_workers: int | None = None,
    include_indices: bool = False,
):
    resolved_torch, _, _, resolved_loader, _ = _require_torch()
    resolved_device = resolve_device(device)
    resolved_num_workers = resolve_loader_num_workers(device=resolved_device, num_workers=num_workers)
    generator = resolved_torch.Generator()
    generator.manual_seed(seed)
    loader_kwargs: dict[str, object] = {
        "dataset": TimeXerWindowDataset(prepared_dataset, windows, include_indices=include_indices),
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

    class SinusoidalPositionalEmbedding(nn.Module):
        def __init__(self, d_model: int, max_len: int) -> None:
            super().__init__()
            position = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
            div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model))
            pe = torch.zeros((1, max_len, d_model), dtype=torch.float32)
            pe[:, :, 0::2] = torch.sin(position * div_term)
            pe[:, :, 1::2] = torch.cos(position * div_term[: pe[:, :, 1::2].shape[2]])
            self.register_buffer("pe", pe, persistent=False)

        def forward(self, inputs):
            return self.pe[:, : inputs.shape[1], :]

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


    class GatedResidualBlock(nn.Module):
        def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, *, dropout: float) -> None:
            super().__init__()
            self.skip = nn.Identity() if input_dim == output_dim else nn.Linear(input_dim, output_dim)
            self.fc1 = nn.Linear(input_dim, hidden_dim)
            self.fc2 = nn.Linear(hidden_dim, output_dim)
            self.gate = nn.Linear(input_dim, output_dim)
            self.norm = nn.LayerNorm(output_dim)
            self.dropout = nn.Dropout(dropout)

        def forward(self, inputs):
            hidden = self.dropout(F.silu(self.fc1(inputs)))
            transformed = self.fc2(hidden)
            gate = torch.sigmoid(self.gate(inputs))
            return self.norm(self.skip(inputs) + gate * transformed)


    class SharedWeightTFTNoGraph(nn.Module):
        def __init__(
            self,
            *,
            local_input_channels: int,
            context_history_channels: int,
            context_future_channels: int,
            static_feature_count: int,
            forecast_steps: int,
            d_model: int,
            lstm_hidden_dim: int,
            attention_heads: int,
            dropout: float,
            bounded_output_epsilon: float,
        ) -> None:
            super().__init__()
            self.forecast_steps = forecast_steps
            self.bounded_output_epsilon = bounded_output_epsilon
            self.static_encoder = FeedForward(static_feature_count, max(d_model, static_feature_count * 2), d_model, dropout=dropout)
            self.history_grn = GatedResidualBlock(
                local_input_channels + context_history_channels + d_model,
                d_model * 2,
                d_model,
                dropout=dropout,
            )
            self.future_grn = GatedResidualBlock(
                context_future_channels + d_model + d_model,
                d_model * 2,
                d_model,
                dropout=dropout,
            )
            self.horizon_embedding = nn.Embedding(forecast_steps, d_model)
            self.encoder = nn.LSTM(d_model, lstm_hidden_dim, batch_first=True)
            self.decoder = nn.LSTM(d_model, lstm_hidden_dim, batch_first=True)
            self.query_projection = nn.Linear(lstm_hidden_dim, d_model)
            self.key_projection = nn.Linear(lstm_hidden_dim, d_model)
            self.value_projection = nn.Linear(lstm_hidden_dim, d_model)
            self.attention = nn.MultiheadAttention(
                embed_dim=d_model,
                num_heads=attention_heads,
                dropout=dropout,
                batch_first=True,
            )
            self.output_grn = GatedResidualBlock(d_model * 3, d_model * 2, d_model, dropout=dropout)
            self.output_head = nn.Linear(d_model, 1)

        def forward(self, local_history, context_history, context_future, static_features):
            batch_size = local_history.shape[0]
            static_context = self.static_encoder(static_features)
            history_static = static_context[:, None, :].expand(-1, local_history.shape[1], -1)
            history_inputs = torch.cat((local_history, context_history, history_static), dim=-1)
            history_encoded = self.history_grn(history_inputs)
            encoder_output, encoder_state = self.encoder(history_encoded)
            horizon_indices = torch.arange(context_future.shape[1], device=context_future.device, dtype=torch.long)
            horizon = self.horizon_embedding(horizon_indices)[None, :, :].expand(batch_size, -1, -1)
            future_static = static_context[:, None, :].expand(-1, context_future.shape[1], -1)
            future_inputs = torch.cat((context_future, future_static, horizon), dim=-1)
            future_encoded = self.future_grn(future_inputs)
            decoder_output, _decoder_state = self.decoder(future_encoded, encoder_state)
            queries = self.query_projection(decoder_output)
            keys = self.key_projection(encoder_output)
            values = self.value_projection(encoder_output)
            attended, _weights = self.attention(queries, keys, values, need_weights=False)
            fused = self.output_grn(torch.cat((queries, attended, future_encoded), dim=-1))
            return (1.0 + self.bounded_output_epsilon) * torch.sigmoid(self.output_head(fused))


    class TimeXerEndogenousEmbedding(nn.Module):
        def __init__(self, *, history_steps: int, patch_len: int, d_model: int, dropout: float) -> None:
            super().__init__()
            if history_steps % patch_len != 0:
                raise ValueError(f"patch_len must evenly divide history_steps={history_steps}.")
            self.history_steps = history_steps
            self.patch_len = patch_len
            self.patch_count = history_steps // patch_len
            self.value_embedding = nn.Linear(patch_len, d_model, bias=False)
            self.position_embedding = SinusoidalPositionalEmbedding(d_model, max_len=self.patch_count)
            self.global_token = nn.Parameter(torch.zeros(1, 1, d_model))
            self.dropout = nn.Dropout(dropout)

        def forward(self, endogenous_history):
            if endogenous_history.ndim != 3 or endogenous_history.shape[2] != 1:
                raise ValueError("TimeXer endogenous history must have shape [batch, history_steps, 1].")
            if endogenous_history.shape[1] != self.history_steps:
                raise ValueError(
                    f"TimeXer endogenous history length must be {self.history_steps}, found {endogenous_history.shape[1]}."
                )
            patches = endogenous_history.squeeze(-1).unfold(1, self.patch_len, self.patch_len)
            tokens = self.value_embedding(patches) + self.position_embedding(patches)
            global_token = self.global_token.expand(tokens.shape[0], -1, -1)
            return self.dropout(torch.cat((tokens, global_token), dim=1))


    class TimeXerExogenousEmbedding(nn.Module):
        def __init__(self, *, history_steps: int, d_model: int, dropout: float) -> None:
            super().__init__()
            self.history_steps = history_steps
            self.value_embedding = nn.Linear(history_steps, d_model)
            self.dropout = nn.Dropout(dropout)

        def forward(self, exogenous_history, history_marks):
            if exogenous_history.shape[1] != self.history_steps:
                raise ValueError(
                    f"TimeXer exogenous history length must be {self.history_steps}, found {exogenous_history.shape[1]}."
                )
            if history_marks.shape[1] != self.history_steps:
                raise ValueError(
                    f"TimeXer history marks length must be {self.history_steps}, found {history_marks.shape[1]}."
                )
            combined = torch.cat((exogenous_history, history_marks), dim=-1).transpose(1, 2)
            return self.dropout(self.value_embedding(combined))


    class TimeXerEncoderLayer(nn.Module):
        def __init__(self, *, d_model: int, attention_heads: int, ff_hidden_dim: int, dropout: float) -> None:
            super().__init__()
            self.self_attention = nn.MultiheadAttention(
                embed_dim=d_model,
                num_heads=attention_heads,
                dropout=dropout,
                batch_first=True,
            )
            self.cross_attention = nn.MultiheadAttention(
                embed_dim=d_model,
                num_heads=attention_heads,
                dropout=dropout,
                batch_first=True,
            )
            self.norm1 = nn.LayerNorm(d_model)
            self.norm2 = nn.LayerNorm(d_model)
            self.norm3 = nn.LayerNorm(d_model)
            self.dropout = nn.Dropout(dropout)
            self.feed_forward = nn.Sequential(
                nn.Linear(d_model, ff_hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(ff_hidden_dim, d_model),
                nn.Dropout(dropout),
            )

        def forward(self, tokens, exogenous_tokens):
            self_attended, _weights = self.self_attention(tokens, tokens, tokens, need_weights=False)
            tokens = self.norm1(tokens + self.dropout(self_attended))
            global_token = tokens[:, -1:, :]
            cross_attended, _weights = self.cross_attention(
                global_token,
                exogenous_tokens,
                exogenous_tokens,
                need_weights=False,
            )
            global_token = self.norm2(global_token + self.dropout(cross_attended))
            tokens = torch.cat((tokens[:, :-1, :], global_token), dim=1)
            return self.norm3(tokens + self.feed_forward(tokens))


    class SharedWeightTimeXerNoGraph(nn.Module):
        def __init__(
            self,
            *,
            history_steps: int,
            forecast_steps: int,
            d_model: int,
            attention_heads: int,
            patch_len: int,
            encoder_layers: int,
            ff_hidden_dim: int,
            dropout: float,
            bounded_output_epsilon: float,
        ) -> None:
            super().__init__()
            if history_steps % patch_len != 0:
                raise ValueError(f"patch_len must evenly divide history_steps={history_steps}.")
            self.forecast_steps = forecast_steps
            self.bounded_output_epsilon = bounded_output_epsilon
            self.endogenous_embedding = TimeXerEndogenousEmbedding(
                history_steps=history_steps,
                patch_len=patch_len,
                d_model=d_model,
                dropout=dropout,
            )
            self.exogenous_embedding = TimeXerExogenousEmbedding(
                history_steps=history_steps,
                d_model=d_model,
                dropout=dropout,
            )
            self.encoder_layers = nn.ModuleList(
                [
                    TimeXerEncoderLayer(
                        d_model=d_model,
                        attention_heads=attention_heads,
                        ff_hidden_dim=ff_hidden_dim,
                        dropout=dropout,
                    )
                    for _ in range(encoder_layers)
                ]
            )
            self.head = nn.Sequential(
                nn.Flatten(start_dim=1),
                nn.Linear(d_model * ((history_steps // patch_len) + 1), forecast_steps),
            )

        def forward(self, endogenous_history, exogenous_history, history_marks):
            means = endogenous_history.mean(dim=1, keepdim=True).detach()
            centered = endogenous_history - means
            stdev = torch.sqrt(torch.var(centered, dim=1, keepdim=True, unbiased=False) + 1e-5)
            normalized_endogenous = centered / stdev
            tokens = self.endogenous_embedding(normalized_endogenous)
            exogenous_tokens = self.exogenous_embedding(exogenous_history, history_marks)
            for layer in self.encoder_layers:
                tokens = layer(tokens, exogenous_tokens)
            logits = self.head(tokens).unsqueeze(-1)
            predictions = (1.0 + self.bounded_output_epsilon) * torch.sigmoid(logits)
            predictions = predictions * stdev + means
            return torch.clamp(predictions, min=0.0, max=1.0 + self.bounded_output_epsilon)

else:

    class SinusoidalPositionalEmbedding:  # pragma: no cover
        def __init__(self, *_args, **_kwargs) -> None:
            _require_torch()

    class FeedForward:  # pragma: no cover
        def __init__(self, *_args, **_kwargs) -> None:
            _require_torch()

    class GatedResidualBlock:  # pragma: no cover
        def __init__(self, *_args, **_kwargs) -> None:
            _require_torch()

    class SharedWeightTFTNoGraph:  # pragma: no cover
        def __init__(self, *_args, **_kwargs) -> None:
            _require_torch()

    class TimeXerEndogenousEmbedding:  # pragma: no cover
        def __init__(self, *_args, **_kwargs) -> None:
            _require_torch()

    class TimeXerExogenousEmbedding:  # pragma: no cover
        def __init__(self, *_args, **_kwargs) -> None:
            _require_torch()

    class TimeXerEncoderLayer:  # pragma: no cover
        def __init__(self, *_args, **_kwargs) -> None:
            _require_torch()

    class SharedWeightTimeXerNoGraph:  # pragma: no cover
        def __init__(self, *_args, **_kwargs) -> None:
            _require_torch()


def build_tft_model(
    *,
    prepared_dataset: state_base.PreparedDataset,
    d_model: int,
    lstm_hidden_dim: int,
    attention_heads: int,
    dropout: float,
    bounded_output_epsilon: float,
):
    _require_torch()
    return SharedWeightTFTNoGraph(
        local_input_channels=prepared_dataset.local_input_channels,
        context_history_channels=prepared_dataset.context_history_channels,
        context_future_channels=prepared_dataset.context_future_channels,
        static_feature_count=prepared_dataset.static_feature_count,
        forecast_steps=prepared_dataset.forecast_steps,
        d_model=d_model,
        lstm_hidden_dim=lstm_hidden_dim,
        attention_heads=attention_heads,
        dropout=dropout,
        bounded_output_epsilon=bounded_output_epsilon,
    )


def build_timexer_model(
    *,
    prepared_dataset: state_base.PreparedDataset,
    d_model: int,
    attention_heads: int,
    patch_len: int,
    encoder_layers: int,
    ff_hidden_dim: int,
    dropout: float,
    bounded_output_epsilon: float,
):
    _require_torch()
    return SharedWeightTimeXerNoGraph(
        history_steps=prepared_dataset.history_steps,
        forecast_steps=prepared_dataset.forecast_steps,
        d_model=d_model,
        attention_heads=attention_heads,
        patch_len=patch_len,
        encoder_layers=encoder_layers,
        ff_hidden_dim=ff_hidden_dim,
        dropout=dropout,
        bounded_output_epsilon=bounded_output_epsilon,
    )


def initialize_model_parameters(model) -> None:
    _, resolved_nn, _, _, _ = _require_torch()
    for parameter in model.parameters():
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


def _make_grad_scaler(torch_module, *, enabled: bool):
    if hasattr(torch_module, "amp") and hasattr(torch_module.amp, "GradScaler"):
        try:
            return torch_module.amp.GradScaler("cuda", enabled=enabled)
        except TypeError:
            return torch_module.amp.GradScaler(enabled=enabled)
    return torch_module.cuda.amp.GradScaler(enabled=enabled)


def _move_batch_to_device(batch, *, torch_module, device: str, include_indices: bool = False):
    tensor_items = batch[:-2] if include_indices else batch
    moved = tuple(
        item.to(device=device, dtype=torch_module.float32, non_blocking=device == "cuda")
        for item in tensor_items
    )
    if include_indices:
        return (*moved, batch[-2], batch[-1])
    return moved


def masked_huber_loss(predictions, targets, valid_mask, *, torch_module, delta: float = 0.03):
    mask = valid_mask.to(device=predictions.device, dtype=predictions.dtype)
    count = mask.sum()
    if float(count.item()) <= 0:
        return (predictions * 0.0).sum()
    errors = predictions - targets
    abs_errors = torch_module.abs(errors)
    huber = torch_module.where(abs_errors <= delta, 0.5 * torch_module.square(errors) / delta, abs_errors - 0.5 * delta)
    return (huber * mask).sum() / count


def masked_mse_loss(predictions, targets, valid_mask, *, torch_module):
    mask = valid_mask.to(device=predictions.device, dtype=predictions.dtype)
    count = mask.sum()
    if float(count.item()) <= 0:
        return (predictions * 0.0).sum()
    squared_error = torch_module.square(predictions - targets)
    return (squared_error * mask).sum() / count


def evaluate_tft_model(
    model,
    prepared_dataset: state_base.PreparedDataset,
    windows: world_model_base.FarmWindowDescriptorIndex,
    *,
    batch_size: int,
    device: str,
    seed: int,
    num_workers: int | None = None,
    amp_enabled: bool = False,
    progress_label: str | None = None,
) -> EvaluationMetrics:
    resolved_torch, _, _, _, _ = _require_torch()
    predictions = np.zeros((len(windows), prepared_dataset.forecast_steps, prepared_dataset.node_count, 1), dtype=np.float32)
    targets = np.zeros_like(predictions, dtype=np.float32)
    valid = np.zeros_like(predictions, dtype=np.float32)
    loader = _build_turbine_dataloader(
        prepared_dataset,
        windows=windows,
        batch_size=batch_size,
        device=device,
        shuffle=False,
        seed=seed,
        num_workers=num_workers,
        include_indices=True,
    )
    model.eval()
    progress = _create_progress_bar(total=_loader_batch_total(loader), desc=progress_label or "evaluate")
    try:
        with resolved_torch.no_grad():
            for raw_batch in loader:
                (
                    batch_local_history,
                    batch_context_history,
                    batch_context_future,
                    batch_static,
                    batch_targets,
                    batch_valid,
                    batch_window_pos,
                    batch_node_index,
                ) = _move_batch_to_device(raw_batch, torch_module=resolved_torch, device=device, include_indices=True)
                with _amp_context(torch_module=resolved_torch, device=device, enabled=amp_enabled):
                    batch_predictions = model(
                        batch_local_history,
                        batch_context_history,
                        batch_context_future,
                        batch_static,
                    ).float()
                window_pos = batch_window_pos.detach().cpu().numpy().astype(np.int64, copy=False)
                node_index = batch_node_index.detach().cpu().numpy().astype(np.int64, copy=False)
                batch_predictions_np = batch_predictions.detach().cpu().numpy().astype(np.float32, copy=False)
                batch_targets_np = batch_targets.detach().cpu().numpy().astype(np.float32, copy=False)
                batch_valid_np = batch_valid.detach().cpu().numpy().astype(np.float32, copy=False)
                for row_index, (window_id, node_id) in enumerate(zip(window_pos, node_index, strict=True)):
                    predictions[int(window_id), :, int(node_id), :] = batch_predictions_np[row_index]
                    targets[int(window_id), :, int(node_id), :] = batch_targets_np[row_index]
                    valid[int(window_id), :, int(node_id), :] = batch_valid_np[row_index]
                progress.update(1)
    finally:
        progress.close()
    return _metrics_from_arrays(
        predictions,
        targets,
        valid,
        rated_power_kw=prepared_dataset.rated_power_kw,
    )


def evaluate_timexer_model(
    model,
    prepared_dataset: state_base.PreparedDataset,
    windows: world_model_base.FarmWindowDescriptorIndex,
    *,
    batch_size: int,
    device: str,
    seed: int,
    num_workers: int | None = None,
    amp_enabled: bool = False,
    progress_label: str | None = None,
) -> EvaluationMetrics:
    resolved_torch, _, _, _, _ = _require_torch()
    predictions = np.zeros((len(windows), prepared_dataset.forecast_steps, prepared_dataset.node_count, 1), dtype=np.float32)
    targets = np.zeros_like(predictions, dtype=np.float32)
    valid = np.zeros_like(predictions, dtype=np.float32)
    loader = _build_timexer_dataloader(
        prepared_dataset,
        windows=windows,
        batch_size=batch_size,
        device=device,
        shuffle=False,
        seed=seed,
        num_workers=num_workers,
        include_indices=True,
    )
    model.eval()
    progress = _create_progress_bar(total=_loader_batch_total(loader), desc=progress_label or "evaluate")
    try:
        with resolved_torch.no_grad():
            for raw_batch in loader:
                (
                    batch_endogenous_history,
                    batch_exogenous_history,
                    batch_history_marks,
                    batch_targets,
                    batch_valid,
                    batch_window_pos,
                    batch_node_index,
                ) = _move_batch_to_device(raw_batch, torch_module=resolved_torch, device=device, include_indices=True)
                with _amp_context(torch_module=resolved_torch, device=device, enabled=amp_enabled):
                    batch_predictions = model(
                        batch_endogenous_history,
                        batch_exogenous_history,
                        batch_history_marks,
                    ).float()
                window_pos = batch_window_pos.detach().cpu().numpy().astype(np.int64, copy=False)
                node_index = batch_node_index.detach().cpu().numpy().astype(np.int64, copy=False)
                batch_predictions_np = batch_predictions.detach().cpu().numpy().astype(np.float32, copy=False)
                batch_targets_np = batch_targets.detach().cpu().numpy().astype(np.float32, copy=False)
                batch_valid_np = batch_valid.detach().cpu().numpy().astype(np.float32, copy=False)
                for row_index, (window_id, node_id) in enumerate(zip(window_pos, node_index, strict=True)):
                    predictions[int(window_id), :, int(node_id), :] = batch_predictions_np[row_index]
                    targets[int(window_id), :, int(node_id), :] = batch_targets_np[row_index]
                    valid[int(window_id), :, int(node_id), :] = batch_valid_np[row_index]
                progress.update(1)
    finally:
        progress.close()
    return _metrics_from_arrays(
        predictions,
        targets,
        valid,
        rated_power_kw=prepared_dataset.rated_power_kw,
    )


def train_tft_model(
    prepared_dataset: state_base.PreparedDataset,
    *,
    device: str,
    seed: int,
    batch_size: int,
    eval_batch_size: int | None,
    learning_rate: float,
    max_epochs: int,
    early_stopping_patience: int,
    d_model: int,
    lstm_hidden_dim: int,
    attention_heads: int,
    dropout: float,
    grad_clip_norm: float,
    weight_decay: float,
    bounded_output_epsilon: float,
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
    resolved_eval_batch_size = resolve_eval_batch_size(batch_size, device=resolved_device, eval_batch_size=eval_batch_size)
    model = build_tft_model(
        prepared_dataset=prepared_dataset,
        d_model=d_model,
        lstm_hidden_dim=lstm_hidden_dim,
        attention_heads=attention_heads,
        dropout=dropout,
        bounded_output_epsilon=bounded_output_epsilon,
    ).to(device=resolved_device)
    initialize_model_parameters(model)
    optimizer = resolved_torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scaler = _make_grad_scaler(resolved_torch, enabled=amp_enabled)
    resolved_checkpoint_path = Path(checkpoint_path) if checkpoint_path is not None else None
    resolved_training_history_path = Path(training_history_path) if training_history_path is not None else None
    checkpoint_job_identity = _job_identity_for_prepared_dataset(prepared_dataset)
    best_state: dict[str, Any] | None = None
    best_epoch = 0
    best_val_mae_pu = float("inf")
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
                raise RuntimeError(f"Checkpoint at {resolved_checkpoint_path} does not match the active job.")
            model.load_state_dict(checkpoint_payload["model_state_dict"])
            optimizer.load_state_dict(checkpoint_payload["optimizer_state_dict"])
            if "scaler_state_dict" in checkpoint_payload:
                scaler.load_state_dict(checkpoint_payload["scaler_state_dict"])
            best_state = checkpoint_payload.get("best_state_dict")
            best_epoch = int(checkpoint_payload["best_epoch"])
            best_val_mae_pu = float(checkpoint_payload["best_val_mae_pu"])
            best_val_rmse_pu = float(checkpoint_payload["best_val_rmse_pu"])
            epochs_without_improvement = int(checkpoint_payload["epochs_without_improvement"])
            epochs_ran = int(checkpoint_payload["epochs_ran"])
            start_epoch = int(checkpoint_payload["next_epoch"])
            if bool(checkpoint_payload.get("training_complete", False)):
                if best_state is None:
                    raise RuntimeError(f"Checkpoint at {resolved_checkpoint_path} has no best state.")
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
    epoch_progress = _create_progress_bar(total=max_epochs, desc=f"{progress_label or prepared_dataset.dataset_id} epochs", leave=True)
    try:
        if start_epoch > 1:
            epoch_progress.update(start_epoch - 1)
        for epoch_index in range(start_epoch, max_epochs + 1):
            model.train()
            train_loader = _build_turbine_dataloader(
                prepared_dataset,
                windows=prepared_dataset.train_windows,
                batch_size=batch_size,
                device=resolved_device,
                shuffle=True,
                seed=seed + epoch_index,
                num_workers=num_workers,
            )
            batch_progress = _create_progress_bar(total=_loader_batch_total(train_loader), desc=f"{progress_label or prepared_dataset.dataset_id} train e{epoch_index}")
            train_loss_sum = 0.0
            train_loss_weight = 0
            train_batch_count = 0
            train_valid_sum = 0.0
            train_target_count = 0
            latest_loss = float("nan")
            try:
                for raw_batch in train_loader:
                    (
                        batch_local_history,
                        batch_context_history,
                        batch_context_future,
                        batch_static,
                        batch_targets,
                        batch_valid,
                    ) = _move_batch_to_device(raw_batch, torch_module=resolved_torch, device=resolved_device)
                    optimizer.zero_grad(set_to_none=True)
                    with _amp_context(torch_module=resolved_torch, device=resolved_device, enabled=amp_enabled):
                        predictions = model(
                            batch_local_history,
                            batch_context_history,
                            batch_context_future,
                            batch_static,
                        )
                        loss = masked_huber_loss(predictions, batch_targets, batch_valid, torch_module=resolved_torch)
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
                    batch_weight = int(batch_local_history.shape[0])
                    train_valid_sum += float(batch_valid.sum().item())
                    train_target_count += int(batch_valid.numel())
                    latest_loss = float(loss.item())
                    train_loss_sum += latest_loss * batch_weight
                    train_loss_weight += batch_weight
                    train_batch_count += 1
                    batch_progress.set_postfix_str(f"loss={latest_loss:.4f}")
                    batch_progress.update(1)
            finally:
                batch_progress.close()
            epochs_ran = epoch_index
            val_metrics = evaluate_tft_model(
                model,
                prepared_dataset,
                prepared_dataset.val_rolling_windows,
                batch_size=resolved_eval_batch_size,
                device=resolved_device,
                seed=seed,
                num_workers=num_workers,
                amp_enabled=amp_enabled,
                progress_label=f"{progress_label or prepared_dataset.dataset_id} val e{epoch_index}",
            )
            val_mae_pu = float(val_metrics.mae_pu)
            val_rmse_pu = float(val_metrics.rmse_pu)
            is_best_epoch = False
            if best_state is None or (
                math.isfinite(val_mae_pu)
                and (not math.isfinite(best_val_mae_pu) or val_mae_pu < best_val_mae_pu - 1e-12)
            ):
                best_val_mae_pu = val_mae_pu
                best_val_rmse_pu = val_rmse_pu
                best_epoch = epoch_index
                best_state = copy.deepcopy(model.state_dict())
                epochs_without_improvement = 0
                is_best_epoch = True
            else:
                epochs_without_improvement += 1
            train_loss_mean = train_loss_sum / train_loss_weight if train_loss_weight else math.nan
            train_valid_fraction = train_valid_sum / train_target_count if train_target_count else math.nan
            val_valid_fraction = (
                float(val_metrics.prediction_count)
                / float(len(prepared_dataset.val_rolling_windows) * prepared_dataset.forecast_steps * prepared_dataset.node_count)
                if len(prepared_dataset.val_rolling_windows) > 0 and prepared_dataset.node_count > 0
                else math.nan
            )
            _tensorboard_add_scalar(tensorboard_writer, "train/loss_mean", train_loss_mean, epoch_index)
            _tensorboard_add_scalar(tensorboard_writer, "train/loss_last", latest_loss, epoch_index)
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
                "early_stopping/is_best_epoch",
                1 if is_best_epoch else 0,
                epoch_index,
            )
            _tensorboard_add_metrics(tensorboard_writer, "val", val_metrics, step=epoch_index)
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
                        "baseline_type": "shared_weight_tft_no_graph",
                        "patch_len": None,
                        "encoder_layers": None,
                        "ff_hidden_dim": None,
                        "epoch": epoch_index,
                        "train_loss_mean": train_loss_mean,
                        "train_loss_last": latest_loss,
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
            should_stop = epochs_without_improvement >= early_stopping_patience or epoch_index >= max_epochs
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
                        "best_val_mae_pu": best_val_mae_pu,
                        "best_val_rmse_pu": best_val_rmse_pu,
                        "epochs_without_improvement": epochs_without_improvement,
                        "epochs_ran": epochs_ran,
                        "next_epoch": epoch_index + 1,
                        "training_complete": should_stop,
                    },
                )
            epoch_progress.update(1)
            epoch_progress.set_postfix_str(f"loss={latest_loss:.4f} val_mae={val_mae_pu:.4f} best={best_val_mae_pu:.4f}")
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


def train_timexer_model(
    prepared_dataset: state_base.PreparedDataset,
    *,
    device: str,
    seed: int,
    batch_size: int,
    eval_batch_size: int | None,
    learning_rate: float,
    max_epochs: int,
    early_stopping_patience: int,
    d_model: int,
    attention_heads: int,
    patch_len: int,
    encoder_layers: int,
    ff_hidden_dim: int,
    dropout: float,
    grad_clip_norm: float,
    weight_decay: float,
    bounded_output_epsilon: float,
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
    resolved_eval_batch_size = resolve_eval_batch_size(batch_size, device=resolved_device, eval_batch_size=eval_batch_size)
    model = build_timexer_model(
        prepared_dataset=prepared_dataset,
        d_model=d_model,
        attention_heads=attention_heads,
        patch_len=patch_len,
        encoder_layers=encoder_layers,
        ff_hidden_dim=ff_hidden_dim,
        dropout=dropout,
        bounded_output_epsilon=bounded_output_epsilon,
    ).to(device=resolved_device)
    initialize_model_parameters(model)
    optimizer = resolved_torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scaler = _make_grad_scaler(resolved_torch, enabled=amp_enabled)
    resolved_checkpoint_path = Path(checkpoint_path) if checkpoint_path is not None else None
    resolved_training_history_path = Path(training_history_path) if training_history_path is not None else None
    checkpoint_job_identity = _job_identity_for_prepared_dataset(prepared_dataset)
    best_state: dict[str, Any] | None = None
    best_epoch = 0
    best_val_mae_pu = float("inf")
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
                raise RuntimeError(f"Checkpoint at {resolved_checkpoint_path} does not match the active job.")
            model.load_state_dict(checkpoint_payload["model_state_dict"])
            optimizer.load_state_dict(checkpoint_payload["optimizer_state_dict"])
            if "scaler_state_dict" in checkpoint_payload:
                scaler.load_state_dict(checkpoint_payload["scaler_state_dict"])
            best_state = checkpoint_payload.get("best_state_dict")
            best_epoch = int(checkpoint_payload["best_epoch"])
            best_val_mae_pu = float(checkpoint_payload["best_val_mae_pu"])
            best_val_rmse_pu = float(checkpoint_payload["best_val_rmse_pu"])
            epochs_without_improvement = int(checkpoint_payload["epochs_without_improvement"])
            epochs_ran = int(checkpoint_payload["epochs_ran"])
            start_epoch = int(checkpoint_payload["next_epoch"])
            if bool(checkpoint_payload.get("training_complete", False)):
                if best_state is None:
                    raise RuntimeError(f"Checkpoint at {resolved_checkpoint_path} has no best state.")
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
    epoch_progress = _create_progress_bar(total=max_epochs, desc=f"{progress_label or prepared_dataset.dataset_id} epochs", leave=True)
    try:
        if start_epoch > 1:
            epoch_progress.update(start_epoch - 1)
        for epoch_index in range(start_epoch, max_epochs + 1):
            model.train()
            train_loader = _build_timexer_dataloader(
                prepared_dataset,
                windows=prepared_dataset.train_windows,
                batch_size=batch_size,
                device=resolved_device,
                shuffle=True,
                seed=seed + epoch_index,
                num_workers=num_workers,
            )
            batch_progress = _create_progress_bar(total=_loader_batch_total(train_loader), desc=f"{progress_label or prepared_dataset.dataset_id} train e{epoch_index}")
            train_loss_sum = 0.0
            train_loss_weight = 0
            train_batch_count = 0
            train_valid_sum = 0.0
            train_target_count = 0
            latest_loss = float("nan")
            try:
                for raw_batch in train_loader:
                    (
                        batch_endogenous_history,
                        batch_exogenous_history,
                        batch_history_marks,
                        batch_targets,
                        batch_valid,
                    ) = _move_batch_to_device(raw_batch, torch_module=resolved_torch, device=resolved_device)
                    optimizer.zero_grad(set_to_none=True)
                    with _amp_context(torch_module=resolved_torch, device=resolved_device, enabled=amp_enabled):
                        predictions = model(
                            batch_endogenous_history,
                            batch_exogenous_history,
                            batch_history_marks,
                        )
                        loss = masked_mse_loss(predictions, batch_targets, batch_valid, torch_module=resolved_torch)
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
                    batch_weight = int(batch_endogenous_history.shape[0])
                    train_valid_sum += float(batch_valid.sum().item())
                    train_target_count += int(batch_valid.numel())
                    latest_loss = float(loss.item())
                    train_loss_sum += latest_loss * batch_weight
                    train_loss_weight += batch_weight
                    train_batch_count += 1
                    batch_progress.set_postfix_str(f"loss={latest_loss:.4f}")
                    batch_progress.update(1)
            finally:
                batch_progress.close()
            epochs_ran = epoch_index
            val_metrics = evaluate_timexer_model(
                model,
                prepared_dataset,
                prepared_dataset.val_rolling_windows,
                batch_size=resolved_eval_batch_size,
                device=resolved_device,
                seed=seed,
                num_workers=num_workers,
                amp_enabled=amp_enabled,
                progress_label=f"{progress_label or prepared_dataset.dataset_id} val e{epoch_index}",
            )
            val_mae_pu = float(val_metrics.mae_pu)
            val_rmse_pu = float(val_metrics.rmse_pu)
            is_best_epoch = False
            if best_state is None or (
                math.isfinite(val_mae_pu)
                and (not math.isfinite(best_val_mae_pu) or val_mae_pu < best_val_mae_pu - 1e-12)
            ):
                best_val_mae_pu = val_mae_pu
                best_val_rmse_pu = val_rmse_pu
                best_epoch = epoch_index
                best_state = copy.deepcopy(model.state_dict())
                epochs_without_improvement = 0
                is_best_epoch = True
            else:
                epochs_without_improvement += 1
            train_loss_mean = train_loss_sum / train_loss_weight if train_loss_weight else math.nan
            train_valid_fraction = train_valid_sum / train_target_count if train_target_count else math.nan
            val_valid_fraction = (
                float(val_metrics.prediction_count)
                / float(len(prepared_dataset.val_rolling_windows) * prepared_dataset.forecast_steps * prepared_dataset.node_count)
                if len(prepared_dataset.val_rolling_windows) > 0 and prepared_dataset.node_count > 0
                else math.nan
            )
            _tensorboard_add_scalar(tensorboard_writer, "train/loss_mean", train_loss_mean, epoch_index)
            _tensorboard_add_scalar(tensorboard_writer, "train/loss_last", latest_loss, epoch_index)
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
                "early_stopping/is_best_epoch",
                1 if is_best_epoch else 0,
                epoch_index,
            )
            _tensorboard_add_metrics(tensorboard_writer, "val", val_metrics, step=epoch_index)
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
                        "baseline_type": "shared_weight_timexer_no_graph",
                        "patch_len": patch_len,
                        "encoder_layers": encoder_layers,
                        "ff_hidden_dim": ff_hidden_dim,
                        "epoch": epoch_index,
                        "train_loss_mean": train_loss_mean,
                        "train_loss_last": latest_loss,
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
            should_stop = epochs_without_improvement >= early_stopping_patience or epoch_index >= max_epochs
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
                        "best_val_mae_pu": best_val_mae_pu,
                        "best_val_rmse_pu": best_val_rmse_pu,
                        "epochs_without_improvement": epochs_without_improvement,
                        "epochs_ran": epochs_ran,
                        "next_epoch": epoch_index + 1,
                        "training_complete": should_stop,
                    },
                )
            epoch_progress.update(1)
            epoch_progress.set_postfix_str(f"loss={latest_loss:.4f} val_mae={val_mae_pu:.4f} best={best_val_mae_pu:.4f}")
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
    prepared_dataset: state_base.PreparedDataset,
    *,
    variant_spec: ExperimentVariant,
    training_outcome: TrainingOutcome,
    runtime_seconds: float,
    seed: int,
    profile: HyperparameterProfile,
    evaluation_results: Sequence[tuple[str, str, world_model_base.FarmWindowDescriptorIndex, EvaluationMetrics]],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    is_tft = variant_spec.model_variant == TFT_VARIANT
    is_timexer = variant_spec.model_variant == TIMEXER_VARIANT
    base_row = {
        "dataset_id": prepared_dataset.dataset_id,
        "model_id": MODEL_ID,
        "model_variant": variant_spec.model_variant,
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
        "baseline_type": variant_spec.baseline_type,
        "uses_graph": False,
        "uses_pairwise": False,
        "uses_global_latent": False,
        "uses_future_observations": False,
        "bounded_output_epsilon": profile.bounded_output_epsilon if (is_tft or is_timexer) else None,
        "d_model": profile.d_model if (is_tft or is_timexer) else None,
        "lstm_hidden_dim": profile.lstm_hidden_dim if is_tft else None,
        "attention_heads": profile.attention_heads if (is_tft or is_timexer) else None,
        "patch_len": profile.patch_len if is_timexer else None,
        "encoder_layers": profile.encoder_layers if is_timexer else None,
        "ff_hidden_dim": profile.ff_hidden_dim if is_timexer else None,
        "dropout": profile.dropout if (is_tft or is_timexer) else None,
        "weight_decay": profile.weight_decay if (is_tft or is_timexer) else None,
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
        "batch_size": profile.batch_size if (is_tft or is_timexer) else None,
        "learning_rate": profile.learning_rate if (is_tft or is_timexer) else None,
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


def execute_persistence_job(
    prepared_dataset: state_base.PreparedDataset,
    *,
    variant_spec: ExperimentVariant,
    seed: int,
    profile: HyperparameterProfile,
    training_history_path: str | Path | None = None,
    tensorboard_log_dir: str | Path | None = None,
) -> list[dict[str, object]]:
    started = time.monotonic()
    writer = _open_tensorboard_writer(
        tensorboard_log_dir,
        dataset_id=prepared_dataset.dataset_id,
        model_variant=variant_spec.model_variant,
    )
    _tensorboard_log_run_config(
        writer,
        prepared_dataset=prepared_dataset,
        variant_spec=variant_spec,
        profile=profile,
        seed=seed,
        device="analytic",
        eval_batch_size=None,
        num_workers=None,
    )
    fallback = _train_history_target_mean(prepared_dataset)
    try:
        evaluation_results = [
            (split_name, eval_protocol, windows, evaluate_persistence(prepared_dataset, windows, train_fallback=fallback))
            for split_name, eval_protocol, windows in iter_evaluation_specs(prepared_dataset)
        ]
        _tensorboard_log_final_evaluations(writer, evaluation_results, step=0)
    finally:
        _close_tensorboard_writer(writer)
    val_rolling = next(
        metrics
        for split_name, eval_protocol, _windows, metrics in evaluation_results
        if split_name == "val" and eval_protocol == ROLLING_EVAL_PROTOCOL
    )
    if training_history_path is not None:
        _append_training_history_row(
            training_history_path,
            {
                "dataset_id": prepared_dataset.dataset_id,
                "model_id": MODEL_ID,
                "model_variant": variant_spec.model_variant,
                "feature_protocol_id": prepared_dataset.feature_protocol_id,
                "task_id": TASK_ID,
                "window_protocol": WINDOW_PROTOCOL,
                "baseline_type": variant_spec.baseline_type,
                "patch_len": None,
                "encoder_layers": None,
                "ff_hidden_dim": None,
                "epoch": 0,
                "train_loss_mean": None,
                "train_loss_last": None,
                "val_mae_pu": val_rolling.mae_pu,
                "val_rmse_pu": val_rolling.rmse_pu,
                "best_val_mae_pu": val_rolling.mae_pu,
                "best_val_rmse_pu": val_rolling.rmse_pu,
                "is_best_epoch": True,
                "epochs_without_improvement": 0,
                "train_batch_count": 0,
                "train_window_count": len(prepared_dataset.train_windows),
                "val_window_count": len(prepared_dataset.val_rolling_windows),
                "device": "analytic",
                "seed": seed,
                "batch_size": None,
                "learning_rate": None,
                "amp_enabled": False,
            },
        )
    outcome = TrainingOutcome(
        best_epoch=0,
        epochs_ran=0,
        best_val_rmse_pu=float(val_rolling.rmse_pu),
        best_val_mae_pu=float(val_rolling.mae_pu),
        device="analytic",
        amp_enabled=False,
        model=None,
    )
    return build_result_rows(
        prepared_dataset,
        variant_spec=variant_spec,
        training_outcome=outcome,
        runtime_seconds=time.monotonic() - started,
        seed=seed,
        profile=profile,
        evaluation_results=evaluation_results,
    )


def execute_tft_job(
    prepared_dataset: state_base.PreparedDataset,
    *,
    variant_spec: ExperimentVariant,
    device: str | None,
    seed: int,
    profile: HyperparameterProfile,
    eval_batch_size: int | None,
    num_workers: int | None,
    checkpoint_path: str | Path | None,
    training_history_path: str | Path | None,
    resume_from_checkpoint: bool,
    tensorboard_log_dir: str | Path | None,
) -> list[dict[str, object]]:
    started = time.monotonic()
    resolved_device = resolve_device(device)
    resolved_eval_batch_size = resolve_eval_batch_size(
        profile.batch_size,
        device=resolved_device,
        eval_batch_size=eval_batch_size,
    )
    writer = _open_tensorboard_writer(
        tensorboard_log_dir,
        dataset_id=prepared_dataset.dataset_id,
        model_variant=variant_spec.model_variant,
    )
    _tensorboard_log_run_config(
        writer,
        prepared_dataset=prepared_dataset,
        variant_spec=variant_spec,
        profile=profile,
        seed=seed,
        device=resolved_device,
        eval_batch_size=resolved_eval_batch_size,
        num_workers=resolve_loader_num_workers(device=resolved_device, num_workers=num_workers),
    )
    try:
        training_outcome = train_tft_model(
            prepared_dataset,
            device=resolved_device,
            seed=seed,
            batch_size=profile.batch_size,
            eval_batch_size=resolved_eval_batch_size,
            learning_rate=profile.learning_rate,
            max_epochs=profile.max_epochs,
            early_stopping_patience=profile.early_stopping_patience,
            d_model=profile.d_model,
            lstm_hidden_dim=profile.lstm_hidden_dim,
            attention_heads=profile.attention_heads,
            dropout=profile.dropout,
            grad_clip_norm=profile.grad_clip_norm,
            weight_decay=profile.weight_decay,
            bounded_output_epsilon=profile.bounded_output_epsilon,
            num_workers=num_workers,
            checkpoint_path=checkpoint_path,
            training_history_path=training_history_path,
            resume_from_checkpoint=resume_from_checkpoint,
            progress_label=f"{prepared_dataset.dataset_id}/{variant_spec.model_variant}",
            tensorboard_writer=writer,
        )
        evaluation_results: list[tuple[str, str, world_model_base.FarmWindowDescriptorIndex, EvaluationMetrics]] = []
        for split_name, eval_protocol, windows in iter_evaluation_specs(prepared_dataset):
            metrics = evaluate_tft_model(
                training_outcome.model,
                prepared_dataset,
                windows,
                batch_size=resolved_eval_batch_size,
                device=training_outcome.device,
                seed=seed,
                num_workers=num_workers,
                amp_enabled=training_outcome.amp_enabled,
                progress_label=f"{prepared_dataset.dataset_id}/{variant_spec.model_variant} {split_name}/{eval_protocol}",
            )
            evaluation_results.append((split_name, eval_protocol, windows, metrics))
        _tensorboard_log_final_evaluations(writer, evaluation_results, step=training_outcome.epochs_ran)
    finally:
        _close_tensorboard_writer(writer)
    return build_result_rows(
        prepared_dataset,
        variant_spec=variant_spec,
        training_outcome=training_outcome,
        runtime_seconds=time.monotonic() - started,
        seed=seed,
        profile=profile,
        evaluation_results=evaluation_results,
    )


def execute_timexer_job(
    prepared_dataset: state_base.PreparedDataset,
    *,
    variant_spec: ExperimentVariant,
    device: str | None,
    seed: int,
    profile: HyperparameterProfile,
    eval_batch_size: int | None,
    num_workers: int | None,
    checkpoint_path: str | Path | None,
    training_history_path: str | Path | None,
    resume_from_checkpoint: bool,
    tensorboard_log_dir: str | Path | None,
) -> list[dict[str, object]]:
    started = time.monotonic()
    resolved_device = resolve_device(device)
    resolved_eval_batch_size = resolve_eval_batch_size(
        profile.batch_size,
        device=resolved_device,
        eval_batch_size=eval_batch_size,
    )
    writer = _open_tensorboard_writer(
        tensorboard_log_dir,
        dataset_id=prepared_dataset.dataset_id,
        model_variant=variant_spec.model_variant,
    )
    _tensorboard_log_run_config(
        writer,
        prepared_dataset=prepared_dataset,
        variant_spec=variant_spec,
        profile=profile,
        seed=seed,
        device=resolved_device,
        eval_batch_size=resolved_eval_batch_size,
        num_workers=resolve_loader_num_workers(device=resolved_device, num_workers=num_workers),
    )
    try:
        training_outcome = train_timexer_model(
            prepared_dataset,
            device=resolved_device,
            seed=seed,
            batch_size=profile.batch_size,
            eval_batch_size=resolved_eval_batch_size,
            learning_rate=profile.learning_rate,
            max_epochs=profile.max_epochs,
            early_stopping_patience=profile.early_stopping_patience,
            d_model=profile.d_model,
            attention_heads=profile.attention_heads,
            patch_len=profile.patch_len,
            encoder_layers=profile.encoder_layers,
            ff_hidden_dim=profile.ff_hidden_dim,
            dropout=profile.dropout,
            grad_clip_norm=profile.grad_clip_norm,
            weight_decay=profile.weight_decay,
            bounded_output_epsilon=profile.bounded_output_epsilon,
            num_workers=num_workers,
            checkpoint_path=checkpoint_path,
            training_history_path=training_history_path,
            resume_from_checkpoint=resume_from_checkpoint,
            progress_label=f"{prepared_dataset.dataset_id}/{variant_spec.model_variant}",
            tensorboard_writer=writer,
        )
        evaluation_results: list[tuple[str, str, world_model_base.FarmWindowDescriptorIndex, EvaluationMetrics]] = []
        for split_name, eval_protocol, windows in iter_evaluation_specs(prepared_dataset):
            metrics = evaluate_timexer_model(
                training_outcome.model,
                prepared_dataset,
                windows,
                batch_size=resolved_eval_batch_size,
                device=training_outcome.device,
                seed=seed,
                num_workers=num_workers,
                amp_enabled=training_outcome.amp_enabled,
                progress_label=f"{prepared_dataset.dataset_id}/{variant_spec.model_variant} {split_name}/{eval_protocol}",
            )
            evaluation_results.append((split_name, eval_protocol, windows, metrics))
        _tensorboard_log_final_evaluations(writer, evaluation_results, step=training_outcome.epochs_ran)
    finally:
        _close_tensorboard_writer(writer)
    return build_result_rows(
        prepared_dataset,
        variant_spec=variant_spec,
        training_outcome=training_outcome,
        runtime_seconds=time.monotonic() - started,
        seed=seed,
        profile=profile,
        evaluation_results=evaluation_results,
    )


def execute_training_job(
    prepared_dataset: state_base.PreparedDataset,
    *,
    variant_spec: ExperimentVariant | None = None,
    device: str | None = None,
    seed: int = DEFAULT_SEED,
    batch_size: int = DEFAULT_BATCH_SIZE,
    eval_batch_size: int | None = None,
    learning_rate: float = DEFAULT_LEARNING_RATE,
    max_epochs: int = DEFAULT_MAX_EPOCHS,
    early_stopping_patience: int = DEFAULT_EARLY_STOPPING_PATIENCE,
    d_model: int = DEFAULT_D_MODEL,
    lstm_hidden_dim: int = DEFAULT_LSTM_HIDDEN_DIM,
    attention_heads: int = DEFAULT_ATTENTION_HEADS,
    patch_len: int = DEFAULT_TIMEXER_PATCH_LEN,
    encoder_layers: int = DEFAULT_TIMEXER_ENCODER_LAYERS,
    ff_hidden_dim: int = DEFAULT_TIMEXER_FF_HIDDEN_DIM,
    dropout: float = DEFAULT_DROPOUT,
    grad_clip_norm: float = DEFAULT_GRAD_CLIP_NORM,
    weight_decay: float = DEFAULT_WEIGHT_DECAY,
    bounded_output_epsilon: float = DEFAULT_BOUNDED_OUTPUT_EPSILON,
    num_workers: int | None = None,
    checkpoint_path: str | Path | None = None,
    training_history_path: str | Path | None = None,
    resume_from_checkpoint: bool = False,
    tensorboard_log_dir: str | Path | None = None,
) -> list[dict[str, object]]:
    resolved_variant = variant_spec or _VARIANT_SPECS_BY_NAME.get(prepared_dataset.model_variant, VARIANT_SPECS[0])
    profile = HyperparameterProfile(
        batch_size=batch_size,
        learning_rate=learning_rate,
        max_epochs=max_epochs,
        early_stopping_patience=early_stopping_patience,
        d_model=d_model,
        lstm_hidden_dim=lstm_hidden_dim,
        attention_heads=attention_heads,
        patch_len=patch_len,
        encoder_layers=encoder_layers,
        ff_hidden_dim=ff_hidden_dim,
        dropout=dropout,
        grad_clip_norm=grad_clip_norm,
        weight_decay=weight_decay,
        bounded_output_epsilon=bounded_output_epsilon,
    )
    if resolved_variant.model_variant == PERSISTENCE_VARIANT:
        return execute_persistence_job(
            prepared_dataset,
            variant_spec=resolved_variant,
            seed=seed,
            profile=profile,
            training_history_path=training_history_path,
            tensorboard_log_dir=tensorboard_log_dir,
        )
    if resolved_variant.model_variant == TFT_VARIANT:
        return execute_tft_job(
            prepared_dataset,
            variant_spec=resolved_variant,
            device=device,
            seed=seed,
            profile=profile,
            eval_batch_size=eval_batch_size,
            num_workers=num_workers,
            checkpoint_path=checkpoint_path,
            training_history_path=training_history_path,
            resume_from_checkpoint=resume_from_checkpoint,
            tensorboard_log_dir=tensorboard_log_dir,
        )
    if resolved_variant.model_variant == TIMEXER_VARIANT:
        return execute_timexer_job(
            prepared_dataset,
            variant_spec=resolved_variant,
            device=device,
            seed=seed,
            profile=profile,
            eval_batch_size=eval_batch_size,
            num_workers=num_workers,
            checkpoint_path=checkpoint_path,
            training_history_path=training_history_path,
            resume_from_checkpoint=resume_from_checkpoint,
            tensorboard_log_dir=tensorboard_log_dir,
        )
    raise ValueError(f"Unsupported baseline variant {resolved_variant.model_variant!r}.")


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


def tensorboard_root_path(
    output_path: str | Path,
    *,
    work_root: str | Path = _RUN_WORK_ROOT,
) -> Path:
    return _resume_paths_for_output(output_path=output_path, work_root=work_root).slot_dir / "tensorboard"


def resolve_tensorboard_root(
    *,
    output_path: str | Path,
    work_root: str | Path,
    tensorboard_log_dir: str | Path | None = None,
    disable_tensorboard: bool = False,
) -> Path | None:
    if disable_tensorboard:
        return None
    if tensorboard_log_dir is not None:
        return Path(tensorboard_log_dir).expanduser().resolve()
    return tensorboard_root_path(output_path, work_root=work_root)


def _job_key(dataset_id: str, model_variant: str) -> tuple[str, str]:
    return dataset_id, model_variant


def _job_identity(*, dataset_id: str, model_variant: str, feature_protocol_id: str) -> dict[str, str]:
    return {
        "dataset_id": dataset_id,
        "model_variant": model_variant,
        "feature_protocol_id": feature_protocol_id,
    }


def _job_identity_for_prepared_dataset(prepared_dataset: state_base.PreparedDataset) -> dict[str, str]:
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
            pl.col("dataset_id").replace_strict(_DATASET_ORDER, default=len(_DATASET_ORDER)).alias("__dataset_order"),
            pl.col("model_variant").replace_strict(_MODEL_VARIANT_ORDER, default=len(_MODEL_VARIANT_ORDER)).alias("__model_variant_order"),
            pl.col("split_name").replace_strict(_SPLIT_ORDER, default=len(_SPLIT_ORDER)).alias("__split_order"),
            pl.col("eval_protocol").replace_strict(_EVAL_PROTOCOL_ORDER, default=len(_EVAL_PROTOCOL_ORDER)).alias("__eval_protocol_order"),
            pl.col("metric_scope").replace_strict(_METRIC_SCOPE_ORDER, default=len(_METRIC_SCOPE_ORDER)).alias("__metric_scope_order"),
            pl.col("lead_step").fill_null(0).alias("__lead_order"),
        )
        .sort(["__dataset_order", "__model_variant_order", "__split_order", "__eval_protocol_order", "__metric_scope_order", "__lead_order"])
        .drop(["__dataset_order", "__model_variant_order", "__split_order", "__eval_protocol_order", "__metric_scope_order", "__lead_order"])
    )


def sort_training_history_frame(frame: pl.DataFrame) -> pl.DataFrame:
    if frame.is_empty():
        return frame.select(_TRAINING_HISTORY_COLUMNS)
    return (
        frame.select(_TRAINING_HISTORY_COLUMNS)
        .with_columns(
            pl.col("dataset_id").replace_strict(_DATASET_ORDER, default=len(_DATASET_ORDER)).alias("__dataset_order"),
            pl.col("model_variant").replace_strict(_MODEL_VARIANT_ORDER, default=len(_MODEL_VARIANT_ORDER)).alias("__model_variant_order"),
        )
        .sort(["__dataset_order", "__model_variant_order", "epoch"])
        .drop(["__dataset_order", "__model_variant_order"])
    )


def _load_resume_state(paths: ResumePaths) -> dict[str, object]:
    payload = json.loads(paths.state_path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != _RUN_STATE_SCHEMA_VERSION:
        raise ValueError(f"Unsupported {FAMILY_ID} resume state schema at {paths.state_path}: {payload.get('schema_version')!r}.")
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
        raise ValueError(f"Partial results at {paths.partial_results_path} are missing expected columns: {missing_columns!r}.")
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
        raise ValueError(f"Training history at {history_path} is missing expected columns: {missing_columns!r}.")
    return sort_training_history_frame(frame.select(_TRAINING_HISTORY_COLUMNS))


def _write_training_history(path: str | Path, frame: pl.DataFrame) -> None:
    _atomic_write_csv(Path(path), sort_training_history_frame(frame))


def _training_history_job_expr(job_identity: dict[str, str]) -> pl.Expr:
    return (
        (pl.col("dataset_id") == job_identity["dataset_id"])
        & (pl.col("model_variant") == job_identity["model_variant"])
        & (pl.col("feature_protocol_id") == job_identity["feature_protocol_id"])
    )


def _prune_training_history_for_job(path: str | Path, *, job_identity: dict[str, str], min_epoch: int) -> None:
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
        retained = frame.filter(~(_training_history_job_expr(job_identity) & (pl.col("epoch") == int(row["epoch"]))))
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
    d_model: int | None,
    lstm_hidden_dim: int | None,
    attention_heads: int | None,
    patch_len: int | None,
    encoder_layers: int | None,
    ff_hidden_dim: int | None,
    dropout: float | None,
    grad_clip_norm: float | None,
    weight_decay: float | None,
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
                    "baseline_type": spec.baseline_type,
                    **asdict(
                        resolve_hyperparameter_profile(
                            spec.model_variant,
                            dataset_id=dataset_id,
                            batch_size=batch_size,
                            learning_rate=learning_rate,
                            max_epochs=max_epochs,
                            early_stopping_patience=early_stopping_patience,
                            d_model=d_model,
                            lstm_hidden_dim=lstm_hidden_dim,
                            attention_heads=attention_heads,
                            patch_len=patch_len,
                            encoder_layers=encoder_layers,
                            ff_hidden_dim=ff_hidden_dim,
                            dropout=dropout,
                            grad_clip_norm=grad_clip_norm,
                            weight_decay=weight_decay,
                            bounded_output_epsilon=bounded_output_epsilon,
                        )
                    ),
                }
                for spec in variant_specs
            }
            for dataset_id in dataset_ids
        },
    }


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
    d_model: int | None = None,
    lstm_hidden_dim: int | None = None,
    attention_heads: int | None = None,
    patch_len: int | None = None,
    encoder_layers: int | None = None,
    ff_hidden_dim: int | None = None,
    dropout: float | None = None,
    grad_clip_norm: float | None = None,
    weight_decay: float | None = None,
    bounded_output_epsilon: float | None = None,
    num_workers: int | None = None,
    tensorboard_log_dir: str | Path | None = None,
    disable_tensorboard: bool = False,
    resume: bool = False,
    force_rerun: bool = False,
    work_root: str | Path = _RUN_WORK_ROOT,
    dataset_loader: Callable[..., state_base.PreparedDataset] | None = None,
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
        d_model=d_model,
        lstm_hidden_dim=lstm_hidden_dim,
        attention_heads=attention_heads,
        patch_len=patch_len,
        encoder_layers=encoder_layers,
        ff_hidden_dim=ff_hidden_dim,
        dropout=dropout,
        grad_clip_norm=grad_clip_norm,
        weight_decay=weight_decay,
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
            raise ValueError(f"Resume state at {resume_paths.state_path} does not match the requested run configuration.")
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
    resume_active_job = None if existing_state is None else existing_state.get("active_job")
    prepared_cache: dict[str, state_base.PreparedDataset] = {}
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
                    job_progress.update(1)
                    continue
                if dataset_loader is None:
                    if dataset_id not in prepared_cache:
                        prepared_cache[dataset_id] = prepare_dataset(
                            dataset_id,
                            variant_spec=variant_spec,
                            cache_root=cache_root,
                            max_train_origins=max_train_origins,
                            max_eval_origins=max_eval_origins,
                        )
                    prepared = replace(prepared_cache[dataset_id], model_variant=variant_spec.model_variant)
                else:
                    prepared = dataset_loader(
                        dataset_id,
                        variant_spec=variant_spec,
                        cache_root=cache_root,
                        max_train_origins=max_train_origins,
                        max_eval_origins=max_eval_origins,
                    )
                profile = resolve_hyperparameter_profile(
                    variant_spec.model_variant,
                    dataset_id=dataset_id,
                    batch_size=batch_size,
                    learning_rate=learning_rate,
                    max_epochs=max_epochs,
                    early_stopping_patience=early_stopping_patience,
                    d_model=d_model,
                    lstm_hidden_dim=lstm_hidden_dim,
                    attention_heads=attention_heads,
                    patch_len=patch_len,
                    encoder_layers=encoder_layers,
                    ff_hidden_dim=ff_hidden_dim,
                    dropout=dropout,
                    grad_clip_norm=grad_clip_norm,
                    weight_decay=weight_decay,
                    bounded_output_epsilon=bounded_output_epsilon,
                )
                checkpoint_path = _job_checkpoint_path(
                    resume_paths,
                    dataset_id=dataset_id,
                    model_variant=variant_spec.model_variant,
                )
                resume_from_checkpoint = resume_active_job == current_job_identity and checkpoint_path.exists()
                _write_resume_state(resume_paths, status="running", effective_config=effective_config, active_job=current_job_identity)
                rows.extend(
                    runner(
                        prepared,
                        variant_spec=variant_spec,
                        device=device,
                        seed=seed,
                        batch_size=profile.batch_size,
                        eval_batch_size=eval_batch_size,
                        learning_rate=profile.learning_rate,
                        max_epochs=profile.max_epochs,
                        early_stopping_patience=profile.early_stopping_patience,
                        d_model=profile.d_model,
                        lstm_hidden_dim=profile.lstm_hidden_dim,
                        attention_heads=profile.attention_heads,
                        patch_len=profile.patch_len,
                        encoder_layers=profile.encoder_layers,
                        ff_hidden_dim=profile.ff_hidden_dim,
                        dropout=profile.dropout,
                        grad_clip_norm=profile.grad_clip_norm,
                        weight_decay=profile.weight_decay,
                        bounded_output_epsilon=profile.bounded_output_epsilon,
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
                _delete_job_checkpoint(resume_paths, dataset_id=dataset_id, model_variant=variant_spec.model_variant)
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
    parser = argparse.ArgumentParser(description="Run world_model_baselines_v1 on Kelmarsh.")
    parser.add_argument("--dataset", action="append", choices=list(DEFAULT_DATASETS), dest="datasets")
    parser.add_argument("--variant", action="append", choices=list(DEFAULT_VARIANTS), dest="variants")
    parser.add_argument("--device", choices=("auto", "cuda", "mps", "cpu"), default="auto")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--patience", type=int, default=None)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--eval-batch-size", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--max-train-origins", type=int, default=None)
    parser.add_argument("--max-eval-origins", type=int, default=None)
    parser.add_argument("--output-path", type=Path, default=_OUTPUT_PATH)
    parser.add_argument("--d-model", type=int, default=None)
    parser.add_argument("--lstm-hidden-dim", type=int, default=None)
    parser.add_argument("--attention-heads", type=int, default=None)
    parser.add_argument("--patch-len", type=int, default=None)
    parser.add_argument("--encoder-layers", type=int, default=None)
    parser.add_argument("--ff-hidden-dim", type=int, default=None)
    parser.add_argument("--dropout", type=float, default=None)
    parser.add_argument("--grad-clip-norm", type=float, default=None)
    parser.add_argument("--weight-decay", type=float, default=None)
    parser.add_argument("--bounded-output-epsilon", type=float, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--tensorboard-log-dir", type=Path, default=None)
    parser.add_argument("--disable-tensorboard", action="store_true")
    parser.add_argument("--run-label", type=str, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--force-rerun", action="store_true")
    parser.add_argument("--no-record-run", action="store_true")
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
                "baseline_type": spec.baseline_type,
                **asdict(
                    resolve_hyperparameter_profile(
                        spec.model_variant,
                        dataset_id=dataset_id,
                        batch_size=args.batch_size,
                        learning_rate=args.learning_rate,
                        max_epochs=args.epochs,
                        early_stopping_patience=args.patience,
                        d_model=args.d_model,
                        lstm_hidden_dim=args.lstm_hidden_dim,
                        attention_heads=args.attention_heads,
                        patch_len=args.patch_len,
                        encoder_layers=args.encoder_layers,
                        ff_hidden_dim=args.ff_hidden_dim,
                        dropout=args.dropout,
                        grad_clip_norm=args.grad_clip_norm,
                        weight_decay=args.weight_decay,
                        bounded_output_epsilon=args.bounded_output_epsilon,
                    )
                ),
            }
            for spec in variant_specs
        }
        for dataset_id in dataset_ids
    }


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
        d_model=args.d_model,
        lstm_hidden_dim=args.lstm_hidden_dim,
        attention_heads=args.attention_heads,
        patch_len=args.patch_len,
        encoder_layers=args.encoder_layers,
        ff_hidden_dim=args.ff_hidden_dim,
        dropout=args.dropout,
        grad_clip_norm=args.grad_clip_norm,
        weight_decay=args.weight_decay,
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
            entrypoint=f"experiment/families/{FAMILY_ID}/run_world_model_baselines_v1.py",
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
