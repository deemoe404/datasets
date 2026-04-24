from __future__ import annotations

import argparse
from dataclasses import asdict, replace
from datetime import UTC, datetime
import json
import math
from types import SimpleNamespace
from pathlib import Path
import sys
import time
from typing import Any, Sequence

import numpy as np
import polars as pl

EXPERIMENT_DIR = Path(__file__).resolve().parent
EXPERIMENT_ROOT = EXPERIMENT_DIR.parents[1]
REPO_ROOT = EXPERIMENT_ROOT.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiment.infra.common.run_records import record_cli_run  # noqa: E402
from experiment.infra.common.window_protocols import NON_OVERLAP_EVAL_PROTOCOL, ROLLING_EVAL_PROTOCOL  # noqa: E402

from world_model_official_baselines_v2 import (  # noqa: E402
    DEFAULT_DATASETS,
    DEFAULT_VARIANTS,
    FAMILY_ID,
    FEATURE_PROTOCOL_ID,
    FORECAST_STEPS,
    HISTORY_STEPS,
    CHRONOS2_VARIANT,
    DGCRN_DIRECT_VARIANT,
    DGCRN_RESIDUAL_VARIANT,
    ITRANSFORMER_TARGET_DIRECT_VARIANT,
    ITRANSFORMER_TARGET_RESIDUAL_VARIANT,
    PERSISTENCE_VARIANT,
    RIDGE_RESIDUAL_VARIANT,
    SEASONAL_PERSISTENCE_VARIANT,
    TIMEXER_TARGET_DIRECT_VARIANT,
    TIMEXER_TARGET_RESIDUAL_VARIANT,
    TASK_ID,
    OfficialVariantSpec,
    resolve_variant_specs,
)

FORMAL_SUPPORTED_VARIANTS = {
    PERSISTENCE_VARIANT,
    SEASONAL_PERSISTENCE_VARIANT,
    RIDGE_RESIDUAL_VARIANT,
    CHRONOS2_VARIANT,
    DGCRN_DIRECT_VARIANT,
    DGCRN_RESIDUAL_VARIANT,
    ITRANSFORMER_TARGET_DIRECT_VARIANT,
    ITRANSFORMER_TARGET_RESIDUAL_VARIANT,
    TIMEXER_TARGET_DIRECT_VARIANT,
    TIMEXER_TARGET_RESIDUAL_VARIANT,
}
FORMAL_BLOCKER_BY_VARIANT_PREFIX = {
    "baseline_mlp_residual": "residual_control_training_not_implemented",
    "baseline_gru_residual": "residual_control_training_not_implemented",
    "baseline_tcn_residual": "residual_control_training_not_implemented",
    "timexer_official_full_exog": "official_full_exog_adapter_not_implemented",
    "itransformer_official_target_plus_exog": "official_exog_adapter_not_implemented",
    "tft_pf": "pytorch_forecasting_training_adapter_not_implemented",
    "mtgnn_official_core": "official_core_training_adapter_not_implemented",
}
DEFAULT_RIDGE_ALPHAS = (0.0, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0)
RIDGE_LAGS = (1, 2, 3, 6, 12, 18, 36, 72, 144)


def formal_support_status(spec: OfficialVariantSpec) -> tuple[str, str | None]:
    if spec.model_variant in FORMAL_SUPPORTED_VARIANTS:
        return "supported", None
    for prefix, reason in FORMAL_BLOCKER_BY_VARIANT_PREFIX.items():
        if spec.model_variant.startswith(prefix):
            return "blocked", reason
    return "blocked", "formal_tuning_support_not_declared"


def _load_state_space_base():
    state_space_dir = EXPERIMENT_DIR.parent / "world_model_state_space_v1"
    if str(state_space_dir) not in sys.path:
        sys.path.insert(0, str(state_space_dir))
    import world_model_state_space_v1 as state_base  # type: ignore

    return state_base


def _prepare_dataset(dataset_id: str, *, max_train_origins: int | None, max_eval_origins: int | None):
    state_base = _load_state_space_base()
    return state_base.prepare_dataset(
        dataset_id,
        cache_root=REPO_ROOT / "cache",
        max_train_origins=max_train_origins,
        max_eval_origins=max_eval_origins,
    )


def _windows_by_split(prepared: Any) -> tuple[tuple[str, str, Any], ...]:
    return (
        ("val", ROLLING_EVAL_PROTOCOL, prepared.val_rolling_windows),
        ("val", NON_OVERLAP_EVAL_PROTOCOL, prepared.val_non_overlap_windows),
        ("test", ROLLING_EVAL_PROTOCOL, prepared.test_rolling_windows),
        ("test", NON_OVERLAP_EVAL_PROTOCOL, prepared.test_non_overlap_windows),
    )


def _selected_window_specs(
    prepared: Any,
    *,
    split_names: Sequence[str] | None = None,
    eval_protocols: Sequence[str] | None = None,
) -> tuple[tuple[str, str, Any], ...]:
    split_filter = set(split_names or ())
    eval_filter = set(eval_protocols or ())
    return tuple(
        (split_name, eval_protocol, windows)
        for split_name, eval_protocol, windows in _windows_by_split(prepared)
        if (not split_filter or split_name in split_filter) and (not eval_filter or eval_protocol in eval_filter)
    )


def _limit_windows(windows: Any, max_origins: int | None) -> Any:
    if max_origins is None or len(windows) <= max_origins:
        return windows
    keep = np.arange(int(max_origins), dtype=np.int64)
    return replace(
        windows,
        target_indices=windows.target_indices[keep],
        output_start_us=windows.output_start_us[keep],
        output_end_us=windows.output_end_us[keep],
    )


def _validation_windows_for_checkpoint(
    prepared: Any,
    *,
    eval_protocol: str,
    max_origins: int | None,
) -> Any:
    if eval_protocol == ROLLING_EVAL_PROTOCOL:
        windows = prepared.val_rolling_windows
    elif eval_protocol == NON_OVERLAP_EVAL_PROTOCOL:
        windows = prepared.val_non_overlap_windows
    else:
        raise ValueError(f"Unsupported checkpoint eval protocol {eval_protocol!r}.")
    return _limit_windows(windows, max_origins)


def _target_and_valid(prepared: Any, windows: Any) -> tuple[np.ndarray, np.ndarray]:
    targets = np.zeros((len(windows), prepared.forecast_steps, prepared.node_count), dtype=np.float32)
    valid = np.zeros_like(targets, dtype=np.float32)
    for window_pos, target_index in enumerate(windows.target_indices):
        future = slice(int(target_index), int(target_index) + prepared.forecast_steps)
        targets[window_pos] = prepared.target_pu_filled[future]
        valid[window_pos] = prepared.target_valid_mask[future].astype(np.float32, copy=False)
    return targets, valid


def _last_value_anchor(prepared: Any, windows: Any) -> np.ndarray:
    anchor = np.zeros((len(windows), prepared.node_count), dtype=np.float32)
    fallback = prepared.persistence_train_fallback_pu.astype(np.float32, copy=True)
    for window_pos, target_index in enumerate(windows.target_indices):
        history = slice(int(target_index) - prepared.history_steps, int(target_index))
        values = prepared.local_history_tensor[history, :, 0]
        unavailable = prepared.local_history_tensor[history, :, 17]
        available = unavailable < 0.5
        last_values = fallback.copy()
        for node_index in range(prepared.node_count):
            valid_positions = np.flatnonzero(available[:, node_index])
            if valid_positions.size:
                last_values[node_index] = values[int(valid_positions[-1]), node_index]
        anchor[window_pos] = last_values
    return anchor


def _seasonal_anchor(prepared: Any, windows: Any) -> np.ndarray:
    anchor = np.zeros((len(windows), prepared.node_count), dtype=np.float32)
    fallback = prepared.persistence_train_fallback_pu.astype(np.float32, copy=True)
    for window_pos, target_index in enumerate(windows.target_indices):
        source_index = int(target_index) - prepared.history_steps
        values = prepared.target_pu_filled[source_index]
        valid = prepared.target_valid_mask[source_index]
        anchor[window_pos] = np.where(valid, values, fallback)
    return anchor


def _repeat_anchor(anchor: np.ndarray, forecast_steps: int) -> np.ndarray:
    return np.repeat(anchor[:, None, :], forecast_steps, axis=1).astype(np.float32, copy=False)


def _metrics(predictions: np.ndarray, targets: np.ndarray, valid: np.ndarray, *, rated_power_kw: float) -> dict[str, Any]:
    valid_f = valid.astype(np.float64, copy=False)
    errors_pu = (predictions.astype(np.float64, copy=False) - targets.astype(np.float64, copy=False)) * valid_f
    prediction_count = int(valid_f.sum())
    if prediction_count <= 0:
        mae_pu = rmse_pu = mae_kw = rmse_kw = math.nan
    else:
        mae_pu = float(np.abs(errors_pu).sum() / prediction_count)
        rmse_pu = float(math.sqrt(np.square(errors_pu).sum() / prediction_count))
        mae_kw = mae_pu * float(rated_power_kw)
        rmse_kw = rmse_pu * float(rated_power_kw)
    lead_valid = valid_f.sum(axis=(0, 2))
    lead_abs = np.abs(errors_pu).sum(axis=(0, 2))
    lead_sq = np.square(errors_pu).sum(axis=(0, 2))
    lead_mae = np.divide(lead_abs, lead_valid, out=np.full_like(lead_abs, np.nan, dtype=np.float64), where=lead_valid > 0)
    lead_rmse = np.sqrt(
        np.divide(lead_sq, lead_valid, out=np.full_like(lead_sq, np.nan, dtype=np.float64), where=lead_valid > 0)
    )
    abs_errors = np.abs(errors_pu[valid_f > 0])
    return {
        "window_count": int(predictions.shape[0]),
        "prediction_count": prediction_count,
        "mae_pu": mae_pu,
        "rmse_pu": rmse_pu,
        "mae_kw": mae_kw,
        "rmse_kw": rmse_kw,
        "lead1_mae_pu": float(lead_mae[0]) if lead_mae.size else math.nan,
        "lead1_rmse_pu": float(lead_rmse[0]) if lead_rmse.size else math.nan,
        "short_rmse_pu": _lead_bucket_rmse(lead_sq, lead_valid, 1, 6),
        "mid_rmse_pu": _lead_bucket_rmse(lead_sq, lead_valid, 7, 18),
        "long_rmse_pu": _lead_bucket_rmse(lead_sq, lead_valid, 19, 36),
        "ae_p50": float(np.quantile(abs_errors, 0.50)) if abs_errors.size else math.nan,
        "ae_p90": float(np.quantile(abs_errors, 0.90)) if abs_errors.size else math.nan,
        "ae_p95": float(np.quantile(abs_errors, 0.95)) if abs_errors.size else math.nan,
    }


def _lead_bucket_rmse(lead_sq: np.ndarray, lead_valid: np.ndarray, start_step: int, end_step: int) -> float:
    start_index = max(0, start_step - 1)
    end_index = min(len(lead_sq), end_step)
    denominator = float(lead_valid[start_index:end_index].sum())
    if denominator <= 0:
        return math.nan
    return float(math.sqrt(float(lead_sq[start_index:end_index].sum()) / denominator))


def _ridge_features(prepared: Any, windows: Any) -> np.ndarray:
    features = np.ones((len(windows), 1 + len(RIDGE_LAGS) * prepared.node_count), dtype=np.float64)
    for row_index, target_index in enumerate(windows.target_indices):
        values = []
        for lag in RIDGE_LAGS:
            values.extend(prepared.target_pu_filled[int(target_index) - lag].tolist())
        features[row_index, 1:] = np.asarray(values, dtype=np.float64)
    return features


def _flatten_future_residuals(prepared: Any, windows: Any, anchor: np.ndarray) -> np.ndarray:
    targets, _valid = _target_and_valid(prepared, windows)
    return (targets - _repeat_anchor(anchor, prepared.forecast_steps)).reshape(len(windows), -1).astype(np.float64)


def _fit_ridge(features: np.ndarray, residuals: np.ndarray, *, alpha: float) -> np.ndarray:
    lhs = features.T @ features
    regularizer = np.eye(lhs.shape[0], dtype=np.float64) * float(alpha)
    regularizer[0, 0] = 0.0
    rhs = features.T @ residuals
    return np.linalg.solve(lhs + regularizer, rhs)


def _predict_ridge(prepared: Any, windows: Any, weights: np.ndarray) -> np.ndarray:
    anchor = _last_value_anchor(prepared, windows)
    residual = (_ridge_features(prepared, windows) @ weights).reshape(len(windows), prepared.forecast_steps, prepared.node_count)
    return (_repeat_anchor(anchor, prepared.forecast_steps) + residual).astype(np.float32)


def _load_chronos2_pipeline(*, device: str):
    from chronos import Chronos2Pipeline  # type: ignore

    pipeline = Chronos2Pipeline.from_pretrained("amazon/chronos-2", torch_dtype="float32")
    model = getattr(pipeline, "model", None)
    if model is not None and hasattr(model, "to"):
        model.to(device)
        if hasattr(model, "eval"):
            model.eval()
    return pipeline


def _chronos_target_and_covariate_names(prepared: Any) -> tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...]]:
    state_base = _load_state_space_base()
    local_value_names = prepared.local_input_feature_names[
        state_base._LOCAL_VALUE_START + 1 : state_base._LOCAL_MASK_START
    ]
    global_value_names = prepared.context_history_feature_names[
        state_base._CONTEXT_GLOBAL_VALUE_START : state_base._CONTEXT_GLOBAL_MASK_START
    ]
    calendar_names = prepared.context_history_feature_names[state_base._CONTEXT_CALENDAR_START :]
    return tuple(local_value_names), tuple(global_value_names), tuple(calendar_names)


def _chronos_input_for(prepared: Any, *, target_index: int, node_index: int) -> dict[str, Any]:
    state_base = _load_state_space_base()
    history = slice(target_index - prepared.history_steps, target_index)
    future = slice(target_index, target_index + prepared.forecast_steps)
    local_value_names, global_value_names, calendar_names = _chronos_target_and_covariate_names(prepared)
    target = prepared.local_history_tensor[history, node_index, state_base._LOCAL_VALUE_START].astype(np.float32, copy=True)
    target_mask = prepared.local_history_tensor[history, node_index, state_base._LOCAL_MASK_START]
    target[target_mask >= 0.5] = np.nan

    past_covariates: dict[str, np.ndarray] = {}
    local_values = prepared.local_history_tensor[
        history,
        node_index,
        state_base._LOCAL_VALUE_START + 1 : state_base._LOCAL_MASK_START,
    ]
    local_masks = prepared.local_history_tensor[
        history,
        node_index,
        state_base._LOCAL_MASK_START + 1 : state_base._LOCAL_DELTA_START,
    ]
    for column_index, column_name in enumerate(local_value_names):
        values = local_values[:, column_index].astype(np.float32, copy=True)
        values[local_masks[:, column_index] >= 0.5] = np.nan
        past_covariates[column_name] = values

    global_values = prepared.context_history_tensor[
        history,
        state_base._CONTEXT_GLOBAL_VALUE_START : state_base._CONTEXT_GLOBAL_MASK_START,
    ]
    global_masks = prepared.context_history_tensor[
        history,
        state_base._CONTEXT_GLOBAL_MASK_START : state_base._CONTEXT_GLOBAL_DELTA_START,
    ]
    for column_index, column_name in enumerate(global_value_names):
        values = global_values[:, column_index].astype(np.float32, copy=True)
        values[global_masks[:, column_index] >= 0.5] = np.nan
        past_covariates[column_name] = values

    history_calendar = prepared.context_history_tensor[history, state_base._CONTEXT_CALENDAR_START :]
    future_calendar = prepared.context_future_tensor[future]
    future_covariates: dict[str, np.ndarray] = {}
    for column_index, column_name in enumerate(calendar_names):
        past_covariates[column_name] = history_calendar[:, column_index].astype(np.float32, copy=True)
        future_covariates[column_name] = future_calendar[:, column_index].astype(np.float32, copy=True)
    return {
        "target": target,
        "past_covariates": past_covariates,
        "future_covariates": future_covariates,
    }


def _chronos_point_to_numpy(point_forecast: object, *, forecast_steps: int) -> np.ndarray:
    if hasattr(point_forecast, "detach"):
        point_forecast = point_forecast.detach()
    if hasattr(point_forecast, "cpu"):
        point_forecast = point_forecast.cpu()
    values = np.asarray(point_forecast, dtype=np.float32)
    if values.ndim == 2:
        if values.shape[0] != 1:
            raise ValueError(f"Expected a single target row from Chronos-2, found {values.shape!r}.")
        values = values[0]
    if values.shape != (forecast_steps,):
        raise ValueError(f"Expected Chronos-2 forecast shape ({forecast_steps},), found {values.shape!r}.")
    return values


def _evaluate_chronos2(
    pipeline: Any,
    prepared: Any,
    windows: Any,
    *,
    batch_size: int,
) -> np.ndarray:
    predictions = np.zeros((len(windows), prepared.forecast_steps, prepared.node_count), dtype=np.float32)
    batch_inputs: list[dict[str, Any]] = []
    batch_window_ids: list[int] = []
    batch_node_ids: list[int] = []

    def flush() -> None:
        if not batch_inputs:
            return
        _quantiles, point_forecasts = pipeline.predict_quantiles(
            batch_inputs,
            prediction_length=prepared.forecast_steps,
            batch_size=batch_size,
            context_length=prepared.history_steps,
            cross_learning=False,
            limit_prediction_length=False,
        )
        if len(point_forecasts) != len(batch_inputs):
            raise RuntimeError(
                f"Chronos-2 returned {len(point_forecasts)} point forecasts for {len(batch_inputs)} inputs."
            )
        for row_index, point_forecast in enumerate(point_forecasts):
            predictions[batch_window_ids[row_index], :, batch_node_ids[row_index]] = _chronos_point_to_numpy(
                point_forecast,
                forecast_steps=prepared.forecast_steps,
            )
        batch_inputs.clear()
        batch_window_ids.clear()
        batch_node_ids.clear()

    for window_pos, target_index in enumerate(windows.target_indices):
        for node_index in range(prepared.node_count):
            batch_inputs.append(
                _chronos_input_for(
                    prepared,
                    target_index=int(target_index),
                    node_index=node_index,
                )
            )
            batch_window_ids.append(window_pos)
            batch_node_ids.append(node_index)
            if len(batch_inputs) >= batch_size:
                flush()
    flush()
    return predictions


def _load_itransformer_model(*, device: str, d_model: int, n_heads: int, e_layers: int, dropout: float):
    source_root = REPO_ROOT / "experiment" / "official_baselines" / "itransformer" / "source"
    if str(source_root) not in sys.path:
        sys.path.insert(0, str(source_root))
    from model.iTransformer import Model  # type: ignore

    configs = SimpleNamespace(
        seq_len=HISTORY_STEPS,
        pred_len=FORECAST_STEPS,
        output_attention=False,
        use_norm=True,
        d_model=d_model,
        embed="timeF",
        freq="h",
        dropout=dropout,
        class_strategy="projection",
        factor=1,
        n_heads=n_heads,
        d_ff=d_model * 4,
        e_layers=e_layers,
        activation="gelu",
    )
    return Model(configs).to(device)


def _load_dgcrn_model(*, prepared: Any, device: str, in_dim: int, hidden_dim: int, dropout: float, gcn_depth: int):
    import torch

    source_root = REPO_ROOT / "experiment" / "official_baselines" / "dgcrn" / "source" / "methods" / "DGCRN"
    if str(source_root) not in sys.path:
        sys.path.insert(0, str(source_root))
    from net import DGCRN  # type: ignore

    adjacency = torch.eye(prepared.node_count, dtype=torch.float32, device=device)
    model = DGCRN(
        gcn_depth=gcn_depth,
        num_nodes=prepared.node_count,
        device=device,
        predefined_A=[adjacency, adjacency.transpose(0, 1)],
        dropout=dropout,
        subgraph_size=min(6, prepared.node_count),
        node_dim=16,
        middle_dim=8,
        seq_length=HISTORY_STEPS,
        in_dim=in_dim,
        out_dim=FORECAST_STEPS,
        layers=1,
        rnn_size=hidden_dim,
        hyperGNN_dim=16,
    ).to(device)
    model.use_curriculum_learning = False
    return model


def _load_timexer_model(*, device: str, d_model: int, n_heads: int, e_layers: int, dropout: float, patch_len: int):
    source_root = REPO_ROOT / "experiment" / "official_baselines" / "timexer" / "source"
    if str(source_root) not in sys.path:
        sys.path.insert(0, str(source_root))
    from models.TimeXer import Model  # type: ignore

    configs = SimpleNamespace(
        task_name="long_term_forecast",
        features="M",
        seq_len=HISTORY_STEPS,
        pred_len=FORECAST_STEPS,
        use_norm=True,
        patch_len=patch_len,
        enc_in=6,
        d_model=d_model,
        embed="timeF",
        freq="h",
        dropout=dropout,
        factor=1,
        n_heads=n_heads,
        d_ff=d_model * 4,
        e_layers=e_layers,
        activation="gelu",
    )
    return Model(configs).to(device)


def _last_available_series(values: np.ndarray, unavailable: np.ndarray, fallback: np.ndarray) -> np.ndarray:
    last_values = fallback.astype(np.float32, copy=True)
    for node_index in range(values.shape[1]):
        valid_positions = np.flatnonzero(unavailable[:, node_index] < 0.5)
        if valid_positions.size:
            last_values[node_index] = values[int(valid_positions[-1]), node_index]
    return last_values


def _static_summary(prepared: Any) -> np.ndarray:
    static = np.asarray(prepared.static_tensor, dtype=np.float32)
    if static.ndim != 2 or static.shape[0] != prepared.node_count or static.shape[1] == 0:
        return np.zeros((prepared.node_count,), dtype=np.float32)
    column = static[:, 0].astype(np.float32, copy=True)
    finite = np.isfinite(column)
    if not finite.any():
        return np.zeros((prepared.node_count,), dtype=np.float32)
    mean = float(column[finite].mean())
    std = float(column[finite].std()) or 1.0
    column = np.where(finite, column, mean)
    return ((column - mean) / std).astype(np.float32)


def _iter_target_only_batches(
    prepared: Any,
    windows: Any,
    *,
    batch_size: int,
    shuffle: bool,
    seed: int,
):
    indices = np.arange(len(windows), dtype=np.int64)
    if shuffle:
        rng = np.random.default_rng(seed)
        rng.shuffle(indices)
    for start in range(0, len(indices), batch_size):
        batch_window_indices = indices[start : start + batch_size]
        x = np.zeros((len(batch_window_indices), prepared.history_steps, prepared.node_count), dtype=np.float32)
        y = np.zeros((len(batch_window_indices), prepared.forecast_steps, prepared.node_count), dtype=np.float32)
        valid = np.zeros_like(y, dtype=np.float32)
        anchor = np.zeros((len(batch_window_indices), prepared.node_count), dtype=np.float32)
        for row_index, window_pos in enumerate(batch_window_indices):
            target_index = int(windows.target_indices[int(window_pos)])
            history = slice(target_index - prepared.history_steps, target_index)
            future = slice(target_index, target_index + prepared.forecast_steps)
            x[row_index] = prepared.local_history_tensor[history, :, 0]
            target_unavailable = prepared.local_history_tensor[history, :, 17]
            fallback = prepared.persistence_train_fallback_pu.astype(np.float32, copy=True)
            last_values = fallback.copy()
            for node_index in range(prepared.node_count):
                valid_positions = np.flatnonzero(target_unavailable[:, node_index] < 0.5)
                if valid_positions.size:
                    last_values[node_index] = x[row_index, int(valid_positions[-1]), node_index]
            anchor[row_index] = last_values
            y[row_index] = prepared.target_pu_filled[future]
            valid[row_index] = prepared.target_valid_mask[future].astype(np.float32, copy=False)
        yield x, y, valid, anchor


def _iter_dgcrn_batches(
    prepared: Any,
    windows: Any,
    *,
    batch_size: int,
    shuffle: bool,
    seed: int,
):
    state_base = _load_state_space_base()
    calendar_count = prepared.context_future_tensor.shape[1]
    in_dim = 1 + calendar_count + 3
    indices = np.arange(len(windows), dtype=np.int64)
    if shuffle:
        rng = np.random.default_rng(seed)
        rng.shuffle(indices)
    static_summary = _static_summary(prepared)
    local_fallback = np.zeros((prepared.node_count,), dtype=np.float32)
    for start in range(0, len(indices), batch_size):
        batch_window_indices = indices[start : start + batch_size]
        x = np.zeros((len(batch_window_indices), in_dim, prepared.node_count, prepared.history_steps), dtype=np.float32)
        ycl = np.zeros((len(batch_window_indices), in_dim, prepared.node_count, prepared.forecast_steps), dtype=np.float32)
        y = np.zeros((len(batch_window_indices), prepared.forecast_steps, prepared.node_count), dtype=np.float32)
        valid = np.zeros_like(y, dtype=np.float32)
        anchor = np.zeros((len(batch_window_indices), prepared.node_count), dtype=np.float32)
        for row_index, window_pos in enumerate(batch_window_indices):
            target_index = int(windows.target_indices[int(window_pos)])
            history = slice(target_index - prepared.history_steps, target_index)
            future = slice(target_index, target_index + prepared.forecast_steps)
            target_history = prepared.local_history_tensor[history, :, 0]
            target_unavailable = prepared.local_history_tensor[history, :, 17]
            last_target = _last_available_series(
                target_history,
                target_unavailable,
                prepared.persistence_train_fallback_pu.astype(np.float32, copy=False),
            )
            anchor[row_index] = last_target
            x[row_index, 0] = target_history.T

            history_calendar = prepared.context_history_tensor[history, state_base._CONTEXT_CALENDAR_START :]
            future_calendar = prepared.context_future_tensor[future]
            for calendar_index in range(calendar_count):
                x[row_index, 1 + calendar_index] = history_calendar[:, calendar_index][None, :]
                ycl[row_index, 1 + calendar_index] = future_calendar[:, calendar_index][None, :]

            local_values = prepared.local_history_tensor[history, :, 1]
            local_unavailable = prepared.local_history_tensor[history, :, 18]
            local_last = _last_available_series(local_values, local_unavailable, local_fallback)
            x[row_index, 1 + calendar_count] = local_values.T
            ycl[row_index, 1 + calendar_count] = local_last[:, None]

            global_values = prepared.context_history_tensor[history, state_base._CONTEXT_GLOBAL_VALUE_START]
            global_unavailable = prepared.context_history_tensor[history, state_base._CONTEXT_GLOBAL_MASK_START]
            global_last_scalar = float(global_values[np.flatnonzero(global_unavailable < 0.5)[-1]]) if np.any(global_unavailable < 0.5) else 0.0
            x[row_index, 2 + calendar_count] = global_values[None, :]
            ycl[row_index, 2 + calendar_count] = np.full((prepared.node_count, prepared.forecast_steps), global_last_scalar, dtype=np.float32)

            x[row_index, 3 + calendar_count] = static_summary[:, None]
            ycl[row_index, 3 + calendar_count] = static_summary[:, None]

            y[row_index] = prepared.target_pu_filled[future]
            valid[row_index] = prepared.target_valid_mask[future].astype(np.float32, copy=False)
        yield x, ycl, y, valid, anchor


def _masked_mse_torch(predictions: Any, targets: Any, valid: Any) -> Any:
    denominator = valid.sum().clamp_min(1.0)
    return (((predictions - targets) ** 2) * valid).sum() / denominator


def _train_itransformer(
    prepared: Any,
    *,
    variant_name: str,
    validation_windows: Any,
    seed: int,
    device: str,
    batch_size: int,
    learning_rate: float,
    max_epochs: int,
    d_model: int,
    n_heads: int,
    e_layers: int,
    dropout: float,
) -> tuple[Any, dict[str, Any]]:
    import torch

    torch.manual_seed(seed)
    if device == "cuda" and torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    resolved_device = device if device != "cuda" or torch.cuda.is_available() else "cpu"
    model = _load_itransformer_model(
        device=resolved_device,
        d_model=d_model,
        n_heads=n_heads,
        e_layers=e_layers,
        dropout=dropout,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    best_state = None
    best_val_rmse = math.inf
    best_val_mae = math.inf
    best_epoch = -1
    history: list[dict[str, Any]] = []
    residual_output = variant_name == ITRANSFORMER_TARGET_RESIDUAL_VARIANT
    for epoch in range(max_epochs):
        model.train()
        train_loss_sum = 0.0
        train_batches = 0
        for x_np, y_np, valid_np, anchor_np in _iter_target_only_batches(
            prepared,
            prepared.train_windows,
            batch_size=batch_size,
            shuffle=True,
            seed=seed + epoch,
        ):
            x = torch.as_tensor(x_np, device=resolved_device)
            y = torch.as_tensor(y_np, device=resolved_device)
            valid = torch.as_tensor(valid_np, device=resolved_device)
            anchor = torch.as_tensor(anchor_np, device=resolved_device)
            target = y - anchor[:, None, :] if residual_output else y
            optimizer.zero_grad(set_to_none=True)
            raw = model(x, None, None, None)
            loss = _masked_mse_torch(raw, target, valid)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss_sum += float(loss.detach().cpu())
            train_batches += 1
        val_predictions = _evaluate_itransformer(
            model,
            prepared,
            validation_windows,
            variant_name=variant_name,
            device=resolved_device,
            batch_size=batch_size,
        )
        val_targets, val_valid = _target_and_valid(prepared, validation_windows)
        val_metrics = _metrics(val_predictions, val_targets, val_valid, rated_power_kw=prepared.rated_power_kw)
        history.append(
            {
                "epoch": epoch,
                "train_loss_mean": train_loss_sum / max(train_batches, 1),
                "val_rmse_pu": val_metrics["rmse_pu"],
                "val_mae_pu": val_metrics["mae_pu"],
            }
        )
        if val_metrics["rmse_pu"] < best_val_rmse:
            best_val_rmse = float(val_metrics["rmse_pu"])
            best_val_mae = float(val_metrics["mae_pu"])
            best_epoch = epoch
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, {
        "best_epoch": best_epoch,
        "epochs_ran": max_epochs,
        "best_val_rmse_pu": best_val_rmse,
        "best_val_mae_pu": best_val_mae,
        "history": history,
        "device": resolved_device,
    }


def _train_dgcrn(
    prepared: Any,
    *,
    variant_name: str,
    validation_windows: Any,
    seed: int,
    device: str,
    batch_size: int,
    learning_rate: float,
    max_epochs: int,
    hidden_dim: int,
    dropout: float,
    gcn_depth: int,
) -> tuple[Any, dict[str, Any]]:
    import torch

    torch.manual_seed(seed)
    if device == "cuda" and torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    resolved_device = device if device != "cuda" or torch.cuda.is_available() else "cpu"
    in_dim = 1 + prepared.context_future_tensor.shape[1] + 3
    model = _load_dgcrn_model(
        prepared=prepared,
        device=resolved_device,
        in_dim=in_dim,
        hidden_dim=hidden_dim,
        dropout=dropout,
        gcn_depth=gcn_depth,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    best_state = None
    best_val_rmse = math.inf
    best_val_mae = math.inf
    best_epoch = -1
    history: list[dict[str, Any]] = []
    residual_output = variant_name == DGCRN_RESIDUAL_VARIANT
    batches_seen = 0
    for epoch in range(max_epochs):
        model.train()
        train_loss_sum = 0.0
        train_batches = 0
        for x_np, ycl_np, y_np, valid_np, anchor_np in _iter_dgcrn_batches(
            prepared,
            prepared.train_windows,
            batch_size=batch_size,
            shuffle=True,
            seed=seed + epoch,
        ):
            x = torch.as_tensor(x_np, device=resolved_device)
            ycl = torch.as_tensor(ycl_np, device=resolved_device)
            y = torch.as_tensor(y_np, device=resolved_device)
            valid = torch.as_tensor(valid_np, device=resolved_device)
            anchor = torch.as_tensor(anchor_np, device=resolved_device)
            target = y - anchor[:, None, :] if residual_output else y
            optimizer.zero_grad(set_to_none=True)
            raw = model(x, ycl=ycl, batches_seen=batches_seen, task_level=prepared.forecast_steps).squeeze(-1)
            loss = _masked_mse_torch(raw, target, valid)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            batches_seen += 1
            train_loss_sum += float(loss.detach().cpu())
            train_batches += 1
        val_predictions = _evaluate_dgcrn(
            model,
            prepared,
            validation_windows,
            variant_name=variant_name,
            device=resolved_device,
            batch_size=batch_size,
        )
        val_targets, val_valid = _target_and_valid(prepared, validation_windows)
        val_metrics = _metrics(val_predictions, val_targets, val_valid, rated_power_kw=prepared.rated_power_kw)
        history.append(
            {
                "epoch": epoch,
                "train_loss_mean": train_loss_sum / max(train_batches, 1),
                "val_rmse_pu": val_metrics["rmse_pu"],
                "val_mae_pu": val_metrics["mae_pu"],
            }
        )
        if val_metrics["rmse_pu"] < best_val_rmse:
            best_val_rmse = float(val_metrics["rmse_pu"])
            best_val_mae = float(val_metrics["mae_pu"])
            best_epoch = epoch
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, {
        "best_epoch": best_epoch,
        "epochs_ran": max_epochs,
        "best_val_rmse_pu": best_val_rmse,
        "best_val_mae_pu": best_val_mae,
        "history": history,
        "device": resolved_device,
    }


def _train_timexer(
    prepared: Any,
    *,
    variant_name: str,
    validation_windows: Any,
    seed: int,
    device: str,
    batch_size: int,
    learning_rate: float,
    max_epochs: int,
    d_model: int,
    n_heads: int,
    e_layers: int,
    dropout: float,
    patch_len: int,
) -> tuple[Any, dict[str, Any]]:
    import torch

    torch.manual_seed(seed)
    if device == "cuda" and torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    resolved_device = device if device != "cuda" or torch.cuda.is_available() else "cpu"
    model = _load_timexer_model(
        device=resolved_device,
        d_model=d_model,
        n_heads=n_heads,
        e_layers=e_layers,
        dropout=dropout,
        patch_len=patch_len,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    best_state = None
    best_val_rmse = math.inf
    best_val_mae = math.inf
    best_epoch = -1
    history: list[dict[str, Any]] = []
    residual_output = variant_name == TIMEXER_TARGET_RESIDUAL_VARIANT
    for epoch in range(max_epochs):
        model.train()
        train_loss_sum = 0.0
        train_batches = 0
        for x_np, y_np, valid_np, anchor_np in _iter_target_only_batches(
            prepared,
            prepared.train_windows,
            batch_size=batch_size,
            shuffle=True,
            seed=seed + epoch,
        ):
            x = torch.as_tensor(x_np, device=resolved_device)
            y = torch.as_tensor(y_np, device=resolved_device)
            valid = torch.as_tensor(valid_np, device=resolved_device)
            anchor = torch.as_tensor(anchor_np, device=resolved_device)
            target = y - anchor[:, None, :] if residual_output else y
            optimizer.zero_grad(set_to_none=True)
            raw = model(x, None, None, None)
            loss = _masked_mse_torch(raw, target, valid)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss_sum += float(loss.detach().cpu())
            train_batches += 1
        val_predictions = _evaluate_timexer(
            model,
            prepared,
            validation_windows,
            variant_name=variant_name,
            device=resolved_device,
            batch_size=batch_size,
        )
        val_targets, val_valid = _target_and_valid(prepared, validation_windows)
        val_metrics = _metrics(val_predictions, val_targets, val_valid, rated_power_kw=prepared.rated_power_kw)
        history.append(
            {
                "epoch": epoch,
                "train_loss_mean": train_loss_sum / max(train_batches, 1),
                "val_rmse_pu": val_metrics["rmse_pu"],
                "val_mae_pu": val_metrics["mae_pu"],
            }
        )
        if val_metrics["rmse_pu"] < best_val_rmse:
            best_val_rmse = float(val_metrics["rmse_pu"])
            best_val_mae = float(val_metrics["mae_pu"])
            best_epoch = epoch
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, {
        "best_epoch": best_epoch,
        "epochs_ran": max_epochs,
        "best_val_rmse_pu": best_val_rmse,
        "best_val_mae_pu": best_val_mae,
        "history": history,
        "device": resolved_device,
    }


def _evaluate_itransformer(
    model: Any,
    prepared: Any,
    windows: Any,
    *,
    variant_name: str,
    device: str,
    batch_size: int,
) -> np.ndarray:
    import torch

    residual_output = variant_name == ITRANSFORMER_TARGET_RESIDUAL_VARIANT
    predictions = np.zeros((len(windows), prepared.forecast_steps, prepared.node_count), dtype=np.float32)
    model.eval()
    offset = 0
    with torch.no_grad():
        for x_np, _y_np, _valid_np, anchor_np in _iter_target_only_batches(
            prepared,
            windows,
            batch_size=batch_size,
            shuffle=False,
            seed=0,
        ):
            x = torch.as_tensor(x_np, device=device)
            raw = model(x, None, None, None).detach().cpu().numpy().astype(np.float32, copy=False)
            if residual_output:
                raw = raw + anchor_np[:, None, :]
            predictions[offset : offset + raw.shape[0]] = raw
            offset += raw.shape[0]
    return predictions


def _evaluate_dgcrn(
    model: Any,
    prepared: Any,
    windows: Any,
    *,
    variant_name: str,
    device: str,
    batch_size: int,
) -> np.ndarray:
    import torch

    residual_output = variant_name == DGCRN_RESIDUAL_VARIANT
    predictions = np.zeros((len(windows), prepared.forecast_steps, prepared.node_count), dtype=np.float32)
    model.eval()
    offset = 0
    with torch.no_grad():
        for x_np, ycl_np, _y_np, _valid_np, anchor_np in _iter_dgcrn_batches(
            prepared,
            windows,
            batch_size=batch_size,
            shuffle=False,
            seed=0,
        ):
            x = torch.as_tensor(x_np, device=device)
            ycl = torch.as_tensor(ycl_np, device=device)
            raw = model(x, ycl=ycl, batches_seen=None, task_level=prepared.forecast_steps).squeeze(-1)
            raw_np = raw.detach().cpu().numpy().astype(np.float32, copy=False)
            if residual_output:
                raw_np = raw_np + anchor_np[:, None, :]
            predictions[offset : offset + raw_np.shape[0]] = raw_np
            offset += raw_np.shape[0]
    return predictions


def _evaluate_timexer(
    model: Any,
    prepared: Any,
    windows: Any,
    *,
    variant_name: str,
    device: str,
    batch_size: int,
) -> np.ndarray:
    import torch

    residual_output = variant_name == TIMEXER_TARGET_RESIDUAL_VARIANT
    predictions = np.zeros((len(windows), prepared.forecast_steps, prepared.node_count), dtype=np.float32)
    model.eval()
    offset = 0
    with torch.no_grad():
        for x_np, _y_np, _valid_np, anchor_np in _iter_target_only_batches(
            prepared,
            windows,
            batch_size=batch_size,
            shuffle=False,
            seed=0,
        ):
            x = torch.as_tensor(x_np, device=device)
            raw = model(x, None, None, None).detach().cpu().numpy().astype(np.float32, copy=False)
            if residual_output:
                raw = raw + anchor_np[:, None, :]
            predictions[offset : offset + raw.shape[0]] = raw
            offset += raw.shape[0]
    return predictions


def _gate_status_for_neural_model(
    *,
    evaluator: Any,
    model: Any,
    prepared: Any,
    variant_name: str,
    device: str,
    batch_size: int,
    train_gate_windows: Any,
    gate_c_windows: Any,
    persistence_train_rmse: float,
    persistence_gate_c_lead1_rmse: float,
    persistence_gate_c_lead1_mae: float,
) -> tuple[bool, bool, dict[str, Any], dict[str, Any]]:
    train_predictions = evaluator(
        model,
        prepared,
        train_gate_windows,
        variant_name=variant_name,
        device=device,
        batch_size=batch_size,
    )
    train_targets, train_valid = _target_and_valid(prepared, train_gate_windows)
    train_metrics = _metrics(train_predictions, train_targets, train_valid, rated_power_kw=prepared.rated_power_kw)
    gate_b_passed = bool(
        train_metrics["rmse_pu"] <= 0.03
        or train_metrics["rmse_pu"] <= 0.5 * persistence_train_rmse
    )

    gate_c_predictions = evaluator(
        model,
        prepared,
        gate_c_windows,
        variant_name=variant_name,
        device=device,
        batch_size=batch_size,
    )
    gate_c_targets, gate_c_valid = _target_and_valid(prepared, gate_c_windows)
    gate_c_metrics = _metrics(gate_c_predictions, gate_c_targets, gate_c_valid, rated_power_kw=prepared.rated_power_kw)
    gate_c_passed = bool(
        gate_c_metrics["lead1_rmse_pu"] <= 1.05 * persistence_gate_c_lead1_rmse
        and gate_c_metrics["lead1_mae_pu"] <= 1.05 * persistence_gate_c_lead1_mae
    )
    return gate_b_passed, gate_c_passed, train_metrics, gate_c_metrics


def _base_row(spec: OfficialVariantSpec, *, dataset_id: str, seed: int) -> dict[str, Any]:
    budget = spec.feature_budget
    return {
        "dataset_id": dataset_id,
        "model_id": "WORLD_MODEL_OFFICIAL_BASELINE",
        "model_variant": spec.model_variant,
        "task_id": TASK_ID,
        "history_steps": HISTORY_STEPS,
        "forecast_steps": FORECAST_STEPS,
        "source_repo": spec.source_repo,
        "source_commit": spec.source_commit,
        "source_file": spec.source_file,
        "model_class": spec.model_class,
        "adapter_class": spec.adapter_class,
        "train_script": spec.train_script,
        "search_config_id": spec.search_config_id,
        "seed": seed,
        "selection_metric": spec.selection_metric,
        "feature_budget_id": spec.feature_budget_id,
        "output_parameterization": spec.output_parameterization,
        "uses_target_history": budget.uses_target_history,
        "uses_local_history": budget.uses_local_history,
        "uses_global_history": budget.uses_global_history,
        "uses_future_calendar": budget.uses_future_calendar,
        "uses_static": budget.uses_static,
        "uses_pairwise": budget.uses_pairwise,
        "uses_future_target": False,
        "selected_by": "validation_only",
        "no_test_feedback": True,
        "test_evaluated_at": None,
    }


def _metric_rows(
    spec: OfficialVariantSpec,
    *,
    prepared: Any,
    seed: int,
    trial_id: str,
    search_config_id: str,
    alpha: float | None,
    predictions_by_split: dict[tuple[str, str], np.ndarray],
    runtime_seconds: float,
    gate_b_passed: bool | None,
    gate_c_passed: bool | None,
    best_trial: bool,
    window_specs: Sequence[tuple[str, str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for split_name, eval_protocol, windows in window_specs:
        predictions = predictions_by_split[(split_name, eval_protocol)]
        targets, valid = _target_and_valid(prepared, windows)
        metrics = _metrics(predictions, targets, valid, rated_power_kw=prepared.rated_power_kw)
        rows.append(
            {
                **_base_row(spec, dataset_id=prepared.dataset_id, seed=seed),
                "split_name": split_name,
                "eval_protocol": eval_protocol,
                "metric_scope": "overall",
                "lead_step": None,
                "trial_id": trial_id,
                "trial_status": "completed",
                "trial_blocker": None,
                "alpha": alpha,
                "formal_search_config_id": search_config_id,
                "is_best_validation_trial": best_trial,
                "gate_a_passed": True,
                "gate_b_passed": gate_b_passed,
                "gate_c_passed": gate_c_passed,
                "runtime_seconds": runtime_seconds,
                **metrics,
            }
        )
    return rows


def _blocked_row(spec: OfficialVariantSpec, *, dataset_id: str, seed: int, blocker: str) -> dict[str, Any]:
    return {
        **_base_row(spec, dataset_id=dataset_id, seed=seed),
        "split_name": "formal_tuning",
        "eval_protocol": "not_started",
        "metric_scope": "blocked",
        "lead_step": None,
        "trial_id": "blocked",
        "trial_status": "blocked",
        "trial_blocker": blocker,
        "alpha": None,
        "formal_search_config_id": spec.search_config_id,
        "is_best_validation_trial": False,
        "gate_a_passed": True,
        "gate_b_passed": False if spec.trainable else None,
        "gate_c_passed": False if spec.trainable else None,
        "runtime_seconds": 0.0,
        "window_count": None,
        "prediction_count": None,
        "mae_pu": None,
        "rmse_pu": None,
        "mae_kw": None,
        "rmse_kw": None,
        "lead1_mae_pu": None,
        "lead1_rmse_pu": None,
        "short_rmse_pu": None,
        "mid_rmse_pu": None,
        "long_rmse_pu": None,
        "ae_p50": None,
        "ae_p90": None,
        "ae_p95": None,
    }


def run_formal_tuning(
    *,
    dataset_ids: Sequence[str] = DEFAULT_DATASETS,
    variant_names: Sequence[str] | None = None,
    output_path: str | Path,
    seed: int = 3407,
    ridge_alphas: Sequence[float] = DEFAULT_RIDGE_ALPHAS,
    max_train_origins: int | None = None,
    max_eval_origins: int | None = None,
    chronos_batch_size: int = 32,
    device: str = "cuda",
    train_batch_size: int = 128,
    max_epochs: int = 3,
    learning_rate: float = 1e-4,
    split_names: Sequence[str] | None = None,
    eval_protocols: Sequence[str] | None = None,
    checkpoint_eval_protocol: str = ROLLING_EVAL_PROTOCOL,
    max_checkpoint_origins: int | None = None,
    gate_origin_count: int = 64,
    run_label: str | None = None,
    no_record_run: bool = False,
) -> pl.DataFrame:
    specs = resolve_variant_specs(variant_names or DEFAULT_VARIANTS)
    rows: list[dict[str, Any]] = []
    for dataset_id in dataset_ids:
        prepared = _prepare_dataset(dataset_id, max_train_origins=max_train_origins, max_eval_origins=max_eval_origins)
        window_specs = _selected_window_specs(prepared, split_names=split_names, eval_protocols=eval_protocols)
        if not window_specs:
            raise ValueError("No evaluation windows selected by split_names/eval_protocols.")
        checkpoint_windows = _validation_windows_for_checkpoint(
            prepared,
            eval_protocol=checkpoint_eval_protocol,
            max_origins=max_checkpoint_origins,
        )
        train_gate_windows = _limit_windows(prepared.train_windows, gate_origin_count)
        gate_c_windows = _validation_windows_for_checkpoint(
            prepared,
            eval_protocol=checkpoint_eval_protocol,
            max_origins=gate_origin_count,
        )
        persistence_train_gate_predictions = _repeat_anchor(
            _last_value_anchor(prepared, train_gate_windows),
            prepared.forecast_steps,
        )
        train_gate_targets, train_gate_valid = _target_and_valid(prepared, train_gate_windows)
        persistence_train_gate_metrics = _metrics(
            persistence_train_gate_predictions,
            train_gate_targets,
            train_gate_valid,
            rated_power_kw=prepared.rated_power_kw,
        )
        persistence_gate_c_predictions = _repeat_anchor(
            _last_value_anchor(prepared, gate_c_windows),
            prepared.forecast_steps,
        )
        gate_c_targets, gate_c_valid = _target_and_valid(prepared, gate_c_windows)
        persistence_gate_c_metrics = _metrics(
            persistence_gate_c_predictions,
            gate_c_targets,
            gate_c_valid,
            rated_power_kw=prepared.rated_power_kw,
        )
        persistence_val_lead1_rmse: float | None = persistence_gate_c_metrics["lead1_rmse_pu"]
        persistence_val_lead1_mae: float | None = persistence_gate_c_metrics["lead1_mae_pu"]
        for spec in specs:
            status, blocker = formal_support_status(spec)
            if status != "supported":
                rows.append(_blocked_row(spec, dataset_id=dataset_id, seed=seed, blocker=blocker or "blocked"))
                continue
            started = time.perf_counter()
            if spec.model_variant == PERSISTENCE_VARIANT:
                predictions_by_split = {
                    (split_name, eval_protocol): _repeat_anchor(_last_value_anchor(prepared, windows), prepared.forecast_steps)
                    for split_name, eval_protocol, windows in window_specs
                }
                val_predictions = _repeat_anchor(_last_value_anchor(prepared, gate_c_windows), prepared.forecast_steps)
                val_targets, val_valid = _target_and_valid(prepared, gate_c_windows)
                val_metrics = _metrics(
                    val_predictions,
                    val_targets,
                    val_valid,
                    rated_power_kw=prepared.rated_power_kw,
                )
                persistence_val_lead1_rmse = val_metrics["lead1_rmse_pu"]
                persistence_val_lead1_mae = val_metrics["lead1_mae_pu"]
                rows.extend(
                    _metric_rows(
                        spec,
                        prepared=prepared,
                        seed=seed,
                        trial_id="analytic_last_value",
                        search_config_id="analytic_no_tuning",
                        alpha=None,
                        predictions_by_split=predictions_by_split,
                        runtime_seconds=time.perf_counter() - started,
                        gate_b_passed=None,
                        gate_c_passed=True,
                        best_trial=True,
                        window_specs=window_specs,
                    )
                )
            elif spec.model_variant == SEASONAL_PERSISTENCE_VARIANT:
                predictions_by_split = {
                    (split_name, eval_protocol): _repeat_anchor(_seasonal_anchor(prepared, windows), prepared.forecast_steps)
                    for split_name, eval_protocol, windows in window_specs
                }
                rows.extend(
                    _metric_rows(
                        spec,
                        prepared=prepared,
                        seed=seed,
                        trial_id="analytic_seasonal",
                        search_config_id="analytic_no_tuning",
                        alpha=None,
                        predictions_by_split=predictions_by_split,
                        runtime_seconds=time.perf_counter() - started,
                        gate_b_passed=None,
                        gate_c_passed=None,
                        best_trial=True,
                        window_specs=window_specs,
                    )
                )
            elif spec.model_variant == RIDGE_RESIDUAL_VARIANT:
                train_anchor = _last_value_anchor(prepared, prepared.train_windows)
                train_features = _ridge_features(prepared, prepared.train_windows)
                train_residuals = _flatten_future_residuals(prepared, prepared.train_windows, train_anchor)
                trial_summaries: list[tuple[float, float, np.ndarray, dict[tuple[str, str], np.ndarray], bool, bool]] = []
                for alpha in ridge_alphas:
                    weights = _fit_ridge(train_features, train_residuals, alpha=float(alpha))
                    val_predictions = _predict_ridge(prepared, prepared.val_rolling_windows, weights)
                    val_targets, val_valid = _target_and_valid(prepared, prepared.val_rolling_windows)
                    val_metrics = _metrics(val_predictions, val_targets, val_valid, rated_power_kw=prepared.rated_power_kw)
                    gate_b_predictions = _predict_ridge(prepared, train_gate_windows, weights)
                    gate_b_metrics = _metrics(
                        gate_b_predictions,
                        train_gate_targets,
                        train_gate_valid,
                        rated_power_kw=prepared.rated_power_kw,
                    )
                    gate_b_passed = bool(
                        gate_b_metrics["rmse_pu"] <= 0.03
                        or gate_b_metrics["rmse_pu"] <= 0.5 * persistence_train_gate_metrics["rmse_pu"]
                    )
                    gate_c_predictions = _predict_ridge(prepared, gate_c_windows, weights)
                    gate_c_metrics = _metrics(
                        gate_c_predictions,
                        gate_c_targets,
                        gate_c_valid,
                        rated_power_kw=prepared.rated_power_kw,
                    )
                    gate_c_passed = bool(
                        persistence_val_lead1_rmse is not None
                        and persistence_val_lead1_mae is not None
                        and gate_c_metrics["lead1_rmse_pu"] <= 1.05 * persistence_val_lead1_rmse
                        and gate_c_metrics["lead1_mae_pu"] <= 1.05 * persistence_val_lead1_mae
                    )
                    predictions_by_split = {
                        (split_name, eval_protocol): _predict_ridge(prepared, windows, weights)
                        for split_name, eval_protocol, windows in window_specs
                    }
                    trial_summaries.append(
                        (
                            float(alpha),
                            float(val_metrics["rmse_pu"]),
                            weights,
                            predictions_by_split,
                            gate_b_passed,
                            gate_c_passed,
                        )
                    )
                best_index = min(range(len(trial_summaries)), key=lambda index: trial_summaries[index][1])
                for index, (alpha, _val_rmse, _weights, predictions_by_split, gate_b_passed, gate_c_passed) in enumerate(trial_summaries):
                    rows.extend(
                        _metric_rows(
                            spec,
                            prepared=prepared,
                            seed=seed,
                            trial_id=f"ridge_alpha_{alpha:g}",
                            search_config_id=f"ridge_b0_alpha_{alpha:g}",
                            alpha=alpha,
                            predictions_by_split=predictions_by_split,
                            runtime_seconds=time.perf_counter() - started,
                            gate_b_passed=gate_b_passed,
                            gate_c_passed=gate_c_passed,
                            best_trial=index == best_index,
                            window_specs=window_specs,
                        )
                    )
            elif spec.model_variant == CHRONOS2_VARIANT:
                pipeline = _load_chronos2_pipeline(device=device)
                predictions_by_split = {
                    (split_name, eval_protocol): _evaluate_chronos2(
                        pipeline,
                        prepared,
                        windows,
                        batch_size=chronos_batch_size,
                    )
                    for split_name, eval_protocol, windows in window_specs
                }
                rows.extend(
                    _metric_rows(
                        spec,
                        prepared=prepared,
                        seed=seed,
                        trial_id="chronos2_zero_shot_median",
                        search_config_id="chronos2_zero_shot_b2",
                        alpha=None,
                        predictions_by_split=predictions_by_split,
                        runtime_seconds=time.perf_counter() - started,
                        gate_b_passed=None,
                        gate_c_passed=None,
                        best_trial=True,
                        window_specs=window_specs,
                    )
                )
            elif spec.model_variant in {DGCRN_DIRECT_VARIANT, DGCRN_RESIDUAL_VARIANT}:
                model, train_summary = _train_dgcrn(
                    prepared,
                    variant_name=spec.model_variant,
                    validation_windows=checkpoint_windows,
                    seed=seed,
                    device=device,
                    batch_size=train_batch_size,
                    learning_rate=learning_rate,
                    max_epochs=max_epochs,
                    hidden_dim=64,
                    dropout=0.1,
                    gcn_depth=2,
                )
                predictions_by_split = {
                    (split_name, eval_protocol): _evaluate_dgcrn(
                        model,
                        prepared,
                        windows,
                        variant_name=spec.model_variant,
                        device=train_summary["device"],
                        batch_size=train_batch_size,
                    )
                    for split_name, eval_protocol, windows in window_specs
                }
                gate_b_passed, gate_c_passed, _gate_b_metrics, _gate_c_metrics = _gate_status_for_neural_model(
                    evaluator=_evaluate_dgcrn,
                    model=model,
                    prepared=prepared,
                    variant_name=spec.model_variant,
                    device=train_summary["device"],
                    batch_size=train_batch_size,
                    train_gate_windows=train_gate_windows,
                    gate_c_windows=gate_c_windows,
                    persistence_train_rmse=float(persistence_train_gate_metrics["rmse_pu"]),
                    persistence_gate_c_lead1_rmse=float(persistence_gate_c_metrics["lead1_rmse_pu"]),
                    persistence_gate_c_lead1_mae=float(persistence_gate_c_metrics["lead1_mae_pu"]),
                )
                rows.extend(
                    _metric_rows(
                        spec,
                        prepared=prepared,
                        seed=seed,
                        trial_id=(
                            "dgcrn_official_core_residual_default"
                            if spec.model_variant == DGCRN_RESIDUAL_VARIANT
                            else "dgcrn_official_core_direct_default"
                        ),
                        search_config_id="dgcrn_official_core_default",
                        alpha=None,
                        predictions_by_split=predictions_by_split,
                        runtime_seconds=time.perf_counter() - started,
                        gate_b_passed=gate_b_passed,
                        gate_c_passed=gate_c_passed,
                        best_trial=True,
                        window_specs=window_specs,
                    )
                )
            elif spec.model_variant in {TIMEXER_TARGET_DIRECT_VARIANT, TIMEXER_TARGET_RESIDUAL_VARIANT}:
                model, train_summary = _train_timexer(
                    prepared,
                    variant_name=spec.model_variant,
                    validation_windows=checkpoint_windows,
                    seed=seed,
                    device=device,
                    batch_size=train_batch_size,
                    learning_rate=learning_rate,
                    max_epochs=max_epochs,
                    d_model=64,
                    n_heads=4,
                    e_layers=2,
                    dropout=0.1,
                    patch_len=16,
                )
                predictions_by_split = {
                    (split_name, eval_protocol): _evaluate_timexer(
                        model,
                        prepared,
                        windows,
                        variant_name=spec.model_variant,
                        device=train_summary["device"],
                        batch_size=train_batch_size,
                    )
                    for split_name, eval_protocol, windows in window_specs
                }
                gate_b_passed, gate_c_passed, _gate_b_metrics, _gate_c_metrics = _gate_status_for_neural_model(
                    evaluator=_evaluate_timexer,
                    model=model,
                    prepared=prepared,
                    variant_name=spec.model_variant,
                    device=train_summary["device"],
                    batch_size=train_batch_size,
                    train_gate_windows=train_gate_windows,
                    gate_c_windows=gate_c_windows,
                    persistence_train_rmse=float(persistence_train_gate_metrics["rmse_pu"]),
                    persistence_gate_c_lead1_rmse=float(persistence_gate_c_metrics["lead1_rmse_pu"]),
                    persistence_gate_c_lead1_mae=float(persistence_gate_c_metrics["lead1_mae_pu"]),
                )
                rows.extend(
                    _metric_rows(
                        spec,
                        prepared=prepared,
                        seed=seed,
                        trial_id=(
                            "timexer_target_only_residual_default"
                            if spec.model_variant == TIMEXER_TARGET_RESIDUAL_VARIANT
                            else "timexer_target_only_direct_default"
                        ),
                        search_config_id="timexer_target_only_default",
                        alpha=None,
                        predictions_by_split=predictions_by_split,
                        runtime_seconds=time.perf_counter() - started,
                        gate_b_passed=gate_b_passed,
                        gate_c_passed=gate_c_passed,
                        best_trial=True,
                        window_specs=window_specs,
                    )
                )
            elif spec.model_variant in {ITRANSFORMER_TARGET_DIRECT_VARIANT, ITRANSFORMER_TARGET_RESIDUAL_VARIANT}:
                model, train_summary = _train_itransformer(
                    prepared,
                    variant_name=spec.model_variant,
                    validation_windows=checkpoint_windows,
                    seed=seed,
                    device=device,
                    batch_size=train_batch_size,
                    learning_rate=learning_rate,
                    max_epochs=max_epochs,
                    d_model=64,
                    n_heads=4,
                    e_layers=2,
                    dropout=0.1,
                )
                predictions_by_split = {
                    (split_name, eval_protocol): _evaluate_itransformer(
                        model,
                        prepared,
                        windows,
                        variant_name=spec.model_variant,
                        device=train_summary["device"],
                        batch_size=train_batch_size,
                    )
                    for split_name, eval_protocol, windows in window_specs
                }
                gate_b_passed, gate_c_passed, _gate_b_metrics, _gate_c_metrics = _gate_status_for_neural_model(
                    evaluator=_evaluate_itransformer,
                    model=model,
                    prepared=prepared,
                    variant_name=spec.model_variant,
                    device=train_summary["device"],
                    batch_size=train_batch_size,
                    train_gate_windows=train_gate_windows,
                    gate_c_windows=gate_c_windows,
                    persistence_train_rmse=float(persistence_train_gate_metrics["rmse_pu"]),
                    persistence_gate_c_lead1_rmse=float(persistence_gate_c_metrics["lead1_rmse_pu"]),
                    persistence_gate_c_lead1_mae=float(persistence_gate_c_metrics["lead1_mae_pu"]),
                )
                rows.extend(
                    _metric_rows(
                        spec,
                        prepared=prepared,
                        seed=seed,
                        trial_id=(
                            "itransformer_target_only_residual_default"
                            if spec.model_variant == ITRANSFORMER_TARGET_RESIDUAL_VARIANT
                            else "itransformer_target_only_direct_default"
                        ),
                        search_config_id="itransformer_target_only_default",
                        alpha=None,
                        predictions_by_split=predictions_by_split,
                        runtime_seconds=time.perf_counter() - started,
                        gate_b_passed=gate_b_passed,
                        gate_c_passed=gate_c_passed,
                        best_trial=True,
                        window_specs=window_specs,
                    )
                )
    frame = pl.DataFrame(rows)
    resolved_output_path = Path(output_path)
    resolved_output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.write_csv(resolved_output_path)
    summary_path = resolved_output_path.with_suffix(".summary.json")
    summary = {
        "created_at_utc": datetime.now(tz=UTC).isoformat(),
        "family_id": FAMILY_ID,
        "row_count": frame.height,
        "completed_rows": int((frame["trial_status"] == "completed").sum()) if frame.height else 0,
        "blocked_rows": int((frame["trial_status"] == "blocked").sum()) if frame.height else 0,
        "supported_variants": sorted(FORMAL_SUPPORTED_VARIANTS),
        "ridge_alphas": [float(value) for value in ridge_alphas],
        "chronos_batch_size": chronos_batch_size,
        "device": device,
        "train_batch_size": train_batch_size,
        "max_epochs": max_epochs,
        "learning_rate": learning_rate,
        "split_names": list(split_names) if split_names else None,
        "eval_protocols": list(eval_protocols) if eval_protocols else None,
        "checkpoint_eval_protocol": checkpoint_eval_protocol,
        "max_checkpoint_origins": max_checkpoint_origins,
        "gate_origin_count": gate_origin_count,
        "max_train_origins": max_train_origins,
        "max_eval_origins": max_eval_origins,
    }
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if not no_record_run:
        manifest_path = record_cli_run(
            family_id=FAMILY_ID,
            repo_root=REPO_ROOT,
            invocation_kind="formal_tuning_runner",
            entrypoint=f"experiment/families/{FAMILY_ID}/run_world_model_official_baselines_v2_formal_tuning.py",
            args={
                "dataset_ids": list(dataset_ids),
                "variant_names": [spec.model_variant for spec in specs],
                "seed": seed,
                "ridge_alphas": [float(value) for value in ridge_alphas],
                "chronos_batch_size": chronos_batch_size,
                "device": device,
                "train_batch_size": train_batch_size,
                "max_epochs": max_epochs,
                "learning_rate": learning_rate,
                "split_names": list(split_names) if split_names else None,
                "eval_protocols": list(eval_protocols) if eval_protocols else None,
                "checkpoint_eval_protocol": checkpoint_eval_protocol,
                "max_checkpoint_origins": max_checkpoint_origins,
                "gate_origin_count": gate_origin_count,
                "max_train_origins": max_train_origins,
                "max_eval_origins": max_eval_origins,
            },
            output_path=resolved_output_path,
            result_row_count=frame.height,
            dataset_ids=tuple(dataset_ids),
            feature_protocol_ids=(FEATURE_PROTOCOL_ID,),
            model_variants=tuple(spec.model_variant for spec in specs),
            eval_protocols=tuple(eval_protocols or (ROLLING_EVAL_PROTOCOL, NON_OVERLAP_EVAL_PROTOCOL)),
            result_splits=tuple(split_names or ("val", "test", "formal_tuning_blocked")),
            artifacts={"summary": summary_path},
            notes=(
                "This formal tuning runner is fail-closed: executable analytic/Ridge controls run, "
                "missing official trainable adapters are recorded as blocked rows, not performance results.",
            ),
            run_label=run_label,
        )
        _enrich_formal_manifest(manifest_path, summary=summary)
    return frame


def _enrich_formal_manifest(manifest_path: str | Path, *, summary: dict[str, Any]) -> None:
    path = Path(manifest_path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["selected_by"] = "validation_only"
    payload["no_test_feedback"] = True
    payload["test_evaluated_at"] = datetime.now(tz=UTC).isoformat()
    payload["formal_tuning_status"] = {
        **summary,
        "blocked_rows_are_not_performance_results": True,
    }
    payload.setdefault("quality_gates", {})
    payload["quality_gates"].update(
        {
            "gate_a": {"status": "previous_debug_snapshots_written"},
            "gate_b": {"status": "computed_for_ridge_and_supported_neural_trainable_paths"},
            "gate_c": {"status": "computed_for_ridge_and_supported_neural_trainable_paths"},
            "gate_d": {"status": "validation_only_selection"},
            "gate_e": {"status": "test_once_for_selected_completed_trials"},
        }
    )
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run fail-closed official baseline v2 formal tuning start.")
    parser.add_argument("--dataset", action="append", choices=list(DEFAULT_DATASETS), dest="datasets")
    parser.add_argument("--variant", action="append", choices=list(DEFAULT_VARIANTS), dest="variants")
    parser.add_argument("--output-path", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--ridge-alpha", action="append", type=float, dest="ridge_alphas")
    parser.add_argument("--max-train-origins", type=int, default=None)
    parser.add_argument("--max-eval-origins", type=int, default=None)
    parser.add_argument("--chronos-batch-size", type=int, default=32)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--train-batch-size", type=int, default=128)
    parser.add_argument("--max-epochs", type=int, default=3)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--split-name", action="append", choices=["val", "test"], dest="split_names")
    parser.add_argument(
        "--eval-protocol",
        action="append",
        choices=[ROLLING_EVAL_PROTOCOL, NON_OVERLAP_EVAL_PROTOCOL],
        dest="eval_protocols",
    )
    parser.add_argument(
        "--checkpoint-eval-protocol",
        choices=[ROLLING_EVAL_PROTOCOL, NON_OVERLAP_EVAL_PROTOCOL],
        default=ROLLING_EVAL_PROTOCOL,
    )
    parser.add_argument("--max-checkpoint-origins", type=int, default=None)
    parser.add_argument("--gate-origin-count", type=int, default=64)
    parser.add_argument("--run-label", type=str, default=None)
    parser.add_argument("--no-record-run", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    run_formal_tuning(
        dataset_ids=tuple(args.datasets) if args.datasets else DEFAULT_DATASETS,
        variant_names=tuple(args.variants) if args.variants else None,
        output_path=args.output_path,
        seed=args.seed,
        ridge_alphas=tuple(args.ridge_alphas) if args.ridge_alphas else DEFAULT_RIDGE_ALPHAS,
        max_train_origins=args.max_train_origins,
        max_eval_origins=args.max_eval_origins,
        chronos_batch_size=args.chronos_batch_size,
        device=args.device,
        train_batch_size=args.train_batch_size,
        max_epochs=args.max_epochs,
        learning_rate=args.learning_rate,
        split_names=tuple(args.split_names) if args.split_names else None,
        eval_protocols=tuple(args.eval_protocols) if args.eval_protocols else None,
        checkpoint_eval_protocol=args.checkpoint_eval_protocol,
        max_checkpoint_origins=args.max_checkpoint_origins,
        gate_origin_count=args.gate_origin_count,
        run_label=args.run_label,
        no_record_run=args.no_record_run,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
