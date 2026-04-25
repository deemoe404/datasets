from __future__ import annotations

import argparse
import importlib.util
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
    ITRANSFORMER_EXOG_RESIDUAL_VARIANT,
    ITRANSFORMER_TARGET_DIRECT_VARIANT,
    ITRANSFORMER_TARGET_RESIDUAL_VARIANT,
    MTGNN_CALENDAR_RESIDUAL_VARIANT,
    MTGNN_TARGET_VARIANT,
    PERSISTENCE_VARIANT,
    RIDGE_RESIDUAL_VARIANT,
    SEASONAL_PERSISTENCE_VARIANT,
    TFT_DIRECT_VARIANT,
    TFT_RESIDUAL_VARIANT,
    TIMEXER_FULL_RESIDUAL_VARIANT,
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
    ITRANSFORMER_EXOG_RESIDUAL_VARIANT,
    ITRANSFORMER_TARGET_DIRECT_VARIANT,
    ITRANSFORMER_TARGET_RESIDUAL_VARIANT,
    TIMEXER_FULL_RESIDUAL_VARIANT,
    TIMEXER_TARGET_DIRECT_VARIANT,
    TIMEXER_TARGET_RESIDUAL_VARIANT,
    TFT_DIRECT_VARIANT,
    TFT_RESIDUAL_VARIANT,
    MTGNN_CALENDAR_RESIDUAL_VARIANT,
    MTGNN_TARGET_VARIANT,
}
FORMAL_BLOCKER_BY_VARIANT_PREFIX = {
    "baseline_mlp_residual": "residual_control_training_not_implemented",
    "baseline_gru_residual": "residual_control_training_not_implemented",
    "baseline_tcn_residual": "residual_control_training_not_implemented",
}
DEFAULT_RIDGE_ALPHAS = (0.0, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0)
RIDGE_LAGS = (1, 2, 3, 6, 12, 18, 36, 72, 144)
DEFAULT_TFT_EVAL_WINDOW_CHUNK_SIZE = 1024


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


def _slice_windows(windows: Any, start: int, stop: int) -> Any:
    return replace(
        windows,
        target_indices=windows.target_indices[start:stop],
        output_start_us=windows.output_start_us[start:stop],
        output_end_us=windows.output_end_us[start:stop],
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


def _load_timexer_model(
    *,
    device: str,
    d_model: int,
    n_heads: int,
    e_layers: int,
    dropout: float,
    patch_len: int,
    enc_in: int,
):
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
        enc_in=enc_in,
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


def _load_tft_components():
    from lightning.pytorch import Trainer  # type: ignore
    from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer  # type: ignore
    from pytorch_forecasting.data import NaNLabelEncoder  # type: ignore
    from pytorch_forecasting.metrics import RMSE  # type: ignore

    return Trainer, TimeSeriesDataSet, TemporalFusionTransformer, NaNLabelEncoder, RMSE


def _mtgnn_net_module():
    source_root = REPO_ROOT / "experiment" / "official_baselines" / "mtgnn" / "source"
    if str(source_root) not in sys.path:
        sys.path.insert(0, str(source_root))
    module_name = "_world_model_official_mtgnn_net"
    module_path = source_root / "net.py"
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load MTGNN official source from {module_path}.")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _wrap_mtgnn_core(core: Any, *, calendar_channels: int, use_calendar: bool, device: str):
    import torch

    class MTGNNOfficialCoreModule(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.core = core
            self.calendar_head = torch.nn.Linear(calendar_channels, core.num_nodes) if use_calendar else None

        def forward(self, x: Any, future_calendar: Any | None = None) -> Any:
            raw = self.core(x)
            if raw.ndim != 4:
                raise RuntimeError(f"Expected MTGNN core output [B, H, N, T], found {tuple(raw.shape)}.")
            raw = raw[..., -1]
            if self.calendar_head is not None:
                if future_calendar is None:
                    raise RuntimeError("MTGNN calendar residual variant requires future calendar features.")
                raw = raw + self.calendar_head(future_calendar)
            return raw

    return MTGNNOfficialCoreModule().to(device)


def _load_mtgnn_model(
    *,
    prepared: Any,
    device: str,
    gcn_depth: int,
    subgraph_size: int,
    node_dim: int,
    residual_channels: int,
    skip_channels: int,
    end_channels: int,
    layers: int,
    dropout: float,
    use_calendar: bool,
):
    import torch

    module = _mtgnn_net_module()
    model_class = module.gtnet
    core = model_class(
        gcn_true=True,
        buildA_true=True,
        gcn_depth=gcn_depth,
        num_nodes=prepared.node_count,
        device=torch.device(device),
        predefined_A=None,
        static_feat=None,
        dropout=dropout,
        subgraph_size=min(int(subgraph_size), prepared.node_count),
        node_dim=node_dim,
        dilation_exponential=1,
        conv_channels=residual_channels,
        residual_channels=residual_channels,
        skip_channels=skip_channels,
        end_channels=end_channels,
        seq_length=prepared.history_steps,
        in_dim=1,
        out_dim=prepared.forecast_steps,
        layers=layers,
    )
    return _wrap_mtgnn_core(
        core,
        calendar_channels=int(prepared.context_future_tensor.shape[1]),
        use_calendar=use_calendar,
        device=device,
    )


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


def _iter_mtgnn_batches(
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
    calendar_channels = int(prepared.context_future_tensor.shape[1])
    fallback = prepared.persistence_train_fallback_pu.astype(np.float32, copy=False)
    for start in range(0, len(indices), batch_size):
        batch_window_indices = indices[start : start + batch_size]
        x = np.zeros((len(batch_window_indices), 1, prepared.node_count, prepared.history_steps), dtype=np.float32)
        future_calendar = np.zeros(
            (len(batch_window_indices), prepared.forecast_steps, calendar_channels),
            dtype=np.float32,
        )
        y = np.zeros((len(batch_window_indices), prepared.forecast_steps, prepared.node_count), dtype=np.float32)
        valid = np.zeros_like(y, dtype=np.float32)
        anchor = np.zeros((len(batch_window_indices), prepared.node_count), dtype=np.float32)
        for row_index, window_pos in enumerate(batch_window_indices):
            target_index = int(windows.target_indices[int(window_pos)])
            history = slice(target_index - prepared.history_steps, target_index)
            future = slice(target_index, target_index + prepared.forecast_steps)
            target_history = prepared.target_pu_filled[history].astype(np.float32, copy=False)
            target_unavailable = 1.0 - prepared.target_valid_mask[history].astype(np.float32, copy=False)
            x[row_index, 0] = target_history.T
            anchor[row_index] = _last_available_series(target_history, target_unavailable, fallback)
            future_calendar[row_index] = prepared.context_future_tensor[future].astype(np.float32, copy=False)
            y[row_index] = prepared.target_pu_filled[future]
            valid[row_index] = prepared.target_valid_mask[future].astype(np.float32, copy=False)
        yield x, future_calendar, y, valid, anchor


_TIMEXER_FULL_LOCAL_VALUE_INDICES = (1, 2, 3, 4, 5, 6)
_TIMEXER_FULL_CONTEXT_HISTORY_INDICES = tuple(range(0, 9)) + tuple(range(27, 40))


def _standardized_static_tensor(prepared: Any) -> np.ndarray:
    static = np.asarray(prepared.static_tensor, dtype=np.float32)
    if static.ndim != 2 or static.shape[0] != prepared.node_count or static.shape[1] == 0:
        return np.zeros((prepared.node_count, 0), dtype=np.float32)
    finite = np.isfinite(static)
    means = np.nanmean(np.where(finite, static, np.nan), axis=0)
    means = np.where(np.isfinite(means), means, 0.0).astype(np.float32)
    filled = np.where(finite, static, means[None, :]).astype(np.float32)
    std = filled.std(axis=0).astype(np.float32)
    std = np.where(std > 1e-6, std, 1.0).astype(np.float32)
    return ((filled - means[None, :]) / std[None, :]).astype(np.float32)


def _timexer_full_exog_channel_count(prepared: Any) -> int:
    calendar_count = int(prepared.context_future_tensor.shape[1])
    static_count = int(np.asarray(prepared.static_tensor).shape[1]) if np.asarray(prepared.static_tensor).ndim == 2 else 0
    return (
        prepared.node_count
        + len(_TIMEXER_FULL_LOCAL_VALUE_INDICES) * prepared.node_count
        + len(_TIMEXER_FULL_CONTEXT_HISTORY_INDICES)
        + calendar_count
        + static_count * prepared.node_count
    )


def _iter_timexer_batches(
    prepared: Any,
    windows: Any,
    *,
    batch_size: int,
    shuffle: bool,
    seed: int,
    full_exog: bool,
):
    if not full_exog:
        yield from _iter_target_only_batches(
            prepared,
            windows,
            batch_size=batch_size,
            shuffle=shuffle,
            seed=seed,
        )
        return

    state_base = _load_state_space_base()
    indices = np.arange(len(windows), dtype=np.int64)
    if shuffle:
        rng = np.random.default_rng(seed)
        rng.shuffle(indices)
    channel_count = _timexer_full_exog_channel_count(prepared)
    static_z = _standardized_static_tensor(prepared)
    for start in range(0, len(indices), batch_size):
        batch_window_indices = indices[start : start + batch_size]
        x = np.zeros((len(batch_window_indices), prepared.history_steps, channel_count), dtype=np.float32)
        y = np.zeros((len(batch_window_indices), prepared.forecast_steps, prepared.node_count), dtype=np.float32)
        valid = np.zeros_like(y, dtype=np.float32)
        anchor = np.zeros((len(batch_window_indices), prepared.node_count), dtype=np.float32)
        for row_index, window_pos in enumerate(batch_window_indices):
            target_index = int(windows.target_indices[int(window_pos)])
            history = slice(target_index - prepared.history_steps, target_index)
            future = slice(target_index, target_index + prepared.forecast_steps)
            target_history = prepared.local_history_tensor[history, :, 0]
            target_unavailable = prepared.local_history_tensor[history, :, state_base._LOCAL_MASK_START]
            last_values = _last_available_series(
                target_history,
                target_unavailable,
                prepared.persistence_train_fallback_pu.astype(np.float32, copy=False),
            )
            anchor[row_index] = last_values

            cursor = 0
            x[row_index, :, cursor : cursor + prepared.node_count] = target_history
            cursor += prepared.node_count

            for value_index in _TIMEXER_FULL_LOCAL_VALUE_INDICES:
                values = prepared.local_history_tensor[history, :, value_index].astype(np.float32, copy=True)
                mask_index = state_base._LOCAL_MASK_START + value_index
                masks = prepared.local_history_tensor[history, :, mask_index]
                values[masks >= 0.5] = 0.0
                x[row_index, :, cursor : cursor + prepared.node_count] = values
                cursor += prepared.node_count

            context_history = prepared.context_history_tensor[history]
            for context_index in _TIMEXER_FULL_CONTEXT_HISTORY_INDICES:
                x[row_index, :, cursor] = context_history[:, context_index]
                cursor += 1

            future_calendar_summary = prepared.context_future_tensor[future].mean(axis=0).astype(np.float32, copy=False)
            for value in future_calendar_summary:
                x[row_index, :, cursor] = float(value)
                cursor += 1

            for static_column in range(static_z.shape[1]):
                values = static_z[:, static_column]
                x[row_index, :, cursor : cursor + prepared.node_count] = values[None, :]
                cursor += prepared.node_count

            if cursor != channel_count:
                raise RuntimeError(f"TimeXer full-exog channel cursor mismatch: {cursor} != {channel_count}.")
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


_TFT_LOCAL_VALUE_INDICES = _TIMEXER_FULL_LOCAL_VALUE_INDICES
_TFT_CONTEXT_HISTORY_INDICES = _TIMEXER_FULL_CONTEXT_HISTORY_INDICES


def _tft_feature_names(prepared: Any) -> tuple[list[str], list[str], list[str]]:
    local_names = [f"local_{index}" for index in _TFT_LOCAL_VALUE_INDICES]
    context_names = [f"context_{index}" for index in _TFT_CONTEXT_HISTORY_INDICES]
    calendar_names = [f"calendar_{index}" for index in range(int(prepared.context_future_tensor.shape[1]))]
    return local_names, context_names, calendar_names


def _tft_frame(
    prepared: Any,
    windows: Any,
    *,
    residual_output: bool,
    residual_anchor_steps: int,
) -> tuple[Any, dict[str, tuple[int, int]]]:
    import pandas as pd

    state_base = _load_state_space_base()
    static_z = _standardized_static_tensor(prepared)
    local_names, context_names, calendar_names = _tft_feature_names(prepared)
    static_names = [f"static_{index}" for index in range(static_z.shape[1])]
    rows: list[dict[str, Any]] = []
    sample_map: dict[str, tuple[int, int]] = {}
    for window_pos, target_index_value in enumerate(windows.target_indices):
        target_index = int(target_index_value)
        history = slice(target_index - prepared.history_steps, target_index)
        future = slice(target_index, target_index + prepared.forecast_steps)
        target_history_all = prepared.local_history_tensor[history, :, 0]
        target_unavailable = prepared.local_history_tensor[history, :, state_base._LOCAL_MASK_START]
        anchors = _last_available_series(
            target_history_all,
            target_unavailable,
            prepared.persistence_train_fallback_pu.astype(np.float32, copy=False),
        )
        history_calendar = prepared.context_history_tensor[history, state_base._CONTEXT_CALENDAR_START :]
        future_calendar = prepared.context_future_tensor[future]
        context_history = prepared.context_history_tensor[history]
        for node_index in range(prepared.node_count):
            sample_id = f"w{window_pos}_n{node_index}"
            sample_map[sample_id] = (window_pos, node_index)
            anchor = float(anchors[node_index])
            target_values = np.concatenate(
                (
                    prepared.target_pu_filled[history, node_index],
                    prepared.target_pu_filled[future, node_index],
                )
            ).astype(np.float32, copy=True)
            if residual_output:
                target_values = target_values - anchor
                if residual_anchor_steps > 0:
                    start = prepared.history_steps
                    stop = min(prepared.history_steps + residual_anchor_steps, target_values.shape[0])
                    target_values[start:stop] = 0.0

            local_columns: dict[str, np.ndarray] = {}
            for local_name, value_index in zip(local_names, _TFT_LOCAL_VALUE_INDICES, strict=True):
                values = prepared.local_history_tensor[history, node_index, value_index].astype(np.float32, copy=True)
                mask_index = state_base._LOCAL_MASK_START + value_index
                masks = prepared.local_history_tensor[history, node_index, mask_index]
                values[masks >= 0.5] = 0.0
                local_columns[local_name] = np.concatenate(
                    (values, np.zeros((prepared.forecast_steps,), dtype=np.float32))
                )

            context_columns: dict[str, np.ndarray] = {}
            for context_name, context_index in zip(context_names, _TFT_CONTEXT_HISTORY_INDICES, strict=True):
                values = context_history[:, context_index].astype(np.float32, copy=True)
                context_columns[context_name] = np.concatenate(
                    (values, np.zeros((prepared.forecast_steps,), dtype=np.float32))
                )

            calendar_values = np.concatenate((history_calendar, future_calendar), axis=0).astype(np.float32, copy=False)
            for step_index in range(prepared.history_steps + prepared.forecast_steps):
                row: dict[str, Any] = {
                    "sample_id": sample_id,
                    "turbine_id": f"t{node_index}",
                    "time_idx": step_index,
                    "relative_time": float((step_index - prepared.history_steps + 1) / prepared.forecast_steps),
                    "is_decoder": float(step_index >= prepared.history_steps),
                    "target": float(target_values[step_index]),
                    "anchor": anchor,
                }
                for local_name in local_names:
                    row[local_name] = float(local_columns[local_name][step_index])
                for context_name in context_names:
                    row[context_name] = float(context_columns[context_name][step_index])
                for calendar_index, calendar_name in enumerate(calendar_names):
                    row[calendar_name] = float(calendar_values[step_index, calendar_index])
                for static_index, static_name in enumerate(static_names):
                    row[static_name] = float(static_z[node_index, static_index])
                rows.append(row)
    return pd.DataFrame.from_records(rows), sample_map


def _tft_dataset(
    frame: Any,
    *,
    categorical_encoders: dict[str, Any] | None = None,
    training: Any | None = None,
    predict: bool = False,
):
    _Trainer, TimeSeriesDataSet, _TemporalFusionTransformer, NaNLabelEncoder, _RMSE = _load_tft_components()
    local_names = [column for column in frame.columns if column.startswith("local_")]
    context_names = [column for column in frame.columns if column.startswith("context_")]
    calendar_names = [column for column in frame.columns if column.startswith("calendar_")]
    static_names = [column for column in frame.columns if column.startswith("static_")]
    if training is not None:
        return TimeSeriesDataSet.from_dataset(training, frame, predict=predict, stop_randomization=True)
    encoders = categorical_encoders or {
        "sample_id": NaNLabelEncoder(add_nan=True),
        "turbine_id": NaNLabelEncoder(add_nan=True),
    }
    return TimeSeriesDataSet(
        frame,
        time_idx="time_idx",
        target="target",
        group_ids=["sample_id"],
        min_encoder_length=HISTORY_STEPS,
        max_encoder_length=HISTORY_STEPS,
        min_prediction_length=FORECAST_STEPS,
        max_prediction_length=FORECAST_STEPS,
        static_categoricals=["turbine_id"],
        static_reals=["anchor", *static_names],
        time_varying_known_reals=["relative_time", "is_decoder", *calendar_names],
        time_varying_unknown_reals=["target", *local_names, *context_names],
        target_normalizer=None,
        categorical_encoders=encoders,
        add_relative_time_idx=False,
        add_target_scales=False,
        add_encoder_length=False,
        randomize_length=False,
    )


def _train_tft(
    prepared: Any,
    *,
    variant_name: str,
    validation_windows: Any,
    residual_anchor_steps: int,
    seed: int,
    device: str,
    batch_size: int,
    learning_rate: float,
    max_epochs: int,
    hidden_size: int,
    lstm_layers: int,
    attention_head_size: int,
    hidden_continuous_size: int,
    dropout: float,
) -> tuple[Any, Any, dict[str, Any]]:
    import torch

    Trainer, _TimeSeriesDataSet, TemporalFusionTransformer, NaNLabelEncoder, RMSE = _load_tft_components()
    torch.manual_seed(seed)
    if device == "cuda" and torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    resolved_device = device if device != "cuda" or torch.cuda.is_available() else "cpu"
    residual_output = variant_name == TFT_RESIDUAL_VARIANT
    train_frame, _train_sample_map = _tft_frame(
        prepared,
        prepared.train_windows,
        residual_output=residual_output,
        residual_anchor_steps=residual_anchor_steps,
    )
    validation_frame, _validation_sample_map = _tft_frame(
        prepared,
        validation_windows,
        residual_output=residual_output,
        residual_anchor_steps=residual_anchor_steps,
    )
    sample_encoder = NaNLabelEncoder(add_nan=True)
    sample_encoder.fit(np.concatenate((train_frame["sample_id"].to_numpy(), validation_frame["sample_id"].to_numpy())))
    turbine_encoder = NaNLabelEncoder(add_nan=True)
    turbine_encoder.fit(np.concatenate((train_frame["turbine_id"].to_numpy(), validation_frame["turbine_id"].to_numpy())))
    training = _tft_dataset(
        train_frame,
        categorical_encoders={"sample_id": sample_encoder, "turbine_id": turbine_encoder},
    )
    validation = _tft_dataset(validation_frame, training=training, predict=True)
    train_loader = training.to_dataloader(train=True, batch_size=batch_size, num_workers=0)
    validation_loader = validation.to_dataloader(train=False, batch_size=batch_size, num_workers=0)
    model = TemporalFusionTransformer.from_dataset(
        training,
        learning_rate=learning_rate,
        hidden_size=hidden_size,
        lstm_layers=lstm_layers,
        attention_head_size=attention_head_size,
        hidden_continuous_size=hidden_continuous_size,
        dropout=dropout,
        loss=RMSE(),
        log_interval=-1,
        reduce_on_plateau_patience=3,
    )
    trainer = Trainer(
        max_epochs=max_epochs,
        accelerator="gpu" if resolved_device == "cuda" else "cpu",
        devices=1,
        logger=False,
        enable_checkpointing=False,
        enable_model_summary=False,
        enable_progress_bar=False,
        gradient_clip_val=0.1,
        num_sanity_val_steps=0,
    )
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=validation_loader)
    model.eval()
    return model, training, {
        "epochs_ran": max_epochs,
        "device": resolved_device,
        "train_sample_count": int(train_frame["sample_id"].nunique()),
        "validation_sample_count": int(validation_frame["sample_id"].nunique()),
    }


def _evaluate_tft(
    model: Any,
    training_dataset: Any,
    prepared: Any,
    windows: Any,
    *,
    variant_name: str,
    device: str,
    batch_size: int,
    residual_anchor_steps: int,
    eval_window_chunk_size: int | None = DEFAULT_TFT_EVAL_WINDOW_CHUNK_SIZE,
) -> np.ndarray:
    chunk_size = int(eval_window_chunk_size or 0)
    if chunk_size > 0 and len(windows) > chunk_size:
        predictions = np.zeros((len(windows), prepared.forecast_steps, prepared.node_count), dtype=np.float32)
        for start in range(0, len(windows), chunk_size):
            stop = min(start + chunk_size, len(windows))
            chunk_windows = _slice_windows(windows, start, stop)
            predictions[start:stop] = _evaluate_tft_single_chunk(
                model,
                training_dataset,
                prepared,
                chunk_windows,
                variant_name=variant_name,
                device=device,
                batch_size=batch_size,
                residual_anchor_steps=residual_anchor_steps,
            )
        return predictions
    return _evaluate_tft_single_chunk(
        model,
        training_dataset,
        prepared,
        windows,
        variant_name=variant_name,
        device=device,
        batch_size=batch_size,
        residual_anchor_steps=residual_anchor_steps,
    )


def _evaluate_tft_single_chunk(
    model: Any,
    training_dataset: Any,
    prepared: Any,
    windows: Any,
    *,
    variant_name: str,
    device: str,
    batch_size: int,
    residual_anchor_steps: int,
) -> np.ndarray:
    residual_output = variant_name == TFT_RESIDUAL_VARIANT
    frame, sample_map = _tft_frame(
        prepared,
        windows,
        residual_output=residual_output,
        residual_anchor_steps=residual_anchor_steps,
    )
    dataset = _tft_dataset(frame, training=training_dataset, predict=True)
    output = _predict_tft_dataset(
        model,
        dataset,
        batch_size=batch_size,
        device=device,
    )
    if output.ndim == 3 and output.shape[-1] == 1:
        output = output[..., 0]
    if output.shape[1] != prepared.forecast_steps:
        raise RuntimeError(f"Expected TFT prediction horizon {prepared.forecast_steps}, found {output.shape!r}.")
    predictions = np.zeros((len(windows), prepared.forecast_steps, prepared.node_count), dtype=np.float32)
    ordered_sample_ids = list(dict.fromkeys(frame["sample_id"].tolist()))
    if output.shape[0] != len(ordered_sample_ids):
        raise RuntimeError(
            f"Expected {len(ordered_sample_ids)} TFT prediction rows, found {output.shape[0]}."
        )
    anchors = dict(frame.groupby("sample_id", sort=False)["anchor"].first())
    for row_index, sample_id in enumerate(ordered_sample_ids):
        window_pos, node_index = sample_map[sample_id]
        values = output[row_index].astype(np.float32, copy=True)
        if residual_output:
            if residual_anchor_steps > 0:
                values[:residual_anchor_steps] = 0.0
            values = values + float(anchors[sample_id])
        predictions[window_pos, :, node_index] = values
    return predictions


def _move_tft_batch_to_device(value: Any, device: Any) -> Any:
    import torch

    if isinstance(value, torch.Tensor):
        return value.to(device)
    if isinstance(value, dict):
        return {key: _move_tft_batch_to_device(item, device) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(_move_tft_batch_to_device(item, device) for item in value)
    if isinstance(value, list):
        return [_move_tft_batch_to_device(item, device) for item in value]
    return value


def _tft_prediction_from_forward(output: Any) -> Any:
    if isinstance(output, dict) and "prediction" in output:
        return output["prediction"]
    if hasattr(output, "prediction"):
        return output.prediction
    try:
        return output["prediction"]
    except Exception as exc:  # pragma: no cover - defensive adapter guard.
        raise RuntimeError("TFT forward output did not expose a prediction tensor.") from exc


def _predict_tft_dataset(
    model: Any,
    dataset: Any,
    *,
    batch_size: int,
    device: str,
) -> np.ndarray:
    import torch

    resolved_device = torch.device("cuda" if device == "cuda" and torch.cuda.is_available() else "cpu")
    model.to(resolved_device)
    model.eval()
    loader = dataset.to_dataloader(train=False, batch_size=batch_size, num_workers=0)
    outputs: list[np.ndarray] = []
    with torch.inference_mode():
        for batch in loader:
            x, _y = batch
            x = _move_tft_batch_to_device(x, resolved_device)
            prediction = _tft_prediction_from_forward(model(x))
            outputs.append(prediction.detach().cpu().numpy().astype(np.float32, copy=False))
    if not outputs:
        return np.zeros((0, FORECAST_STEPS), dtype=np.float32)
    return np.concatenate(outputs, axis=0)


def _masked_mse_torch(predictions: Any, targets: Any, valid: Any) -> Any:
    denominator = valid.sum().clamp_min(1.0)
    return (((predictions - targets) ** 2) * valid).sum() / denominator


def _anchored_loss_valid(valid: Any, *, residual_output: bool, residual_anchor_steps: int) -> Any:
    if not residual_output or residual_anchor_steps <= 0:
        return valid
    anchored = valid.clone()
    anchored[:, :residual_anchor_steps, :] = 0.0
    return anchored


def _apply_residual_anchor_torch(raw: Any, *, residual_output: bool, residual_anchor_steps: int) -> Any:
    if not residual_output or residual_anchor_steps <= 0:
        return raw
    anchored = raw.clone()
    anchored[:, :residual_anchor_steps, :] = 0.0
    return anchored


def _apply_residual_anchor_numpy(raw: np.ndarray, *, residual_output: bool, residual_anchor_steps: int) -> np.ndarray:
    if not residual_output or residual_anchor_steps <= 0:
        return raw
    anchored = raw.copy()
    anchored[:, :residual_anchor_steps, :] = 0.0
    return anchored


def _train_itransformer(
    prepared: Any,
    *,
    variant_name: str,
    validation_windows: Any,
    residual_anchor_steps: int,
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
    residual_output = variant_name in {ITRANSFORMER_TARGET_RESIDUAL_VARIANT, ITRANSFORMER_EXOG_RESIDUAL_VARIANT}
    full_exog = variant_name == ITRANSFORMER_EXOG_RESIDUAL_VARIANT
    for epoch in range(max_epochs):
        model.train()
        train_loss_sum = 0.0
        train_batches = 0
        for x_np, y_np, valid_np, anchor_np in _iter_timexer_batches(
            prepared,
            prepared.train_windows,
            batch_size=batch_size,
            shuffle=True,
            seed=seed + epoch,
            full_exog=full_exog,
        ):
            x = torch.as_tensor(x_np, device=resolved_device)
            y = torch.as_tensor(y_np, device=resolved_device)
            valid = torch.as_tensor(valid_np, device=resolved_device)
            anchor = torch.as_tensor(anchor_np, device=resolved_device)
            target = y - anchor[:, None, :] if residual_output else y
            optimizer.zero_grad(set_to_none=True)
            raw = model(x, None, None, None)
            if full_exog:
                raw = raw[..., : prepared.node_count]
            raw = _apply_residual_anchor_torch(
                raw,
                residual_output=residual_output,
                residual_anchor_steps=residual_anchor_steps,
            )
            loss = _masked_mse_torch(
                raw,
                target,
                _anchored_loss_valid(
                    valid,
                    residual_output=residual_output,
                    residual_anchor_steps=residual_anchor_steps,
                ),
            )
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
            residual_anchor_steps=residual_anchor_steps,
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
    residual_anchor_steps: int,
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
            raw = _apply_residual_anchor_torch(
                raw,
                residual_output=residual_output,
                residual_anchor_steps=residual_anchor_steps,
            )
            loss = _masked_mse_torch(
                raw,
                target,
                _anchored_loss_valid(
                    valid,
                    residual_output=residual_output,
                    residual_anchor_steps=residual_anchor_steps,
                ),
            )
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
            residual_anchor_steps=residual_anchor_steps,
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
    residual_anchor_steps: int,
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
    full_exog = variant_name == TIMEXER_FULL_RESIDUAL_VARIANT
    model = _load_timexer_model(
        device=resolved_device,
        d_model=d_model,
        n_heads=n_heads,
        e_layers=e_layers,
        dropout=dropout,
        patch_len=patch_len,
        enc_in=_timexer_full_exog_channel_count(prepared) if full_exog else prepared.node_count,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    best_state = None
    best_val_rmse = math.inf
    best_val_mae = math.inf
    best_epoch = -1
    history: list[dict[str, Any]] = []
    residual_output = variant_name in {TIMEXER_TARGET_RESIDUAL_VARIANT, TIMEXER_FULL_RESIDUAL_VARIANT}
    for epoch in range(max_epochs):
        model.train()
        train_loss_sum = 0.0
        train_batches = 0
        for x_np, y_np, valid_np, anchor_np in _iter_timexer_batches(
            prepared,
            prepared.train_windows,
            batch_size=batch_size,
            shuffle=True,
            seed=seed + epoch,
            full_exog=full_exog,
        ):
            x = torch.as_tensor(x_np, device=resolved_device)
            y = torch.as_tensor(y_np, device=resolved_device)
            valid = torch.as_tensor(valid_np, device=resolved_device)
            anchor = torch.as_tensor(anchor_np, device=resolved_device)
            target = y - anchor[:, None, :] if residual_output else y
            optimizer.zero_grad(set_to_none=True)
            raw = model(x, None, None, None)[..., : prepared.node_count]
            raw = _apply_residual_anchor_torch(
                raw,
                residual_output=residual_output,
                residual_anchor_steps=residual_anchor_steps,
            )
            loss = _masked_mse_torch(
                raw,
                target,
                _anchored_loss_valid(
                    valid,
                    residual_output=residual_output,
                    residual_anchor_steps=residual_anchor_steps,
                ),
            )
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
            residual_anchor_steps=residual_anchor_steps,
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


def _train_mtgnn(
    prepared: Any,
    *,
    variant_name: str,
    validation_windows: Any,
    residual_anchor_steps: int,
    seed: int,
    device: str,
    batch_size: int,
    learning_rate: float,
    max_epochs: int,
    gcn_depth: int,
    subgraph_size: int,
    node_dim: int,
    residual_channels: int,
    skip_channels: int,
    end_channels: int,
    layers: int,
    dropout: float,
) -> tuple[Any, dict[str, Any]]:
    import torch

    torch.manual_seed(seed)
    if device == "cuda" and torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    resolved_device = device if device != "cuda" or torch.cuda.is_available() else "cpu"
    residual_output = variant_name == MTGNN_CALENDAR_RESIDUAL_VARIANT
    use_calendar = variant_name == MTGNN_CALENDAR_RESIDUAL_VARIANT
    model = _load_mtgnn_model(
        prepared=prepared,
        device=resolved_device,
        gcn_depth=gcn_depth,
        subgraph_size=subgraph_size,
        node_dim=node_dim,
        residual_channels=residual_channels,
        skip_channels=skip_channels,
        end_channels=end_channels,
        layers=layers,
        dropout=dropout,
        use_calendar=use_calendar,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    best_state = None
    best_val_rmse = math.inf
    best_val_mae = math.inf
    best_epoch = -1
    history: list[dict[str, Any]] = []
    for epoch in range(max_epochs):
        model.train()
        train_loss_sum = 0.0
        train_batches = 0
        for x_np, calendar_np, y_np, valid_np, anchor_np in _iter_mtgnn_batches(
            prepared,
            prepared.train_windows,
            batch_size=batch_size,
            shuffle=True,
            seed=seed + epoch,
        ):
            x = torch.as_tensor(x_np, device=resolved_device)
            future_calendar = torch.as_tensor(calendar_np, device=resolved_device)
            y = torch.as_tensor(y_np, device=resolved_device)
            valid = torch.as_tensor(valid_np, device=resolved_device)
            anchor = torch.as_tensor(anchor_np, device=resolved_device)
            target = y - anchor[:, None, :] if residual_output else y
            optimizer.zero_grad(set_to_none=True)
            raw = model(x, future_calendar if use_calendar else None)
            raw = _apply_residual_anchor_torch(
                raw,
                residual_output=residual_output,
                residual_anchor_steps=residual_anchor_steps,
            )
            loss = _masked_mse_torch(
                raw,
                target,
                _anchored_loss_valid(
                    valid,
                    residual_output=residual_output,
                    residual_anchor_steps=residual_anchor_steps,
                ),
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss_sum += float(loss.detach().cpu())
            train_batches += 1
        val_predictions = _evaluate_mtgnn(
            model,
            prepared,
            validation_windows,
            variant_name=variant_name,
            device=resolved_device,
            batch_size=batch_size,
            residual_anchor_steps=residual_anchor_steps,
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
    residual_anchor_steps: int,
) -> np.ndarray:
    import torch

    residual_output = variant_name in {ITRANSFORMER_TARGET_RESIDUAL_VARIANT, ITRANSFORMER_EXOG_RESIDUAL_VARIANT}
    full_exog = variant_name == ITRANSFORMER_EXOG_RESIDUAL_VARIANT
    predictions = np.zeros((len(windows), prepared.forecast_steps, prepared.node_count), dtype=np.float32)
    model.eval()
    offset = 0
    with torch.no_grad():
        for x_np, _y_np, _valid_np, anchor_np in _iter_timexer_batches(
            prepared,
            windows,
            batch_size=batch_size,
            shuffle=False,
            seed=0,
            full_exog=full_exog,
        ):
            x = torch.as_tensor(x_np, device=device)
            raw = model(x, None, None, None).detach().cpu().numpy().astype(np.float32, copy=False)
            if full_exog:
                raw = raw[..., : prepared.node_count]
            raw = _apply_residual_anchor_numpy(
                raw,
                residual_output=residual_output,
                residual_anchor_steps=residual_anchor_steps,
            )
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
    residual_anchor_steps: int,
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
            raw_np = _apply_residual_anchor_numpy(
                raw_np,
                residual_output=residual_output,
                residual_anchor_steps=residual_anchor_steps,
            )
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
    residual_anchor_steps: int,
) -> np.ndarray:
    import torch

    residual_output = variant_name in {TIMEXER_TARGET_RESIDUAL_VARIANT, TIMEXER_FULL_RESIDUAL_VARIANT}
    full_exog = variant_name == TIMEXER_FULL_RESIDUAL_VARIANT
    predictions = np.zeros((len(windows), prepared.forecast_steps, prepared.node_count), dtype=np.float32)
    model.eval()
    offset = 0
    with torch.no_grad():
        for x_np, _y_np, _valid_np, anchor_np in _iter_timexer_batches(
            prepared,
            windows,
            batch_size=batch_size,
            shuffle=False,
            seed=0,
            full_exog=full_exog,
        ):
            x = torch.as_tensor(x_np, device=device)
            raw = model(x, None, None, None)[..., : prepared.node_count].detach().cpu().numpy().astype(np.float32, copy=False)
            raw = _apply_residual_anchor_numpy(
                raw,
                residual_output=residual_output,
                residual_anchor_steps=residual_anchor_steps,
            )
            if residual_output:
                raw = raw + anchor_np[:, None, :]
            predictions[offset : offset + raw.shape[0]] = raw
            offset += raw.shape[0]
    return predictions


def _evaluate_mtgnn(
    model: Any,
    prepared: Any,
    windows: Any,
    *,
    variant_name: str,
    device: str,
    batch_size: int,
    residual_anchor_steps: int,
) -> np.ndarray:
    import torch

    residual_output = variant_name == MTGNN_CALENDAR_RESIDUAL_VARIANT
    use_calendar = variant_name == MTGNN_CALENDAR_RESIDUAL_VARIANT
    predictions = np.zeros((len(windows), prepared.forecast_steps, prepared.node_count), dtype=np.float32)
    model.eval()
    offset = 0
    with torch.no_grad():
        for x_np, calendar_np, _y_np, _valid_np, anchor_np in _iter_mtgnn_batches(
            prepared,
            windows,
            batch_size=batch_size,
            shuffle=False,
            seed=0,
        ):
            x = torch.as_tensor(x_np, device=device)
            future_calendar = torch.as_tensor(calendar_np, device=device)
            raw = model(x, future_calendar if use_calendar else None).detach().cpu().numpy().astype(np.float32, copy=False)
            raw = _apply_residual_anchor_numpy(
                raw,
                residual_output=residual_output,
                residual_anchor_steps=residual_anchor_steps,
            )
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
    residual_anchor_steps: int,
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
        residual_anchor_steps=residual_anchor_steps,
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
        residual_anchor_steps=residual_anchor_steps,
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
    residual_anchor_steps: int,
    best_trial: bool,
    window_specs: Sequence[tuple[str, str, Any]],
    gate_b_scope: str | None = None,
    gate_b_overfit64_passed: bool | None = None,
    train_gate_after_fit_passed: bool | None = None,
    train_gate_after_fit_rmse_pu: float | None = None,
    train_gate_after_fit_mae_pu: float | None = None,
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
                "gate_b_scope": gate_b_scope,
                "gate_b_overfit64_passed": gate_b_overfit64_passed,
                "train_gate_after_fit_passed": train_gate_after_fit_passed,
                "train_gate_after_fit_rmse_pu": train_gate_after_fit_rmse_pu,
                "train_gate_after_fit_mae_pu": train_gate_after_fit_mae_pu,
                "gate_c_passed": gate_c_passed,
                "residual_anchor_steps": residual_anchor_steps,
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
        "gate_b_scope": "not_started" if spec.trainable else None,
        "gate_b_overfit64_passed": False if spec.trainable else None,
        "train_gate_after_fit_passed": None,
        "train_gate_after_fit_rmse_pu": None,
        "train_gate_after_fit_mae_pu": None,
        "gate_c_passed": False if spec.trainable else None,
        "residual_anchor_steps": 0,
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
    residual_anchor_steps: int = 0,
    dgcrn_hidden_dim: int = 64,
    dgcrn_dropout: float = 0.1,
    dgcrn_gcn_depth: int = 2,
    itransformer_d_model: int = 64,
    itransformer_n_heads: int = 4,
    itransformer_e_layers: int = 2,
    itransformer_dropout: float = 0.1,
    tft_hidden_size: int = 32,
    tft_lstm_layers: int = 1,
    tft_attention_head_size: int = 4,
    tft_hidden_continuous_size: int = 16,
    tft_dropout: float = 0.1,
    tft_eval_window_chunk_size: int = DEFAULT_TFT_EVAL_WINDOW_CHUNK_SIZE,
    mtgnn_gcn_depth: int = 2,
    mtgnn_subgraph_size: int = 6,
    mtgnn_node_dim: int = 40,
    mtgnn_residual_channels: int = 32,
    mtgnn_skip_channels: int = 64,
    mtgnn_end_channels: int = 128,
    mtgnn_layers: int = 3,
    mtgnn_dropout: float = 0.3,
    timexer_d_model: int = 64,
    timexer_n_heads: int = 4,
    timexer_e_layers: int = 2,
    timexer_dropout: float = 0.1,
    timexer_patch_len: int = 16,
    gate_b_overfit64_passed: bool | None = None,
    gate_b_overfit64_rmse_pu: float | None = None,
    gate_b_overfit64_mae_pu: float | None = None,
    gate_b_overfit64_source: str | None = None,
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
                        residual_anchor_steps=0,
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
                        residual_anchor_steps=0,
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
                            residual_anchor_steps=0,
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
                        residual_anchor_steps=0,
                        best_trial=True,
                        window_specs=window_specs,
                    )
                )
            elif spec.model_variant in {DGCRN_DIRECT_VARIANT, DGCRN_RESIDUAL_VARIANT}:
                model, train_summary = _train_dgcrn(
                    prepared,
                    variant_name=spec.model_variant,
                    validation_windows=checkpoint_windows,
                    residual_anchor_steps=residual_anchor_steps,
                    seed=seed,
                    device=device,
                    batch_size=train_batch_size,
                    learning_rate=learning_rate,
                    max_epochs=max_epochs,
                    hidden_dim=dgcrn_hidden_dim,
                    dropout=dgcrn_dropout,
                    gcn_depth=dgcrn_gcn_depth,
                )
                dgcrn_search_config_id = (
                    f"dgcrn_official_core_h{dgcrn_hidden_dim}"
                    f"_dropout{dgcrn_dropout:g}"
                    f"_gcn{dgcrn_gcn_depth}"
                    f"_lr{learning_rate:g}"
                    f"_anchor{residual_anchor_steps if spec.output_parameterization == 'residual' else 0}"
                )
                predictions_by_split = {
                    (split_name, eval_protocol): _evaluate_dgcrn(
                        model,
                        prepared,
                        windows,
                        variant_name=spec.model_variant,
                        device=train_summary["device"],
                        batch_size=train_batch_size,
                        residual_anchor_steps=residual_anchor_steps,
                    )
                    for split_name, eval_protocol, windows in window_specs
                }
                train_gate_after_fit_passed, gate_c_passed, train_gate_metrics, _gate_c_metrics = _gate_status_for_neural_model(
                    evaluator=_evaluate_dgcrn,
                    model=model,
                    prepared=prepared,
                    variant_name=spec.model_variant,
                    device=train_summary["device"],
                    batch_size=train_batch_size,
                    residual_anchor_steps=residual_anchor_steps,
                    train_gate_windows=train_gate_windows,
                    gate_c_windows=gate_c_windows,
                    persistence_train_rmse=float(persistence_train_gate_metrics["rmse_pu"]),
                    persistence_gate_c_lead1_rmse=float(persistence_gate_c_metrics["lead1_rmse_pu"]),
                    persistence_gate_c_lead1_mae=float(persistence_gate_c_metrics["lead1_mae_pu"]),
                )
                gate_b_for_row = (
                    gate_b_overfit64_passed if gate_b_overfit64_passed is not None else train_gate_after_fit_passed
                )
                gate_b_scope = (
                    "overfit64_preflight"
                    if gate_b_overfit64_passed is not None
                    else "train_gate_after_fit"
                )
                rows.extend(
                    _metric_rows(
                        spec,
                        prepared=prepared,
                        seed=seed,
                        trial_id=(
                            f"dgcrn_official_core_residual_{dgcrn_search_config_id}"
                            if spec.model_variant == DGCRN_RESIDUAL_VARIANT
                            else f"dgcrn_official_core_direct_{dgcrn_search_config_id}"
                        ),
                        search_config_id=dgcrn_search_config_id,
                        alpha=None,
                        predictions_by_split=predictions_by_split,
                        runtime_seconds=time.perf_counter() - started,
                        gate_b_passed=gate_b_for_row,
                        gate_c_passed=gate_c_passed,
                        residual_anchor_steps=residual_anchor_steps if spec.output_parameterization == "residual" else 0,
                        best_trial=True,
                        window_specs=window_specs,
                        gate_b_scope=gate_b_scope,
                        gate_b_overfit64_passed=gate_b_overfit64_passed,
                        train_gate_after_fit_passed=train_gate_after_fit_passed,
                        train_gate_after_fit_rmse_pu=float(train_gate_metrics["rmse_pu"]),
                        train_gate_after_fit_mae_pu=float(train_gate_metrics["mae_pu"]),
                    )
                )
            elif spec.model_variant in {
                TIMEXER_TARGET_DIRECT_VARIANT,
                TIMEXER_TARGET_RESIDUAL_VARIANT,
                TIMEXER_FULL_RESIDUAL_VARIANT,
            }:
                timexer_feature_mode = "full_exog" if spec.model_variant == TIMEXER_FULL_RESIDUAL_VARIANT else "target_only"
                timexer_search_config_id = (
                    f"timexer_{timexer_feature_mode}_d{timexer_d_model}_h{timexer_n_heads}_e{timexer_e_layers}"
                    f"_dropout{timexer_dropout:g}_patch{timexer_patch_len}_lr{learning_rate:g}"
                    f"_anchor{residual_anchor_steps if spec.output_parameterization == 'residual' else 0}"
                )
                model, train_summary = _train_timexer(
                    prepared,
                    variant_name=spec.model_variant,
                    validation_windows=checkpoint_windows,
                    residual_anchor_steps=residual_anchor_steps,
                    seed=seed,
                    device=device,
                    batch_size=train_batch_size,
                    learning_rate=learning_rate,
                    max_epochs=max_epochs,
                    d_model=timexer_d_model,
                    n_heads=timexer_n_heads,
                    e_layers=timexer_e_layers,
                    dropout=timexer_dropout,
                    patch_len=timexer_patch_len,
                )
                predictions_by_split = {
                    (split_name, eval_protocol): _evaluate_timexer(
                        model,
                        prepared,
                        windows,
                        variant_name=spec.model_variant,
                        device=train_summary["device"],
                        batch_size=train_batch_size,
                        residual_anchor_steps=residual_anchor_steps,
                    )
                    for split_name, eval_protocol, windows in window_specs
                }
                train_gate_after_fit_passed, gate_c_passed, train_gate_metrics, _gate_c_metrics = _gate_status_for_neural_model(
                    evaluator=_evaluate_timexer,
                    model=model,
                    prepared=prepared,
                    variant_name=spec.model_variant,
                    device=train_summary["device"],
                    batch_size=train_batch_size,
                    residual_anchor_steps=residual_anchor_steps,
                    train_gate_windows=train_gate_windows,
                    gate_c_windows=gate_c_windows,
                    persistence_train_rmse=float(persistence_train_gate_metrics["rmse_pu"]),
                    persistence_gate_c_lead1_rmse=float(persistence_gate_c_metrics["lead1_rmse_pu"]),
                    persistence_gate_c_lead1_mae=float(persistence_gate_c_metrics["lead1_mae_pu"]),
                )
                gate_b_for_row = (
                    gate_b_overfit64_passed if gate_b_overfit64_passed is not None else train_gate_after_fit_passed
                )
                gate_b_scope = (
                    "overfit64_preflight"
                    if gate_b_overfit64_passed is not None
                    else "train_gate_after_fit"
                )
                rows.extend(
                    _metric_rows(
                        spec,
                        prepared=prepared,
                        seed=seed,
                        trial_id=(
                            f"timexer_{timexer_feature_mode}_residual_{timexer_search_config_id}"
                            if spec.output_parameterization == "residual"
                            else f"timexer_target_only_direct_{timexer_search_config_id}"
                        ),
                        search_config_id=timexer_search_config_id,
                        alpha=None,
                        predictions_by_split=predictions_by_split,
                        runtime_seconds=time.perf_counter() - started,
                        gate_b_passed=gate_b_for_row,
                        gate_c_passed=gate_c_passed,
                        residual_anchor_steps=residual_anchor_steps if spec.output_parameterization == "residual" else 0,
                        best_trial=True,
                        window_specs=window_specs,
                        gate_b_scope=gate_b_scope,
                        gate_b_overfit64_passed=gate_b_overfit64_passed,
                        train_gate_after_fit_passed=train_gate_after_fit_passed,
                        train_gate_after_fit_rmse_pu=float(train_gate_metrics["rmse_pu"]),
                        train_gate_after_fit_mae_pu=float(train_gate_metrics["mae_pu"]),
                    )
                )
            elif spec.model_variant in {
                ITRANSFORMER_TARGET_DIRECT_VARIANT,
                ITRANSFORMER_TARGET_RESIDUAL_VARIANT,
                ITRANSFORMER_EXOG_RESIDUAL_VARIANT,
            }:
                itransformer_feature_mode = (
                    "target_plus_exog" if spec.model_variant == ITRANSFORMER_EXOG_RESIDUAL_VARIANT else "target_only"
                )
                itransformer_search_config_id = (
                    f"itransformer_{itransformer_feature_mode}_d{itransformer_d_model}_h{itransformer_n_heads}"
                    f"_e{itransformer_e_layers}_dropout{itransformer_dropout}_lr{learning_rate}"
                    f"_anchor{residual_anchor_steps if spec.output_parameterization == 'residual' else 0}"
                )
                model, train_summary = _train_itransformer(
                    prepared,
                    variant_name=spec.model_variant,
                    validation_windows=checkpoint_windows,
                    residual_anchor_steps=residual_anchor_steps,
                    seed=seed,
                    device=device,
                    batch_size=train_batch_size,
                    learning_rate=learning_rate,
                    max_epochs=max_epochs,
                    d_model=itransformer_d_model,
                    n_heads=itransformer_n_heads,
                    e_layers=itransformer_e_layers,
                    dropout=itransformer_dropout,
                )
                predictions_by_split = {
                    (split_name, eval_protocol): _evaluate_itransformer(
                        model,
                        prepared,
                        windows,
                        variant_name=spec.model_variant,
                        device=train_summary["device"],
                        batch_size=train_batch_size,
                        residual_anchor_steps=residual_anchor_steps,
                        eval_window_chunk_size=tft_eval_window_chunk_size,
                    )
                    for split_name, eval_protocol, windows in window_specs
                }
                gate_b_passed, gate_c_passed, gate_b_metrics, _gate_c_metrics = _gate_status_for_neural_model(
                    evaluator=_evaluate_itransformer,
                    model=model,
                    prepared=prepared,
                    variant_name=spec.model_variant,
                    device=train_summary["device"],
                    batch_size=train_batch_size,
                    residual_anchor_steps=residual_anchor_steps,
                    train_gate_windows=train_gate_windows,
                    gate_c_windows=gate_c_windows,
                    persistence_train_rmse=float(persistence_train_gate_metrics["rmse_pu"]),
                    persistence_gate_c_lead1_rmse=float(persistence_gate_c_metrics["lead1_rmse_pu"]),
                    persistence_gate_c_lead1_mae=float(persistence_gate_c_metrics["lead1_mae_pu"]),
                )
                gate_b_for_row = (
                    gate_b_overfit64_passed if gate_b_overfit64_passed is not None else gate_b_passed
                )
                gate_b_scope = (
                    "overfit64_preflight"
                    if gate_b_overfit64_passed is not None
                    else "train_gate_after_fit"
                )
                rows.extend(
                    _metric_rows(
                        spec,
                        prepared=prepared,
                        seed=seed,
                        trial_id=f"itransformer_{itransformer_feature_mode}_{spec.output_parameterization}_{itransformer_search_config_id}",
                        search_config_id=itransformer_search_config_id,
                        alpha=None,
                        predictions_by_split=predictions_by_split,
                        runtime_seconds=time.perf_counter() - started,
                        gate_b_passed=gate_b_for_row,
                        gate_c_passed=gate_c_passed,
                        residual_anchor_steps=residual_anchor_steps if spec.output_parameterization == "residual" else 0,
                        best_trial=True,
                        window_specs=window_specs,
                        gate_b_scope=gate_b_scope,
                        gate_b_overfit64_passed=gate_b_overfit64_passed,
                        train_gate_after_fit_passed=gate_b_passed,
                        train_gate_after_fit_rmse_pu=float(gate_b_metrics["rmse_pu"]),
                        train_gate_after_fit_mae_pu=float(gate_b_metrics["mae_pu"]),
                    )
                )
            elif spec.model_variant in {MTGNN_TARGET_VARIANT, MTGNN_CALENDAR_RESIDUAL_VARIANT}:
                mtgnn_feature_mode = (
                    "calendar" if spec.model_variant == MTGNN_CALENDAR_RESIDUAL_VARIANT else "target_only"
                )
                mtgnn_search_config_id = (
                    f"mtgnn_{mtgnn_feature_mode}_gcn{mtgnn_gcn_depth}_sub{mtgnn_subgraph_size}"
                    f"_node{mtgnn_node_dim}_res{mtgnn_residual_channels}_skip{mtgnn_skip_channels}"
                    f"_end{mtgnn_end_channels}_layers{mtgnn_layers}_dropout{mtgnn_dropout:g}_lr{learning_rate:g}"
                    f"_anchor{residual_anchor_steps if spec.output_parameterization == 'residual' else 0}"
                )
                model, train_summary = _train_mtgnn(
                    prepared,
                    variant_name=spec.model_variant,
                    validation_windows=checkpoint_windows,
                    residual_anchor_steps=residual_anchor_steps,
                    seed=seed,
                    device=device,
                    batch_size=train_batch_size,
                    learning_rate=learning_rate,
                    max_epochs=max_epochs,
                    gcn_depth=mtgnn_gcn_depth,
                    subgraph_size=mtgnn_subgraph_size,
                    node_dim=mtgnn_node_dim,
                    residual_channels=mtgnn_residual_channels,
                    skip_channels=mtgnn_skip_channels,
                    end_channels=mtgnn_end_channels,
                    layers=mtgnn_layers,
                    dropout=mtgnn_dropout,
                )
                predictions_by_split = {
                    (split_name, eval_protocol): _evaluate_mtgnn(
                        model,
                        prepared,
                        windows,
                        variant_name=spec.model_variant,
                        device=train_summary["device"],
                        batch_size=train_batch_size,
                        residual_anchor_steps=residual_anchor_steps,
                    )
                    for split_name, eval_protocol, windows in window_specs
                }
                gate_b_passed, gate_c_passed, gate_b_metrics, _gate_c_metrics = _gate_status_for_neural_model(
                    evaluator=_evaluate_mtgnn,
                    model=model,
                    prepared=prepared,
                    variant_name=spec.model_variant,
                    device=train_summary["device"],
                    batch_size=train_batch_size,
                    residual_anchor_steps=residual_anchor_steps,
                    train_gate_windows=train_gate_windows,
                    gate_c_windows=gate_c_windows,
                    persistence_train_rmse=float(persistence_train_gate_metrics["rmse_pu"]),
                    persistence_gate_c_lead1_rmse=float(persistence_gate_c_metrics["lead1_rmse_pu"]),
                    persistence_gate_c_lead1_mae=float(persistence_gate_c_metrics["lead1_mae_pu"]),
                )
                gate_b_for_row = (
                    gate_b_overfit64_passed if gate_b_overfit64_passed is not None else gate_b_passed
                )
                gate_b_scope = (
                    "overfit64_preflight"
                    if gate_b_overfit64_passed is not None
                    else "train_gate_after_fit"
                )
                rows.extend(
                    _metric_rows(
                        spec,
                        prepared=prepared,
                        seed=seed,
                        trial_id=f"{spec.model_variant}_{mtgnn_search_config_id}",
                        search_config_id=mtgnn_search_config_id,
                        alpha=None,
                        predictions_by_split=predictions_by_split,
                        runtime_seconds=time.perf_counter() - started,
                        gate_b_passed=gate_b_for_row,
                        gate_c_passed=gate_c_passed,
                        residual_anchor_steps=residual_anchor_steps if spec.output_parameterization == "residual" else 0,
                        best_trial=True,
                        window_specs=window_specs,
                        gate_b_scope=gate_b_scope,
                        gate_b_overfit64_passed=gate_b_overfit64_passed,
                        train_gate_after_fit_passed=gate_b_passed,
                        train_gate_after_fit_rmse_pu=float(gate_b_metrics["rmse_pu"]),
                        train_gate_after_fit_mae_pu=float(gate_b_metrics["mae_pu"]),
                    )
                )
            elif spec.model_variant in {TFT_DIRECT_VARIANT, TFT_RESIDUAL_VARIANT}:
                tft_search_config_id = (
                    f"tft_pf_h{tft_hidden_size}_lstm{tft_lstm_layers}_heads{tft_attention_head_size}"
                    f"_hc{tft_hidden_continuous_size}_dropout{tft_dropout}_lr{learning_rate}"
                    f"_anchor{residual_anchor_steps if spec.output_parameterization == 'residual' else 0}"
                )
                model, training_dataset, train_summary = _train_tft(
                    prepared,
                    variant_name=spec.model_variant,
                    validation_windows=checkpoint_windows,
                    residual_anchor_steps=residual_anchor_steps,
                    seed=seed,
                    device=device,
                    batch_size=train_batch_size,
                    learning_rate=learning_rate,
                    max_epochs=max_epochs,
                    hidden_size=tft_hidden_size,
                    lstm_layers=tft_lstm_layers,
                    attention_head_size=tft_attention_head_size,
                    hidden_continuous_size=tft_hidden_continuous_size,
                    dropout=tft_dropout,
                )
                predictions_by_split = {
                    (split_name, eval_protocol): _evaluate_tft(
                        model,
                        training_dataset,
                        prepared,
                        windows,
                        variant_name=spec.model_variant,
                        device=train_summary["device"],
                        batch_size=train_batch_size,
                        residual_anchor_steps=residual_anchor_steps,
                    )
                    for split_name, eval_protocol, windows in window_specs
                }
                gate_b_passed, gate_c_passed, gate_b_metrics, _gate_c_metrics = _gate_status_for_neural_model(
                    evaluator=lambda model, prepared, windows, **kwargs: _evaluate_tft(
                        model,
                        training_dataset,
                        prepared,
                        windows,
                        eval_window_chunk_size=tft_eval_window_chunk_size,
                        **kwargs,
                    ),
                    model=model,
                    prepared=prepared,
                    variant_name=spec.model_variant,
                    device=train_summary["device"],
                    batch_size=train_batch_size,
                    residual_anchor_steps=residual_anchor_steps,
                    train_gate_windows=train_gate_windows,
                    gate_c_windows=gate_c_windows,
                    persistence_train_rmse=float(persistence_train_gate_metrics["rmse_pu"]),
                    persistence_gate_c_lead1_rmse=float(persistence_gate_c_metrics["lead1_rmse_pu"]),
                    persistence_gate_c_lead1_mae=float(persistence_gate_c_metrics["lead1_mae_pu"]),
                )
                gate_b_for_row = (
                    gate_b_overfit64_passed if gate_b_overfit64_passed is not None else gate_b_passed
                )
                gate_b_scope = (
                    "overfit64_preflight"
                    if gate_b_overfit64_passed is not None
                    else "train_gate_after_fit"
                )
                rows.extend(
                    _metric_rows(
                        spec,
                        prepared=prepared,
                        seed=seed,
                        trial_id=f"{spec.model_variant}_{tft_search_config_id}",
                        search_config_id=tft_search_config_id,
                        alpha=None,
                        predictions_by_split=predictions_by_split,
                        runtime_seconds=time.perf_counter() - started,
                        gate_b_passed=gate_b_for_row,
                        gate_c_passed=gate_c_passed,
                        residual_anchor_steps=residual_anchor_steps if spec.output_parameterization == "residual" else 0,
                        best_trial=True,
                        window_specs=window_specs,
                        gate_b_scope=gate_b_scope,
                        gate_b_overfit64_passed=gate_b_overfit64_passed,
                        train_gate_after_fit_passed=gate_b_passed,
                        train_gate_after_fit_rmse_pu=float(gate_b_metrics["rmse_pu"]),
                        train_gate_after_fit_mae_pu=float(gate_b_metrics["mae_pu"]),
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
        "residual_anchor_steps": residual_anchor_steps,
        "dgcrn_hidden_dim": dgcrn_hidden_dim,
        "dgcrn_dropout": dgcrn_dropout,
        "dgcrn_gcn_depth": dgcrn_gcn_depth,
        "itransformer_d_model": itransformer_d_model,
        "itransformer_n_heads": itransformer_n_heads,
        "itransformer_e_layers": itransformer_e_layers,
        "itransformer_dropout": itransformer_dropout,
        "tft_hidden_size": tft_hidden_size,
        "tft_lstm_layers": tft_lstm_layers,
        "tft_attention_head_size": tft_attention_head_size,
        "tft_hidden_continuous_size": tft_hidden_continuous_size,
        "tft_dropout": tft_dropout,
        "tft_eval_window_chunk_size": tft_eval_window_chunk_size,
        "mtgnn_gcn_depth": mtgnn_gcn_depth,
        "mtgnn_subgraph_size": mtgnn_subgraph_size,
        "mtgnn_node_dim": mtgnn_node_dim,
        "mtgnn_residual_channels": mtgnn_residual_channels,
        "mtgnn_skip_channels": mtgnn_skip_channels,
        "mtgnn_end_channels": mtgnn_end_channels,
        "mtgnn_layers": mtgnn_layers,
        "mtgnn_dropout": mtgnn_dropout,
        "timexer_d_model": timexer_d_model,
        "timexer_n_heads": timexer_n_heads,
        "timexer_e_layers": timexer_e_layers,
        "timexer_dropout": timexer_dropout,
        "timexer_patch_len": timexer_patch_len,
        "gate_b_overfit64_passed": gate_b_overfit64_passed,
        "gate_b_overfit64_rmse_pu": gate_b_overfit64_rmse_pu,
        "gate_b_overfit64_mae_pu": gate_b_overfit64_mae_pu,
        "gate_b_overfit64_source": gate_b_overfit64_source,
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
                "residual_anchor_steps": residual_anchor_steps,
                "dgcrn_hidden_dim": dgcrn_hidden_dim,
                "dgcrn_dropout": dgcrn_dropout,
                "dgcrn_gcn_depth": dgcrn_gcn_depth,
                "itransformer_d_model": itransformer_d_model,
                "itransformer_n_heads": itransformer_n_heads,
                "itransformer_e_layers": itransformer_e_layers,
                "itransformer_dropout": itransformer_dropout,
                "tft_hidden_size": tft_hidden_size,
                "tft_lstm_layers": tft_lstm_layers,
                "tft_attention_head_size": tft_attention_head_size,
                "tft_hidden_continuous_size": tft_hidden_continuous_size,
                "tft_dropout": tft_dropout,
                "tft_eval_window_chunk_size": tft_eval_window_chunk_size,
                "mtgnn_gcn_depth": mtgnn_gcn_depth,
                "mtgnn_subgraph_size": mtgnn_subgraph_size,
                "mtgnn_node_dim": mtgnn_node_dim,
                "mtgnn_residual_channels": mtgnn_residual_channels,
                "mtgnn_skip_channels": mtgnn_skip_channels,
                "mtgnn_end_channels": mtgnn_end_channels,
                "mtgnn_layers": mtgnn_layers,
                "mtgnn_dropout": mtgnn_dropout,
                "timexer_d_model": timexer_d_model,
                "timexer_n_heads": timexer_n_heads,
                "timexer_e_layers": timexer_e_layers,
                "timexer_dropout": timexer_dropout,
                "timexer_patch_len": timexer_patch_len,
                "gate_b_overfit64_passed": gate_b_overfit64_passed,
                "gate_b_overfit64_rmse_pu": gate_b_overfit64_rmse_pu,
                "gate_b_overfit64_mae_pu": gate_b_overfit64_mae_pu,
                "gate_b_overfit64_source": gate_b_overfit64_source,
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
    parser.add_argument("--residual-anchor-steps", type=int, default=0)
    parser.add_argument("--dgcrn-hidden-dim", type=int, default=64)
    parser.add_argument("--dgcrn-dropout", type=float, default=0.1)
    parser.add_argument("--dgcrn-gcn-depth", type=int, default=2)
    parser.add_argument("--itransformer-d-model", type=int, default=64)
    parser.add_argument("--itransformer-n-heads", type=int, default=4)
    parser.add_argument("--itransformer-e-layers", type=int, default=2)
    parser.add_argument("--itransformer-dropout", type=float, default=0.1)
    parser.add_argument("--tft-hidden-size", type=int, default=32)
    parser.add_argument("--tft-lstm-layers", type=int, default=1)
    parser.add_argument("--tft-attention-head-size", type=int, default=4)
    parser.add_argument("--tft-hidden-continuous-size", type=int, default=16)
    parser.add_argument("--tft-dropout", type=float, default=0.1)
    parser.add_argument("--tft-eval-window-chunk-size", type=int, default=DEFAULT_TFT_EVAL_WINDOW_CHUNK_SIZE)
    parser.add_argument("--mtgnn-gcn-depth", type=int, default=2)
    parser.add_argument("--mtgnn-subgraph-size", type=int, default=6)
    parser.add_argument("--mtgnn-node-dim", type=int, default=40)
    parser.add_argument("--mtgnn-residual-channels", type=int, default=32)
    parser.add_argument("--mtgnn-skip-channels", type=int, default=64)
    parser.add_argument("--mtgnn-end-channels", type=int, default=128)
    parser.add_argument("--mtgnn-layers", type=int, default=3)
    parser.add_argument("--mtgnn-dropout", type=float, default=0.3)
    parser.add_argument("--timexer-d-model", type=int, default=64)
    parser.add_argument("--timexer-n-heads", type=int, default=4)
    parser.add_argument("--timexer-e-layers", type=int, default=2)
    parser.add_argument("--timexer-dropout", type=float, default=0.1)
    parser.add_argument("--timexer-patch-len", type=int, default=16)
    parser.add_argument("--gate-b-overfit64-passed", action="store_true", default=None)
    parser.add_argument("--gate-b-overfit64-rmse-pu", type=float, default=None)
    parser.add_argument("--gate-b-overfit64-mae-pu", type=float, default=None)
    parser.add_argument("--gate-b-overfit64-source", type=str, default=None)
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
        residual_anchor_steps=args.residual_anchor_steps,
        dgcrn_hidden_dim=args.dgcrn_hidden_dim,
        dgcrn_dropout=args.dgcrn_dropout,
        dgcrn_gcn_depth=args.dgcrn_gcn_depth,
        itransformer_d_model=args.itransformer_d_model,
        itransformer_n_heads=args.itransformer_n_heads,
        itransformer_e_layers=args.itransformer_e_layers,
        itransformer_dropout=args.itransformer_dropout,
        tft_hidden_size=args.tft_hidden_size,
        tft_lstm_layers=args.tft_lstm_layers,
        tft_attention_head_size=args.tft_attention_head_size,
        tft_hidden_continuous_size=args.tft_hidden_continuous_size,
        tft_dropout=args.tft_dropout,
        tft_eval_window_chunk_size=args.tft_eval_window_chunk_size,
        mtgnn_gcn_depth=args.mtgnn_gcn_depth,
        mtgnn_subgraph_size=args.mtgnn_subgraph_size,
        mtgnn_node_dim=args.mtgnn_node_dim,
        mtgnn_residual_channels=args.mtgnn_residual_channels,
        mtgnn_skip_channels=args.mtgnn_skip_channels,
        mtgnn_end_channels=args.mtgnn_end_channels,
        mtgnn_layers=args.mtgnn_layers,
        mtgnn_dropout=args.mtgnn_dropout,
        timexer_d_model=args.timexer_d_model,
        timexer_n_heads=args.timexer_n_heads,
        timexer_e_layers=args.timexer_e_layers,
        timexer_dropout=args.timexer_dropout,
        timexer_patch_len=args.timexer_patch_len,
        gate_b_overfit64_passed=args.gate_b_overfit64_passed,
        gate_b_overfit64_rmse_pu=args.gate_b_overfit64_rmse_pu,
        gate_b_overfit64_mae_pu=args.gate_b_overfit64_mae_pu,
        gate_b_overfit64_source=args.gate_b_overfit64_source,
        run_label=args.run_label,
        no_record_run=args.no_record_run,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
