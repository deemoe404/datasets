from __future__ import annotations

import argparse
from datetime import UTC, datetime
import hashlib
import json
import math
from pathlib import Path
import sys
import time
from typing import Any, Sequence

import numpy as np
import polars as pl

DIAGNOSTICS_DIR = Path(__file__).resolve().parent
FAMILY_DIR = DIAGNOSTICS_DIR.parent
while str(DIAGNOSTICS_DIR) in sys.path:
    sys.path.remove(str(DIAGNOSTICS_DIR))
if str(FAMILY_DIR) not in sys.path:
    sys.path.insert(0, str(FAMILY_DIR))

import formal_tuning as formal  # noqa: E402

ROLLING_PROTOCOL = formal.ROLLING_EVAL_PROTOCOL
DEFAULT_EXACT_AE_LIMIT = 32_000_000
SHARD_SCHEMA_VERSION = 1


def _gate_e_manifest_fields(*, split_name: str, created_at: str | None) -> dict[str, Any]:
    return {
        "selection_metric": "val_overall_rmse",
        "selected_by": "validation_only",
        "no_test_feedback": True,
        "test_evaluated_at": created_at if split_name == "test" else None,
    }


def _json_default(value: Any) -> Any:
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable.")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, default=_json_default, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _safe_protocol_name(eval_protocol: str) -> str:
    return eval_protocol.replace("/", "_").replace(" ", "_")


def _artifact_stem(dataset_id: str, split_name: str, eval_protocol: str, variant_name: str) -> str:
    variant_token = "residual" if variant_name == formal.TFT_RESIDUAL_VARIANT else "direct"
    return f"tft_pf_{variant_token}_{dataset_id}_{split_name}_{_safe_protocol_name(eval_protocol)}"


def shard_artifact_paths(
    output_dir: Path,
    *,
    dataset_id: str,
    split_name: str,
    eval_protocol: str,
    variant_name: str,
    shard_start: int,
    shard_stop: int,
) -> dict[str, Path]:
    stem = (
        f"{_artifact_stem(dataset_id, split_name, eval_protocol, variant_name)}"
        f"_shard_{shard_start:06d}_{shard_stop:06d}"
    )
    return {
        "json": output_dir / f"{stem}.json",
        "csv": output_dir / f"{stem}.csv",
        "abs_errors": output_dir / f"{stem}.abs_errors.npy",
        "origin_errors": output_dir / f"{stem}.origin_errors.csv",
    }


def aggregate_artifact_paths(
    output_dir: Path,
    *,
    dataset_id: str,
    split_name: str,
    eval_protocol: str,
    variant_name: str,
) -> dict[str, Path]:
    stem = f"{_artifact_stem(dataset_id, split_name, eval_protocol, variant_name)}_aggregate"
    return {
        "json": output_dir / f"{stem}.json",
        "csv": output_dir / f"{stem}.csv",
    }


def _windows_for_split(prepared: Any, *, split_name: str, eval_protocol: str) -> Any:
    if split_name == "val" and eval_protocol == formal.ROLLING_EVAL_PROTOCOL:
        return prepared.val_rolling_windows
    if split_name == "val" and eval_protocol == formal.NON_OVERLAP_EVAL_PROTOCOL:
        return prepared.val_non_overlap_windows
    if split_name == "test" and eval_protocol == formal.ROLLING_EVAL_PROTOCOL:
        return prepared.test_rolling_windows
    if split_name == "test" and eval_protocol == formal.NON_OVERLAP_EVAL_PROTOCOL:
        return prepared.test_non_overlap_windows
    raise ValueError(f"Unsupported split/protocol pair: {split_name!r}, {eval_protocol!r}.")


def resolve_shard_bounds(
    *,
    total_window_count: int,
    shard_size: int | None,
    shard_index: int | None,
    start: int | None,
    stop: int | None,
) -> tuple[int, int]:
    if start is not None or stop is not None:
        if shard_size is not None or shard_index is not None:
            raise ValueError("Use either --start/--stop or --shard-size/--shard-index, not both.")
        if start is None or stop is None:
            raise ValueError("--start and --stop must be provided together.")
        shard_start = int(start)
        shard_stop = int(stop)
    else:
        if shard_size is None or shard_index is None:
            raise ValueError("Provide --start/--stop or --shard-size/--shard-index.")
        if shard_size <= 0:
            raise ValueError("--shard-size must be positive.")
        if shard_index < 0:
            raise ValueError("--shard-index must be non-negative.")
        shard_start = int(shard_index) * int(shard_size)
        shard_stop = min(shard_start + int(shard_size), int(total_window_count))
    if shard_start < 0 or shard_stop <= shard_start or shard_stop > total_window_count:
        raise ValueError(
            f"Invalid shard bounds [{shard_start}, {shard_stop}) for total_window_count={total_window_count}."
        )
    return shard_start, shard_stop


def tft_frozen_config(
    *,
    variant_name: str,
    seed: int,
    max_train_origins: int | None,
    max_checkpoint_origins: int | None,
    checkpoint_eval_protocol: str,
    residual_anchor_steps: int,
    train_batch_size: int,
    max_epochs: int,
    learning_rate: float,
    hidden_size: int,
    lstm_layers: int,
    attention_head_size: int,
    hidden_continuous_size: int,
    dropout: float,
    eval_window_chunk_size: int,
) -> dict[str, Any]:
    residual_steps = residual_anchor_steps if variant_name == formal.TFT_RESIDUAL_VARIANT else 0
    return {
        "variant_name": variant_name,
        "seed": int(seed),
        "max_train_origins": max_train_origins,
        "max_checkpoint_origins": max_checkpoint_origins,
        "checkpoint_eval_protocol": checkpoint_eval_protocol,
        "residual_anchor_steps": int(residual_steps),
        "train_batch_size": int(train_batch_size),
        "max_epochs": int(max_epochs),
        "learning_rate": float(learning_rate),
        "hidden_size": int(hidden_size),
        "lstm_layers": int(lstm_layers),
        "attention_head_size": int(attention_head_size),
        "hidden_continuous_size": int(hidden_continuous_size),
        "dropout": float(dropout),
        "eval_window_chunk_size": int(eval_window_chunk_size),
        "train_or_load_strategy": "train_each_shard_process",
    }


def _config_hash(config: dict[str, Any]) -> str:
    payload = json.dumps(config, sort_keys=True, separators=(",", ":"), default=_json_default)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _tft_search_config_id(config: dict[str, Any]) -> str:
    return (
        f"tft_pf_h{config['hidden_size']}_lstm{config['lstm_layers']}_heads{config['attention_head_size']}"
        f"_hc{config['hidden_continuous_size']}_dropout{config['dropout']}_lr{config['learning_rate']}"
        f"_anchor{config['residual_anchor_steps']}"
    )


def _empty_components(forecast_steps: int) -> dict[str, Any]:
    return {
        "window_count": 0,
        "prediction_count": 0,
        "abs_error_sum": 0.0,
        "sq_error_sum": 0.0,
        "lead_valid": np.zeros((forecast_steps,), dtype=np.float64),
        "lead_abs": np.zeros((forecast_steps,), dtype=np.float64),
        "lead_sq": np.zeros((forecast_steps,), dtype=np.float64),
    }


def _abs_errors(predictions: np.ndarray, targets: np.ndarray, valid: np.ndarray) -> np.ndarray:
    valid_f = valid.astype(np.float64, copy=False)
    errors = (predictions.astype(np.float64, copy=False) - targets.astype(np.float64, copy=False)) * valid_f
    return np.abs(errors[valid_f > 0]).astype(np.float64, copy=False)


def _metric_components(predictions: np.ndarray, targets: np.ndarray, valid: np.ndarray, *, forecast_steps: int) -> dict[str, Any]:
    if predictions.shape != targets.shape or predictions.shape != valid.shape:
        raise ValueError(
            f"Shape mismatch: predictions={predictions.shape!r}, targets={targets.shape!r}, valid={valid.shape!r}."
        )
    if predictions.ndim != 3:
        raise ValueError(f"Expected [window, lead, node] arrays, found {predictions.shape!r}.")
    if predictions.shape[1] != forecast_steps:
        raise ValueError(f"Expected {forecast_steps} lead steps, found {predictions.shape[1]}.")
    valid_f = valid.astype(np.float64, copy=False)
    errors = (predictions.astype(np.float64, copy=False) - targets.astype(np.float64, copy=False)) * valid_f
    abs_masked = np.abs(errors)
    sq_masked = np.square(errors)
    return {
        "window_count": int(predictions.shape[0]),
        "prediction_count": int(valid_f.sum()),
        "abs_error_sum": float(abs_masked.sum()),
        "sq_error_sum": float(sq_masked.sum()),
        "lead_valid": valid_f.sum(axis=(0, 2)).astype(np.float64),
        "lead_abs": abs_masked.sum(axis=(0, 2)).astype(np.float64),
        "lead_sq": sq_masked.sum(axis=(0, 2)).astype(np.float64),
    }


def _merge_components(total: dict[str, Any], part: dict[str, Any]) -> None:
    total["window_count"] += int(part["window_count"])
    total["prediction_count"] += int(part["prediction_count"])
    total["abs_error_sum"] += float(part["abs_error_sum"])
    total["sq_error_sum"] += float(part["sq_error_sum"])
    total["lead_valid"] += np.asarray(part["lead_valid"], dtype=np.float64)
    total["lead_abs"] += np.asarray(part["lead_abs"], dtype=np.float64)
    total["lead_sq"] += np.asarray(part["lead_sq"], dtype=np.float64)


def _metrics_from_components(
    components: dict[str, Any],
    *,
    rated_power_kw: float,
    exact_abs_errors: np.ndarray | None,
    exact_abs_error_limit: int | None,
    metrics_backend: str,
) -> dict[str, Any]:
    prediction_count = int(components["prediction_count"])
    if prediction_count <= 0:
        mae_pu = rmse_pu = mae_kw = rmse_kw = math.nan
    else:
        mae_pu = float(components["abs_error_sum"]) / prediction_count
        rmse_pu = math.sqrt(float(components["sq_error_sum"]) / prediction_count)
        mae_kw = mae_pu * float(rated_power_kw)
        rmse_kw = rmse_pu * float(rated_power_kw)
    lead_valid = np.asarray(components["lead_valid"], dtype=np.float64)
    lead_abs = np.asarray(components["lead_abs"], dtype=np.float64)
    lead_sq = np.asarray(components["lead_sq"], dtype=np.float64)
    lead_mae = np.divide(
        lead_abs,
        lead_valid,
        out=np.full_like(lead_abs, np.nan, dtype=np.float64),
        where=lead_valid > 0,
    )
    lead_rmse = np.sqrt(
        np.divide(
            lead_sq,
            lead_valid,
            out=np.full_like(lead_sq, np.nan, dtype=np.float64),
            where=lead_valid > 0,
        )
    )
    if exact_abs_errors is None:
        ae_status = "exact_unavailable"
        ae_p50 = ae_p90 = ae_p95 = math.nan
        ae_count: int | None = None
    else:
        if exact_abs_error_limit is not None and exact_abs_errors.size > exact_abs_error_limit:
            raise RuntimeError(
                f"Exact AE count {exact_abs_errors.size} exceeds exact_abs_error_limit={exact_abs_error_limit}."
            )
        if exact_abs_errors.size != prediction_count:
            raise RuntimeError(
                f"Exact AE count {exact_abs_errors.size} does not match prediction_count={prediction_count}."
            )
        ae_status = "exact"
        ae_p50 = float(np.quantile(exact_abs_errors, 0.50)) if exact_abs_errors.size else math.nan
        ae_p90 = float(np.quantile(exact_abs_errors, 0.90)) if exact_abs_errors.size else math.nan
        ae_p95 = float(np.quantile(exact_abs_errors, 0.95)) if exact_abs_errors.size else math.nan
        ae_count = int(exact_abs_errors.size)
    return {
        "window_count": int(components["window_count"]),
        "prediction_count": prediction_count,
        "mae_pu": float(mae_pu),
        "rmse_pu": float(rmse_pu),
        "mae_kw": float(mae_kw),
        "rmse_kw": float(rmse_kw),
        "lead1_mae_pu": float(lead_mae[0]) if lead_mae.size else math.nan,
        "lead1_rmse_pu": float(lead_rmse[0]) if lead_rmse.size else math.nan,
        "short_rmse_pu": formal._lead_bucket_rmse(lead_sq, lead_valid, 1, 6),
        "mid_rmse_pu": formal._lead_bucket_rmse(lead_sq, lead_valid, 7, 18),
        "long_rmse_pu": formal._lead_bucket_rmse(lead_sq, lead_valid, 19, 36),
        "ae_p50": ae_p50,
        "ae_p90": ae_p90,
        "ae_p95": ae_p95,
        "metrics_backend": metrics_backend,
        "ae_quantile_status": ae_status,
        "ae_quantile_exact_count": ae_count,
        "ae_quantile_exact_limit": exact_abs_error_limit,
    }


def _per_origin_error_rows(
    predictions: np.ndarray,
    targets: np.ndarray,
    valid: np.ndarray,
    windows: Any,
    *,
    shard_start: int,
) -> list[dict[str, Any]]:
    abs_masked = np.abs((predictions.astype(np.float64) - targets.astype(np.float64)) * valid.astype(np.float64))
    valid_counts = valid.astype(np.float64).sum(axis=(1, 2))
    origin_abs = np.divide(
        abs_masked.sum(axis=(1, 2)),
        valid_counts,
        out=np.full((predictions.shape[0],), np.nan, dtype=np.float64),
        where=valid_counts > 0,
    )
    rows: list[dict[str, Any]] = []
    for local_index, value in enumerate(origin_abs):
        rows.append(
            {
                "origin_index": int(shard_start) + int(local_index),
                "target_index": int(windows.target_indices[local_index]),
                "output_start_us": int(windows.output_start_us[local_index]),
                "output_end_us": int(windows.output_end_us[local_index]),
                "origin_prediction_count": int(valid_counts[local_index]),
                "proposed_abs_error_pu": float(value),
            }
        )
    return rows


def build_shard_record_from_arrays(
    predictions: np.ndarray,
    targets: np.ndarray,
    valid: np.ndarray,
    *,
    dataset_id: str,
    split_name: str,
    eval_protocol: str,
    variant_name: str,
    frozen_config: dict[str, Any],
    shard_start: int,
    shard_stop: int,
    total_window_count: int,
    rated_power_kw: float,
    forecast_steps: int,
    node_count: int,
    exact_abs_error_limit: int | None,
    target_index_start: int | None = None,
    target_index_stop_exclusive: int | None = None,
    output_start_us: int | None = None,
    output_end_us: int | None = None,
    train_summary: dict[str, Any] | None = None,
    runtime_seconds: float | None = None,
) -> tuple[dict[str, Any], np.ndarray]:
    components = _metric_components(predictions, targets, valid, forecast_steps=forecast_steps)
    exact_abs_errors = _abs_errors(predictions, targets, valid)
    if exact_abs_error_limit is not None and exact_abs_errors.size > exact_abs_error_limit:
        raise RuntimeError(
            f"Shard exact AE count {exact_abs_errors.size} exceeds exact_abs_error_limit={exact_abs_error_limit}."
        )
    config_hash = _config_hash(frozen_config)
    metrics = _metrics_from_components(
        components,
        rated_power_kw=rated_power_kw,
        exact_abs_errors=exact_abs_errors,
        exact_abs_error_limit=exact_abs_error_limit,
        metrics_backend="tft_pf_shard_streaming",
    )
    search_config_id = _tft_search_config_id(frozen_config)
    created_at = datetime.now(tz=UTC).isoformat()
    record = {
        "schema_version": SHARD_SCHEMA_VERSION,
        "status": "complete",
        "created_at": created_at,
        "dataset_id": dataset_id,
        "split_name": split_name,
        "eval_protocol": eval_protocol,
        "model_variant": variant_name,
        "trial_id": f"{variant_name}_{search_config_id}",
        "formal_search_config_id": search_config_id,
        "frozen_config": frozen_config,
        "frozen_config_hash": config_hash,
        **_gate_e_manifest_fields(split_name=split_name, created_at=created_at),
        "shard_start": int(shard_start),
        "shard_stop": int(shard_stop),
        "shard_window_count": int(shard_stop) - int(shard_start),
        "total_window_count": int(total_window_count),
        "target_index_start": target_index_start,
        "target_index_stop_exclusive": target_index_stop_exclusive,
        "output_start_us": output_start_us,
        "output_end_us": output_end_us,
        "forecast_steps": int(forecast_steps),
        "node_count": int(node_count),
        "rated_power_kw": float(rated_power_kw),
        "exact_abs_error_limit": exact_abs_error_limit,
        "train_summary": train_summary or {},
        "runtime_seconds": runtime_seconds,
        "tft_contract": {
            "uses_future_target": False,
            "target_scope": "history_context_only",
            "future_covariates_scope": "calendar_only",
            "feature_budget_id": "B2",
        },
        "metrics": metrics,
        "components": {
            "window_count": int(components["window_count"]),
            "prediction_count": int(components["prediction_count"]),
            "abs_error_sum": float(components["abs_error_sum"]),
            "sq_error_sum": float(components["sq_error_sum"]),
            "lead_valid": np.asarray(components["lead_valid"], dtype=np.float64).tolist(),
            "lead_abs": np.asarray(components["lead_abs"], dtype=np.float64).tolist(),
            "lead_sq": np.asarray(components["lead_sq"], dtype=np.float64).tolist(),
        },
    }
    return record, exact_abs_errors


def _metric_identity_fields(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "dataset_id": record["dataset_id"],
        "model_variant": record["model_variant"],
        "split_name": record["split_name"],
        "eval_protocol": record["eval_protocol"],
        "shard_start": record.get("shard_start"),
        "shard_stop": record.get("shard_stop"),
        "total_window_count": record.get("total_window_count"),
        "trial_id": record.get("trial_id"),
        "formal_search_config_id": record.get("formal_search_config_id"),
        "frozen_config_hash": record.get("frozen_config_hash"),
        "selection_metric": record.get("selection_metric", "val_overall_rmse"),
        "selected_by": record.get("selected_by", "validation_only"),
        "no_test_feedback": record.get("no_test_feedback", True),
        "test_evaluated_at": record.get("test_evaluated_at"),
    }


def write_shard_artifacts(
    record: dict[str, Any],
    exact_abs_errors: np.ndarray,
    origin_error_rows: Sequence[dict[str, Any]],
    output_dir: Path,
) -> dict[str, Path]:
    paths = shard_artifact_paths(
        output_dir,
        dataset_id=str(record["dataset_id"]),
        split_name=str(record["split_name"]),
        eval_protocol=str(record["eval_protocol"]),
        variant_name=str(record["model_variant"]),
        shard_start=int(record["shard_start"]),
        shard_stop=int(record["shard_stop"]),
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(paths["abs_errors"], exact_abs_errors.astype(np.float64, copy=False))
    if origin_error_rows:
        pl.DataFrame(origin_error_rows).write_csv(paths["origin_errors"])
    else:
        paths["origin_errors"].write_text(
            "origin_index,target_index,output_start_us,output_end_us,origin_prediction_count,proposed_abs_error_pu\n",
            encoding="utf-8",
        )
    json_record = dict(record)
    json_record["abs_error_path"] = paths["abs_errors"].name
    json_record["origin_error_path"] = paths["origin_errors"].name
    _write_json(paths["json"], json_record)
    pl.DataFrame([{**_metric_identity_fields(json_record), **json_record["metrics"]}]).write_csv(paths["csv"])
    return paths


def _sum_components(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    forecast_steps = int(records[0]["forecast_steps"])
    components = _empty_components(forecast_steps)
    for record in records:
        _merge_components(components, record["components"])
    return components


def _validate_shard_coverage(
    records: Sequence[dict[str, Any]],
    *,
    dataset_id: str,
    split_name: str,
    eval_protocol: str,
    variant_name: str,
) -> None:
    if not records:
        raise RuntimeError("No completed TFT-PF shard records found.")
    first = records[0]
    total_window_count = int(first["total_window_count"])
    forecast_steps = int(first["forecast_steps"])
    node_count = int(first["node_count"])
    rated_power_kw = float(first["rated_power_kw"])
    config_hash = str(first["frozen_config_hash"])
    frozen_config = first.get("frozen_config")
    formal_search_config_id = first.get("formal_search_config_id")
    trial_id = first.get("trial_id")
    expected_start = 0
    for record in records:
        if int(record.get("schema_version", -1)) != SHARD_SCHEMA_VERSION:
            raise RuntimeError(f"Unsupported shard schema version in {record!r}.")
        if record.get("status") != "complete":
            raise RuntimeError(f"Shard [{record.get('shard_start')}, {record.get('shard_stop')}) is not complete.")
        if (
            record["dataset_id"] != dataset_id
            or record["split_name"] != split_name
            or record["eval_protocol"] != eval_protocol
            or record["model_variant"] != variant_name
        ):
            raise RuntimeError("Shard identity does not match requested aggregate.")
        if int(record["total_window_count"]) != total_window_count:
            raise RuntimeError("Shard total_window_count mismatch.")
        if int(record["forecast_steps"]) != forecast_steps or int(record["node_count"]) != node_count:
            raise RuntimeError("Shard shape metadata mismatch.")
        if float(record["rated_power_kw"]) != rated_power_kw:
            raise RuntimeError("Shard rated_power_kw mismatch.")
        if str(record.get("frozen_config_hash")) != config_hash or record.get("frozen_config") != frozen_config:
            raise RuntimeError("Shard frozen TFT config identity mismatch.")
        if record.get("formal_search_config_id") != formal_search_config_id or record.get("trial_id") != trial_id:
            raise RuntimeError("Shard trial/config labels mismatch.")
        if record.get("tft_contract", {}).get("uses_future_target") is not False:
            raise RuntimeError("Shard contract does not prove uses_future_target=false.")
        if record.get("metrics", {}).get("ae_quantile_status") != "exact":
            raise RuntimeError("Shard exact AE quantiles are not exact.")
        shard_start = int(record["shard_start"])
        shard_stop = int(record["shard_stop"])
        if shard_start != expected_start:
            raise RuntimeError(f"Missing or overlapping shard coverage: expected start {expected_start}, found {shard_start}.")
        expected_start = shard_stop
    if expected_start != total_window_count:
        raise RuntimeError(f"Missing final shard coverage: covered {expected_start}, expected {total_window_count}.")


def aggregate_shard_records(
    records: Sequence[dict[str, Any]],
    *,
    dataset_id: str,
    split_name: str,
    eval_protocol: str,
    variant_name: str,
    exact_abs_error_limit: int | None,
    base_dir: Path | None = None,
) -> dict[str, Any]:
    ordered = sorted(records, key=lambda record: int(record["shard_start"]))
    _validate_shard_coverage(
        ordered,
        dataset_id=dataset_id,
        split_name=split_name,
        eval_protocol=eval_protocol,
        variant_name=variant_name,
    )
    components = _sum_components(ordered)
    exact_chunks: list[np.ndarray] = []
    for record in ordered:
        inline = record.get("_exact_abs_errors")
        if inline is not None:
            chunk = np.asarray(inline, dtype=np.float64)
        else:
            abs_error_path = record.get("abs_error_path")
            if not abs_error_path:
                raise RuntimeError(
                    f"Shard [{record['shard_start']}, {record['shard_stop']}) is missing abs_error_path."
                )
            if base_dir is None:
                raise RuntimeError("base_dir is required when aggregating shards from abs_error_path.")
            chunk_path = Path(abs_error_path)
            if not chunk_path.is_absolute():
                chunk_path = base_dir / chunk_path
            if not chunk_path.exists():
                raise RuntimeError(f"Missing exact AE artifact: {chunk_path}.")
            chunk = np.load(chunk_path)
        exact_chunks.append(np.asarray(chunk, dtype=np.float64))
    exact_abs_errors = np.concatenate(exact_chunks) if exact_chunks else np.zeros((0,), dtype=np.float64)
    first = ordered[0]
    metrics = _metrics_from_components(
        components,
        rated_power_kw=float(first["rated_power_kw"]),
        exact_abs_errors=exact_abs_errors,
        exact_abs_error_limit=exact_abs_error_limit,
        metrics_backend="tft_pf_shard_aggregate",
    )
    created_at = datetime.now(tz=UTC).isoformat()
    return {
        "schema_version": SHARD_SCHEMA_VERSION,
        "status": "complete",
        "created_at": created_at,
        "dataset_id": dataset_id,
        "split_name": split_name,
        "eval_protocol": eval_protocol,
        "model_variant": variant_name,
        "trial_id": first["trial_id"],
        "formal_search_config_id": first["formal_search_config_id"],
        "frozen_config": first["frozen_config"],
        "frozen_config_hash": first["frozen_config_hash"],
        **_gate_e_manifest_fields(split_name=split_name, created_at=created_at),
        "shard_count": len(ordered),
        "shard_start": 0,
        "shard_stop": int(first["total_window_count"]),
        "total_window_count": int(first["total_window_count"]),
        "forecast_steps": int(first["forecast_steps"]),
        "node_count": int(first["node_count"]),
        "rated_power_kw": float(first["rated_power_kw"]),
        "exact_abs_error_limit": exact_abs_error_limit,
        "tft_contract": {
            "uses_future_target": False,
            "target_scope": "history_context_only",
            "future_covariates_scope": "calendar_only",
            "feature_budget_id": "B2",
        },
        "metrics": metrics,
        "components": {
            "window_count": int(components["window_count"]),
            "prediction_count": int(components["prediction_count"]),
            "abs_error_sum": float(components["abs_error_sum"]),
            "sq_error_sum": float(components["sq_error_sum"]),
            "lead_valid": np.asarray(components["lead_valid"], dtype=np.float64).tolist(),
            "lead_abs": np.asarray(components["lead_abs"], dtype=np.float64).tolist(),
            "lead_sq": np.asarray(components["lead_sq"], dtype=np.float64).tolist(),
        },
        "source_shards": [
            {
                "shard_start": int(record["shard_start"]),
                "shard_stop": int(record["shard_stop"]),
                "json_path": record.get("_json_path"),
                "abs_error_path": record.get("abs_error_path"),
                "origin_error_path": record.get("origin_error_path"),
            }
            for record in ordered
        ],
    }


def _load_shard_records(
    output_dir: Path,
    *,
    dataset_id: str,
    split_name: str,
    eval_protocol: str,
    variant_name: str,
) -> list[dict[str, Any]]:
    prefix = f"{_artifact_stem(dataset_id, split_name, eval_protocol, variant_name)}_shard_"
    records: list[dict[str, Any]] = []
    for path in sorted(output_dir.glob(f"{prefix}*.json")):
        record = json.loads(path.read_text(encoding="utf-8"))
        record["_json_path"] = path.name
        records.append(record)
    return records


def _build_config_from_args(args: argparse.Namespace) -> dict[str, Any]:
    return tft_frozen_config(
        variant_name=args.variant,
        seed=args.seed,
        max_train_origins=args.max_train_origins,
        max_checkpoint_origins=args.max_checkpoint_origins,
        checkpoint_eval_protocol=args.checkpoint_eval_protocol,
        residual_anchor_steps=args.residual_anchor_steps,
        train_batch_size=args.train_batch_size,
        max_epochs=args.max_epochs,
        learning_rate=args.learning_rate,
        hidden_size=args.tft_hidden_size,
        lstm_layers=args.tft_lstm_layers,
        attention_head_size=args.tft_attention_head_size,
        hidden_continuous_size=args.tft_hidden_continuous_size,
        dropout=args.tft_dropout,
        eval_window_chunk_size=args.tft_eval_window_chunk_size,
    )


def _assert_split_allowed(args: argparse.Namespace) -> None:
    if args.split_name == "test" and not args.allow_test:
        raise RuntimeError("Test split is disabled; rerun with --allow-test only after parent approval.")
    if args.eval_protocol != ROLLING_PROTOCOL:
        raise RuntimeError("This diagnostics evaluator is intended for rolling_origin_no_refit TFT-PF shards.")


def _evaluate_shard(args: argparse.Namespace) -> None:
    _assert_split_allowed(args)
    started = time.perf_counter()
    frozen_config = _build_config_from_args(args)
    prepared = formal._prepare_dataset(
        args.dataset_id,
        max_train_origins=args.max_train_origins,
        max_eval_origins=args.max_eval_origins,
    )
    windows = _windows_for_split(prepared, split_name=args.split_name, eval_protocol=args.eval_protocol)
    total_window_count = len(windows)
    shard_start, shard_stop = resolve_shard_bounds(
        total_window_count=total_window_count,
        shard_size=args.shard_size,
        shard_index=args.shard_index,
        start=args.start,
        stop=args.stop,
    )
    shard_windows = formal._slice_windows(windows, shard_start, shard_stop)
    checkpoint_windows = formal._validation_windows_for_checkpoint(
        prepared,
        eval_protocol=args.checkpoint_eval_protocol,
        max_origins=args.max_checkpoint_origins,
    )
    model, training_dataset, train_summary = formal._train_tft(
        prepared,
        variant_name=args.variant,
        validation_windows=checkpoint_windows,
        residual_anchor_steps=frozen_config["residual_anchor_steps"],
        seed=args.seed,
        device=args.device,
        batch_size=args.train_batch_size,
        learning_rate=args.learning_rate,
        max_epochs=args.max_epochs,
        hidden_size=args.tft_hidden_size,
        lstm_layers=args.tft_lstm_layers,
        attention_head_size=args.tft_attention_head_size,
        hidden_continuous_size=args.tft_hidden_continuous_size,
        dropout=args.tft_dropout,
    )
    components = _empty_components(prepared.forecast_steps)
    abs_error_chunks: list[np.ndarray] = []
    origin_error_rows: list[dict[str, Any]] = []
    for chunk_windows, predictions in formal._iter_evaluate_tft_chunks(
        model,
        training_dataset,
        prepared,
        shard_windows,
        variant_name=args.variant,
        device=train_summary["device"],
        batch_size=args.train_batch_size,
        residual_anchor_steps=frozen_config["residual_anchor_steps"],
        eval_window_chunk_size=args.tft_eval_window_chunk_size,
    ):
        targets, valid = formal._target_and_valid(prepared, chunk_windows)
        part = _metric_components(predictions, targets, valid, forecast_steps=prepared.forecast_steps)
        _merge_components(components, part)
        abs_error_chunks.append(_abs_errors(predictions, targets, valid))
        origin_offset = shard_start + int(components["window_count"]) - int(predictions.shape[0])
        origin_error_rows.extend(
            _per_origin_error_rows(predictions, targets, valid, chunk_windows, shard_start=origin_offset)
        )
    exact_abs_errors = np.concatenate(abs_error_chunks) if abs_error_chunks else np.zeros((0,), dtype=np.float64)
    if args.exact_ae_limit is not None and exact_abs_errors.size > args.exact_ae_limit:
        raise RuntimeError(
            f"Shard exact AE count {exact_abs_errors.size} exceeds exact_abs_error_limit={args.exact_ae_limit}."
        )
    metrics = _metrics_from_components(
        components,
        rated_power_kw=prepared.rated_power_kw,
        exact_abs_errors=exact_abs_errors,
        exact_abs_error_limit=args.exact_ae_limit,
        metrics_backend="tft_pf_shard_streaming",
    )
    record, _record_abs_errors = build_shard_record_from_arrays(
        np.zeros((0, prepared.forecast_steps, prepared.node_count), dtype=np.float32),
        np.zeros((0, prepared.forecast_steps, prepared.node_count), dtype=np.float32),
        np.zeros((0, prepared.forecast_steps, prepared.node_count), dtype=np.float32),
        dataset_id=args.dataset_id,
        split_name=args.split_name,
        eval_protocol=args.eval_protocol,
        variant_name=args.variant,
        frozen_config=frozen_config,
        shard_start=shard_start,
        shard_stop=shard_stop,
        total_window_count=total_window_count,
        rated_power_kw=prepared.rated_power_kw,
        forecast_steps=prepared.forecast_steps,
        node_count=prepared.node_count,
        exact_abs_error_limit=args.exact_ae_limit,
        target_index_start=int(shard_windows.target_indices[0]),
        target_index_stop_exclusive=int(shard_windows.target_indices[-1]) + 1,
        output_start_us=int(shard_windows.output_start_us[0]),
        output_end_us=int(shard_windows.output_end_us[-1]),
        train_summary=train_summary,
        runtime_seconds=time.perf_counter() - started,
    )
    record["metrics"] = metrics
    record["components"] = {
        "window_count": int(components["window_count"]),
        "prediction_count": int(components["prediction_count"]),
        "abs_error_sum": float(components["abs_error_sum"]),
        "sq_error_sum": float(components["sq_error_sum"]),
        "lead_valid": np.asarray(components["lead_valid"], dtype=np.float64).tolist(),
        "lead_abs": np.asarray(components["lead_abs"], dtype=np.float64).tolist(),
        "lead_sq": np.asarray(components["lead_sq"], dtype=np.float64).tolist(),
    }
    paths = write_shard_artifacts(record, exact_abs_errors, origin_error_rows, args.output_dir)
    print(json.dumps({key: str(value) for key, value in paths.items()}, indent=2, sort_keys=True))


def _aggregate(args: argparse.Namespace) -> None:
    _assert_split_allowed(args)
    records = _load_shard_records(
        args.output_dir,
        dataset_id=args.dataset_id,
        split_name=args.split_name,
        eval_protocol=args.eval_protocol,
        variant_name=args.variant,
    )
    aggregate = aggregate_shard_records(
        records,
        dataset_id=args.dataset_id,
        split_name=args.split_name,
        eval_protocol=args.eval_protocol,
        variant_name=args.variant,
        exact_abs_error_limit=args.exact_ae_limit,
        base_dir=args.output_dir,
    )
    paths = aggregate_artifact_paths(
        args.output_dir,
        dataset_id=args.dataset_id,
        split_name=args.split_name,
        eval_protocol=args.eval_protocol,
        variant_name=args.variant,
    )
    _write_json(paths["json"], aggregate)
    pl.DataFrame([{**_metric_identity_fields(aggregate), **aggregate["metrics"], "shard_count": aggregate["shard_count"]}]).write_csv(
        paths["csv"]
    )
    print(json.dumps({key: str(value) for key, value in paths.items()}, indent=2, sort_keys=True))


def _plan(args: argparse.Namespace) -> None:
    _assert_split_allowed(args)
    frozen_config = _build_config_from_args(args)
    prepared = formal._prepare_dataset(
        args.dataset_id,
        max_train_origins=args.max_train_origins,
        max_eval_origins=args.max_eval_origins,
    )
    total_window_count = len(_windows_for_split(prepared, split_name=args.split_name, eval_protocol=args.eval_protocol))
    if args.shard_size <= 0:
        raise ValueError("--shard-size must be positive.")
    shards = []
    for shard_start in range(0, total_window_count, args.shard_size):
        shard_stop = min(shard_start + args.shard_size, total_window_count)
        paths = shard_artifact_paths(
            args.output_dir,
            dataset_id=args.dataset_id,
            split_name=args.split_name,
            eval_protocol=args.eval_protocol,
            variant_name=args.variant,
            shard_start=shard_start,
            shard_stop=shard_stop,
        )
        shards.append(
            {
                "shard_index": len(shards),
                "shard_start": shard_start,
                "shard_stop": shard_stop,
                "json_path": str(paths["json"]),
                "csv_path": str(paths["csv"]),
                "abs_error_path": str(paths["abs_errors"]),
                "origin_error_path": str(paths["origin_errors"]),
            }
        )
    payload = {
        "schema_version": SHARD_SCHEMA_VERSION,
        "created_at": datetime.now(tz=UTC).isoformat(),
        "dataset_id": args.dataset_id,
        "split_name": args.split_name,
        "eval_protocol": args.eval_protocol,
        "model_variant": args.variant,
        "total_window_count": total_window_count,
        "forecast_steps": prepared.forecast_steps,
        "node_count": prepared.node_count,
        "max_prediction_count": total_window_count * prepared.forecast_steps * prepared.node_count,
        "max_eval_origins": args.max_eval_origins,
        "shard_size": args.shard_size,
        "shard_count": len(shards),
        "exact_abs_error_limit": args.exact_ae_limit,
        "frozen_config": frozen_config,
        "frozen_config_hash": _config_hash(frozen_config),
        "formal_search_config_id": _tft_search_config_id(frozen_config),
        "tft_contract": {
            "uses_future_target": False,
            "target_scope": "history_context_only",
            "future_covariates_scope": "calendar_only",
            "feature_budget_id": "B2",
        },
        "shards": shards,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    plan_path = args.output_dir / f"{_artifact_stem(args.dataset_id, args.split_name, args.eval_protocol, args.variant)}_plan.json"
    _write_json(plan_path, payload)
    print(str(plan_path))


def _add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--dataset-id", default="kelmarsh")
    parser.add_argument("--split", dest="split_name", choices=("val", "test"), default="val")
    parser.add_argument("--eval-protocol", default=ROLLING_PROTOCOL)
    parser.add_argument("--variant", choices=(formal.TFT_RESIDUAL_VARIANT, formal.TFT_DIRECT_VARIANT), default=formal.TFT_RESIDUAL_VARIANT)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--exact-ae-limit", type=int, default=DEFAULT_EXACT_AE_LIMIT)
    parser.add_argument("--allow-test", action="store_true")
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--max-train-origins", type=int, default=512)
    parser.add_argument("--max-eval-origins", type=int, default=None)
    parser.add_argument("--max-checkpoint-origins", type=int, default=256)
    parser.add_argument("--checkpoint-eval-protocol", default=ROLLING_PROTOCOL)
    parser.add_argument("--residual-anchor-steps", type=int, default=1)
    parser.add_argument("--train-batch-size", type=int, default=128)
    parser.add_argument("--max-epochs", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--tft-hidden-size", type=int, default=32)
    parser.add_argument("--tft-lstm-layers", type=int, default=1)
    parser.add_argument("--tft-attention-head-size", type=int, default=4)
    parser.add_argument("--tft-hidden-continuous-size", type=int, default=16)
    parser.add_argument("--tft-dropout", type=float, default=0.1)
    parser.add_argument("--tft-eval-window-chunk-size", type=int, default=512)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Recoverable TFT-PF rolling shard evaluator.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    plan_parser = subparsers.add_parser("plan", help="Write a full-shard execution plan without model inference.")
    _add_common_args(plan_parser)
    plan_parser.add_argument("--shard-size", type=int, required=True)
    plan_parser.set_defaults(func=_plan)

    eval_parser = subparsers.add_parser("eval-shard", help="Train/reload the frozen TFT config and evaluate one shard.")
    _add_common_args(eval_parser)
    eval_parser.add_argument("--device", default="cuda")
    eval_parser.add_argument("--shard-size", type=int)
    eval_parser.add_argument("--shard-index", type=int)
    eval_parser.add_argument("--start", type=int)
    eval_parser.add_argument("--stop", type=int)
    eval_parser.set_defaults(func=_evaluate_shard)

    aggregate_parser = subparsers.add_parser("aggregate", help="Aggregate completed shards into full metrics.")
    _add_common_args(aggregate_parser)
    aggregate_parser.set_defaults(func=_aggregate)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
