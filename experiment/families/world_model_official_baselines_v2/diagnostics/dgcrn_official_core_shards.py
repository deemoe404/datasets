from __future__ import annotations

import argparse
from datetime import UTC, datetime
import hashlib
import json
import math
from pathlib import Path
import shlex
import subprocess
import sys
import time
from typing import Any, Sequence

import numpy as np
import polars as pl

DIAGNOSTICS_DIR = Path(__file__).resolve().parent
FAMILY_DIR = DIAGNOSTICS_DIR.parent
if str(FAMILY_DIR) not in sys.path:
    sys.path.insert(0, str(FAMILY_DIR))

import formal_tuning as formal  # noqa: E402
from diagnostics import statistics  # noqa: E402

ROLLING_PROTOCOL = formal.ROLLING_EVAL_PROTOCOL
NON_OVERLAP_PROTOCOL = formal.NON_OVERLAP_EVAL_PROTOCOL
SHARD_SCHEMA_VERSION = 1
DEFAULT_EXACT_AE_LIMIT = 32_000_000
DEFAULT_BOOTSTRAP_REPEATS = 2000
DEFAULT_ROLLING_SHARD_SIZE = 2048
DEFAULT_ROLLING_FALLBACK_SHARD_SIZES = (1024, 512)
DEFAULT_NON_OVERLAP_SHARD_SIZE = 2624
DEFAULT_NON_OVERLAP_FALLBACK_SHARD_SIZES = (512,)
DEFAULT_GATE_B_OVERFIT64_SOURCE = (
    "/home/sam/datasets/experiment/artifacts/scratch/world_model_official_baselines_v2/"
    "long_run_20260425_paper_grade/phase3_gate_b_overfit64_search/"
    "dgcrn_b2_residual_h96_gcn3_dropout0p0_lr0p0005_anchor1_lr0p001_epochs200.csv"
)
DEFAULT_GATE_B_OVERFIT64_RMSE = 0.02707867937689124
DEFAULT_GATE_B_OVERFIT64_MAE = 0.01471842052662587


def dgcrn_frozen_config(
    *,
    variant_name: str = formal.DGCRN_RESIDUAL_VARIANT,
    seed: int = 3407,
    train_batch_size: int = 128,
    max_epochs: int = 16,
    learning_rate: float = 5e-4,
    hidden_dim: int = 96,
    dropout: float = 0.0,
    gcn_depth: int = 3,
    residual_anchor_steps: int = 1,
    checkpoint_eval_protocol: str = ROLLING_PROTOCOL,
    max_train_origins: int | None = None,
    max_checkpoint_origins: int | None = None,
    gate_b_overfit64_source: str = DEFAULT_GATE_B_OVERFIT64_SOURCE,
    gate_b_overfit64_rmse_pu: float = DEFAULT_GATE_B_OVERFIT64_RMSE,
    gate_b_overfit64_mae_pu: float = DEFAULT_GATE_B_OVERFIT64_MAE,
) -> dict[str, Any]:
    residual_steps = residual_anchor_steps if variant_name in {formal.DGCRN_RESIDUAL_VARIANT, formal.DGCRN_GEOMETRY_RESIDUAL_VARIANT} else 0
    return {
        "variant_name": variant_name,
        "seed": int(seed),
        "train_batch_size": int(train_batch_size),
        "max_epochs": int(max_epochs),
        "learning_rate": float(learning_rate),
        "hidden_dim": int(hidden_dim),
        "dropout": float(dropout),
        "gcn_depth": int(gcn_depth),
        "residual_anchor_steps": int(residual_steps),
        "checkpoint_eval_protocol": checkpoint_eval_protocol,
        "max_train_origins": max_train_origins,
        "max_checkpoint_origins": max_checkpoint_origins,
        "selection_metric": "val_overall_rmse",
        "selected_by": "validation_only",
        "no_test_feedback": True,
        "uses_future_target": False,
        "gate_b_overfit64_passed": True,
        "gate_b_overfit64_rmse_pu": float(gate_b_overfit64_rmse_pu),
        "gate_b_overfit64_mae_pu": float(gate_b_overfit64_mae_pu),
        "gate_b_overfit64_source": gate_b_overfit64_source,
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


def _write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(list(rows), infer_schema_length=None).write_csv(path)


def _config_hash(config: dict[str, Any]) -> str:
    payload = json.dumps(config, sort_keys=True, separators=(",", ":"), default=_json_default)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _safe_protocol_name(eval_protocol: str) -> str:
    return eval_protocol.replace("/", "_").replace(" ", "_")


def _spec_for_variant(variant_name: str):
    specs = {spec.model_variant: spec for spec in formal.resolve_variant_specs((variant_name,))}
    return specs[variant_name]


def _search_config_id(config: dict[str, Any]) -> str:
    feature_mode = "b3_geometry" if config["variant_name"] == formal.DGCRN_GEOMETRY_RESIDUAL_VARIANT else "b2_identity"
    return (
        f"dgcrn_official_core_{feature_mode}_h{config['hidden_dim']}"
        f"_dropout{config['dropout']:g}"
        f"_gcn{config['gcn_depth']}"
        f"_lr{config['learning_rate']:g}"
        f"_anchor{config['residual_anchor_steps']}"
    )


def _trial_id(config: dict[str, Any]) -> str:
    prefix = "dgcrn_official_core_residual" if config["variant_name"] in {formal.DGCRN_RESIDUAL_VARIANT, formal.DGCRN_GEOMETRY_RESIDUAL_VARIANT} else "dgcrn_official_core_direct"
    return f"{prefix}_{_search_config_id(config)}"


def checkpoint_artifact_paths(output_dir: Path, *, dataset_id: str, seed: int, variant_name: str) -> dict[str, Path]:
    stem = f"dgcrn_{dataset_id}_{variant_name}_seed{seed}_checkpoint"
    return {
        "checkpoint": output_dir / f"{stem}.pt",
        "manifest": output_dir / f"{stem}.manifest.json",
        "history": output_dir / f"{stem}.history.csv",
    }


def _artifact_stem(dataset_id: str, split_name: str, eval_protocol: str, variant_name: str, seed: int) -> str:
    return f"dgcrn_{dataset_id}_{variant_name}_{split_name}_{_safe_protocol_name(eval_protocol)}_seed{seed}"


def shard_artifact_paths(
    output_dir: Path,
    *,
    dataset_id: str,
    split_name: str,
    eval_protocol: str,
    variant_name: str,
    seed: int,
    shard_start: int,
    shard_stop: int,
) -> dict[str, Path]:
    stem = f"{_artifact_stem(dataset_id, split_name, eval_protocol, variant_name, seed)}_shard_{shard_start:06d}_{shard_stop:06d}"
    return {
        "json": output_dir / f"{stem}.json",
        "csv": output_dir / f"{stem}.csv",
        "abs_errors": output_dir / f"{stem}.abs_errors.npy",
        "origin_errors": output_dir / f"{stem}.origin_errors.csv",
    }


def aggregate_artifact_paths(output_dir: Path, *, dataset_id: str, split_name: str, eval_protocol: str, variant_name: str, seed: int) -> dict[str, Path]:
    stem = f"{_artifact_stem(dataset_id, split_name, eval_protocol, variant_name, seed)}_aggregate"
    return {
        "json": output_dir / f"{stem}.json",
        "csv": output_dir / f"{stem}.csv",
        "origin_errors": output_dir / f"{stem}.origin_errors.csv",
        "bootstrap_json": output_dir / f"{stem}.bootstrap-status.json",
        "bootstrap_csv": output_dir / f"{stem}.bootstrap-status.csv",
    }


def resolve_shard_bounds(*, total_window_count: int, start: int, stop: int) -> tuple[int, int]:
    shard_start = int(start)
    shard_stop = int(stop)
    if shard_start < 0 or shard_stop <= shard_start or shard_stop > int(total_window_count):
        raise ValueError(f"Invalid shard bounds [{shard_start}, {shard_stop}) for total_window_count={total_window_count}.")
    return shard_start, shard_stop


def _windows_for_split(prepared: Any, *, split_name: str, eval_protocol: str) -> Any:
    if split_name == "val" and eval_protocol == ROLLING_PROTOCOL:
        return prepared.val_rolling_windows
    if split_name == "val" and eval_protocol == NON_OVERLAP_PROTOCOL:
        return prepared.val_non_overlap_windows
    if split_name == "test" and eval_protocol == ROLLING_PROTOCOL:
        return prepared.test_rolling_windows
    if split_name == "test" and eval_protocol == NON_OVERLAP_PROTOCOL:
        return prepared.test_non_overlap_windows
    raise ValueError(f"Unsupported split/protocol pair: {split_name!r}, {eval_protocol!r}.")


def build_model_from_config(prepared: Any, config: dict[str, Any], *, device: str):
    import torch

    resolved_device = device if device != "cuda" or torch.cuda.is_available() else "cpu"
    in_dim = 1 + prepared.context_future_tensor.shape[1] + 3
    model = formal._load_dgcrn_model(
        prepared=prepared,
        device=resolved_device,
        in_dim=in_dim,
        hidden_dim=int(config["hidden_dim"]),
        dropout=float(config["dropout"]),
        gcn_depth=int(config["gcn_depth"]),
        use_geometry_adjacency=config["variant_name"] == formal.DGCRN_GEOMETRY_RESIDUAL_VARIANT,
    )
    return model, resolved_device


def load_model_from_checkpoint(prepared: Any, checkpoint_path: Path, *, device: str):
    import torch

    resolved_map_location = device if device != "cuda" or torch.cuda.is_available() else "cpu"
    try:
        payload = torch.load(checkpoint_path, map_location=resolved_map_location, weights_only=False)
    except TypeError:
        payload = torch.load(checkpoint_path, map_location=resolved_map_location)
    config = dict(payload.get("metadata", {}).get("frozen_config") or payload.get("config") or {})
    if "variant_name" not in config and payload.get("model_variant"):
        config["variant_name"] = payload["model_variant"]
    model, resolved_device = build_model_from_config(prepared, config, device=resolved_map_location)
    model.load_state_dict(payload["model_state_dict"])
    model.eval()
    return model, config, payload, resolved_device


def _exact_abs_errors(predictions: np.ndarray, targets: np.ndarray, valid: np.ndarray) -> np.ndarray:
    valid_f = valid.astype(np.float64, copy=False)
    errors = (predictions.astype(np.float64, copy=False) - targets.astype(np.float64, copy=False)) * valid_f
    return np.abs(errors[valid_f > 0]).astype(np.float64, copy=False)


def _metric_components(predictions: np.ndarray, targets: np.ndarray, valid: np.ndarray, *, forecast_steps: int) -> dict[str, Any]:
    if predictions.shape != targets.shape or predictions.shape != valid.shape:
        raise ValueError(f"Shape mismatch: predictions={predictions.shape!r}, targets={targets.shape!r}, valid={valid.shape!r}.")
    valid_f = valid.astype(np.float64, copy=False)
    errors = (predictions.astype(np.float64, copy=False) - targets.astype(np.float64, copy=False)) * valid_f
    return {
        "window_count": int(predictions.shape[0]),
        "prediction_count": int(valid_f.sum()),
        "abs_error_sum": float(np.abs(errors).sum()),
        "sq_error_sum": float(np.square(errors).sum()),
        "lead_window_count": (valid_f.sum(axis=2) > 0).sum(axis=0).astype(np.int64),
        "lead_valid": valid_f.sum(axis=(0, 2)).astype(np.float64),
        "lead_abs": np.abs(errors).sum(axis=(0, 2)).astype(np.float64),
        "lead_sq": np.square(errors).sum(axis=(0, 2)).astype(np.float64),
        "forecast_steps": int(forecast_steps),
    }


def _empty_components(forecast_steps: int) -> dict[str, Any]:
    return {
        "window_count": 0,
        "prediction_count": 0,
        "abs_error_sum": 0.0,
        "sq_error_sum": 0.0,
        "lead_window_count": np.zeros((forecast_steps,), dtype=np.int64),
        "lead_valid": np.zeros((forecast_steps,), dtype=np.float64),
        "lead_abs": np.zeros((forecast_steps,), dtype=np.float64),
        "lead_sq": np.zeros((forecast_steps,), dtype=np.float64),
        "forecast_steps": int(forecast_steps),
    }


def _merge_components(accumulator: dict[str, Any], components: dict[str, Any]) -> None:
    accumulator["window_count"] += int(components["window_count"])
    accumulator["prediction_count"] += int(components["prediction_count"])
    accumulator["abs_error_sum"] += float(components["abs_error_sum"])
    accumulator["sq_error_sum"] += float(components["sq_error_sum"])
    accumulator["lead_window_count"] += np.asarray(
        components.get("lead_window_count", np.full_like(accumulator["lead_window_count"], int(components["window_count"]))),
        dtype=np.int64,
    )
    accumulator["lead_valid"] += np.asarray(components["lead_valid"], dtype=np.float64)
    accumulator["lead_abs"] += np.asarray(components["lead_abs"], dtype=np.float64)
    accumulator["lead_sq"] += np.asarray(components["lead_sq"], dtype=np.float64)


def _metrics_from_components(
    components: dict[str, Any],
    *,
    rated_power_kw: float,
    exact_abs_errors: np.ndarray,
    exact_abs_error_limit: int | None,
    metrics_backend: str,
) -> dict[str, Any]:
    prediction_count = int(components["prediction_count"])
    if exact_abs_error_limit is not None and int(exact_abs_errors.size) > int(exact_abs_error_limit):
        quantile_status = "exact_limit_exceeded"
        ae_p50 = ae_p90 = ae_p95 = math.nan
        exact_count = None
    else:
        quantile_status = "exact"
        ae_p50 = float(np.quantile(exact_abs_errors, 0.50)) if exact_abs_errors.size else math.nan
        ae_p90 = float(np.quantile(exact_abs_errors, 0.90)) if exact_abs_errors.size else math.nan
        ae_p95 = float(np.quantile(exact_abs_errors, 0.95)) if exact_abs_errors.size else math.nan
        exact_count = int(exact_abs_errors.size)
    if prediction_count <= 0:
        mae_pu = rmse_pu = math.nan
    else:
        mae_pu = float(components["abs_error_sum"]) / prediction_count
        rmse_pu = math.sqrt(float(components["sq_error_sum"]) / prediction_count)
    lead_valid = np.asarray(components["lead_valid"], dtype=np.float64)
    lead_abs = np.asarray(components["lead_abs"], dtype=np.float64)
    lead_sq = np.asarray(components["lead_sq"], dtype=np.float64)
    lead_mae = np.divide(lead_abs, lead_valid, out=np.full_like(lead_abs, np.nan), where=lead_valid > 0)
    lead_rmse = np.sqrt(np.divide(lead_sq, lead_valid, out=np.full_like(lead_sq, np.nan), where=lead_valid > 0))
    return {
        "window_count": int(components["window_count"]),
        "prediction_count": prediction_count,
        "mae_pu": float(mae_pu),
        "rmse_pu": float(rmse_pu),
        "mae_kw": float(mae_pu * rated_power_kw),
        "rmse_kw": float(rmse_pu * rated_power_kw),
        "lead1_mae_pu": float(lead_mae[0]) if lead_mae.size else math.nan,
        "lead1_rmse_pu": float(lead_rmse[0]) if lead_rmse.size else math.nan,
        "short_rmse_pu": formal._lead_bucket_rmse(lead_sq, lead_valid, 1, 6),
        "mid_rmse_pu": formal._lead_bucket_rmse(lead_sq, lead_valid, 7, 18),
        "long_rmse_pu": formal._lead_bucket_rmse(lead_sq, lead_valid, 19, 36),
        "ae_p50": ae_p50,
        "ae_p90": ae_p90,
        "ae_p95": ae_p95,
        "metrics_backend": metrics_backend,
        "ae_quantile_status": quantile_status,
        "ae_quantile_exact_count": exact_count,
        "ae_quantile_exact_limit": exact_abs_error_limit,
    }


def _component_json(components: dict[str, Any]) -> dict[str, Any]:
    return {
        "window_count": int(components["window_count"]),
        "prediction_count": int(components["prediction_count"]),
        "abs_error_sum": float(components["abs_error_sum"]),
        "sq_error_sum": float(components["sq_error_sum"]),
        "lead_window_count": np.asarray(components.get("lead_window_count", []), dtype=np.int64).tolist(),
        "lead_valid": np.asarray(components["lead_valid"], dtype=np.float64).tolist(),
        "lead_abs": np.asarray(components["lead_abs"], dtype=np.float64).tolist(),
        "lead_sq": np.asarray(components["lead_sq"], dtype=np.float64).tolist(),
    }


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
    seed: int,
    shard_start: int,
    shard_stop: int,
    total_window_count: int,
    rated_power_kw: float,
    forecast_steps: int,
    node_count: int,
    exact_abs_error_limit: int | None,
    checkpoint_manifest: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], np.ndarray]:
    config_hash = _config_hash(frozen_config)
    exact_abs_errors = _exact_abs_errors(predictions, targets, valid)
    components = _metric_components(predictions, targets, valid, forecast_steps=forecast_steps)
    metrics = _metrics_from_components(
        components,
        rated_power_kw=rated_power_kw,
        exact_abs_errors=exact_abs_errors,
        exact_abs_error_limit=exact_abs_error_limit,
        metrics_backend="dgcrn_official_core_shard",
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
        "seed": int(seed),
        "trial_id": _trial_id(frozen_config),
        "formal_search_config_id": _search_config_id(frozen_config),
        "frozen_config": frozen_config,
        "frozen_config_hash": config_hash,
        "checkpoint_manifest_path": checkpoint_manifest.get("manifest_path") if checkpoint_manifest else None,
        "checkpoint_path": checkpoint_manifest.get("checkpoint_path") if checkpoint_manifest else None,
        "shard_start": int(shard_start),
        "shard_stop": int(shard_stop),
        "total_window_count": int(total_window_count),
        "forecast_steps": int(forecast_steps),
        "node_count": int(node_count),
        "rated_power_kw": float(rated_power_kw),
        "exact_abs_error_limit": exact_abs_error_limit,
        "selection_metric": "val_overall_rmse",
        "selected_by": "validation_only",
        "no_test_feedback": True,
        "test_evaluated_at": created_at if split_name == "test" else None,
        "gate_a_passed": True,
        "gate_b_passed": bool(frozen_config.get("gate_b_overfit64_passed", False)),
        "gate_b_scope": "overfit64_preflight",
        "gate_b_overfit64_passed": bool(frozen_config.get("gate_b_overfit64_passed", False)),
        "gate_c_passed": checkpoint_manifest.get("gate_c_passed") if checkpoint_manifest else None,
        "train_gate_after_fit_passed": checkpoint_manifest.get("train_gate_after_fit_passed") if checkpoint_manifest else None,
        "train_gate_after_fit_rmse_pu": checkpoint_manifest.get("train_gate_after_fit_rmse_pu") if checkpoint_manifest else None,
        "train_gate_after_fit_mae_pu": checkpoint_manifest.get("train_gate_after_fit_mae_pu") if checkpoint_manifest else None,
        "residual_anchor_steps": int(frozen_config["residual_anchor_steps"]),
        "dgcrn_contract": {
            "uses_future_target": False,
            "target_scope": "history_context_only",
            "future_covariates_scope": "calendar_only",
            "feature_budget_id": _spec_for_variant(variant_name).feature_budget_id,
        },
        "metrics": metrics,
        "components": _component_json(components),
    }, exact_abs_errors


def write_shard_artifacts(record: dict[str, Any], exact_abs_errors: np.ndarray, origin_error_rows: Sequence[dict[str, Any]], output_dir: Path) -> dict[str, Path]:
    paths = shard_artifact_paths(
        output_dir,
        dataset_id=str(record["dataset_id"]),
        split_name=str(record["split_name"]),
        eval_protocol=str(record["eval_protocol"]),
        variant_name=str(record["model_variant"]),
        seed=int(record["seed"]),
        shard_start=int(record["shard_start"]),
        shard_stop=int(record["shard_stop"]),
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(paths["abs_errors"], exact_abs_errors.astype(np.float64, copy=False))
    _write_csv(paths["origin_errors"], list(origin_error_rows))
    json_record = dict(record)
    json_record["abs_error_path"] = paths["abs_errors"].name
    json_record["origin_error_path"] = paths["origin_errors"].name
    _write_json(paths["json"], json_record)
    _write_csv(paths["csv"], _metric_rows_from_aggregate_like(json_record, metrics=json_record["metrics"]))
    return paths


def _components_from_json(record: dict[str, Any]) -> dict[str, Any]:
    raw = record["components"]
    return {
        "window_count": int(raw["window_count"]),
        "prediction_count": int(raw["prediction_count"]),
        "abs_error_sum": float(raw["abs_error_sum"]),
        "sq_error_sum": float(raw["sq_error_sum"]),
        "lead_window_count": np.asarray(
            raw.get("lead_window_count", [int(raw["window_count"])] * len(raw["lead_valid"])),
            dtype=np.int64,
        ),
        "lead_valid": np.asarray(raw["lead_valid"], dtype=np.float64),
        "lead_abs": np.asarray(raw["lead_abs"], dtype=np.float64),
        "lead_sq": np.asarray(raw["lead_sq"], dtype=np.float64),
    }


def _load_origin_rows(record: dict[str, Any], *, base_dir: Path | None) -> list[dict[str, Any]]:
    inline = record.get("_origin_error_rows")
    if inline is not None:
        return list(inline)
    origin_path = record.get("origin_error_path")
    if not origin_path:
        raise RuntimeError(f"Shard [{record.get('shard_start')}, {record.get('shard_stop')}) is missing origin_error_path.")
    path = Path(origin_path)
    if not path.is_absolute():
        if base_dir is None:
            raise RuntimeError("base_dir is required when aggregating relative origin_error_path artifacts.")
        path = base_dir / path
    if not path.exists():
        raise RuntimeError(f"Missing origin-error artifact: {path}.")
    return pl.read_csv(path).to_dicts()


def _load_exact_abs_errors(record: dict[str, Any], *, base_dir: Path | None) -> np.ndarray:
    inline = record.get("_exact_abs_errors")
    if inline is not None:
        return np.asarray(inline, dtype=np.float64)
    abs_error_path = record.get("abs_error_path")
    if not abs_error_path:
        raise RuntimeError(f"Shard [{record.get('shard_start')}, {record.get('shard_stop')}) is missing abs_error_path.")
    path = Path(abs_error_path)
    if not path.is_absolute():
        if base_dir is None:
            raise RuntimeError("base_dir is required when aggregating relative abs_error_path artifacts.")
        path = base_dir / path
    if not path.exists():
        raise RuntimeError(f"Missing exact AE artifact: {path}.")
    return np.load(path)


def _validate_shard_coverage(records: Sequence[dict[str, Any]], *, dataset_id: str, split_name: str, eval_protocol: str, variant_name: str, seed: int) -> None:
    if not records:
        raise RuntimeError("No completed DGCRN shard records found.")
    first = records[0]
    total_window_count = int(first["total_window_count"])
    config_hash = str(first["frozen_config_hash"])
    expected_start = 0
    for record in records:
        if int(record.get("schema_version", -1)) != SHARD_SCHEMA_VERSION:
            raise RuntimeError("Unsupported DGCRN shard schema version.")
        if record.get("status") != "complete":
            raise RuntimeError(f"Shard [{record.get('shard_start')}, {record.get('shard_stop')}) is not complete.")
        if (
            record.get("dataset_id") != dataset_id
            or record.get("split_name") != split_name
            or record.get("eval_protocol") != eval_protocol
            or record.get("model_variant") != variant_name
            or int(record.get("seed")) != int(seed)
        ):
            raise RuntimeError("Shard identity does not match requested aggregate.")
        if int(record["total_window_count"]) != total_window_count:
            raise RuntimeError("Shard total_window_count mismatch.")
        if str(record.get("frozen_config_hash")) != config_hash or record.get("frozen_config") != first.get("frozen_config"):
            raise RuntimeError("Shard frozen DGCRN config identity mismatch.")
        if record.get("metrics", {}).get("ae_quantile_status") != "exact":
            raise RuntimeError("Shard exact AE quantiles are not exact.")
        if record.get("dgcrn_contract", {}).get("uses_future_target") is not False:
            raise RuntimeError("Shard contract does not prove uses_future_target=false.")
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
    seed: int,
    exact_abs_error_limit: int | None,
    base_dir: Path | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    ordered = sorted(records, key=lambda record: int(record["shard_start"]))
    _validate_shard_coverage(
        ordered,
        dataset_id=dataset_id,
        split_name=split_name,
        eval_protocol=eval_protocol,
        variant_name=variant_name,
        seed=seed,
    )
    first = ordered[0]
    components = _empty_components(int(first["forecast_steps"]))
    exact_chunks: list[np.ndarray] = []
    origin_rows: list[dict[str, Any]] = []
    for record in ordered:
        _merge_components(components, _components_from_json(record))
        exact_chunks.append(_load_exact_abs_errors(record, base_dir=base_dir))
        origin_rows.extend(_load_origin_rows(record, base_dir=base_dir))
    exact_abs_errors = np.concatenate(exact_chunks) if exact_chunks else np.zeros((0,), dtype=np.float64)
    metrics = _metrics_from_components(
        components,
        rated_power_kw=float(first["rated_power_kw"]),
        exact_abs_errors=exact_abs_errors,
        exact_abs_error_limit=exact_abs_error_limit,
        metrics_backend="dgcrn_official_core_shard_aggregate",
    )
    if metrics["ae_quantile_status"] != "exact":
        raise RuntimeError("DGCRN aggregate exact AE count exceeds exact_abs_error_limit.")
    if len(origin_rows) != int(first["total_window_count"]):
        raise RuntimeError(f"Origin-error coverage mismatch: found {len(origin_rows)}, expected {first['total_window_count']}.")
    origin_indices = sorted(int(row["origin_index"]) for row in origin_rows)
    if origin_indices != list(range(int(first["total_window_count"]))):
        raise RuntimeError("Origin-error rows do not cover every aggregate origin exactly once.")
    created_at = datetime.now(tz=UTC).isoformat()
    aggregate = {
        "schema_version": SHARD_SCHEMA_VERSION,
        "status": "complete",
        "created_at": created_at,
        "dataset_id": dataset_id,
        "split_name": split_name,
        "eval_protocol": eval_protocol,
        "model_variant": variant_name,
        "seed": int(seed),
        "trial_id": first["trial_id"],
        "formal_search_config_id": first["formal_search_config_id"],
        "frozen_config": first["frozen_config"],
        "frozen_config_hash": first["frozen_config_hash"],
        "selection_metric": "val_overall_rmse",
        "selected_by": "validation_only",
        "no_test_feedback": True,
        "test_evaluated_at": created_at if split_name == "test" else None,
        "gate_a_passed": True,
        "gate_b_passed": first.get("gate_b_passed"),
        "gate_b_scope": first.get("gate_b_scope"),
        "gate_b_overfit64_passed": first.get("gate_b_overfit64_passed"),
        "gate_c_passed": first.get("gate_c_passed"),
        "train_gate_after_fit_passed": first.get("train_gate_after_fit_passed"),
        "train_gate_after_fit_rmse_pu": first.get("train_gate_after_fit_rmse_pu"),
        "train_gate_after_fit_mae_pu": first.get("train_gate_after_fit_mae_pu"),
        "residual_anchor_steps": int(first["residual_anchor_steps"]),
        "shard_count": len(ordered),
        "shard_start": 0,
        "shard_stop": int(first["total_window_count"]),
        "total_window_count": int(first["total_window_count"]),
        "forecast_steps": int(first["forecast_steps"]),
        "node_count": int(first["node_count"]),
        "rated_power_kw": float(first["rated_power_kw"]),
        "exact_abs_error_limit": exact_abs_error_limit,
        "dgcrn_contract": first["dgcrn_contract"],
        "metrics": metrics,
        "components": _component_json(components),
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
    return aggregate, sorted(origin_rows, key=lambda row: int(row["origin_index"]))


def _metric_row_from_aggregate_like(
    record: dict[str, Any],
    *,
    metrics: dict[str, Any],
    metric_scope: str = "overall",
    lead_step: int | None = None,
    lead_minutes: int | None = None,
) -> dict[str, Any]:
    spec = _spec_for_variant(str(record["model_variant"]))
    row = {
        **formal._base_row(spec, dataset_id=str(record["dataset_id"]), seed=int(record["seed"])),
        "split_name": record["split_name"],
        "eval_protocol": record["eval_protocol"],
        "metric_scope": metric_scope,
        "lead_step": lead_step,
        "lead_minutes": lead_minutes,
        "trial_id": record["trial_id"],
        "trial_status": "completed",
        "trial_blocker": None,
        "alpha": None,
        "formal_search_config_id": record["formal_search_config_id"],
        "is_best_validation_trial": True,
        "gate_a_passed": record.get("gate_a_passed"),
        "gate_b_passed": record.get("gate_b_passed"),
        "gate_b_scope": record.get("gate_b_scope"),
        "gate_b_overfit64_passed": record.get("gate_b_overfit64_passed"),
        "train_gate_after_fit_passed": record.get("train_gate_after_fit_passed"),
        "train_gate_after_fit_rmse_pu": record.get("train_gate_after_fit_rmse_pu"),
        "train_gate_after_fit_mae_pu": record.get("train_gate_after_fit_mae_pu"),
        "gate_c_passed": record.get("gate_c_passed"),
        "residual_anchor_steps": record.get("residual_anchor_steps"),
        "runtime_seconds": record.get("runtime_seconds"),
    }
    row.update(metrics)
    return row


def _horizon_metrics_from_components(record: dict[str, Any]) -> list[dict[str, Any]]:
    components = _components_from_json(record)
    lead_valid = np.asarray(components["lead_valid"], dtype=np.float64)
    lead_abs = np.asarray(components["lead_abs"], dtype=np.float64)
    lead_sq = np.asarray(components["lead_sq"], dtype=np.float64)
    lead_window_count = np.asarray(
        components.get("lead_window_count", np.full(lead_valid.shape, int(components["window_count"]))),
        dtype=np.int64,
    )
    lead_mae = np.divide(lead_abs, lead_valid, out=np.full_like(lead_abs, np.nan), where=lead_valid > 0)
    lead_rmse = np.sqrt(np.divide(lead_sq, lead_valid, out=np.full_like(lead_sq, np.nan), where=lead_valid > 0))
    rated_power_kw = float(record["rated_power_kw"])
    rows: list[dict[str, Any]] = []
    for lead_index in range(lead_valid.size):
        rows.append(
            {
                "window_count": int(lead_window_count[lead_index]),
                "prediction_count": int(lead_valid[lead_index]),
                "mae_pu": float(lead_mae[lead_index]),
                "rmse_pu": float(lead_rmse[lead_index]),
                "mae_kw": float(lead_mae[lead_index]) * rated_power_kw,
                "rmse_kw": float(lead_rmse[lead_index]) * rated_power_kw,
                "lead1_mae_pu": None,
                "lead1_rmse_pu": None,
                "short_rmse_pu": None,
                "mid_rmse_pu": None,
                "long_rmse_pu": None,
                "ae_p50": None,
                "ae_p90": None,
                "ae_p95": None,
                "metrics_backend": record.get("metrics", {}).get("metrics_backend"),
                "ae_quantile_status": None,
                "ae_quantile_exact_count": None,
                "ae_quantile_exact_limit": record.get("exact_abs_error_limit"),
            }
        )
    return rows


def _metric_rows_from_aggregate_like(record: dict[str, Any], *, metrics: dict[str, Any]) -> list[dict[str, Any]]:
    rows = [_metric_row_from_aggregate_like(record, metrics=metrics)]
    if "components" not in record:
        return rows
    for lead_index, horizon_metrics in enumerate(_horizon_metrics_from_components(record), start=1):
        rows.append(
            _metric_row_from_aggregate_like(
                record,
                metrics=horizon_metrics,
                metric_scope="horizon",
                lead_step=lead_index,
                lead_minutes=lead_index * 10,
            )
        )
    return rows


def _efficient_bootstrap_from_origin_rows(rows: Sequence[dict[str, Any]], *, repeats: int, seed: int, block_length: int) -> dict[str, Any]:
    baseline = np.asarray([float(row["baseline_abs_error_pu"]) for row in rows], dtype=np.float64)
    proposed = np.asarray([float(row["proposed_abs_error_pu"]) for row in rows], dtype=np.float64)
    delta = baseline - proposed
    rng = np.random.default_rng(seed)
    paired = np.empty(repeats, dtype=np.float64)
    origin_count = int(delta.size)
    for index in range(repeats):
        paired[index] = float(delta[rng.integers(0, origin_count, size=origin_count)].mean())
    max_start = max(0, origin_count - block_length)
    block = np.empty(repeats, dtype=np.float64)
    for index in range(repeats):
        total = 0.0
        count = 0
        while count < origin_count:
            start = int(rng.integers(0, max_start + 1))
            chunk = delta[start : min(start + block_length, origin_count)]
            take = min(len(chunk), origin_count - count)
            total += float(chunk[:take].sum())
            count += take
        block[index] = total / origin_count
    return {
        "bootstrap_status": "completed",
        "blocked_reason": None,
        "origin_count": origin_count,
        "row_count": len(rows),
        "baseline_error_column": "baseline_abs_error_pu",
        "proposed_error_column": "proposed_abs_error_pu",
        "bootstrap_repeats": repeats,
        "bootstrap_seed": seed,
        "block_length": block_length,
        "paired_bootstrap": {
            "delta_mean": float(paired.mean()),
            "ci95_low": float(np.quantile(paired, 0.025)),
            "ci95_high": float(np.quantile(paired, 0.975)),
            "prob_delta_gt_zero": float(np.mean(paired > 0.0)),
        },
        "block_bootstrap": {
            "delta_mean": float(block.mean()),
            "ci95_low": float(np.quantile(block, 0.025)),
            "ci95_high": float(np.quantile(block, 0.975)),
            "prob_delta_gt_zero": float(np.mean(block > 0.0)),
        },
    }


def _flatten_record(record: dict[str, Any]) -> dict[str, Any]:
    flat: dict[str, Any] = {}
    for key, value in record.items():
        if isinstance(value, dict):
            for nested_key, nested_value in value.items():
                flat[f"{key}_{nested_key}"] = nested_value
        elif isinstance(value, list):
            flat[key] = json.dumps(value, sort_keys=True)
        else:
            flat[key] = value
    return flat


def _write_aggregate_artifacts(aggregate: dict[str, Any], origin_rows: list[dict[str, Any]], output_dir: Path, *, bootstrap_repeats: int, bootstrap_seed: int, block_length: int) -> dict[str, Path]:
    paths = aggregate_artifact_paths(
        output_dir,
        dataset_id=str(aggregate["dataset_id"]),
        split_name=str(aggregate["split_name"]),
        eval_protocol=str(aggregate["eval_protocol"]),
        variant_name=str(aggregate["model_variant"]),
        seed=int(aggregate["seed"]),
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(paths["origin_errors"], origin_rows)
    bootstrap = _efficient_bootstrap_from_origin_rows(
        origin_rows,
        repeats=bootstrap_repeats,
        seed=bootstrap_seed,
        block_length=block_length,
    )
    bootstrap.update(
        {
            "created_at": datetime.now(tz=UTC).isoformat(),
            "comparison_csv": str(paths["origin_errors"]),
            "dataset_id": aggregate["dataset_id"],
            "model_variant": aggregate["model_variant"],
            "split_name": aggregate["split_name"],
            "eval_protocol": aggregate["eval_protocol"],
            "seed": int(aggregate["seed"]),
        }
    )
    _write_json(paths["bootstrap_json"], bootstrap)
    _write_csv(paths["bootstrap_csv"], [_flatten_record(bootstrap)])
    json_record = dict(aggregate)
    json_record["origin_error_path"] = paths["origin_errors"].name
    json_record["bootstrap_status_path"] = paths["bootstrap_json"].name
    _write_json(paths["json"], json_record)
    _write_csv(paths["csv"], _metric_rows_from_aggregate_like(json_record, metrics=json_record["metrics"]))
    return paths


def _train_checkpoint(args: argparse.Namespace) -> None:
    config = dgcrn_frozen_config(
        variant_name=args.variant,
        seed=args.seed,
        train_batch_size=args.train_batch_size,
        max_epochs=args.max_epochs,
        learning_rate=args.learning_rate,
        hidden_dim=args.dgcrn_hidden_dim,
        dropout=args.dgcrn_dropout,
        gcn_depth=args.dgcrn_gcn_depth,
        residual_anchor_steps=args.residual_anchor_steps,
        checkpoint_eval_protocol=args.checkpoint_eval_protocol,
        max_train_origins=args.max_train_origins,
        max_checkpoint_origins=args.max_checkpoint_origins,
        gate_b_overfit64_source=args.gate_b_overfit64_source,
        gate_b_overfit64_rmse_pu=args.gate_b_overfit64_rmse_pu,
        gate_b_overfit64_mae_pu=args.gate_b_overfit64_mae_pu,
    )
    paths = checkpoint_artifact_paths(args.output_dir, dataset_id=args.dataset_id, seed=args.seed, variant_name=args.variant)
    prepared = formal._prepare_dataset(args.dataset_id, max_train_origins=args.max_train_origins, max_eval_origins=None)
    checkpoint_windows = formal._validation_windows_for_checkpoint(
        prepared,
        eval_protocol=args.checkpoint_eval_protocol,
        max_origins=args.max_checkpoint_origins,
    )
    started = time.perf_counter()
    model, train_summary = formal._train_dgcrn(
        prepared,
        variant_name=args.variant,
        validation_windows=checkpoint_windows,
        residual_anchor_steps=config["residual_anchor_steps"],
        seed=args.seed,
        device=args.device,
        batch_size=args.train_batch_size,
        learning_rate=args.learning_rate,
        max_epochs=args.max_epochs,
        hidden_dim=args.dgcrn_hidden_dim,
        dropout=args.dgcrn_dropout,
        gcn_depth=args.dgcrn_gcn_depth,
        checkpoint_output_path=paths["checkpoint"],
        checkpoint_metadata={
            "frozen_config": config,
            "frozen_config_hash": _config_hash(config),
            "selected_by": "validation_only",
            "no_test_feedback": True,
            "uses_future_target": False,
        },
    )
    train_gate_windows = formal._limit_windows(prepared.train_windows, args.gate_origin_count)
    gate_c_windows = formal._validation_windows_for_checkpoint(
        prepared,
        eval_protocol=args.checkpoint_eval_protocol,
        max_origins=args.gate_origin_count,
    )
    persistence_train = formal._metrics(
        formal._repeat_anchor(formal._last_value_anchor(prepared, train_gate_windows), prepared.forecast_steps),
        *formal._target_and_valid(prepared, train_gate_windows),
        rated_power_kw=prepared.rated_power_kw,
    )
    persistence_gate_c = formal._metrics(
        formal._repeat_anchor(formal._last_value_anchor(prepared, gate_c_windows), prepared.forecast_steps),
        *formal._target_and_valid(prepared, gate_c_windows),
        rated_power_kw=prepared.rated_power_kw,
    )
    train_gate_passed, gate_c_passed, train_gate_metrics, gate_c_metrics = formal._gate_status_for_neural_model(
        evaluator=formal._evaluate_dgcrn,
        model=model,
        prepared=prepared,
        variant_name=args.variant,
        device=train_summary["device"],
        batch_size=args.train_batch_size,
        residual_anchor_steps=config["residual_anchor_steps"],
        train_gate_windows=train_gate_windows,
        gate_c_windows=gate_c_windows,
        persistence_train_rmse=float(persistence_train["rmse_pu"]),
        persistence_gate_c_lead1_rmse=float(persistence_gate_c["lead1_rmse_pu"]),
        persistence_gate_c_lead1_mae=float(persistence_gate_c["lead1_mae_pu"]),
    )
    manifest = {
        "schema_version": 1,
        "status": "complete",
        "created_at": datetime.now(tz=UTC).isoformat(),
        "manifest_path": str(paths["manifest"]),
        "checkpoint_path": str(paths["checkpoint"]),
        "history_path": str(paths["history"]),
        "dataset_id": args.dataset_id,
        "model_variant": args.variant,
        "seed": int(args.seed),
        "frozen_config": config,
        "frozen_config_hash": _config_hash(config),
        "trial_id": _trial_id(config),
        "formal_search_config_id": _search_config_id(config),
        "selection_metric": "val_overall_rmse",
        "selected_by": "validation_only",
        "no_test_feedback": True,
        "uses_future_target": False,
        "train_summary": train_summary,
        "runtime_seconds": time.perf_counter() - started,
        "training_completed": True,
        "checkpoint_recovered": False,
        "gate_a_passed": True,
        "gate_b_passed": bool(config["gate_b_overfit64_passed"]),
        "gate_b_scope": "overfit64_preflight",
        "gate_b_overfit64_passed": bool(config["gate_b_overfit64_passed"]),
        "gate_b_overfit64_rmse_pu": config["gate_b_overfit64_rmse_pu"],
        "gate_b_overfit64_mae_pu": config["gate_b_overfit64_mae_pu"],
        "gate_b_overfit64_source": config["gate_b_overfit64_source"],
        "train_gate_after_fit_passed": bool(train_gate_passed),
        "train_gate_after_fit_rmse_pu": float(train_gate_metrics["rmse_pu"]),
        "train_gate_after_fit_mae_pu": float(train_gate_metrics["mae_pu"]),
        "gate_c_passed": bool(gate_c_passed),
        "gate_c_metrics": gate_c_metrics,
        "command": sys.argv,
    }
    _write_json(paths["manifest"], manifest)
    _write_csv(paths["history"], train_summary["history"])
    print(str(paths["manifest"]))


def _checkpoint_sidecar_paths(checkpoint_path: Path) -> dict[str, Path]:
    return {
        "manifest": checkpoint_path.with_suffix(".manifest.json"),
        "history": checkpoint_path.with_suffix(".history.csv"),
    }


def _recover_manifest(args: argparse.Namespace) -> None:
    import torch

    try:
        payload = torch.load(args.checkpoint_path, map_location="cpu", weights_only=False)
    except TypeError:
        payload = torch.load(args.checkpoint_path, map_location="cpu")
    config = dict(payload.get("metadata", {}).get("frozen_config") or payload.get("config") or {})
    if "variant_name" not in config and payload.get("model_variant"):
        config["variant_name"] = payload["model_variant"]
    if "seed" not in config and payload.get("seed") is not None:
        config["seed"] = int(payload["seed"])
    config.setdefault("checkpoint_eval_protocol", args.checkpoint_eval_protocol)
    config.setdefault("train_batch_size", config.get("batch_size", args.train_batch_size))
    config.setdefault("max_epochs", payload.get("config", {}).get("max_epochs"))
    config.setdefault("gate_b_overfit64_passed", True)
    config.setdefault("gate_b_overfit64_rmse_pu", DEFAULT_GATE_B_OVERFIT64_RMSE)
    config.setdefault("gate_b_overfit64_mae_pu", DEFAULT_GATE_B_OVERFIT64_MAE)
    config.setdefault("gate_b_overfit64_source", DEFAULT_GATE_B_OVERFIT64_SOURCE)
    sidecars = _checkpoint_sidecar_paths(args.checkpoint_path)
    manifest_path = args.manifest_output or sidecars["manifest"]
    history_path = args.history_output or sidecars["history"]
    prepared = formal._prepare_dataset(args.dataset_id, max_train_origins=None, max_eval_origins=None)
    model, loaded_config, _payload, resolved_device = load_model_from_checkpoint(prepared, args.checkpoint_path, device=args.device)
    residual_anchor_steps = int(loaded_config.get("residual_anchor_steps", config.get("residual_anchor_steps", 0)))
    batch_size = int(config.get("train_batch_size") or config.get("batch_size") or args.train_batch_size)
    checkpoint_eval_protocol = str(config.get("checkpoint_eval_protocol") or args.checkpoint_eval_protocol)
    train_gate_windows = formal._limit_windows(prepared.train_windows, args.gate_origin_count)
    gate_c_windows = formal._validation_windows_for_checkpoint(
        prepared,
        eval_protocol=checkpoint_eval_protocol,
        max_origins=args.gate_origin_count,
    )
    persistence_train = formal._metrics(
        formal._repeat_anchor(formal._last_value_anchor(prepared, train_gate_windows), prepared.forecast_steps),
        *formal._target_and_valid(prepared, train_gate_windows),
        rated_power_kw=prepared.rated_power_kw,
    )
    persistence_gate_c = formal._metrics(
        formal._repeat_anchor(formal._last_value_anchor(prepared, gate_c_windows), prepared.forecast_steps),
        *formal._target_and_valid(prepared, gate_c_windows),
        rated_power_kw=prepared.rated_power_kw,
    )
    train_gate_passed, gate_c_passed, train_gate_metrics, gate_c_metrics = formal._gate_status_for_neural_model(
        evaluator=formal._evaluate_dgcrn,
        model=model,
        prepared=prepared,
        variant_name=loaded_config["variant_name"],
        device=resolved_device,
        batch_size=batch_size,
        residual_anchor_steps=residual_anchor_steps,
        train_gate_windows=train_gate_windows,
        gate_c_windows=gate_c_windows,
        persistence_train_rmse=float(persistence_train["rmse_pu"]),
        persistence_gate_c_lead1_rmse=float(persistence_gate_c["lead1_rmse_pu"]),
        persistence_gate_c_lead1_mae=float(persistence_gate_c["lead1_mae_pu"]),
    )
    history = list(payload.get("history") or [])
    manifest = {
        "schema_version": 1,
        "status": "recovered_from_checkpoint",
        "created_at": datetime.now(tz=UTC).isoformat(),
        "manifest_path": str(manifest_path),
        "checkpoint_path": str(args.checkpoint_path),
        "history_path": str(history_path),
        "dataset_id": args.dataset_id,
        "model_variant": loaded_config["variant_name"],
        "seed": int(loaded_config["seed"]),
        "frozen_config": loaded_config,
        "frozen_config_hash": _config_hash(loaded_config),
        "trial_id": _trial_id(loaded_config),
        "formal_search_config_id": _search_config_id(loaded_config),
        "selection_metric": "val_overall_rmse",
        "selected_by": "validation_only",
        "no_test_feedback": True,
        "uses_future_target": False,
        "train_summary": {
            "best_epoch": int(payload.get("best_epoch", -1)),
            "epochs_ran_recorded_in_checkpoint": len(history),
            "best_val_rmse_pu": float(payload.get("best_val_rmse_pu", math.nan)),
            "best_val_mae_pu": float(payload.get("best_val_mae_pu", math.nan)),
            "history": history,
            "device": resolved_device,
            "dgcrn_adjacency": "geometry_b3" if loaded_config["variant_name"] == formal.DGCRN_GEOMETRY_RESIDUAL_VARIANT else "identity_b2",
            "checkpoint_output_path": str(args.checkpoint_path),
        },
        "training_completed": False,
        "checkpoint_recovered": True,
        "recovery_reason": args.recovery_reason,
        "gate_a_passed": True,
        "gate_b_passed": bool(loaded_config.get("gate_b_overfit64_passed", False)),
        "gate_b_scope": "overfit64_preflight",
        "gate_b_overfit64_passed": bool(loaded_config.get("gate_b_overfit64_passed", False)),
        "gate_b_overfit64_rmse_pu": loaded_config.get("gate_b_overfit64_rmse_pu"),
        "gate_b_overfit64_mae_pu": loaded_config.get("gate_b_overfit64_mae_pu"),
        "gate_b_overfit64_source": loaded_config.get("gate_b_overfit64_source"),
        "train_gate_after_fit_passed": bool(train_gate_passed),
        "train_gate_after_fit_rmse_pu": float(train_gate_metrics["rmse_pu"]),
        "train_gate_after_fit_mae_pu": float(train_gate_metrics["mae_pu"]),
        "gate_c_passed": bool(gate_c_passed),
        "gate_c_metrics": gate_c_metrics,
        "command": sys.argv,
    }
    _write_json(manifest_path, manifest)
    _write_csv(history_path, history)
    print(str(manifest_path))


def _eval_shard(args: argparse.Namespace) -> None:
    if args.split_name == "test" and not args.allow_test:
        raise SystemExit("--allow-test is required for test split DGCRN shard evaluation.")
    manifest = json.loads(args.checkpoint_manifest.read_text(encoding="utf-8"))
    checkpoint_path = Path(manifest["checkpoint_path"])
    if not checkpoint_path.is_absolute():
        checkpoint_path = args.checkpoint_manifest.parent / checkpoint_path
    prepared = formal._prepare_dataset(args.dataset_id, max_train_origins=None, max_eval_origins=None)
    all_windows = _windows_for_split(prepared, split_name=args.split_name, eval_protocol=args.eval_protocol)
    shard_start, shard_stop = resolve_shard_bounds(total_window_count=len(all_windows), start=args.start, stop=args.stop)
    windows = formal._slice_windows(all_windows, shard_start, shard_stop)
    model, config, _payload, resolved_device = load_model_from_checkpoint(prepared, checkpoint_path, device=args.device)
    predictions = formal._evaluate_dgcrn(
        model,
        prepared,
        windows,
        variant_name=config["variant_name"],
        device=resolved_device,
        batch_size=args.eval_batch_size,
        residual_anchor_steps=int(config["residual_anchor_steps"]),
    )
    targets, valid = formal._target_and_valid(prepared, windows)
    origin_rows = formal._origin_error_rows(
        _spec_for_variant(config["variant_name"]),
        prepared=prepared,
        seed=int(config["seed"]),
        split_name=args.split_name,
        eval_protocol=args.eval_protocol,
        windows=windows,
        predictions=predictions,
        trial_id=_trial_id(config),
        search_config_id=_search_config_id(config),
        residual_anchor_steps=int(config["residual_anchor_steps"]),
        best_trial=True,
    )
    for offset, row in enumerate(origin_rows):
        row["origin_index"] = shard_start + offset
        row["window_count"] = len(all_windows)
        row["shard_start"] = shard_start
        row["shard_stop"] = shard_stop
    record, exact_abs_errors = build_shard_record_from_arrays(
        predictions,
        targets,
        valid,
        dataset_id=args.dataset_id,
        split_name=args.split_name,
        eval_protocol=args.eval_protocol,
        variant_name=config["variant_name"],
        frozen_config=config,
        seed=int(config["seed"]),
        shard_start=shard_start,
        shard_stop=shard_stop,
        total_window_count=len(all_windows),
        rated_power_kw=prepared.rated_power_kw,
        forecast_steps=prepared.forecast_steps,
        node_count=prepared.node_count,
        exact_abs_error_limit=args.exact_abs_error_limit,
        checkpoint_manifest=manifest,
    )
    paths = write_shard_artifacts(record, exact_abs_errors, origin_rows, args.output_dir)
    print(str(paths["json"]))


def _aggregate(args: argparse.Namespace) -> None:
    if args.split_name == "test" and not args.allow_test:
        raise SystemExit("--allow-test is required for test split DGCRN aggregate.")
    pattern = f"{_artifact_stem(args.dataset_id, args.split_name, args.eval_protocol, args.variant, args.seed)}_shard_*.json"
    records = []
    for path in sorted(args.output_dir.glob(pattern)):
        record = json.loads(path.read_text(encoding="utf-8"))
        record["_json_path"] = str(path)
        records.append(record)
    aggregate, origin_rows = aggregate_shard_records(
        records,
        dataset_id=args.dataset_id,
        split_name=args.split_name,
        eval_protocol=args.eval_protocol,
        variant_name=args.variant,
        seed=args.seed,
        exact_abs_error_limit=args.exact_abs_error_limit,
        base_dir=args.output_dir,
    )
    paths = _write_aggregate_artifacts(
        aggregate,
        origin_rows,
        args.output_dir,
        bootstrap_repeats=args.bootstrap_repeats,
        bootstrap_seed=args.bootstrap_seed,
        block_length=args.block_length,
    )
    print(str(paths["json"]))


def planned_shard_intervals(total_window_count: int, shard_size: int) -> list[tuple[int, int]]:
    if int(total_window_count) <= 0:
        raise ValueError("total_window_count must be positive.")
    if int(shard_size) <= 0:
        raise ValueError("shard_size must be positive.")
    return [
        (start, min(start + int(shard_size), int(total_window_count)))
        for start in range(0, int(total_window_count), int(shard_size))
    ]


def _shard_artifacts_complete(
    output_dir: Path,
    *,
    dataset_id: str,
    split_name: str,
    eval_protocol: str,
    variant_name: str,
    seed: int,
    shard_start: int,
    shard_stop: int,
) -> bool:
    paths = shard_artifact_paths(
        output_dir,
        dataset_id=dataset_id,
        split_name=split_name,
        eval_protocol=eval_protocol,
        variant_name=variant_name,
        seed=seed,
        shard_start=shard_start,
        shard_stop=shard_stop,
    )
    if not all(path.exists() for path in (paths["json"], paths["abs_errors"], paths["origin_errors"])):
        return False
    try:
        record = json.loads(paths["json"].read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return False
    return record.get("status") == "complete" and int(record.get("shard_start", -1)) == int(shard_start) and int(record.get("shard_stop", -1)) == int(shard_stop)


def _run_subprocess(command: Sequence[str], *, log_path: Path, cwd: Path) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log_handle:
        log_handle.write(f"$ {shlex.join(command)}\n\n")
        log_handle.flush()
        completed = subprocess.run(
            list(command),
            cwd=cwd,
            text=True,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            check=False,
        )
    return int(completed.returncode)


def _status_record(
    *,
    command: Sequence[str],
    log_path: Path,
    returncode: int,
    status: str,
    started_at: str,
    duration_seconds: float,
    shard_start: int | None = None,
    shard_stop: int | None = None,
    attempt: int | None = None,
    shard_size: int | None = None,
) -> dict[str, Any]:
    signal_number = abs(returncode) if returncode < 0 else None
    return {
        "created_at": datetime.now(tz=UTC).isoformat(),
        "started_at": started_at,
        "duration_seconds": float(duration_seconds),
        "status": status,
        "returncode": int(returncode),
        "signal_number": signal_number,
        "command": list(command),
        "log_path": str(log_path),
        "shard_start": shard_start,
        "shard_stop": shard_stop,
        "attempt": attempt,
        "shard_size": shard_size,
    }


def _append_jsonl(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, default=_json_default, sort_keys=True) + "\n")


def _run_eval_interval_with_retry(
    *,
    args: argparse.Namespace,
    manifest: dict[str, Any],
    variant_name: str,
    seed: int,
    start: int,
    stop: int,
    shard_sizes: Sequence[int],
    size_index: int,
    status_path: Path,
    repo_root: Path,
) -> None:
    shard_size = int(shard_sizes[size_index])
    if _shard_artifacts_complete(
        args.output_dir,
        dataset_id=args.dataset_id,
        split_name=args.split_name,
        eval_protocol=args.eval_protocol,
        variant_name=variant_name,
        seed=seed,
        shard_start=start,
        shard_stop=stop,
    ):
        _append_jsonl(
            status_path,
            {
                "created_at": datetime.now(tz=UTC).isoformat(),
                "status": "skipped_existing_artifacts",
                "shard_start": int(start),
                "shard_stop": int(stop),
                "shard_size": shard_size,
            },
        )
        return

    attempts = int(args.retry_attempts)
    if args.eval_protocol == NON_OVERLAP_PROTOCOL and size_index == 0:
        attempts = 1
    for attempt in range(1, attempts + 1):
        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            "eval-shard",
            "--dataset-id",
            args.dataset_id,
            "--checkpoint-manifest",
            str(args.checkpoint_manifest),
            "--output-dir",
            str(args.output_dir),
            "--split-name",
            args.split_name,
            "--eval-protocol",
            args.eval_protocol,
            "--start",
            str(start),
            "--stop",
            str(stop),
            "--device",
            args.device,
            "--eval-batch-size",
            str(args.eval_batch_size),
            "--exact-abs-error-limit",
            str(args.exact_abs_error_limit),
        ]
        if args.split_name == "test":
            command.append("--allow-test")
        log_path = (
            args.output_dir
            / "logs"
            / f"dgcrn_{args.split_name}_{_safe_protocol_name(args.eval_protocol)}_seed{seed}_shard_{start:06d}_{stop:06d}_attempt{attempt}.log"
        )
        started_at = datetime.now(tz=UTC).isoformat()
        started = time.perf_counter()
        returncode = _run_subprocess(command, log_path=log_path, cwd=repo_root)
        duration = time.perf_counter() - started
        if returncode == 0 and _shard_artifacts_complete(
            args.output_dir,
            dataset_id=args.dataset_id,
            split_name=args.split_name,
            eval_protocol=args.eval_protocol,
            variant_name=variant_name,
            seed=seed,
            shard_start=start,
            shard_stop=stop,
        ):
            _append_jsonl(
                status_path,
                _status_record(
                    command=command,
                    log_path=log_path,
                    returncode=returncode,
                    status="completed",
                    started_at=started_at,
                    duration_seconds=duration,
                    shard_start=start,
                    shard_stop=stop,
                    attempt=attempt,
                    shard_size=shard_size,
                ),
            )
            return
        _append_jsonl(
            status_path,
            _status_record(
                command=command,
                log_path=log_path,
                returncode=returncode,
                status="failed",
                started_at=started_at,
                duration_seconds=duration,
                shard_start=start,
                shard_stop=stop,
                attempt=attempt,
                shard_size=shard_size,
            ),
        )

    if size_index + 1 >= len(shard_sizes):
        raise RuntimeError(f"DGCRN shard [{start}, {stop}) failed at minimum shard size {shard_size}.")
    next_size = int(shard_sizes[size_index + 1])
    for child_start, child_stop in planned_shard_intervals(stop - start, next_size):
        _run_eval_interval_with_retry(
            args=args,
            manifest=manifest,
            variant_name=variant_name,
            seed=seed,
            start=start + child_start,
            stop=start + child_stop,
            shard_sizes=shard_sizes,
            size_index=size_index + 1,
            status_path=status_path,
            repo_root=repo_root,
        )


def _run_test_shards(args: argparse.Namespace) -> None:
    if args.split_name == "test" and not args.allow_test:
        raise SystemExit("--allow-test is required for test split DGCRN shard driver.")
    manifest = json.loads(args.checkpoint_manifest.read_text(encoding="utf-8"))
    config = dict(manifest["frozen_config"])
    variant_name = str(config["variant_name"])
    seed = int(config["seed"])
    prepared = formal._prepare_dataset(args.dataset_id, max_train_origins=None, max_eval_origins=None)
    total_count = len(_windows_for_split(prepared, split_name=args.split_name, eval_protocol=args.eval_protocol))
    if args.eval_protocol == ROLLING_PROTOCOL:
        shard_sizes = (int(args.rolling_shard_size), *tuple(int(value) for value in args.rolling_fallback_shard_sizes))
    else:
        shard_sizes = (int(args.non_overlap_shard_size), *tuple(int(value) for value in args.non_overlap_fallback_shard_sizes))
    if any(size <= 0 for size in shard_sizes):
        raise ValueError("All shard sizes must be positive.")
    status_path = (
        args.output_dir
        / f"dgcrn_{args.dataset_id}_{variant_name}_{args.split_name}_{_safe_protocol_name(args.eval_protocol)}_seed{seed}_shard_driver_status.jsonl"
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    _append_jsonl(
        status_path,
        {
            "created_at": datetime.now(tz=UTC).isoformat(),
            "status": "driver_started",
            "dataset_id": args.dataset_id,
            "model_variant": variant_name,
            "seed": seed,
            "split_name": args.split_name,
            "eval_protocol": args.eval_protocol,
            "total_window_count": total_count,
            "shard_sizes": list(shard_sizes),
            "selected_by": "validation_only",
            "no_test_feedback": True,
        },
    )
    repo_root = FAMILY_DIR.parents[2]
    for start, stop in planned_shard_intervals(total_count, shard_sizes[0]):
        _run_eval_interval_with_retry(
            args=args,
            manifest=manifest,
            variant_name=variant_name,
            seed=seed,
            start=start,
            stop=stop,
            shard_sizes=shard_sizes,
            size_index=0,
            status_path=status_path,
            repo_root=repo_root,
        )
    aggregate_command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "aggregate",
        "--dataset-id",
        args.dataset_id,
        "--variant",
        variant_name,
        "--seed",
        str(seed),
        "--output-dir",
        str(args.output_dir),
        "--split-name",
        args.split_name,
        "--eval-protocol",
        args.eval_protocol,
        "--exact-abs-error-limit",
        str(args.exact_abs_error_limit),
        "--bootstrap-repeats",
        str(args.bootstrap_repeats),
        "--bootstrap-seed",
        str(args.bootstrap_seed),
        "--block-length",
        str(args.block_length),
    ]
    if args.split_name == "test":
        aggregate_command.append("--allow-test")
    aggregate_log = (
        args.output_dir
        / "logs"
        / f"dgcrn_{args.split_name}_{_safe_protocol_name(args.eval_protocol)}_seed{seed}_aggregate.log"
    )
    started_at = datetime.now(tz=UTC).isoformat()
    started = time.perf_counter()
    returncode = _run_subprocess(aggregate_command, log_path=aggregate_log, cwd=repo_root)
    aggregate_paths = aggregate_artifact_paths(
        args.output_dir,
        dataset_id=args.dataset_id,
        split_name=args.split_name,
        eval_protocol=args.eval_protocol,
        variant_name=variant_name,
        seed=seed,
    )
    aggregate_complete = returncode == 0 and all(
        path.exists()
        for path in (
            aggregate_paths["json"],
            aggregate_paths["csv"],
            aggregate_paths["origin_errors"],
            aggregate_paths["bootstrap_json"],
        )
    )
    _append_jsonl(
        status_path,
        _status_record(
            command=aggregate_command,
            log_path=aggregate_log,
            returncode=returncode,
            status="aggregate_completed" if aggregate_complete else "aggregate_failed",
            started_at=started_at,
            duration_seconds=time.perf_counter() - started,
        ),
    )
    if not aggregate_complete:
        raise RuntimeError(f"DGCRN aggregate failed for {args.split_name}/{args.eval_protocol}; see {aggregate_log}.")
    _append_jsonl(
        status_path,
        {
            "created_at": datetime.now(tz=UTC).isoformat(),
            "status": "driver_completed",
            "aggregate_json": str(aggregate_paths["json"]),
        },
    )
    print(str(aggregate_paths["json"]))


def _checkpoint_manifest_complete(path: Path) -> bool:
    if not path.exists():
        return False
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return False
    return manifest.get("status") in {"complete", "recovered_from_checkpoint"} and manifest.get("selected_by") == "validation_only"


def _aggregate_complete(
    output_dir: Path,
    *,
    dataset_id: str,
    split_name: str,
    eval_protocol: str,
    variant_name: str,
    seed: int,
) -> bool:
    paths = aggregate_artifact_paths(
        output_dir,
        dataset_id=dataset_id,
        split_name=split_name,
        eval_protocol=eval_protocol,
        variant_name=variant_name,
        seed=seed,
    )
    if not all(path.exists() for path in (paths["json"], paths["csv"], paths["origin_errors"], paths["bootstrap_json"], paths["bootstrap_csv"])):
        return False
    try:
        aggregate = json.loads(paths["json"].read_text(encoding="utf-8"))
        bootstrap = json.loads(paths["bootstrap_json"].read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return False
    expected_count = 94458 if split_name == "test" and eval_protocol == ROLLING_PROTOCOL else 2624 if split_name == "test" and eval_protocol == NON_OVERLAP_PROTOCOL else None
    if expected_count is not None and int(aggregate.get("metrics", {}).get("window_count", -1)) != expected_count:
        return False
    return aggregate.get("status") == "complete" and bootstrap.get("bootstrap_status") == "completed"


def _run_seed_driver(args: argparse.Namespace, *, seed: int) -> None:
    repo_root = FAMILY_DIR.parents[2]
    config = dgcrn_frozen_config(
        variant_name=args.variant,
        seed=seed,
        train_batch_size=args.train_batch_size,
        max_epochs=args.max_epochs,
        learning_rate=args.learning_rate,
        hidden_dim=args.dgcrn_hidden_dim,
        dropout=args.dgcrn_dropout,
        gcn_depth=args.dgcrn_gcn_depth,
        residual_anchor_steps=args.residual_anchor_steps,
        checkpoint_eval_protocol=args.checkpoint_eval_protocol,
        gate_b_overfit64_source=args.gate_b_overfit64_source,
        gate_b_overfit64_rmse_pu=args.gate_b_overfit64_rmse_pu,
        gate_b_overfit64_mae_pu=args.gate_b_overfit64_mae_pu,
    )
    paths = checkpoint_artifact_paths(args.output_dir, dataset_id=args.dataset_id, seed=seed, variant_name=args.variant)
    status_path = args.output_dir / f"dgcrn_{args.dataset_id}_{args.variant}_seed{seed}_rescue_driver_status.jsonl"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    _append_jsonl(
        status_path,
        {
            "created_at": datetime.now(tz=UTC).isoformat(),
            "status": "seed_driver_started",
            "dataset_id": args.dataset_id,
            "model_variant": args.variant,
            "seed": seed,
            "frozen_config": config,
            "selected_by": "validation_only",
            "no_test_feedback": True,
        },
    )
    if not _checkpoint_manifest_complete(paths["manifest"]):
        train_batch_sizes = [int(args.train_batch_size)]
        if args.fallback_train_batch_size and int(args.fallback_train_batch_size) not in train_batch_sizes:
            train_batch_sizes.append(int(args.fallback_train_batch_size))
        for index, batch_size in enumerate(train_batch_sizes):
            command = [
                sys.executable,
                str(Path(__file__).resolve()),
                "train-checkpoint",
                "--output-dir",
                str(args.output_dir),
                "--dataset-id",
                args.dataset_id,
                "--variant",
                args.variant,
                "--seed",
                str(seed),
                "--device",
                args.device,
                "--train-batch-size",
                str(batch_size),
                "--max-epochs",
                str(args.max_epochs),
                "--learning-rate",
                str(args.learning_rate),
                "--dgcrn-hidden-dim",
                str(args.dgcrn_hidden_dim),
                "--dgcrn-dropout",
                str(args.dgcrn_dropout),
                "--dgcrn-gcn-depth",
                str(args.dgcrn_gcn_depth),
                "--residual-anchor-steps",
                str(args.residual_anchor_steps),
                "--checkpoint-eval-protocol",
                args.checkpoint_eval_protocol,
                "--gate-origin-count",
                str(args.gate_origin_count),
                "--gate-b-overfit64-source",
                args.gate_b_overfit64_source,
                "--gate-b-overfit64-rmse-pu",
                str(args.gate_b_overfit64_rmse_pu),
                "--gate-b-overfit64-mae-pu",
                str(args.gate_b_overfit64_mae_pu),
            ]
            log_path = args.output_dir / "logs" / f"dgcrn_seed{seed}_train_batch{batch_size}.log"
            started_at = datetime.now(tz=UTC).isoformat()
            started = time.perf_counter()
            returncode = _run_subprocess(command, log_path=log_path, cwd=repo_root)
            _append_jsonl(
                status_path,
                _status_record(
                    command=command,
                    log_path=log_path,
                    returncode=returncode,
                    status="train_completed" if returncode == 0 and _checkpoint_manifest_complete(paths["manifest"]) else "train_failed",
                    started_at=started_at,
                    duration_seconds=time.perf_counter() - started,
                ),
            )
            if _checkpoint_manifest_complete(paths["manifest"]):
                break
            if paths["checkpoint"].exists():
                recover_command = [
                    sys.executable,
                    str(Path(__file__).resolve()),
                    "recover-manifest",
                    "--dataset-id",
                    args.dataset_id,
                    "--checkpoint-path",
                    str(paths["checkpoint"]),
                    "--manifest-output",
                    str(paths["manifest"]),
                    "--history-output",
                    str(paths["history"]),
                    "--device",
                    args.device,
                    "--train-batch-size",
                    str(batch_size),
                    "--checkpoint-eval-protocol",
                    args.checkpoint_eval_protocol,
                    "--gate-origin-count",
                    str(args.gate_origin_count),
                    "--recovery-reason",
                    f"seed_driver_train_returncode_{returncode}",
                ]
                recover_log = args.output_dir / "logs" / f"dgcrn_seed{seed}_recover_manifest.log"
                recover_started_at = datetime.now(tz=UTC).isoformat()
                recover_started = time.perf_counter()
                recover_returncode = _run_subprocess(recover_command, log_path=recover_log, cwd=repo_root)
                _append_jsonl(
                    status_path,
                    _status_record(
                        command=recover_command,
                        log_path=recover_log,
                        returncode=recover_returncode,
                        status="recover_manifest_completed" if recover_returncode == 0 and _checkpoint_manifest_complete(paths["manifest"]) else "recover_manifest_failed",
                        started_at=recover_started_at,
                        duration_seconds=time.perf_counter() - recover_started,
                    ),
                )
                break
            if returncode == 0 or index + 1 >= len(train_batch_sizes):
                break
        if not _checkpoint_manifest_complete(paths["manifest"]):
            raise RuntimeError(f"DGCRN seed {seed} did not produce a usable checkpoint manifest.")
    else:
        _append_jsonl(
            status_path,
            {
                "created_at": datetime.now(tz=UTC).isoformat(),
                "status": "skipped_existing_checkpoint_manifest",
                "manifest_path": str(paths["manifest"]),
            },
        )
    for eval_protocol in (ROLLING_PROTOCOL, NON_OVERLAP_PROTOCOL):
        if _aggregate_complete(
            args.output_dir,
            dataset_id=args.dataset_id,
            split_name="test",
            eval_protocol=eval_protocol,
            variant_name=args.variant,
            seed=seed,
        ):
            _append_jsonl(
                status_path,
                {
                    "created_at": datetime.now(tz=UTC).isoformat(),
                    "status": "skipped_existing_test_aggregate",
                    "split_name": "test",
                    "eval_protocol": eval_protocol,
                },
            )
            continue
        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            "run-test-shards",
            "--dataset-id",
            args.dataset_id,
            "--checkpoint-manifest",
            str(paths["manifest"]),
            "--output-dir",
            str(args.output_dir),
            "--split-name",
            "test",
            "--eval-protocol",
            eval_protocol,
            "--device",
            args.device,
            "--eval-batch-size",
            str(args.eval_batch_size),
            "--exact-abs-error-limit",
            str(args.exact_abs_error_limit),
            "--rolling-shard-size",
            str(args.rolling_shard_size),
            "--non-overlap-shard-size",
            str(args.non_overlap_shard_size),
            "--retry-attempts",
            str(args.retry_attempts),
            "--bootstrap-repeats",
            str(args.bootstrap_repeats),
            "--bootstrap-seed",
            str(args.bootstrap_seed),
            "--block-length",
            str(args.block_length),
            "--allow-test",
        ]
        if args.rolling_fallback_shard_sizes:
            command.append("--rolling-fallback-shard-sizes")
            command.extend(str(fallback_size) for fallback_size in args.rolling_fallback_shard_sizes)
        if args.non_overlap_fallback_shard_sizes:
            command.append("--non-overlap-fallback-shard-sizes")
            command.extend(str(fallback_size) for fallback_size in args.non_overlap_fallback_shard_sizes)
        log_path = args.output_dir / "logs" / f"dgcrn_seed{seed}_test_{_safe_protocol_name(eval_protocol)}_driver.log"
        started_at = datetime.now(tz=UTC).isoformat()
        started = time.perf_counter()
        returncode = _run_subprocess(command, log_path=log_path, cwd=repo_root)
        complete = _aggregate_complete(
            args.output_dir,
            dataset_id=args.dataset_id,
            split_name="test",
            eval_protocol=eval_protocol,
            variant_name=args.variant,
            seed=seed,
        )
        _append_jsonl(
            status_path,
            _status_record(
                command=command,
                log_path=log_path,
                returncode=returncode,
                status=f"test_{_safe_protocol_name(eval_protocol)}_completed" if returncode == 0 and complete else f"test_{_safe_protocol_name(eval_protocol)}_failed",
                started_at=started_at,
                duration_seconds=time.perf_counter() - started,
            ),
        )
        if returncode != 0 or not complete:
            raise RuntimeError(f"DGCRN seed {seed} test {eval_protocol} failed; see {log_path}.")
    _append_jsonl(
        status_path,
        {
            "created_at": datetime.now(tz=UTC).isoformat(),
            "status": "seed_driver_completed",
            "seed": seed,
        },
    )


def _run_seeds(args: argparse.Namespace) -> None:
    for seed in args.seeds:
        _run_seed_driver(args, seed=int(seed))
    print("completed_seeds=" + ",".join(str(seed) for seed in args.seeds))


def _add_config_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--dataset-id", default="kelmarsh")
    parser.add_argument("--variant", default=formal.DGCRN_RESIDUAL_VARIANT, choices=[formal.DGCRN_RESIDUAL_VARIANT, formal.DGCRN_GEOMETRY_RESIDUAL_VARIANT, formal.DGCRN_DIRECT_VARIANT])
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--train-batch-size", type=int, default=128)
    parser.add_argument("--max-epochs", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=5e-4)
    parser.add_argument("--dgcrn-hidden-dim", type=int, default=96)
    parser.add_argument("--dgcrn-dropout", type=float, default=0.0)
    parser.add_argument("--dgcrn-gcn-depth", type=int, default=3)
    parser.add_argument("--residual-anchor-steps", type=int, default=1)
    parser.add_argument("--checkpoint-eval-protocol", default=ROLLING_PROTOCOL, choices=[ROLLING_PROTOCOL, NON_OVERLAP_PROTOCOL])
    parser.add_argument("--max-train-origins", type=int, default=None)
    parser.add_argument("--max-checkpoint-origins", type=int, default=None)
    parser.add_argument("--gate-origin-count", type=int, default=64)
    parser.add_argument("--gate-b-overfit64-source", default=DEFAULT_GATE_B_OVERFIT64_SOURCE)
    parser.add_argument("--gate-b-overfit64-rmse-pu", type=float, default=DEFAULT_GATE_B_OVERFIT64_RMSE)
    parser.add_argument("--gate-b-overfit64-mae-pu", type=float, default=DEFAULT_GATE_B_OVERFIT64_MAE)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Recoverable DGCRN official-core checkpoint and shard evaluator.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train = subparsers.add_parser("train-checkpoint")
    _add_config_args(train)
    train.add_argument("--output-dir", type=Path, required=True)
    train.set_defaults(func=_train_checkpoint)

    recover = subparsers.add_parser("recover-manifest")
    recover.add_argument("--dataset-id", default="kelmarsh")
    recover.add_argument("--checkpoint-path", type=Path, required=True)
    recover.add_argument("--manifest-output", type=Path)
    recover.add_argument("--history-output", type=Path)
    recover.add_argument("--device", default="cuda")
    recover.add_argument("--train-batch-size", type=int, default=128)
    recover.add_argument("--checkpoint-eval-protocol", default=ROLLING_PROTOCOL, choices=[ROLLING_PROTOCOL, NON_OVERLAP_PROTOCOL])
    recover.add_argument("--gate-origin-count", type=int, default=64)
    recover.add_argument("--recovery-reason", default="train_process_exited_before_final_manifest")
    recover.set_defaults(func=_recover_manifest)

    eval_parser = subparsers.add_parser("eval-shard")
    eval_parser.add_argument("--dataset-id", default="kelmarsh")
    eval_parser.add_argument("--checkpoint-manifest", type=Path, required=True)
    eval_parser.add_argument("--output-dir", type=Path, required=True)
    eval_parser.add_argument("--split-name", choices=["val", "test"], required=True)
    eval_parser.add_argument("--eval-protocol", choices=[ROLLING_PROTOCOL, NON_OVERLAP_PROTOCOL], required=True)
    eval_parser.add_argument("--start", type=int, required=True)
    eval_parser.add_argument("--stop", type=int, required=True)
    eval_parser.add_argument("--device", default="cuda")
    eval_parser.add_argument("--eval-batch-size", type=int, default=128)
    eval_parser.add_argument("--exact-abs-error-limit", type=int, default=DEFAULT_EXACT_AE_LIMIT)
    eval_parser.add_argument("--allow-test", action="store_true")
    eval_parser.set_defaults(func=_eval_shard)

    aggregate = subparsers.add_parser("aggregate")
    aggregate.add_argument("--dataset-id", default="kelmarsh")
    aggregate.add_argument("--variant", default=formal.DGCRN_RESIDUAL_VARIANT)
    aggregate.add_argument("--seed", type=int, default=3407)
    aggregate.add_argument("--output-dir", type=Path, required=True)
    aggregate.add_argument("--split-name", choices=["val", "test"], required=True)
    aggregate.add_argument("--eval-protocol", choices=[ROLLING_PROTOCOL, NON_OVERLAP_PROTOCOL], required=True)
    aggregate.add_argument("--exact-abs-error-limit", type=int, default=DEFAULT_EXACT_AE_LIMIT)
    aggregate.add_argument("--bootstrap-repeats", type=int, default=DEFAULT_BOOTSTRAP_REPEATS)
    aggregate.add_argument("--bootstrap-seed", type=int, default=3407)
    aggregate.add_argument("--block-length", type=int, default=24)
    aggregate.add_argument("--allow-test", action="store_true")
    aggregate.set_defaults(func=_aggregate)

    driver = subparsers.add_parser("run-test-shards")
    driver.add_argument("--dataset-id", default="kelmarsh")
    driver.add_argument("--checkpoint-manifest", type=Path, required=True)
    driver.add_argument("--output-dir", type=Path, required=True)
    driver.add_argument("--split-name", choices=["val", "test"], required=True)
    driver.add_argument("--eval-protocol", choices=[ROLLING_PROTOCOL, NON_OVERLAP_PROTOCOL], required=True)
    driver.add_argument("--device", default="cuda")
    driver.add_argument("--eval-batch-size", type=int, default=128)
    driver.add_argument("--exact-abs-error-limit", type=int, default=DEFAULT_EXACT_AE_LIMIT)
    driver.add_argument("--rolling-shard-size", type=int, default=DEFAULT_ROLLING_SHARD_SIZE)
    driver.add_argument("--rolling-fallback-shard-sizes", type=int, nargs="*", default=list(DEFAULT_ROLLING_FALLBACK_SHARD_SIZES))
    driver.add_argument("--non-overlap-shard-size", type=int, default=DEFAULT_NON_OVERLAP_SHARD_SIZE)
    driver.add_argument("--non-overlap-fallback-shard-sizes", type=int, nargs="*", default=list(DEFAULT_NON_OVERLAP_FALLBACK_SHARD_SIZES))
    driver.add_argument("--retry-attempts", type=int, default=2)
    driver.add_argument("--bootstrap-repeats", type=int, default=DEFAULT_BOOTSTRAP_REPEATS)
    driver.add_argument("--bootstrap-seed", type=int, default=3407)
    driver.add_argument("--block-length", type=int, default=24)
    driver.add_argument("--allow-test", action="store_true")
    driver.set_defaults(func=_run_test_shards)

    seed_driver = subparsers.add_parser("run-seeds")
    _add_config_args(seed_driver)
    seed_driver.add_argument("--seeds", type=int, nargs="+", required=True)
    seed_driver.add_argument("--output-dir", type=Path, required=True)
    seed_driver.add_argument("--fallback-train-batch-size", type=int, default=64)
    seed_driver.add_argument("--eval-batch-size", type=int, default=128)
    seed_driver.add_argument("--exact-abs-error-limit", type=int, default=DEFAULT_EXACT_AE_LIMIT)
    seed_driver.add_argument("--rolling-shard-size", type=int, default=DEFAULT_ROLLING_SHARD_SIZE)
    seed_driver.add_argument("--rolling-fallback-shard-sizes", type=int, nargs="*", default=list(DEFAULT_ROLLING_FALLBACK_SHARD_SIZES))
    seed_driver.add_argument("--non-overlap-shard-size", type=int, default=DEFAULT_NON_OVERLAP_SHARD_SIZE)
    seed_driver.add_argument("--non-overlap-fallback-shard-sizes", type=int, nargs="*", default=list(DEFAULT_NON_OVERLAP_FALLBACK_SHARD_SIZES))
    seed_driver.add_argument("--retry-attempts", type=int, default=2)
    seed_driver.add_argument("--bootstrap-repeats", type=int, default=DEFAULT_BOOTSTRAP_REPEATS)
    seed_driver.add_argument("--bootstrap-seed", type=int, default=3407)
    seed_driver.add_argument("--block-length", type=int, default=24)
    seed_driver.set_defaults(func=_run_seeds)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    args.func(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
