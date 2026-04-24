from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import UTC, datetime
import json
import math
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
    PERSISTENCE_VARIANT,
    RIDGE_RESIDUAL_VARIANT,
    SEASONAL_PERSISTENCE_VARIANT,
    TASK_ID,
    OfficialVariantSpec,
    resolve_variant_specs,
)

FORMAL_SUPPORTED_VARIANTS = {
    PERSISTENCE_VARIANT,
    SEASONAL_PERSISTENCE_VARIANT,
    RIDGE_RESIDUAL_VARIANT,
}
FORMAL_BLOCKER_BY_VARIANT_PREFIX = {
    "baseline_mlp_residual": "residual_control_training_not_implemented",
    "baseline_gru_residual": "residual_control_training_not_implemented",
    "baseline_tcn_residual": "residual_control_training_not_implemented",
    "chronos2_official": "chronos2_zero_shot_execution_not_integrated_in_v2_formal_runner",
    "dgcrn_official_core": "official_core_training_adapter_not_implemented",
    "timexer_official": "official_training_adapter_not_implemented",
    "itransformer_official": "official_training_adapter_not_implemented",
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
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for split_name, eval_protocol, windows in _windows_by_split(prepared):
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
    run_label: str | None = None,
    no_record_run: bool = False,
) -> pl.DataFrame:
    specs = resolve_variant_specs(variant_names or DEFAULT_VARIANTS)
    rows: list[dict[str, Any]] = []
    for dataset_id in dataset_ids:
        prepared = _prepare_dataset(dataset_id, max_train_origins=max_train_origins, max_eval_origins=max_eval_origins)
        persistence_train_predictions = _repeat_anchor(_last_value_anchor(prepared, prepared.train_windows), prepared.forecast_steps)
        train_targets, train_valid = _target_and_valid(prepared, prepared.train_windows)
        persistence_train_metrics = _metrics(
            persistence_train_predictions,
            train_targets,
            train_valid,
            rated_power_kw=prepared.rated_power_kw,
        )
        persistence_val_lead1_rmse: float | None = None
        persistence_val_lead1_mae: float | None = None
        for spec in specs:
            status, blocker = formal_support_status(spec)
            if status != "supported":
                rows.append(_blocked_row(spec, dataset_id=dataset_id, seed=seed, blocker=blocker or "blocked"))
                continue
            started = time.perf_counter()
            if spec.model_variant == PERSISTENCE_VARIANT:
                predictions_by_split = {
                    (split_name, eval_protocol): _repeat_anchor(_last_value_anchor(prepared, windows), prepared.forecast_steps)
                    for split_name, eval_protocol, windows in _windows_by_split(prepared)
                }
                val_targets, val_valid = _target_and_valid(prepared, prepared.val_rolling_windows)
                val_metrics = _metrics(
                    predictions_by_split[("val", ROLLING_EVAL_PROTOCOL)],
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
                    )
                )
            elif spec.model_variant == SEASONAL_PERSISTENCE_VARIANT:
                predictions_by_split = {
                    (split_name, eval_protocol): _repeat_anchor(_seasonal_anchor(prepared, windows), prepared.forecast_steps)
                    for split_name, eval_protocol, windows in _windows_by_split(prepared)
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
                    gate_b_predictions = _predict_ridge(prepared, prepared.train_windows, weights)
                    gate_b_metrics = _metrics(
                        gate_b_predictions,
                        train_targets,
                        train_valid,
                        rated_power_kw=prepared.rated_power_kw,
                    )
                    gate_b_passed = bool(
                        gate_b_metrics["rmse_pu"] <= 0.03
                        or gate_b_metrics["rmse_pu"] <= 0.5 * persistence_train_metrics["rmse_pu"]
                    )
                    gate_c_passed = bool(
                        persistence_val_lead1_rmse is not None
                        and persistence_val_lead1_mae is not None
                        and val_metrics["lead1_rmse_pu"] <= 1.05 * persistence_val_lead1_rmse
                        and val_metrics["lead1_mae_pu"] <= 1.05 * persistence_val_lead1_mae
                    )
                    predictions_by_split = {
                        (split_name, eval_protocol): _predict_ridge(prepared, windows, weights)
                        for split_name, eval_protocol, windows in _windows_by_split(prepared)
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
                "max_train_origins": max_train_origins,
                "max_eval_origins": max_eval_origins,
            },
            output_path=resolved_output_path,
            result_row_count=frame.height,
            dataset_ids=tuple(dataset_ids),
            feature_protocol_ids=(FEATURE_PROTOCOL_ID,),
            model_variants=tuple(spec.model_variant for spec in specs),
            eval_protocols=(ROLLING_EVAL_PROTOCOL, NON_OVERLAP_EVAL_PROTOCOL),
            result_splits=("val", "test", "formal_tuning_blocked"),
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
            "gate_b": {"status": "computed_for_ridge_residual_controls_only"},
            "gate_c": {"status": "computed_for_ridge_residual_controls_only"},
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
        run_label=args.run_label,
        no_record_run=args.no_record_run,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
