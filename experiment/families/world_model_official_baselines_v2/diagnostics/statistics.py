from __future__ import annotations

import math
from collections import OrderedDict
from collections.abc import Iterable, Mapping

import numpy as np

SUMMARY_GROUP_COLUMNS = (
    "dataset_id",
    "model_id",
    "model_variant",
    "task_id",
    "split_name",
    "eval_protocol",
    "metric_scope",
    "feature_budget_id",
    "output_parameterization",
    "selection_metric",
    "selected_by",
    "no_test_feedback",
    "gate_a_passed",
    "gate_b_passed",
    "gate_b_scope",
    "gate_b_overfit64_passed",
    "gate_c_passed",
    "residual_anchor_steps",
    "formal_search_config_id",
    "is_best_validation_trial",
)
SUMMARY_METRIC_COLUMNS = (
    "mae_pu",
    "rmse_pu",
    "mae_kw",
    "rmse_kw",
    "lead1_mae_pu",
    "lead1_rmse_pu",
    "short_rmse_pu",
    "mid_rmse_pu",
    "long_rmse_pu",
    "ae_p50",
    "ae_p90",
    "ae_p95",
    "runtime_seconds",
    "window_count",
    "prediction_count",
)
PAIRED_ERROR_COLUMN_CANDIDATES = (
    ("baseline_abs_error_pu", "proposed_abs_error_pu"),
    ("baseline_ae_pu", "proposed_ae_pu"),
    ("baseline_error_pu", "proposed_error_pu"),
    ("baseline_error", "proposed_error"),
    ("control_abs_error_pu", "candidate_abs_error_pu"),
)


def paired_bootstrap_delta(
    *,
    baseline_errors: list[float] | np.ndarray,
    proposed_errors: list[float] | np.ndarray,
    repeats: int = 5000,
    seed: int = 3407,
) -> dict[str, float]:
    baseline = np.asarray(baseline_errors, dtype=np.float64)
    proposed = np.asarray(proposed_errors, dtype=np.float64)
    if baseline.shape != proposed.shape:
        raise ValueError("baseline_errors and proposed_errors must have the same shape.")
    if baseline.ndim != 1 or baseline.size == 0:
        raise ValueError("paired bootstrap expects a non-empty 1D origin-level error array.")
    rng = np.random.default_rng(seed)
    deltas = np.empty(repeats, dtype=np.float64)
    origin_count = baseline.size
    for repeat_index in range(repeats):
        indices = rng.integers(0, origin_count, size=origin_count)
        deltas[repeat_index] = float(np.mean(baseline[indices] - proposed[indices]))
    return {
        "delta_mean": float(np.mean(deltas)),
        "ci95_low": float(np.quantile(deltas, 0.025)),
        "ci95_high": float(np.quantile(deltas, 0.975)),
        "prob_delta_gt_zero": float(np.mean(deltas > 0.0)),
    }


def block_bootstrap_delta(
    *,
    baseline_errors: list[float] | np.ndarray,
    proposed_errors: list[float] | np.ndarray,
    block_length: int,
    repeats: int = 5000,
    seed: int = 3407,
) -> dict[str, float]:
    baseline = np.asarray(baseline_errors, dtype=np.float64)
    proposed = np.asarray(proposed_errors, dtype=np.float64)
    if baseline.shape != proposed.shape:
        raise ValueError("baseline_errors and proposed_errors must have the same shape.")
    if baseline.ndim != 1 or baseline.size == 0:
        raise ValueError("block bootstrap expects a non-empty 1D origin-level error array.")
    indices = block_bootstrap_indices(baseline.size, block_length=block_length, repeats=repeats, seed=seed)
    deltas = np.mean(baseline[indices] - proposed[indices], axis=1)
    return {
        "delta_mean": float(np.mean(deltas)),
        "ci95_low": float(np.quantile(deltas, 0.025)),
        "ci95_high": float(np.quantile(deltas, 0.975)),
        "prob_delta_gt_zero": float(np.mean(deltas > 0.0)),
    }


def block_bootstrap_indices(origin_count: int, *, block_length: int, repeats: int, seed: int = 3407) -> np.ndarray:
    if origin_count <= 0 or block_length <= 0 or repeats <= 0:
        raise ValueError("origin_count, block_length, and repeats must be positive.")
    rng = np.random.default_rng(seed)
    result = np.empty((repeats, origin_count), dtype=np.int64)
    max_start = max(0, origin_count - block_length)
    for repeat_index in range(repeats):
        chosen: list[int] = []
        while len(chosen) < origin_count:
            start = int(rng.integers(0, max_start + 1))
            chosen.extend(range(start, min(start + block_length, origin_count)))
        result[repeat_index] = np.asarray(chosen[:origin_count], dtype=np.int64)
    return result


def error_quantiles(abs_errors: list[float] | np.ndarray) -> dict[str, float]:
    values = np.asarray(abs_errors, dtype=np.float64)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("error_quantiles expects a non-empty 1D absolute-error array.")
    return {
        "ae_p50": float(np.round(np.quantile(values, 0.50), 12)),
        "ae_p90": float(np.round(np.quantile(values, 0.90), 12)),
        "ae_p95": float(np.round(np.quantile(values, 0.95), 12)),
    }


def aggregate_seed_rows(
    rows: Iterable[Mapping[str, str]],
    *,
    aggregation_id: str | None = None,
    aggregation_note: str | None = None,
) -> list[dict[str, object]]:
    grouped: OrderedDict[tuple[str, ...], list[Mapping[str, str]]] = OrderedDict()
    materialized = list(rows)
    for row in materialized:
        key = tuple(_string_value(row.get(column)) for column in SUMMARY_GROUP_COLUMNS if column in row)
        grouped.setdefault(key, []).append(row)

    result: list[dict[str, object]] = []
    for group_rows in grouped.values():
        first = group_rows[0]
        record: dict[str, object] = {
            column: _coerce_summary_value(first[column])
            for column in SUMMARY_GROUP_COLUMNS
            if column in first
        }
        for metric in SUMMARY_METRIC_COLUMNS:
            values = [_parse_float(row.get(metric)) for row in group_rows if _parse_float(row.get(metric)) is not None]
            if not values:
                continue
            mean_value = float(sum(values) / len(values))
            record[f"{metric}_mean"] = mean_value
            record[f"{metric}_std"] = _sample_std(values, mean_value)
            record[f"{metric}_min"] = float(min(values))
            record[f"{metric}_max"] = float(max(values))
        seeds = [_string_value(row.get("seed")) for row in group_rows if _string_value(row.get("seed"))]
        record["seed_count"] = len(dict.fromkeys(seeds))
        record["seed_list"] = ",".join(dict.fromkeys(seeds))
        if aggregation_id is not None:
            record["aggregation_id"] = aggregation_id
        if aggregation_note is not None:
            record["aggregation_note"] = aggregation_note
        result.append(record)
    return result


def validate_seed_summary(
    seed_rows: Iterable[Mapping[str, str]],
    summary_rows: Iterable[Mapping[str, str]],
    *,
    tolerance: float = 1e-9,
) -> dict[str, object]:
    expected_records = aggregate_seed_rows(seed_rows)
    summary_index = {
        _summary_key(row): row
        for row in summary_rows
    }
    mismatches: list[dict[str, object]] = []
    checked_fields = 0
    for expected in expected_records:
        key = _summary_key(expected)
        actual = summary_index.get(key)
        if actual is None:
            mismatches.append(
                {
                    "group": dict(zip([column for column in SUMMARY_GROUP_COLUMNS if column in expected], key, strict=False)),
                    "field": None,
                    "reason": "missing_summary_row",
                }
            )
            continue
        for metric in SUMMARY_METRIC_COLUMNS:
            for suffix in ("mean", "std", "min", "max"):
                field = f"{metric}_{suffix}"
                if field not in expected or field not in actual:
                    continue
                checked_fields += 1
                expected_value = _parse_float(expected.get(field))
                actual_value = _parse_float(actual.get(field))
                if expected_value is None and actual_value is None:
                    continue
                if expected_value is None or actual_value is None:
                    mismatches.append({"group": dict(_group_items(expected)), "field": field, "expected": expected_value, "actual": actual_value})
                    continue
                if not math.isclose(expected_value, actual_value, rel_tol=0.0, abs_tol=tolerance):
                    mismatches.append({"group": dict(_group_items(expected)), "field": field, "expected": expected_value, "actual": actual_value})
        if "seed_count" in actual:
            checked_fields += 1
            actual_seed_count = _parse_float(actual.get("seed_count"))
            if actual_seed_count != expected["seed_count"]:
                mismatches.append(
                    {
                        "group": dict(_group_items(expected)),
                        "field": "seed_count",
                        "expected": expected["seed_count"],
                        "actual": actual.get("seed_count"),
                    }
                )
    return {
        "summary_validation_status": "passed" if not mismatches else "failed",
        "checked_group_count": len(expected_records),
        "checked_field_count": checked_fields,
        "mismatch_count": len(mismatches),
        "mismatches": mismatches,
    }


def bootstrap_from_comparison_rows(
    rows: Iterable[Mapping[str, str]],
    *,
    repeats: int = 5000,
    seed: int = 3407,
    block_length: int = 24,
) -> dict[str, object]:
    materialized = list(rows)
    if not materialized:
        return blocked_bootstrap_status("missing_per_origin_paired_errors", row_count=0)
    baseline_column, proposed_column = _resolve_paired_error_columns(materialized[0].keys())
    if baseline_column is None or proposed_column is None:
        return blocked_bootstrap_status(
            "missing_per_origin_paired_errors",
            row_count=len(materialized),
            available_columns=sorted(materialized[0].keys()),
        )
    baseline = [_parse_float(row.get(baseline_column)) for row in materialized]
    proposed = [_parse_float(row.get(proposed_column)) for row in materialized]
    paired_values = [(base, prop) for base, prop in zip(baseline, proposed, strict=True) if base is not None and prop is not None]
    if not paired_values:
        return blocked_bootstrap_status(
            "missing_per_origin_paired_errors",
            row_count=len(materialized),
            baseline_error_column=baseline_column,
            proposed_error_column=proposed_column,
        )
    baseline_errors = [base for base, _prop in paired_values]
    proposed_errors = [prop for _base, prop in paired_values]
    paired = paired_bootstrap_delta(
        baseline_errors=baseline_errors,
        proposed_errors=proposed_errors,
        repeats=repeats,
        seed=seed,
    )
    blocked = None
    try:
        block = block_bootstrap_delta(
            baseline_errors=baseline_errors,
            proposed_errors=proposed_errors,
            block_length=block_length,
            repeats=repeats,
            seed=seed,
        )
    except ValueError as exc:
        block = None
        blocked = str(exc)
    return {
        "bootstrap_status": "completed",
        "blocked_reason": None,
        "origin_count": len(paired_values),
        "row_count": len(materialized),
        "baseline_error_column": baseline_column,
        "proposed_error_column": proposed_column,
        "bootstrap_repeats": repeats,
        "bootstrap_seed": seed,
        "block_length": block_length,
        "paired_bootstrap": paired,
        "block_bootstrap": block,
        "block_bootstrap_blocked_reason": blocked,
    }


def blocked_bootstrap_status(reason: str, **extra: object) -> dict[str, object]:
    return {
        "bootstrap_status": "blocked",
        "blocked_reason": reason,
        "detail": (
            "Paired and block bootstrap require per-origin paired error rows for "
            "the compared baseline/proposed models. Aggregate seed-level metrics "
            "are not sufficient and must not be resampled as origin-level evidence."
        ),
        "required_column_pairs": [list(pair) for pair in PAIRED_ERROR_COLUMN_CANDIDATES],
        **extra,
    }


def _resolve_paired_error_columns(columns: Iterable[str]) -> tuple[str | None, str | None]:
    available = set(columns)
    for baseline_column, proposed_column in PAIRED_ERROR_COLUMN_CANDIDATES:
        if baseline_column in available and proposed_column in available:
            return baseline_column, proposed_column
    return None, None


def _sample_std(values: list[float], mean_value: float) -> float:
    if len(values) <= 1:
        return 0.0
    variance = sum((value - mean_value) ** 2 for value in values) / (len(values) - 1)
    return float(math.sqrt(variance))


def _summary_key(row: Mapping[str, object]) -> tuple[str, ...]:
    return tuple(_string_value(row.get(column)) for column in SUMMARY_GROUP_COLUMNS if column in row)


def _group_items(row: Mapping[str, object]) -> list[tuple[str, object]]:
    return [(column, row[column]) for column in SUMMARY_GROUP_COLUMNS if column in row]


def _parse_float(value: object) -> float | None:
    if value is None:
        return None
    text = str(value).strip()
    if text == "":
        return None
    try:
        parsed = float(text)
    except ValueError:
        return None
    if math.isnan(parsed):
        return None
    return parsed


def _string_value(value: object) -> str:
    if value is None:
        return ""
    return str(value)


def _coerce_summary_value(value: str) -> object:
    text = str(value)
    if text == "True":
        return True
    if text == "False":
        return False
    parsed = _parse_float(text)
    if parsed is not None and text.strip() == str(int(parsed)):
        return int(parsed)
    return value
