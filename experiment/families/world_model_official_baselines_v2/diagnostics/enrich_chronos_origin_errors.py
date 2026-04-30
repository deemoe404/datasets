from __future__ import annotations

import argparse
import csv
import gzip
import json
from datetime import UTC, datetime
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import numpy as np

DIAGNOSTICS_DIR = Path(__file__).resolve().parent
FAMILY_DIR = DIAGNOSTICS_DIR.parent
while str(DIAGNOSTICS_DIR) in sys.path:
    sys.path.remove(str(DIAGNOSTICS_DIR))
if str(FAMILY_DIR) not in sys.path:
    sys.path.insert(0, str(FAMILY_DIR))

import formal_tuning as formal  # noqa: E402
from diagnostics import chronos2_rolling_shards as chronos_shards  # noqa: E402
from diagnostics import statistics  # noqa: E402

OUTPUT_COLUMNS = (
    "dataset_id",
    "model_id",
    "model_variant",
    "task_id",
    "history_steps",
    "forecast_steps",
    "seed",
    "split_name",
    "eval_protocol",
    "metric_scope",
    "origin_index",
    "target_index",
    "output_start_us",
    "output_end_us",
    "window_count",
    "origin_prediction_count",
    "node_count",
    "trial_id",
    "formal_search_config_id",
    "feature_budget_id",
    "output_parameterization",
    "selection_metric",
    "selected_by",
    "no_test_feedback",
    "uses_future_target",
    "residual_anchor_steps",
    "baseline_model_variant",
    "baseline_abs_error_pu",
    "proposed_abs_error_pu",
    "control_abs_error_pu",
    "candidate_abs_error_pu",
    "source_abs_error_path",
    "source_shard_start",
    "source_shard_stop",
)


def metadata_from_aggregate(aggregate: Mapping[str, Any]) -> dict[str, Any]:
    chronos_contract = dict(aggregate.get("chronos_contract") or {})
    variant = str(aggregate.get("model_variant", formal.CHRONOS2_VARIANT))
    specs = {spec.model_variant: spec for spec in formal.resolve_variant_specs(None)}
    spec = specs.get(variant)
    seed = aggregate.get("seed")
    return {
        "dataset_id": aggregate["dataset_id"],
        "model_id": "WORLD_MODEL_OFFICIAL_BASELINE",
        "model_variant": variant,
        "task_id": formal.TASK_ID,
        "history_steps": int(aggregate.get("history_steps", formal.HISTORY_STEPS)),
        "forecast_steps": int(aggregate.get("forecast_steps", formal.FORECAST_STEPS)),
        "seed": int(3407 if seed is None else seed),
        "split_name": aggregate["split_name"],
        "eval_protocol": aggregate["eval_protocol"],
        "metric_scope": "forecast_origin",
        "window_count": int(aggregate["total_window_count"]),
        "node_count": int(aggregate["node_count"]),
        "trial_id": aggregate.get("trial_id", "chronos2_zero_shot_median"),
        "formal_search_config_id": aggregate.get("formal_search_config_id", "chronos2_zero_shot_b2"),
        "feature_budget_id": spec.feature_budget_id if spec else "B2",
        "output_parameterization": spec.output_parameterization if spec else _infer_output_parameterization(variant),
        "selection_metric": aggregate.get("selection_metric", "val_overall_rmse"),
        "selected_by": aggregate.get("selected_by", "validation_only"),
        "no_test_feedback": aggregate.get("no_test_feedback"),
        "uses_future_target": chronos_contract.get("uses_future_target"),
        "residual_anchor_steps": 0,
        "baseline_model_variant": formal.PERSISTENCE_VARIANT,
    }


def proposed_origin_abs_errors_from_flat(
    abs_errors: Sequence[float] | np.ndarray,
    valid_counts: Sequence[int] | np.ndarray,
    *,
    source_abs_error_path: str,
) -> np.ndarray:
    flat = np.asarray(abs_errors, dtype=np.float64)
    if flat.ndim != 1:
        raise ValueError(f"Expected flat abs_errors for {source_abs_error_path}, found shape {flat.shape!r}.")
    counts_raw = np.asarray(valid_counts)
    if counts_raw.ndim != 1:
        raise ValueError(f"Expected 1D valid_counts, found shape {counts_raw.shape!r}.")
    if np.any(counts_raw < 0):
        raise ValueError("valid_counts must be non-negative.")
    counts_float = counts_raw.astype(np.float64, copy=False)
    if not np.all(np.isfinite(counts_float)) or not np.all(counts_float == np.floor(counts_float)):
        raise ValueError("valid_counts must contain integer values.")
    counts = counts_float.astype(np.int64, copy=False)
    expected_size = int(counts.sum())
    if flat.size != expected_size:
        raise ValueError(
            f"Flat abs_errors size mismatch for {source_abs_error_path}: found {flat.size}, expected {expected_size}."
        )

    proposed = np.full((counts.size,), np.nan, dtype=np.float64)
    offset = 0
    for index, count in enumerate(counts):
        next_offset = offset + int(count)
        if count > 0:
            proposed[index] = float(flat[offset:next_offset].mean())
        offset = next_offset
    return proposed


def validate_and_enrich_flat_abs_errors(
    abs_errors: Sequence[float] | np.ndarray,
    windows: Any,
    valid_counts: Sequence[int] | np.ndarray,
    baseline_abs_error_pu: Sequence[float] | np.ndarray,
    *,
    metadata: Mapping[str, Any],
    expected_start: int,
    source_abs_error_path: str,
    source_shard_start: int,
    source_shard_stop: int,
    source_shard_record: Mapping[str, Any] | None = None,
) -> list[dict[str, Any]]:
    expected_count = int(source_shard_stop) - int(source_shard_start)
    if len(windows) != expected_count:
        raise ValueError(f"Window slice length mismatch: found {len(windows)}, expected {expected_count}.")
    counts_raw = np.asarray(valid_counts)
    if counts_raw.shape != (expected_count,):
        raise ValueError(f"valid_counts shape mismatch: found {counts_raw.shape!r}, expected {(expected_count,)!r}.")
    counts_float = counts_raw.astype(np.float64, copy=False)
    if np.any(counts_float < 0) or not np.all(np.isfinite(counts_float)) or not np.all(counts_float == np.floor(counts_float)):
        raise ValueError("valid_counts must contain non-negative integer values.")
    counts = counts_float.astype(np.int64, copy=False)
    baseline = np.asarray(baseline_abs_error_pu, dtype=np.float64)
    if baseline.shape != (expected_count,):
        raise ValueError(f"Baseline error shape mismatch: found {baseline.shape!r}, expected {(expected_count,)!r}.")

    if source_shard_record is not None:
        _validate_shard_record_metadata(
            source_shard_record,
            windows,
            source_shard_start=source_shard_start,
            source_shard_stop=source_shard_stop,
            valid_counts=counts,
        )

    proposed = proposed_origin_abs_errors_from_flat(
        abs_errors,
        counts,
        source_abs_error_path=source_abs_error_path,
    )

    enriched: list[dict[str, Any]] = []
    for local_index in range(expected_count):
        origin_index = int(expected_start) + local_index
        baseline_value = float(baseline[local_index])
        proposed_value = float(proposed[local_index])
        enriched.append(
            {
                **metadata,
                "origin_index": origin_index,
                "target_index": int(windows.target_indices[local_index]),
                "output_start_us": int(windows.output_start_us[local_index]),
                "output_end_us": int(windows.output_end_us[local_index]),
                "origin_prediction_count": int(counts[local_index]),
                "baseline_abs_error_pu": baseline_value,
                "proposed_abs_error_pu": proposed_value,
                "control_abs_error_pu": baseline_value,
                "candidate_abs_error_pu": proposed_value,
                "source_abs_error_path": source_abs_error_path,
                "source_shard_start": int(source_shard_start),
                "source_shard_stop": int(source_shard_stop),
            }
        )
    return enriched


def persistence_origin_abs_errors(prepared: Any, windows: Any) -> np.ndarray:
    targets, valid = formal._target_and_valid(prepared, windows)
    baseline_predictions = formal._repeat_anchor(formal._last_value_anchor(prepared, windows), prepared.forecast_steps)
    baseline_abs_error, _valid_counts = formal._per_origin_abs_error_pu(baseline_predictions, targets, valid)
    return baseline_abs_error


def valid_counts_for_windows(prepared: Any, windows: Any) -> np.ndarray:
    _targets, valid = formal._target_and_valid(prepared, windows)
    return valid.astype(np.float64, copy=False).sum(axis=(1, 2)).astype(np.int64, copy=False)


def enrich_chronos_origin_errors(
    *,
    aggregate_json_path: Path,
    output_csv_path: Path,
) -> int:
    aggregate = json.loads(aggregate_json_path.read_text(encoding="utf-8"))
    metadata = metadata_from_aggregate(aggregate)
    dataset_id = str(metadata["dataset_id"])
    split_name = str(metadata["split_name"])
    eval_protocol = str(metadata["eval_protocol"])
    if eval_protocol != formal.ROLLING_EVAL_PROTOCOL:
        raise ValueError(f"This enricher is intended for rolling origin errors, found {eval_protocol!r}.")
    if (aggregate.get("chronos_contract") or {}).get("uses_future_target") is not False:
        raise ValueError("Aggregate does not prove uses_future_target=false.")
    if aggregate.get("no_test_feedback") is not True:
        raise ValueError("Aggregate does not prove no_test_feedback=true.")

    prepared = formal._prepare_dataset(dataset_id, max_train_origins=None, max_eval_origins=None)
    windows = chronos_shards._windows_for_split(prepared, split_name=split_name, eval_protocol=eval_protocol)
    total_window_count = int(aggregate["total_window_count"])
    if len(windows) != total_window_count:
        raise ValueError(f"Prepared window count {len(windows)} does not match aggregate {total_window_count}.")

    source_shards = sorted(aggregate.get("source_shards") or [], key=lambda item: int(item["shard_start"]))
    if not source_shards:
        raise ValueError("Aggregate has no source_shards with abs_error_path entries.")

    aggregate_dir = aggregate_json_path.parent
    output_csv_path.parent.mkdir(parents=True, exist_ok=True)
    row_count = 0
    expected_start = 0
    with output_csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(OUTPUT_COLUMNS), extrasaction="ignore")
        writer.writeheader()
        for shard in source_shards:
            shard_start = int(shard["shard_start"])
            shard_stop = int(shard["shard_stop"])
            if shard_start != expected_start:
                raise ValueError(f"Missing or overlapping source shard coverage: expected {expected_start}, found {shard_start}.")
            abs_error_path = shard.get("abs_error_path")
            if not abs_error_path:
                raise ValueError(f"Source shard [{shard_start}, {shard_stop}) is missing abs_error_path.")
            abs_error_file = _resolve_artifact_path(aggregate_dir, abs_error_path, artifact_name="shard abs-errors NPY")
            shard_record = _load_source_shard_record(aggregate_dir, shard)
            shard_windows = formal._slice_windows(windows, shard_start, shard_stop)
            counts = valid_counts_for_windows(prepared, shard_windows)
            baseline_abs = persistence_origin_abs_errors(prepared, shard_windows)
            enriched = validate_and_enrich_flat_abs_errors(
                np.load(abs_error_file),
                shard_windows,
                counts,
                baseline_abs,
                metadata=metadata,
                expected_start=shard_start,
                source_abs_error_path=str(abs_error_path),
                source_shard_start=shard_start,
                source_shard_stop=shard_stop,
                source_shard_record=shard_record,
            )
            writer.writerows(enriched)
            row_count += len(enriched)
            expected_start = shard_stop
    if expected_start != total_window_count:
        raise ValueError(f"Missing final source shard coverage: covered {expected_start}, expected {total_window_count}.")
    return row_count


def write_bootstrap_artifacts(
    *,
    comparison_csv_path: Path,
    output_prefix: Path,
    repeats: int = 5000,
    seed: int = 3407,
    block_length: int = 24,
) -> dict[str, Path]:
    rows = _read_csv(comparison_csv_path)
    status = statistics.bootstrap_from_comparison_rows(
        rows,
        repeats=repeats,
        seed=seed,
        block_length=block_length,
    )
    status.update(
        {
            "created_at": datetime.now(tz=UTC).isoformat(),
            "comparison_csv": str(comparison_csv_path),
        }
    )
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    json_path = output_prefix.with_name(f"{output_prefix.name}-bootstrap-status.json")
    csv_path = output_prefix.with_name(f"{output_prefix.name}-bootstrap-status.csv")
    _write_json(json_path, status)
    _write_csv(csv_path, [_flatten_record(status)])
    return {"bootstrap_status_json": json_path, "bootstrap_status_csv": csv_path}


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Enrich Chronos-2 rolling shard flat absolute errors with last-value persistence paired errors."
    )
    parser.add_argument("--aggregate-json", required=True, type=Path)
    parser.add_argument("--output-csv", required=True, type=Path)
    parser.add_argument("--bootstrap-prefix", type=Path)
    parser.add_argument("--bootstrap-repeats", type=int, default=5000)
    parser.add_argument("--bootstrap-block-length", type=int, default=24)
    parser.add_argument("--bootstrap-seed", type=int, default=3407)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    row_count = enrich_chronos_origin_errors(
        aggregate_json_path=args.aggregate_json,
        output_csv_path=args.output_csv,
    )
    artifacts: dict[str, Any] = {"comparison_csv": str(args.output_csv), "row_count": row_count}
    if args.bootstrap_prefix is not None:
        bootstrap_artifacts = write_bootstrap_artifacts(
            comparison_csv_path=args.output_csv,
            output_prefix=args.bootstrap_prefix,
            repeats=args.bootstrap_repeats,
            seed=args.bootstrap_seed,
            block_length=args.bootstrap_block_length,
        )
        artifacts.update({name: str(path) for name, path in bootstrap_artifacts.items()})
    print(json.dumps(artifacts, indent=2, sort_keys=True))
    return 0


def _validate_shard_record_metadata(
    record: Mapping[str, Any],
    windows: Any,
    *,
    source_shard_start: int,
    source_shard_stop: int,
    valid_counts: np.ndarray,
) -> None:
    if "shard_start" in record and int(record["shard_start"]) != int(source_shard_start):
        raise ValueError(f"Source shard JSON start mismatch: found {record['shard_start']}, expected {source_shard_start}.")
    if "shard_stop" in record and int(record["shard_stop"]) != int(source_shard_stop):
        raise ValueError(f"Source shard JSON stop mismatch: found {record['shard_stop']}, expected {source_shard_stop}.")
    expected_fields = {
        "target_index_start": int(windows.target_indices[0]),
        "target_index_stop_exclusive": int(windows.target_indices[-1]) + 1,
        "output_start_us": int(windows.output_start_us[0]),
        "output_end_us": int(windows.output_end_us[-1]),
    }
    for key, expected in expected_fields.items():
        value = record.get(key)
        if value is not None and int(value) != expected:
            raise ValueError(f"Source shard JSON {key} mismatch: found {value}, expected {expected}.")
    prediction_count = (record.get("components") or {}).get("prediction_count")
    if prediction_count is not None and int(prediction_count) != int(valid_counts.sum()):
        raise ValueError(
            f"Source shard JSON prediction_count mismatch: found {prediction_count}, expected {int(valid_counts.sum())}."
        )


def _load_source_shard_record(base_dir: Path, shard: Mapping[str, Any]) -> dict[str, Any] | None:
    json_path = shard.get("json_path")
    if not json_path:
        return None
    path = _resolve_artifact_path(base_dir, json_path, artifact_name="shard JSON")
    return json.loads(path.read_text(encoding="utf-8"))


def _read_csv(path: Path) -> list[dict[str, str]]:
    resolved = path if path.exists() else Path(f"{path}.gz")
    opener = gzip.open if resolved.suffix == ".gz" else Path.open
    with opener(resolved, "rt", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _flatten_record(record: Mapping[str, Any]) -> dict[str, Any]:
    flattened: dict[str, Any] = {}
    for key, value in record.items():
        if isinstance(value, Mapping):
            for nested_key, nested_value in value.items():
                flattened[f"{key}_{nested_key}"] = nested_value
        elif isinstance(value, list):
            flattened[key] = json.dumps(value, sort_keys=True)
        else:
            flattened[key] = value
    return flattened


def _resolve_artifact_path(base_dir: Path, raw_path: Any, *, artifact_name: str) -> Path:
    path = Path(str(raw_path))
    if not path.is_absolute():
        path = base_dir / path
    if not path.exists():
        raise FileNotFoundError(f"Missing {artifact_name}: {path}")
    return path


def _infer_output_parameterization(model_variant: str) -> str:
    if "residual" in model_variant:
        return "residual"
    if "direct" in model_variant:
        return "direct"
    if "zero_shot" in model_variant:
        return "direct"
    return ""


if __name__ == "__main__":
    raise SystemExit(main())
