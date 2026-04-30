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
from diagnostics import statistics  # noqa: E402
from diagnostics import tft_pf_rolling_shards as tft_shards  # noqa: E402

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
    "source_origin_error_path",
    "source_shard_start",
    "source_shard_stop",
)


def metadata_from_aggregate(aggregate: Mapping[str, Any]) -> dict[str, Any]:
    frozen_config = dict(aggregate.get("frozen_config") or {})
    tft_contract = dict(aggregate.get("tft_contract") or {})
    variant = str(aggregate["model_variant"])
    specs = {spec.model_variant: spec for spec in formal.resolve_variant_specs(None)}
    spec = specs.get(variant)
    seed_value = aggregate.get("seed")
    if seed_value is None:
        seed_value = frozen_config.get("seed", 3407)
    return {
        "dataset_id": aggregate["dataset_id"],
        "model_id": "WORLD_MODEL_OFFICIAL_BASELINE",
        "model_variant": variant,
        "task_id": formal.TASK_ID,
        "history_steps": int(aggregate.get("history_steps", formal.HISTORY_STEPS)),
        "forecast_steps": int(aggregate.get("forecast_steps", formal.FORECAST_STEPS)),
        "seed": int(seed_value),
        "split_name": aggregate["split_name"],
        "eval_protocol": aggregate["eval_protocol"],
        "metric_scope": "forecast_origin",
        "window_count": int(aggregate["total_window_count"]),
        "node_count": int(aggregate["node_count"]),
        "trial_id": aggregate.get("trial_id"),
        "formal_search_config_id": aggregate.get("formal_search_config_id"),
        "feature_budget_id": tft_contract.get("feature_budget_id") or (spec.feature_budget_id if spec else None),
        "output_parameterization": spec.output_parameterization if spec else _infer_output_parameterization(variant),
        "selection_metric": aggregate.get("selection_metric"),
        "selected_by": aggregate.get("selected_by"),
        "no_test_feedback": aggregate.get("no_test_feedback"),
        "uses_future_target": tft_contract.get("uses_future_target"),
        "residual_anchor_steps": int(frozen_config.get("residual_anchor_steps", 0)),
        "baseline_model_variant": formal.PERSISTENCE_VARIANT,
    }


def validate_and_enrich_rows(
    source_rows: Sequence[Mapping[str, Any]],
    windows: Any,
    baseline_abs_error_pu: Sequence[float] | np.ndarray,
    *,
    metadata: Mapping[str, Any],
    expected_start: int,
    source_origin_error_path: str,
    source_shard_start: int,
    source_shard_stop: int,
) -> list[dict[str, Any]]:
    expected_count = int(source_shard_stop) - int(source_shard_start)
    if len(source_rows) != expected_count:
        raise ValueError(
            f"Origin-error row count mismatch for {source_origin_error_path}: "
            f"found {len(source_rows)}, expected {expected_count}."
        )
    if len(windows) != expected_count:
        raise ValueError(f"Window slice length mismatch: found {len(windows)}, expected {expected_count}.")
    baseline = np.asarray(baseline_abs_error_pu, dtype=np.float64)
    if baseline.shape != (expected_count,):
        raise ValueError(f"Baseline error shape mismatch: found {baseline.shape!r}, expected {(expected_count,)!r}.")

    enriched: list[dict[str, Any]] = []
    for local_index, row in enumerate(source_rows):
        origin_index = _parse_int(row.get("origin_index"), "origin_index")
        expected_origin = int(expected_start) + local_index
        if origin_index != expected_origin:
            raise ValueError(
                f"origin_index mismatch in {source_origin_error_path}: found {origin_index}, expected {expected_origin}."
            )
        target_index = _parse_int(row.get("target_index"), "target_index")
        output_start_us = _parse_int(row.get("output_start_us"), "output_start_us")
        output_end_us = _parse_int(row.get("output_end_us"), "output_end_us")
        if target_index != int(windows.target_indices[local_index]):
            raise ValueError(
                f"target_index mismatch at origin {origin_index}: "
                f"found {target_index}, expected {int(windows.target_indices[local_index])}."
            )
        if output_start_us != int(windows.output_start_us[local_index]):
            raise ValueError(
                f"output_start_us mismatch at origin {origin_index}: "
                f"found {output_start_us}, expected {int(windows.output_start_us[local_index])}."
            )
        if output_end_us != int(windows.output_end_us[local_index]):
            raise ValueError(
                f"output_end_us mismatch at origin {origin_index}: "
                f"found {output_end_us}, expected {int(windows.output_end_us[local_index])}."
            )
        proposed = _parse_float(row.get("proposed_abs_error_pu"), "proposed_abs_error_pu")
        baseline_value = float(baseline[local_index])
        enriched.append(
            {
                **metadata,
                "origin_index": origin_index,
                "target_index": target_index,
                "output_start_us": output_start_us,
                "output_end_us": output_end_us,
                "origin_prediction_count": _parse_int(row.get("origin_prediction_count"), "origin_prediction_count"),
                "baseline_abs_error_pu": baseline_value,
                "proposed_abs_error_pu": proposed,
                "control_abs_error_pu": baseline_value,
                "candidate_abs_error_pu": proposed,
                "source_origin_error_path": source_origin_error_path,
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


def enrich_tft_origin_errors(
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
    if aggregate.get("tft_contract", {}).get("uses_future_target") is not False:
        raise ValueError("Aggregate does not prove uses_future_target=false.")
    if aggregate.get("no_test_feedback") is not True:
        raise ValueError("Aggregate does not prove no_test_feedback=true.")

    frozen_config = dict(aggregate.get("frozen_config") or {})
    prepared = formal._prepare_dataset(
        dataset_id,
        max_train_origins=frozen_config.get("max_train_origins"),
        max_eval_origins=None,
    )
    windows = tft_shards._windows_for_split(prepared, split_name=split_name, eval_protocol=eval_protocol)
    total_window_count = int(aggregate["total_window_count"])
    if len(windows) != total_window_count:
        raise ValueError(f"Prepared window count {len(windows)} does not match aggregate {total_window_count}.")

    source_shards = sorted(aggregate.get("source_shards") or [], key=lambda item: int(item["shard_start"]))
    if not source_shards:
        raise ValueError("Aggregate has no source_shards with origin_error_path entries.")

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
            origin_error_path = shard.get("origin_error_path")
            if not origin_error_path:
                raise ValueError(f"Source shard [{shard_start}, {shard_stop}) is missing origin_error_path.")
            origin_csv = _resolve_artifact_path(aggregate_dir, origin_error_path)
            source_rows = _read_csv(origin_csv)
            shard_windows = formal._slice_windows(windows, shard_start, shard_stop)
            baseline_abs = persistence_origin_abs_errors(prepared, shard_windows)
            enriched = validate_and_enrich_rows(
                source_rows,
                shard_windows,
                baseline_abs,
                metadata=metadata,
                expected_start=shard_start,
                source_origin_error_path=str(origin_error_path),
                source_shard_start=shard_start,
                source_shard_stop=shard_stop,
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
        description="Enrich TFT-PF rolling shard origin errors with last-value persistence paired errors."
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
    row_count = enrich_tft_origin_errors(
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


def _resolve_artifact_path(base_dir: Path, raw_path: Any) -> Path:
    path = Path(str(raw_path))
    if not path.is_absolute():
        path = base_dir / path
    if not path.exists():
        raise FileNotFoundError(f"Missing shard origin-error CSV: {path}")
    return path


def _infer_output_parameterization(model_variant: str) -> str:
    if "residual" in model_variant:
        return "residual"
    if "direct" in model_variant:
        return "direct"
    return ""


def _parse_int(value: Any, column: str) -> int:
    if value is None or str(value).strip() == "":
        raise ValueError(f"Missing required integer column {column!r}.")
    parsed = float(str(value))
    if not parsed.is_integer():
        raise ValueError(f"Expected integer value for {column!r}, found {value!r}.")
    return int(parsed)


def _parse_float(value: Any, column: str) -> float:
    if value is None or str(value).strip() == "":
        raise ValueError(f"Missing required float column {column!r}.")
    return float(str(value))


if __name__ == "__main__":
    raise SystemExit(main())
