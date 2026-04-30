from __future__ import annotations

import argparse
import csv
import gzip
import json
from datetime import UTC, datetime
from pathlib import Path
import sys
from typing import Any, Iterable

FAMILY_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = FAMILY_DIR.parents[2]
if str(FAMILY_DIR) not in sys.path:
    sys.path.insert(0, str(FAMILY_DIR))

from diagnostics import statistics  # noqa: E402

FAMILY_ID = "world_model_official_baselines_v2"
DEFAULT_RUN_DIR = (
    REPO_ROOT
    / "experiment"
    / "artifacts"
    / "scratch"
    / FAMILY_ID
    / "long_run_20260425_paper_grade"
    / "dgcrn_rescue"
)
DEFAULT_PUBLISH_ROOT = REPO_ROOT / "experiment" / "artifacts" / "published" / FAMILY_ID
DEFAULT_PREFIX = "20260425-paper-grade-dgcrn-rescue"
DEFAULT_EXISTING_FINAL_TABLE = DEFAULT_PUBLISH_ROOT / "20260425-paper-grade-final-table.csv"
DEFAULT_STEM = "dgcrn_kelmarsh_dgcrn_official_core_residual_b2_v2"
EXPECTED_COUNTS = {
    "rolling_origin_no_refit": {"window_count": 94458, "prediction_count": 20300940},
    "non_overlap": {"window_count": 2624, "prediction_count": 563951},
}


def publish_rescue(
    *,
    run_dir: Path,
    publish_root: Path,
    prefix: str,
    artifact_stem: str,
    expected_seeds: list[int],
    existing_final_table: Path | None,
    bootstrap_repeats: int,
    bootstrap_seed: int,
    block_length: int,
) -> dict[str, Any]:
    run_dir = run_dir.resolve()
    publish_root = publish_root.resolve()
    publish_root.mkdir(parents=True, exist_ok=True)

    validation: list[dict[str, Any]] = []
    completed_seeds: list[int] = []
    seed_rows: list[dict[str, Any]] = []
    manifest_by_seed: dict[int, dict[str, Any]] = {}
    origin_rows_by_protocol: dict[str, list[dict[str, Any]]] = {protocol: [] for protocol in EXPECTED_COUNTS}

    for seed in expected_seeds:
        manifest_path = run_dir / f"{artifact_stem}_seed{seed}_checkpoint.manifest.json"
        manifest = _read_json_if_exists(manifest_path)
        if manifest is None:
            validation.append({"seed": seed, "status": "missing_manifest", "path": str(manifest_path)})
            continue
        manifest_by_seed[seed] = manifest
        seed_complete = True
        for protocol, counts in EXPECTED_COUNTS.items():
            aggregate_path = run_dir / f"{artifact_stem}_test_{protocol}_seed{seed}_aggregate.json"
            aggregate_csv_path = run_dir / f"{artifact_stem}_test_{protocol}_seed{seed}_aggregate.csv"
            origin_path = run_dir / f"{artifact_stem}_test_{protocol}_seed{seed}_aggregate.origin_errors.csv"
            bootstrap_path = run_dir / f"{artifact_stem}_test_{protocol}_seed{seed}_aggregate.bootstrap-status.json"
            aggregate = _read_json_if_exists(aggregate_path)
            bootstrap = _read_json_if_exists(bootstrap_path)
            row_status = _validate_aggregate(
                seed=seed,
                protocol=protocol,
                aggregate=aggregate,
                aggregate_path=aggregate_path,
                aggregate_csv_path=aggregate_csv_path,
                origin_path=origin_path,
                bootstrap=bootstrap,
                bootstrap_path=bootstrap_path,
                expected_counts=counts,
            )
            validation.append(row_status)
            if row_status["status"] != "complete":
                seed_complete = False
                continue
            rows = _read_csv(aggregate_csv_path)
            for row in rows:
                enriched = dict(row)
                enriched.update(
                    {
                        "checkpoint_manifest_status": manifest.get("status"),
                        "training_completed": _training_completed(manifest),
                        "checkpoint_recovered": bool(manifest.get("checkpoint_recovered", False)),
                    }
                )
                seed_rows.append(enriched)
            origin_rows_by_protocol[protocol].extend(_read_csv(origin_path))
        if seed_complete:
            completed_seeds.append(seed)

    all_expected_seeds_complete = set(completed_seeds) == set(expected_seeds)
    all_manifests_complete = all(
        manifest_by_seed.get(seed, {}).get("status") == "complete" and _training_completed(manifest_by_seed[seed])
        for seed in expected_seeds
    )
    main_table_eligible = all_expected_seeds_complete and all_manifests_complete
    main_table_status = "main" if main_table_eligible else "appendix_diagnostic"
    eligibility_reason = (
        "all_expected_seeds_complete"
        if main_table_eligible
        else _eligibility_reason(expected_seeds, completed_seeds, manifest_by_seed)
    )

    for row in seed_rows:
        row["main_table_status"] = main_table_status
        row["dgcrn_rescue_eligibility_reason"] = eligibility_reason

    artifacts: dict[str, str] = {}
    seed_rows_path = publish_root / f"{prefix}-seed-rows.csv"
    _write_csv(seed_rows_path, seed_rows)
    artifacts["seed_rows_csv"] = str(seed_rows_path)

    summary_rows = statistics.aggregate_seed_rows(
        seed_rows,
        aggregation_id=f"{prefix}-multiseed",
        aggregation_note=(
            "DGCRN rescue shard evaluation. Main-table eligible only when all expected seeds "
            "have complete full training manifests plus full rolling/non-overlap test coverage."
        ),
    )
    for row in summary_rows:
        row["main_table_status"] = main_table_status
        row["dgcrn_rescue_eligibility_reason"] = eligibility_reason
    summary_path = publish_root / f"{prefix}-statistics-summary.csv"
    _write_csv(summary_path, summary_rows)
    artifacts["statistics_summary_csv"] = str(summary_path)

    bootstrap_summary_rows: list[dict[str, Any]] = []
    for protocol, rows in origin_rows_by_protocol.items():
        comparison_path = publish_root / f"{prefix}-{protocol}-origin-comparison.csv"
        _write_csv(comparison_path, rows)
        artifacts[f"{protocol}_origin_comparison_csv"] = str(comparison_path)
        bootstrap = statistics.bootstrap_from_comparison_rows(
            rows,
            repeats=bootstrap_repeats,
            seed=bootstrap_seed,
            block_length=block_length,
        )
        bootstrap.update(
            {
                "created_at": datetime.now(tz=UTC).isoformat(),
                "protocol": protocol,
                "comparison_csv": str(comparison_path),
                "seed_rows_csv": str(seed_rows_path),
                "completed_seed_count": len(completed_seeds),
                "expected_seed_count": len(expected_seeds),
                "main_table_status": main_table_status,
            }
        )
        bootstrap_json_path = publish_root / f"{prefix}-{protocol}-bootstrap-status.json"
        bootstrap_csv_path = publish_root / f"{prefix}-{protocol}-bootstrap-status.csv"
        _write_json(bootstrap_json_path, bootstrap)
        _write_csv(bootstrap_csv_path, [_flatten_record(bootstrap)])
        artifacts[f"{protocol}_bootstrap_status_json"] = str(bootstrap_json_path)
        artifacts[f"{protocol}_bootstrap_status_csv"] = str(bootstrap_csv_path)
        bootstrap_summary_rows.append({"protocol": protocol, **_flatten_record(bootstrap)})

    bootstrap_summary_path = publish_root / f"{prefix}-bootstrap-status-summary.csv"
    _write_csv(bootstrap_summary_path, bootstrap_summary_rows)
    artifacts["bootstrap_status_summary_csv"] = str(bootstrap_summary_path)

    if main_table_eligible and existing_final_table is not None:
        existing_rows = _read_csv(existing_final_table)
        final_rows = [*existing_rows, *_main_table_rows(summary_rows)]
        final_table_path = publish_root / f"{prefix}-final-table.csv"
        _write_csv(final_table_path, final_rows)
        artifacts["final_table_csv"] = str(final_table_path)

    manifest = {
        "created_at": datetime.now(tz=UTC).isoformat(),
        "run_dir": str(run_dir),
        "publish_root": str(publish_root),
        "prefix": prefix,
        "artifact_stem": artifact_stem,
        "expected_seeds": expected_seeds,
        "completed_seeds": completed_seeds,
        "main_table_eligible": main_table_eligible,
        "main_table_status": main_table_status,
        "eligibility_reason": eligibility_reason,
        "selected_by": "validation_only",
        "no_test_feedback": True,
        "test_metrics_used_for_config_selection": False,
        "validation": validation,
        "artifacts": artifacts,
    }
    manifest_path = publish_root / f"{prefix}-manifest.json"
    _write_json(manifest_path, manifest)
    artifacts["manifest_json"] = str(manifest_path)
    return manifest


def _validate_aggregate(
    *,
    seed: int,
    protocol: str,
    aggregate: dict[str, Any] | None,
    aggregate_path: Path,
    aggregate_csv_path: Path,
    origin_path: Path,
    bootstrap: dict[str, Any] | None,
    bootstrap_path: Path,
    expected_counts: dict[str, int],
) -> dict[str, Any]:
    base = {
        "seed": seed,
        "protocol": protocol,
        "aggregate_path": str(aggregate_path),
        "origin_error_path": str(origin_path),
        "bootstrap_path": str(bootstrap_path),
    }
    if aggregate is None:
        return {**base, "status": "missing_aggregate"}
    if aggregate.get("status") != "complete":
        return {**base, "status": "aggregate_not_complete", "aggregate_status": aggregate.get("status")}
    if not aggregate_csv_path.exists():
        return {**base, "status": "missing_aggregate_csv", "aggregate_csv_path": str(aggregate_csv_path)}
    if bootstrap is None:
        return {**base, "status": "missing_bootstrap"}
    if bootstrap.get("bootstrap_status") != "completed":
        return {**base, "status": "bootstrap_not_completed", "bootstrap_status": bootstrap.get("bootstrap_status")}
    if str(aggregate.get("split_name")) != "test":
        return {**base, "status": "not_test_split", "split_name": aggregate.get("split_name")}
    if str(aggregate.get("selected_by")) != "validation_only" or bool(aggregate.get("no_test_feedback")) is not True:
        return {**base, "status": "selection_contract_failed"}
    contract = aggregate.get("dgcrn_contract", {})
    if bool(contract.get("uses_future_target", aggregate.get("uses_future_target", False))):
        return {**base, "status": "future_target_contract_failed"}
    metrics = aggregate.get("metrics", {})
    for key, expected in expected_counts.items():
        actual = int(metrics.get(key, aggregate.get(key, -1)))
        if actual != expected:
            return {**base, "status": "coverage_mismatch", "field": key, "expected": expected, "actual": actual}
    if not origin_path.exists():
        return {**base, "status": "missing_origin_errors"}
    origin_count = _csv_data_row_count(origin_path)
    expected_origin_count = expected_counts["window_count"]
    if origin_count != expected_origin_count:
        return {
            **base,
            "status": "origin_error_count_mismatch",
            "expected": expected_origin_count,
            "actual": origin_count,
        }
    return {**base, "status": "complete"}


def _training_completed(manifest: dict[str, Any]) -> bool:
    if manifest.get("status") == "complete":
        return bool(manifest.get("training_completed", True))
    return bool(manifest.get("training_completed", False))


def _eligibility_reason(expected_seeds: list[int], completed_seeds: list[int], manifest_by_seed: dict[int, dict[str, Any]]) -> str:
    missing = [seed for seed in expected_seeds if seed not in completed_seeds]
    if missing:
        return "missing_full_test_coverage_for_seeds_" + "_".join(str(seed) for seed in missing)
    non_complete = [
        seed
        for seed in expected_seeds
        if manifest_by_seed.get(seed, {}).get("status") != "complete" or not _training_completed(manifest_by_seed[seed])
    ]
    if non_complete:
        return "non_complete_training_manifest_for_seeds_" + "_".join(str(seed) for seed in non_complete)
    return "unknown_not_main_eligible"


def _main_table_rows(summary_rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in summary_rows:
        rows.append(
            {
                "dataset_id": row.get("dataset_id"),
                "model_id": row.get("model_id"),
                "model_variant": row.get("model_variant"),
                "task_id": row.get("task_id"),
                "split_name": row.get("split_name"),
                "eval_protocol": row.get("eval_protocol"),
                "metric_scope": row.get("metric_scope"),
                "feature_budget_id": row.get("feature_budget_id"),
                "output_parameterization": row.get("output_parameterization"),
                "selection_metric": row.get("selection_metric"),
                "selected_by": row.get("selected_by"),
                "no_test_feedback": row.get("no_test_feedback"),
                "gate_a_passed": row.get("gate_a_passed"),
                "gate_b_passed": row.get("gate_b_passed"),
                "gate_b_scope": row.get("gate_b_scope"),
                "gate_b_overfit64_passed": row.get("gate_b_overfit64_passed"),
                "gate_c_passed": row.get("gate_c_passed"),
                "residual_anchor_steps": row.get("residual_anchor_steps"),
                "formal_search_config_id": row.get("formal_search_config_id"),
                "is_best_validation_trial": row.get("is_best_validation_trial"),
                "mae_pu_mean": row.get("mae_pu_mean"),
                "mae_pu_std": row.get("mae_pu_std"),
                "rmse_pu_mean": row.get("rmse_pu_mean"),
                "rmse_pu_std": row.get("rmse_pu_std"),
                "window_count_mean": row.get("window_count_mean"),
                "prediction_count_mean": row.get("prediction_count_mean"),
                "seed_count": row.get("seed_count"),
                "seed_list": row.get("seed_list"),
                "main_table_status": row.get("main_table_status"),
            }
        )
    return rows


def _read_json_if_exists(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _read_csv(path: Path) -> list[dict[str, str]]:
    resolved = path if path.exists() else Path(f"{path}.gz")
    opener = gzip.open if resolved.suffix == ".gz" else Path.open
    with opener(resolved, "rt", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if fieldnames:
            writer.writeheader()
            writer.writerows(rows)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _csv_data_row_count(path: Path) -> int:
    with path.open(newline="", encoding="utf-8") as handle:
        return max(0, sum(1 for _ in handle) - 1)


def _flatten_record(record: dict[str, Any], *, prefix: str | None = None) -> dict[str, Any]:
    flat: dict[str, Any] = {}
    for key, value in record.items():
        next_key = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            flat.update(_flatten_record(value, prefix=next_key))
        else:
            flat[next_key] = value
    return flat


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Publish DGCRN rescue seed rows, summaries, and eligibility manifest.")
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--publish-root", type=Path, default=DEFAULT_PUBLISH_ROOT)
    parser.add_argument("--prefix", default=DEFAULT_PREFIX)
    parser.add_argument("--artifact-stem", default=DEFAULT_STEM)
    parser.add_argument("--expected-seeds", nargs="+", type=int, default=[3407, 42, 2026, 17, 926])
    parser.add_argument("--existing-final-table", type=Path, default=DEFAULT_EXISTING_FINAL_TABLE)
    parser.add_argument("--bootstrap-repeats", type=int, default=5000)
    parser.add_argument("--bootstrap-seed", type=int, default=3407)
    parser.add_argument("--block-length", type=int, default=24)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    manifest = publish_rescue(
        run_dir=args.run_dir,
        publish_root=args.publish_root,
        prefix=args.prefix,
        artifact_stem=args.artifact_stem,
        expected_seeds=args.expected_seeds,
        existing_final_table=args.existing_final_table,
        bootstrap_repeats=args.bootstrap_repeats,
        bootstrap_seed=args.bootstrap_seed,
        block_length=args.block_length,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
