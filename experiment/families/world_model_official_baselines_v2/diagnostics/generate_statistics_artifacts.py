from __future__ import annotations

import argparse
import csv
import gzip
import json
from datetime import UTC, datetime
from pathlib import Path
import sys
from typing import Any

FAMILY_DIR = Path(__file__).resolve().parents[1]
if str(FAMILY_DIR) not in sys.path:
    sys.path.insert(0, str(FAMILY_DIR))

from diagnostics.statistics import (
    aggregate_seed_rows,
    blocked_bootstrap_status,
    bootstrap_from_comparison_rows,
    validate_seed_summary,
)


def build_statistics_artifacts(
    *,
    seed_rows_path: Path,
    output_prefix: Path,
    summary_csv_path: Path | None = None,
    comparison_csv_path: Path | None = None,
    aggregation_id: str | None = None,
    aggregation_note: str | None = None,
    bootstrap_repeats: int = 5000,
    bootstrap_seed: int = 3407,
    block_length: int = 24,
    fail_on_summary_mismatch: bool = False,
) -> dict[str, Path]:
    seed_rows = _read_csv(seed_rows_path)
    output_prefix.parent.mkdir(parents=True, exist_ok=True)

    generated_summary = aggregate_seed_rows(
        seed_rows,
        aggregation_id=aggregation_id,
        aggregation_note=aggregation_note,
    )
    summary_output_path = output_prefix.with_name(f"{output_prefix.name}-statistics-summary.csv")
    _write_csv(summary_output_path, generated_summary)

    artifacts = {"statistics_summary": summary_output_path}
    validation: dict[str, Any] | None = None
    if summary_csv_path is not None:
        validation = validate_seed_summary(seed_rows, _read_csv(summary_csv_path))
        validation.update(
            {
                "created_at": datetime.now(tz=UTC).isoformat(),
                "seed_rows_csv": str(seed_rows_path),
                "summary_csv": str(summary_csv_path),
                "statistics_summary_csv": str(summary_output_path),
            }
        )
        validation_path = output_prefix.with_name(f"{output_prefix.name}-summary-validation.json")
        _write_json(validation_path, validation)
        artifacts["summary_validation"] = validation_path
        if fail_on_summary_mismatch and validation["summary_validation_status"] != "passed":
            raise ValueError(f"summary validation failed; see {validation_path}")

    if comparison_csv_path is None:
        bootstrap_status = blocked_bootstrap_status(
            "missing_per_origin_paired_errors",
            seed_rows_csv=str(seed_rows_path),
            summary_csv=str(summary_csv_path) if summary_csv_path is not None else None,
            comparison_csv=None,
            seed_row_count=len(seed_rows),
        )
    else:
        bootstrap_status = bootstrap_from_comparison_rows(
            _read_csv(comparison_csv_path),
            repeats=bootstrap_repeats,
            seed=bootstrap_seed,
            block_length=block_length,
        )
        bootstrap_status.update({"comparison_csv": str(comparison_csv_path)})
    bootstrap_status.update(
        {
            "created_at": datetime.now(tz=UTC).isoformat(),
            "seed_rows_csv": str(seed_rows_path),
            "summary_csv": str(summary_csv_path) if summary_csv_path is not None else None,
            "statistics_summary_csv": str(summary_output_path),
        }
    )
    status_json_path = output_prefix.with_name(f"{output_prefix.name}-bootstrap-status.json")
    status_csv_path = output_prefix.with_name(f"{output_prefix.name}-bootstrap-status.csv")
    _write_json(status_json_path, bootstrap_status)
    _write_csv(status_csv_path, [_flatten_record(bootstrap_status)])
    artifacts["bootstrap_status_json"] = status_json_path
    artifacts["bootstrap_status_csv"] = status_csv_path
    return artifacts


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate lightweight v2 statistical summary/bootstrap artifacts from existing CSV outputs."
    )
    parser.add_argument("--seed-rows", required=True, type=Path, help="Seed-level result rows CSV.")
    parser.add_argument("--summary-csv", type=Path, help="Existing seed summary CSV to verify.")
    parser.add_argument("--comparison-csv", type=Path, help="Optional per-origin paired error comparison CSV.")
    parser.add_argument("--output-prefix", required=True, type=Path, help="Output prefix for generated artifacts.")
    parser.add_argument("--aggregation-id", help="Aggregation id to stamp onto generated summary rows.")
    parser.add_argument("--aggregation-note", help="Aggregation note to stamp onto generated summary rows.")
    parser.add_argument("--bootstrap-repeats", type=int, default=5000)
    parser.add_argument("--bootstrap-seed", type=int, default=3407)
    parser.add_argument("--block-length", type=int, default=24)
    parser.add_argument("--fail-on-summary-mismatch", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    artifacts = build_statistics_artifacts(
        seed_rows_path=args.seed_rows,
        output_prefix=args.output_prefix,
        summary_csv_path=args.summary_csv,
        comparison_csv_path=args.comparison_csv,
        aggregation_id=args.aggregation_id,
        aggregation_note=args.aggregation_note,
        bootstrap_repeats=args.bootstrap_repeats,
        bootstrap_seed=args.bootstrap_seed,
        block_length=args.block_length,
        fail_on_summary_mismatch=args.fail_on_summary_mismatch,
    )
    print(json.dumps({name: str(path) for name, path in artifacts.items()}, indent=2, sort_keys=True))
    return 0


def _read_csv(path: Path) -> list[dict[str, str]]:
    resolved = path if path.exists() else Path(f"{path}.gz")
    opener = gzip.open if resolved.suffix == ".gz" else Path.open
    with opener(resolved, "rt", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
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


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _flatten_record(record: dict[str, Any]) -> dict[str, Any]:
    flattened: dict[str, Any] = {}
    for key, value in record.items():
        if isinstance(value, dict):
            for nested_key, nested_value in value.items():
                flattened[f"{key}_{nested_key}"] = nested_value
        elif isinstance(value, list):
            flattened[key] = json.dumps(value, sort_keys=True)
        else:
            flattened[key] = value
    return flattened


if __name__ == "__main__":
    raise SystemExit(main())
