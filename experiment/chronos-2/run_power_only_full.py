from __future__ import annotations

import argparse
import json
import math
import shutil
import subprocess
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import sys
from typing import Sequence

import polars as pl

EXPERIMENT_DIR = Path(__file__).resolve().parent
if str(EXPERIMENT_DIR) not in sys.path:
    sys.path.insert(0, str(EXPERIMENT_DIR))

from chronos2_power_only import (
    DEFAULT_DATASETS,
    MODEL_ID,
    MULTIVARIATE_KNN6_SUFFIX,
    _RESULT_COLUMNS,
    TARGET_POLICY,
    TASK_ID,
    UNIVARIATE_SUFFIX,
)


REPO_ROOT = EXPERIMENT_DIR.parents[1]
PYTHON_BIN = EXPERIMENT_DIR / ".conda" / "bin" / "python"
CLI_ENTRYPOINT = EXPERIMENT_DIR / "run_power_only.py"
FINAL_OUTPUT = REPO_ROOT / "experiment" / "chronos-2.csv"
DEFAULT_WORK_ROOT = EXPERIMENT_DIR / ".work"


@dataclass(frozen=True)
class Attempt:
    device: str
    batch_size: int


def _timestamp_label() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M%S")


def print_status(message: str) -> None:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {message}", flush=True)


def expected_dataset_ids() -> list[str]:
    rows: list[str] = []
    for dataset_id in DEFAULT_DATASETS:
        rows.append(f"{dataset_id}{MULTIVARIATE_KNN6_SUFFIX}")
        rows.append(f"{dataset_id}{UNIVARIATE_SUFFIX}")
    return sorted(rows)


def write_log(log_path: Path, payload: dict[str, object]) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def load_chunk_frame(csv_path: Path) -> pl.DataFrame:
    frame = pl.read_csv(csv_path)
    return frame.select(_RESULT_COLUMNS)


def build_cli_command(
    *,
    dataset_id: str,
    mode: str,
    output_path: Path,
    attempt: Attempt,
    window_offset: int = 0,
    max_windows_per_dataset: int | None = None,
    turbine_ids: Sequence[str] | None = None,
) -> list[str]:
    command = [
        str(PYTHON_BIN),
        str(CLI_ENTRYPOINT),
        "--dataset",
        dataset_id,
        "--mode",
        mode,
        "--device",
        attempt.device,
        "--batch-size",
        str(attempt.batch_size),
        "--output-path",
        str(output_path),
    ]
    if window_offset:
        command.extend(["--window-offset", str(window_offset)])
    if max_windows_per_dataset is not None:
        command.extend(["--max-windows-per-dataset", str(max_windows_per_dataset)])
    if turbine_ids:
        for turbine_id in turbine_ids:
            command.extend(["--turbine-id", turbine_id])
    return command


def execute_chunk(
    *,
    label: str,
    dataset_id: str,
    mode: str,
    work_dir: Path,
    primary: Attempt,
    fallback: Attempt | None = None,
    window_offset: int = 0,
    max_windows_per_dataset: int | None = None,
    turbine_ids: Sequence[str] | None = None,
) -> Path:
    chunk_dir = work_dir / "chunks"
    log_dir = work_dir / "logs"
    chunk_dir.mkdir(parents=True, exist_ok=True)
    output_path = chunk_dir / f"{label}.csv"

    if output_path.exists():
        print_status(f"REUSE {label}: {output_path}")
        return output_path

    def _run_attempt(attempt: Attempt, *, log_suffix: str = "") -> subprocess.CompletedProcess[str]:
        command = build_cli_command(
            dataset_id=dataset_id,
            mode=mode,
            output_path=output_path,
            attempt=attempt,
            window_offset=window_offset,
            max_windows_per_dataset=max_windows_per_dataset,
            turbine_ids=turbine_ids,
        )
        print_status(f"RUN {label}{log_suffix}: {' '.join(command)}")
        started_at = datetime.now().isoformat(timespec="seconds")
        result = subprocess.run(command, cwd=REPO_ROOT, capture_output=True, text=True)
        finished_at = datetime.now().isoformat(timespec="seconds")
        write_log(
            log_dir / f"{label}{log_suffix}.json",
            {
                "label": f"{label}{log_suffix}",
                "dataset_id": dataset_id,
                "mode": mode,
                "command": command,
                "returncode": result.returncode,
                "started_at": started_at,
                "finished_at": finished_at,
                "stdout": result.stdout,
                "stderr": result.stderr,
            },
        )
        return result

    primary_result = _run_attempt(primary)
    if primary_result.returncode == 0 and output_path.exists():
        return output_path
    if fallback is None:
        raise RuntimeError(f"{label} failed. See {log_dir / f'{label}.json'}")

    print_status(f"FALLBACK {label}: {fallback.device} batch_size={fallback.batch_size}")
    fallback_result = _run_attempt(fallback, log_suffix="__fallback")
    if fallback_result.returncode == 0 and output_path.exists():
        return output_path
    raise RuntimeError(
        f"{label} failed on both primary and fallback attempts. "
        f"See {log_dir / f'{label}.json'} and {log_dir / f'{label}__fallback.json'}."
    )


def merge_chunk_results(chunk_paths: Sequence[Path]) -> pl.DataFrame:
    grouped_rows: dict[str, list[dict[str, object]]] = defaultdict(list)
    for chunk_path in chunk_paths:
        frame = load_chunk_frame(chunk_path)
        for row in frame.to_dicts():
            if int(row["window_count"]) == 0 or int(row["prediction_count"]) == 0:
                continue
            grouped_rows[str(row["dataset_id"])].append(row)

    merged_rows: list[dict[str, object]] = []
    for dataset_id, rows in grouped_rows.items():
        first = rows[0]
        prediction_count = sum(int(row["prediction_count"]) for row in rows)
        window_count = sum(int(row["window_count"]) for row in rows)
        abs_error_sum = sum(float(row["mae_kw"]) * int(row["prediction_count"]) for row in rows)
        squared_error_sum = sum((float(row["rmse_kw"]) ** 2) * int(row["prediction_count"]) for row in rows)
        normalized_abs_error_sum = sum(float(row["mae_pu"]) * int(row["prediction_count"]) for row in rows)
        normalized_squared_error_sum = sum((float(row["rmse_pu"]) ** 2) * int(row["prediction_count"]) for row in rows)
        devices = sorted({str(row["device"]) for row in rows})
        start_timestamps = [str(row["start_timestamp"]) for row in rows if row["start_timestamp"] is not None]
        end_timestamps = [str(row["end_timestamp"]) for row in rows if row["end_timestamp"] is not None]
        merged_rows.append(
            {
                "dataset_id": dataset_id,
                "model_id": first["model_id"],
                "task_id": first["task_id"],
                "history_steps": int(first["history_steps"]),
                "forecast_steps": int(first["forecast_steps"]),
                "stride_steps": int(first["stride_steps"]),
                "target_policy": first["target_policy"],
                "window_count": window_count,
                "prediction_count": prediction_count,
                "start_timestamp": min(start_timestamps) if start_timestamps else None,
                "end_timestamp": max(end_timestamps) if end_timestamps else None,
                "mae_kw": abs_error_sum / prediction_count,
                "rmse_kw": math.sqrt(squared_error_sum / prediction_count),
                "mae_pu": normalized_abs_error_sum / prediction_count,
                "rmse_pu": math.sqrt(normalized_squared_error_sum / prediction_count),
                "device": devices[0] if len(devices) == 1 else ",".join(devices),
                "runtime_seconds": round(sum(float(row["runtime_seconds"]) for row in rows), 6),
            }
        )

    return pl.DataFrame(merged_rows).select(_RESULT_COLUMNS).sort("dataset_id")


def validate_final_results(frame: pl.DataFrame) -> None:
    expected_ids = expected_dataset_ids()
    if frame.height != len(expected_ids):
        raise RuntimeError(f"Expected {len(expected_ids)} result rows, found {frame.height}.")
    actual_ids = frame["dataset_id"].to_list()
    if actual_ids != expected_ids:
        raise RuntimeError(f"Unexpected dataset ids: {actual_ids!r}")
    if frame["model_id"].n_unique() != 1 or frame["model_id"][0] != MODEL_ID:
        raise RuntimeError("Final results contain an unexpected model_id.")
    if frame["task_id"].n_unique() != 1 or frame["task_id"][0] != TASK_ID:
        raise RuntimeError("Final results contain an unexpected task_id.")
    if frame["target_policy"].n_unique() != 1 or frame["target_policy"][0] != TARGET_POLICY:
        raise RuntimeError("Final results contain an unexpected target_policy.")
    for column in ["window_count", "prediction_count", "mae_kw", "rmse_kw", "mae_pu", "rmse_pu"]:
        if frame[column].null_count() != 0:
            raise RuntimeError(f"Column {column} contains null values.")
    if any(dataset_id.endswith("_multivariate") for dataset_id in actual_ids):
        raise RuntimeError("Old full multivariate rows are still present in the final output.")


def run_window_chunks(
    *,
    dataset_id: str,
    mode: str,
    work_dir: Path,
    primary: Attempt,
    fallback: Attempt,
    windows_per_chunk: int,
) -> list[Path]:
    chunk_paths: list[Path] = []
    offset = 0
    while True:
        label = f"{dataset_id}_{mode}_chunk_{offset:05d}"
        chunk_path = execute_chunk(
            label=label,
            dataset_id=dataset_id,
            mode=mode,
            work_dir=work_dir,
            primary=primary,
            fallback=fallback,
            window_offset=offset,
            max_windows_per_dataset=windows_per_chunk,
        )
        chunk_frame = load_chunk_frame(chunk_path)
        if chunk_frame.height != 1:
            raise RuntimeError(f"Expected a single-row chunk in {chunk_path}, found {chunk_frame.height}.")
        row = chunk_frame.to_dicts()[0]
        if int(row["window_count"]) == 0 or int(row["prediction_count"]) == 0:
            print_status(f"STOP {dataset_id} {mode} at offset {offset}: no more scored windows.")
            break
        chunk_paths.append(chunk_path)
        offset += windows_per_chunk
    return chunk_paths


def build_hill_univariate_chunks() -> list[tuple[str, ...]]:
    return [
        ("T01", "T02", "T03"),
        ("T04", "T05", "T06"),
        ("T07", "T08", "T09"),
        ("T10", "T11", "T12"),
        ("T13", "T14", "T15"),
        ("T16", "T17", "T18"),
        ("T19", "T20", "T21"),
    ]


def build_sdwpf_univariate_chunks() -> list[tuple[str, ...]]:
    turbine_ids = [str(index) for index in range(1, 135)]
    return [tuple(turbine_ids[index : index + 16]) for index in range(0, len(turbine_ids), 16)]


def build_hill_multivariate_target_groups() -> list[tuple[str, ...]]:
    return [(f"T{index:02d}",) for index in range(1, 22)]


def build_sdwpf_multivariate_target_groups() -> list[tuple[str, ...]]:
    turbine_ids = [str(index) for index in range(1, 135)]
    return [tuple(turbine_ids[index : index + 8]) for index in range(0, len(turbine_ids), 8)]


def run_target_group_chunks(
    *,
    dataset_id: str,
    mode: str,
    work_dir: Path,
    primary: Attempt,
    fallback: Attempt,
    target_groups: Sequence[Sequence[str]],
) -> list[Path]:
    chunk_paths: list[Path] = []

    def _run_group(group: tuple[str, ...], label: str) -> list[Path]:
        try:
            return [
                execute_chunk(
                    label=label,
                    dataset_id=dataset_id,
                    mode=mode,
                    work_dir=work_dir,
                    primary=primary,
                    fallback=fallback,
                    turbine_ids=group,
                )
            ]
        except RuntimeError:
            if len(group) == 1:
                raise
            midpoint = len(group) // 2
            left = group[:midpoint]
            right = group[midpoint:]
            print_status(f"SPLIT {label}: {group} -> {left} | {right}")
            return _run_group(left, f"{label}a") + _run_group(right, f"{label}b")

    for group_index, target_group in enumerate(target_groups, start=1):
        chunk_paths.extend(
            _run_group(tuple(target_group), f"{dataset_id}_{mode}_targets_{group_index:03d}")
        )
    return chunk_paths


def run_full_experiment(
    *,
    work_dir: Path,
    final_output: Path = FINAL_OUTPUT,
) -> pl.DataFrame:
    work_dir.mkdir(parents=True, exist_ok=True)
    if final_output.exists():
        backup_path = work_dir / "chronos-2.previous.csv"
        if not backup_path.exists():
            shutil.copy2(final_output, backup_path)
            print_status(f"Backed up existing output to {backup_path}")

    chunk_paths: list[Path] = []

    for dataset_id in ("kelmarsh", "penmanshiel"):
        chunk_paths.append(
            execute_chunk(
                label=f"{dataset_id}_all",
                dataset_id=dataset_id,
                mode="all",
                work_dir=work_dir,
                primary=Attempt(device="mps", batch_size=32),
                fallback=Attempt(device="cpu", batch_size=16),
            )
        )

    for chunk_index, turbine_ids in enumerate(build_hill_univariate_chunks(), start=1):
        chunk_paths.append(
            execute_chunk(
                label=f"hill_of_towie_univariate_chunk_{chunk_index:02d}",
                dataset_id="hill_of_towie",
                mode="univariate",
                work_dir=work_dir,
                primary=Attempt(device="mps", batch_size=4),
                fallback=Attempt(device="cpu", batch_size=4),
                turbine_ids=turbine_ids,
            )
        )

    for chunk_index, turbine_ids in enumerate(build_sdwpf_univariate_chunks(), start=1):
        chunk_paths.append(
            execute_chunk(
                label=f"sdwpf_kddcup_univariate_chunk_{chunk_index:02d}",
                dataset_id="sdwpf_kddcup",
                mode="univariate",
                work_dir=work_dir,
                primary=Attempt(device="mps", batch_size=16),
                fallback=Attempt(device="cpu", batch_size=8),
                turbine_ids=turbine_ids,
            )
        )

    chunk_paths.extend(
        run_target_group_chunks(
            dataset_id="hill_of_towie",
            mode="multivariate_knn6",
            work_dir=work_dir,
            primary=Attempt(device="mps", batch_size=4),
            fallback=Attempt(device="cpu", batch_size=1),
            target_groups=build_hill_multivariate_target_groups(),
        )
    )
    chunk_paths.extend(
        run_target_group_chunks(
            dataset_id="sdwpf_kddcup",
            mode="multivariate_knn6",
            work_dir=work_dir,
            primary=Attempt(device="mps", batch_size=4),
            fallback=Attempt(device="cpu", batch_size=1),
            target_groups=build_sdwpf_multivariate_target_groups(),
        )
    )

    merged = merge_chunk_results(chunk_paths)
    validate_final_results(merged)

    temp_output = work_dir / "chronos-2.final.csv"
    merged.write_csv(temp_output)
    shutil.copy2(temp_output, final_output)
    print_status(f"Wrote final output to {final_output}")
    return merged


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Safely run the full Chronos-2 power_only benchmark.")
    parser.add_argument(
        "--work-dir",
        type=Path,
        default=None,
        help="Chunk/log working directory. Defaults to experiment/chronos-2/.work/full-run-<timestamp>.",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=FINAL_OUTPUT,
        help="Final merged CSV path. Defaults to experiment/chronos-2.csv in the repo root.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    work_dir = args.work_dir or (DEFAULT_WORK_ROOT / f"full-run-{_timestamp_label()}")
    result = run_full_experiment(work_dir=work_dir, final_output=args.output_path)
    print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
