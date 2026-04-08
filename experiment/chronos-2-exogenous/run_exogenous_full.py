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

try:
    from tqdm.auto import tqdm
except ImportError:
    class tqdm:  # type: ignore[no-redef]
        def __init__(self, *args, **kwargs) -> None:
            del args, kwargs

        def update(self, *args, **kwargs) -> None:
            del args, kwargs

        def set_postfix_str(self, *args, **kwargs) -> None:
            del args, kwargs

        def close(self) -> None:
            return None

        @staticmethod
        def write(message: str) -> None:
            print(message, flush=True)

EXPERIMENT_DIR = Path(__file__).resolve().parent
if str(EXPERIMENT_DIR) not in sys.path:
    sys.path.insert(0, str(EXPERIMENT_DIR))

from chronos2_exogenous import (
    DEFAULT_COVARIATE_STAGES,
    DEFAULT_DATASETS,
    LAYOUT,
    MODEL_ID,
    TARGET_POLICY,
    TASK_ID,
    _GROUP_KEY_COLUMNS,
    _RESULT_COLUMNS,
)


REPO_ROOT = EXPERIMENT_DIR.parents[1]
PYTHON_BIN = EXPERIMENT_DIR / ".conda" / "bin" / "python"
CLI_ENTRYPOINT = EXPERIMENT_DIR / "run_exogenous.py"
FINAL_OUTPUT = REPO_ROOT / "experiment" / "chronos-2-exogenous.csv"
DEFAULT_WORK_ROOT = EXPERIMENT_DIR / ".work"


@dataclass(frozen=True)
class Attempt:
    device: str
    series_budget: int


@dataclass(frozen=True)
class ChunkSpec:
    label: str
    dataset_id: str
    turbine_ids: tuple[str, ...] | None = None


def _timestamp_label() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M%S")


def print_status(message: str) -> None:
    tqdm.write(f"[{datetime.now().strftime('%H:%M:%S')}] {message}")


def expected_result_keys(
    *,
    dataset_ids: Sequence[str] = DEFAULT_DATASETS,
    covariate_stages: Sequence[str] = DEFAULT_COVARIATE_STAGES,
    include_power_only_reference: bool = False,
) -> list[tuple[str, str, str, str]]:
    keys: list[tuple[str, str, str, str]] = []
    for dataset_id in dataset_ids:
        if include_power_only_reference:
            keys.append((dataset_id, LAYOUT, "reference", "power_only"))
        for stage in covariate_stages:
            keys.append((dataset_id, LAYOUT, stage, stage))
    return sorted(keys)


def write_log(log_path: Path, payload: dict[str, object]) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def load_chunk_frame(csv_path: Path) -> pl.DataFrame:
    frame = pl.read_csv(csv_path)
    return frame.select(_RESULT_COLUMNS)


def build_cli_command(
    *,
    dataset_id: str,
    output_path: Path,
    attempt: Attempt,
    include_power_only_reference: bool = False,
    covariate_stages: Sequence[str] = DEFAULT_COVARIATE_STAGES,
    window_offset: int = 0,
    max_windows_per_dataset: int | None = None,
    turbine_ids: Sequence[str] | None = None,
) -> list[str]:
    command = [
        str(PYTHON_BIN),
        str(CLI_ENTRYPOINT),
        "--dataset",
        dataset_id,
        "--device",
        attempt.device,
        "--series-budget",
        str(attempt.series_budget),
        "--output-path",
        str(output_path),
    ]
    for stage in covariate_stages:
        command.extend(["--covariate-stage", stage])
    if include_power_only_reference:
        command.append("--include-power-only-reference")
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
    work_dir: Path,
    primary: Attempt,
    fallback: Attempt | None = None,
    include_power_only_reference: bool = False,
    covariate_stages: Sequence[str] = DEFAULT_COVARIATE_STAGES,
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
            output_path=output_path,
            attempt=attempt,
            include_power_only_reference=include_power_only_reference,
            covariate_stages=covariate_stages,
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

    print_status(f"FALLBACK {label}: {fallback.device} series_budget={fallback.series_budget}")
    fallback_result = _run_attempt(fallback, log_suffix="__fallback")
    if fallback_result.returncode == 0 and output_path.exists():
        return output_path
    raise RuntimeError(
        f"{label} failed on both primary and fallback attempts. "
        f"See {log_dir / f'{label}.json'} and {log_dir / f'{label}__fallback.json'}."
    )


def merge_chunk_results(chunk_paths: Sequence[Path]) -> pl.DataFrame:
    grouped_rows: dict[tuple[object, ...], list[dict[str, object]]] = defaultdict(list)
    for chunk_path in chunk_paths:
        frame = load_chunk_frame(chunk_path)
        for row in frame.to_dicts():
            if int(row["window_count"]) == 0 or int(row["prediction_count"]) == 0:
                continue
            key = tuple(row[column] for column in _GROUP_KEY_COLUMNS)
            grouped_rows[key].append(row)

    merged_rows: list[dict[str, object]] = []
    for rows in grouped_rows.values():
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
                **{column: first[column] for column in _GROUP_KEY_COLUMNS},
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

    return pl.DataFrame(merged_rows).select(_RESULT_COLUMNS).sort(["dataset_id", "covariate_stage", "covariate_pack"])


def validate_final_results(
    frame: pl.DataFrame,
    *,
    dataset_ids: Sequence[str] = DEFAULT_DATASETS,
    covariate_stages: Sequence[str] = DEFAULT_COVARIATE_STAGES,
    include_power_only_reference: bool = False,
) -> None:
    expected_keys = expected_result_keys(
        dataset_ids=dataset_ids,
        covariate_stages=covariate_stages,
        include_power_only_reference=include_power_only_reference,
    )
    if frame.height != len(expected_keys):
        raise RuntimeError(f"Expected {len(expected_keys)} result rows, found {frame.height}.")

    actual_keys = sorted(
        (
            str(row["dataset_id"]),
            str(row["layout"]),
            str(row["covariate_stage"]),
            str(row["covariate_pack"]),
        )
        for row in frame.select(["dataset_id", "layout", "covariate_stage", "covariate_pack"]).to_dicts()
    )
    if actual_keys != expected_keys:
        raise RuntimeError(f"Unexpected result keys: {actual_keys!r}")
    if frame["model_id"].n_unique() != 1 or frame["model_id"][0] != MODEL_ID:
        raise RuntimeError("Final results contain an unexpected model_id.")
    if frame["task_id"].n_unique() != 1 or frame["task_id"][0] != TASK_ID:
        raise RuntimeError("Final results contain an unexpected task_id.")
    if frame["target_policy"].n_unique() != 1 or frame["target_policy"][0] != TARGET_POLICY:
        raise RuntimeError("Final results contain an unexpected target_policy.")
    if frame["layout"].n_unique() != 1 or frame["layout"][0] != LAYOUT:
        raise RuntimeError("Final results contain an unexpected layout.")
    for column in ["window_count", "prediction_count", "mae_kw", "rmse_kw", "mae_pu", "rmse_pu"]:
        if frame[column].null_count() != 0:
            raise RuntimeError(f"Column {column} contains null values.")


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


def build_full_chunk_specs() -> list[ChunkSpec]:
    chunk_specs = [
        ChunkSpec(label=f"{dataset_id}_all_stages", dataset_id=dataset_id)
        for dataset_id in ("kelmarsh", "penmanshiel")
    ]
    chunk_specs.extend(
        ChunkSpec(
            label=f"hill_of_towie_chunk_{chunk_index:02d}",
            dataset_id="hill_of_towie",
            turbine_ids=turbine_ids,
        )
        for chunk_index, turbine_ids in enumerate(build_hill_univariate_chunks(), start=1)
    )
    chunk_specs.extend(
        ChunkSpec(
            label=f"sdwpf_kddcup_chunk_{chunk_index:02d}",
            dataset_id="sdwpf_kddcup",
            turbine_ids=turbine_ids,
        )
        for chunk_index, turbine_ids in enumerate(build_sdwpf_univariate_chunks(), start=1)
    )
    return chunk_specs


def run_full_experiment(
    *,
    work_dir: Path,
    final_output: Path = FINAL_OUTPUT,
    series_budget: int = 1024,
    include_power_only_reference: bool = False,
    covariate_stages: Sequence[str] = DEFAULT_COVARIATE_STAGES,
) -> pl.DataFrame:
    work_dir.mkdir(parents=True, exist_ok=True)
    if final_output.exists():
        backup_path = work_dir / "chronos-2-exogenous.previous.csv"
        if not backup_path.exists():
            shutil.copy2(final_output, backup_path)
            print_status(f"Backed up existing output to {backup_path}")

    chunk_specs = build_full_chunk_specs()
    chunk_paths: list[Path] = []
    progress_bar = tqdm(
        total=len(chunk_specs),
        desc="Chronos-2 exogenous",
        unit="chunk",
        dynamic_ncols=True,
        disable=not sys.stderr.isatty(),
    )
    try:
        for chunk_spec in chunk_specs:
            progress_bar.set_postfix_str(chunk_spec.label)
            chunk_paths.append(
                execute_chunk(
                    label=chunk_spec.label,
                    dataset_id=chunk_spec.dataset_id,
                    work_dir=work_dir,
                    primary=Attempt(device="mps", series_budget=series_budget),
                    fallback=Attempt(device="cpu", series_budget=series_budget),
                    include_power_only_reference=include_power_only_reference,
                    covariate_stages=covariate_stages,
                    turbine_ids=chunk_spec.turbine_ids,
                )
            )
            progress_bar.update(1)
        progress_bar.set_postfix_str("merge")
    finally:
        progress_bar.close()

    merged = merge_chunk_results(chunk_paths)
    validate_final_results(
        merged,
        dataset_ids=DEFAULT_DATASETS,
        covariate_stages=covariate_stages,
        include_power_only_reference=include_power_only_reference,
    )

    temp_output = work_dir / "chronos-2-exogenous.final.csv"
    merged.write_csv(temp_output)
    shutil.copy2(temp_output, final_output)
    print_status(f"Wrote final output to {final_output}")
    return merged


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Safely run the full Chronos-2 exogenous benchmark.")
    parser.add_argument(
        "--work-dir",
        type=Path,
        default=None,
        help="Chunk/log working directory. Defaults to experiment/chronos-2-exogenous/.work/full-run-<timestamp>.",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=FINAL_OUTPUT,
        help="Final merged CSV path. Defaults to experiment/chronos-2-exogenous.csv in the repo root.",
    )
    parser.add_argument(
        "--series-budget",
        type=int,
        default=1024,
        help="Target-plus-covariate series budget used for each runner invocation.",
    )
    parser.add_argument(
        "--include-power-only-reference",
        action="store_true",
        help="Also include one power-only reference row per dataset.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    work_dir = args.work_dir or (DEFAULT_WORK_ROOT / f"full-run-{_timestamp_label()}")
    result = run_full_experiment(
        work_dir=work_dir,
        final_output=args.output_path,
        series_budget=args.series_budget,
        include_power_only_reference=bool(args.include_power_only_reference),
    )
    print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
