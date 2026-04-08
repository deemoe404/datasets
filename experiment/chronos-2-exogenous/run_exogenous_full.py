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
from typing import Any, Mapping, Sequence

import polars as pl

try:
    from tqdm.auto import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False

    class tqdm:  # type: ignore[no-redef]
        def __init__(self, *args, **kwargs) -> None:
            del args, kwargs
            self.n = 0
            self.total = None

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
    PROFILE_LOG_PREFIX,
    select_device,
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
FIXED_NON_CPU_SERIES_BUDGETS = (1024, 768, 512)
CPU_FALLBACK_SERIES_BUDGET = FIXED_NON_CPU_SERIES_BUDGETS[0]
PROGRESS_PHASES = {
    "progress_chunk_plan",
    "progress_stage_start",
    "progress_batch",
    "progress_stage_complete",
}


@dataclass(frozen=True)
class Attempt:
    device: str
    series_budget: int


@dataclass(frozen=True)
class ChunkSpec:
    label: str
    dataset_id: str
    turbine_ids: tuple[str, ...] | None = None


def resolve_full_run_device(device: str | None = None) -> str:
    return device or select_device()


def resolve_attempts(
    *,
    device: str | None = None,
    series_budget: int,
) -> tuple[Attempt, ...]:
    resolved_device = resolve_full_run_device(device)
    if resolved_device == "cpu":
        return (Attempt(device="cpu", series_budget=series_budget),)
    return tuple(
        [Attempt(device=resolved_device, series_budget=budget) for budget in FIXED_NON_CPU_SERIES_BUDGETS]
        + [Attempt(device="cpu", series_budget=CPU_FALLBACK_SERIES_BUDGET)]
    )


def _timestamp_label() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M%S")


def print_status(message: str) -> None:
    tqdm.write(f"[{datetime.now().strftime('%H:%M:%S')}] {message}")


def format_non_cpu_retry_budgets() -> str:
    return " -> ".join(str(budget) for budget in FIXED_NON_CPU_SERIES_BUDGETS)


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
    emit_progress_events: bool = False,
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
    if emit_progress_events:
        command.append("--emit-progress-events")
    return command


def progress_is_enabled() -> bool:
    return HAS_TQDM and sys.stderr.isatty()


def parse_progress_event(stderr_line: str) -> dict[str, Any] | None:
    if not stderr_line.startswith(PROFILE_LOG_PREFIX):
        return None
    try:
        payload = json.loads(stderr_line[len(PROFILE_LOG_PREFIX) :].strip())
    except json.JSONDecodeError:
        return None
    if payload.get("phase") not in PROGRESS_PHASES:
        return None
    return payload


def format_progress_postfix(payload: Mapping[str, object], attempt: Attempt) -> str:
    parts = [f"{attempt.device}/{attempt.series_budget}"]
    stage = payload.get("stage")
    if stage:
        parts.append(f"stage={stage}")
    turbine_id = payload.get("turbine_id")
    if turbine_id:
        parts.append(f"turbine={turbine_id}")
    return " ".join(str(part) for part in parts)


def execute_chunk(
    *,
    label: str,
    dataset_id: str,
    work_dir: Path,
    attempts: Sequence[Attempt],
    include_power_only_reference: bool = False,
    covariate_stages: Sequence[str] = DEFAULT_COVARIATE_STAGES,
    window_offset: int = 0,
    max_windows_per_dataset: int | None = None,
    turbine_ids: Sequence[str] | None = None,
    progress_enabled: bool = False,
) -> Path:
    chunk_dir = work_dir / "chunks"
    log_dir = work_dir / "logs"
    chunk_dir.mkdir(parents=True, exist_ok=True)
    output_path = chunk_dir / f"{label}.csv"

    if output_path.exists():
        print_status(f"REUSE {label}: {output_path}")
        return output_path

    if not attempts:
        raise ValueError("execute_chunk requires at least one attempt.")

    chunk_progress_bar: Any | None = None

    def _close_chunk_progress_bar() -> None:
        nonlocal chunk_progress_bar
        if chunk_progress_bar is not None:
            chunk_progress_bar.close()
            chunk_progress_bar = None

    def _ensure_chunk_progress_bar(total_batches: int) -> Any | None:
        nonlocal chunk_progress_bar
        if not progress_enabled:
            return None
        if chunk_progress_bar is None:
            chunk_progress_bar = tqdm(
                total=total_batches,
                desc=label,
                unit="batch",
                dynamic_ncols=True,
                leave=False,
                position=1,
                disable=not progress_enabled,
            )
        elif getattr(chunk_progress_bar, "total", None) != total_batches:
            chunk_progress_bar.total = total_batches
            refresh = getattr(chunk_progress_bar, "refresh", None)
            if callable(refresh):
                refresh()
        return chunk_progress_bar

    def _handle_progress_event(payload: dict[str, Any], attempt: Attempt) -> None:
        total_batches = int(payload.get("chunk_total_batches", 0) or 0)
        progress_bar = _ensure_chunk_progress_bar(total_batches)
        if progress_bar is None:
            return
        if payload.get("phase") == "progress_batch":
            completed_batches = int(payload.get("completed_chunk_batches", 0) or 0)
            current_batches = int(getattr(progress_bar, "n", 0))
            if completed_batches > current_batches:
                progress_bar.update(completed_batches - current_batches)
        progress_bar.set_postfix_str(format_progress_postfix(payload, attempt))

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
            emit_progress_events=progress_enabled,
        )
        print_status(f"RUN {label}{log_suffix}: {' '.join(command)}")
        started_at = datetime.now().isoformat(timespec="seconds")
        process = subprocess.Popen(
            command,
            cwd=REPO_ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        stdout_text = ""
        stderr_lines: list[str] = []
        assert process.stderr is not None
        for stderr_line in process.stderr:
            payload = parse_progress_event(stderr_line)
            if payload is not None:
                _handle_progress_event(payload, attempt)
                continue
            stderr_lines.append(stderr_line)
        if process.stdout is not None:
            stdout_text = process.stdout.read()
            process.stdout.close()
        process.stderr.close()
        returncode = process.wait()
        finished_at = datetime.now().isoformat(timespec="seconds")
        result = subprocess.CompletedProcess(
            command,
            returncode,
            stdout=stdout_text,
            stderr="".join(stderr_lines),
        )
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

    attempted_log_paths: list[Path] = []
    for attempt_index, attempt in enumerate(attempts):
        _close_chunk_progress_bar()
        if attempt_index == 0:
            log_suffix = ""
        elif attempt.device == "cpu":
            log_suffix = "__fallback_cpu"
            print_status(f"FALLBACK {label}: {attempt.device} series_budget={attempt.series_budget}")
        else:
            log_suffix = f"__retry_{attempt_index:02d}"
            print_status(f"RETRY {label}: {attempt.device} series_budget={attempt.series_budget}")

        log_path = log_dir / f"{label}{log_suffix}.json"
        attempted_log_paths.append(log_path)
        result = _run_attempt(attempt, log_suffix=log_suffix)
        if result.returncode == 0 and output_path.exists():
            _close_chunk_progress_bar()
            return output_path

    _close_chunk_progress_bar()
    attempted_logs = ", ".join(str(path) for path in attempted_log_paths)
    if len(attempted_log_paths) == 1:
        raise RuntimeError(f"{label} failed. See {attempted_logs}.")
    raise RuntimeError(f"{label} failed on all attempts. See {attempted_logs}.")


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
    series_budget: int = CPU_FALLBACK_SERIES_BUDGET,
    include_power_only_reference: bool = False,
    covariate_stages: Sequence[str] = DEFAULT_COVARIATE_STAGES,
    device: str | None = None,
) -> pl.DataFrame:
    work_dir.mkdir(parents=True, exist_ok=True)
    if final_output.exists():
        backup_path = work_dir / "chronos-2-exogenous.previous.csv"
        if not backup_path.exists():
            shutil.copy2(final_output, backup_path)
            print_status(f"Backed up existing output to {backup_path}")

    resolved_device = resolve_full_run_device(device)
    attempts = resolve_attempts(device=resolved_device, series_budget=series_budget)
    if resolved_device == "cpu":
        print_status(f"Using cpu for full run without fallback (series_budget={attempts[0].series_budget}).")
    else:
        retry_budgets = format_non_cpu_retry_budgets()
        print_status(
            f"Using {resolved_device} for full run with retry budgets {retry_budgets} before cpu fallback {CPU_FALLBACK_SERIES_BUDGET}."
        )
        if series_budget != CPU_FALLBACK_SERIES_BUDGET:
            print_status(
                f"Ignoring requested series_budget={series_budget} for non-cpu full run; "
                f"using fixed retry budgets {retry_budgets} before cpu fallback {CPU_FALLBACK_SERIES_BUDGET}."
            )

    chunk_specs = build_full_chunk_specs()
    chunk_paths: list[Path] = []
    nested_progress_enabled = progress_is_enabled()
    progress_bar = tqdm(
        total=len(chunk_specs),
        desc="Chronos-2 exogenous",
        unit="chunk",
        dynamic_ncols=True,
        position=0,
        disable=not nested_progress_enabled,
    )
    try:
        for chunk_spec in chunk_specs:
            progress_bar.set_postfix_str(chunk_spec.label)
            chunk_paths.append(
                execute_chunk(
                    label=chunk_spec.label,
                    dataset_id=chunk_spec.dataset_id,
                    work_dir=work_dir,
                    attempts=attempts,
                    include_power_only_reference=include_power_only_reference,
                    covariate_stages=covariate_stages,
                    turbine_ids=chunk_spec.turbine_ids,
                    progress_enabled=nested_progress_enabled,
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
        default=CPU_FALLBACK_SERIES_BUDGET,
        help=(
            "Series budget for cpu full runs. Non-cpu full runs ignore this flag and use fixed retry budgets "
            "1024 -> 768 -> 512 before cpu fallback 1024."
        ),
    )
    parser.add_argument(
        "--include-power-only-reference",
        action="store_true",
        help="Also include one power-only reference row per dataset.",
    )
    parser.add_argument(
        "--device",
        choices=["cuda", "mps", "cpu"],
        default=None,
        help="Override automatic device selection for full-run chunks.",
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
        device=args.device,
    )
    print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
