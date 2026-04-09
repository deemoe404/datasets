from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import subprocess
import sys
from typing import Sequence

import polars as pl

EXPERIMENT_DIR = Path(__file__).resolve().parent
if str(EXPERIMENT_DIR) not in sys.path:
    sys.path.insert(0, str(EXPERIMENT_DIR))

from ltsf_linear import (  # noqa: E402
    DEFAULT_COVARIATE_STAGES,
    DEFAULT_DATASETS,
    MODEL_ID,
    MODEL_VARIANTS,
    REFERENCE_STAGE,
    SPLIT_PROTOCOL,
    TASK_ID,
    _RESULT_COLUMNS,
    build_requested_packs,
    resolve_device,
    sort_result_frame,
)


REPO_ROOT = EXPERIMENT_DIR.parents[1]
PYTHON_BIN = EXPERIMENT_DIR / ".conda" / "bin" / "python"
CLI_ENTRYPOINT = EXPERIMENT_DIR / "run_ltsf_linear.py"
FINAL_OUTPUT = REPO_ROOT / "experiment" / "ltsf-linear.csv"
DEFAULT_WORK_ROOT = EXPERIMENT_DIR / ".work"


@dataclass(frozen=True)
class Attempt:
    device: str


@dataclass(frozen=True)
class JobSpec:
    label: str
    dataset_id: str
    covariate_stage: str
    covariate_pack: str
    model_variant: str


def print_status(message: str) -> None:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {message}", flush=True)


def build_job_specs() -> tuple[JobSpec, ...]:
    return tuple(
        JobSpec(
            label=f"{dataset_id}_{pack.stage}_{pack.pack_name}_{model_variant}",
            dataset_id=dataset_id,
            covariate_stage=pack.stage,
            covariate_pack=pack.pack_name,
            model_variant=model_variant,
        )
        for dataset_id in DEFAULT_DATASETS
        for pack in build_requested_packs(
            dataset_id,
            covariate_stages=DEFAULT_COVARIATE_STAGES,
            include_power_only_reference=True,
        )
        for model_variant in MODEL_VARIANTS
    )


def expected_job_keys() -> list[tuple[str, str, str, str]]:
    return [
        (job.dataset_id, job.covariate_stage, job.covariate_pack, job.model_variant)
        for job in build_job_specs()
    ]


def resolve_attempts(device: str | None = None) -> tuple[Attempt, ...]:
    resolved_device = resolve_device(device)
    primary = Attempt(device=resolved_device)
    if resolved_device == "cpu":
        return (primary,)
    return (primary, Attempt(device="cpu"))


def write_log(log_path: Path, payload: dict[str, object]) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def build_cli_command(
    *,
    dataset_id: str,
    covariate_stage: str,
    model_variant: str,
    output_path: Path,
    attempt: Attempt,
    epochs: int | None = None,
    max_windows_per_split: int | None = None,
) -> list[str]:
    command = [
        str(PYTHON_BIN),
        str(CLI_ENTRYPOINT),
        "--dataset",
        dataset_id,
        "--model",
        model_variant,
        "--device",
        attempt.device,
        "--output-path",
        str(output_path),
    ]
    if covariate_stage == REFERENCE_STAGE:
        command.append("--reference-only")
    else:
        command.extend(["--covariate-stage", covariate_stage, "--no-power-only-reference"])
    if epochs is not None:
        command.extend(["--epochs", str(epochs)])
    if max_windows_per_split is not None:
        command.extend(["--max-windows-per-split", str(max_windows_per_split)])
    return command


def execute_job(
    *,
    job: JobSpec,
    work_dir: Path,
    attempts: Sequence[Attempt],
    epochs: int | None = None,
    max_windows_per_split: int | None = None,
) -> Path:
    chunk_dir = work_dir / "chunks"
    log_dir = work_dir / "logs"
    chunk_dir.mkdir(parents=True, exist_ok=True)
    output_path = chunk_dir / f"{job.label}.csv"

    if output_path.exists():
        print_status(f"REUSE {job.label}: {output_path}")
        return output_path

    for attempt_index, attempt in enumerate(attempts):
        suffix = "" if attempt_index == 0 else f"__retry_{attempt.device}"
        command = build_cli_command(
            dataset_id=job.dataset_id,
            covariate_stage=job.covariate_stage,
            model_variant=job.model_variant,
            output_path=output_path,
            attempt=attempt,
            epochs=epochs,
            max_windows_per_split=max_windows_per_split,
        )
        print_status(f"RUN {job.label}{suffix}: {' '.join(command)}")
        started_at = datetime.now().isoformat(timespec="seconds")
        result = subprocess.run(command, cwd=REPO_ROOT, capture_output=True, text=True)
        finished_at = datetime.now().isoformat(timespec="seconds")
        write_log(
            log_dir / f"{job.label}{suffix}.json",
            {
                "label": f"{job.label}{suffix}",
                "dataset_id": job.dataset_id,
                "covariate_stage": job.covariate_stage,
                "covariate_pack": job.covariate_pack,
                "model_variant": job.model_variant,
                "command": command,
                "returncode": result.returncode,
                "started_at": started_at,
                "finished_at": finished_at,
                "stdout": result.stdout,
                "stderr": result.stderr,
            },
        )
        if result.returncode == 0 and output_path.exists():
            return output_path

    raise RuntimeError(f"{job.label} failed. See logs under {log_dir}.")


def load_chunk_frame(csv_path: Path) -> pl.DataFrame:
    return pl.read_csv(csv_path).select(_RESULT_COLUMNS)


def merge_chunk_results(chunk_paths: Sequence[Path]) -> pl.DataFrame:
    frame = pl.concat([load_chunk_frame(path) for path in chunk_paths], how="vertical")
    return sort_result_frame(frame.select(_RESULT_COLUMNS))


def validate_final_results(frame: pl.DataFrame) -> None:
    expected = expected_job_keys()
    if frame.height != len(expected):
        raise RuntimeError(f"Expected {len(expected)} result rows, found {frame.height}.")
    actual = list(
        zip(
            frame["dataset_id"].to_list(),
            frame["covariate_stage"].to_list(),
            frame["covariate_pack"].to_list(),
            frame["model_variant"].to_list(),
            strict=True,
        )
    )
    if actual != expected:
        raise RuntimeError(f"Unexpected dataset/stage/pack/model rows: {actual!r}")
    if frame["model_id"].n_unique() != 1 or frame["model_id"][0] != MODEL_ID:
        raise RuntimeError("Final results contain an unexpected model_id.")
    if frame["task_id"].n_unique() != 1 or frame["task_id"][0] != TASK_ID:
        raise RuntimeError("Final results contain an unexpected task_id.")
    if frame["split_protocol"].n_unique() != 1 or frame["split_protocol"][0] != SPLIT_PROTOCOL:
        raise RuntimeError("Final results contain an unexpected split_protocol.")
    for column in (
        "feature_set",
        "covariate_count",
        "covariate_policy",
        "window_count",
        "prediction_count",
        "mae_kw",
        "rmse_kw",
        "mae_pu",
        "rmse_pu",
        "best_epoch",
        "epochs_ran",
        "best_val_rmse_pu",
    ):
        if frame[column].null_count() != 0:
            raise RuntimeError(f"Column {column} contains null values.")


def run_full_experiment(
    *,
    device: str | None = None,
    output_path: str | Path = FINAL_OUTPUT,
    work_dir: str | Path = DEFAULT_WORK_ROOT,
    epochs: int | None = None,
    max_windows_per_split: int | None = None,
) -> pl.DataFrame:
    work_dir_path = Path(work_dir)
    attempts = resolve_attempts(device)
    chunk_paths = [
        execute_job(
            job=job,
            work_dir=work_dir_path,
            attempts=attempts,
            epochs=epochs,
            max_windows_per_split=max_windows_per_split,
        )
        for job in build_job_specs()
    ]
    results = merge_chunk_results(chunk_paths)
    validate_final_results(results)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    results.write_csv(output)
    return results


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the full LTSF-Linear benchmark grid.")
    parser.add_argument(
        "--device",
        choices=("auto", "cuda", "mps", "cpu"),
        default="auto",
        help="Primary device. Defaults to auto (cuda -> mps -> cpu).",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=FINAL_OUTPUT,
        help="Final merged CSV path. Defaults to experiment/ltsf-linear.csv in the repo root.",
    )
    parser.add_argument(
        "--work-dir",
        type=Path,
        default=DEFAULT_WORK_ROOT,
        help="Work directory for chunk CSVs and logs.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Optional epoch override passed through to each job.",
    )
    parser.add_argument(
        "--max-windows-per-split",
        type=int,
        default=None,
        help="Optional smoke-test limit passed through to each job.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    run_full_experiment(
        device=args.device,
        output_path=args.output_path,
        work_dir=args.work_dir,
        epochs=args.epochs,
        max_windows_per_split=args.max_windows_per_split,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
