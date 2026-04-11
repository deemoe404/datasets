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
try:
    from tqdm.auto import tqdm

    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False

    class tqdm:  # type: ignore[no-redef]
        def __init__(self, *args, **kwargs) -> None:
            del args
            self.total = kwargs.get("total")
            self.desc = kwargs.get("desc")
            self.disable = kwargs.get("disable", False)
            self.n = 0

        def update(self, value=1) -> None:
            self.n += int(value)

        def set_postfix_str(self, value: str) -> None:
            del value

        def close(self) -> None:
            return None

        @staticmethod
        def write(message: str) -> None:
            print(message, flush=True)

EXPERIMENT_DIR = Path(__file__).resolve().parent
if str(EXPERIMENT_DIR) not in sys.path:
    sys.path.insert(0, str(EXPERIMENT_DIR))
EXPERIMENT_ROOT = EXPERIMENT_DIR.parents[1]
COMMON_DIR = EXPERIMENT_ROOT / "infra" / "common"
if str(COMMON_DIR) not in sys.path:
    sys.path.insert(0, str(COMMON_DIR))

from ltsf_linear import (  # noqa: E402
    DEFAULT_COVARIATE_STAGES,
    DEFAULT_DATASETS,
    FORECAST_STEPS,
    HORIZON_METRIC_SCOPE,
    MODEL_ID,
    MODEL_VARIANTS,
    NON_OVERLAP_EVAL_PROTOCOL,
    OVERALL_METRIC_SCOPE,
    REFERENCE_STAGE,
    ROLLING_EVAL_PROTOCOL,
    SPLIT_PROTOCOL,
    TASK_ID,
    WINDOW_PROTOCOL,
    _RESULT_COLUMNS,
    build_requested_packs,
    resolve_device,
    sort_result_frame,
)
from run_records import record_cli_run, resolve_family_feature_protocol_ids  # noqa: E402
from published_outputs import default_family_output_path  # noqa: E402


REPO_ROOT = EXPERIMENT_ROOT.parent
FAMILY_ID = "ltsf_linear_local"
PYTHON_BIN = EXPERIMENT_DIR / ".conda" / "bin" / "python"
CLI_ENTRYPOINT = EXPERIMENT_DIR / "run_ltsf_linear.py"
FINAL_OUTPUT = default_family_output_path(repo_root=REPO_ROOT, family_id=FAMILY_ID)
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
    tqdm.write(f"[{datetime.now().strftime('%H:%M:%S')}] {message}")


def _timestamp_label() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M%S")


def progress_is_enabled() -> bool:
    return HAS_TQDM and sys.stderr.isatty()


def _create_progress_bar(*, total: int | None, desc: str, leave: bool = False):
    return tqdm(
        total=total,
        desc=desc,
        leave=leave,
        disable=not progress_is_enabled(),
        dynamic_ncols=True,
    )


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


def expected_result_keys() -> list[tuple[str, str, str, str, str, str, str, int | None]]:
    keys: list[tuple[str, str, str, str, str, str, str, int | None]] = []
    for job in build_job_specs():
        for split_name in ("val", "test"):
            for eval_protocol in (ROLLING_EVAL_PROTOCOL, NON_OVERLAP_EVAL_PROTOCOL):
                keys.append(
                    (
                        job.dataset_id,
                        job.covariate_stage,
                        job.covariate_pack,
                        job.model_variant,
                        split_name,
                        eval_protocol,
                        OVERALL_METRIC_SCOPE,
                        None,
                    )
                )
                for lead_step in range(1, FORECAST_STEPS + 1):
                    keys.append(
                        (
                            job.dataset_id,
                            job.covariate_stage,
                            job.covariate_pack,
                            job.model_variant,
                            split_name,
                            eval_protocol,
                            HORIZON_METRIC_SCOPE,
                            lead_step,
                        )
                    )
    return keys


def resolve_attempts(device: str | None = None) -> tuple[Attempt, ...]:
    resolved_device = resolve_device(device)
    primary = Attempt(device=resolved_device)
    if resolved_device == "cpu":
        return (primary,)
    return (primary, Attempt(device="cpu"))


def write_log(log_path: Path, payload: dict[str, object]) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def chunk_is_reusable(csv_path: Path) -> bool:
    if not csv_path.exists():
        return False
    try:
        frame = pl.read_csv(
            csv_path,
            n_rows=1,
            schema_overrides={
                "lead_step": pl.Int64,
                "lead_minutes": pl.Int64,
            },
        )
    except Exception:
        return False
    return set(_RESULT_COLUMNS).issubset(frame.columns)


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
    reuse_existing_chunks: bool = False,
) -> Path:
    chunk_dir = work_dir / "chunks"
    log_dir = work_dir / "logs"
    chunk_dir.mkdir(parents=True, exist_ok=True)
    output_path = chunk_dir / f"{job.label}.csv"

    if reuse_existing_chunks and chunk_is_reusable(output_path):
        print_status(f"REUSE {job.label}: {output_path}")
        return output_path
    if output_path.exists():
        status = "STALE" if reuse_existing_chunks else "OVERWRITE"
        print_status(f"{status} {job.label}: {output_path}")
        output_path.unlink()

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
        if output_path.exists():
            output_path.unlink()
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
    return pl.read_csv(
        csv_path,
        schema_overrides={
            "lead_step": pl.Int64,
            "lead_minutes": pl.Int64,
        },
    ).select(_RESULT_COLUMNS)


def merge_chunk_results(chunk_paths: Sequence[Path]) -> pl.DataFrame:
    frame = pl.concat([load_chunk_frame(path) for path in chunk_paths], how="vertical")
    return sort_result_frame(frame.select(_RESULT_COLUMNS))


def validate_final_results(frame: pl.DataFrame) -> None:
    expected = expected_result_keys()
    if frame.height != len(expected):
        raise RuntimeError(f"Expected {len(expected)} result rows, found {frame.height}.")
    actual = list(
        zip(
            frame["dataset_id"].to_list(),
            frame["covariate_stage"].to_list(),
            frame["covariate_pack"].to_list(),
            frame["model_variant"].to_list(),
            frame["split_name"].to_list(),
            frame["eval_protocol"].to_list(),
            frame["metric_scope"].to_list(),
            frame["lead_step"].to_list(),
            strict=True,
        )
    )
    normalized_actual = [
        (
            dataset_id,
            covariate_stage,
            covariate_pack,
            model_variant,
            split_name,
            eval_protocol,
            metric_scope,
            None if lead_step is None else int(lead_step),
        )
        for (
            dataset_id,
            covariate_stage,
            covariate_pack,
            model_variant,
            split_name,
            eval_protocol,
            metric_scope,
            lead_step,
        ) in actual
    ]
    if normalized_actual != expected:
        raise RuntimeError(f"Unexpected result key rows: {normalized_actual!r}")
    if frame["model_id"].n_unique() != 1 or frame["model_id"][0] != MODEL_ID:
        raise RuntimeError("Final results contain an unexpected model_id.")
    if frame["task_id"].n_unique() != 1 or frame["task_id"][0] != TASK_ID:
        raise RuntimeError("Final results contain an unexpected task_id.")
    if frame["window_protocol"].n_unique() != 1 or frame["window_protocol"][0] != WINDOW_PROTOCOL:
        raise RuntimeError("Final results contain an unexpected window_protocol.")
    if frame["split_protocol"].n_unique() != 1 or frame["split_protocol"][0] != SPLIT_PROTOCOL:
        raise RuntimeError("Final results contain an unexpected split_protocol.")
    for column in (
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
    work_dir: str | Path,
    epochs: int | None = None,
    max_windows_per_split: int | None = None,
    reuse_existing_chunks: bool = False,
) -> pl.DataFrame:
    work_dir_path = Path(work_dir)
    attempts = resolve_attempts(device)
    job_specs = build_job_specs()
    progress = _create_progress_bar(total=len(job_specs), desc="ltsf-linear full jobs", leave=True)
    chunk_paths: list[Path] = []
    try:
        for job in job_specs:
            progress.set_postfix_str(job.label)
            chunk_paths.append(
                execute_job(
                    job=job,
                    work_dir=work_dir_path,
                    attempts=attempts,
                    epochs=epochs,
                    max_windows_per_split=max_windows_per_split,
                    reuse_existing_chunks=reuse_existing_chunks,
                )
            )
            progress.update(1)
    finally:
        progress.close()
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
        help=f"Final merged CSV path. Defaults to {FINAL_OUTPUT}.",
    )
    parser.add_argument(
        "--work-dir",
        type=Path,
        default=None,
        help="Chunk/log working directory. Defaults to experiment/families/ltsf-linear/.work/full-run-<timestamp>.",
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
    parser.add_argument(
        "--reuse-existing-chunks",
        action="store_true",
        help="Resume from compatible existing chunk CSVs under the work directory instead of rerunning them.",
    )
    parser.add_argument(
        "--run-label",
        type=str,
        default=None,
        help="Optional label suffix for the formal run record under experiment/artifacts/runs/ltsf_linear_local/.",
    )
    parser.add_argument(
        "--no-record-run",
        action="store_true",
        help="Skip writing a formal run record manifest under experiment/artifacts/runs/.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    work_dir = args.work_dir or (DEFAULT_WORK_ROOT / f"full-run-{_timestamp_label()}")
    results = run_full_experiment(
        device=args.device,
        output_path=args.output_path,
        work_dir=work_dir,
        epochs=args.epochs,
        max_windows_per_split=args.max_windows_per_split,
        reuse_existing_chunks=bool(args.reuse_existing_chunks),
    )
    if not args.no_record_run:
        record_cli_run(
            family_id=FAMILY_ID,
            repo_root=REPO_ROOT,
            invocation_kind="full_orchestrator",
            entrypoint="experiment/families/ltsf-linear/run_ltsf_linear_full.py",
            args=vars(args),
            output_path=args.output_path,
            result_row_count=results.height,
            dataset_ids=DEFAULT_DATASETS,
            feature_protocol_ids=resolve_family_feature_protocol_ids(
                FAMILY_ID,
                ("reference", *DEFAULT_COVARIATE_STAGES),
                repo_root=REPO_ROOT,
            ),
            model_variants=MODEL_VARIANTS,
            eval_protocols=(ROLLING_EVAL_PROTOCOL, NON_OVERLAP_EVAL_PROTOCOL),
            result_splits=("val", "test"),
            artifacts={"work_dir": work_dir},
            run_label=args.run_label,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
