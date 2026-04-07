from __future__ import annotations

import argparse
import os
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

from .api import build_gold_base, build_manifest, build_silver, build_task_cache
from .models import TaskSpec
from .registry import list_dataset_specs

SUPPORTED_DATASETS = tuple(spec.dataset_id for spec in list_dataset_specs())

_FARM_TASK = TaskSpec.next_6h_from_24h(granularity="farm")
_TURBINE_TASK = TaskSpec.next_6h_from_24h(granularity="turbine")
_TURBINE_STRIDE_TASK = TaskSpec(
    task_id="next_6h_from_24h_stride_6h",
    history_duration="24h",
    forecast_duration="6h",
    stride_duration="6h",
    granularity="turbine",
)


@dataclass(frozen=True)
class RebuildStage:
    name: str
    run: Callable[[str, Path], Path]


@dataclass(frozen=True)
class RebuildFailure:
    dataset: str
    stage: str
    error: str


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m wind_datasets.rebuild_cache",
        description="Rebuild standard cache layers for one or more datasets.",
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Remove cache/<dataset> before rebuilding the selected datasets.",
    )
    parser.add_argument(
        "--cache-root",
        default=os.environ.get("CACHE_ROOT", "cache"),
        help="Cache root directory. Defaults to ./cache relative to the current working directory.",
    )
    parser.add_argument(
        "--include-turbine",
        action="store_true",
        help="Also build explicit turbine compatibility caches.",
    )
    parser.add_argument(
        "datasets",
        nargs="*",
        metavar="dataset",
        default=["all"],
        help=f"Datasets to rebuild: {', '.join(SUPPORTED_DATASETS)}, or all.",
    )
    return parser


def normalize_datasets(requested: Sequence[str]) -> list[str]:
    datasets = list(requested) or ["all"]
    normalized: list[str] = []
    supported = set(SUPPORTED_DATASETS)
    for dataset in datasets:
        expanded = SUPPORTED_DATASETS if dataset == "all" else (dataset,)
        for item in expanded:
            if item not in supported:
                supported_names = ", ".join([*SUPPORTED_DATASETS, "all"])
                raise ValueError(f"Unsupported dataset {item!r}. Expected one of: {supported_names}.")
            if item not in normalized:
                normalized.append(item)
    return normalized


def rebuild_stages(include_turbine: bool) -> tuple[RebuildStage, ...]:
    stages = [
        RebuildStage(
            name="manifest",
            run=lambda dataset, cache_root: build_manifest(dataset, cache_root=cache_root),
        ),
        RebuildStage(
            name="silver",
            run=lambda dataset, cache_root: build_silver(dataset, cache_root=cache_root),
        ),
        RebuildStage(
            name="gold_base/default/farm/default",
            run=lambda dataset, cache_root: build_gold_base(dataset, cache_root=cache_root),
        ),
        RebuildStage(
            name="tasks/default/farm/next_6h_from_24h",
            run=lambda dataset, cache_root: build_task_cache(dataset, _FARM_TASK, cache_root=cache_root),
        ),
    ]
    if include_turbine:
        stages.extend(
            [
                RebuildStage(
                    name="gold_base/default/turbine/default",
                    run=lambda dataset, cache_root: build_gold_base(
                        dataset,
                        cache_root=cache_root,
                        layout="turbine",
                    ),
                ),
                RebuildStage(
                    name="tasks/default/turbine/next_6h_from_24h",
                    run=lambda dataset, cache_root: build_task_cache(
                        dataset,
                        _TURBINE_TASK,
                        cache_root=cache_root,
                    ),
                ),
                RebuildStage(
                    name="tasks/default/turbine/next_6h_from_24h_stride_6h",
                    run=lambda dataset, cache_root: build_task_cache(
                        dataset,
                        _TURBINE_STRIDE_TASK,
                        cache_root=cache_root,
                    ),
                ),
            ]
        )
    return tuple(stages)


def _clean_cache_dirs(cache_root: Path, datasets: Sequence[str]) -> None:
    for dataset in datasets:
        dataset_dir = cache_root / dataset
        print(f"[clean] removing {dataset_dir}", flush=True)
        shutil.rmtree(dataset_dir, ignore_errors=True)


def _log_stage(dataset: str, stage: str) -> None:
    print(f"[rebuild] dataset={dataset} stage={stage}", flush=True)


def _format_error(exc: Exception) -> str:
    message = str(exc).strip()
    return message or exc.__class__.__name__


def run_rebuild(
    datasets: Sequence[str],
    cache_root: Path,
    *,
    clean: bool = False,
    include_turbine: bool = False,
) -> list[RebuildFailure]:
    if clean:
        _clean_cache_dirs(cache_root, datasets)
    cache_root.mkdir(parents=True, exist_ok=True)

    failures: list[RebuildFailure] = []
    for dataset in datasets:
        for stage in rebuild_stages(include_turbine):
            _log_stage(dataset, stage.name)
            try:
                result = stage.run(dataset, cache_root)
            except Exception as exc:
                failure = RebuildFailure(dataset=dataset, stage=stage.name, error=_format_error(exc))
                failures.append(failure)
                print(
                    f"[rebuild] dataset={failure.dataset} stage={failure.stage} failed: {failure.error}",
                    file=sys.stderr,
                    flush=True,
                )
                break
            print(result, flush=True)
    return failures


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    try:
        args = parser.parse_args(list(argv) if argv is not None else None)
    except SystemExit as exc:
        return int(exc.code)

    try:
        datasets = normalize_datasets(args.datasets)
    except ValueError as exc:
        parser.print_usage(sys.stderr)
        print(f"{parser.prog}: error: {exc}", file=sys.stderr)
        return 2

    failures = run_rebuild(
        datasets,
        Path(args.cache_root),
        clean=args.clean,
        include_turbine=args.include_turbine,
    )
    if failures:
        print(
            f"[rebuild] completed with {len(failures)} failed dataset(s).",
            file=sys.stderr,
            flush=True,
        )
        for failure in failures:
            print(
                f"[rebuild] summary dataset={failure.dataset} stage={failure.stage}: {failure.error}",
                file=sys.stderr,
                flush=True,
            )
        return 1
    print("[rebuild] completed successfully.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
