from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from pathlib import Path
import sys
from typing import Sequence

import polars as pl

EXPERIMENT_DIR = Path(__file__).resolve().parent
EXPERIMENT_ROOT = EXPERIMENT_DIR.parents[1]
REPO_ROOT = EXPERIMENT_ROOT.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

BASELINE_DIR = EXPERIMENT_DIR.parent / "world_model_baselines_v1"
if str(BASELINE_DIR) not in sys.path:
    sys.path.insert(0, str(BASELINE_DIR))

import world_model_baselines_v1 as base  # noqa: E402
from experiment.infra.common.published_outputs import (  # noqa: E402
    default_family_output_path,
    generate_run_stem,
)
from experiment.infra.common.run_records import record_cli_run  # noqa: E402
from experiment.infra.common.window_protocols import (  # noqa: E402
    NON_OVERLAP_EVAL_PROTOCOL,
    ROLLING_EVAL_PROTOCOL,
)


MODEL_ID = "WORLD_MODEL_HARDENED_BASELINE"
FAMILY_ID = "world_model_hardened_baselines_v1"
PERSISTENCE_HARDENED_VARIANT = "world_model_last_value_persistence_hardened_v1_farm_sync"
DGCRN_OFFICIAL_CORE_VARIANT = "world_model_dgcrn_official_core_v1_farm_sync"
TIMEXER_OFFICIAL_VARIANT = "world_model_timexer_official_v1_farm_sync"
ITRANSFORMER_OFFICIAL_VARIANT = "world_model_itransformer_official_v1_farm_sync"
CHRONOS_OFFICIAL_VARIANT = "world_model_chronos_2_zero_shot_official_v1_farm_sync"
DEFAULT_VARIANTS = (
    PERSISTENCE_HARDENED_VARIANT,
    DGCRN_OFFICIAL_CORE_VARIANT,
    TIMEXER_OFFICIAL_VARIANT,
    ITRANSFORMER_OFFICIAL_VARIANT,
    CHRONOS_OFFICIAL_VARIANT,
)
RESULT_PROVENANCE_COLUMNS = (
    "implementation_track",
    "source_repo",
    "source_commit",
    "adapter_kind",
    "search_config_id",
)
DEFAULT_DATASETS = base.DEFAULT_DATASETS
DEFAULT_SELECTION_METRICS = base.DEFAULT_SELECTION_METRICS
SELECTION_METRIC_RMSE = base.SELECTION_METRIC_RMSE
SELECTION_METRIC_MAE = base.SELECTION_METRIC_MAE
FEATURE_PROTOCOL_ID = base.FEATURE_PROTOCOL_ID
TASK_ID = base.TASK_ID
HISTORY_STEPS = base.HISTORY_STEPS
FORECAST_STEPS = base.FORECAST_STEPS

_REPO_ROOT = REPO_ROOT
_CACHE_ROOT = _REPO_ROOT / "cache"
_RUN_WORK_ROOT = EXPERIMENT_DIR / ".work" / "run_world_model_hardened_baselines_v1"


@dataclass(frozen=True)
class HardenedVariantSpec:
    model_variant: str
    backend_variant: str
    feature_protocol_id: str
    baseline_type: str
    implementation_track: str
    source_repo: str
    source_commit: str
    adapter_kind: str
    search_config_id: str


VARIANT_SPECS = (
    HardenedVariantSpec(
        model_variant=PERSISTENCE_HARDENED_VARIANT,
        backend_variant=base.PERSISTENCE_VARIANT,
        feature_protocol_id=FEATURE_PROTOCOL_ID,
        baseline_type="last_value_persistence_hardened",
        implementation_track="hardened_official",
        source_repo="analytic://last-value-persistence",
        source_commit="not-applicable",
        adapter_kind="analytic_anchor",
        search_config_id="analytic_default",
    ),
    HardenedVariantSpec(
        model_variant=DGCRN_OFFICIAL_CORE_VARIANT,
        backend_variant=base.DGCRN_VARIANT,
        feature_protocol_id=FEATURE_PROTOCOL_ID,
        baseline_type="dgcrn_official_core",
        implementation_track="hardened_official",
        source_repo="https://github.com/tsinghua-fib-lab/Traffic-Benchmark.git",
        source_commit="b9f8e40b4df9b58f5ad88432dc070cbbbcdc0228",
        adapter_kind="official_core_port",
        search_config_id="phase1_default",
    ),
    HardenedVariantSpec(
        model_variant=TIMEXER_OFFICIAL_VARIANT,
        backend_variant=base.TIMEXER_VARIANT,
        feature_protocol_id=FEATURE_PROTOCOL_ID,
        baseline_type="timexer_official",
        implementation_track="hardened_official",
        source_repo="https://github.com/thuml/TimeXer.git",
        source_commit="76011909357972bd55a27adba2e1be994d81b327",
        adapter_kind="official_source_adapter",
        search_config_id="phase1_default",
    ),
    HardenedVariantSpec(
        model_variant=ITRANSFORMER_OFFICIAL_VARIANT,
        backend_variant=base.ITRANSFORMER_VARIANT,
        feature_protocol_id=FEATURE_PROTOCOL_ID,
        baseline_type="itransformer_official",
        implementation_track="hardened_official",
        source_repo="https://github.com/thuml/iTransformer.git",
        source_commit="c2426e68ca13f74aaec08045c5c724d8ad328124",
        adapter_kind="official_source_adapter",
        search_config_id="phase1_default",
    ),
    HardenedVariantSpec(
        model_variant=CHRONOS_OFFICIAL_VARIANT,
        backend_variant=base.CHRONOS_VARIANT,
        feature_protocol_id=FEATURE_PROTOCOL_ID,
        baseline_type="chronos_2_zero_shot_official",
        implementation_track="hardened_official",
        source_repo="https://huggingface.co/amazon/chronos-2",
        source_commit="chronos-forecasting==2.2.2",
        adapter_kind="official_package_adapter",
        search_config_id="zero_shot_default",
    ),
)
_VARIANT_SPECS_BY_NAME = {spec.model_variant: spec for spec in VARIANT_SPECS}
_VARIANT_SPECS_BY_BACKEND = {spec.backend_variant: spec for spec in VARIANT_SPECS}


def resolve_variant_specs(variant_names: Sequence[str] | None = None) -> tuple[HardenedVariantSpec, ...]:
    requested = tuple(variant_names or DEFAULT_VARIANTS)
    resolved: list[HardenedVariantSpec] = []
    seen: set[str] = set()
    for variant_name in requested:
        try:
            spec = _VARIANT_SPECS_BY_NAME[variant_name]
        except KeyError as exc:
            raise ValueError(f"Unknown hardened model variant {variant_name!r}.") from exc
        if spec.model_variant in seen:
            continue
        resolved.append(spec)
        seen.add(spec.model_variant)
    return tuple(resolved)


def resolve_selection_metrics(selection_metrics: Sequence[str] | None) -> tuple[str, ...]:
    return base.resolve_selection_metrics(selection_metrics)


def _resolve_output_path(
    output_path: str | Path | None,
    *,
    run_stem: str | None = None,
    resume: bool = False,
    force_rerun: bool = False,
) -> Path:
    if output_path is not None:
        return Path(output_path)
    if resume or force_rerun:
        raise ValueError("--resume/--force-rerun requires --output-path for timestamped formal outputs.")
    return default_family_output_path(repo_root=_REPO_ROOT, family_id=FAMILY_ID, run_stem=run_stem)


def training_history_output_path(output_path: str | Path) -> Path:
    return base.training_history_output_path(output_path)


def _backend_variants(specs: Sequence[HardenedVariantSpec]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(spec.backend_variant for spec in specs))


def annotate_hardened_results(frame: pl.DataFrame, variant_names: Sequence[str] | None = None) -> pl.DataFrame:
    specs = resolve_variant_specs(variant_names)
    specs_by_backend = {spec.backend_variant: spec for spec in specs}
    if "model_variant" not in frame.columns:
        raise ValueError("Hardened result annotation requires a model_variant column.")
    observed = set(frame.get_column("model_variant").unique().to_list())
    unknown = sorted(observed - set(specs_by_backend))
    if unknown:
        raise ValueError(f"Cannot annotate backend variants not requested by hardened family: {unknown!r}.")

    def _field(field_name: str):
        return pl.col("model_variant").map_elements(
            lambda backend: getattr(specs_by_backend[str(backend)], field_name),
            return_dtype=pl.Utf8,
        )

    expressions = [
        _field("model_variant").alias("model_variant"),
        _field("implementation_track").alias("implementation_track"),
        _field("source_repo").alias("source_repo"),
        _field("source_commit").alias("source_commit"),
        _field("adapter_kind").alias("adapter_kind"),
        _field("search_config_id").alias("search_config_id"),
    ]
    if "model_id" in frame.columns:
        expressions.append(pl.lit(MODEL_ID).alias("model_id"))
    if "baseline_type" in frame.columns:
        expressions.append(_field("baseline_type").alias("baseline_type"))
    return frame.with_columns(expressions)


def _annotate_training_history_file(path: Path, variant_names: Sequence[str]) -> None:
    if not path.exists():
        return
    history = pl.read_csv(path)
    if "model_variant" not in history.columns:
        return
    annotated = annotate_hardened_results(history, variant_names)
    annotated.write_csv(path)


def _resolved_record_hyperparameters(
    dataset_ids: Sequence[str],
    variant_specs: Sequence[HardenedVariantSpec],
    args: argparse.Namespace,
) -> dict[str, object]:
    return {
        dataset_id: {
            spec.model_variant: {
                "feature_protocol_id": spec.feature_protocol_id,
                "baseline_type": spec.baseline_type,
                "backend_variant": spec.backend_variant,
                "implementation_track": spec.implementation_track,
                "source_repo": spec.source_repo,
                "source_commit": spec.source_commit,
                "adapter_kind": spec.adapter_kind,
                "search_config_id": spec.search_config_id,
                **asdict(
                    base.resolve_hyperparameter_profile(
                        spec.backend_variant,
                        dataset_id=dataset_id,
                        batch_size=args.batch_size,
                        learning_rate=args.learning_rate,
                        max_epochs=args.epochs,
                        early_stopping_patience=args.patience,
                        d_model=args.d_model,
                        lstm_hidden_dim=args.lstm_hidden_dim,
                        attention_heads=args.attention_heads,
                        patch_len=args.patch_len,
                        encoder_layers=args.encoder_layers,
                        ff_hidden_dim=args.ff_hidden_dim,
                        residual_channels=args.residual_channels,
                        skip_channels=args.skip_channels,
                        end_channels=args.end_channels,
                        gcn_depth=args.gcn_depth,
                        mtgnn_layers=args.mtgnn_layers,
                        subgraph_size=args.subgraph_size,
                        node_embed_dim=args.node_embed_dim,
                        dilation_exponential=args.dilation_exponential,
                        propalpha=args.propalpha,
                        hidden_dim=args.hidden_dim,
                        embed_dim=args.embed_dim,
                        num_layers=args.num_layers,
                        cheb_k=args.cheb_k,
                        teacher_forcing_ratio=args.teacher_forcing_ratio,
                        dropout=args.dropout,
                        grad_clip_norm=args.grad_clip_norm,
                        weight_decay=args.weight_decay,
                        bounded_output_epsilon=args.bounded_output_epsilon,
                    )
                ),
            }
            for spec in variant_specs
        }
        for dataset_id in dataset_ids
    }


def run_experiment(
    *,
    dataset_ids: Sequence[str] = DEFAULT_DATASETS,
    variant_names: Sequence[str] | None = None,
    selection_metrics: Sequence[str] | None = None,
    cache_root: str | Path = _CACHE_ROOT,
    output_path: str | Path | None = None,
    device: str | None = None,
    max_epochs: int | None = None,
    max_train_origins: int | None = None,
    max_eval_origins: int | None = None,
    seed: int = base.DEFAULT_SEED,
    batch_size: int | None = None,
    eval_batch_size: int | None = None,
    learning_rate: float | None = None,
    early_stopping_patience: int | None = None,
    d_model: int | None = None,
    lstm_hidden_dim: int | None = None,
    attention_heads: int | None = None,
    patch_len: int | None = None,
    encoder_layers: int | None = None,
    ff_hidden_dim: int | None = None,
    residual_channels: int | None = None,
    skip_channels: int | None = None,
    end_channels: int | None = None,
    gcn_depth: int | None = None,
    mtgnn_layers: int | None = None,
    subgraph_size: int | None = None,
    node_embed_dim: int | None = None,
    dilation_exponential: int | None = None,
    propalpha: float | None = None,
    hidden_dim: int | None = None,
    embed_dim: int | None = None,
    num_layers: int | None = None,
    cheb_k: int | None = None,
    teacher_forcing_ratio: float | None = None,
    dropout: float | None = None,
    grad_clip_norm: float | None = None,
    weight_decay: float | None = None,
    bounded_output_epsilon: float | None = None,
    num_workers: int | None = None,
    tensorboard_log_dir: str | Path | None = None,
    disable_tensorboard: bool = False,
    resume: bool = False,
    force_rerun: bool = False,
    work_root: str | Path = _RUN_WORK_ROOT,
) -> pl.DataFrame:
    specs = resolve_variant_specs(variant_names)
    resolved_output_path = _resolve_output_path(output_path, resume=resume, force_rerun=force_rerun)
    backend_results = base.run_experiment(
        dataset_ids=dataset_ids,
        variant_names=_backend_variants(specs),
        selection_metrics=selection_metrics,
        cache_root=cache_root,
        output_path=resolved_output_path,
        device=device,
        max_epochs=max_epochs,
        max_train_origins=max_train_origins,
        max_eval_origins=max_eval_origins,
        seed=seed,
        batch_size=batch_size,
        eval_batch_size=eval_batch_size,
        learning_rate=learning_rate,
        early_stopping_patience=early_stopping_patience,
        d_model=d_model,
        lstm_hidden_dim=lstm_hidden_dim,
        attention_heads=attention_heads,
        patch_len=patch_len,
        encoder_layers=encoder_layers,
        ff_hidden_dim=ff_hidden_dim,
        residual_channels=residual_channels,
        skip_channels=skip_channels,
        end_channels=end_channels,
        gcn_depth=gcn_depth,
        mtgnn_layers=mtgnn_layers,
        subgraph_size=subgraph_size,
        node_embed_dim=node_embed_dim,
        dilation_exponential=dilation_exponential,
        propalpha=propalpha,
        hidden_dim=hidden_dim,
        embed_dim=embed_dim,
        num_layers=num_layers,
        cheb_k=cheb_k,
        teacher_forcing_ratio=teacher_forcing_ratio,
        dropout=dropout,
        grad_clip_norm=grad_clip_norm,
        weight_decay=weight_decay,
        bounded_output_epsilon=bounded_output_epsilon,
        num_workers=num_workers,
        tensorboard_log_dir=tensorboard_log_dir,
        disable_tensorboard=disable_tensorboard,
        resume=resume,
        force_rerun=force_rerun,
        work_root=work_root,
    )
    annotated = annotate_hardened_results(backend_results, tuple(spec.model_variant for spec in specs))
    resolved_output_path.parent.mkdir(parents=True, exist_ok=True)
    annotated.write_csv(resolved_output_path)
    _annotate_training_history_file(
        training_history_output_path(resolved_output_path),
        tuple(spec.model_variant for spec in specs),
    )
    return annotated


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", action="append", choices=list(DEFAULT_DATASETS), dest="datasets")
    parser.add_argument("--variant", action="append", choices=list(DEFAULT_VARIANTS), dest="variants")
    parser.add_argument(
        "--selection-metric",
        action="append",
        choices=list(DEFAULT_SELECTION_METRICS),
        dest="selection_metrics",
        help="Selection metric contract to run. Repeat to run multiple contracts; defaults to both val_rmse_pu then val_mae_pu.",
    )
    parser.add_argument("--device", choices=("auto", "cuda", "mps", "cpu"), default="auto")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--patience", type=int, default=None)
    parser.add_argument("--seed", type=int, default=base.DEFAULT_SEED)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--eval-batch-size", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--max-train-origins", type=int, default=None)
    parser.add_argument("--max-eval-origins", type=int, default=None)
    parser.add_argument("--output-path", type=Path, default=None)
    parser.add_argument("--d-model", type=int, default=None)
    parser.add_argument("--lstm-hidden-dim", type=int, default=None)
    parser.add_argument("--attention-heads", type=int, default=None)
    parser.add_argument("--patch-len", type=int, default=None)
    parser.add_argument("--encoder-layers", type=int, default=None)
    parser.add_argument("--ff-hidden-dim", type=int, default=None)
    parser.add_argument("--residual-channels", type=int, default=None)
    parser.add_argument("--skip-channels", type=int, default=None)
    parser.add_argument("--end-channels", type=int, default=None)
    parser.add_argument("--gcn-depth", type=int, default=None)
    parser.add_argument("--mtgnn-layers", type=int, default=None)
    parser.add_argument("--subgraph-size", type=int, default=None)
    parser.add_argument("--node-embed-dim", type=int, default=None)
    parser.add_argument("--dilation-exponential", type=int, default=None)
    parser.add_argument("--propalpha", type=float, default=None)
    parser.add_argument("--hidden-dim", type=int, default=None)
    parser.add_argument("--embed-dim", type=int, default=None)
    parser.add_argument("--num-layers", type=int, default=None)
    parser.add_argument("--cheb-k", type=int, default=None)
    parser.add_argument("--teacher-forcing-ratio", type=float, default=None)
    parser.add_argument("--dropout", type=float, default=None)
    parser.add_argument("--grad-clip-norm", type=float, default=None)
    parser.add_argument("--weight-decay", type=float, default=None)
    parser.add_argument("--bounded-output-epsilon", type=float, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--tensorboard-log-dir", type=Path, default=None)
    parser.add_argument("--disable-tensorboard", action="store_true")
    parser.add_argument("--run-label", type=str, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--force-rerun", action="store_true")
    parser.add_argument("--no-record-run", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    run_stem = generate_run_stem()
    try:
        resolved_output_path = _resolve_output_path(
            args.output_path,
            run_stem=run_stem,
            resume=args.resume,
            force_rerun=args.force_rerun,
        )
    except ValueError as exc:
        parser.error(str(exc))
    resolved_dataset_ids = base._validate_dataset_ids(tuple(args.datasets) if args.datasets else DEFAULT_DATASETS)
    variant_specs = resolve_variant_specs(tuple(args.variants) if args.variants else None)
    resolved_selection_metrics = resolve_selection_metrics(tuple(args.selection_metrics) if args.selection_metrics else None)
    resolved_tensorboard_root = base.resolve_tensorboard_root(
        output_path=resolved_output_path,
        work_root=_RUN_WORK_ROOT,
        tensorboard_log_dir=args.tensorboard_log_dir,
        disable_tensorboard=args.disable_tensorboard,
    )
    resolved_hyperparameters = _resolved_record_hyperparameters(
        resolved_dataset_ids,
        variant_specs,
        args,
    )
    print(f"[{FAMILY_ID}] output_path={resolved_output_path}")
    results = run_experiment(
        dataset_ids=resolved_dataset_ids,
        variant_names=tuple(spec.model_variant for spec in variant_specs),
        selection_metrics=resolved_selection_metrics,
        device=args.device,
        max_epochs=args.epochs,
        output_path=resolved_output_path,
        max_train_origins=args.max_train_origins,
        max_eval_origins=args.max_eval_origins,
        seed=args.seed,
        batch_size=args.batch_size,
        eval_batch_size=args.eval_batch_size,
        learning_rate=args.learning_rate,
        early_stopping_patience=args.patience,
        d_model=args.d_model,
        lstm_hidden_dim=args.lstm_hidden_dim,
        attention_heads=args.attention_heads,
        patch_len=args.patch_len,
        encoder_layers=args.encoder_layers,
        ff_hidden_dim=args.ff_hidden_dim,
        residual_channels=args.residual_channels,
        skip_channels=args.skip_channels,
        end_channels=args.end_channels,
        gcn_depth=args.gcn_depth,
        mtgnn_layers=args.mtgnn_layers,
        subgraph_size=args.subgraph_size,
        node_embed_dim=args.node_embed_dim,
        dilation_exponential=args.dilation_exponential,
        propalpha=args.propalpha,
        hidden_dim=args.hidden_dim,
        embed_dim=args.embed_dim,
        num_layers=args.num_layers,
        cheb_k=args.cheb_k,
        teacher_forcing_ratio=args.teacher_forcing_ratio,
        dropout=args.dropout,
        grad_clip_norm=args.grad_clip_norm,
        weight_decay=args.weight_decay,
        bounded_output_epsilon=args.bounded_output_epsilon,
        num_workers=args.num_workers,
        tensorboard_log_dir=args.tensorboard_log_dir,
        disable_tensorboard=args.disable_tensorboard,
        resume=args.resume,
        force_rerun=args.force_rerun,
    )
    if not args.no_record_run:
        recorded_args = vars(args).copy()
        recorded_args["output_path"] = resolved_output_path
        recorded_args["selection_metrics"] = list(resolved_selection_metrics)
        recorded_args["resolved_dataset_variant_hyperparameters"] = resolved_hyperparameters
        record_cli_run(
            family_id=FAMILY_ID,
            repo_root=_REPO_ROOT,
            invocation_kind="family_runner",
            entrypoint=f"experiment/families/{FAMILY_ID}/run_world_model_hardened_baselines_v1.py",
            args=recorded_args,
            output_path=resolved_output_path,
            result_row_count=results.height,
            dataset_ids=resolved_dataset_ids,
            feature_protocol_ids=tuple(spec.feature_protocol_id for spec in variant_specs),
            model_variants=tuple(spec.model_variant for spec in variant_specs),
            eval_protocols=(ROLLING_EVAL_PROTOCOL, NON_OVERLAP_EVAL_PROTOCOL),
            result_splits=("val", "test"),
            artifacts={
                "training_history": training_history_output_path(resolved_output_path),
                **({} if resolved_tensorboard_root is None else {"tensorboard_root": resolved_tensorboard_root}),
            },
            run_label=args.run_label,
            run_stem=run_stem if args.output_path is None else None,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
