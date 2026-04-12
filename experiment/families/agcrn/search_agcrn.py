from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import time
from typing import Sequence

import polars as pl

import agcrn


DEFAULT_OUTPUT_DIR = (
    agcrn.EXPERIMENT_ROOT
    / "artifacts"
    / "scratch"
    / agcrn.FAMILY_ID
    / "search_20260413"
)
ALIGNMENT_VARIANTS = agcrn.resolve_variant_specs(agcrn.SEARCH_VARIANTS)


@dataclass(frozen=True)
class SearchConfig:
    name: str
    batch_size: int
    learning_rate: float
    hidden_dim: int
    embed_dim: int
    num_layers: int
    cheb_k: int


COMMON_SCREEN_CONFIGS = (
    SearchConfig(
        name="baseline_bs1024_h64_e10_l2_k2_lr1e-3",
        batch_size=1024,
        learning_rate=1e-3,
        hidden_dim=64,
        embed_dim=10,
        num_layers=2,
        cheb_k=2,
    ),
    SearchConfig(
        name="baseline_bs512_h64_e10_l2_k2_lr1e-3",
        batch_size=512,
        learning_rate=1e-3,
        hidden_dim=64,
        embed_dim=10,
        num_layers=2,
        cheb_k=2,
    ),
    SearchConfig(
        name="baseline_bs512_h64_e10_l2_k2_lr5e-4",
        batch_size=512,
        learning_rate=5e-4,
        hidden_dim=64,
        embed_dim=10,
        num_layers=2,
        cheb_k=2,
    ),
    SearchConfig(
        name="compact_bs512_h48_e8_l1_k2_lr1e-3",
        batch_size=512,
        learning_rate=1e-3,
        hidden_dim=48,
        embed_dim=8,
        num_layers=1,
        cheb_k=2,
    ),
    SearchConfig(
        name="compact_bs1024_h48_e8_l1_k2_lr2e-3",
        batch_size=1024,
        learning_rate=2e-3,
        hidden_dim=48,
        embed_dim=8,
        num_layers=1,
        cheb_k=2,
    ),
    SearchConfig(
        name="larger_bs512_h96_e16_l2_k2_lr1e-3",
        batch_size=512,
        learning_rate=1e-3,
        hidden_dim=96,
        embed_dim=16,
        num_layers=2,
        cheb_k=2,
    ),
    SearchConfig(
        name="larger_bs512_h96_e16_l2_k2_lr5e-4",
        batch_size=512,
        learning_rate=5e-4,
        hidden_dim=96,
        embed_dim=16,
        num_layers=2,
        cheb_k=2,
    ),
    SearchConfig(
        name="graph_bs512_h64_e16_l2_k3_lr5e-4",
        batch_size=512,
        learning_rate=5e-4,
        hidden_dim=64,
        embed_dim=16,
        num_layers=2,
        cheb_k=3,
    ),
)

SCREEN_CONFIGS_BY_VARIANT = {
    variant_name: COMMON_SCREEN_CONFIGS
    for variant_name in agcrn.SEARCH_VARIANTS
}


def _maybe_clear_cuda_cache() -> None:
    if agcrn.torch is None:
        return
    if agcrn.torch.cuda.is_available():
        agcrn.torch.cuda.empty_cache()


def _config_from_row(row: dict[str, object]) -> SearchConfig:
    return SearchConfig(
        name=str(row["config_name"]),
        batch_size=int(row["batch_size"]),
        learning_rate=float(row["learning_rate"]),
        hidden_dim=int(row["hidden_dim"]),
        embed_dim=int(row["embed_dim"]),
        num_layers=int(row["num_layers"]),
        cheb_k=int(row["cheb_k"]),
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run aligned AGCRN hyperparameter search across the active feature protocols.",
    )
    parser.add_argument(
        "--dataset",
        action="append",
        choices=list(agcrn.DEFAULT_DATASETS),
        dest="datasets",
        help="Datasets to search. Defaults to kelmarsh.",
    )
    parser.add_argument(
        "--variant",
        action="append",
        choices=list(agcrn.SEARCH_VARIANTS),
        dest="variants",
        help="Limit the tuned variants. Alignment remains fixed to the full active seven-variant family surface.",
    )
    parser.add_argument(
        "--config-name",
        action="append",
        dest="config_names",
        help="Optional config-name filter applied after variant selection. Can be repeated.",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cuda", "mps", "cpu"),
        default="auto",
        help="Training device. Defaults to auto.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=agcrn.DEFAULT_SEED,
        help="Random seed for all runs.",
    )
    parser.add_argument(
        "--screen-train-origins",
        type=int,
        default=65_536,
        help="Dense train-origin cap used during the screening stage.",
    )
    parser.add_argument(
        "--screen-eval-origins",
        type=int,
        default=8_192,
        help="Dense val-origin cap used during the screening stage.",
    )
    parser.add_argument(
        "--screen-epochs",
        type=int,
        default=10,
        help="Maximum epochs used during screening.",
    )
    parser.add_argument(
        "--screen-patience",
        type=int,
        default=3,
        help="Early-stopping patience used during screening.",
    )
    parser.add_argument(
        "--full-epochs",
        type=int,
        default=20,
        help="Maximum epochs used for final confirmatory runs.",
    )
    parser.add_argument(
        "--full-patience",
        type=int,
        default=5,
        help="Early-stopping patience used for final confirmatory runs.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=2,
        help="How many screened configs per variant to carry into the full stage.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directory for summary CSV/JSON outputs. Defaults to {DEFAULT_OUTPUT_DIR}.",
    )
    parser.add_argument(
        "--skip-final",
        action="store_true",
        help="Only run screening and skip the final full-window confirmation stage.",
    )
    parser.add_argument(
        "--full-only",
        action="store_true",
        help="Skip screening and run only the final full-window stage for the selected configs.",
    )
    return parser


def _prepared_by_variant(
    dataset_id: str,
    *,
    max_train_origins: int | None,
    max_eval_origins: int | None,
) -> dict[str, agcrn.PreparedDataset]:
    prepared = agcrn._prepare_datasets_for_variants(
        dataset_id,
        variant_specs=ALIGNMENT_VARIANTS,
        max_train_origins=max_train_origins,
        max_eval_origins=max_eval_origins,
    )
    return {item.model_variant: item for item in prepared}


def _screen_one(
    prepared_dataset: agcrn.PreparedDataset,
    *,
    config: SearchConfig,
    device: str,
    seed: int,
    max_epochs: int,
    patience: int,
) -> dict[str, object]:
    runtime_start = time.monotonic()
    training_outcome = agcrn.train_model(
        prepared_dataset,
        device=device,
        seed=seed,
        batch_size=config.batch_size,
        learning_rate=config.learning_rate,
        max_epochs=max_epochs,
        early_stopping_patience=patience,
        hidden_dim=config.hidden_dim,
        embed_dim=config.embed_dim,
        num_layers=config.num_layers,
        cheb_k=config.cheb_k,
        grad_clip_norm=agcrn.DEFAULT_GRAD_CLIP_NORM,
        progress_label=f"screen/{prepared_dataset.model_variant}/{config.name}",
    )
    val_rolling_loader = agcrn._build_dataloader(
        prepared_dataset,
        windows=prepared_dataset.val_rolling_windows,
        batch_size=config.batch_size,
        shuffle=False,
        seed=seed,
    )
    val_non_overlap_loader = agcrn._build_dataloader(
        prepared_dataset,
        windows=prepared_dataset.val_non_overlap_windows,
        batch_size=config.batch_size,
        shuffle=False,
        seed=seed,
    )
    val_rolling = agcrn.evaluate_model(
        training_outcome.model,
        val_rolling_loader,
        device=training_outcome.device,
        rated_power_kw=prepared_dataset.rated_power_kw,
        forecast_steps=prepared_dataset.forecast_steps,
        progress_label=f"screen/{prepared_dataset.model_variant}/{config.name}/val_rolling",
    )
    val_non_overlap = agcrn.evaluate_model(
        training_outcome.model,
        val_non_overlap_loader,
        device=training_outcome.device,
        rated_power_kw=prepared_dataset.rated_power_kw,
        forecast_steps=prepared_dataset.forecast_steps,
        progress_label=f"screen/{prepared_dataset.model_variant}/{config.name}/val_non_overlap",
    )
    row = {
        "dataset_id": prepared_dataset.dataset_id,
        "model_variant": prepared_dataset.model_variant,
        "feature_protocol_id": prepared_dataset.feature_protocol_id,
        "stage": "screen",
        "config_name": config.name,
        "batch_size": config.batch_size,
        "learning_rate": config.learning_rate,
        "hidden_dim": config.hidden_dim,
        "embed_dim": config.embed_dim,
        "num_layers": config.num_layers,
        "cheb_k": config.cheb_k,
        "best_epoch": training_outcome.best_epoch,
        "epochs_ran": training_outcome.epochs_ran,
        "best_val_rmse_pu": training_outcome.best_val_rmse_pu,
        "val_rolling_window_count": len(prepared_dataset.val_rolling_windows),
        "val_non_overlap_window_count": len(prepared_dataset.val_non_overlap_windows),
        "val_rolling_rmse_pu": float(val_rolling.rmse_pu),
        "val_rolling_mae_pu": float(val_rolling.mae_pu),
        "val_non_overlap_rmse_pu": float(val_non_overlap.rmse_pu),
        "val_non_overlap_mae_pu": float(val_non_overlap.mae_pu),
        "train_window_count": len(prepared_dataset.train_windows),
        "runtime_seconds": round(time.monotonic() - runtime_start, 6),
        "device": training_outcome.device,
        "seed": seed,
    }
    del training_outcome
    del val_rolling_loader
    del val_non_overlap_loader
    _maybe_clear_cuda_cache()
    return row


def _overall_metric_row(frame: pl.DataFrame, *, split_name: str, eval_protocol: str) -> dict[str, object]:
    return (
        frame.filter(
            (pl.col("split_name") == split_name)
            & (pl.col("eval_protocol") == eval_protocol)
            & (pl.col("metric_scope") == agcrn.OVERALL_METRIC_SCOPE)
        )
        .to_dicts()[0]
    )


def _final_one(
    prepared_dataset: agcrn.PreparedDataset,
    *,
    config: SearchConfig,
    device: str,
    seed: int,
    max_epochs: int,
    patience: int,
) -> tuple[dict[str, object], pl.DataFrame]:
    rows = agcrn.execute_training_job(
        prepared_dataset,
        device=device,
        seed=seed,
        batch_size=config.batch_size,
        learning_rate=config.learning_rate,
        max_epochs=max_epochs,
        early_stopping_patience=patience,
        hidden_dim=config.hidden_dim,
        embed_dim=config.embed_dim,
        num_layers=config.num_layers,
        cheb_k=config.cheb_k,
        grad_clip_norm=agcrn.DEFAULT_GRAD_CLIP_NORM,
    )
    frame = pl.DataFrame(rows)
    val_rolling = _overall_metric_row(frame, split_name="val", eval_protocol=agcrn.ROLLING_EVAL_PROTOCOL)
    val_non_overlap = _overall_metric_row(frame, split_name="val", eval_protocol=agcrn.NON_OVERLAP_EVAL_PROTOCOL)
    test_rolling = _overall_metric_row(frame, split_name="test", eval_protocol=agcrn.ROLLING_EVAL_PROTOCOL)
    test_non_overlap = _overall_metric_row(frame, split_name="test", eval_protocol=agcrn.NON_OVERLAP_EVAL_PROTOCOL)
    summary = {
        "dataset_id": prepared_dataset.dataset_id,
        "model_variant": prepared_dataset.model_variant,
        "feature_protocol_id": prepared_dataset.feature_protocol_id,
        "stage": "final",
        "config_name": config.name,
        "batch_size": config.batch_size,
        "learning_rate": config.learning_rate,
        "hidden_dim": config.hidden_dim,
        "embed_dim": config.embed_dim,
        "num_layers": config.num_layers,
        "cheb_k": config.cheb_k,
        "best_epoch": int(val_rolling["best_epoch"]),
        "epochs_ran": int(val_rolling["epochs_ran"]),
        "best_val_rmse_pu": float(val_rolling["best_val_rmse_pu"]),
        "train_window_count": int(val_rolling["train_window_count"]),
        "val_rolling_window_count": int(val_rolling["window_count"]),
        "val_non_overlap_window_count": int(val_non_overlap["window_count"]),
        "test_rolling_window_count": int(test_rolling["window_count"]),
        "test_non_overlap_window_count": int(test_non_overlap["window_count"]),
        "val_rolling_rmse_pu": float(val_rolling["rmse_pu"]),
        "val_rolling_mae_pu": float(val_rolling["mae_pu"]),
        "val_non_overlap_rmse_pu": float(val_non_overlap["rmse_pu"]),
        "val_non_overlap_mae_pu": float(val_non_overlap["mae_pu"]),
        "test_rolling_rmse_pu": float(test_rolling["rmse_pu"]),
        "test_rolling_mae_pu": float(test_rolling["mae_pu"]),
        "test_non_overlap_rmse_pu": float(test_non_overlap["rmse_pu"]),
        "test_non_overlap_mae_pu": float(test_non_overlap["mae_pu"]),
        "runtime_seconds": float(val_rolling["runtime_seconds"]),
        "device": str(val_rolling["device"]),
        "seed": int(val_rolling["seed"]),
    }
    _maybe_clear_cuda_cache()
    return summary, frame


def _sort_screen_rows(frame: pl.DataFrame) -> pl.DataFrame:
    return frame.sort(
        by=[
            "dataset_id",
            "model_variant",
            "val_rolling_rmse_pu",
            "val_non_overlap_rmse_pu",
            "runtime_seconds",
            "config_name",
        ]
    )


def _write_json(path: Path, payload: object) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _emit_stage_row(row: dict[str, object]) -> None:
    print(json.dumps(row, ensure_ascii=False, sort_keys=True), flush=True)


def run_search(
    *,
    dataset_ids: Sequence[str],
    tuned_variants: Sequence[str],
    config_names: Sequence[str] | None,
    device: str,
    seed: int,
    screen_train_origins: int,
    screen_eval_origins: int,
    screen_epochs: int,
    screen_patience: int,
    full_epochs: int,
    full_patience: int,
    top_k: int,
    output_dir: Path,
    skip_final: bool,
    full_only: bool,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    output_dir.mkdir(parents=True, exist_ok=True)
    screen_rows: list[dict[str, object]] = []
    final_rows: list[dict[str, object]] = []
    final_detail_frames: list[pl.DataFrame] = []
    allowed_config_names = set(config_names or ())

    selected_configs_by_variant = {
        variant_name: [
            config
            for config in SCREEN_CONFIGS_BY_VARIANT[variant_name]
            if not allowed_config_names or config.name in allowed_config_names
        ]
        for variant_name in tuned_variants
    }
    for variant_name, selected_configs in selected_configs_by_variant.items():
        if not selected_configs:
            raise ValueError(
                f"No configs selected for variant {variant_name!r} after applying --config-name filters."
            )

    if not full_only:
        for dataset_id in dataset_ids:
            prepared_screen = _prepared_by_variant(
                dataset_id,
                max_train_origins=screen_train_origins,
                max_eval_origins=screen_eval_origins,
            )
            for variant_name in tuned_variants:
                prepared_dataset = prepared_screen[variant_name]
                for config in selected_configs_by_variant[variant_name]:
                    row = _screen_one(
                        prepared_dataset,
                        config=config,
                        device=device,
                        seed=seed,
                        max_epochs=screen_epochs,
                        patience=screen_patience,
                    )
                    screen_rows.append(row)
                    _emit_stage_row(row)
                    screen_frame = _sort_screen_rows(pl.DataFrame(screen_rows))
                    screen_frame.write_csv(output_dir / "screen_summary.csv")

    screen_frame = _sort_screen_rows(pl.DataFrame(screen_rows)) if screen_rows else pl.DataFrame()
    _write_json(
        output_dir / "search_plan.json",
        {
            "dataset_ids": list(dataset_ids),
            "alignment_variants": [spec.model_variant for spec in ALIGNMENT_VARIANTS],
            "tuned_variants": list(tuned_variants),
            "screen_train_origins": screen_train_origins,
            "screen_eval_origins": screen_eval_origins,
            "screen_epochs": screen_epochs,
            "screen_patience": screen_patience,
            "full_epochs": full_epochs,
            "full_patience": full_patience,
            "top_k": top_k,
            "seed": seed,
            "device": device,
            "screen_configs": {
                variant_name: [
                    asdict(config)
                    for config in selected_configs_by_variant[variant_name]
                ]
                for variant_name in tuned_variants
            },
            "full_only": full_only,
        },
    )
    if not screen_frame.is_empty():
        screen_frame.write_csv(output_dir / "screen_summary.csv")

    if skip_final:
        return screen_frame, pl.DataFrame()

    for dataset_id in dataset_ids:
        prepared_full = _prepared_by_variant(
            dataset_id,
            max_train_origins=None,
            max_eval_origins=None,
        )
        for variant_name in tuned_variants:
            if full_only:
                configs_for_final = selected_configs_by_variant[variant_name]
            else:
                top_configs = (
                    screen_frame
                    .filter(
                        (pl.col("dataset_id") == dataset_id)
                        & (pl.col("model_variant") == variant_name)
                    )
                    .head(top_k)
                    .to_dicts()
                )
                configs_for_final = [_config_from_row(row) for row in top_configs]
            for config in configs_for_final:
                summary, detail_frame = _final_one(
                    prepared_full[variant_name],
                    config=config,
                    device=device,
                    seed=seed,
                    max_epochs=full_epochs,
                    patience=full_patience,
                )
                final_rows.append(summary)
                final_detail_frames.append(detail_frame)
                _emit_stage_row(summary)
                pl.DataFrame(final_rows).sort(
                    by=[
                        "dataset_id",
                        "model_variant",
                        "test_rolling_rmse_pu",
                        "val_rolling_rmse_pu",
                        "config_name",
                    ]
                ).write_csv(output_dir / "final_summary.csv")
                pl.concat(final_detail_frames, how="diagonal").write_csv(output_dir / "final_detailed_rows.csv")

    final_frame = pl.DataFrame(final_rows).sort(
        by=[
            "dataset_id",
            "model_variant",
            "test_rolling_rmse_pu",
            "val_rolling_rmse_pu",
            "config_name",
        ]
    )
    final_frame.write_csv(output_dir / "final_summary.csv")
    pl.concat(final_detail_frames, how="diagonal").write_csv(output_dir / "final_detailed_rows.csv")
    return screen_frame, final_frame


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    dataset_ids = tuple(args.datasets) if args.datasets else agcrn.DEFAULT_DATASETS
    tuned_variants = tuple(args.variants) if args.variants else agcrn.SEARCH_VARIANTS
    run_search(
        dataset_ids=dataset_ids,
        tuned_variants=tuned_variants,
        config_names=tuple(args.config_names) if args.config_names else None,
        device=args.device,
        seed=args.seed,
        screen_train_origins=args.screen_train_origins,
        screen_eval_origins=args.screen_eval_origins,
        screen_epochs=args.screen_epochs,
        screen_patience=args.screen_patience,
        full_epochs=args.full_epochs,
        full_patience=args.full_patience,
        top_k=args.top_k,
        output_dir=args.output_dir,
        skip_final=args.skip_final,
        full_only=args.full_only,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
