from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import date
import json
import math
from pathlib import Path
import time
from typing import Sequence

import numpy as np
import polars as pl

import world_model_state_space_v1 as state_space


DEFAULT_OUTPUT_DIR = (
    state_space.EXPERIMENT_ROOT
    / "artifacts"
    / "scratch"
    / state_space.FAMILY_ID
    / f"search_{date.today().strftime('%Y%m%d')}"
)
DEFAULT_SCREEN_EPOCHS = 8
DEFAULT_SCREEN_PATIENCE = 3
DEFAULT_FULL_EPOCHS = state_space.DEFAULT_MAX_EPOCHS
DEFAULT_FULL_PATIENCE = state_space.DEFAULT_EARLY_STOPPING_PATIENCE
DEFAULT_TOP_K = 2
DEFAULT_FARM_LOSS_WEIGHTS = (0.0, 0.02, 0.05, 0.1)
ALIGNMENT_VARIANT = state_space.resolve_variant_specs((state_space.MODEL_VARIANT,))[0]
SEARCH_VARIANTS = (ALIGNMENT_VARIANT.model_variant,)


@dataclass(frozen=True)
class SearchConfig:
    name: str
    farm_loss_weight: float


def _config_name_for_farm_loss_weight(value: float) -> str:
    return f"farm_loss_weight_{value:.2f}".replace(".", "p")


COMMON_SCREEN_CONFIGS = tuple(
    SearchConfig(
        name=_config_name_for_farm_loss_weight(value),
        farm_loss_weight=value,
    )
    for value in DEFAULT_FARM_LOSS_WEIGHTS
)


def _maybe_clear_cuda_cache() -> None:
    if state_space.torch is None:
        return
    if state_space.torch.cuda.is_available():
        state_space.torch.cuda.empty_cache()


def _is_oom_error(exc: BaseException) -> bool:
    if state_space.torch is not None:
        oom_error = getattr(state_space.torch, "OutOfMemoryError", None)
        if oom_error is not None and isinstance(exc, oom_error):
            return True
    return "out of memory" in str(exc).lower()


def _config_from_row(row: dict[str, object]) -> SearchConfig:
    return SearchConfig(
        name=str(row["config_name"]),
        farm_loss_weight=float(row["farm_loss_weight"]),
    )


def _matches_selected_farm_loss_weight(config: SearchConfig, selected_weights: Sequence[float]) -> bool:
    return any(math.isclose(config.farm_loss_weight, float(value), rel_tol=0.0, abs_tol=1e-12) for value in selected_weights)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the world-model state-space v1 farm-loss sweep on kelmarsh.",
    )
    parser.add_argument(
        "--farm-loss-weight",
        action="append",
        dest="farm_loss_weights",
        type=float,
        help="Optional farm-loss weights to keep from the default sweep grid.",
    )
    parser.add_argument(
        "--config-name",
        action="append",
        dest="config_names",
        help="Optional config-name filter applied after the farm-loss-weight filter.",
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
        default=state_space.DEFAULT_SEED,
        help="Random seed for all runs.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directory for summary CSV/JSON outputs. Defaults to {DEFAULT_OUTPUT_DIR}.",
    )
    return parser


def _prepare_dataset(dataset_id: str) -> state_space.PreparedDataset:
    return state_space.prepare_dataset(
        dataset_id,
        variant_spec=ALIGNMENT_VARIANT,
    )


def _overall_metric_row(frame: pl.DataFrame, *, split_name: str, eval_protocol: str) -> dict[str, object]:
    return (
        frame.filter(
            (pl.col("split_name") == split_name)
            & (pl.col("eval_protocol") == eval_protocol)
            & (pl.col("metric_scope") == state_space.OVERALL_METRIC_SCOPE)
        ).to_dicts()[0]
    )


def _rolling_horizon_group_means(metrics: state_space.EvaluationMetrics) -> dict[str, float]:
    return state_space._horizon_rmse_pu_group_means(metrics)


def _result_frame_horizon_group_means(
    frame: pl.DataFrame,
    *,
    split_name: str,
    eval_protocol: str,
) -> dict[str, float]:
    horizon_rows = frame.filter(
        (pl.col("split_name") == split_name)
        & (pl.col("eval_protocol") == eval_protocol)
        & (pl.col("metric_scope") == state_space.HORIZON_METRIC_SCOPE)
    ).sort("lead_step")
    horizon_rmse_pu = horizon_rows["rmse_pu"].to_numpy()
    return {
        column_name: state_space._mean_horizon_values(
            np.asarray(horizon_rmse_pu, dtype=np.float64),
            start_lead=start_lead,
            end_lead=end_lead,
        )
        for column_name, _tag_name, start_lead, end_lead in state_space._VAL_RMSE_GROUP_SPECS
    }


def _screen_one(
    prepared_dataset: state_space.PreparedDataset,
    *,
    config: SearchConfig,
    device: str,
    seed: int,
    max_epochs: int,
    patience: int,
) -> dict[str, object]:
    runtime_start = time.monotonic()
    profile = state_space.resolve_hyperparameter_profile(
        prepared_dataset.model_variant,
        dataset_id=prepared_dataset.dataset_id,
        max_epochs=max_epochs,
        early_stopping_patience=patience,
        farm_loss_weight=config.farm_loss_weight,
    )
    training_outcome = state_space.train_model(
        prepared_dataset,
        device=device,
        seed=seed,
        batch_size=profile.batch_size,
        learning_rate=profile.learning_rate,
        max_epochs=profile.max_epochs,
        early_stopping_patience=profile.early_stopping_patience,
        z_dim=profile.z_dim,
        h_dim=profile.h_dim,
        global_state_dim=profile.global_state_dim,
        obs_encoding_dim=profile.obs_encoding_dim,
        innovation_dim=profile.innovation_dim,
        source_summary_dim=profile.source_summary_dim,
        edge_message_dim=profile.edge_message_dim,
        edge_hidden_dim=profile.edge_hidden_dim,
        tau_embed_dim=profile.tau_embed_dim,
        met_summary_dim=profile.met_summary_dim,
        turbine_embed_dim=profile.turbine_embed_dim,
        dropout=profile.dropout,
        grad_clip_norm=profile.grad_clip_norm,
        hist_recon_loss_weight=profile.hist_recon_loss_weight,
        farm_loss_weight=profile.farm_loss_weight,
        met_loss_weight=profile.met_loss_weight,
        innovation_loss_weight=profile.innovation_loss_weight,
        weight_decay=profile.weight_decay,
        wake_lambda_x=profile.wake_lambda_x,
        wake_lambda_y=profile.wake_lambda_y,
        wake_kappa=profile.wake_kappa,
        bounded_output_epsilon=profile.bounded_output_epsilon,
        progress_label=f"screen/{prepared_dataset.dataset_id}/{config.name}",
    )
    resolved_eval_batch_size = state_space.resolve_eval_batch_size(
        profile.batch_size,
        device=training_outcome.device,
        eval_batch_size=None,
    )
    val_rolling_loader = state_space._build_dataloader(
        prepared_dataset,
        windows=prepared_dataset.val_rolling_windows,
        batch_size=resolved_eval_batch_size,
        device=training_outcome.device,
        shuffle=False,
        seed=seed,
    )
    val_non_overlap_loader = state_space._build_dataloader(
        prepared_dataset,
        windows=prepared_dataset.val_non_overlap_windows,
        batch_size=resolved_eval_batch_size,
        device=training_outcome.device,
        shuffle=False,
        seed=seed,
    )
    val_rolling = state_space.evaluate_model(
        training_outcome.model,
        val_rolling_loader,
        device=training_outcome.device,
        rated_power_kw=prepared_dataset.rated_power_kw,
        forecast_steps=prepared_dataset.forecast_steps,
        amp_enabled=training_outcome.amp_enabled,
        progress_label=f"screen/{prepared_dataset.dataset_id}/{config.name}/val_rolling",
    )
    val_non_overlap = state_space.evaluate_model(
        training_outcome.model,
        val_non_overlap_loader,
        device=training_outcome.device,
        rated_power_kw=prepared_dataset.rated_power_kw,
        forecast_steps=prepared_dataset.forecast_steps,
        amp_enabled=training_outcome.amp_enabled,
        progress_label=f"screen/{prepared_dataset.dataset_id}/{config.name}/val_non_overlap",
    )
    rolling_group_means = _rolling_horizon_group_means(val_rolling)
    row = {
        "dataset_id": prepared_dataset.dataset_id,
        "model_variant": prepared_dataset.model_variant,
        "feature_protocol_id": prepared_dataset.feature_protocol_id,
        "stage": "screen",
        "config_name": config.name,
        "farm_loss_weight": config.farm_loss_weight,
        "best_epoch": training_outcome.best_epoch,
        "epochs_ran": training_outcome.epochs_ran,
        "best_val_rmse_pu": training_outcome.best_val_rmse_pu,
        "val_rolling_window_count": len(prepared_dataset.val_rolling_windows),
        "val_non_overlap_window_count": len(prepared_dataset.val_non_overlap_windows),
        "val_rolling_rmse_pu": float(val_rolling.rmse_pu),
        "val_rolling_mae_pu": float(val_rolling.mae_pu),
        "val_non_overlap_rmse_pu": float(val_non_overlap.rmse_pu),
        "val_non_overlap_mae_pu": float(val_non_overlap.mae_pu),
        "val_rmse_pu_leads_13_24_mean": rolling_group_means["val_rmse_pu_leads_13_24_mean"],
        "val_rmse_pu_leads_25_36_mean": rolling_group_means["val_rmse_pu_leads_25_36_mean"],
        "train_window_count": len(prepared_dataset.train_windows),
        "runtime_seconds": round(time.monotonic() - runtime_start, 6),
        "device": training_outcome.device,
        "seed": seed,
    }
    _maybe_clear_cuda_cache()
    return row


def _final_one(
    prepared_dataset: state_space.PreparedDataset,
    *,
    config: SearchConfig,
    device: str,
    seed: int,
    max_epochs: int,
    patience: int,
) -> tuple[dict[str, object], pl.DataFrame]:
    rows = state_space.execute_training_job(
        prepared_dataset,
        device=device,
        seed=seed,
        max_epochs=max_epochs,
        early_stopping_patience=patience,
        farm_loss_weight=config.farm_loss_weight,
    )
    frame = pl.DataFrame(rows)
    val_rolling = _overall_metric_row(
        frame,
        split_name="val",
        eval_protocol=state_space.ROLLING_EVAL_PROTOCOL,
    )
    val_non_overlap = _overall_metric_row(
        frame,
        split_name="val",
        eval_protocol=state_space.NON_OVERLAP_EVAL_PROTOCOL,
    )
    test_rolling = _overall_metric_row(
        frame,
        split_name="test",
        eval_protocol=state_space.ROLLING_EVAL_PROTOCOL,
    )
    test_non_overlap = _overall_metric_row(
        frame,
        split_name="test",
        eval_protocol=state_space.NON_OVERLAP_EVAL_PROTOCOL,
    )
    rolling_group_means = _result_frame_horizon_group_means(
        frame,
        split_name="val",
        eval_protocol=state_space.ROLLING_EVAL_PROTOCOL,
    )
    summary = {
        "dataset_id": prepared_dataset.dataset_id,
        "model_variant": prepared_dataset.model_variant,
        "feature_protocol_id": prepared_dataset.feature_protocol_id,
        "stage": "final",
        "config_name": config.name,
        "farm_loss_weight": config.farm_loss_weight,
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
        "val_rmse_pu_leads_13_24_mean": rolling_group_means["val_rmse_pu_leads_13_24_mean"],
        "val_rmse_pu_leads_25_36_mean": rolling_group_means["val_rmse_pu_leads_25_36_mean"],
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
            "best_val_rmse_pu",
            "config_name",
        ]
    )


def _sort_final_rows(frame: pl.DataFrame) -> pl.DataFrame:
    return frame.sort(
        by=[
            "dataset_id",
            "model_variant",
            "best_val_rmse_pu",
            "config_name",
        ]
    )


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _emit_stage_row(row: dict[str, object]) -> None:
    print(json.dumps(row, ensure_ascii=False, sort_keys=True), flush=True)


def _emit_oom_event(
    *,
    stage: str,
    prepared_dataset: state_space.PreparedDataset,
    config: SearchConfig,
    exc: BaseException,
) -> None:
    _emit_stage_row(
        {
            "stage": stage,
            "status": "oom",
            "dataset_id": prepared_dataset.dataset_id,
            "model_variant": prepared_dataset.model_variant,
            "feature_protocol_id": prepared_dataset.feature_protocol_id,
            "config_name": config.name,
            "farm_loss_weight": config.farm_loss_weight,
            "error": str(exc),
        }
    )


def _selected_defaults_payload(final_frame: pl.DataFrame) -> dict[str, object]:
    selected_rows = (
        _sort_final_rows(final_frame)
        .group_by(["dataset_id", "model_variant"], maintain_order=True)
        .first()
        .sort(["dataset_id", "model_variant"])
        .to_dicts()
    )
    return {
        "selection_rule": [
            "best_val_rmse_pu",
            "config_name",
        ],
        "selected_defaults": {
            str(row["dataset_id"]): {
                str(row["model_variant"]): {
                    "feature_protocol_id": row["feature_protocol_id"],
                    "config_name": row["config_name"],
                    "farm_loss_weight": row["farm_loss_weight"],
                }
            }
            for row in selected_rows
        },
    }


def run_search(
    *,
    dataset_ids: Sequence[str] = state_space.DEFAULT_DATASETS,
    farm_loss_weights: Sequence[float] | None = None,
    config_names: Sequence[str] | None = None,
    device: str = "auto",
    seed: int = state_space.DEFAULT_SEED,
    screen_epochs: int = DEFAULT_SCREEN_EPOCHS,
    screen_patience: int = DEFAULT_SCREEN_PATIENCE,
    full_epochs: int = DEFAULT_FULL_EPOCHS,
    full_patience: int = DEFAULT_FULL_PATIENCE,
    top_k: int = DEFAULT_TOP_K,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    if top_k <= 0:
        raise ValueError("top_k must be positive.")
    resolved_dataset_ids = state_space._validate_dataset_ids(tuple(dataset_ids))
    output_dir.mkdir(parents=True, exist_ok=True)

    selected_farm_loss_weights = tuple(farm_loss_weights or ())
    allowed_config_names = set(config_names or ())
    selected_configs = [
        config
        for config in COMMON_SCREEN_CONFIGS
        if (not selected_farm_loss_weights or _matches_selected_farm_loss_weight(config, selected_farm_loss_weights))
        and (not allowed_config_names or config.name in allowed_config_names)
    ]
    if not selected_configs:
        raise ValueError("No farm-loss sweep configs selected after applying filters.")

    prepared_by_dataset = {
        dataset_id: _prepare_dataset(dataset_id)
        for dataset_id in resolved_dataset_ids
    }
    screen_rows: list[dict[str, object]] = []
    final_rows: list[dict[str, object]] = []
    final_detail_frames: list[pl.DataFrame] = []

    for dataset_id in resolved_dataset_ids:
        prepared_dataset = prepared_by_dataset[dataset_id]
        for config in selected_configs:
            try:
                row = _screen_one(
                    prepared_dataset,
                    config=config,
                    device=device,
                    seed=seed,
                    max_epochs=screen_epochs,
                    patience=screen_patience,
                )
            except RuntimeError as exc:
                if not _is_oom_error(exc):
                    raise
                _maybe_clear_cuda_cache()
                _emit_oom_event(
                    stage="screen",
                    prepared_dataset=prepared_dataset,
                    config=config,
                    exc=exc,
                )
                continue
            screen_rows.append(row)
            _emit_stage_row(row)
            _sort_screen_rows(pl.DataFrame(screen_rows)).write_csv(output_dir / "screen_summary.csv")

    screen_frame = _sort_screen_rows(pl.DataFrame(screen_rows)) if screen_rows else pl.DataFrame()
    if not screen_frame.is_empty():
        screen_frame.write_csv(output_dir / "screen_summary.csv")
    _write_json(
        output_dir / "search_plan.json",
        {
            "dataset_ids": list(resolved_dataset_ids),
            "model_variant": state_space.MODEL_VARIANT,
            "feature_protocol_id": state_space.FEATURE_PROTOCOL_ID,
            "screen_epochs": screen_epochs,
            "screen_patience": screen_patience,
            "full_epochs": full_epochs,
            "full_patience": full_patience,
            "top_k": top_k,
            "seed": seed,
            "device": device,
            "search_configs": [asdict(config) for config in selected_configs],
            "selection_rule": ["best_val_rmse_pu", "config_name"],
        },
    )

    for dataset_id in resolved_dataset_ids:
        if screen_frame.is_empty():
            continue
        configs_for_final = [
            _config_from_row(row)
            for row in (
                screen_frame
                .filter(pl.col("dataset_id") == dataset_id)
                .head(top_k)
                .to_dicts()
            )
        ]
        prepared_dataset = prepared_by_dataset[dataset_id]
        for config in configs_for_final:
            try:
                summary, detail_frame = _final_one(
                    prepared_dataset,
                    config=config,
                    device=device,
                    seed=seed,
                    max_epochs=full_epochs,
                    patience=full_patience,
                )
            except RuntimeError as exc:
                if not _is_oom_error(exc):
                    raise
                _maybe_clear_cuda_cache()
                _emit_oom_event(
                    stage="final",
                    prepared_dataset=prepared_dataset,
                    config=config,
                    exc=exc,
                )
                continue
            final_rows.append(summary)
            final_detail_frames.append(detail_frame)
            _emit_stage_row(summary)
            _sort_final_rows(pl.DataFrame(final_rows)).write_csv(output_dir / "final_summary.csv")
            pl.concat(final_detail_frames, how="diagonal").write_csv(output_dir / "final_detailed_rows.csv")

    final_frame = _sort_final_rows(pl.DataFrame(final_rows)) if final_rows else pl.DataFrame()
    if not final_frame.is_empty():
        final_frame.write_csv(output_dir / "final_summary.csv")
        pl.concat(final_detail_frames, how="diagonal").write_csv(output_dir / "final_detailed_rows.csv")
        _write_json(output_dir / "selected_defaults.json", _selected_defaults_payload(final_frame))
    return screen_frame, final_frame


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    run_search(
        farm_loss_weights=tuple(args.farm_loss_weights) if args.farm_loss_weights else None,
        config_names=tuple(args.config_names) if args.config_names else None,
        device=args.device,
        seed=args.seed,
        output_dir=args.output_dir,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
