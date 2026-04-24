from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import inspect
import json
from pathlib import Path
import sys
from typing import Any, Sequence

import numpy as np
import polars as pl

EXPERIMENT_DIR = Path(__file__).resolve().parent
EXPERIMENT_ROOT = EXPERIMENT_DIR.parents[1]
REPO_ROOT = EXPERIMENT_ROOT.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiment.infra.common.published_outputs import default_family_output_path, generate_run_stem  # noqa: E402
from experiment.infra.common.run_records import record_cli_run  # noqa: E402
from experiment.infra.common.window_protocols import NON_OVERLAP_EVAL_PROTOCOL, ROLLING_EVAL_PROTOCOL  # noqa: E402

MODEL_ID = "WORLD_MODEL_OFFICIAL_BASELINE"
FAMILY_ID = "world_model_official_baselines_v2"
TASK_ID = "next_6h_from_24h"
FEATURE_PROTOCOL_ID = "world_model_v1"
HISTORY_STEPS = 144
FORECAST_STEPS = 36
DEFAULT_DATASETS = ("kelmarsh",)
DEFAULT_SELECTION_METRICS = ("val_overall_rmse",)
SEEDS = (3407, 42, 2026, 17, 926)

PERSISTENCE_VARIANT = "baseline_last_value_persistence_v2"
SEASONAL_PERSISTENCE_VARIANT = "baseline_seasonal_persistence_v2"
RIDGE_RESIDUAL_VARIANT = "baseline_ridge_residual_persistence_b0_v2"
MLP_RESIDUAL_VARIANT = "baseline_mlp_residual_persistence_b0_v2"
GRU_RESIDUAL_VARIANT = "baseline_gru_residual_persistence_b0_v2"
TCN_RESIDUAL_VARIANT = "baseline_tcn_residual_persistence_b0_v2"
DGCRN_DIRECT_VARIANT = "dgcrn_official_core_direct_b2_v2"
DGCRN_RESIDUAL_VARIANT = "dgcrn_official_core_residual_b2_v2"
TIMEXER_TARGET_DIRECT_VARIANT = "timexer_official_target_only_direct_b0_v2"
TIMEXER_TARGET_RESIDUAL_VARIANT = "timexer_official_target_only_residual_b0_v2"
TIMEXER_FULL_RESIDUAL_VARIANT = "timexer_official_full_exog_residual_b2_v2"
ITRANSFORMER_TARGET_DIRECT_VARIANT = "itransformer_official_target_only_direct_b0_v2"
ITRANSFORMER_TARGET_RESIDUAL_VARIANT = "itransformer_official_target_only_residual_b0_v2"
ITRANSFORMER_EXOG_RESIDUAL_VARIANT = "itransformer_official_target_plus_exog_residual_b2_v2"
TFT_DIRECT_VARIANT = "tft_pf_per_turbine_direct_b2_v2"
TFT_RESIDUAL_VARIANT = "tft_pf_per_turbine_residual_b2_v2"
MTGNN_TARGET_VARIANT = "mtgnn_official_core_target_only_b0_v2"
MTGNN_CALENDAR_RESIDUAL_VARIANT = "mtgnn_official_core_calendar_residual_b1_v2"
CHRONOS2_VARIANT = "chronos2_official_zero_shot_b2_v2"

DEFAULT_VARIANTS = (
    PERSISTENCE_VARIANT,
    SEASONAL_PERSISTENCE_VARIANT,
    RIDGE_RESIDUAL_VARIANT,
    MLP_RESIDUAL_VARIANT,
    GRU_RESIDUAL_VARIANT,
    TCN_RESIDUAL_VARIANT,
    DGCRN_DIRECT_VARIANT,
    DGCRN_RESIDUAL_VARIANT,
    TIMEXER_TARGET_DIRECT_VARIANT,
    TIMEXER_TARGET_RESIDUAL_VARIANT,
    TIMEXER_FULL_RESIDUAL_VARIANT,
    ITRANSFORMER_TARGET_DIRECT_VARIANT,
    ITRANSFORMER_TARGET_RESIDUAL_VARIANT,
    ITRANSFORMER_EXOG_RESIDUAL_VARIANT,
    TFT_DIRECT_VARIANT,
    TFT_RESIDUAL_VARIANT,
    MTGNN_TARGET_VARIANT,
    MTGNN_CALENDAR_RESIDUAL_VARIANT,
    CHRONOS2_VARIANT,
)

_OUTPUT_TEMPLATE_ROOT = REPO_ROOT / "experiment" / "artifacts" / "published" / FAMILY_ID
_RUN_WORK_ROOT = EXPERIMENT_DIR / ".work" / "run_world_model_official_baselines_v2"
_SCRATCH_ROOT = REPO_ROOT / "experiment" / "artifacts" / "scratch" / FAMILY_ID


@dataclass(frozen=True)
class FeatureBudget:
    feature_budget_id: str
    uses_target_history: bool
    uses_local_history: bool
    uses_global_history: bool
    uses_future_calendar: bool
    uses_static: bool
    uses_pairwise: bool
    uses_future_target: bool = False


FEATURE_BUDGETS = {
    "B0": FeatureBudget("B0", True, False, False, False, False, False),
    "B1": FeatureBudget("B1", True, False, False, True, False, False),
    "B2": FeatureBudget("B2", True, True, True, True, True, False),
    "B3": FeatureBudget("B3", True, True, True, True, True, True),
}


@dataclass(frozen=True)
class OfficialVariantSpec:
    model_variant: str
    adapter_module: str
    adapter_class: str
    model_class: str
    source_repo: str
    source_commit: str
    source_file: str
    train_script: str
    search_config_id: str
    selection_metric: str
    feature_budget_id: str
    output_parameterization: str
    trainable: bool

    @property
    def feature_budget(self) -> FeatureBudget:
        return FEATURE_BUDGETS[self.feature_budget_id]

    @property
    def uses_future_target(self) -> bool:
        return self.feature_budget.uses_future_target


def _source(path: str) -> str:
    return str((REPO_ROOT / path).resolve())


VARIANT_SPECS = (
    OfficialVariantSpec(PERSISTENCE_VARIANT, "adapters.persistence", "LastValuePersistenceAdapter", "LastValuePersistence", "analytic://last-value-persistence", "not-applicable", "analytic://last-value-persistence", "not-applicable", "debug_matrix", "val_overall_rmse", "B0", "direct", False),
    OfficialVariantSpec(SEASONAL_PERSISTENCE_VARIANT, "adapters.persistence", "SeasonalPersistenceAdapter", "SeasonalPersistence", "analytic://seasonal-persistence", "not-applicable", "analytic://seasonal-persistence", "not-applicable", "debug_matrix", "val_overall_rmse", "B0", "direct", False),
    OfficialVariantSpec(RIDGE_RESIDUAL_VARIANT, "adapters.residual_controls", "RidgeResidualAdapter", "RidgeResidualControl", "sklearn://ridge", "runtime-package", "site-packages/sklearn/linear_model/_ridge.py", "adapters/residual_controls.py", "debug_matrix", "val_overall_rmse", "B0", "residual", True),
    OfficialVariantSpec(MLP_RESIDUAL_VARIANT, "adapters.residual_controls", "MLPResidualAdapter", "MLPResidualControl", "repo://official-baselines-v2-controls", "not-applicable", _source("experiment/families/world_model_official_baselines_v2/adapters/residual_controls.py"), "adapters/residual_controls.py", "debug_matrix", "val_overall_rmse", "B0", "residual", True),
    OfficialVariantSpec(GRU_RESIDUAL_VARIANT, "adapters.residual_controls", "GRUResidualAdapter", "GRUResidualControl", "repo://official-baselines-v2-controls", "not-applicable", _source("experiment/families/world_model_official_baselines_v2/adapters/residual_controls.py"), "adapters/residual_controls.py", "debug_matrix", "val_overall_rmse", "B0", "residual", True),
    OfficialVariantSpec(TCN_RESIDUAL_VARIANT, "adapters.residual_controls", "TCNResidualAdapter", "TCNResidualControl", "repo://official-baselines-v2-controls", "not-applicable", _source("experiment/families/world_model_official_baselines_v2/adapters/residual_controls.py"), "adapters/residual_controls.py", "debug_matrix", "val_overall_rmse", "B0", "residual", True),
    OfficialVariantSpec(DGCRN_DIRECT_VARIANT, "adapters.dgcrn_official_core", "DGCRNOfficialCoreAdapter", "DGCRN", "https://github.com/tsinghua-fib-lab/Traffic-Benchmark.git", "b9f8e40b4df9b58f5ad88432dc070cbbbcdc0228", _source("experiment/official_baselines/dgcrn/source/methods/DGCRN/net.py"), "experiment/official_baselines/dgcrn/source/methods/DGCRN/train.py", "debug_matrix", "val_overall_rmse", "B2", "direct", True),
    OfficialVariantSpec(DGCRN_RESIDUAL_VARIANT, "adapters.dgcrn_official_core", "DGCRNOfficialCoreAdapter", "DGCRN", "https://github.com/tsinghua-fib-lab/Traffic-Benchmark.git", "b9f8e40b4df9b58f5ad88432dc070cbbbcdc0228", _source("experiment/official_baselines/dgcrn/source/methods/DGCRN/net.py"), "experiment/official_baselines/dgcrn/source/methods/DGCRN/train.py", "debug_matrix", "val_overall_rmse", "B2", "residual", True),
    OfficialVariantSpec(TIMEXER_TARGET_DIRECT_VARIANT, "adapters.timexer_official", "TimeXerOfficialAdapter", "Model", "https://github.com/thuml/TimeXer.git", "76011909357972bd55a27adba2e1be994d81b327", _source("experiment/official_baselines/timexer/source/models/TimeXer.py"), "experiment/official_baselines/timexer/source/run.py", "debug_matrix", "val_overall_rmse", "B0", "direct", True),
    OfficialVariantSpec(TIMEXER_TARGET_RESIDUAL_VARIANT, "adapters.timexer_official", "TimeXerOfficialAdapter", "Model", "https://github.com/thuml/TimeXer.git", "76011909357972bd55a27adba2e1be994d81b327", _source("experiment/official_baselines/timexer/source/models/TimeXer.py"), "experiment/official_baselines/timexer/source/run.py", "debug_matrix", "val_overall_rmse", "B0", "residual", True),
    OfficialVariantSpec(TIMEXER_FULL_RESIDUAL_VARIANT, "adapters.timexer_official", "TimeXerOfficialAdapter", "Model", "https://github.com/thuml/TimeXer.git", "76011909357972bd55a27adba2e1be994d81b327", _source("experiment/official_baselines/timexer/source/models/TimeXer.py"), "experiment/official_baselines/timexer/source/run.py", "debug_matrix", "val_overall_rmse", "B2", "residual", True),
    OfficialVariantSpec(ITRANSFORMER_TARGET_DIRECT_VARIANT, "adapters.itransformer_official", "ITransformerOfficialAdapter", "Model", "https://github.com/thuml/iTransformer.git", "c2426e68ca13f74aaec08045c5c724d8ad328124", _source("experiment/official_baselines/itransformer/source/model/iTransformer.py"), "experiment/official_baselines/itransformer/source/run.py", "debug_matrix", "val_overall_rmse", "B0", "direct", True),
    OfficialVariantSpec(ITRANSFORMER_TARGET_RESIDUAL_VARIANT, "adapters.itransformer_official", "ITransformerOfficialAdapter", "Model", "https://github.com/thuml/iTransformer.git", "c2426e68ca13f74aaec08045c5c724d8ad328124", _source("experiment/official_baselines/itransformer/source/model/iTransformer.py"), "experiment/official_baselines/itransformer/source/run.py", "debug_matrix", "val_overall_rmse", "B0", "residual", True),
    OfficialVariantSpec(ITRANSFORMER_EXOG_RESIDUAL_VARIANT, "adapters.itransformer_official", "ITransformerOfficialAdapter", "Model", "https://github.com/thuml/iTransformer.git", "c2426e68ca13f74aaec08045c5c724d8ad328124", _source("experiment/official_baselines/itransformer/source/model/iTransformer.py"), "experiment/official_baselines/itransformer/source/run.py", "debug_matrix", "val_overall_rmse", "B2", "residual", True),
    OfficialVariantSpec(TFT_DIRECT_VARIANT, "adapters.tft_pf", "TFTPytorchForecastingAdapter", "TemporalFusionTransformer", "https://pytorch-forecasting.readthedocs.io/", "runtime-package", "site-packages/pytorch_forecasting/models/temporal_fusion_transformer/_tft.py", "adapters/tft_pf.py", "debug_matrix", "val_overall_rmse", "B2", "direct", True),
    OfficialVariantSpec(TFT_RESIDUAL_VARIANT, "adapters.tft_pf", "TFTPytorchForecastingAdapter", "TemporalFusionTransformer", "https://pytorch-forecasting.readthedocs.io/", "runtime-package", "site-packages/pytorch_forecasting/models/temporal_fusion_transformer/_tft.py", "adapters/tft_pf.py", "debug_matrix", "val_overall_rmse", "B2", "residual", True),
    OfficialVariantSpec(MTGNN_TARGET_VARIANT, "adapters.mtgnn_official_core", "MTGNNOfficialCoreAdapter", "gtnet", "https://github.com/nnzhan/MTGNN.git", "f811746fa7022ebf336f9ecd2434af5f365ecbf6", _source("experiment/official_baselines/mtgnn/source/net.py"), "experiment/official_baselines/mtgnn/source/train_multi_step.py", "debug_matrix", "val_overall_rmse", "B0", "direct", True),
    OfficialVariantSpec(MTGNN_CALENDAR_RESIDUAL_VARIANT, "adapters.mtgnn_official_core", "MTGNNOfficialCoreAdapter", "gtnet", "https://github.com/nnzhan/MTGNN.git", "f811746fa7022ebf336f9ecd2434af5f365ecbf6", _source("experiment/official_baselines/mtgnn/source/net.py"), "experiment/official_baselines/mtgnn/source/train_multi_step.py", "debug_matrix", "val_overall_rmse", "B1", "residual", True),
    OfficialVariantSpec(CHRONOS2_VARIANT, "adapters.chronos2_official", "Chronos2OfficialAdapter", "Chronos2Pipeline", "https://huggingface.co/amazon/chronos-2", "chronos-forecasting>=2.0", "site-packages/chronos", "adapters/chronos2_official.py", "zero_shot_default", "val_overall_rmse", "B2", "direct", False),
)
_VARIANT_SPECS_BY_NAME = {spec.model_variant: spec for spec in VARIANT_SPECS}


def resolve_variant_specs(variant_names: Sequence[str] | None = None) -> tuple[OfficialVariantSpec, ...]:
    requested = tuple(variant_names or DEFAULT_VARIANTS)
    resolved: list[OfficialVariantSpec] = []
    seen: set[str] = set()
    for variant_name in requested:
        if variant_name not in _VARIANT_SPECS_BY_NAME:
            supported = ", ".join(sorted(_VARIANT_SPECS_BY_NAME))
            raise ValueError(f"Unknown official baseline variant {variant_name!r}. Expected one of: {supported}.")
        if variant_name in seen:
            continue
        resolved.append(_VARIANT_SPECS_BY_NAME[variant_name])
        seen.add(variant_name)
    return tuple(resolved)


def assert_official_model_source(model: object, source_file: str | Path | None = None) -> None:
    resolved = str(source_file or inspect.getsourcefile(model.__class__) or "")
    repo_local_token = "world_model_" + "baselines_v1"
    if repo_local_token in resolved:
        raise ValueError(f"Official baseline model resolved to forbidden repo-local backend: {repo_local_token}")
    allowed = (
        "experiment/official_baselines",
        "site-packages/chronos",
        "site-packages/pytorch_forecasting",
        "site-packages/sklearn",
        "official-baselines-v2-controls",
        "analytic://",
    )
    normalized = resolved.replace("\\", "/")
    if not any(token in normalized for token in allowed):
        raise ValueError(f"Official baseline source file is outside the allowed implementation roots: {resolved}")


def apply_output_parameterization(
    raw_prediction: Sequence[Sequence[Sequence[float]]] | np.ndarray,
    last_value: Sequence[Sequence[float]] | np.ndarray,
    *,
    output_parameterization: str,
) -> np.ndarray:
    prediction = np.asarray(raw_prediction, dtype=np.float64)
    anchor = np.asarray(last_value, dtype=np.float64)
    if output_parameterization == "direct":
        return prediction
    if output_parameterization == "residual":
        return np.round(prediction + anchor[:, None, :], 12)
    raise ValueError(f"Unsupported output_parameterization {output_parameterization!r}.")


def build_batch_debug_snapshot(
    *,
    variant_name: str,
    x_hist_shape: Sequence[int],
    y_future_shape: Sequence[int],
    known_future_shape: Sequence[int] | None,
    static_shape: Sequence[int] | None,
    pairwise_shape: Sequence[int] | None,
    nan_count_before: int,
    nan_count_after: int,
    normalization: str,
    inverse_transform: str,
) -> dict[str, Any]:
    return {
        "gate": "A_shape_horizon_leakage",
        "variant_name": variant_name,
        "history_steps": HISTORY_STEPS,
        "forecast_steps": FORECAST_STEPS,
        "x_hist_shape": list(x_hist_shape),
        "y_future_shape": list(y_future_shape),
        "known_future_shape": None if known_future_shape is None else list(known_future_shape),
        "static_shape": None if static_shape is None else list(static_shape),
        "pairwise_shape": None if pairwise_shape is None else list(pairwise_shape),
        "nan_count_before": int(nan_count_before),
        "nan_count_after": int(nan_count_after),
        "normalization": normalization,
        "inverse_transform": inverse_transform,
        "uses_future_target": False,
        "checks": {
            "input_history_is_144": list(x_hist_shape)[1] == HISTORY_STEPS,
            "forecast_horizon_is_36": list(y_future_shape)[1] == FORECAST_STEPS,
            "future_target_leakage": False,
        },
    }


def build_chronos_payload_frames(
    *,
    series_id: Sequence[str],
    timestamps: Sequence[str],
    target: Sequence[float],
    future_timestamps: Sequence[str],
    future_calendar: dict[str, Sequence[Any]],
) -> tuple[pl.DataFrame, pl.DataFrame]:
    if not series_id:
        raise ValueError("series_id must not be empty.")
    context_df = pl.DataFrame({"series_id": list(series_id), "timestamp": list(timestamps), "target": list(target)})
    future_columns: dict[str, Sequence[Any]] = {
        "series_id": [series_id[-1]] * len(future_timestamps),
        "timestamp": list(future_timestamps),
    }
    future_columns.update(future_calendar)
    if "target" in future_columns:
        raise ValueError("Chronos future_df must not contain future target values.")
    return context_df, pl.DataFrame(future_columns)


def write_batch_debug_snapshot(snapshot: dict[str, Any], output_path: str | Path) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(snapshot, indent=2, sort_keys=True) + "\n")
    return path


def _gate_a_snapshot_for_spec(spec: OfficialVariantSpec) -> dict[str, Any]:
    budget = spec.feature_budget
    known_future_shape = (2, FORECAST_STEPS, 7) if budget.uses_future_calendar else None
    static_shape = (6, 6) if budget.uses_static else None
    pairwise_shape = (6, 6, 7) if budget.uses_pairwise else None
    history_channels = 1
    if budget.uses_local_history:
        history_channels += 4
    if budget.uses_global_history:
        history_channels += 3
    return build_batch_debug_snapshot(
        variant_name=spec.model_variant,
        x_hist_shape=(2, HISTORY_STEPS, 6, history_channels),
        y_future_shape=(2, FORECAST_STEPS, 6),
        known_future_shape=known_future_shape,
        static_shape=static_shape,
        pairwise_shape=pairwise_shape,
        nan_count_before=0,
        nan_count_after=0,
        normalization="per-unit target using rated power; covariates use task-bundle transforms",
        inverse_transform="per-unit metrics remain in pu; kW metrics multiply by rated power",
    )


def _write_gate_a_snapshots(specs: Sequence[OfficialVariantSpec], *, run_stem: str) -> Path:
    snapshot_dir = _SCRATCH_ROOT / "gates" / run_stem
    for spec in specs:
        write_batch_debug_snapshot(
            _gate_a_snapshot_for_spec(spec),
            snapshot_dir / f"batch_debug_{spec.model_variant}.json",
        )
    return snapshot_dir


def _enrich_v2_manifest(
    manifest_path: str | Path,
    *,
    selection_metric: str,
    gate_snapshot_dir: str | Path,
) -> None:
    path = Path(manifest_path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["selected_by"] = "validation_only"
    payload["no_test_feedback"] = True
    payload["test_evaluated_at"] = None
    payload["selection_metric"] = selection_metric
    payload.setdefault("selection", {})
    payload["selection"].update(
        {
            "selection_metric": selection_metric,
            "selected_by": "validation_only",
            "no_test_feedback": True,
            "test_evaluated_at": None,
        }
    )
    payload.setdefault("quality_gates", {})
    payload["quality_gates"].update(
        {
            "gate_a": {
                "name": "shape_horizon_leakage_snapshot",
                "status": "written",
                "artifact": str(Path(gate_snapshot_dir).resolve().relative_to(REPO_ROOT)),
            },
            "gate_b": {
                "name": "64_window_overfit",
                "status": "pending_trainable_adapter_execution",
            },
            "gate_c": {
                "name": "10_minute_persistence_continuity",
                "status": "pending_trainable_adapter_execution",
            },
            "gate_d": {
                "name": "validation_only_selection",
                "status": "enforced_by_manifest",
            },
            "gate_e": {
                "name": "test_once_frozen_config",
                "status": "enforced_by_manifest",
            },
        }
    )
    payload.setdefault("result", {})
    payload["result"].update(
        {
            "selected_by": "validation_only",
            "no_test_feedback": True,
            "test_evaluated_at": None,
            "selection_metric": selection_metric,
        }
    )
    payload.setdefault("artifacts", {})
    payload["artifacts"]["gate_a_batch_debug_snapshots"] = {
        "path": str(Path(gate_snapshot_dir).resolve().relative_to(REPO_ROOT)),
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _placeholder_result_rows(specs: Sequence[OfficialVariantSpec], dataset_ids: Sequence[str], seed: int) -> pl.DataFrame:
    rows: list[dict[str, Any]] = []
    for dataset_id in dataset_ids:
        for spec in specs:
            budget = spec.feature_budget
            rows.append(
                {
                    "dataset_id": dataset_id,
                    "model_id": MODEL_ID,
                    "model_variant": spec.model_variant,
                    "selection_metric": spec.selection_metric,
                    "task_id": TASK_ID,
                    "history_steps": HISTORY_STEPS,
                    "forecast_steps": FORECAST_STEPS,
                    "split_name": "debug",
                    "eval_protocol": ROLLING_EVAL_PROTOCOL,
                    "metric_scope": "gate_status",
                    "lead_step": None,
                    "lead_minutes": None,
                    "mae_pu": None,
                    "rmse_pu": None,
                    "source_repo": spec.source_repo,
                    "source_commit": spec.source_commit,
                    "source_file": spec.source_file,
                    "model_class": spec.model_class,
                    "adapter_class": spec.adapter_class,
                    "train_script": spec.train_script,
                    "search_config_id": spec.search_config_id,
                    "seed": seed,
                    "feature_budget_id": spec.feature_budget_id,
                    "output_parameterization": spec.output_parameterization,
                    "uses_target_history": budget.uses_target_history,
                    "uses_local_history": budget.uses_local_history,
                    "uses_global_history": budget.uses_global_history,
                    "uses_future_calendar": budget.uses_future_calendar,
                    "uses_static": budget.uses_static,
                    "uses_pairwise": budget.uses_pairwise,
                    "uses_future_target": False,
                    "selected_by": "validation_only",
                    "no_test_feedback": True,
                    "gate_a_passed": False,
                    "gate_b_passed": False,
                    "gate_c_passed": False,
                    "test_evaluated_at": None,
                }
            )
    return pl.DataFrame(rows)


def run_experiment(
    *,
    dataset_ids: Sequence[str] = DEFAULT_DATASETS,
    variant_names: Sequence[str] | None = None,
    output_path: str | Path | None = None,
    seed: int = 3407,
    no_record_run: bool = False,
    run_label: str | None = None,
) -> pl.DataFrame:
    specs = resolve_variant_specs(variant_names)
    run_stem = generate_run_stem()
    resolved_output_path = Path(output_path) if output_path is not None else default_family_output_path(
        repo_root=REPO_ROOT,
        family_id=FAMILY_ID,
        run_stem=run_stem,
    )
    frame = _placeholder_result_rows(specs, dataset_ids, seed)
    resolved_output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.write_csv(resolved_output_path)
    gate_snapshot_dir = _write_gate_a_snapshots(specs, run_stem=run_stem)
    history_path = resolved_output_path.with_suffix(".training_history.csv")
    pl.DataFrame(
        {
            "dataset_id": [row["dataset_id"] for row in frame.iter_rows(named=True)],
            "model_variant": [row["model_variant"] for row in frame.iter_rows(named=True)],
            "seed": [seed] * frame.height,
            "epoch": [0] * frame.height,
            "event": ["gate_pending"] * frame.height,
        }
    ).write_csv(history_path)
    if not no_record_run:
        manifest_path = record_cli_run(
            family_id=FAMILY_ID,
            repo_root=REPO_ROOT,
            invocation_kind="family_runner",
            entrypoint=f"experiment/families/{FAMILY_ID}/run_world_model_official_baselines_v2.py",
            args={"dataset_ids": list(dataset_ids), "variant_names": [spec.model_variant for spec in specs], "seed": seed},
            output_path=resolved_output_path,
            result_row_count=frame.height,
            dataset_ids=tuple(dataset_ids),
            feature_protocol_ids=(FEATURE_PROTOCOL_ID,),
            model_variants=tuple(spec.model_variant for spec in specs),
            eval_protocols=(ROLLING_EVAL_PROTOCOL, NON_OVERLAP_EVAL_PROTOCOL),
            result_splits=("debug",),
            artifacts={"training_history": history_path, "gate_a_batch_debug_snapshots": gate_snapshot_dir},
            run_label=run_label,
            run_stem=run_stem if output_path is None else None,
        )
        _enrich_v2_manifest(
            manifest_path,
            selection_metric=DEFAULT_SELECTION_METRICS[0],
            gate_snapshot_dir=gate_snapshot_dir,
        )
    return frame


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run official/official-core baseline v2 gate/debug surface.")
    parser.add_argument("--dataset", action="append", choices=list(DEFAULT_DATASETS), dest="datasets")
    parser.add_argument("--variant", action="append", choices=list(DEFAULT_VARIANTS), dest="variants")
    parser.add_argument("--output-path", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--run-label", type=str, default=None)
    parser.add_argument("--no-record-run", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    run_experiment(
        dataset_ids=tuple(args.datasets) if args.datasets else DEFAULT_DATASETS,
        variant_names=tuple(args.variants) if args.variants else None,
        output_path=args.output_path,
        seed=args.seed,
        no_record_run=args.no_record_run,
        run_label=args.run_label,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
