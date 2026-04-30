from __future__ import annotations

from importlib.util import module_from_spec, spec_from_file_location
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest


def _load_module():
    module_path = (
        Path(__file__).resolve().parents[1]
        / "experiment"
        / "families"
        / "world_model_official_baselines_v2"
        / "world_model_official_baselines_v2.py"
    )
    spec = spec_from_file_location("world_model_official_baselines_v2", module_path)
    assert spec is not None
    assert spec.loader is not None
    module = module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _load_formal_tuning_module():
    module_path = (
        Path(__file__).resolve().parents[1]
        / "experiment"
        / "families"
        / "world_model_official_baselines_v2"
        / "formal_tuning.py"
    )
    family_dir = module_path.parent
    if str(family_dir) not in sys.path:
        sys.path.insert(0, str(family_dir))
    spec = spec_from_file_location("world_model_official_baselines_v2_formal_tuning", module_path)
    assert spec is not None
    assert spec.loader is not None
    module = module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _load_statistics_module():
    family_dir = (
        Path(__file__).resolve().parents[1]
        / "experiment"
        / "families"
        / "world_model_official_baselines_v2"
    )
    if str(family_dir) not in sys.path:
        sys.path.insert(0, str(family_dir))
    from diagnostics import statistics

    return statistics


def _load_statistics_artifact_module():
    module_path = (
        Path(__file__).resolve().parents[1]
        / "experiment"
        / "families"
        / "world_model_official_baselines_v2"
        / "diagnostics"
        / "generate_statistics_artifacts.py"
    )
    family_dir = module_path.parents[1]
    if str(family_dir) not in sys.path:
        sys.path.insert(0, str(family_dir))
    spec = spec_from_file_location("world_model_official_baselines_v2_generate_statistics_artifacts", module_path)
    assert spec is not None
    assert spec.loader is not None
    module = module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _load_chronos_shards_module():
    module_path = (
        Path(__file__).resolve().parents[1]
        / "experiment"
        / "families"
        / "world_model_official_baselines_v2"
        / "diagnostics"
        / "chronos2_rolling_shards.py"
    )
    family_dir = module_path.parents[1]
    if str(family_dir) not in sys.path:
        sys.path.insert(0, str(family_dir))
    spec = spec_from_file_location("world_model_official_baselines_v2_chronos2_rolling_shards", module_path)
    assert spec is not None
    assert spec.loader is not None
    module = module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _load_tft_shards_module():
    module_path = (
        Path(__file__).resolve().parents[1]
        / "experiment"
        / "families"
        / "world_model_official_baselines_v2"
        / "diagnostics"
        / "tft_pf_rolling_shards.py"
    )
    family_dir = module_path.parents[1]
    if str(family_dir) not in sys.path:
        sys.path.insert(0, str(family_dir))
    spec = spec_from_file_location("world_model_official_baselines_v2_tft_pf_rolling_shards", module_path)
    assert spec is not None
    assert spec.loader is not None
    module = module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _load_dgcrn_shards_module():
    module_path = (
        Path(__file__).resolve().parents[1]
        / "experiment"
        / "families"
        / "world_model_official_baselines_v2"
        / "diagnostics"
        / "dgcrn_official_core_shards.py"
    )
    family_dir = module_path.parents[1]
    if str(family_dir) not in sys.path:
        sys.path.insert(0, str(family_dir))
    spec = spec_from_file_location("world_model_official_baselines_v2_dgcrn_official_core_shards", module_path)
    assert spec is not None
    assert spec.loader is not None
    module = module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _load_tft_enrich_module():
    module_path = (
        Path(__file__).resolve().parents[1]
        / "experiment"
        / "families"
        / "world_model_official_baselines_v2"
        / "diagnostics"
        / "enrich_tft_origin_errors.py"
    )
    family_dir = module_path.parents[1]
    if str(family_dir) not in sys.path:
        sys.path.insert(0, str(family_dir))
    spec = spec_from_file_location("world_model_official_baselines_v2_enrich_tft_origin_errors", module_path)
    assert spec is not None
    assert spec.loader is not None
    module = module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _load_chronos_enrich_module():
    module_path = (
        Path(__file__).resolve().parents[1]
        / "experiment"
        / "families"
        / "world_model_official_baselines_v2"
        / "diagnostics"
        / "enrich_chronos_origin_errors.py"
    )
    family_dir = module_path.parents[1]
    if str(family_dir) not in sys.path:
        sys.path.insert(0, str(family_dir))
    spec = spec_from_file_location("world_model_official_baselines_v2_enrich_chronos_origin_errors", module_path)
    assert spec is not None
    assert spec.loader is not None
    module = module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _load_long_run_driver_module():
    module_path = (
        Path(__file__).resolve().parents[1]
        / "experiment"
        / "families"
        / "world_model_official_baselines_v2"
        / "diagnostics"
        / "long_run_driver.py"
    )
    family_dir = module_path.parents[1]
    if str(family_dir) not in sys.path:
        sys.path.insert(0, str(family_dir))
    spec = spec_from_file_location("world_model_official_baselines_v2_long_run_driver", module_path)
    assert spec is not None
    assert spec.loader is not None
    module = module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_default_variants_cover_debug_matrix_without_repo_local_backends() -> None:
    module = _load_module()

    assert "world_model_baselines_v1" not in Path(module.__file__).read_text()
    assert module.FAMILY_ID == "world_model_official_baselines_v2"
    assert module.DEFAULT_VARIANTS == (
        "baseline_last_value_persistence_v2",
        "baseline_seasonal_persistence_v2",
        "baseline_ridge_residual_persistence_b0_v2",
        "baseline_mlp_residual_persistence_b0_v2",
        "baseline_gru_residual_persistence_b0_v2",
        "baseline_tcn_residual_persistence_b0_v2",
        "dgcrn_official_core_direct_b2_v2",
        "dgcrn_official_core_residual_b2_v2",
        "dgcrn_official_core_residual_b3_geometry_v2",
        "timexer_official_target_only_direct_b0_v2",
        "timexer_official_target_only_residual_b0_v2",
        "timexer_official_full_exog_residual_b2_v2",
        "itransformer_official_target_only_direct_b0_v2",
        "itransformer_official_target_only_residual_b0_v2",
        "itransformer_official_target_plus_exog_residual_b2_v2",
        "tft_pf_per_turbine_direct_b2_v2",
        "tft_pf_per_turbine_residual_b2_v2",
        "mtgnn_official_core_target_only_b0_v2",
        "mtgnn_official_core_calendar_residual_b1_v2",
        "chronos2_official_zero_shot_b2_v2",
    )
    specs = module.resolve_variant_specs(None)
    assert all(spec.feature_budget_id in {"B0", "B1", "B2", "B3"} for spec in specs)
    assert all(not spec.uses_future_target for spec in specs)
    assert any(spec.output_parameterization == "residual" for spec in specs)
    specs_by_name = {spec.model_variant: spec for spec in specs}
    assert specs_by_name["dgcrn_official_core_residual_b3_geometry_v2"].feature_budget.uses_pairwise is True
    assert specs_by_name["itransformer_official_target_plus_exog_residual_b2_v2"].residual_input_mode == "anchor_centered"
    assert specs_by_name["itransformer_official_target_plus_exog_residual_b2_v2"].official_internal_norm is False
    assert specs_by_name["itransformer_official_target_only_direct_b0_v2"].official_internal_norm is True


def test_source_file_guard_rejects_repo_local_backend() -> None:
    module = _load_module()

    class RepoLocalModel:
        pass

    with pytest.raises(ValueError, match="world_model_baselines_v1"):
        module.assert_official_model_source(
            RepoLocalModel(),
            source_file="/repo/experiment/families/world_model_baselines_v1/world_model_baselines_v1.py",
        )


def test_source_file_guard_accepts_official_sources_and_chronos_package() -> None:
    module = _load_module()

    class OfficialModel:
        pass

    module.assert_official_model_source(
        OfficialModel(),
        source_file="/repo/experiment/official_baselines/timexer/source/models/TimeXer.py",
    )
    module.assert_official_model_source(
        OfficialModel(),
        source_file="/env/lib/python3.11/site-packages/chronos/chronos.py",
    )


def test_batch_debug_snapshot_has_gate_a_contract() -> None:
    module = _load_module()
    snapshot = module.build_batch_debug_snapshot(
        variant_name="dgcrn_official_core_residual_b2_v2",
        x_hist_shape=(2, 144, 6, 1),
        y_future_shape=(2, 36, 6),
        known_future_shape=(2, 36, 7),
        static_shape=(6, 6),
        pairwise_shape=(6, 6, 7),
        nan_count_before=0,
        nan_count_after=0,
        normalization="per-unit target using rated power",
        inverse_transform="multiply by rated power for kW metrics",
    )

    assert snapshot["variant_name"] == "dgcrn_official_core_residual_b2_v2"
    assert snapshot["x_hist_shape"] == [2, 144, 6, 1]
    assert snapshot["y_future_shape"] == [2, 36, 6]
    assert snapshot["known_future_shape"] == [2, 36, 7]
    assert snapshot["uses_future_target"] is False
    assert snapshot["gate"] == "A_shape_horizon_leakage"


def test_residual_reanchor_adds_last_value_only() -> None:
    module = _load_module()
    direct = module.apply_output_parameterization(
        raw_prediction=[[[0.1, 0.2], [0.3, 0.4]]],
        last_value=[[0.5, 0.6]],
        output_parameterization="direct",
    )
    residual = module.apply_output_parameterization(
        raw_prediction=[[[0.1, 0.2], [0.3, 0.4]]],
        last_value=[[0.5, 0.6]],
        output_parameterization="residual",
    )

    assert direct.tolist() == [[[0.1, 0.2], [0.3, 0.4]]]
    assert residual.tolist() == [[[0.6, 0.8], [0.8, 1.0]]]


def test_chronos_payload_excludes_future_target() -> None:
    module = _load_module()
    context_df, future_df = module.build_chronos_payload_frames(
        series_id=["wt01", "wt01", "wt01"],
        timestamps=["2020-01-01T00:00:00", "2020-01-01T00:10:00", "2020-01-01T00:20:00"],
        target=[0.1, 0.2, 0.3],
        future_timestamps=["2020-01-01T00:30:00", "2020-01-01T00:40:00"],
        future_calendar={"hour": [0, 0]},
    )

    assert "target" in context_df.columns
    assert "target" not in future_df.columns
    assert future_df["series_id"].to_list() == ["wt01", "wt01"]


def test_statistics_helpers_report_bootstrap_probability_and_error_quantiles() -> None:
    statistics = _load_statistics_module()

    result = statistics.paired_bootstrap_delta(
        baseline_errors=[0.4, 0.5, 0.6, 0.7],
        proposed_errors=[0.2, 0.3, 0.4, 0.5],
        repeats=200,
        seed=3407,
    )
    block_result = statistics.block_bootstrap_delta(
        baseline_errors=[0.4, 0.5, 0.6, 0.7],
        proposed_errors=[0.2, 0.3, 0.4, 0.5],
        block_length=2,
        repeats=200,
        seed=3407,
    )
    quantiles = statistics.error_quantiles([0.1, 0.2, 0.3, 0.4])

    assert result["delta_mean"] > 0
    assert result["prob_delta_gt_zero"] == 1.0
    assert result["ci95_low"] > 0
    assert block_result["prob_delta_gt_zero"] == 1.0
    assert quantiles == {"ae_p50": 0.25, "ae_p90": 0.37, "ae_p95": 0.385}


def test_statistics_artifact_generation_blocks_without_origin_errors(tmp_path: Path) -> None:
    module = _load_statistics_artifact_module()
    seed_rows = tmp_path / "seed_rows.csv"
    summary_csv = tmp_path / "summary.csv"
    seed_rows.write_text(
        "\n".join(
            [
                "dataset_id,model_id,model_variant,task_id,split_name,eval_protocol,metric_scope,seed,selection_metric,selected_by,no_test_feedback,gate_a_passed,gate_b_passed,gate_b_scope,gate_b_overfit64_passed,gate_c_passed,residual_anchor_steps,formal_search_config_id,is_best_validation_trial,mae_pu,rmse_pu,ae_p50",
                "kelmarsh,m,v,next_6h_from_24h,test,non_overlap,overall,1,val_rmse,validation_only,True,True,True,overfit64_preflight,True,True,1,cfg,True,0.10,0.20,0.05",
                "kelmarsh,m,v,next_6h_from_24h,test,non_overlap,overall,2,val_rmse,validation_only,True,True,True,overfit64_preflight,True,True,1,cfg,True,0.14,0.24,0.07",
                "",
            ]
        ),
        encoding="utf-8",
    )
    summary_csv.write_text(
        "\n".join(
            [
                "dataset_id,model_id,model_variant,task_id,split_name,eval_protocol,metric_scope,selection_metric,selected_by,no_test_feedback,gate_a_passed,gate_b_passed,gate_b_scope,gate_b_overfit64_passed,gate_c_passed,residual_anchor_steps,formal_search_config_id,is_best_validation_trial,mae_pu_mean,mae_pu_std,mae_pu_min,mae_pu_max,rmse_pu_mean,rmse_pu_std,rmse_pu_min,rmse_pu_max,ae_p50_mean,ae_p50_std,ae_p50_min,ae_p50_max,seed_count",
                "kelmarsh,m,v,next_6h_from_24h,test,non_overlap,overall,val_rmse,validation_only,True,True,True,overfit64_preflight,True,True,1,cfg,True,0.12,0.028284271247461898,0.10,0.14,0.22,0.028284271247461888,0.20,0.24,0.060000000000000005,0.014142135623730954,0.05,0.07,2",
                "",
            ]
        ),
        encoding="utf-8",
    )

    artifacts = module.build_statistics_artifacts(
        seed_rows_path=seed_rows,
        summary_csv_path=summary_csv,
        output_prefix=tmp_path / "dgcrn",
        fail_on_summary_mismatch=True,
    )

    validation = json.loads(artifacts["summary_validation"].read_text(encoding="utf-8"))
    bootstrap = json.loads(artifacts["bootstrap_status_json"].read_text(encoding="utf-8"))
    generated_summary = artifacts["statistics_summary"].read_text(encoding="utf-8")

    assert validation["summary_validation_status"] == "passed"
    assert "ae_p50_mean" in generated_summary
    assert bootstrap["bootstrap_status"] == "blocked"
    assert bootstrap["blocked_reason"] == "missing_per_origin_paired_errors"
    assert "paired_bootstrap" not in bootstrap
    assert "prob_delta_gt_zero" not in bootstrap


def test_origin_error_rows_export_paired_bootstrap_schema() -> None:
    module = _load_module()
    formal = _load_formal_tuning_module()
    statistics = _load_statistics_module()
    spec = {spec.model_variant: spec for spec in module.resolve_variant_specs(None)}[
        "dgcrn_official_core_residual_b2_v2"
    ]
    target_pu_filled = np.array(
        [
            [0.1, 0.2],
            [0.2, 0.4],
            [0.4, 0.5],
            [0.6, 0.7],
            [0.8, 0.9],
        ],
        dtype=np.float32,
    )
    target_valid_mask = np.ones_like(target_pu_filled, dtype=bool)
    target_valid_mask[4, 1] = False
    local_history_tensor = np.zeros((5, 2, 18), dtype=np.float32)
    local_history_tensor[:, :, 0] = target_pu_filled
    prepared = SimpleNamespace(
        dataset_id="kelmarsh",
        forecast_steps=2,
        history_steps=2,
        node_count=2,
        target_pu_filled=target_pu_filled,
        target_valid_mask=target_valid_mask,
        local_history_tensor=local_history_tensor,
        persistence_train_fallback_pu=np.array([0.0, 0.0], dtype=np.float32),
    )
    class WindowSpec(SimpleNamespace):
        def __len__(self) -> int:
            return len(self.target_indices)

    windows = WindowSpec(
        target_indices=np.array([2, 3], dtype=np.int64),
        output_start_us=np.array([1_000_000, 2_000_000], dtype=np.int64),
        output_end_us=np.array([3_000_000, 4_000_000], dtype=np.int64),
    )
    predictions = np.array(
        [
            [[0.5, 0.6], [0.7, 0.8]],
            [[0.65, 0.85], [1.05, 1.25]],
        ],
        dtype=np.float32,
    )

    rows = formal._origin_error_rows(
        spec,
        prepared=prepared,
        seed=42,
        split_name="test",
        eval_protocol="non_overlap",
        windows=windows,
        predictions=predictions,
        trial_id="trial20",
        search_config_id="dgcrn_trial20",
        residual_anchor_steps=1,
        best_trial=True,
    )
    bootstrap = statistics.bootstrap_from_comparison_rows(rows, repeats=50, seed=7, block_length=1)

    assert len(rows) == 2
    assert rows[0]["origin_index"] == 0
    assert rows[0]["origin_prediction_count"] == 4
    assert rows[1]["origin_prediction_count"] == 3
    assert rows[0]["baseline_model_variant"] == "baseline_last_value_persistence_v2"
    assert rows[0]["baseline_abs_error_pu"] == pytest.approx(0.25)
    assert rows[0]["proposed_abs_error_pu"] == pytest.approx(0.1)
    assert rows[1]["baseline_abs_error_pu"] == pytest.approx((0.2 + 0.2 + 0.4) / 3)
    assert rows[1]["proposed_abs_error_pu"] == pytest.approx((0.05 + 0.15 + 0.25) / 3)
    assert bootstrap["bootstrap_status"] == "completed"
    assert bootstrap["origin_count"] == 2
    assert bootstrap["baseline_error_column"] == "baseline_abs_error_pu"
    assert bootstrap["proposed_error_column"] == "proposed_abs_error_pu"
    assert bootstrap["paired_bootstrap"]["prob_delta_gt_zero"] == 1.0


def test_origin_error_export_rejects_missing_coverage_and_shape_mismatch() -> None:
    module = _load_module()
    formal = _load_formal_tuning_module()
    spec = {spec.model_variant: spec for spec in module.resolve_variant_specs(None)}[
        "dgcrn_official_core_residual_b2_v2"
    ]

    class WindowSpec(SimpleNamespace):
        def __len__(self) -> int:
            return len(self.target_indices)

    prepared = SimpleNamespace(
        dataset_id="kelmarsh",
        forecast_steps=2,
        history_steps=2,
        node_count=2,
        target_pu_filled=np.ones((5, 2), dtype=np.float32),
        target_valid_mask=np.ones((5, 2), dtype=bool),
        local_history_tensor=np.zeros((5, 2, 18), dtype=np.float32),
        persistence_train_fallback_pu=np.zeros((2,), dtype=np.float32),
    )
    windows = WindowSpec(
        target_indices=np.array([2], dtype=np.int64),
        output_start_us=np.array([1_000], dtype=np.int64),
        output_end_us=np.array([2_000], dtype=np.int64),
    )
    predictions = np.ones((1, 2, 2), dtype=np.float32)

    with pytest.raises(ValueError, match="shape mismatch"):
        formal._per_origin_abs_error_pu(predictions[:, :1], predictions, predictions)

    prepared.target_valid_mask[2:4] = False
    with pytest.raises(ValueError, match="missing coverage"):
        formal._origin_error_rows(
            spec,
            prepared=prepared,
            seed=42,
            split_name="test",
            eval_protocol="non_overlap",
            windows=windows,
            predictions=predictions,
            trial_id="trial",
            search_config_id="cfg",
            residual_anchor_steps=1,
            best_trial=True,
        )


def test_formal_tuning_support_is_fail_closed() -> None:
    module = _load_module()
    formal = _load_formal_tuning_module()
    specs = {spec.model_variant: spec for spec in module.resolve_variant_specs(None)}

    assert formal.formal_support_status(specs["baseline_last_value_persistence_v2"]) == ("supported", None)
    assert formal.formal_support_status(specs["baseline_ridge_residual_persistence_b0_v2"]) == ("supported", None)
    assert formal.formal_support_status(specs["chronos2_official_zero_shot_b2_v2"]) == ("supported", None)
    assert formal.formal_support_status(specs["dgcrn_official_core_direct_b2_v2"]) == ("supported", None)
    assert formal.formal_support_status(specs["dgcrn_official_core_residual_b2_v2"]) == ("supported", None)
    assert formal.formal_support_status(specs["dgcrn_official_core_residual_b3_geometry_v2"]) == ("supported", None)
    assert formal.formal_support_status(specs["timexer_official_target_only_direct_b0_v2"]) == ("supported", None)
    assert formal.formal_support_status(specs["timexer_official_target_only_residual_b0_v2"]) == ("supported", None)
    assert formal.formal_support_status(specs["timexer_official_full_exog_residual_b2_v2"]) == ("supported", None)
    assert formal.formal_support_status(specs["itransformer_official_target_only_direct_b0_v2"]) == ("supported", None)
    assert formal.formal_support_status(specs["itransformer_official_target_only_residual_b0_v2"]) == ("supported", None)
    assert formal.formal_support_status(specs["itransformer_official_target_plus_exog_residual_b2_v2"]) == (
        "supported",
        None,
    )
    assert formal.formal_support_status(specs["tft_pf_per_turbine_direct_b2_v2"]) == ("supported", None)
    assert formal.formal_support_status(specs["tft_pf_per_turbine_residual_b2_v2"]) == ("supported", None)
    assert formal.formal_support_status(specs["mtgnn_official_core_target_only_b0_v2"]) == ("supported", None)
    assert formal.formal_support_status(specs["mtgnn_official_core_calendar_residual_b1_v2"]) == ("supported", None)


def test_formal_tuning_cli_exposes_neural_search_knobs() -> None:
    formal = _load_formal_tuning_module()

    args = formal.build_arg_parser().parse_args(
        [
            "--output-path",
            "scratch.csv",
            "--dgcrn-hidden-dim",
            "96",
            "--dgcrn-dropout",
            "0.0",
            "--dgcrn-gcn-depth",
            "3",
            "--itransformer-d-model",
            "128",
            "--itransformer-n-heads",
            "8",
            "--itransformer-e-layers",
            "3",
            "--itransformer-dropout",
            "0.2",
            "--tft-hidden-size",
            "64",
            "--tft-lstm-layers",
            "2",
            "--tft-attention-head-size",
            "4",
            "--tft-hidden-continuous-size",
            "16",
            "--tft-dropout",
            "0.3",
            "--tft-streaming-exact-ae-limit",
            "12000000",
            "--mtgnn-gcn-depth",
            "3",
            "--mtgnn-subgraph-size",
            "4",
            "--mtgnn-node-dim",
            "32",
            "--mtgnn-residual-channels",
            "16",
            "--mtgnn-skip-channels",
            "32",
            "--mtgnn-end-channels",
            "64",
            "--mtgnn-layers",
            "2",
            "--mtgnn-dropout",
            "0.2",
            "--gate-b-overfit64-passed",
            "--gate-b-overfit64-rmse-pu",
            "0.028",
            "--gate-b-overfit64-mae-pu",
            "0.019",
            "--gate-b-overfit64-source",
            "scratch/overfit64.csv",
            "--origin-error-output-path",
            "origin-errors.csv",
            "--include-horizon-rows",
        ]
    )

    assert args.dgcrn_hidden_dim == 96
    assert args.dgcrn_dropout == 0.0
    assert args.dgcrn_gcn_depth == 3
    assert args.itransformer_d_model == 128
    assert args.itransformer_n_heads == 8
    assert args.itransformer_e_layers == 3
    assert args.itransformer_dropout == 0.2
    assert args.tft_hidden_size == 64
    assert args.tft_lstm_layers == 2
    assert args.tft_attention_head_size == 4
    assert args.tft_hidden_continuous_size == 16
    assert args.tft_dropout == 0.3
    assert args.tft_streaming_exact_ae_limit == 12_000_000
    assert args.mtgnn_gcn_depth == 3
    assert args.mtgnn_subgraph_size == 4
    assert args.mtgnn_node_dim == 32
    assert args.mtgnn_residual_channels == 16
    assert args.mtgnn_skip_channels == 32
    assert args.mtgnn_end_channels == 64
    assert args.mtgnn_layers == 2
    assert args.mtgnn_dropout == 0.2
    assert args.gate_b_overfit64_passed is True
    assert args.gate_b_overfit64_rmse_pu == 0.028
    assert args.gate_b_overfit64_mae_pu == 0.019
    assert args.gate_b_overfit64_source == "scratch/overfit64.csv"
    assert args.origin_error_output_path == Path("origin-errors.csv")
    assert args.include_horizon_rows is True


def test_timexer_itransformer_residual_batches_center_target_history_by_anchor() -> None:
    formal = _load_formal_tuning_module()

    class WindowSpec(SimpleNamespace):
        def __len__(self) -> int:
            return len(self.target_indices)

    target = np.array(
        [
            [0.1, 0.2],
            [0.3, 0.4],
            [0.5, 0.6],
            [0.7, 0.8],
            [0.9, 1.0],
        ],
        dtype=np.float32,
    )
    local_history_tensor = np.zeros((5, 2, 18), dtype=np.float32)
    local_history_tensor[:, :, 0] = target
    prepared = SimpleNamespace(
        history_steps=3,
        forecast_steps=2,
        node_count=2,
        local_history_tensor=local_history_tensor,
        target_pu_filled=target,
        target_valid_mask=np.ones_like(target, dtype=bool),
        persistence_train_fallback_pu=np.array([0.0, 0.0], dtype=np.float32),
    )
    windows = WindowSpec(
        target_indices=np.array([3], dtype=np.int64),
        output_start_us=np.array([1], dtype=np.int64),
        output_end_us=np.array([2], dtype=np.int64),
    )

    x_direct, y_direct, valid_direct, anchor_direct = next(
        formal._iter_timexer_batches(
            prepared,
            windows,
            batch_size=1,
            shuffle=False,
            seed=0,
            full_exog=False,
            residual_output=False,
        )
    )
    x_residual, y_residual, valid_residual, anchor_residual = next(
        formal._iter_timexer_batches(
            prepared,
            windows,
            batch_size=1,
            shuffle=False,
            seed=0,
            full_exog=False,
            residual_output=True,
        )
    )

    np.testing.assert_allclose(anchor_direct, np.array([[0.5, 0.6]], dtype=np.float32))
    np.testing.assert_allclose(anchor_residual, np.array([[0.5, 0.6]], dtype=np.float32))
    np.testing.assert_allclose(x_direct[0], target[:3])
    np.testing.assert_allclose(x_residual[0], target[:3] - np.array([0.5, 0.6], dtype=np.float32))
    np.testing.assert_allclose(y_residual, y_direct)
    np.testing.assert_allclose(valid_residual, valid_direct)


def test_dgcrn_geometry_adjacency_uses_pairwise_distance_and_self_loops() -> None:
    pytest.importorskip("torch")
    formal = _load_formal_tuning_module()
    pairwise = np.zeros((3, 3, 1), dtype=np.float32)
    pairwise[:, :, 0] = np.array(
        [
            [0.0, 2.0, 4.0],
            [2.0, 0.0, 8.0],
            [4.0, 8.0, 0.0],
        ],
        dtype=np.float32,
    )
    prepared = SimpleNamespace(
        node_count=3,
        pairwise_tensor=pairwise,
        pairwise_feature_names=("distance_in_rotor_diameters",),
    )

    adjacency = formal._dgcrn_geometry_adjacency(prepared, device="cpu").cpu().numpy()

    assert np.isfinite(adjacency).all()
    assert np.diag(adjacency).tolist() == pytest.approx([1.0, 1.0, 1.0])
    assert adjacency[0, 1] == pytest.approx(np.exp(-2.0 / 4.0))
    assert adjacency[1, 2] == pytest.approx(np.exp(-8.0 / 4.0))
    np.testing.assert_allclose(adjacency, adjacency.T)


def test_long_run_driver_plan_contains_paper_grade_queue(tmp_path: Path) -> None:
    driver = _load_long_run_driver_module()
    run_root = tmp_path / "long_run"
    queue_path = driver.write_plan(run_root)
    payload = json.loads(queue_path.read_text(encoding="utf-8"))
    item_ids = {item["item_id"] for item in payload["queue"]}

    assert payload["policy"]["selected_by"] == "validation_only"
    assert payload["policy"]["no_test_feedback"] is True
    assert payload["policy"]["test_metrics_used_for_config_selection"] is False
    assert "phase0_snapshot" in item_ids
    assert "phase1_controls_full_val_test" in item_ids
    assert any(item_id.startswith("dgcrn_b3_geometry_residual_") for item_id in item_ids)
    assert any(item_id.startswith("itransformer_target_plus_exog_residual_") for item_id in item_ids)
    assert any(item_id.startswith("timexer_full_exog_residual_") for item_id in item_ids)
    assert "phase3_materialize_full_validation_queue" in item_ids
    assert "phase4_materialize_test_multiseed_queue" in item_ids


def test_streaming_metrics_match_batch_metrics_across_chunks() -> None:
    formal = _load_formal_tuning_module()
    predictions = np.array(
        [
            [[0.1, 0.4], [0.2, 0.5], [0.3, 0.6]],
            [[0.5, 0.2], [0.4, 0.1], [0.3, 0.0]],
            [[0.8, 0.7], [0.6, 0.5], [0.4, 0.3]],
            [[0.0, 0.1], [0.2, 0.3], [0.4, 0.5]],
        ],
        dtype=np.float32,
    )
    targets = predictions + np.array(
        [
            [[0.01, -0.02], [0.03, -0.04], [0.05, -0.06]],
            [[-0.02, 0.01], [-0.04, 0.03], [-0.06, 0.05]],
            [[0.02, 0.04], [0.06, 0.08], [0.10, 0.12]],
            [[-0.03, -0.01], [-0.05, -0.07], [-0.09, -0.11]],
        ],
        dtype=np.float32,
    )
    valid = np.ones_like(predictions, dtype=np.float32)
    valid[1, 2, 0] = 0.0
    valid[3, 0, 1] = 0.0

    batch = formal._metrics(predictions, targets, valid, rated_power_kw=2050.0)
    streaming = formal._metrics_from_prediction_chunks(
        [
            (predictions[:1], targets[:1], valid[:1]),
            (predictions[1:3], targets[1:3], valid[1:3]),
            (predictions[3:], targets[3:], valid[3:]),
        ],
        forecast_steps=3,
        rated_power_kw=2050.0,
        exact_abs_error_limit=1000,
    )

    for key in (
        "window_count",
        "prediction_count",
        "mae_pu",
        "rmse_pu",
        "mae_kw",
        "rmse_kw",
        "lead1_mae_pu",
        "lead1_rmse_pu",
        "short_rmse_pu",
        "mid_rmse_pu",
        "long_rmse_pu",
        "ae_p50",
        "ae_p90",
        "ae_p95",
    ):
        if np.isnan(batch[key]):
            assert np.isnan(streaming[key])
        else:
            assert streaming[key] == pytest.approx(batch[key])
    assert streaming["metrics_backend"] == "streaming"
    assert streaming["ae_quantile_status"] == "exact"
    assert streaming["ae_quantile_exact_count"] == int(valid.sum())


def test_streaming_metrics_fail_closed_when_exact_quantiles_exceed_limit() -> None:
    formal = _load_formal_tuning_module()
    predictions = np.zeros((2, 3, 2), dtype=np.float32)
    targets = np.ones((2, 3, 2), dtype=np.float32) * 0.25
    valid = np.ones((2, 3, 2), dtype=np.float32)

    streaming = formal._metrics_from_prediction_chunks(
        [(predictions[:1], targets[:1], valid[:1]), (predictions[1:], targets[1:], valid[1:])],
        forecast_steps=3,
        rated_power_kw=2050.0,
        exact_abs_error_limit=4,
    )

    assert streaming["prediction_count"] == 12
    assert streaming["mae_pu"] == pytest.approx(0.25)
    assert streaming["rmse_pu"] == pytest.approx(0.25)
    assert np.isnan(streaming["ae_p50"])
    assert np.isnan(streaming["ae_p90"])
    assert np.isnan(streaming["ae_p95"])
    assert streaming["ae_quantile_status"] == "exact_limit_exceeded"
    assert streaming["ae_quantile_exact_count"] is None


def test_chronos_shard_aggregation_matches_batch_metrics() -> None:
    formal = _load_formal_tuning_module()
    shards = _load_chronos_shards_module()
    predictions = np.array(
        [
            [[0.10, 0.40], [0.20, 0.50], [0.30, 0.60]],
            [[0.50, 0.20], [0.40, 0.10], [0.30, 0.00]],
            [[0.80, 0.70], [0.60, 0.50], [0.40, 0.30]],
            [[0.00, 0.10], [0.20, 0.30], [0.40, 0.50]],
            [[0.30, 0.20], [0.10, 0.00], [0.50, 0.40]],
        ],
        dtype=np.float32,
    )
    targets = predictions + np.array(
        [
            [[0.01, -0.02], [0.03, -0.04], [0.05, -0.06]],
            [[-0.02, 0.01], [-0.04, 0.03], [-0.06, 0.05]],
            [[0.02, 0.04], [0.06, 0.08], [0.10, 0.12]],
            [[-0.03, -0.01], [-0.05, -0.07], [-0.09, -0.11]],
            [[0.07, -0.05], [0.03, -0.01], [0.02, -0.04]],
        ],
        dtype=np.float32,
    )
    valid = np.ones_like(predictions, dtype=np.float32)
    valid[1, 2, 0] = 0.0
    valid[3, 0, 1] = 0.0
    records = []
    for start, stop in ((0, 2), (2, 5)):
        record, exact_abs_errors = shards.build_shard_record_from_arrays(
            predictions[start:stop],
            targets[start:stop],
            valid[start:stop],
            dataset_id="kelmarsh",
            split_name="val",
            eval_protocol=formal.ROLLING_EVAL_PROTOCOL,
            shard_start=start,
            shard_stop=stop,
            total_window_count=5,
            rated_power_kw=2050.0,
            forecast_steps=3,
            node_count=2,
            exact_abs_error_limit=1000,
        )
        record["_exact_abs_errors"] = exact_abs_errors
        records.append(record)

    aggregate = shards.aggregate_shard_records(
        records,
        dataset_id="kelmarsh",
        split_name="val",
        eval_protocol=formal.ROLLING_EVAL_PROTOCOL,
        exact_abs_error_limit=1000,
    )
    batch = formal._metrics(predictions, targets, valid, rated_power_kw=2050.0)

    assert aggregate["status"] == "complete"
    assert aggregate["shard_count"] == 2
    assert aggregate["metrics"]["metrics_backend"] == "chronos2_shard_aggregate"
    assert aggregate["metrics"]["ae_quantile_status"] == "exact"
    assert aggregate["selection_metric"] == "val_overall_rmse"
    assert aggregate["selected_by"] == "validation_only"
    assert aggregate["no_test_feedback"] is True
    for key in (
        "window_count",
        "prediction_count",
        "mae_pu",
        "rmse_pu",
        "mae_kw",
        "rmse_kw",
        "lead1_mae_pu",
        "lead1_rmse_pu",
        "short_rmse_pu",
        "mid_rmse_pu",
        "long_rmse_pu",
        "ae_p50",
        "ae_p90",
        "ae_p95",
    ):
        if np.isnan(batch[key]):
            assert np.isnan(aggregate["metrics"][key])
        else:
            assert aggregate["metrics"][key] == pytest.approx(batch[key])


def test_chronos_shard_aggregation_fails_on_missing_coverage() -> None:
    formal = _load_formal_tuning_module()
    shards = _load_chronos_shards_module()
    predictions = np.zeros((1, 3, 2), dtype=np.float32)
    targets = np.ones((1, 3, 2), dtype=np.float32) * 0.1
    valid = np.ones_like(predictions, dtype=np.float32)
    records = []
    for start, stop in ((0, 1), (2, 3)):
        record, exact_abs_errors = shards.build_shard_record_from_arrays(
            predictions,
            targets,
            valid,
            dataset_id="kelmarsh",
            split_name="val",
            eval_protocol=formal.ROLLING_EVAL_PROTOCOL,
            shard_start=start,
            shard_stop=stop,
            total_window_count=3,
            rated_power_kw=2050.0,
            forecast_steps=3,
            node_count=2,
            exact_abs_error_limit=1000,
        )
        record["_exact_abs_errors"] = exact_abs_errors
        records.append(record)

    with pytest.raises(RuntimeError, match="Missing or overlapping shard coverage"):
        shards.aggregate_shard_records(
            records,
            dataset_id="kelmarsh",
            split_name="val",
            eval_protocol=formal.ROLLING_EVAL_PROTOCOL,
            exact_abs_error_limit=1000,
        )


def test_chronos_shard_aggregation_fails_when_exact_limit_too_small() -> None:
    formal = _load_formal_tuning_module()
    shards = _load_chronos_shards_module()
    predictions = np.zeros((2, 3, 2), dtype=np.float32)
    targets = np.ones((2, 3, 2), dtype=np.float32) * 0.1
    valid = np.ones_like(predictions, dtype=np.float32)
    record, exact_abs_errors = shards.build_shard_record_from_arrays(
        predictions,
        targets,
        valid,
        dataset_id="kelmarsh",
        split_name="val",
        eval_protocol=formal.ROLLING_EVAL_PROTOCOL,
        shard_start=0,
        shard_stop=2,
        total_window_count=2,
        rated_power_kw=2050.0,
        forecast_steps=3,
        node_count=2,
        exact_abs_error_limit=1000,
    )
    record["_exact_abs_errors"] = exact_abs_errors

    with pytest.raises(RuntimeError, match="exceeds exact_abs_error_limit"):
        shards.aggregate_shard_records(
            [record],
            dataset_id="kelmarsh",
            split_name="val",
            eval_protocol=formal.ROLLING_EVAL_PROTOCOL,
            exact_abs_error_limit=4,
        )


def _tft_test_config(formal, shards):
    return shards.tft_frozen_config(
        variant_name=formal.TFT_RESIDUAL_VARIANT,
        seed=3407,
        max_train_origins=512,
        max_checkpoint_origins=256,
        checkpoint_eval_protocol=formal.ROLLING_EVAL_PROTOCOL,
        residual_anchor_steps=1,
        train_batch_size=128,
        max_epochs=8,
        learning_rate=3e-4,
        hidden_size=32,
        lstm_layers=1,
        attention_head_size=4,
        hidden_continuous_size=16,
        dropout=0.1,
        eval_window_chunk_size=512,
    )


def test_tft_shard_aggregation_matches_batch_metrics() -> None:
    formal = _load_formal_tuning_module()
    shards = _load_tft_shards_module()
    frozen_config = _tft_test_config(formal, shards)
    predictions = np.array(
        [
            [[0.10, 0.40], [0.20, 0.50], [0.30, 0.60]],
            [[0.50, 0.20], [0.40, 0.10], [0.30, 0.00]],
            [[0.80, 0.70], [0.60, 0.50], [0.40, 0.30]],
            [[0.00, 0.10], [0.20, 0.30], [0.40, 0.50]],
            [[0.30, 0.20], [0.10, 0.00], [0.50, 0.40]],
        ],
        dtype=np.float32,
    )
    targets = predictions + np.array(
        [
            [[0.01, -0.02], [0.03, -0.04], [0.05, -0.06]],
            [[-0.02, 0.01], [-0.04, 0.03], [-0.06, 0.05]],
            [[0.02, 0.04], [0.06, 0.08], [0.10, 0.12]],
            [[-0.03, -0.01], [-0.05, -0.07], [-0.09, -0.11]],
            [[0.07, -0.05], [0.03, -0.01], [0.02, -0.04]],
        ],
        dtype=np.float32,
    )
    valid = np.ones_like(predictions, dtype=np.float32)
    valid[1, 2, 0] = 0.0
    valid[3, 0, 1] = 0.0
    records = []
    for start, stop in ((0, 2), (2, 5)):
        record, exact_abs_errors = shards.build_shard_record_from_arrays(
            predictions[start:stop],
            targets[start:stop],
            valid[start:stop],
            dataset_id="kelmarsh",
            split_name="val",
            eval_protocol=formal.ROLLING_EVAL_PROTOCOL,
            variant_name=formal.TFT_RESIDUAL_VARIANT,
            frozen_config=frozen_config,
            shard_start=start,
            shard_stop=stop,
            total_window_count=5,
            rated_power_kw=2050.0,
            forecast_steps=3,
            node_count=2,
            exact_abs_error_limit=1000,
        )
        record["_exact_abs_errors"] = exact_abs_errors
        records.append(record)

    aggregate = shards.aggregate_shard_records(
        records,
        dataset_id="kelmarsh",
        split_name="val",
        eval_protocol=formal.ROLLING_EVAL_PROTOCOL,
        variant_name=formal.TFT_RESIDUAL_VARIANT,
        exact_abs_error_limit=1000,
    )
    batch = formal._metrics(predictions, targets, valid, rated_power_kw=2050.0)

    assert aggregate["status"] == "complete"
    assert aggregate["shard_count"] == 2
    assert aggregate["metrics"]["metrics_backend"] == "tft_pf_shard_aggregate"
    assert aggregate["metrics"]["ae_quantile_status"] == "exact"
    assert aggregate["tft_contract"]["uses_future_target"] is False
    assert aggregate["selection_metric"] == "val_overall_rmse"
    assert aggregate["selected_by"] == "validation_only"
    assert aggregate["no_test_feedback"] is True
    for key in (
        "window_count",
        "prediction_count",
        "mae_pu",
        "rmse_pu",
        "mae_kw",
        "rmse_kw",
        "lead1_mae_pu",
        "lead1_rmse_pu",
        "short_rmse_pu",
        "mid_rmse_pu",
        "long_rmse_pu",
        "ae_p50",
        "ae_p90",
        "ae_p95",
    ):
        if np.isnan(batch[key]):
            assert np.isnan(aggregate["metrics"][key])
        else:
            assert aggregate["metrics"][key] == pytest.approx(batch[key])


def test_tft_shard_aggregation_fails_on_missing_coverage() -> None:
    formal = _load_formal_tuning_module()
    shards = _load_tft_shards_module()
    frozen_config = _tft_test_config(formal, shards)
    predictions = np.zeros((1, 3, 2), dtype=np.float32)
    targets = np.ones((1, 3, 2), dtype=np.float32) * 0.1
    valid = np.ones_like(predictions, dtype=np.float32)
    records = []
    for start, stop in ((0, 1), (2, 3)):
        record, exact_abs_errors = shards.build_shard_record_from_arrays(
            predictions,
            targets,
            valid,
            dataset_id="kelmarsh",
            split_name="val",
            eval_protocol=formal.ROLLING_EVAL_PROTOCOL,
            variant_name=formal.TFT_RESIDUAL_VARIANT,
            frozen_config=frozen_config,
            shard_start=start,
            shard_stop=stop,
            total_window_count=3,
            rated_power_kw=2050.0,
            forecast_steps=3,
            node_count=2,
            exact_abs_error_limit=1000,
        )
        record["_exact_abs_errors"] = exact_abs_errors
        records.append(record)

    with pytest.raises(RuntimeError, match="Missing or overlapping shard coverage"):
        shards.aggregate_shard_records(
            records,
            dataset_id="kelmarsh",
            split_name="val",
            eval_protocol=formal.ROLLING_EVAL_PROTOCOL,
            variant_name=formal.TFT_RESIDUAL_VARIANT,
            exact_abs_error_limit=1000,
        )


def test_tft_shard_aggregation_fails_when_exact_limit_too_small() -> None:
    formal = _load_formal_tuning_module()
    shards = _load_tft_shards_module()
    frozen_config = _tft_test_config(formal, shards)
    predictions = np.zeros((2, 3, 2), dtype=np.float32)
    targets = np.ones((2, 3, 2), dtype=np.float32) * 0.1
    valid = np.ones_like(predictions, dtype=np.float32)
    record, exact_abs_errors = shards.build_shard_record_from_arrays(
        predictions,
        targets,
        valid,
        dataset_id="kelmarsh",
        split_name="val",
        eval_protocol=formal.ROLLING_EVAL_PROTOCOL,
        variant_name=formal.TFT_RESIDUAL_VARIANT,
        frozen_config=frozen_config,
        shard_start=0,
        shard_stop=2,
        total_window_count=2,
        rated_power_kw=2050.0,
        forecast_steps=3,
        node_count=2,
        exact_abs_error_limit=1000,
    )
    record["_exact_abs_errors"] = exact_abs_errors

    with pytest.raises(RuntimeError, match="exceeds exact_abs_error_limit"):
        shards.aggregate_shard_records(
            [record],
            dataset_id="kelmarsh",
            split_name="val",
            eval_protocol=formal.ROLLING_EVAL_PROTOCOL,
            variant_name=formal.TFT_RESIDUAL_VARIANT,
            exact_abs_error_limit=4,
        )


def _dgcrn_test_config(shards):
    return shards.dgcrn_frozen_config(
        seed=3407,
        train_batch_size=2,
        max_epochs=1,
        learning_rate=5e-4,
        hidden_dim=8,
        dropout=0.0,
        gcn_depth=1,
        residual_anchor_steps=1,
    )


def test_dgcrn_checkpoint_load_reproduces_fixed_batch(tmp_path: Path) -> None:
    shards = _load_dgcrn_shards_module()
    source_file = (
        Path(__file__).resolve().parents[1]
        / "experiment"
        / "official_baselines"
        / "dgcrn"
        / "source"
        / "methods"
        / "DGCRN"
        / "net.py"
    )
    if not source_file.exists():
        pytest.skip("Official DGCRN source is not present in this checkout.")
    import torch

    config = _dgcrn_test_config(shards)
    prepared = SimpleNamespace(
        node_count=2,
        context_future_tensor=np.zeros((1, 7), dtype=np.float32),
    )
    model, _device = shards.build_model_from_config(prepared, config, device="cpu")
    model.eval()
    torch.manual_seed(3407)
    x = torch.randn(2, 11, 2, 144)
    ycl = torch.randn(2, 11, 2, 36)
    with torch.no_grad():
        before = model(x, ycl=ycl, batches_seen=None, task_level=36).detach().clone()
    checkpoint = tmp_path / "dgcrn.pt"
    torch.save({"model_state_dict": model.state_dict(), "metadata": {"frozen_config": config}}, checkpoint)

    loaded, loaded_config, _payload, _resolved_device = shards.load_model_from_checkpoint(
        prepared,
        checkpoint,
        device="cpu",
    )
    with torch.no_grad():
        after = loaded(x, ycl=ycl, batches_seen=None, task_level=36).detach()

    assert loaded_config == config
    assert torch.allclose(before, after, atol=1e-6)


def test_dgcrn_shard_aggregation_matches_batch_metrics() -> None:
    formal = _load_formal_tuning_module()
    shards = _load_dgcrn_shards_module()
    frozen_config = _dgcrn_test_config(shards)
    predictions = np.array(
        [
            [[0.10, 0.40], [0.20, 0.50], [0.30, 0.60]],
            [[0.50, 0.20], [0.40, 0.10], [0.30, 0.00]],
            [[0.80, 0.70], [0.60, 0.50], [0.40, 0.30]],
            [[0.00, 0.10], [0.20, 0.30], [0.40, 0.50]],
            [[0.30, 0.20], [0.10, 0.00], [0.50, 0.40]],
        ],
        dtype=np.float32,
    )
    targets = predictions + np.array(
        [
            [[0.01, -0.02], [0.03, -0.04], [0.05, -0.06]],
            [[-0.02, 0.01], [-0.04, 0.03], [-0.06, 0.05]],
            [[0.02, 0.04], [0.06, 0.08], [0.10, 0.12]],
            [[-0.03, -0.01], [-0.05, -0.07], [-0.09, -0.11]],
            [[0.07, -0.05], [0.03, -0.01], [0.02, -0.04]],
        ],
        dtype=np.float32,
    )
    valid = np.ones_like(predictions, dtype=np.float32)
    valid[1, 2, 0] = 0.0
    valid[3, 0, 1] = 0.0
    records = []
    for start, stop in ((0, 2), (2, 5)):
        record, exact_abs_errors = shards.build_shard_record_from_arrays(
            predictions[start:stop],
            targets[start:stop],
            valid[start:stop],
            dataset_id="kelmarsh",
            split_name="val",
            eval_protocol=formal.ROLLING_EVAL_PROTOCOL,
            variant_name=formal.DGCRN_RESIDUAL_VARIANT,
            frozen_config=frozen_config,
            seed=3407,
            shard_start=start,
            shard_stop=stop,
            total_window_count=5,
            rated_power_kw=2050.0,
            forecast_steps=3,
            node_count=2,
            exact_abs_error_limit=1000,
        )
        record["_exact_abs_errors"] = exact_abs_errors
        record["_origin_error_rows"] = [
            {
                "origin_index": origin_index,
                "baseline_abs_error_pu": 0.20 + 0.01 * origin_index,
                "proposed_abs_error_pu": 0.10 + 0.01 * origin_index,
                "selected_by": "validation_only",
                "no_test_feedback": True,
            }
            for origin_index in range(start, stop)
        ]
        records.append(record)

    aggregate, origin_rows = shards.aggregate_shard_records(
        records,
        dataset_id="kelmarsh",
        split_name="val",
        eval_protocol=formal.ROLLING_EVAL_PROTOCOL,
        variant_name=formal.DGCRN_RESIDUAL_VARIANT,
        seed=3407,
        exact_abs_error_limit=1000,
    )
    batch = formal._metrics(predictions, targets, valid, rated_power_kw=2050.0)

    assert len(origin_rows) == 5
    assert aggregate["status"] == "complete"
    assert aggregate["shard_count"] == 2
    assert aggregate["metrics"]["metrics_backend"] == "dgcrn_official_core_shard_aggregate"
    assert aggregate["metrics"]["ae_quantile_status"] == "exact"
    assert aggregate["dgcrn_contract"]["uses_future_target"] is False
    assert aggregate["selected_by"] == "validation_only"
    assert aggregate["no_test_feedback"] is True
    for key in (
        "window_count",
        "prediction_count",
        "mae_pu",
        "rmse_pu",
        "mae_kw",
        "rmse_kw",
        "lead1_mae_pu",
        "lead1_rmse_pu",
        "short_rmse_pu",
        "mid_rmse_pu",
        "long_rmse_pu",
        "ae_p50",
        "ae_p90",
        "ae_p95",
    ):
        if np.isnan(batch[key]):
            assert np.isnan(aggregate["metrics"][key])
        else:
            assert aggregate["metrics"][key] == pytest.approx(batch[key])


def test_dgcrn_shard_aggregation_rejects_bad_coverage_config_and_origin_errors() -> None:
    formal = _load_formal_tuning_module()
    shards = _load_dgcrn_shards_module()
    frozen_config = _dgcrn_test_config(shards)
    predictions = np.zeros((1, 3, 2), dtype=np.float32)
    targets = np.ones_like(predictions, dtype=np.float32) * 0.1
    valid = np.ones_like(predictions, dtype=np.float32)

    def make_record(start: int, stop: int, *, with_origin_errors: bool = True):
        record, exact_abs_errors = shards.build_shard_record_from_arrays(
            predictions,
            targets,
            valid,
            dataset_id="kelmarsh",
            split_name="val",
            eval_protocol=formal.ROLLING_EVAL_PROTOCOL,
            variant_name=formal.DGCRN_RESIDUAL_VARIANT,
            frozen_config=frozen_config,
            seed=3407,
            shard_start=start,
            shard_stop=stop,
            total_window_count=2,
            rated_power_kw=2050.0,
            forecast_steps=3,
            node_count=2,
            exact_abs_error_limit=1000,
        )
        record["_exact_abs_errors"] = exact_abs_errors
        if with_origin_errors:
            record["_origin_error_rows"] = [
                {
                    "origin_index": start,
                    "baseline_abs_error_pu": 0.2,
                    "proposed_abs_error_pu": 0.1,
                }
            ]
        return record

    with pytest.raises(RuntimeError, match="Missing or overlapping shard coverage"):
        shards.aggregate_shard_records(
            [make_record(0, 1), make_record(2, 3)],
            dataset_id="kelmarsh",
            split_name="val",
            eval_protocol=formal.ROLLING_EVAL_PROTOCOL,
            variant_name=formal.DGCRN_RESIDUAL_VARIANT,
            seed=3407,
            exact_abs_error_limit=1000,
        )

    bad_config_records = [make_record(0, 1), make_record(1, 2)]
    bad_config_records[1]["frozen_config_hash"] = "different"
    with pytest.raises(RuntimeError, match="config identity mismatch"):
        shards.aggregate_shard_records(
            bad_config_records,
            dataset_id="kelmarsh",
            split_name="val",
            eval_protocol=formal.ROLLING_EVAL_PROTOCOL,
            variant_name=formal.DGCRN_RESIDUAL_VARIANT,
            seed=3407,
            exact_abs_error_limit=1000,
        )

    with pytest.raises(RuntimeError, match="missing origin_error_path"):
        shards.aggregate_shard_records(
            [make_record(0, 1, with_origin_errors=False), make_record(1, 2, with_origin_errors=False)],
            dataset_id="kelmarsh",
            split_name="val",
            eval_protocol=formal.ROLLING_EVAL_PROTOCOL,
            variant_name=formal.DGCRN_RESIDUAL_VARIANT,
            seed=3407,
            exact_abs_error_limit=1000,
        )


def test_dgcrn_shard_driver_plans_exact_intervals() -> None:
    shards = _load_dgcrn_shards_module()

    assert shards.planned_shard_intervals(94458, 2048)[:3] == [(0, 2048), (2048, 4096), (4096, 6144)]
    assert shards.planned_shard_intervals(94458, 2048)[-1] == (94208, 94458)
    assert len(shards.planned_shard_intervals(94458, 2048)) == 47
    assert shards.planned_shard_intervals(2624, 2624) == [(0, 2624)]
    assert shards.planned_shard_intervals(2624, 512)[-1] == (2560, 2624)

    with pytest.raises(ValueError, match="shard_size must be positive"):
        shards.planned_shard_intervals(10, 0)


def test_tft_origin_error_enrichment_validates_alignment_and_schema() -> None:
    enrich = _load_tft_enrich_module()

    class WindowSpec(SimpleNamespace):
        def __len__(self) -> int:
            return len(self.target_indices)

    windows = WindowSpec(
        target_indices=np.array([10, 11], dtype=np.int64),
        output_start_us=np.array([1_000, 2_000], dtype=np.int64),
        output_end_us=np.array([1_999, 2_999], dtype=np.int64),
    )
    metadata = {
        "dataset_id": "kelmarsh",
        "model_id": "WORLD_MODEL_OFFICIAL_BASELINE",
        "model_variant": "tft_pf_per_turbine_residual_b2_v2",
        "task_id": "next_6h_from_24h",
        "history_steps": 144,
        "forecast_steps": 36,
        "seed": 3407,
        "split_name": "test",
        "eval_protocol": "rolling_origin_no_refit",
        "metric_scope": "forecast_origin",
        "window_count": 2,
        "node_count": 6,
        "trial_id": "trial",
        "formal_search_config_id": "cfg",
        "feature_budget_id": "B2",
        "output_parameterization": "residual",
        "selection_metric": "val_overall_rmse",
        "selected_by": "validation_only",
        "no_test_feedback": True,
        "uses_future_target": False,
        "residual_anchor_steps": 1,
        "baseline_model_variant": "baseline_last_value_persistence_v2",
    }
    rows = [
        {
            "origin_index": "3",
            "target_index": "10",
            "output_start_us": "1000",
            "output_end_us": "1999",
            "origin_prediction_count": "216",
            "proposed_abs_error_pu": "0.20",
        },
        {
            "origin_index": "4",
            "target_index": "11",
            "output_start_us": "2000",
            "output_end_us": "2999",
            "origin_prediction_count": "215",
            "proposed_abs_error_pu": "0.30",
        },
    ]

    enriched = enrich.validate_and_enrich_rows(
        rows,
        windows,
        np.array([0.1, 0.4], dtype=np.float64),
        metadata=metadata,
        expected_start=3,
        source_origin_error_path="shard.origin_errors.csv",
        source_shard_start=3,
        source_shard_stop=5,
    )

    assert len(enriched) == 2
    assert enriched[0]["dataset_id"] == "kelmarsh"
    assert enriched[0]["baseline_abs_error_pu"] == pytest.approx(0.1)
    assert enriched[0]["proposed_abs_error_pu"] == pytest.approx(0.2)
    assert enriched[0]["control_abs_error_pu"] == pytest.approx(0.1)
    assert enriched[0]["candidate_abs_error_pu"] == pytest.approx(0.2)
    assert enriched[1]["origin_prediction_count"] == 215
    assert enriched[1]["source_shard_start"] == 3

    bad_rows = [dict(rows[0], target_index="12"), rows[1]]
    with pytest.raises(ValueError, match="target_index mismatch"):
        enrich.validate_and_enrich_rows(
            bad_rows,
            windows,
            np.array([0.1, 0.4], dtype=np.float64),
            metadata=metadata,
            expected_start=3,
            source_origin_error_path="shard.origin_errors.csv",
            source_shard_start=3,
            source_shard_stop=5,
        )


def test_tft_origin_error_enrichment_writes_bootstrap_status(tmp_path: Path) -> None:
    enrich = _load_tft_enrich_module()
    comparison_csv = tmp_path / "comparison.csv"
    comparison_csv.write_text(
        "\n".join(
            [
                "origin_index,baseline_abs_error_pu,proposed_abs_error_pu",
                "0,0.40,0.20",
                "1,0.50,0.30",
                "2,0.60,0.40",
                "",
            ]
        ),
        encoding="utf-8",
    )

    artifacts = enrich.write_bootstrap_artifacts(
        comparison_csv_path=comparison_csv,
        output_prefix=tmp_path / "tft",
        repeats=50,
        seed=7,
        block_length=1,
    )

    status = json.loads(artifacts["bootstrap_status_json"].read_text(encoding="utf-8"))
    status_csv = artifacts["bootstrap_status_csv"].read_text(encoding="utf-8")

    assert status["bootstrap_status"] == "completed"
    assert status["origin_count"] == 3
    assert status["baseline_error_column"] == "baseline_abs_error_pu"
    assert status["proposed_error_column"] == "proposed_abs_error_pu"
    assert status["paired_bootstrap"]["prob_delta_gt_zero"] == 1.0
    assert "paired_bootstrap_prob_delta_gt_zero" in status_csv


def test_chronos_flat_abs_error_enrichment_reconstructs_origin_means() -> None:
    enrich = _load_chronos_enrich_module()

    class WindowSpec(SimpleNamespace):
        def __len__(self) -> int:
            return len(self.target_indices)

    windows = WindowSpec(
        target_indices=np.array([10, 11, 12], dtype=np.int64),
        output_start_us=np.array([1_000, 2_000, 3_000], dtype=np.int64),
        output_end_us=np.array([1_999, 2_999, 3_999], dtype=np.int64),
    )
    metadata = {
        "dataset_id": "kelmarsh",
        "model_id": "WORLD_MODEL_OFFICIAL_BASELINE",
        "model_variant": "chronos2_official_zero_shot_b2_v2",
        "task_id": "next_6h_from_24h",
        "history_steps": 144,
        "forecast_steps": 36,
        "seed": 3407,
        "split_name": "test",
        "eval_protocol": "rolling_origin_no_refit",
        "metric_scope": "forecast_origin",
        "window_count": 3,
        "node_count": 6,
        "trial_id": "chronos2_zero_shot_median",
        "formal_search_config_id": "chronos2_zero_shot_b2",
        "feature_budget_id": "B2",
        "output_parameterization": "direct",
        "selection_metric": "val_overall_rmse",
        "selected_by": "validation_only",
        "no_test_feedback": True,
        "uses_future_target": False,
        "residual_anchor_steps": 0,
        "baseline_model_variant": "baseline_last_value_persistence_v2",
    }
    source_record = {
        "shard_start": 5,
        "shard_stop": 8,
        "target_index_start": 10,
        "target_index_stop_exclusive": 13,
        "output_start_us": 1_000,
        "output_end_us": 3_999,
        "components": {"prediction_count": 6},
    }

    enriched = enrich.validate_and_enrich_flat_abs_errors(
        np.array([0.2, 0.4, 0.3, 0.6, 0.8, 1.0], dtype=np.float64),
        windows,
        np.array([2, 1, 3], dtype=np.int64),
        np.array([0.1, 0.2, 0.3], dtype=np.float64),
        metadata=metadata,
        expected_start=5,
        source_abs_error_path="shard.abs_errors.npy",
        source_shard_start=5,
        source_shard_stop=8,
        source_shard_record=source_record,
    )

    assert [row["origin_index"] for row in enriched] == [5, 6, 7]
    assert [row["origin_prediction_count"] for row in enriched] == [2, 1, 3]
    assert [row["proposed_abs_error_pu"] for row in enriched] == pytest.approx([0.3, 0.3, 0.8])
    assert enriched[2]["baseline_abs_error_pu"] == pytest.approx(0.3)
    assert enriched[2]["candidate_abs_error_pu"] == pytest.approx(0.8)
    assert enriched[0]["source_abs_error_path"] == "shard.abs_errors.npy"

    with pytest.raises(ValueError, match="Flat abs_errors size mismatch"):
        enrich.validate_and_enrich_flat_abs_errors(
            np.array([0.2, 0.4], dtype=np.float64),
            windows,
            np.array([2, 1, 3], dtype=np.int64),
            np.array([0.1, 0.2, 0.3], dtype=np.float64),
            metadata=metadata,
            expected_start=5,
            source_abs_error_path="shard.abs_errors.npy",
            source_shard_start=5,
            source_shard_stop=8,
            source_shard_record=source_record,
        )

    bad_record = dict(source_record, output_end_us=4_001)
    with pytest.raises(ValueError, match="output_end_us mismatch"):
        enrich.validate_and_enrich_flat_abs_errors(
            np.array([0.2, 0.4, 0.3, 0.6, 0.8, 1.0], dtype=np.float64),
            windows,
            np.array([2, 1, 3], dtype=np.int64),
            np.array([0.1, 0.2, 0.3], dtype=np.float64),
            metadata=metadata,
            expected_start=5,
            source_abs_error_path="shard.abs_errors.npy",
            source_shard_start=5,
            source_shard_stop=8,
            source_shard_record=bad_record,
        )


def test_chronos_origin_error_enrichment_writes_bootstrap_status(tmp_path: Path) -> None:
    enrich = _load_chronos_enrich_module()
    comparison_csv = tmp_path / "comparison.csv"
    comparison_csv.write_text(
        "\n".join(
            [
                "origin_index,baseline_abs_error_pu,proposed_abs_error_pu,source_abs_error_path",
                "0,0.40,0.20,shard0.npy",
                "1,0.50,0.30,shard0.npy",
                "2,0.60,0.40,shard0.npy",
                "",
            ]
        ),
        encoding="utf-8",
    )

    artifacts = enrich.write_bootstrap_artifacts(
        comparison_csv_path=comparison_csv,
        output_prefix=tmp_path / "chronos",
        repeats=50,
        seed=7,
        block_length=1,
    )

    status = json.loads(artifacts["bootstrap_status_json"].read_text(encoding="utf-8"))
    status_csv = artifacts["bootstrap_status_csv"].read_text(encoding="utf-8")

    assert status["bootstrap_status"] == "completed"
    assert status["origin_count"] == 3
    assert status["baseline_error_column"] == "baseline_abs_error_pu"
    assert status["proposed_error_column"] == "proposed_abs_error_pu"
    assert status["paired_bootstrap"]["prob_delta_gt_zero"] == 1.0
    assert "paired_bootstrap_prob_delta_gt_zero" in status_csv
