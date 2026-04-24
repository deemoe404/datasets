from __future__ import annotations

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import sys

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
    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root / "experiment" / "families" / "world_model_official_baselines_v2"))
    from diagnostics.statistics import error_quantiles, paired_bootstrap_delta

    result = paired_bootstrap_delta(
        baseline_errors=[0.4, 0.5, 0.6, 0.7],
        proposed_errors=[0.2, 0.3, 0.4, 0.5],
        repeats=200,
        seed=3407,
    )
    quantiles = error_quantiles([0.1, 0.2, 0.3, 0.4])

    assert result["delta_mean"] > 0
    assert result["prob_delta_gt_zero"] == 1.0
    assert result["ci95_low"] > 0
    assert quantiles == {"ae_p50": 0.25, "ae_p90": 0.37, "ae_p95": 0.385}


def test_formal_tuning_support_is_fail_closed() -> None:
    module = _load_module()
    formal = _load_formal_tuning_module()
    specs = {spec.model_variant: spec for spec in module.resolve_variant_specs(None)}

    assert formal.formal_support_status(specs["baseline_last_value_persistence_v2"]) == ("supported", None)
    assert formal.formal_support_status(specs["baseline_ridge_residual_persistence_b0_v2"]) == ("supported", None)
    assert formal.formal_support_status(specs["chronos2_official_zero_shot_b2_v2"]) == ("supported", None)
    status, blocker = formal.formal_support_status(specs["dgcrn_official_core_residual_b2_v2"])
    assert status == "blocked"
    assert blocker == "official_core_training_adapter_not_implemented"
