from __future__ import annotations

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import sys


def _load_module():
    module_path = (
        Path(__file__).resolve().parents[1]
        / "experiment"
        / "infra"
        / "common"
        / "experiment_registry.py"
    )
    spec = spec_from_file_location("experiment_registry", module_path)
    assert spec is not None
    assert spec.loader is not None
    module = module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_registry_snapshot_contains_expected_families_and_feature_protocols() -> None:
    module = _load_module()

    snapshot = module.load_registry_snapshot()

    assert set(snapshot.families) == {
        "chronos2_power_only",
        "chronos2_exogenous",
        "ltsf_linear_local",
        "tft_turbine_pilot",
        "agcrn_official_aligned",
        "persistence_turbine_baseline",
    }
    assert set(snapshot.feature_protocols) == {
        "power_only",
        "power_stats_history",
        "staged_past_covariates.stage1_core",
        "staged_past_covariates.stage2_ops",
        "staged_past_covariates.stage3_regime",
        "static_calendar",
        "static_calendar_stage1",
        "static_calendar_stage2",
    }


def test_registry_family_bindings_capture_current_local_labels() -> None:
    module = _load_module()

    snapshot = module.load_registry_snapshot()
    tft = snapshot.families["tft_turbine_pilot"]
    chronos = snapshot.families["chronos2_power_only"]
    persistence = snapshot.families["persistence_turbine_baseline"]

    assert tft.implementation_label_kind == "input_pack"
    assert tft.implementation_bindings["reference"] == "power_only"
    assert tft.implementation_bindings["known_static"] == "static_calendar"
    assert tft.implementation_bindings["hist_stage1"] == "staged_past_covariates.stage1_core"
    assert tft.implementation_bindings["mixed_stage2"] == "static_calendar_stage2"

    assert chronos.implementation_bindings["univariate"] == "power_only"
    assert chronos.implementation_bindings["multivariate_knn6_power_stats"] == "power_stats_history"

    assert persistence.status == "archived"
    assert persistence.training_mode == "analytic_baseline"
    assert persistence.supported_result_splits == ("full",)


def test_registry_dataset_family_feature_matrix_matches_expected_scope() -> None:
    module = _load_module()

    snapshot = module.load_registry_snapshot()
    rows = module.build_dataset_family_feature_matrix(snapshot)
    triples = {(row.dataset_id, row.family_id, row.feature_protocol_id) for row in rows}

    assert len(rows) == 51

    assert (
        "kelmarsh",
        "tft_turbine_pilot",
        "static_calendar_stage2",
    ) in triples
    assert (
        "sdwpf_kddcup",
        "chronos2_exogenous",
        "staged_past_covariates.stage3_regime",
    ) in triples
    assert (
        "penmanshiel",
        "agcrn_official_aligned",
        "power_only",
    ) not in triples
    assert (
        "hill_of_towie",
        "tft_turbine_pilot",
        "power_only",
    ) not in triples


def test_registry_markdown_renderer_mentions_expected_rows() -> None:
    module = _load_module()

    snapshot = module.load_registry_snapshot()
    rows = module.build_dataset_family_feature_matrix(snapshot)
    rendered = module.render_matrix_markdown(rows)

    assert "chronos2_power_only" in rendered
    assert "staged_past_covariates.stage1_core" in rendered
    assert "official_aligned_power_only_farm_sync" in rendered
    assert "archived" in rendered
