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


def test_registry_snapshot_contains_active_families() -> None:
    module = _load_module()

    snapshot = module.load_registry_snapshot()

    assert set(snapshot.families) == {
        "agcrn_official_aligned",
        "agcrn_masked",
        "world_model_agcrn_v1",
        "world_model_baselines_v1",
        "world_model_hardened_baselines_v1",
        "world_model_official_baselines_v2",
        "world_model_rollout_v1",
        "world_model_state_space_v1",
    }


def test_registry_family_bindings_capture_current_active_contract() -> None:
    module = _load_module()

    snapshot = module.load_registry_snapshot()
    agcrn = snapshot.families["agcrn_official_aligned"]
    agcrn_masked = snapshot.families["agcrn_masked"]
    world_model = snapshot.families["world_model_agcrn_v1"]
    world_model_baselines = snapshot.families["world_model_baselines_v1"]
    world_model_hardened_baselines = snapshot.families["world_model_hardened_baselines_v1"]
    world_model_official_baselines = snapshot.families["world_model_official_baselines_v2"]
    world_model_rollout = snapshot.families["world_model_rollout_v1"]
    world_model_state_space = snapshot.families["world_model_state_space_v1"]

    assert agcrn.status == "pilot"
    assert agcrn.task_contract.granularity == "farm"
    assert agcrn.dataset_scope == ("kelmarsh", "penmanshiel")
    assert agcrn.default_output_path == "experiment/artifacts/published/agcrn_official_aligned/{run_timestamp}.csv"
    assert agcrn.supported_feature_protocols == (
        "power_only",
        "power_ws_hist",
        "power_atemp_hist",
        "power_itemp_hist",
        "power_wd_hist_sincos",
        "power_wd_yaw_hist_sincos",
        "power_wd_yaw_pitchmean_hist_sincos",
        "power_wd_yaw_lrpm_hist_sincos",
        "power_ws_wd_hist_sincos",
    )
    assert agcrn.implementation_bindings == {
        "official_aligned_power_only_farm_sync": "power_only",
        "official_aligned_power_ws_hist_farm_sync": "power_ws_hist",
        "official_aligned_power_atemp_hist_farm_sync": "power_atemp_hist",
        "official_aligned_power_itemp_hist_farm_sync": "power_itemp_hist",
        "official_aligned_power_wd_hist_sincos_farm_sync": "power_wd_hist_sincos",
        "official_aligned_power_wd_yaw_hist_sincos_farm_sync": "power_wd_yaw_hist_sincos",
        "official_aligned_power_wd_yaw_pitchmean_hist_sincos_farm_sync": "power_wd_yaw_pitchmean_hist_sincos",
        "official_aligned_power_wd_yaw_lrpm_hist_sincos_farm_sync": "power_wd_yaw_lrpm_hist_sincos",
        "official_aligned_power_ws_wd_hist_sincos_farm_sync": "power_ws_wd_hist_sincos",
    }
    assert agcrn_masked.status == "pilot"
    assert agcrn_masked.task_contract.granularity == "farm"
    assert agcrn_masked.dataset_scope == ("kelmarsh", "penmanshiel")
    assert agcrn_masked.default_output_path == "experiment/artifacts/published/agcrn_masked/{run_timestamp}.csv"
    assert agcrn_masked.supported_feature_protocols == ("power_wd_yaw_pmean_hist_sincos_masked",)
    assert agcrn_masked.implementation_bindings == {
        "masked_power_wd_yaw_pmean_hist_sincos_farm_sync": "power_wd_yaw_pmean_hist_sincos_masked",
    }
    assert world_model.status == "pilot"
    assert world_model.task_contract.granularity == "farm"
    assert world_model.dataset_scope == ("kelmarsh", "penmanshiel")
    assert world_model.default_output_path == "experiment/artifacts/published/world_model_agcrn_v1/{run_timestamp}.csv"
    assert world_model.supported_feature_protocols == ("world_model_v1",)
    assert world_model.implementation_bindings == {
        "world_model_v1_seq2seq_farm_sync": "world_model_v1",
    }
    assert world_model_baselines.status == "prototype"
    assert world_model_baselines.task_contract.granularity == "farm"
    assert world_model_baselines.dataset_scope == ("kelmarsh",)
    assert (
        world_model_baselines.default_output_path
        == "experiment/artifacts/published/world_model_baselines_v1/{run_timestamp}.csv"
    )
    assert world_model_baselines.supported_feature_protocols == ("world_model_v1",)
    assert world_model_baselines.implementation_bindings == {
        "world_model_persistence_last_value_v1_farm_sync": "world_model_v1",
        "world_model_shared_weight_tft_no_graph_v1_farm_sync": "world_model_v1",
        "world_model_shared_weight_timexer_no_graph_v1_farm_sync": "world_model_v1",
        "world_model_dgcrn_v1_farm_sync": "world_model_v1",
        "world_model_chronos_2_zero_shot_v1_farm_sync": "world_model_v1",
        "world_model_itransformer_no_graph_v1_farm_sync": "world_model_v1",
        "world_model_mtgnn_calendar_graph_v1_farm_sync": "world_model_v1",
    }
    assert world_model_hardened_baselines.status == "prototype"
    assert world_model_hardened_baselines.task_contract.granularity == "farm"
    assert world_model_hardened_baselines.dataset_scope == ("kelmarsh",)
    assert (
        world_model_hardened_baselines.default_output_path
        == "experiment/artifacts/published/world_model_hardened_baselines_v1/{run_timestamp}.csv"
    )
    assert world_model_hardened_baselines.supported_feature_protocols == ("world_model_v1",)
    assert world_model_hardened_baselines.implementation_bindings == {
        "world_model_last_value_persistence_hardened_v1_farm_sync": "world_model_v1",
        "world_model_dgcrn_official_core_v1_farm_sync": "world_model_v1",
        "world_model_timexer_official_v1_farm_sync": "world_model_v1",
        "world_model_itransformer_official_v1_farm_sync": "world_model_v1",
        "world_model_chronos_2_zero_shot_official_v1_farm_sync": "world_model_v1",
    }
    assert world_model_official_baselines.status == "prototype"
    assert world_model_official_baselines.task_contract.granularity == "farm"
    assert world_model_official_baselines.dataset_scope == ("kelmarsh",)
    assert (
        world_model_official_baselines.default_output_path
        == "experiment/artifacts/published/world_model_official_baselines_v2/{run_timestamp}.csv"
    )
    assert world_model_official_baselines.supported_feature_protocols == ("world_model_v1",)
    assert world_model_official_baselines.implementation_bindings["dgcrn_official_core_residual_b2_v2"] == "world_model_v1"
    assert world_model_official_baselines.implementation_bindings["chronos2_official_zero_shot_b2_v2"] == "world_model_v1"
    assert world_model_rollout.status == "prototype"
    assert world_model_rollout.task_contract.granularity == "farm"
    assert world_model_rollout.dataset_scope == ("kelmarsh", "penmanshiel")
    assert world_model_rollout.default_output_path == "experiment/artifacts/published/world_model_rollout_v1/{run_timestamp}.csv"
    assert world_model_rollout.supported_feature_protocols == ("world_model_v1",)
    assert world_model_rollout.implementation_bindings == {
        "world_model_rollout_v1_farm_sync": "world_model_v1",
    }
    assert world_model_state_space.status == "prototype"
    assert world_model_state_space.task_contract.granularity == "farm"
    assert world_model_state_space.dataset_scope == ("kelmarsh",)
    assert (
        world_model_state_space.default_output_path
        == "experiment/artifacts/published/world_model_state_space_v1/{run_timestamp}.csv"
    )
    assert world_model_state_space.supported_feature_protocols == ("world_model_v1",)
    assert world_model_state_space.implementation_bindings == {
        "world_model_state_space_v1_farm_sync": "world_model_v1",
        "world_model_state_space_v1_residual_persistence_farm_sync": "world_model_v1",
        "world_model_state_space_v1_residual_persistence_gated_sum_farm_sync": "world_model_v1",
        "world_model_state_space_v1_residual_persistence_rotor_units_wake_farm_sync": "world_model_v1",
        "world_model_state_space_v1_residual_persistence_gated_sum_rotor_units_wake_farm_sync": "world_model_v1",
        "world_model_state_space_v1_global_local_residual_farm_sync": "world_model_v1",
        "world_model_state_space_v1_global_local_increment_farm_sync": "world_model_v1",
        "world_model_state_space_v1_wake_off_farm_sync": "world_model_v1",
        "world_model_state_space_v1_graph_off_farm_sync": "world_model_v1",
        "world_model_state_space_v1_no_farm_aux_farm_sync": "world_model_v1",
        "world_model_state_space_v1_no_met_aux_farm_sync": "world_model_v1",
    }


def test_registry_dataset_family_feature_matrix_matches_active_scope() -> None:
    module = _load_module()

    snapshot = module.load_registry_snapshot()
    rows = module.build_dataset_family_feature_matrix(snapshot)

    assert len(rows) == 28
    assert rows[0].family_id == "agcrn_masked"
    assert rows[0].feature_protocol_id == "power_wd_yaw_pmean_hist_sincos_masked"
    assert rows[1].family_id == "agcrn_masked"
    assert rows[1].dataset_id == "penmanshiel"
    assert sum(row.family_id == "agcrn_official_aligned" for row in rows) == 18
    assert sum(row.family_id == "agcrn_masked" for row in rows) == 2
    assert sum(row.family_id == "world_model_agcrn_v1" for row in rows) == 2
    assert sum(row.family_id == "world_model_baselines_v1" for row in rows) == 1
    assert sum(row.family_id == "world_model_hardened_baselines_v1" for row in rows) == 1
    assert sum(row.family_id == "world_model_official_baselines_v2" for row in rows) == 1
    assert sum(row.family_id == "world_model_rollout_v1" for row in rows) == 2
    assert sum(row.family_id == "world_model_state_space_v1" for row in rows) == 1
    assert {row.family_status for row in rows} == {"pilot", "prototype"}


def test_registry_markdown_renderer_mentions_active_family() -> None:
    module = _load_module()

    snapshot = module.load_registry_snapshot()
    rows = module.build_dataset_family_feature_matrix(snapshot)
    rendered = module.render_matrix_markdown(rows)

    assert "agcrn_official_aligned" in rendered
    assert "agcrn_masked" in rendered
    assert "world_model_agcrn_v1" in rendered
    assert "world_model_baselines_v1" in rendered
    assert "world_model_hardened_baselines_v1" in rendered
    assert "world_model_official_baselines_v2" in rendered
    assert "world_model_rollout_v1" in rendered
    assert "world_model_state_space_v1" in rendered
    assert "official_aligned_power_only_farm_sync" in rendered
    assert "official_aligned_power_ws_hist_farm_sync" in rendered
    assert "official_aligned_power_atemp_hist_farm_sync" in rendered
    assert "official_aligned_power_itemp_hist_farm_sync" in rendered
    assert "official_aligned_power_wd_hist_sincos_farm_sync" in rendered
    assert "official_aligned_power_wd_yaw_hist_sincos_farm_sync" in rendered
    assert "official_aligned_power_wd_yaw_pitchmean_hist_sincos_farm_sync" in rendered
    assert "official_aligned_power_wd_yaw_lrpm_hist_sincos_farm_sync" in rendered
    assert "official_aligned_power_ws_wd_hist_sincos_farm_sync" in rendered
    assert "power_only" in rendered
    assert "power_ws_hist" in rendered
    assert "power_atemp_hist" in rendered
    assert "power_itemp_hist" in rendered
    assert "power_wd_hist_sincos" in rendered
    assert "power_wd_yaw_hist_sincos" in rendered
    assert "power_wd_yaw_pitchmean_hist_sincos" in rendered
    assert "masked_power_wd_yaw_pmean_hist_sincos_farm_sync" in rendered
    assert "power_wd_yaw_pmean_hist_sincos_masked" in rendered
    assert "world_model_v1_seq2seq_farm_sync" in rendered
    assert "world_model_persistence_last_value_v1_farm_sync" in rendered
    assert "world_model_shared_weight_tft_no_graph_v1_farm_sync" in rendered
    assert "world_model_shared_weight_timexer_no_graph_v1_farm_sync" in rendered
    assert "world_model_dgcrn_v1_farm_sync" in rendered
    assert "world_model_chronos_2_zero_shot_v1_farm_sync" in rendered
    assert "world_model_itransformer_no_graph_v1_farm_sync" in rendered
    assert "world_model_mtgnn_calendar_graph_v1_farm_sync" in rendered
    assert "world_model_dgcrn_official_core_v1_farm_sync" in rendered
    assert "world_model_timexer_official_v1_farm_sync" in rendered
    assert "world_model_itransformer_official_v1_farm_sync" in rendered
    assert "world_model_chronos_2_zero_shot_official_v1_farm_sync" in rendered
    assert "dgcrn_official_core_residual_b2_v2" in rendered
    assert "timexer_official_full_exog_residual_b2_v2" in rendered
    assert "chronos2_official_zero_shot_b2_v2" in rendered
    assert "world_model_rollout_v1_farm_sync" in rendered
    assert "world_model_state_space_v1_farm_sync" in rendered
    assert "world_model_state_space_v1_residual_persistence_farm_sync" in rendered
    assert "world_model_state_space_v1_residual_persistence_gated_sum_farm_sync" in rendered
    assert "world_model_state_space_v1_residual_persistence_rotor_units_wake_farm_sync" in rendered
    assert "world_model_state_space_v1_residual_persistence_gated_sum_rotor_units_wake_farm_sync" in rendered
    assert "world_model_state_space_v1_global_local_residual_farm_sync" in rendered
    assert "world_model_state_space_v1_global_local_increment_farm_sync" in rendered
    assert "world_model_state_space_v1_wake_off_farm_sync" in rendered
    assert "world_model_state_space_v1_graph_off_farm_sync" in rendered
    assert "world_model_state_space_v1_no_farm_aux_farm_sync" in rendered
    assert "world_model_state_space_v1_no_met_aux_farm_sync" in rendered
    assert "world_model_v1" in rendered
    assert "power_wd_yaw_lrpm_hist_sincos" in rendered
    assert "power_ws_wd_hist_sincos" in rendered
