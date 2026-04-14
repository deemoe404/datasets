from __future__ import annotations

from wind_datasets.source_column_policy import load_source_column_policy


def test_source_column_policies_do_not_reference_removed_feature_protocols() -> None:
    allowed_protocol_tags = {
        "all",
        "power_only",
        "power_ws_hist",
        "power_atemp_hist",
        "power_itemp_hist",
        "power_wd_hist_sincos",
        "power_wd_yaw_hist_sincos",
        "power_wd_yaw_pitchmean_hist_sincos",
        "power_wd_yaw_pmean_hist_sincos_masked",
        "power_wd_yaw_lrpm_hist_sincos",
        "power_ws_wd_hist_sincos",
        "world_model_v1",
    }
    removed_protocol_tags = {
        "power_stats_history",
        "staged_past_covariates.stage1_core",
        "staged_past_covariates.stage2_ops",
        "staged_past_covariates.stage3_regime",
        "static_calendar",
        "static_calendar_stage1",
        "static_calendar_stage2",
    }

    for dataset_id in ("kelmarsh", "penmanshiel", "hill_of_towie", "sdwpf_kddcup"):
        policy = load_source_column_policy(dataset_id)
        for entry in policy.entries:
            assert set(entry.required_for_protocols).isdisjoint(removed_protocol_tags)
            assert set(entry.required_for_protocols).issubset(allowed_protocol_tags)
            assert set(entry.canonical_outputs).isdisjoint(removed_protocol_tags)
