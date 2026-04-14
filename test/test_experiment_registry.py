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


def test_registry_snapshot_contains_only_active_family() -> None:
    module = _load_module()

    snapshot = module.load_registry_snapshot()

    assert set(snapshot.families) == {"agcrn_official_aligned"}


def test_registry_family_bindings_capture_current_active_contract() -> None:
    module = _load_module()

    snapshot = module.load_registry_snapshot()
    agcrn = snapshot.families["agcrn_official_aligned"]

    assert agcrn.status == "pilot"
    assert agcrn.task_contract.granularity == "farm"
    assert agcrn.dataset_scope == ("kelmarsh", "penmanshiel")
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


def test_registry_dataset_family_feature_matrix_matches_active_scope() -> None:
    module = _load_module()

    snapshot = module.load_registry_snapshot()
    rows = module.build_dataset_family_feature_matrix(snapshot)

    assert len(rows) == 18
    assert [row.dataset_id for row in rows] == [
        "kelmarsh",
        "kelmarsh",
        "kelmarsh",
        "kelmarsh",
        "kelmarsh",
        "kelmarsh",
        "kelmarsh",
        "kelmarsh",
        "kelmarsh",
        "penmanshiel",
        "penmanshiel",
        "penmanshiel",
        "penmanshiel",
        "penmanshiel",
        "penmanshiel",
        "penmanshiel",
        "penmanshiel",
        "penmanshiel",
    ]
    assert [row.family_id for row in rows] == [
        "agcrn_official_aligned",
        "agcrn_official_aligned",
        "agcrn_official_aligned",
        "agcrn_official_aligned",
        "agcrn_official_aligned",
        "agcrn_official_aligned",
        "agcrn_official_aligned",
        "agcrn_official_aligned",
        "agcrn_official_aligned",
        "agcrn_official_aligned",
        "agcrn_official_aligned",
        "agcrn_official_aligned",
        "agcrn_official_aligned",
        "agcrn_official_aligned",
        "agcrn_official_aligned",
        "agcrn_official_aligned",
        "agcrn_official_aligned",
        "agcrn_official_aligned",
    ]
    assert [row.feature_protocol_id for row in rows] == [
        "power_only",
        "power_ws_hist",
        "power_atemp_hist",
        "power_itemp_hist",
        "power_wd_hist_sincos",
        "power_wd_yaw_hist_sincos",
        "power_wd_yaw_pitchmean_hist_sincos",
        "power_wd_yaw_lrpm_hist_sincos",
        "power_ws_wd_hist_sincos",
        "power_only",
        "power_ws_hist",
        "power_atemp_hist",
        "power_itemp_hist",
        "power_wd_hist_sincos",
        "power_wd_yaw_hist_sincos",
        "power_wd_yaw_pitchmean_hist_sincos",
        "power_wd_yaw_lrpm_hist_sincos",
        "power_ws_wd_hist_sincos",
    ]
    assert all(row.family_status == "pilot" for row in rows)


def test_registry_markdown_renderer_mentions_active_family() -> None:
    module = _load_module()

    snapshot = module.load_registry_snapshot()
    rows = module.build_dataset_family_feature_matrix(snapshot)
    rendered = module.render_matrix_markdown(rows)

    assert "agcrn_official_aligned" in rendered
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
    assert "power_wd_yaw_lrpm_hist_sincos" in rendered
    assert "power_ws_wd_hist_sincos" in rendered
