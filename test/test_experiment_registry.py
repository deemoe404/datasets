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
    assert agcrn.supported_feature_protocols == ("power_only",)
    assert agcrn.implementation_bindings == {
        "official_aligned_power_only_farm_sync": "power_only",
    }


def test_registry_dataset_family_feature_matrix_matches_active_scope() -> None:
    module = _load_module()

    snapshot = module.load_registry_snapshot()
    rows = module.build_dataset_family_feature_matrix(snapshot)

    assert len(rows) == 1
    row = rows[0]
    assert row.dataset_id == "kelmarsh"
    assert row.family_id == "agcrn_official_aligned"
    assert row.feature_protocol_id == "power_only"
    assert row.family_status == "pilot"


def test_registry_markdown_renderer_mentions_active_family() -> None:
    module = _load_module()

    snapshot = module.load_registry_snapshot()
    rows = module.build_dataset_family_feature_matrix(snapshot)
    rendered = module.render_matrix_markdown(rows)

    assert "agcrn_official_aligned" in rendered
    assert "official_aligned_power_only_farm_sync" in rendered
    assert "power_only" in rendered
