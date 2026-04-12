from __future__ import annotations

from importlib.util import module_from_spec, spec_from_file_location
import json
from pathlib import Path
import sys


def _load_module():
    module_path = (
        Path(__file__).resolve().parents[1]
        / "experiment"
        / "infra"
        / "common"
        / "run_records.py"
    )
    common_dir = str(module_path.parent)
    if common_dir not in sys.path:
        sys.path.insert(0, common_dir)
    spec = spec_from_file_location("run_records", module_path)
    assert spec is not None
    assert spec.loader is not None
    module = module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_resolve_family_feature_protocol_ids_maps_active_labels_to_registry_ids() -> None:
    module = _load_module()
    repo_root = Path(__file__).resolve().parents[1]

    assert module.resolve_family_feature_protocol_ids(
        "agcrn_official_aligned",
        (
            "official_aligned_power_only_farm_sync",
            "official_aligned_power_ws_hist_farm_sync",
            "official_aligned_power_wd_hist_sincos_farm_sync",
            "official_aligned_power_ws_wd_hist_sincos_farm_sync",
        ),
        repo_root=repo_root,
    ) == (
        "power_only",
        "power_ws_hist",
        "power_wd_hist_sincos",
        "power_ws_wd_hist_sincos",
    )


def test_record_cli_run_writes_manifest_with_output_checksum(tmp_path) -> None:
    module = _load_module()
    repo_root = Path(__file__).resolve().parents[1]
    runs_root = tmp_path / "runs"
    output_path = tmp_path / "results.csv"
    output_path.write_text("dataset_id,metric\nkelmarsh,1.0\n", encoding="utf-8")

    manifest_path = module.record_cli_run(
        family_id="agcrn_official_aligned",
        repo_root=repo_root,
        runs_root=runs_root,
        invocation_kind="family_runner",
        entrypoint="experiment/families/agcrn/run_agcrn.py",
        args={"dataset": ["kelmarsh"]},
        output_path=output_path,
        result_row_count=1,
        dataset_ids=("kelmarsh",),
        feature_protocol_ids=(
            "power_only",
            "power_ws_hist",
            "power_wd_hist_sincos",
            "power_ws_wd_hist_sincos",
        ),
        model_variants=(
            "official_aligned_power_only_farm_sync",
            "official_aligned_power_ws_hist_farm_sync",
            "official_aligned_power_wd_hist_sincos_farm_sync",
            "official_aligned_power_ws_wd_hist_sincos_farm_sync",
        ),
        eval_protocols=("rolling_origin_no_refit", "non_overlap"),
        result_splits=("val", "test"),
        artifacts={"cache_root": repo_root / "cache"},
        run_label="kelmarsh-both-variants",
    )

    assert manifest_path.exists()
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert payload["family"]["family_id"] == "agcrn_official_aligned"
    assert payload["selection"]["dataset_ids"] == ["kelmarsh"]
    assert payload["selection"]["feature_protocol_ids"] == [
        "power_only",
        "power_ws_hist",
        "power_wd_hist_sincos",
        "power_ws_wd_hist_sincos",
    ]
    assert payload["selection"]["model_variants"] == [
        "official_aligned_power_only_farm_sync",
        "official_aligned_power_ws_hist_farm_sync",
        "official_aligned_power_wd_hist_sincos_farm_sync",
        "official_aligned_power_ws_wd_hist_sincos_farm_sync",
    ]
    assert payload["result"]["row_count"] == 1
    assert payload["artifacts"]["primary_output"]["exists"] is True
    assert payload["artifacts"]["primary_output"]["sha256"]
    assert payload["invocation"]["run_label"] == "kelmarsh-both-variants"


def test_record_cli_run_rejects_protocols_not_supported_by_family(tmp_path) -> None:
    module = _load_module()
    repo_root = Path(__file__).resolve().parents[1]
    output_path = tmp_path / "results.csv"
    output_path.write_text("x\n", encoding="utf-8")

    try:
        module.record_cli_run(
            family_id="agcrn_official_aligned",
            repo_root=repo_root,
            runs_root=tmp_path / "runs",
            invocation_kind="family_runner",
            entrypoint="experiment/families/agcrn/run_agcrn.py",
            args={},
            output_path=output_path,
            result_row_count=1,
            dataset_ids=("kelmarsh",),
            feature_protocol_ids=("unexpected_protocol",),
        )
    except ValueError as exc:
        assert "unsupported feature protocols" in str(exc)
    else:
        raise AssertionError("Expected record_cli_run to reject unsupported feature protocols.")
