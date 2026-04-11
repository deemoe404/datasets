from __future__ import annotations

from dataclasses import asdict
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import sys


def _load_module(path_parts: tuple[str, ...], name: str):
    module_path = Path(__file__).resolve().parents[1].joinpath(*path_parts)
    spec = spec_from_file_location(name, module_path)
    assert spec is not None
    assert spec.loader is not None
    module = module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_shared_covariate_manifest_matches_dataset_side_protocol_source() -> None:
    shared = _load_module(("experiment", "infra", "common", "covariate_packs.py"), "shared_covariate_packs")
    source = _load_module(("src", "wind_datasets", "feature_protocols.py"), "dataset_feature_protocols")

    for dataset_id in ("kelmarsh", "penmanshiel", "hill_of_towie", "sdwpf_kddcup"):
        for stage in shared.DEFAULT_COVARIATE_STAGES:
            assert asdict(shared.resolve_covariate_pack(dataset_id, stage)) == asdict(
                source.resolve_covariate_pack(dataset_id, stage)
            )
        assert asdict(shared.reference_pack_for(dataset_id)) == asdict(source.reference_pack_for(dataset_id))


def test_reference_pack_and_stage_columns_are_expected() -> None:
    shared = _load_module(("experiment", "infra", "common", "covariate_packs.py"), "shared_covariate_packs_expected")

    kelmarsh_reference = shared.reference_pack_for("kelmarsh")
    hill_stage3 = shared.resolve_covariate_pack("hill_of_towie", "stage3_regime")
    kelmarsh_stage1 = shared.resolve_covariate_pack("kelmarsh", "stage1_core")
    penmanshiel_stage2 = shared.resolve_covariate_pack("penmanshiel", "stage2_ops")
    sdwpf_stage2 = shared.resolve_covariate_pack("sdwpf_kddcup", "stage2_ops")

    assert kelmarsh_reference.stage == "reference"
    assert kelmarsh_reference.pack_name == "power_only"
    assert len(kelmarsh_stage1.required_columns) == 12
    assert "Grid frequency (Hz)" not in kelmarsh_stage1.required_columns
    assert len(penmanshiel_stage2.required_columns) == 15
    assert "farm_pmu__gms_power_setpoint_kw" not in penmanshiel_stage2.required_columns
    assert "farm_pmu__gms_voltage_v" not in penmanshiel_stage2.required_columns
    assert "farm_pmu__gms_grid_frequency_hz" not in penmanshiel_stage2.required_columns
    assert len(hill_stage3.required_columns) == 23
    assert "wtc_ActualWindDirection_mean" not in hill_stage3.required_columns
    assert "wtc_GridFreq_mean" not in hill_stage3.required_columns
    assert "farm_grid__frequency" not in hill_stage3.required_columns
    assert "tuneup_post_effective" not in hill_stage3.required_columns
    assert sdwpf_stage2.required_columns[-1] == "Prtv"
