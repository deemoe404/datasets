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


def test_shared_covariate_manifest_matches_chronos_shim() -> None:
    shared = _load_module(("experiment", "common", "covariate_packs.py"), "shared_covariate_packs")
    shim = _load_module(
        ("experiment", "chronos-2-exogenous", "chronos2_exogenous_manifest.py"),
        "chronos2_exogenous_manifest",
    )

    for dataset_id in ("kelmarsh", "penmanshiel", "hill_of_towie", "sdwpf_kddcup"):
        for stage in shared.DEFAULT_COVARIATE_STAGES:
            assert asdict(shared.resolve_covariate_pack(dataset_id, stage)) == asdict(
                shim.resolve_covariate_pack(dataset_id, stage)
            )
        assert asdict(shared.reference_pack_for(dataset_id)) == asdict(shim.reference_pack_for(dataset_id))


def test_reference_pack_and_stage_feature_sets_are_expected() -> None:
    shared = _load_module(("experiment", "common", "covariate_packs.py"), "shared_covariate_packs_expected")

    kelmarsh_reference = shared.reference_pack_for("kelmarsh")
    hill_stage3 = shared.resolve_covariate_pack("hill_of_towie", "stage3_regime")
    sdwpf_stage2 = shared.resolve_covariate_pack("sdwpf_kddcup", "stage2_ops")

    assert kelmarsh_reference.stage == "reference"
    assert kelmarsh_reference.pack_name == "power_only"
    assert kelmarsh_reference.feature_set == "lightweight"
    assert hill_stage3.feature_set == "default"
    assert "tuneup_post_effective" in hill_stage3.required_columns
    assert sdwpf_stage2.required_columns[-1] == "Prtv"
