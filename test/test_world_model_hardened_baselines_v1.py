from __future__ import annotations

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import sys

import polars as pl


def _load_module():
    module_path = (
        Path(__file__).resolve().parents[1]
        / "experiment"
        / "families"
        / "world_model_hardened_baselines_v1"
        / "world_model_hardened_baselines_v1.py"
    )
    spec = spec_from_file_location("world_model_hardened_baselines_v1", module_path)
    assert spec is not None
    assert spec.loader is not None
    module = module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_default_variants_are_initial_hardened_track() -> None:
    module = _load_module()

    assert module.DEFAULT_VARIANTS == (
        module.PERSISTENCE_HARDENED_VARIANT,
        module.DGCRN_OFFICIAL_CORE_VARIANT,
        module.TIMEXER_OFFICIAL_VARIANT,
        module.ITRANSFORMER_OFFICIAL_VARIANT,
        module.CHRONOS_OFFICIAL_VARIANT,
    )
    assert module.RESULT_PROVENANCE_COLUMNS == (
        "implementation_track",
        "source_repo",
        "source_commit",
        "adapter_kind",
        "search_config_id",
    )


def test_hardened_variants_resolve_to_repo_baseline_backends() -> None:
    module = _load_module()
    specs = {spec.model_variant: spec for spec in module.resolve_variant_specs(None)}

    assert specs[module.PERSISTENCE_HARDENED_VARIANT].backend_variant == module.base.PERSISTENCE_VARIANT
    assert specs[module.DGCRN_OFFICIAL_CORE_VARIANT].backend_variant == module.base.DGCRN_VARIANT
    assert specs[module.TIMEXER_OFFICIAL_VARIANT].backend_variant == module.base.TIMEXER_VARIANT
    assert specs[module.ITRANSFORMER_OFFICIAL_VARIANT].backend_variant == module.base.ITRANSFORMER_VARIANT
    assert specs[module.CHRONOS_OFFICIAL_VARIANT].backend_variant == module.base.CHRONOS_VARIANT
    assert specs[module.DGCRN_OFFICIAL_CORE_VARIANT].source_repo.endswith("Traffic-Benchmark.git")
    assert specs[module.TIMEXER_OFFICIAL_VARIANT].source_repo.endswith("TimeXer.git")
    assert specs[module.ITRANSFORMER_OFFICIAL_VARIANT].source_repo.endswith("iTransformer.git")
    assert specs[module.CHRONOS_OFFICIAL_VARIANT].source_repo == "https://huggingface.co/amazon/chronos-2"


def test_result_annotation_rewrites_variant_and_adds_provenance() -> None:
    module = _load_module()
    frame = pl.DataFrame(
        {
            "dataset_id": ["kelmarsh", "kelmarsh"],
            "model_id": [module.base.MODEL_ID, module.base.MODEL_ID],
            "model_variant": [module.base.DGCRN_VARIANT, module.base.CHRONOS_VARIANT],
            "baseline_type": ["dgcrn_dynamic_graph", "chronos_2_zero_shot"],
            "rmse_pu": [0.1, 0.2],
        }
    )

    annotated = module.annotate_hardened_results(
        frame,
        (
            module.DGCRN_OFFICIAL_CORE_VARIANT,
            module.CHRONOS_OFFICIAL_VARIANT,
        ),
    )

    assert annotated["model_variant"].to_list() == [
        module.DGCRN_OFFICIAL_CORE_VARIANT,
        module.CHRONOS_OFFICIAL_VARIANT,
    ]
    assert annotated["model_id"].to_list() == [module.MODEL_ID, module.MODEL_ID]
    assert annotated["baseline_type"].to_list() == ["dgcrn_official_core", "chronos_2_zero_shot_official"]
    assert annotated["implementation_track"].to_list() == ["hardened_official", "hardened_official"]
    assert annotated["adapter_kind"].to_list() == ["official_core_port", "official_package_adapter"]
    assert annotated["search_config_id"].to_list() == ["phase1_default", "zero_shot_default"]
    assert annotated["source_repo"].to_list()[0].endswith("Traffic-Benchmark.git")
    assert annotated["source_repo"].to_list()[1] == "https://huggingface.co/amazon/chronos-2"
    assert annotated["source_commit"].null_count() == 0


def test_wrapper_contract_files_exist_for_initial_external_sources() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    official_root = repo_root / "experiment" / "official_baselines"

    for wrapper_name in ("dgcrn", "timexer", "itransformer", "chronos_2", "tft_pf"):
        wrapper_root = official_root / wrapper_name
        assert (wrapper_root / "README.md").is_file()
        assert (wrapper_root / "create_env.sh").is_file()
        assert (wrapper_root / "environment.yml").is_file()

    for wrapper_name in ("dgcrn", "timexer", "itransformer"):
        assert (official_root / wrapper_name / "source").exists()


def test_main_runs_backend_and_writes_hardened_output(tmp_path, monkeypatch) -> None:
    module = _load_module()
    output_path = tmp_path / "hardened.csv"
    recorded: dict[str, object] = {}

    def _fake_run_experiment(**kwargs):
        recorded["backend_variants"] = kwargs["variant_names"]
        recorded["work_root"] = kwargs["work_root"]
        return pl.DataFrame(
            {
                "dataset_id": ["kelmarsh"],
                "model_id": [module.base.MODEL_ID],
                "model_variant": [module.base.DGCRN_VARIANT],
                "baseline_type": ["dgcrn_dynamic_graph"],
                "selection_metric": [module.SELECTION_METRIC_RMSE],
                "rmse_pu": [0.123],
            }
        )

    monkeypatch.setattr(module.base, "run_experiment", _fake_run_experiment)

    assert (
        module.main(
            [
                "--variant",
                module.DGCRN_OFFICIAL_CORE_VARIANT,
                "--selection-metric",
                module.SELECTION_METRIC_RMSE,
                "--output-path",
                str(output_path),
                "--disable-tensorboard",
                "--no-record-run",
            ]
        )
        == 0
    )

    written = pl.read_csv(output_path)
    assert recorded["backend_variants"] == (module.base.DGCRN_VARIANT,)
    assert Path(recorded["work_root"]).name == "run_world_model_hardened_baselines_v1"
    assert written["model_id"].to_list() == [module.MODEL_ID]
    assert written["model_variant"].to_list() == [module.DGCRN_OFFICIAL_CORE_VARIANT]
    assert written["baseline_type"].to_list() == ["dgcrn_official_core"]
    assert written["source_commit"].to_list() == ["b9f8e40b4df9b58f5ad88432dc070cbbbcdc0228"]
