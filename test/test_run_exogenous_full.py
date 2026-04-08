from __future__ import annotations

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import sys

import polars as pl


def _load_module():
    module_path = (
        Path(__file__).resolve().parents[1]
        / "experiment"
        / "chronos-2-exogenous"
        / "run_exogenous_full.py"
    )
    spec = spec_from_file_location("run_exogenous_full", module_path)
    assert spec is not None
    assert spec.loader is not None
    module = module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_merge_chunk_results_combines_partial_rows_by_metadata_key(tmp_path) -> None:
    module = _load_module()
    chunk_a = tmp_path / "chunk_a.csv"
    chunk_b = tmp_path / "chunk_b.csv"
    shared = {
        "dataset_id": "hill_of_towie",
        "model_id": module.MODEL_ID,
        "task_id": module.TASK_ID,
        "history_steps": 144,
        "forecast_steps": 36,
        "stride_steps": 36,
        "target_policy": module.TARGET_POLICY,
        "layout": module.LAYOUT,
        "covariate_stage": "stage3_regime",
        "covariate_pack": "stage3_regime",
        "feature_set": "default",
        "covariate_count": 30,
        "covariate_policy": "dataset_custom_past_only",
    }
    pl.DataFrame(
        [
            {
                **shared,
                "window_count": 10,
                "prediction_count": 100,
                "start_timestamp": "2024-01-01 00:10:00",
                "end_timestamp": "2024-01-02 00:00:00",
                "mae_kw": 2.0,
                "rmse_kw": 3.0,
                "mae_pu": 0.2,
                "rmse_pu": 0.3,
                "device": "mps",
                "runtime_seconds": 1.5,
            }
        ]
    ).write_csv(chunk_a)
    pl.DataFrame(
        [
            {
                **shared,
                "window_count": 12,
                "prediction_count": 50,
                "start_timestamp": "2024-01-03 00:10:00",
                "end_timestamp": "2024-01-04 00:00:00",
                "mae_kw": 4.0,
                "rmse_kw": 5.0,
                "mae_pu": 0.4,
                "rmse_pu": 0.5,
                "device": "cpu",
                "runtime_seconds": 2.0,
            }
        ]
    ).write_csv(chunk_b)

    merged = module.merge_chunk_results([chunk_a, chunk_b])
    row = merged.to_dicts()[0]

    assert row["dataset_id"] == "hill_of_towie"
    assert row["covariate_stage"] == "stage3_regime"
    assert row["window_count"] == 22
    assert row["prediction_count"] == 150
    assert row["start_timestamp"] == "2024-01-01 00:10:00"
    assert row["end_timestamp"] == "2024-01-04 00:00:00"
    assert row["mae_kw"] == (2.0 * 100 + 4.0 * 50) / 150
    assert row["device"] == "cpu,mps"
    assert row["runtime_seconds"] == 3.5


def test_validate_final_results_requires_complete_dataset_stage_grid() -> None:
    module = _load_module()
    frame = pl.DataFrame(
        {
            "dataset_id": ["kelmarsh"],
            "model_id": [module.MODEL_ID],
            "task_id": [module.TASK_ID],
            "history_steps": [144],
            "forecast_steps": [36],
            "stride_steps": [36],
            "target_policy": [module.TARGET_POLICY],
            "window_count": [1],
            "prediction_count": [1],
            "start_timestamp": ["2024-01-01 00:10:00"],
            "end_timestamp": ["2024-01-01 00:10:00"],
            "mae_kw": [1.0],
            "rmse_kw": [1.0],
            "mae_pu": [0.1],
            "rmse_pu": [0.1],
            "device": ["cpu"],
            "runtime_seconds": [0.1],
            "layout": [module.LAYOUT],
            "covariate_stage": ["stage1_core"],
            "covariate_pack": ["stage1_core"],
            "feature_set": ["lightweight"],
            "covariate_count": [13],
            "covariate_policy": ["dataset_custom_past_only"],
        }
    )

    try:
        module.validate_final_results(frame)
    except RuntimeError as exc:
        assert "Expected 12 result rows" in str(exc)
    else:
        raise AssertionError("validate_final_results should reject incomplete result sets.")


def test_build_full_chunk_specs_matches_expected_chunk_plan() -> None:
    module = _load_module()

    chunk_specs = module.build_full_chunk_specs()
    labels = [chunk_spec.label for chunk_spec in chunk_specs]

    assert len(chunk_specs) == 18
    assert labels[:2] == ["kelmarsh_all_stages", "penmanshiel_all_stages"]
    assert labels[2] == "hill_of_towie_chunk_01"
    assert labels[8] == "hill_of_towie_chunk_07"
    assert labels[9] == "sdwpf_kddcup_chunk_01"
    assert labels[-1] == "sdwpf_kddcup_chunk_09"
