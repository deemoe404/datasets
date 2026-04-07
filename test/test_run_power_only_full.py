from __future__ import annotations

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import sys

import polars as pl


def _load_module():
    module_path = (
        Path(__file__).resolve().parents[1]
        / "experiment"
        / "chronos-2"
        / "run_power_only_full.py"
    )
    spec = spec_from_file_location("run_power_only_full", module_path)
    assert spec is not None
    assert spec.loader is not None
    module = module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_merge_chunk_results_combines_chunk_metrics_correctly(tmp_path) -> None:
    module = _load_module()
    chunk_a = tmp_path / "chunk_a.csv"
    chunk_b = tmp_path / "chunk_b.csv"

    pl.DataFrame(
        [
            {
                "dataset_id": "hill_of_towie_multivariate_knn6",
                "model_id": module.MODEL_ID,
                "task_id": module.TASK_ID,
                "history_steps": 144,
                "forecast_steps": 36,
                "stride_steps": 36,
                "target_policy": module.TARGET_POLICY,
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
                "dataset_id": "hill_of_towie_multivariate_knn6",
                "model_id": module.MODEL_ID,
                "task_id": module.TASK_ID,
                "history_steps": 144,
                "forecast_steps": 36,
                "stride_steps": 36,
                "target_policy": module.TARGET_POLICY,
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

    assert row["dataset_id"] == "hill_of_towie_multivariate_knn6"
    assert row["window_count"] == 22
    assert row["prediction_count"] == 150
    assert row["start_timestamp"] == "2024-01-01 00:10:00"
    assert row["end_timestamp"] == "2024-01-04 00:00:00"
    assert row["mae_kw"] == (2.0 * 100 + 4.0 * 50) / 150
    assert row["device"] == "cpu,mps"
    assert row["runtime_seconds"] == 3.5


def test_validate_final_results_requires_exact_dataset_ids() -> None:
    module = _load_module()
    frame = pl.DataFrame(
        {
            "dataset_id": ["kelmarsh_univariate"],
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
        }
    )

    try:
        module.validate_final_results(frame)
    except RuntimeError as exc:
        assert "Expected 8 result rows" in str(exc)
    else:
        raise AssertionError("validate_final_results should reject incomplete result sets.")


def test_run_target_group_chunks_splits_after_failure(monkeypatch, tmp_path) -> None:
    module = _load_module()
    seen_groups: list[tuple[str, ...]] = []

    def _fake_execute_chunk(**kwargs):
        turbine_ids = tuple(kwargs["turbine_ids"])
        seen_groups.append(turbine_ids)
        if len(turbine_ids) > 1:
            raise RuntimeError("chunk too large")
        output_path = tmp_path / f"{kwargs['label']}.csv"
        pl.DataFrame(
            [
                {
                    "dataset_id": "hill_of_towie_multivariate_knn6",
                    "model_id": module.MODEL_ID,
                    "task_id": module.TASK_ID,
                    "history_steps": 144,
                    "forecast_steps": 36,
                    "stride_steps": 36,
                    "target_policy": module.TARGET_POLICY,
                    "window_count": 1,
                    "prediction_count": 1,
                    "start_timestamp": "2024-01-01 00:10:00",
                    "end_timestamp": "2024-01-01 00:10:00",
                    "mae_kw": 1.0,
                    "rmse_kw": 1.0,
                    "mae_pu": 0.1,
                    "rmse_pu": 0.1,
                    "device": "cpu",
                    "runtime_seconds": 0.1,
                }
            ]
        ).write_csv(output_path)
        return output_path

    monkeypatch.setattr(module, "execute_chunk", _fake_execute_chunk)

    chunk_paths = module.run_target_group_chunks(
        dataset_id="hill_of_towie",
        mode="multivariate_knn6",
        work_dir=tmp_path,
        primary=module.Attempt(device="mps", batch_size=4),
        fallback=module.Attempt(device="cpu", batch_size=1),
        target_groups=[("T01", "T02", "T03", "T04")],
    )

    assert seen_groups[0] == ("T01", "T02", "T03", "T04")
    assert len(chunk_paths) == 4
