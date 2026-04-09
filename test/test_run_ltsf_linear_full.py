from __future__ import annotations

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import sys

import polars as pl


def _load_module():
    module_path = (
        Path(__file__).resolve().parents[1]
        / "experiment"
        / "ltsf-linear"
        / "run_ltsf_linear_full.py"
    )
    spec = spec_from_file_location("run_ltsf_linear_full", module_path)
    assert spec is not None
    assert spec.loader is not None
    module = module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _result_row(module, *, dataset_id: str, model_variant: str) -> dict[str, object]:
    return {
        "dataset_id": dataset_id,
        "model_id": module.MODEL_ID,
        "model_variant": model_variant,
        "task_id": module.TASK_ID,
        "history_steps": 144,
        "forecast_steps": 36,
        "stride_steps": 36,
        "split_protocol": module.SPLIT_PROTOCOL,
        "window_count": 10,
        "prediction_count": 360,
        "start_timestamp": "2024-01-01 00:00:00",
        "end_timestamp": "2024-01-02 00:00:00",
        "mae_kw": 1.0,
        "rmse_kw": 1.1,
        "mae_pu": 0.1,
        "rmse_pu": 0.11,
        "device": "cpu",
        "runtime_seconds": 1.0,
        "train_window_count": 70,
        "val_window_count": 10,
        "test_window_count": 20,
        "best_epoch": 2,
        "epochs_ran": 3,
        "best_val_rmse_pu": 0.09,
        "seed": 42,
        "batch_size": 1024,
        "learning_rate": 1e-3,
    }


def test_expected_job_keys_cover_full_eight_row_grid() -> None:
    module = _load_module()

    assert module.expected_job_keys() == [
        ("kelmarsh", "nlinear"),
        ("kelmarsh", "dlinear"),
        ("penmanshiel", "nlinear"),
        ("penmanshiel", "dlinear"),
        ("hill_of_towie", "nlinear"),
        ("hill_of_towie", "dlinear"),
        ("sdwpf_kddcup", "nlinear"),
        ("sdwpf_kddcup", "dlinear"),
    ]


def test_build_cli_command_includes_optional_smoke_args() -> None:
    module = _load_module()

    command = module.build_cli_command(
        dataset_id="kelmarsh",
        model_variant="nlinear",
        output_path=Path("/tmp/out.csv"),
        attempt=module.Attempt(device="cpu"),
        epochs=3,
        max_windows_per_split=64,
    )

    assert "--dataset" in command
    assert "--model" in command
    assert "--epochs" in command
    assert "--max-windows-per-split" in command


def test_merge_chunk_results_orders_rows_by_dataset_and_model(tmp_path) -> None:
    module = _load_module()
    chunk_a = tmp_path / "chunk_a.csv"
    chunk_b = tmp_path / "chunk_b.csv"
    pl.DataFrame(
        [
            _result_row(module, dataset_id="sdwpf_kddcup", model_variant="dlinear"),
            _result_row(module, dataset_id="kelmarsh", model_variant="dlinear"),
        ]
    ).write_csv(chunk_a)
    pl.DataFrame(
        [
            _result_row(module, dataset_id="kelmarsh", model_variant="nlinear"),
            _result_row(module, dataset_id="sdwpf_kddcup", model_variant="nlinear"),
        ]
    ).write_csv(chunk_b)

    merged = module.merge_chunk_results([chunk_a, chunk_b])

    assert list(zip(merged["dataset_id"].to_list(), merged["model_variant"].to_list(), strict=True)) == [
        ("kelmarsh", "nlinear"),
        ("kelmarsh", "dlinear"),
        ("sdwpf_kddcup", "nlinear"),
        ("sdwpf_kddcup", "dlinear"),
    ]


def test_validate_final_results_requires_complete_grid() -> None:
    module = _load_module()
    frame = pl.DataFrame([_result_row(module, dataset_id="kelmarsh", model_variant="nlinear")])

    try:
        module.validate_final_results(frame)
    except RuntimeError as exc:
        assert "Expected 8 result rows" in str(exc)
    else:
        raise AssertionError("validate_final_results should reject incomplete results.")


def test_resolve_attempts_adds_cpu_fallback_for_non_cpu(monkeypatch) -> None:
    module = _load_module()
    monkeypatch.setattr(module, "resolve_device", lambda device=None: "cuda")

    attempts = module.resolve_attempts(device=None)

    assert attempts == (module.Attempt(device="cuda"), module.Attempt(device="cpu"))


def test_execute_job_retries_on_cpu_after_primary_failure(monkeypatch, tmp_path) -> None:
    module = _load_module()
    job = module.JobSpec(label="kelmarsh_nlinear", dataset_id="kelmarsh", model_variant="nlinear")
    seen_devices: list[str] = []

    class _Result:
        def __init__(self, returncode: int) -> None:
            self.returncode = returncode
            self.stdout = ""
            self.stderr = ""

    def _fake_run(command, cwd, capture_output, text):
        del cwd, capture_output, text
        seen_devices.append(command[command.index("--device") + 1])
        output_path = Path(command[command.index("--output-path") + 1])
        if len(seen_devices) == 1:
            return _Result(returncode=1)
        pl.DataFrame([_result_row(module, dataset_id="kelmarsh", model_variant="nlinear")]).write_csv(output_path)
        return _Result(returncode=0)

    monkeypatch.setattr(module.subprocess, "run", _fake_run)

    output_path = module.execute_job(
        job=job,
        work_dir=tmp_path,
        attempts=(module.Attempt(device="cuda"), module.Attempt(device="cpu")),
    )

    assert seen_devices == ["cuda", "cpu"]
    assert output_path.exists()


def test_run_full_experiment_merges_all_jobs(monkeypatch, tmp_path) -> None:
    module = _load_module()

    def _fake_execute_job(*, job, work_dir, attempts, epochs=None, max_windows_per_split=None):
        del work_dir, attempts, epochs, max_windows_per_split
        output_path = tmp_path / f"{job.label}.csv"
        pl.DataFrame([_result_row(module, dataset_id=job.dataset_id, model_variant=job.model_variant)]).write_csv(
            output_path
        )
        return output_path

    monkeypatch.setattr(module, "execute_job", _fake_execute_job)

    output_path = tmp_path / "final.csv"
    results = module.run_full_experiment(output_path=output_path, work_dir=tmp_path / "work")

    assert results.height == 8
    assert output_path.exists()
