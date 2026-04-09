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


class _TqdmRecorder:
    instances: list["_TqdmRecorder"] = []
    writes: list[str] = []

    def __init__(self, *args, **kwargs) -> None:
        del args
        self.total = kwargs.get("total")
        self.desc = kwargs.get("desc")
        self.disable = kwargs.get("disable")
        self.n = 0
        self.postfixes: list[str] = []
        self.closed = False
        _TqdmRecorder.instances.append(self)

    def update(self, value=1) -> None:
        self.n += int(value)

    def set_postfix_str(self, value: str) -> None:
        self.postfixes.append(value)

    def close(self) -> None:
        self.closed = True

    @staticmethod
    def write(message: str) -> None:
        _TqdmRecorder.writes.append(message)


def _result_row(
    module,
    *,
    dataset_id: str,
    covariate_stage: str,
    covariate_pack: str,
    model_variant: str,
    split_name: str,
    eval_protocol: str,
    metric_scope: str,
    lead_step: int | None,
) -> dict[str, object]:
    covariate_count = 0 if covariate_stage == "reference" else 3
    return {
        "dataset_id": dataset_id,
        "model_id": module.MODEL_ID,
        "model_variant": model_variant,
        "task_id": module.TASK_ID,
        "window_protocol": module.WINDOW_PROTOCOL,
        "history_steps": 144,
        "forecast_steps": 36,
        "stride_steps": 1,
        "split_protocol": module.SPLIT_PROTOCOL,
        "split_name": split_name,
        "eval_protocol": eval_protocol,
        "metric_scope": metric_scope,
        "lead_step": lead_step,
        "lead_minutes": None if lead_step is None else lead_step * 10,
        "covariate_stage": covariate_stage,
        "covariate_pack": covariate_pack,
        "feature_set": "lightweight" if dataset_id in {"kelmarsh", "penmanshiel"} else "default",
        "covariate_count": covariate_count,
        "covariate_policy": "none" if covariate_count == 0 else "past_only_train_zscore_fill0_mask",
        "window_count": 10,
        "prediction_count": 360 if metric_scope == module.OVERALL_METRIC_SCOPE else 10,
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


def _job_rows(module, job) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for split_name in ("val", "test"):
        for eval_protocol in (module.ROLLING_EVAL_PROTOCOL, module.NON_OVERLAP_EVAL_PROTOCOL):
            rows.append(
                _result_row(
                    module,
                    dataset_id=job.dataset_id,
                    covariate_stage=job.covariate_stage,
                    covariate_pack=job.covariate_pack,
                    model_variant=job.model_variant,
                    split_name=split_name,
                    eval_protocol=eval_protocol,
                    metric_scope=module.OVERALL_METRIC_SCOPE,
                    lead_step=None,
                )
            )
            for lead_step in range(1, module.FORECAST_STEPS + 1):
                rows.append(
                    _result_row(
                        module,
                        dataset_id=job.dataset_id,
                        covariate_stage=job.covariate_stage,
                        covariate_pack=job.covariate_pack,
                        model_variant=job.model_variant,
                        split_name=split_name,
                        eval_protocol=eval_protocol,
                        metric_scope=module.HORIZON_METRIC_SCOPE,
                        lead_step=lead_step,
                    )
                )
    return rows


def test_expected_job_keys_cover_full_thirty_two_row_grid() -> None:
    module = _load_module()

    expected = module.expected_job_keys()

    assert len(expected) == 32
    assert expected[:4] == [
        ("kelmarsh", "reference", "power_only", "nlinear"),
        ("kelmarsh", "reference", "power_only", "dlinear"),
        ("kelmarsh", "stage1_core", "stage1_core", "nlinear"),
        ("kelmarsh", "stage1_core", "stage1_core", "dlinear"),
    ]
    assert expected[-4:] == [
        ("sdwpf_kddcup", "stage2_ops", "stage2_ops", "nlinear"),
        ("sdwpf_kddcup", "stage2_ops", "stage2_ops", "dlinear"),
        ("sdwpf_kddcup", "stage3_regime", "stage3_regime", "nlinear"),
        ("sdwpf_kddcup", "stage3_regime", "stage3_regime", "dlinear"),
    ]


def test_expected_result_keys_cover_long_4736_row_grid() -> None:
    module = _load_module()

    expected = module.expected_result_keys()

    assert len(expected) == 4736
    assert expected[:3] == [
        ("kelmarsh", "reference", "power_only", "nlinear", "val", module.ROLLING_EVAL_PROTOCOL, module.OVERALL_METRIC_SCOPE, None),
        ("kelmarsh", "reference", "power_only", "nlinear", "val", module.ROLLING_EVAL_PROTOCOL, module.HORIZON_METRIC_SCOPE, 1),
        ("kelmarsh", "reference", "power_only", "nlinear", "val", module.ROLLING_EVAL_PROTOCOL, module.HORIZON_METRIC_SCOPE, 2),
    ]
    assert expected[-3:] == [
        ("sdwpf_kddcup", "stage3_regime", "stage3_regime", "dlinear", "test", module.NON_OVERLAP_EVAL_PROTOCOL, module.HORIZON_METRIC_SCOPE, 34),
        ("sdwpf_kddcup", "stage3_regime", "stage3_regime", "dlinear", "test", module.NON_OVERLAP_EVAL_PROTOCOL, module.HORIZON_METRIC_SCOPE, 35),
        ("sdwpf_kddcup", "stage3_regime", "stage3_regime", "dlinear", "test", module.NON_OVERLAP_EVAL_PROTOCOL, module.HORIZON_METRIC_SCOPE, 36),
    ]


def test_build_cli_command_uses_reference_only_for_reference_jobs() -> None:
    module = _load_module()

    command = module.build_cli_command(
        dataset_id="kelmarsh",
        covariate_stage="reference",
        model_variant="nlinear",
        output_path=Path("/tmp/out.csv"),
        attempt=module.Attempt(device="cpu"),
        epochs=3,
        max_windows_per_split=64,
    )

    assert "--reference-only" in command
    assert "--covariate-stage" not in command
    assert "--max-windows-per-split" in command


def test_build_cli_command_uses_single_stage_without_reference() -> None:
    module = _load_module()

    command = module.build_cli_command(
        dataset_id="kelmarsh",
        covariate_stage="stage2_ops",
        model_variant="dlinear",
        output_path=Path("/tmp/out.csv"),
        attempt=module.Attempt(device="cpu"),
    )

    assert "--covariate-stage" in command
    assert "stage2_ops" in command
    assert "--no-power-only-reference" in command
    assert "--reference-only" not in command


def test_merge_chunk_results_orders_rows_by_long_key(tmp_path) -> None:
    module = _load_module()
    chunk_a = tmp_path / "chunk_a.csv"
    chunk_b = tmp_path / "chunk_b.csv"
    pl.DataFrame(
        [
            _result_row(
                module,
                dataset_id="sdwpf_kddcup",
                covariate_stage="stage1_core",
                covariate_pack="stage1_core",
                model_variant="dlinear",
                split_name="test",
                eval_protocol=module.NON_OVERLAP_EVAL_PROTOCOL,
                metric_scope=module.HORIZON_METRIC_SCOPE,
                lead_step=2,
            ),
            _result_row(
                module,
                dataset_id="kelmarsh",
                covariate_stage="reference",
                covariate_pack="power_only",
                model_variant="nlinear",
                split_name="val",
                eval_protocol=module.ROLLING_EVAL_PROTOCOL,
                metric_scope=module.HORIZON_METRIC_SCOPE,
                lead_step=1,
            ),
        ]
    ).write_csv(chunk_a)
    pl.DataFrame(
        [
            _result_row(
                module,
                dataset_id="kelmarsh",
                covariate_stage="reference",
                covariate_pack="power_only",
                model_variant="nlinear",
                split_name="val",
                eval_protocol=module.ROLLING_EVAL_PROTOCOL,
                metric_scope=module.OVERALL_METRIC_SCOPE,
                lead_step=None,
            ),
        ]
    ).write_csv(chunk_b)

    merged = module.merge_chunk_results([chunk_a, chunk_b])

    assert list(
        zip(
            merged["dataset_id"].to_list(),
            merged["split_name"].to_list(),
            merged["eval_protocol"].to_list(),
            merged["metric_scope"].to_list(),
            merged["lead_step"].to_list(),
            strict=True,
        )
    ) == [
        ("kelmarsh", "val", module.ROLLING_EVAL_PROTOCOL, module.OVERALL_METRIC_SCOPE, None),
        ("kelmarsh", "val", module.ROLLING_EVAL_PROTOCOL, module.HORIZON_METRIC_SCOPE, 1),
        ("sdwpf_kddcup", "test", module.NON_OVERLAP_EVAL_PROTOCOL, module.HORIZON_METRIC_SCOPE, 2),
    ]


def test_validate_final_results_requires_complete_grid() -> None:
    module = _load_module()
    frame = pl.DataFrame(
        [
            _result_row(
                module,
                dataset_id="kelmarsh",
                covariate_stage="reference",
                covariate_pack="power_only",
                model_variant="nlinear",
                split_name="val",
                eval_protocol=module.ROLLING_EVAL_PROTOCOL,
                metric_scope=module.OVERALL_METRIC_SCOPE,
                lead_step=None,
            )
        ]
    )

    try:
        module.validate_final_results(frame)
    except RuntimeError as exc:
        assert "Expected 4736 result rows" in str(exc)
    else:
        raise AssertionError("validate_final_results should reject incomplete results.")


def test_resolve_attempts_adds_cpu_fallback_for_non_cpu(monkeypatch) -> None:
    module = _load_module()
    monkeypatch.setattr(module, "resolve_device", lambda device=None: "cuda")

    attempts = module.resolve_attempts(device=None)

    assert attempts == (module.Attempt(device="cuda"), module.Attempt(device="cpu"))


def test_execute_job_retries_on_cpu_after_primary_failure(monkeypatch, tmp_path) -> None:
    module = _load_module()
    job = module.JobSpec(
        label="kelmarsh_reference_power_only_nlinear",
        dataset_id="kelmarsh",
        covariate_stage="reference",
        covariate_pack="power_only",
        model_variant="nlinear",
    )
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
        pl.DataFrame(_job_rows(module, job)).write_csv(output_path)
        return _Result(returncode=0)

    monkeypatch.setattr(module.subprocess, "run", _fake_run)

    output_path = module.execute_job(
        job=job,
        work_dir=tmp_path,
        attempts=(module.Attempt(device="cuda"), module.Attempt(device="cpu")),
    )

    assert seen_devices == ["cuda", "cpu"]
    assert output_path.exists()


def test_execute_job_overwrites_existing_chunk_by_default(monkeypatch, tmp_path) -> None:
    module = _load_module()
    job = module.JobSpec(
        label="kelmarsh_reference_power_only_nlinear",
        dataset_id="kelmarsh",
        covariate_stage="reference",
        covariate_pack="power_only",
        model_variant="nlinear",
    )
    output_dir = tmp_path / "chunks"
    output_dir.mkdir(parents=True, exist_ok=True)
    stale_output = output_dir / f"{job.label}.csv"
    stale_output.write_text("old\nchunk\n", encoding="utf-8")
    seen_commands: list[list[str]] = []

    class _Result:
        def __init__(self) -> None:
            self.returncode = 0
            self.stdout = ""
            self.stderr = ""

    def _fake_run(command, cwd, capture_output, text):
        del cwd, capture_output, text
        seen_commands.append(command)
        output_path = Path(command[command.index("--output-path") + 1])
        pl.DataFrame(_job_rows(module, job)).write_csv(output_path)
        return _Result()

    monkeypatch.setattr(module.subprocess, "run", _fake_run)

    output_path = module.execute_job(
        job=job,
        work_dir=tmp_path,
        attempts=(module.Attempt(device="cpu"),),
    )

    assert len(seen_commands) == 1
    assert output_path == stale_output
    frame = pl.read_csv(output_path)
    assert "window_protocol" in frame.columns


def test_execute_job_reuses_only_when_flag_is_enabled_and_chunk_is_compatible(monkeypatch, tmp_path) -> None:
    module = _load_module()
    job = module.JobSpec(
        label="kelmarsh_reference_power_only_nlinear",
        dataset_id="kelmarsh",
        covariate_stage="reference",
        covariate_pack="power_only",
        model_variant="nlinear",
    )
    output_dir = tmp_path / "chunks"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{job.label}.csv"
    pl.DataFrame(_job_rows(module, job)).write_csv(output_path)

    def _unexpected_run(*args, **kwargs):
        raise AssertionError("compatible chunk should have been reused")

    monkeypatch.setattr(module.subprocess, "run", _unexpected_run)

    resolved = module.execute_job(
        job=job,
        work_dir=tmp_path,
        attempts=(module.Attempt(device="cpu"),),
        reuse_existing_chunks=True,
    )

    assert resolved == output_path


def test_execute_job_rebuilds_stale_chunk_even_when_reuse_is_enabled(monkeypatch, tmp_path) -> None:
    module = _load_module()
    job = module.JobSpec(
        label="kelmarsh_reference_power_only_nlinear",
        dataset_id="kelmarsh",
        covariate_stage="reference",
        covariate_pack="power_only",
        model_variant="nlinear",
    )
    output_dir = tmp_path / "chunks"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{job.label}.csv"
    pl.DataFrame(
        [
            {
                "dataset_id": "kelmarsh",
                "model_id": module.MODEL_ID,
            }
        ]
    ).write_csv(output_path)
    seen_commands: list[list[str]] = []

    class _Result:
        def __init__(self) -> None:
            self.returncode = 0
            self.stdout = ""
            self.stderr = ""

    def _fake_run(command, cwd, capture_output, text):
        del cwd, capture_output, text
        seen_commands.append(command)
        refreshed_output = Path(command[command.index("--output-path") + 1])
        pl.DataFrame(_job_rows(module, job)).write_csv(refreshed_output)
        return _Result()

    monkeypatch.setattr(module.subprocess, "run", _fake_run)

    resolved = module.execute_job(
        job=job,
        work_dir=tmp_path,
        attempts=(module.Attempt(device="cpu"),),
        reuse_existing_chunks=True,
    )

    assert resolved == output_path
    assert len(seen_commands) == 1
    frame = pl.read_csv(output_path)
    assert "window_protocol" in frame.columns


def test_run_full_experiment_merges_all_jobs(monkeypatch, tmp_path) -> None:
    module = _load_module()

    def _fake_execute_job(*, job, work_dir, attempts, epochs=None, max_windows_per_split=None, reuse_existing_chunks=False):
        del work_dir, attempts, epochs, max_windows_per_split, reuse_existing_chunks
        output_path = tmp_path / f"{job.label}.csv"
        pl.DataFrame(_job_rows(module, job)).write_csv(output_path)
        return output_path

    monkeypatch.setattr(module, "execute_job", _fake_execute_job)

    output_path = tmp_path / "final.csv"
    results = module.run_full_experiment(output_path=output_path, work_dir=tmp_path / "work")

    assert results.height == 4736
    assert output_path.exists()


def test_run_full_experiment_updates_job_progress_bar(monkeypatch, tmp_path) -> None:
    module = _load_module()
    _TqdmRecorder.instances.clear()
    job_specs = (
        module.JobSpec(
            label="kelmarsh_reference_power_only_nlinear",
            dataset_id="kelmarsh",
            covariate_stage="reference",
            covariate_pack="power_only",
            model_variant="nlinear",
        ),
        module.JobSpec(
            label="kelmarsh_reference_power_only_dlinear",
            dataset_id="kelmarsh",
            covariate_stage="reference",
            covariate_pack="power_only",
            model_variant="dlinear",
        ),
    )

    def _fake_execute_job(*, job, work_dir, attempts, epochs=None, max_windows_per_split=None, reuse_existing_chunks=False):
        del work_dir, attempts, epochs, max_windows_per_split, reuse_existing_chunks
        output_path = tmp_path / f"{job.label}.csv"
        pl.DataFrame(_job_rows(module, job)).write_csv(output_path)
        return output_path

    monkeypatch.setattr(module, "tqdm", _TqdmRecorder)
    monkeypatch.setattr(module, "progress_is_enabled", lambda: True)
    monkeypatch.setattr(module, "build_job_specs", lambda: job_specs)
    monkeypatch.setattr(module, "execute_job", _fake_execute_job)

    module.run_full_experiment(output_path=tmp_path / "final.csv", work_dir=tmp_path / "work")

    assert len(_TqdmRecorder.instances) == 1
    progress = _TqdmRecorder.instances[0]
    assert progress.total == 2
    assert progress.n == 2
    assert progress.closed is True
    assert progress.desc == "ltsf-linear full jobs"
    assert progress.postfixes == [job.label for job in job_specs]
