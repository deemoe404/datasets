from __future__ import annotations

import io
from importlib.util import module_from_spec, spec_from_file_location
import json
from pathlib import Path
import sys

import polars as pl


def _load_module():
    module_path = (
        Path(__file__).resolve().parents[1]
        / "experiment"
        / "families"
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


def _command_option(command: list[str], option: str) -> str:
    option_index = command.index(option)
    return str(command[option_index + 1])


class _FakeProcess:
    def __init__(self, *, returncode: int, stdout: str, stderr: str) -> None:
        self._returncode = returncode
        self.stdout = io.StringIO(stdout)
        self.stderr = io.StringIO(stderr)

    def wait(self) -> int:
        return self._returncode


class _TqdmRecorder:
    instances: list["_TqdmRecorder"] = []
    writes: list[str] = []

    def __init__(self, *args, **kwargs) -> None:
        del args
        self.total = kwargs.get("total")
        self.desc = kwargs.get("desc")
        self.unit = kwargs.get("unit")
        self.position = kwargs.get("position")
        self.leave = kwargs.get("leave")
        self.disable = kwargs.get("disable")
        self.n = 0
        self.postfixes: list[str] = []
        self.closed = False
        _TqdmRecorder.instances.append(self)

    def update(self, value=1) -> None:
        self.n += int(value)

    def set_postfix_str(self, value: str) -> None:
        self.postfixes.append(value)

    def refresh(self) -> None:
        return None

    def close(self) -> None:
        self.closed = True

    @staticmethod
    def write(message: str) -> None:
        _TqdmRecorder.writes.append(message)


def _progress_line(module, phase: str, **fields: object) -> str:
    payload = json.dumps({"dataset_id": "kelmarsh", "phase": phase, **fields}, sort_keys=True)
    return f"{module.PROFILE_LOG_PREFIX}{payload}\n"


def _install_fake_popen(monkeypatch, module, attempts: list[dict[str, object]]) -> list[list[str]]:
    seen_commands: list[list[str]] = []

    def _fake_popen(command, cwd, stdout, stderr, text, bufsize):
        del cwd, stdout, stderr, text, bufsize
        normalized_command = [str(part) for part in command]
        seen_commands.append(normalized_command)
        attempt_index = len(seen_commands) - 1
        payload = attempts[attempt_index]
        returncode = int(payload["returncode"])
        if returncode == 0:
            output_path = Path(_command_option(normalized_command, "--output-path"))
            output_path.parent.mkdir(parents=True, exist_ok=True)
            pl.DataFrame({"attempt_index": [attempt_index]}).write_csv(output_path)
        return _FakeProcess(
            returncode=returncode,
            stdout=str(payload.get("stdout", f"stdout-{attempt_index}")),
            stderr=str(payload.get("stderr", f"stderr-{attempt_index}\n")),
        )

    monkeypatch.setattr(module.subprocess, "Popen", _fake_popen)
    return seen_commands


def _read_log(log_path: Path) -> dict[str, object]:
    return json.loads(log_path.read_text())

def _covariate_count_for(dataset_id: str) -> int:
    return 13 if dataset_id in {"kelmarsh", "penmanshiel"} else 30


def _result_row(
    module,
    *,
    dataset_id: str,
    covariate_stage: str,
    covariate_pack: str,
    eval_protocol: str,
    metric_scope: str,
    lead_step: int | None,
    window_protocol: str | None = None,
    split_protocol: str | None = None,
    window_count: int = 10,
    prediction_count: int | None = None,
    start_timestamp: str = "2024-01-01 00:00:00",
    end_timestamp: str = "2024-01-02 00:00:00",
    mae_kw: float = 1.0,
    rmse_kw: float = 1.1,
    mae_pu: float = 0.1,
    rmse_pu: float = 0.11,
    device: str = "cpu",
    runtime_seconds: float = 1.0,
) -> dict[str, object]:
    resolved_prediction_count = prediction_count
    if resolved_prediction_count is None:
        resolved_prediction_count = 360 if metric_scope == module.OVERALL_METRIC_SCOPE else 10
    return {
        "dataset_id": dataset_id,
        "model_id": module.MODEL_ID,
        "task_id": module.TASK_ID,
        "window_protocol": window_protocol or module.DEFAULT_WINDOW_PROTOCOL,
        "history_steps": 144,
        "forecast_steps": 36,
        "stride_steps": 1,
        "split_protocol": split_protocol or module.SPLIT_PROTOCOL,
        "split_name": "test",
        "eval_protocol": eval_protocol,
        "metric_scope": metric_scope,
        "lead_step": lead_step,
        "lead_minutes": None if lead_step is None else lead_step * 10,
        "target_policy": module.TARGET_POLICY,
        "window_count": window_count,
        "prediction_count": resolved_prediction_count,
        "start_timestamp": start_timestamp,
        "end_timestamp": end_timestamp,
        "mae_kw": mae_kw,
        "rmse_kw": rmse_kw,
        "mae_pu": mae_pu,
        "rmse_pu": rmse_pu,
        "device": device,
        "runtime_seconds": runtime_seconds,
        "layout": module.LAYOUT,
        "covariate_stage": covariate_stage,
        "covariate_pack": covariate_pack,
        "covariate_count": _covariate_count_for(dataset_id),
        "covariate_policy": module.COVARIATE_POLICY,
        "train_window_count": 70,
        "val_window_count": 10,
        "test_window_count": 20,
    }


def _rows_for_expected_keys(
    module,
    *,
    include_power_only_reference: bool = False,
    window_protocol: str | None = None,
    split_protocol: str | None = None,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for dataset_id, covariate_stage, covariate_pack, split_name, eval_protocol, metric_scope, lead_step in (
        module.expected_result_keys(include_power_only_reference=include_power_only_reference)
    ):
        assert split_name == "test"
        rows.append(
            _result_row(
                module,
                dataset_id=dataset_id,
                covariate_stage=covariate_stage,
                covariate_pack=covariate_pack,
                eval_protocol=eval_protocol,
                metric_scope=metric_scope,
                lead_step=lead_step,
                window_protocol=window_protocol,
                split_protocol=split_protocol,
            )
        )
    return rows


def test_expected_result_keys_cover_test_only_long_888_row_grid() -> None:
    module = _load_module()

    expected = module.expected_result_keys()

    assert len(expected) == 888
    assert expected[:3] == [
        (
            "kelmarsh",
            "stage1_core",
            "stage1_core",
            "test",
            module.ROLLING_EVAL_PROTOCOL,
            module.OVERALL_METRIC_SCOPE,
            None,
        ),
        (
            "kelmarsh",
            "stage1_core",
            "stage1_core",
            "test",
            module.ROLLING_EVAL_PROTOCOL,
            module.HORIZON_METRIC_SCOPE,
            1,
        ),
        (
            "kelmarsh",
            "stage1_core",
            "stage1_core",
            "test",
            module.ROLLING_EVAL_PROTOCOL,
            module.HORIZON_METRIC_SCOPE,
            2,
        ),
    ]
    assert expected[-3:] == [
        (
            "sdwpf_kddcup",
            "stage3_regime",
            "stage3_regime",
            "test",
            module.NON_OVERLAP_EVAL_PROTOCOL,
            module.HORIZON_METRIC_SCOPE,
            34,
        ),
        (
            "sdwpf_kddcup",
            "stage3_regime",
            "stage3_regime",
            "test",
            module.NON_OVERLAP_EVAL_PROTOCOL,
            module.HORIZON_METRIC_SCOPE,
            35,
        ),
        (
            "sdwpf_kddcup",
            "stage3_regime",
            "stage3_regime",
            "test",
            module.NON_OVERLAP_EVAL_PROTOCOL,
            module.HORIZON_METRIC_SCOPE,
            36,
        ),
    ]


def test_expected_result_keys_include_optional_power_only_reference() -> None:
    module = _load_module()

    expected = module.expected_result_keys(dataset_ids=("kelmarsh",), include_power_only_reference=True)

    assert len(expected) == 296
    assert expected[:3] == [
        (
            "kelmarsh",
            "reference",
            "power_only",
            "test",
            module.ROLLING_EVAL_PROTOCOL,
            module.OVERALL_METRIC_SCOPE,
            None,
        ),
        (
            "kelmarsh",
            "reference",
            "power_only",
            "test",
            module.ROLLING_EVAL_PROTOCOL,
            module.HORIZON_METRIC_SCOPE,
            1,
        ),
        (
            "kelmarsh",
            "reference",
            "power_only",
            "test",
            module.ROLLING_EVAL_PROTOCOL,
            module.HORIZON_METRIC_SCOPE,
            2,
        ),
    ]


def test_merge_chunk_results_combines_partial_rows_by_metadata_key(tmp_path) -> None:
    module = _load_module()
    chunk_a = tmp_path / "chunk_a.csv"
    chunk_b = tmp_path / "chunk_b.csv"
    shared = {
        "dataset_id": "hill_of_towie",
        "covariate_stage": "stage3_regime",
        "covariate_pack": "stage3_regime",
        "eval_protocol": module.NON_OVERLAP_EVAL_PROTOCOL,
        "metric_scope": module.OVERALL_METRIC_SCOPE,
        "lead_step": None,
    }
    pl.DataFrame(
        [
            _result_row(
                module,
                **shared,
                window_count=10,
                prediction_count=100,
                start_timestamp="2024-01-01 00:10:00",
                end_timestamp="2024-01-02 00:00:00",
                mae_kw=2.0,
                rmse_kw=3.0,
                mae_pu=0.2,
                rmse_pu=0.3,
                device="mps",
                runtime_seconds=1.5,
            )
        ]
    ).write_csv(chunk_a)
    pl.DataFrame(
        [
            _result_row(
                module,
                **shared,
                window_count=12,
                prediction_count=50,
                start_timestamp="2024-01-03 00:10:00",
                end_timestamp="2024-01-04 00:00:00",
                mae_kw=4.0,
                rmse_kw=5.0,
                mae_pu=0.4,
                rmse_pu=0.5,
                device="cpu",
                runtime_seconds=2.0,
            )
        ]
    ).write_csv(chunk_b)

    merged = module.merge_chunk_results([chunk_a, chunk_b])
    row = merged.to_dicts()[0]

    assert row["dataset_id"] == "hill_of_towie"
    assert row["covariate_stage"] == "stage3_regime"
    assert row["covariate_pack"] == "stage3_regime"
    assert row["split_name"] == "test"
    assert row["eval_protocol"] == module.NON_OVERLAP_EVAL_PROTOCOL
    assert row["metric_scope"] == module.OVERALL_METRIC_SCOPE
    assert row["window_count"] == 22
    assert row["prediction_count"] == 150
    assert row["window_protocol"] == module.DEFAULT_WINDOW_PROTOCOL
    assert row["split_protocol"] == module.SPLIT_PROTOCOL
    assert row["start_timestamp"] == "2024-01-01 00:10:00"
    assert row["end_timestamp"] == "2024-01-04 00:00:00"
    assert row["mae_kw"] == (2.0 * 100 + 4.0 * 50) / 150
    assert row["device"] == "cpu,mps"
    assert row["runtime_seconds"] == 3.5
    assert row["train_window_count"] == 70
    assert row["val_window_count"] == 10
    assert row["test_window_count"] == 20


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
                eval_protocol=module.NON_OVERLAP_EVAL_PROTOCOL,
                metric_scope=module.HORIZON_METRIC_SCOPE,
                lead_step=2,
            ),
            _result_row(
                module,
                dataset_id="kelmarsh",
                covariate_stage="stage1_core",
                covariate_pack="stage1_core",
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
                covariate_stage="stage1_core",
                covariate_pack="stage1_core",
                eval_protocol=module.ROLLING_EVAL_PROTOCOL,
                metric_scope=module.OVERALL_METRIC_SCOPE,
                lead_step=None,
            )
        ]
    ).write_csv(chunk_b)

    merged = module.merge_chunk_results([chunk_a, chunk_b])

    assert list(
        zip(
            merged["dataset_id"].to_list(),
            merged["covariate_stage"].to_list(),
            merged["covariate_pack"].to_list(),
            merged["split_name"].to_list(),
            merged["eval_protocol"].to_list(),
            merged["metric_scope"].to_list(),
            merged["lead_step"].to_list(),
            strict=True,
        )
    ) == [
        (
            "kelmarsh",
            "stage1_core",
            "stage1_core",
            "test",
            module.ROLLING_EVAL_PROTOCOL,
            module.OVERALL_METRIC_SCOPE,
            None,
        ),
        (
            "kelmarsh",
            "stage1_core",
            "stage1_core",
            "test",
            module.ROLLING_EVAL_PROTOCOL,
            module.HORIZON_METRIC_SCOPE,
            1,
        ),
        (
            "sdwpf_kddcup",
            "stage1_core",
            "stage1_core",
            "test",
            module.NON_OVERLAP_EVAL_PROTOCOL,
            module.HORIZON_METRIC_SCOPE,
            2,
        ),
    ]


def test_validate_final_results_requires_complete_dataset_stage_grid() -> None:
    module = _load_module()
    frame = pl.DataFrame(
        [
            _result_row(
                module,
                dataset_id="kelmarsh",
                covariate_stage="stage1_core",
                covariate_pack="stage1_core",
                eval_protocol=module.ROLLING_EVAL_PROTOCOL,
                metric_scope=module.OVERALL_METRIC_SCOPE,
                lead_step=None,
            )
        ]
    )

    try:
        module.validate_final_results(frame)
    except RuntimeError as exc:
        assert "Expected 888 result rows" in str(exc)
    else:
        raise AssertionError("validate_final_results should reject incomplete result sets.")


def test_validate_final_results_rejects_unexpected_window_protocol() -> None:
    module = _load_module()
    frame = pl.DataFrame(_rows_for_expected_keys(module, window_protocol="legacy_6h_stride"))

    try:
        module.validate_final_results(frame)
    except RuntimeError as exc:
        assert "unexpected window_protocol" in str(exc)
    else:
        raise AssertionError("validate_final_results should reject mismatched window protocols.")


def test_build_full_chunk_specs_matches_expected_chunk_plan() -> None:
    module = _load_module()

    chunk_specs = module.build_full_chunk_specs()
    labels = [chunk_spec.label for chunk_spec in chunk_specs]

    assert len(chunk_specs) == 18
    assert labels[:3] == [
        "kelmarsh_all_stages",
        "penmanshiel_all_stages",
        "hill_of_towie_chunk_01",
    ]
    assert labels[-3:] == [
        "sdwpf_kddcup_chunk_07",
        "sdwpf_kddcup_chunk_08",
        "sdwpf_kddcup_chunk_09",
    ]


def test_build_progress_labels_matches_test_only_eval_slice_plan() -> None:
    module = _load_module()

    labels = module.build_progress_labels()

    assert len(labels) == 36
    assert labels[:3] == [
        "kelmarsh_all_stages_rolling_origin_no_refit",
        "penmanshiel_all_stages_rolling_origin_no_refit",
        "hill_of_towie_chunk_01_rolling_origin_no_refit",
    ]
    assert labels[-3:] == [
        "sdwpf_kddcup_chunk_07_non_overlap",
        "sdwpf_kddcup_chunk_08_non_overlap",
        "sdwpf_kddcup_chunk_09_non_overlap",
    ]


def test_resolve_attempts_auto_uses_cuda_fixed_retry_chain(monkeypatch) -> None:
    module = _load_module()
    monkeypatch.setattr(module, "select_device", lambda: "cuda")

    auto_attempts = module.resolve_attempts(device=None, series_budget=4096)
    explicit_attempts = module.resolve_attempts(device="cuda", series_budget=256)

    expected = (
        module.Attempt(device="cuda", series_budget=1024),
        module.Attempt(device="cuda", series_budget=768),
        module.Attempt(device="cuda", series_budget=512),
        module.Attempt(device="cpu", series_budget=1024),
    )
    assert auto_attempts == expected
    assert explicit_attempts == expected


def test_resolve_attempts_auto_uses_mps_fixed_retry_chain(monkeypatch) -> None:
    module = _load_module()
    monkeypatch.setattr(module, "select_device", lambda: "mps")

    auto_attempts = module.resolve_attempts(device=None, series_budget=2048)
    explicit_attempts = module.resolve_attempts(device="mps", series_budget=256)

    expected = (
        module.Attempt(device="mps", series_budget=1024),
        module.Attempt(device="mps", series_budget=768),
        module.Attempt(device="mps", series_budget=512),
        module.Attempt(device="cpu", series_budget=1024),
    )
    assert auto_attempts == expected
    assert explicit_attempts == expected


def test_resolve_attempts_cpu_uses_requested_series_budget(monkeypatch) -> None:
    module = _load_module()
    monkeypatch.setattr(module, "select_device", lambda: "cpu")

    auto_attempts = module.resolve_attempts(device=None, series_budget=1024)
    explicit_attempts = module.resolve_attempts(device="cpu", series_budget=2048)

    assert auto_attempts == (module.Attempt(device="cpu", series_budget=1024),)
    assert explicit_attempts == (module.Attempt(device="cpu", series_budget=2048),)


def test_execute_chunk_succeeds_on_first_attempt(monkeypatch, tmp_path) -> None:
    module = _load_module()
    seen_commands = _install_fake_popen(monkeypatch, module, attempts=[{"returncode": 0}])

    output_path = module.execute_chunk(
        label="chunk_first_try",
        dataset_id="kelmarsh",
        work_dir=tmp_path,
        attempts=module.resolve_attempts(device="cuda", series_budget=4096),
        eval_protocol=module.ROLLING_EVAL_PROTOCOL,
    )

    assert output_path == tmp_path / "chunks" / "chunk_first_try.csv"
    assert len(seen_commands) == 1
    assert _command_option(seen_commands[0], "--device") == "cuda"
    assert _command_option(seen_commands[0], "--series-budget") == "1024"
    first_log = _read_log(tmp_path / "logs" / "chunk_first_try.json")
    assert first_log["returncode"] == 0
    assert first_log["eval_protocol"] == module.ROLLING_EVAL_PROTOCOL
    assert _command_option(first_log["command"], "--series-budget") == "1024"
    assert not (tmp_path / "logs" / "chunk_first_try__retry_01.json").exists()


def test_build_cli_command_uses_eval_protocol_stage_filters_and_reference_flag() -> None:
    module = _load_module()

    command = module.build_cli_command(
        dataset_id="kelmarsh",
        output_path=Path("/tmp/chronos-2-exogenous.csv"),
        attempt=module.Attempt(device="cpu", series_budget=1024),
        eval_protocol=module.NON_OVERLAP_EVAL_PROTOCOL,
        covariate_stages=("stage2_ops",),
        include_power_only_reference=True,
        max_windows_per_split=64,
    )

    assert "--window-protocol" not in command
    assert _command_option(command, "--eval-protocol") == module.NON_OVERLAP_EVAL_PROTOCOL
    assert command.count("--covariate-stage") == 1
    assert "stage2_ops" in command
    assert "--include-power-only-reference" in command
    assert _command_option(command, "--max-windows-per-split") == "64"


def test_execute_chunk_succeeds_on_second_retry(monkeypatch, tmp_path) -> None:
    module = _load_module()
    seen_commands = _install_fake_popen(monkeypatch, module, attempts=[{"returncode": 1}, {"returncode": 0}])

    output_path = module.execute_chunk(
        label="chunk_second_try",
        dataset_id="kelmarsh",
        work_dir=tmp_path,
        attempts=module.resolve_attempts(device="cuda", series_budget=4096),
        eval_protocol=module.ROLLING_EVAL_PROTOCOL,
    )

    assert output_path == tmp_path / "chunks" / "chunk_second_try.csv"
    assert [(_command_option(command, "--device"), _command_option(command, "--series-budget")) for command in seen_commands] == [
        ("cuda", "1024"),
        ("cuda", "768"),
    ]
    first_log = _read_log(tmp_path / "logs" / "chunk_second_try.json")
    retry_log = _read_log(tmp_path / "logs" / "chunk_second_try__retry_01.json")
    assert first_log["returncode"] == 1
    assert retry_log["returncode"] == 0
    assert _command_option(retry_log["command"], "--series-budget") == "768"


def test_execute_chunk_succeeds_on_third_retry(monkeypatch, tmp_path) -> None:
    module = _load_module()
    seen_commands = _install_fake_popen(monkeypatch, module, attempts=[{"returncode": 1}, {"returncode": 1}, {"returncode": 0}])

    output_path = module.execute_chunk(
        label="chunk_third_try",
        dataset_id="hill_of_towie",
        work_dir=tmp_path,
        attempts=module.resolve_attempts(device="cuda", series_budget=1234),
        eval_protocol=module.NON_OVERLAP_EVAL_PROTOCOL,
    )

    assert output_path == tmp_path / "chunks" / "chunk_third_try.csv"
    assert [(_command_option(command, "--device"), _command_option(command, "--series-budget")) for command in seen_commands] == [
        ("cuda", "1024"),
        ("cuda", "768"),
        ("cuda", "512"),
    ]
    retry_log = _read_log(tmp_path / "logs" / "chunk_third_try__retry_02.json")
    assert retry_log["returncode"] == 0
    assert _command_option(retry_log["command"], "--series-budget") == "512"


def test_execute_chunk_succeeds_on_cpu_fallback(monkeypatch, tmp_path) -> None:
    module = _load_module()
    seen_commands = _install_fake_popen(
        monkeypatch,
        module,
        attempts=[{"returncode": 1}, {"returncode": 1}, {"returncode": 1}, {"returncode": 0}],
    )

    output_path = module.execute_chunk(
        label="chunk_cpu_fallback",
        dataset_id="hill_of_towie",
        work_dir=tmp_path,
        attempts=module.resolve_attempts(device="cuda", series_budget=2048),
        eval_protocol=module.ROLLING_EVAL_PROTOCOL,
    )

    assert output_path == tmp_path / "chunks" / "chunk_cpu_fallback.csv"
    assert [(_command_option(command, "--device"), _command_option(command, "--series-budget")) for command in seen_commands] == [
        ("cuda", "1024"),
        ("cuda", "768"),
        ("cuda", "512"),
        ("cpu", "1024"),
    ]
    fallback_log = _read_log(tmp_path / "logs" / "chunk_cpu_fallback__fallback_cpu.json")
    assert fallback_log["returncode"] == 0
    assert _command_option(fallback_log["command"], "--device") == "cpu"
    assert _command_option(fallback_log["command"], "--series-budget") == "1024"


def test_execute_chunk_streams_progress_events_and_filters_logs(monkeypatch, tmp_path) -> None:
    module = _load_module()
    _TqdmRecorder.instances = []
    _TqdmRecorder.writes = []
    monkeypatch.setattr(module, "tqdm", _TqdmRecorder)
    seen_commands = _install_fake_popen(
        monkeypatch,
        module,
        attempts=[
            {
                "returncode": 0,
                "stderr": "".join(
                    [
                        _progress_line(module, "progress_chunk_plan", chunk_total_batches=2),
                        _progress_line(
                            module,
                            "progress_stage_start",
                            chunk_total_batches=2,
                            stage="stage1_core",
                        ),
                        _progress_line(
                            module,
                            "progress_batch",
                            chunk_total_batches=2,
                            completed_chunk_batches=1,
                            stage="stage1_core",
                            turbine_id="T01",
                        ),
                        _progress_line(
                            module,
                            "progress_batch",
                            chunk_total_batches=2,
                            completed_chunk_batches=2,
                            stage="stage1_core",
                            turbine_id="T02",
                        ),
                        _progress_line(
                            module,
                            "progress_stage_complete",
                            chunk_total_batches=2,
                            completed_chunk_batches=2,
                            stage="stage1_core",
                        ),
                        "stderr-0\n",
                    ]
                ),
            }
        ],
    )

    output_path = module.execute_chunk(
        label="chunk_progress",
        dataset_id="kelmarsh",
        work_dir=tmp_path,
        attempts=module.resolve_attempts(device="cuda", series_budget=4096),
        eval_protocol=module.ROLLING_EVAL_PROTOCOL,
        progress_enabled=True,
    )

    assert output_path == tmp_path / "chunks" / "chunk_progress.csv"
    assert "--emit-progress-events" in seen_commands[0]
    progress_bars = [instance for instance in _TqdmRecorder.instances if instance.position == 1]
    assert len(progress_bars) == 1
    assert progress_bars[0].total == 2
    assert progress_bars[0].n == 2
    assert progress_bars[0].closed is True
    assert any("cuda/1024" in postfix for postfix in progress_bars[0].postfixes)
    assert any("turbine=T02" in postfix for postfix in progress_bars[0].postfixes)
    log = _read_log(tmp_path / "logs" / "chunk_progress.json")
    assert log["stderr"] == "stderr-0\n"
    assert log["eval_protocol"] == module.ROLLING_EVAL_PROTOCOL


def test_execute_chunk_resets_inner_progress_bar_across_retries(monkeypatch, tmp_path) -> None:
    module = _load_module()
    _TqdmRecorder.instances = []
    _TqdmRecorder.writes = []
    monkeypatch.setattr(module, "tqdm", _TqdmRecorder)
    seen_commands = _install_fake_popen(
        monkeypatch,
        module,
        attempts=[
            {
                "returncode": 1,
                "stderr": "".join(
                    [
                        _progress_line(module, "progress_chunk_plan", chunk_total_batches=1),
                        _progress_line(
                            module,
                            "progress_batch",
                            chunk_total_batches=1,
                            completed_chunk_batches=1,
                            stage="stage1_core",
                            turbine_id="T01",
                        ),
                    ]
                ),
            },
            {
                "returncode": 1,
                "stderr": "".join(
                    [
                        _progress_line(module, "progress_chunk_plan", chunk_total_batches=1),
                        _progress_line(
                            module,
                            "progress_batch",
                            chunk_total_batches=1,
                            completed_chunk_batches=1,
                            stage="stage1_core",
                            turbine_id="T01",
                        ),
                    ]
                ),
            },
            {
                "returncode": 1,
                "stderr": "".join(
                    [
                        _progress_line(module, "progress_chunk_plan", chunk_total_batches=1),
                        _progress_line(
                            module,
                            "progress_batch",
                            chunk_total_batches=1,
                            completed_chunk_batches=1,
                            stage="stage1_core",
                            turbine_id="T01",
                        ),
                    ]
                ),
            },
            {
                "returncode": 0,
                "stderr": "".join(
                    [
                        _progress_line(module, "progress_chunk_plan", chunk_total_batches=1),
                        _progress_line(
                            module,
                            "progress_batch",
                            chunk_total_batches=1,
                            completed_chunk_batches=1,
                            stage="stage1_core",
                            turbine_id="T01",
                        ),
                    ]
                ),
            },
        ],
    )

    module.execute_chunk(
        label="chunk_progress_retry",
        dataset_id="hill_of_towie",
        work_dir=tmp_path,
        attempts=module.resolve_attempts(device="cuda", series_budget=4096),
        eval_protocol=module.NON_OVERLAP_EVAL_PROTOCOL,
        progress_enabled=True,
    )

    assert [(_command_option(command, "--device"), _command_option(command, "--series-budget")) for command in seen_commands] == [
        ("cuda", "1024"),
        ("cuda", "768"),
        ("cuda", "512"),
        ("cpu", "1024"),
    ]
    progress_bars = [instance for instance in _TqdmRecorder.instances if instance.position == 1]
    assert len(progress_bars) == 4
    assert all(progress_bar.total == 1 for progress_bar in progress_bars)
    assert all(progress_bar.n == 1 for progress_bar in progress_bars)
    assert all(progress_bar.closed is True for progress_bar in progress_bars)


def test_execute_chunk_raises_after_all_attempts_fail(monkeypatch, tmp_path) -> None:
    module = _load_module()
    _install_fake_popen(
        monkeypatch,
        module,
        attempts=[{"returncode": 1}, {"returncode": 1}, {"returncode": 1}, {"returncode": 1}],
    )

    try:
        module.execute_chunk(
            label="chunk_all_fail",
            dataset_id="sdwpf_kddcup",
            work_dir=tmp_path,
            attempts=module.resolve_attempts(device="cuda", series_budget=1024),
            eval_protocol=module.NON_OVERLAP_EVAL_PROTOCOL,
        )
    except RuntimeError as exc:
        message = str(exc)
        assert "chunk_all_fail failed on all attempts" in message
        assert "chunk_all_fail.json" in message
        assert "chunk_all_fail__retry_01.json" in message
        assert "chunk_all_fail__retry_02.json" in message
        assert "chunk_all_fail__fallback_cpu.json" in message
    else:
        raise AssertionError("execute_chunk should raise after exhausting all attempts.")

    assert (tmp_path / "logs" / "chunk_all_fail.json").exists()
    assert (tmp_path / "logs" / "chunk_all_fail__retry_01.json").exists()
    assert (tmp_path / "logs" / "chunk_all_fail__retry_02.json").exists()
    assert (tmp_path / "logs" / "chunk_all_fail__fallback_cpu.json").exists()


def test_run_window_chunks_stops_after_empty_chunk(monkeypatch, tmp_path) -> None:
    module = _load_module()
    seen_offsets: list[int] = []

    def _fake_execute_chunk(**kwargs):
        window_offset = int(kwargs["window_offset"])
        seen_offsets.append(window_offset)
        output_path = tmp_path / f"{kwargs['label']}.csv"
        window_count = 3 if window_offset == 0 else 0
        prediction_count = 3 if window_offset == 0 else 0
        pl.DataFrame(
            [
                _result_row(
                    module,
                    dataset_id="kelmarsh",
                    covariate_stage="stage1_core",
                    covariate_pack="stage1_core",
                    eval_protocol=kwargs["eval_protocol"],
                    metric_scope=module.OVERALL_METRIC_SCOPE,
                    lead_step=None,
                    window_count=window_count,
                    prediction_count=prediction_count,
                )
            ]
        ).write_csv(output_path)
        return output_path

    monkeypatch.setattr(module, "execute_chunk", _fake_execute_chunk)

    chunk_paths = module.run_window_chunks(
        label_prefix="kelmarsh_all_stages_rolling_origin_no_refit",
        dataset_id="kelmarsh",
        work_dir=tmp_path,
        attempts=(module.Attempt(device="cpu", series_budget=1024),),
        eval_protocol=module.ROLLING_EVAL_PROTOCOL,
        windows_per_chunk=7,
    )

    assert seen_offsets == [0, 7]
    assert chunk_paths == [tmp_path / "kelmarsh_all_stages_rolling_origin_no_refit_offset_00000.csv"]
