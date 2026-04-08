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
    )

    assert output_path == tmp_path / "chunks" / "chunk_first_try.csv"
    assert len(seen_commands) == 1
    assert _command_option(seen_commands[0], "--device") == "cuda"
    assert _command_option(seen_commands[0], "--series-budget") == "1024"
    first_log = _read_log(tmp_path / "logs" / "chunk_first_try.json")
    assert first_log["returncode"] == 0
    assert _command_option(first_log["command"], "--series-budget") == "1024"
    assert not (tmp_path / "logs" / "chunk_first_try__retry_01.json").exists()


def test_execute_chunk_succeeds_on_second_retry(monkeypatch, tmp_path) -> None:
    module = _load_module()
    seen_commands = _install_fake_popen(monkeypatch, module, attempts=[{"returncode": 1}, {"returncode": 0}])

    output_path = module.execute_chunk(
        label="chunk_second_try",
        dataset_id="kelmarsh",
        work_dir=tmp_path,
        attempts=module.resolve_attempts(device="cuda", series_budget=4096),
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
