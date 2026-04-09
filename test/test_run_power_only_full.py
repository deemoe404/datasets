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


def _command_option(command: list[str], option: str) -> str:
    option_index = command.index(option)
    return str(command[option_index + 1])


class _TqdmRecorder:
    instances: list["_TqdmRecorder"] = []
    writes: list[str] = []

    def __init__(self, *args, **kwargs) -> None:
        del args
        self.total = kwargs.get("total")
        self.desc = kwargs.get("desc")
        self.unit = kwargs.get("unit")
        self.position = kwargs.get("position")
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


class _FakeProcess:
    def __init__(self, *, returncode: int, stdout: str, stderr: str) -> None:
        self._returncode = returncode
        self.stdout = io.StringIO(stdout)
        self.stderr = io.StringIO(stderr)

    def wait(self) -> int:
        return self._returncode


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


def _result_row(
    module,
    *,
    dataset_id: str,
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
        "train_window_count": 70,
        "val_window_count": 10,
        "test_window_count": 20,
    }


def _rows_for_expected_keys(
    module,
    *,
    window_protocol: str | None = None,
    split_protocol: str | None = None,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for dataset_id, split_name, eval_protocol, metric_scope, lead_step in module.expected_result_keys():
        assert split_name == "test"
        rows.append(
            _result_row(
                module,
                dataset_id=dataset_id,
                eval_protocol=eval_protocol,
                metric_scope=metric_scope,
                lead_step=lead_step,
                window_protocol=window_protocol,
                split_protocol=split_protocol,
            )
        )
    return rows


def test_expected_dataset_ids_includes_power_stats_variants() -> None:
    module = _load_module()

    assert module.expected_dataset_ids() == [
        "hill_of_towie_multivariate_knn6",
        "hill_of_towie_multivariate_knn6_power_stats",
        "hill_of_towie_univariate",
        "hill_of_towie_univariate_power_stats",
        "kelmarsh_multivariate_knn6",
        "kelmarsh_multivariate_knn6_power_stats",
        "kelmarsh_univariate",
        "kelmarsh_univariate_power_stats",
        "penmanshiel_multivariate_knn6",
        "penmanshiel_multivariate_knn6_power_stats",
        "penmanshiel_univariate",
        "penmanshiel_univariate_power_stats",
        "sdwpf_kddcup_multivariate_knn6",
        "sdwpf_kddcup_univariate",
    ]


def test_expected_result_keys_cover_test_only_long_1036_row_grid() -> None:
    module = _load_module()

    expected = module.expected_result_keys()

    assert len(expected) == 1036
    assert expected[:3] == [
        (
            "hill_of_towie_multivariate_knn6",
            "test",
            module.ROLLING_EVAL_PROTOCOL,
            module.OVERALL_METRIC_SCOPE,
            None,
        ),
        (
            "hill_of_towie_multivariate_knn6",
            "test",
            module.ROLLING_EVAL_PROTOCOL,
            module.HORIZON_METRIC_SCOPE,
            1,
        ),
        (
            "hill_of_towie_multivariate_knn6",
            "test",
            module.ROLLING_EVAL_PROTOCOL,
            module.HORIZON_METRIC_SCOPE,
            2,
        ),
    ]
    assert expected[-3:] == [
        (
            "sdwpf_kddcup_univariate",
            "test",
            module.NON_OVERLAP_EVAL_PROTOCOL,
            module.HORIZON_METRIC_SCOPE,
            34,
        ),
        (
            "sdwpf_kddcup_univariate",
            "test",
            module.NON_OVERLAP_EVAL_PROTOCOL,
            module.HORIZON_METRIC_SCOPE,
            35,
        ),
        (
            "sdwpf_kddcup_univariate",
            "test",
            module.NON_OVERLAP_EVAL_PROTOCOL,
            module.HORIZON_METRIC_SCOPE,
            36,
        ),
    ]


def test_merge_chunk_results_combines_long_rows_by_metadata_key(tmp_path) -> None:
    module = _load_module()
    chunk_a = tmp_path / "chunk_a.csv"
    chunk_b = tmp_path / "chunk_b.csv"
    shared = {
        "dataset_id": "hill_of_towie_multivariate_knn6",
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

    assert row["dataset_id"] == "hill_of_towie_multivariate_knn6"
    assert row["split_name"] == "test"
    assert row["eval_protocol"] == module.NON_OVERLAP_EVAL_PROTOCOL
    assert row["metric_scope"] == module.OVERALL_METRIC_SCOPE
    assert row["lead_step"] is None
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
                dataset_id="sdwpf_kddcup_univariate",
                eval_protocol=module.NON_OVERLAP_EVAL_PROTOCOL,
                metric_scope=module.HORIZON_METRIC_SCOPE,
                lead_step=2,
            ),
            _result_row(
                module,
                dataset_id="hill_of_towie_multivariate_knn6",
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
                dataset_id="hill_of_towie_multivariate_knn6",
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
            merged["split_name"].to_list(),
            merged["eval_protocol"].to_list(),
            merged["metric_scope"].to_list(),
            merged["lead_step"].to_list(),
            strict=True,
        )
    ) == [
        (
            "hill_of_towie_multivariate_knn6",
            "test",
            module.ROLLING_EVAL_PROTOCOL,
            module.OVERALL_METRIC_SCOPE,
            None,
        ),
        (
            "hill_of_towie_multivariate_knn6",
            "test",
            module.ROLLING_EVAL_PROTOCOL,
            module.HORIZON_METRIC_SCOPE,
            1,
        ),
        (
            "sdwpf_kddcup_univariate",
            "test",
            module.NON_OVERLAP_EVAL_PROTOCOL,
            module.HORIZON_METRIC_SCOPE,
            2,
        ),
    ]


def test_validate_final_results_requires_complete_grid() -> None:
    module = _load_module()
    frame = pl.DataFrame(
        [
            _result_row(
                module,
                dataset_id="kelmarsh_univariate",
                eval_protocol=module.ROLLING_EVAL_PROTOCOL,
                metric_scope=module.OVERALL_METRIC_SCOPE,
                lead_step=None,
            )
        ]
    )

    try:
        module.validate_final_results(frame)
    except RuntimeError as exc:
        assert "Expected 1036 result rows" in str(exc)
    else:
        raise AssertionError("validate_final_results should reject incomplete results.")


def test_validate_final_results_rejects_unexpected_window_protocol() -> None:
    module = _load_module()
    frame = pl.DataFrame(_rows_for_expected_keys(module, window_protocol="legacy_6h_stride"))

    try:
        module.validate_final_results(frame)
    except RuntimeError as exc:
        assert "unexpected window_protocol" in str(exc)
    else:
        raise AssertionError("validate_final_results should reject mismatched window protocols.")


def test_build_progress_labels_matches_test_only_eval_slice_plan() -> None:
    module = _load_module()

    labels = module.build_progress_labels()

    assert len(labels) == 134
    assert labels[:3] == [
        "kelmarsh_univariate_rolling_origin_no_refit",
        "kelmarsh_univariate_power_stats_rolling_origin_no_refit",
        "kelmarsh_multivariate_knn6_rolling_origin_no_refit",
    ]
    assert labels[-3:] == [
        "sdwpf_kddcup_multivariate_knn6_targets_015_non_overlap",
        "sdwpf_kddcup_multivariate_knn6_targets_016_non_overlap",
        "sdwpf_kddcup_multivariate_knn6_targets_017_non_overlap",
    ]


def test_progress_is_enabled_requires_tqdm_and_tty(monkeypatch) -> None:
    module = _load_module()
    monkeypatch.setattr(module, "HAS_TQDM", True)
    monkeypatch.setattr(module.sys.stderr, "isatty", lambda: True)
    assert module.progress_is_enabled() is True

    monkeypatch.setattr(module.sys.stderr, "isatty", lambda: False)
    assert module.progress_is_enabled() is False

    monkeypatch.setattr(module, "HAS_TQDM", False)
    monkeypatch.setattr(module.sys.stderr, "isatty", lambda: True)
    assert module.progress_is_enabled() is False


def test_print_status_uses_tqdm_write(monkeypatch) -> None:
    module = _load_module()
    _TqdmRecorder.writes = []
    monkeypatch.setattr(module, "tqdm", _TqdmRecorder)

    module.print_status("hello")

    assert len(_TqdmRecorder.writes) == 1
    assert "hello" in _TqdmRecorder.writes[0]


def test_resolve_attempts_auto_uses_cuda_with_cpu_fallback(monkeypatch) -> None:
    module = _load_module()
    monkeypatch.setattr(module, "select_device", lambda: "cuda")

    primary, fallback = module.resolve_attempts(device=None, batch_size=32, fallback_batch_size=16)

    assert primary == module.Attempt(device="cuda", batch_size=32)
    assert fallback == module.Attempt(device="cpu", batch_size=16)


def test_resolve_attempts_auto_uses_mps_with_cpu_fallback(monkeypatch) -> None:
    module = _load_module()
    monkeypatch.setattr(module, "select_device", lambda: "mps")

    primary, fallback = module.resolve_attempts(device=None, batch_size=32, fallback_batch_size=16)

    assert primary == module.Attempt(device="mps", batch_size=32)
    assert fallback == module.Attempt(device="cpu", batch_size=16)


def test_resolve_attempts_cpu_has_no_fallback(monkeypatch) -> None:
    module = _load_module()
    monkeypatch.setattr(module, "select_device", lambda: "cpu")

    auto_primary, auto_fallback = module.resolve_attempts(device=None, batch_size=32, fallback_batch_size=16)
    explicit_primary, explicit_fallback = module.resolve_attempts(device="cpu", batch_size=32, fallback_batch_size=16)

    assert auto_primary == module.Attempt(device="cpu", batch_size=32)
    assert auto_fallback is None
    assert explicit_primary == module.Attempt(device="cpu", batch_size=32)
    assert explicit_fallback is None


def test_resolve_attempts_explicit_non_cpu_preserves_cpu_fallback() -> None:
    module = _load_module()

    cuda_primary, cuda_fallback = module.resolve_attempts(device="cuda", batch_size=32, fallback_batch_size=16)
    mps_primary, mps_fallback = module.resolve_attempts(device="mps", batch_size=32, fallback_batch_size=16)

    assert cuda_primary == module.Attempt(device="cuda", batch_size=32)
    assert cuda_fallback == module.Attempt(device="cpu", batch_size=16)
    assert mps_primary == module.Attempt(device="mps", batch_size=32)
    assert mps_fallback == module.Attempt(device="cpu", batch_size=16)


def test_resolve_legacy_full_run_batch_plan_matches_previous_chunk_defaults() -> None:
    module = _load_module()

    assert module.resolve_legacy_full_run_batch_plan(dataset_id="kelmarsh", mode="univariate") == (32, 16)
    assert module.resolve_legacy_full_run_batch_plan(dataset_id="hill_of_towie", mode="univariate") == (4, 4)
    assert module.resolve_legacy_full_run_batch_plan(dataset_id="sdwpf_kddcup", mode="multivariate_knn6") == (4, 1)


def test_resolve_full_run_batch_plan_uses_tuned_cuda_profile_with_legacy_cpu_fallbacks() -> None:
    module = _load_module()

    assert module.resolve_full_run_batch_plan(
        dataset_id="kelmarsh",
        mode="univariate",
        device="cuda",
    ) == (128, 16)
    assert module.resolve_full_run_batch_plan(
        dataset_id="penmanshiel",
        mode="univariate_power_stats",
        device="cuda",
    ) == (32, 16)
    assert module.resolve_full_run_batch_plan(
        dataset_id="hill_of_towie",
        mode="multivariate_knn6",
        device="cuda",
    ) == (32, 1)


def test_resolve_full_run_batch_plan_preserves_legacy_profile_for_non_cuda_devices() -> None:
    module = _load_module()

    assert module.resolve_full_run_batch_plan(
        dataset_id="kelmarsh",
        mode="univariate",
        device="mps",
    ) == (32, 16)
    assert module.resolve_full_run_batch_plan(
        dataset_id="sdwpf_kddcup",
        mode="univariate",
        device="cpu",
    ) == (16, 8)


def test_build_cli_command_uses_eval_protocol_and_split_window_limit() -> None:
    module = _load_module()

    command = module.build_cli_command(
        dataset_id="kelmarsh",
        mode="univariate",
        output_path=Path("/tmp/chronos-2.csv"),
        attempt=module.Attempt(device="cpu", batch_size=32),
        eval_protocol=module.NON_OVERLAP_EVAL_PROTOCOL,
        max_windows_per_split=64,
    )

    assert "--window-protocol" not in command
    assert _command_option(command, "--eval-protocol") == module.NON_OVERLAP_EVAL_PROTOCOL
    assert _command_option(command, "--max-windows-per-split") == "64"
    assert "--emit-progress-events" not in command


def test_build_cli_command_appends_emit_progress_events_flag() -> None:
    module = _load_module()

    command = module.build_cli_command(
        dataset_id="kelmarsh",
        mode="univariate",
        output_path=Path("/tmp/chronos-2.csv"),
        attempt=module.Attempt(device="cpu", batch_size=32),
        eval_protocol=module.ROLLING_EVAL_PROTOCOL,
        emit_progress_events=True,
    )

    assert "--emit-progress-events" in command


def test_execute_chunk_retries_on_cpu_after_primary_failure(monkeypatch, tmp_path) -> None:
    module = _load_module()
    seen_commands = _install_fake_popen(
        monkeypatch,
        module,
        attempts=[{"returncode": 1}, {"returncode": 0}],
    )

    output_path = module.execute_chunk(
        label="chunk_retry",
        dataset_id="kelmarsh",
        mode="univariate",
        work_dir=tmp_path,
        primary=module.Attempt(device="cuda", batch_size=32),
        fallback=module.Attempt(device="cpu", batch_size=16),
        eval_protocol=module.ROLLING_EVAL_PROTOCOL,
    )

    assert output_path == tmp_path / "chunks" / "chunk_retry.csv"
    assert [(_command_option(command, "--device"), _command_option(command, "--batch-size")) for command in seen_commands] == [
        ("cuda", "32"),
        ("cpu", "16"),
    ]
    retry_log = _read_log(tmp_path / "logs" / "chunk_retry__fallback.json")
    assert retry_log["returncode"] == 0
    assert _command_option(retry_log["command"], "--eval-protocol") == module.ROLLING_EVAL_PROTOCOL


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
                        _progress_line(
                            module,
                            "progress_stage_start",
                            chunk_total_batches=2,
                            stage="univariate",
                            stage_total_batches=2,
                        ),
                        _progress_line(
                            module,
                            "progress_batch",
                            chunk_total_batches=2,
                            completed_chunk_batches=1,
                            stage="univariate",
                            turbine_id="Kelmarsh 1",
                        ),
                        _progress_line(
                            module,
                            "progress_batch",
                            chunk_total_batches=2,
                            completed_chunk_batches=2,
                            stage="univariate",
                            turbine_id="Kelmarsh 1",
                        ),
                        _progress_line(
                            module,
                            "progress_stage_complete",
                            chunk_total_batches=2,
                            completed_chunk_batches=2,
                            stage="univariate",
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
        mode="univariate",
        work_dir=tmp_path,
        primary=module.Attempt(device="cuda", batch_size=32),
        fallback=module.Attempt(device="cpu", batch_size=16),
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
    assert any("cuda/32" in postfix for postfix in progress_bars[0].postfixes)
    assert any("stage=univariate" in postfix for postfix in progress_bars[0].postfixes)
    log = _read_log(tmp_path / "logs" / "chunk_progress.json")
    assert log["stderr"] == "stderr-0\n"
    assert log["eval_protocol"] == module.ROLLING_EVAL_PROTOCOL


def test_run_target_group_chunks_splits_after_failure(monkeypatch, tmp_path) -> None:
    module = _load_module()
    seen_groups: list[tuple[str, ...]] = []

    def _fake_execute_chunk(**kwargs):
        turbine_ids = tuple(kwargs["turbine_ids"])
        seen_groups.append(turbine_ids)
        if len(turbine_ids) > 1:
            raise RuntimeError("chunk too large")
        window_offset = int(kwargs["window_offset"])
        output_path = tmp_path / f"{kwargs['label']}.csv"
        window_count = 1 if window_offset == 0 else 0
        prediction_count = 1 if window_offset == 0 else 0
        pl.DataFrame(
            [
                _result_row(
                    module,
                    dataset_id="hill_of_towie_multivariate_knn6",
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

    chunk_paths = module.run_target_group_chunks(
        dataset_id="hill_of_towie",
        mode="multivariate_knn6",
        work_dir=tmp_path,
        primary=module.Attempt(device="mps", batch_size=4),
        fallback=module.Attempt(device="cpu", batch_size=1),
        target_groups=[("T01", "T02", "T03", "T04")],
        eval_protocol=module.NON_OVERLAP_EVAL_PROTOCOL,
        windows_per_chunk=5,
    )

    assert seen_groups[0] == ("T01", "T02", "T03", "T04")
    assert set(seen_groups[1:]) == {
        ("T01", "T02"),
        ("T01",),
        ("T02",),
        ("T03", "T04"),
        ("T03",),
        ("T04",),
    }
    assert len(chunk_paths) == 4


def test_run_window_chunks_stops_after_empty_chunk(monkeypatch, tmp_path) -> None:
    module = _load_module()
    seen_offsets: list[int] = []

    def _fake_execute_chunk(**kwargs):
        window_offset = int(kwargs["window_offset"])
        seen_offsets.append(window_offset)
        output_path = tmp_path / f"{kwargs['label']}.csv"
        window_count = 2 if window_offset == 0 else 0
        prediction_count = 2 if window_offset == 0 else 0
        pl.DataFrame(
            [
                _result_row(
                    module,
                    dataset_id="kelmarsh_univariate",
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
        label_prefix="kelmarsh_univariate_rolling_origin_no_refit",
        dataset_id="kelmarsh",
        mode="univariate",
        work_dir=tmp_path,
        primary=module.Attempt(device="cpu", batch_size=32),
        fallback=None,
        eval_protocol=module.ROLLING_EVAL_PROTOCOL,
        windows_per_chunk=5,
    )

    assert seen_offsets == [0, 5]
    assert chunk_paths == [tmp_path / "kelmarsh_univariate_rolling_origin_no_refit_offset_00000.csv"]
