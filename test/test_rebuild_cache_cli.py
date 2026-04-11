from __future__ import annotations

import os
from pathlib import Path
import stat
import subprocess

import pytest

from wind_datasets import rebuild_cache as rebuild_module
from wind_datasets.config import ProjectConfigError


def _stage_name_for_task(task_id: str, feature_protocol_id: str) -> str:
    return f"tasks/{task_id}/{feature_protocol_id}"


def _patch_rebuild_api(
    monkeypatch: pytest.MonkeyPatch,
    calls: list[tuple[str, str]],
    *,
    fail_stage: str | None = None,
    fail_dataset: str | None = None,
    fail_message: str = "synthetic failure",
) -> None:
    def _record(stage_name: str, dataset: str) -> None:
        calls.append((dataset, stage_name))
        if stage_name == fail_stage and dataset == fail_dataset:
            raise ValueError(fail_message)

    def _build_manifest(dataset_id: str, cache_root: str | Path = "cache") -> Path:
        _record("manifest", dataset_id)
        return Path(cache_root) / dataset_id / "manifest" / "manifest.json"

    def _build_silver(dataset_id: str, cache_root: str | Path = "cache") -> Path:
        _record("silver", dataset_id)
        return Path(cache_root) / dataset_id / "silver"

    def _build_gold_base(
        dataset_id: str,
        cache_root: str | Path = "cache",
        quality_profile: str | None = None,
        layout: str | None = None,
    ) -> Path:
        del quality_profile
        del layout
        stage_name = "gold_base"
        _record(stage_name, dataset_id)
        return Path(cache_root) / dataset_id / "gold_base" / "series.parquet"

    def _build_task_cache(
        dataset_id: str,
        task_spec,
        cache_root: str | Path = "cache",
        feature_protocol_id: str = "power_only",
        quality_profile: str | None = None,
    ) -> Path:
        del quality_profile
        stage_name = _stage_name_for_task(task_spec.task_id, feature_protocol_id)
        _record(stage_name, dataset_id)
        return Path(cache_root) / dataset_id / "tasks" / task_spec.task_id / feature_protocol_id

    monkeypatch.setattr(rebuild_module, "build_manifest", _build_manifest)
    monkeypatch.setattr(rebuild_module, "build_silver", _build_silver)
    monkeypatch.setattr(rebuild_module, "build_gold_base", _build_gold_base)
    monkeypatch.setattr(rebuild_module, "build_task_cache", _build_task_cache)


def test_rebuild_cli_defaults_to_all_datasets_and_farm_only(monkeypatch, capsys) -> None:
    calls: list[tuple[str, str]] = []
    _patch_rebuild_api(monkeypatch, calls)

    code = rebuild_module.main([])

    assert code == 0
    expected_stages = [
        "manifest",
        "silver",
        "gold_base",
        "tasks/next_6h_from_24h/power_only",
    ]
    assert calls == [
        (dataset, stage)
        for dataset in rebuild_module.SUPPORTED_DATASETS
        for stage in expected_stages
    ]
    captured = capsys.readouterr()
    assert "completed successfully" in captured.out
    assert "turbine" not in captured.out


def test_rebuild_cli_include_turbine_adds_compatibility_stages(monkeypatch) -> None:
    calls: list[tuple[str, str]] = []
    monkeypatch.setattr(rebuild_module, "SUPPORTED_DATASETS", ("kelmarsh",))
    _patch_rebuild_api(monkeypatch, calls)

    code = rebuild_module.main(["--include-turbine", "kelmarsh"])

    assert code == 0
    assert calls == [
        ("kelmarsh", "manifest"),
        ("kelmarsh", "silver"),
        ("kelmarsh", "gold_base"),
        ("kelmarsh", "tasks/next_6h_from_24h/power_only"),
    ]


def test_rebuild_cli_all_and_duplicate_datasets_stay_deduped_and_ordered(monkeypatch) -> None:
    calls: list[tuple[str, str]] = []
    monkeypatch.setattr(
        rebuild_module,
        "SUPPORTED_DATASETS",
        ("kelmarsh", "penmanshiel", "hill_of_towie", "sdwpf_kddcup"),
    )
    _patch_rebuild_api(monkeypatch, calls)

    code = rebuild_module.main(["all", "kelmarsh", "penmanshiel", "kelmarsh"])

    assert code == 0
    manifest_datasets = [dataset for dataset, stage in calls if stage == "manifest"]
    assert manifest_datasets == list(rebuild_module.SUPPORTED_DATASETS)


def test_rebuild_cli_rejects_unsupported_dataset(capsys) -> None:
    code = rebuild_module.main(["not_a_dataset"])

    assert code == 2
    captured = capsys.readouterr()
    assert "Unsupported dataset" in captured.err


def test_rebuild_cli_clean_only_removes_selected_dataset(monkeypatch, tmp_path) -> None:
    calls: list[tuple[str, str]] = []
    cache_root = tmp_path / "cache"
    selected_dir = cache_root / "kelmarsh"
    untouched_dir = cache_root / "penmanshiel"
    selected_dir.mkdir(parents=True)
    untouched_dir.mkdir(parents=True)
    (selected_dir / "sentinel.txt").write_text("selected", encoding="utf-8")
    (untouched_dir / "sentinel.txt").write_text("untouched", encoding="utf-8")
    monkeypatch.setattr(rebuild_module, "SUPPORTED_DATASETS", ("kelmarsh", "penmanshiel"))
    _patch_rebuild_api(monkeypatch, calls)

    code = rebuild_module.main(["--clean", "--cache-root", str(cache_root), "kelmarsh"])

    assert code == 0
    assert not selected_dir.exists()
    assert untouched_dir.exists()
    assert calls[0] == ("kelmarsh", "manifest")


def test_rebuild_cli_records_failure_and_continues_next_dataset(monkeypatch, capsys) -> None:
    calls: list[tuple[str, str]] = []
    monkeypatch.setattr(rebuild_module, "SUPPORTED_DATASETS", ("kelmarsh", "penmanshiel"))
    _patch_rebuild_api(
        monkeypatch,
        calls,
        fail_stage="silver",
        fail_dataset="kelmarsh",
        fail_message="synthetic silver failure",
    )

    code = rebuild_module.main(["kelmarsh", "penmanshiel"])

    assert code == 1
    assert ("kelmarsh", "gold_base") not in calls
    assert calls[-4:] == [
        ("penmanshiel", "manifest"),
        ("penmanshiel", "silver"),
        ("penmanshiel", "gold_base"),
        ("penmanshiel", "tasks/next_6h_from_24h/power_only"),
    ]
    captured = capsys.readouterr()
    assert "completed with 1 failed dataset" in captured.err
    assert "dataset=kelmarsh stage=silver" in captured.err


def test_rebuild_cli_summarizes_sdwpf_gold_block_without_running_tasks(monkeypatch, capsys) -> None:
    calls: list[tuple[str, str]] = []
    monkeypatch.setattr(rebuild_module, "SUPPORTED_DATASETS", ("sdwpf_kddcup", "kelmarsh"))
    _patch_rebuild_api(
        monkeypatch,
        calls,
        fail_stage="gold_base",
        fail_dataset="sdwpf_kddcup",
        fail_message="Refusing to build sdwpf_kddcup gold/task cache. blocked by audit.",
    )

    code = rebuild_module.main(["sdwpf_kddcup", "kelmarsh"])

    assert code == 1
    assert ("sdwpf_kddcup", "manifest") in calls
    assert ("sdwpf_kddcup", "silver") in calls
    assert ("sdwpf_kddcup", "tasks/next_6h_from_24h/power_only") not in calls
    assert calls[-4:] == [
        ("kelmarsh", "manifest"),
        ("kelmarsh", "silver"),
        ("kelmarsh", "gold_base"),
        ("kelmarsh", "tasks/next_6h_from_24h/power_only"),
    ]
    captured = capsys.readouterr()
    assert "sdwpf_kddcup" in captured.err
    assert "blocked by audit" in captured.err


def test_rebuild_shell_wrapper_reports_missing_python_path(tmp_path) -> None:
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "rebuild_cache.sh"

    result = subprocess.run(
        ["bash", str(script_path), "--python"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 2
    assert result.stderr.strip() == "--python requires a path"


def test_rebuild_shell_wrapper_suggests_create_env_for_missing_default_python(tmp_path) -> None:
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "rebuild_cache.sh"
    repo_root = tmp_path / "repo"
    scripts_dir = repo_root / "scripts"
    scripts_dir.mkdir(parents=True)
    copied_script = scripts_dir / "rebuild_cache.sh"
    copied_script.write_text(script_path.read_text(encoding="utf-8"), encoding="utf-8")
    copied_script.chmod(copied_script.stat().st_mode | stat.S_IXUSR)

    result = subprocess.run(
        ["bash", str(copied_script), "kelmarsh"],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 1
    assert "Run ./scripts/create_env.sh" in result.stderr


def test_rebuild_shell_wrapper_passes_through_to_python_module(tmp_path) -> None:
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "rebuild_cache.sh"
    repo_root = script_path.parents[1]
    stub_path = tmp_path / "python-stub.sh"
    stub_path.write_text(
        "#!/usr/bin/env bash\n"
        "printf 'pwd=%s\\n' \"$PWD\"\n"
        "printf 'args=%s\\n' \"$*\"\n"
        "printf 'pythonpath=%s\\n' \"$PYTHONPATH\"\n",
        encoding="utf-8",
    )
    stub_path.chmod(stub_path.stat().st_mode | stat.S_IXUSR)

    result = subprocess.run(
        [
            "bash",
            str(script_path),
            "--python",
            str(stub_path),
            "--cache-root",
            str(tmp_path / "cache"),
            "kelmarsh",
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0
    assert f"pwd={repo_root}" in result.stdout
    assert "args=-m wind_datasets.rebuild_cache --cache-root" in result.stdout
    assert "kelmarsh" in result.stdout
    assert f"pythonpath={repo_root / 'src'}" in result.stdout


def test_scripts_create_env_script_bootstraps_conda_env_and_editable_install(tmp_path) -> None:
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    scripts_dir = repo_root / "scripts"
    scripts_dir.mkdir()
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "create_env.sh"
    env_file_path = Path(__file__).resolve().parents[1] / "scripts" / "environment.yml"
    copied_script = scripts_dir / "create_env.sh"
    copied_env_file = scripts_dir / "environment.yml"
    copied_script.write_text(script_path.read_text(encoding="utf-8"), encoding="utf-8")
    copied_env_file.write_text(env_file_path.read_text(encoding="utf-8"), encoding="utf-8")
    copied_script.chmod(copied_script.stat().st_mode | stat.S_IXUSR)

    stub_bin = tmp_path / "bin"
    stub_bin.mkdir()
    log_path = tmp_path / "create-env.log"
    conda_stub = stub_bin / "conda"
    conda_stub.write_text(
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        "printf 'conda=%s\\n' \"$*\" >> \"$LOG_PATH\"\n"
        "prefix=''\n"
        "while [[ $# -gt 0 ]]; do\n"
        "  case \"$1\" in\n"
        "    --prefix)\n"
        "      prefix=\"$2\"\n"
        "      shift 2\n"
        "      ;;\n"
        "    *)\n"
        "      shift\n"
        "      ;;\n"
        "  esac\n"
        "done\n"
        "mkdir -p \"$prefix/bin\"\n"
        "cat > \"$prefix/bin/python\" <<'PYEOF'\n"
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        "printf 'python=%s\\n' \"$*\" >> \"$LOG_PATH\"\n"
        "PYEOF\n"
        "chmod +x \"$prefix/bin/python\"\n",
        encoding="utf-8",
    )
    conda_stub.chmod(conda_stub.stat().st_mode | stat.S_IXUSR)

    env = os.environ.copy()
    env["PATH"] = f"{stub_bin}:{env['PATH']}"
    env["LOG_PATH"] = str(log_path)

    result = subprocess.run(
        ["/bin/bash", str(copied_script)],
        cwd=repo_root,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0
    log_text = log_path.read_text(encoding="utf-8")
    assert f"conda=env create --prefix {repo_root / '.conda'} --file {repo_root / 'scripts' / 'environment.yml'}" in log_text
    assert "python=-m pip install --upgrade pip" in log_text
    assert f"python=-m pip install --upgrade --editable {repo_root}" in log_text
    assert "Dataset processing environment is ready" in result.stdout


def test_scripts_create_env_script_requires_conda(tmp_path) -> None:
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    scripts_dir = repo_root / "scripts"
    scripts_dir.mkdir()
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "create_env.sh"
    env_file_path = Path(__file__).resolve().parents[1] / "scripts" / "environment.yml"
    copied_script = scripts_dir / "create_env.sh"
    copied_env_file = scripts_dir / "environment.yml"
    copied_script.write_text(script_path.read_text(encoding="utf-8"), encoding="utf-8")
    copied_env_file.write_text(env_file_path.read_text(encoding="utf-8"), encoding="utf-8")
    copied_script.chmod(copied_script.stat().st_mode | stat.S_IXUSR)

    env = os.environ.copy()
    env["PATH"] = str(tmp_path)

    result = subprocess.run(
        ["/bin/bash", str(copied_script)],
        cwd=repo_root,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 1
    assert "conda executable not found" in result.stderr


def test_rebuild_cli_check_reports_missing_project_config(monkeypatch, capsys) -> None:
    def _raise_missing_config(dataset: str):
        raise ProjectConfigError(
            "Missing project config file /tmp/wind_datasets.local.toml. "
            "Copy /tmp/wind_datasets.local.toml.example and set [paths].source_data_root."
        )

    monkeypatch.setattr(rebuild_module, "get_dataset_spec", _raise_missing_config)

    code = rebuild_module.main(["--check", "kelmarsh"])

    assert code == 1
    captured = capsys.readouterr()
    assert "wind_datasets.local.toml" in captured.err
    assert "source_data_root" in captured.err
