from __future__ import annotations

from pathlib import Path

import pytest

from wind_datasets import config as config_module
from wind_datasets import registry as registry_module
from wind_datasets.config import ProjectConfigError


@pytest.fixture(autouse=True)
def _clear_config_caches() -> None:
    config_module.clear_config_caches()
    yield
    config_module.clear_config_caches()


def _write_local_config(project_root: Path, source_root: str | None = None) -> Path:
    content = "[paths]\n"
    if source_root is not None:
        content += f'source_data_root = "{source_root}"\n'
    path = project_root / config_module.LOCAL_CONFIG_FILENAME
    path.write_text(content, encoding="utf-8")
    return path


def test_get_source_data_root_reads_repo_local_toml(tmp_path, monkeypatch) -> None:
    source_root = tmp_path / "sources"
    source_root.mkdir()
    _write_local_config(tmp_path, str(source_root))
    monkeypatch.setattr(config_module, "get_project_root", lambda: tmp_path)

    assert config_module.get_source_data_root() == source_root.resolve()


def test_get_source_data_root_requires_local_config_file(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(config_module, "get_project_root", lambda: tmp_path)

    with pytest.raises(ProjectConfigError, match=r"wind_datasets\.local\.toml") as excinfo:
        config_module.get_source_data_root()

    assert "source_data_root" in str(excinfo.value)


def test_get_source_data_root_requires_source_data_root_key(tmp_path, monkeypatch) -> None:
    _write_local_config(tmp_path, None)
    monkeypatch.setattr(config_module, "get_project_root", lambda: tmp_path)

    with pytest.raises(ProjectConfigError, match=r"wind_datasets\.local\.toml") as excinfo:
        config_module.get_source_data_root()

    assert "[paths].source_data_root" in str(excinfo.value)


def test_get_source_data_root_requires_existing_directory(tmp_path, monkeypatch) -> None:
    missing_root = tmp_path / "missing"
    _write_local_config(tmp_path, str(missing_root))
    monkeypatch.setattr(config_module, "get_project_root", lambda: tmp_path)

    with pytest.raises(ProjectConfigError, match=r"wind_datasets\.local\.toml") as excinfo:
        config_module.get_source_data_root()

    assert str(missing_root) in str(excinfo.value)


def test_registry_builds_dataset_source_roots_from_configured_root(monkeypatch) -> None:
    configured_root = Path("/home/sam/Documents/datasets/Wind Power Forecasting")
    monkeypatch.setattr(registry_module.config, "get_source_data_root", lambda: configured_root)

    assert registry_module.get_dataset_spec("kelmarsh").source_root == configured_root / "Kelmarsh wind farm data"
    assert registry_module.get_dataset_spec("penmanshiel").source_root == configured_root / "Penmanshiel wind farm data"
    assert registry_module.get_dataset_spec("hill_of_towie").source_root == configured_root / "Hill of Towie"
    assert registry_module.get_dataset_spec("sdwpf_kddcup").source_root == (
        configured_root / "SDWPF_dataset" / "sdwpf_kddcup"
    )
