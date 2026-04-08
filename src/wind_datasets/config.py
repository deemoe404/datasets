from __future__ import annotations

from functools import lru_cache
from pathlib import Path
import tomllib

PROJECT_ROOT_MARKER = "pyproject.toml"
LOCAL_CONFIG_FILENAME = "wind_datasets.local.toml"
LOCAL_CONFIG_TEMPLATE_FILENAME = "wind_datasets.local.toml.example"
_SOURCE_DATA_ROOT_KEY = "[paths].source_data_root"


class ProjectConfigError(RuntimeError):
    """Raised when the repository-local runtime configuration is missing or invalid."""


def _find_project_root(start: Path) -> Path:
    candidate = start if start.is_dir() else start.parent
    for path in (candidate, *candidate.parents):
        if (path / PROJECT_ROOT_MARKER).is_file():
            return path
    raise ProjectConfigError(
        f"Could not locate repository root from {start}. Expected an ancestor containing {PROJECT_ROOT_MARKER}."
    )


@lru_cache(maxsize=1)
def _cached_project_root() -> Path:
    return _find_project_root(Path(__file__).resolve())


def get_project_root() -> Path:
    return _cached_project_root()


def get_project_config_path() -> Path:
    return get_project_root() / LOCAL_CONFIG_FILENAME


def get_project_config_template_path() -> Path:
    return get_project_root() / LOCAL_CONFIG_TEMPLATE_FILENAME


@lru_cache(maxsize=1)
def load_project_config() -> dict[str, object]:
    config_path = get_project_config_path()
    if not config_path.exists():
        raise ProjectConfigError(
            f"Missing project config file {config_path}. "
            f"Copy {get_project_config_template_path()} and set {_SOURCE_DATA_ROOT_KEY}."
        )
    if not config_path.is_file():
        raise ProjectConfigError(f"Invalid project config file {config_path}: expected a regular file.")

    try:
        return tomllib.loads(config_path.read_text(encoding="utf-8"))
    except tomllib.TOMLDecodeError as exc:
        raise ProjectConfigError(f"Invalid project config file {config_path}: {exc}.") from exc


@lru_cache(maxsize=1)
def get_source_data_root() -> Path:
    config_path = get_project_config_path()
    payload = load_project_config()
    paths = payload.get("paths")
    if not isinstance(paths, dict):
        raise ProjectConfigError(f"Invalid project config file {config_path}: missing [paths] table.")

    raw_source_root = paths.get("source_data_root")
    if not isinstance(raw_source_root, str) or not raw_source_root.strip():
        raise ProjectConfigError(
            f"Invalid project config file {config_path}: expected {_SOURCE_DATA_ROOT_KEY} to be a non-empty string."
        )

    source_root = Path(raw_source_root).expanduser()
    if not source_root.exists():
        raise ProjectConfigError(
            f"Invalid project config file {config_path}: {_SOURCE_DATA_ROOT_KEY}={raw_source_root!r} does not exist."
        )
    if not source_root.is_dir():
        raise ProjectConfigError(
            f"Invalid project config file {config_path}: {_SOURCE_DATA_ROOT_KEY}={raw_source_root!r} is not a directory."
        )
    return source_root.resolve()


def clear_config_caches() -> None:
    get_source_data_root.cache_clear()
    load_project_config.cache_clear()
    _cached_project_root.cache_clear()
