from __future__ import annotations

from pathlib import Path


DEFAULT_PUBLISHED_FILENAME = "latest.csv"

_KNOWN_FAMILY_IDS = {
    "agcrn_official_aligned",
}

_EXPERIMENT_NAME_TO_FAMILY_ID = {
    "agcrn": "agcrn_official_aligned",
}


def default_published_root(*, repo_root: str | Path) -> Path:
    return Path(repo_root).resolve() / "experiment" / "artifacts" / "published"


def default_family_output_dir(*, repo_root: str | Path, family_id: str) -> Path:
    if family_id not in _KNOWN_FAMILY_IDS:
        raise ValueError(f"Unknown experiment family {family_id!r}.")
    return default_published_root(repo_root=repo_root) / family_id


def default_family_output_path(
    *,
    repo_root: str | Path,
    family_id: str,
    filename: str = DEFAULT_PUBLISHED_FILENAME,
) -> Path:
    resolved_filename = Path(filename)
    if resolved_filename.name != filename or not filename.strip():
        raise ValueError(f"Published output filename must be a single path component, found {filename!r}.")
    return default_family_output_dir(repo_root=repo_root, family_id=family_id) / filename


def family_id_for_experiment_name(experiment_name: str) -> str:
    try:
        return _EXPERIMENT_NAME_TO_FAMILY_ID[experiment_name]
    except KeyError as exc:
        raise ValueError(f"Unknown experiment name {experiment_name!r}.") from exc


def default_experiment_output_path(*, repo_root: str | Path, experiment_name: str) -> Path:
    return default_family_output_path(
        repo_root=repo_root,
        family_id=family_id_for_experiment_name(experiment_name),
    )


__all__ = [
    "DEFAULT_PUBLISHED_FILENAME",
    "default_published_root",
    "default_family_output_dir",
    "default_family_output_path",
    "family_id_for_experiment_name",
    "default_experiment_output_path",
]
