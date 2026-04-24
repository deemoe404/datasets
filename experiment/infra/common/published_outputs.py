from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
import re


RUN_TIMESTAMP_FORMAT = "%Y%m%d-%H%M%S"
RUN_TIMESTAMP_TOKEN = "{run_timestamp}"
DEFAULT_PUBLISHED_FILENAME_TEMPLATE = f"{RUN_TIMESTAMP_TOKEN}.csv"

_RUN_STEM_PATTERN = re.compile(r"^\d{8}-\d{6}$")

_KNOWN_FAMILY_IDS = {
    "agcrn_official_aligned",
    "agcrn_masked",
    "world_model_agcrn_v1",
    "world_model_baselines_v1",
    "world_model_hardened_baselines_v1",
    "world_model_rollout_v1",
    "world_model_state_space_v1",
}

_EXPERIMENT_NAME_TO_FAMILY_ID = {
    "agcrn": "agcrn_official_aligned",
    "agcrn_masked": "agcrn_masked",
    "world_model_agcrn_v1": "world_model_agcrn_v1",
    "world_model_baselines_v1": "world_model_baselines_v1",
    "world_model_hardened_baselines_v1": "world_model_hardened_baselines_v1",
    "world_model_rollout_v1": "world_model_rollout_v1",
    "world_model_state_space_v1": "world_model_state_space_v1",
}


def default_published_root(*, repo_root: str | Path) -> Path:
    return Path(repo_root).resolve() / "experiment" / "artifacts" / "published"


def default_family_output_dir(*, repo_root: str | Path, family_id: str) -> Path:
    if family_id not in _KNOWN_FAMILY_IDS:
        raise ValueError(f"Unknown experiment family {family_id!r}.")
    return default_published_root(repo_root=repo_root) / family_id


def generate_run_stem(*, now: datetime | None = None) -> str:
    moment = datetime.now(tz=UTC) if now is None else now
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=UTC)
    else:
        moment = moment.astimezone(UTC)
    return moment.strftime(RUN_TIMESTAMP_FORMAT)


def validate_run_stem(run_stem: str) -> str:
    normalized = run_stem.strip()
    if not _RUN_STEM_PATTERN.fullmatch(normalized):
        raise ValueError(
            f"Run timestamp stem must match {RUN_TIMESTAMP_FORMAT!r}, found {run_stem!r}."
        )
    return normalized


def _validate_single_path_component(value: str, *, label: str) -> str:
    normalized = value.strip()
    resolved_value = Path(normalized)
    if resolved_value.name != normalized or not normalized:
        raise ValueError(f"{label} must be a single path component, found {value!r}.")
    return normalized


def default_family_output_template(*, repo_root: str | Path, family_id: str) -> Path:
    return default_family_output_dir(repo_root=repo_root, family_id=family_id) / DEFAULT_PUBLISHED_FILENAME_TEMPLATE


def default_family_output_path(
    *,
    repo_root: str | Path,
    family_id: str,
    run_stem: str | None = None,
    filename: str | None = None,
) -> Path:
    if run_stem is not None and filename is not None:
        raise ValueError("Published output path accepts either run_stem or filename, not both.")
    if filename is None:
        filename = f"{validate_run_stem(run_stem or generate_run_stem())}.csv"
    resolved_filename = _validate_single_path_component(filename, label="Published output filename")
    return default_family_output_dir(repo_root=repo_root, family_id=family_id) / resolved_filename


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


def default_experiment_output_template(*, repo_root: str | Path, experiment_name: str) -> Path:
    return default_family_output_template(
        repo_root=repo_root,
        family_id=family_id_for_experiment_name(experiment_name),
    )


__all__ = [
    "DEFAULT_PUBLISHED_FILENAME_TEMPLATE",
    "RUN_TIMESTAMP_FORMAT",
    "RUN_TIMESTAMP_TOKEN",
    "default_published_root",
    "default_family_output_dir",
    "generate_run_stem",
    "validate_run_stem",
    "default_family_output_template",
    "default_family_output_path",
    "family_id_for_experiment_name",
    "default_experiment_output_template",
    "default_experiment_output_path",
]
