from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - compatibility fallback for non-project interpreters
    try:
        import tomli as tomllib  # type: ignore[no-redef]
    except ModuleNotFoundError as exc:  # pragma: no cover - exercised only outside the project env
        raise ModuleNotFoundError(
            "experiment_registry.py requires Python 3.11+ or the tomli package."
        ) from exc


ALLOWED_DATASET_IDS = ("kelmarsh", "penmanshiel", "hill_of_towie", "sdwpf_kddcup")
ALLOWED_FAMILY_STATUSES = ("benchmark", "pilot", "ablation", "prototype", "archived")
ALLOWED_TRAINING_MODES = ("zero_shot", "trainable", "analytic_baseline")


@dataclass(frozen=True)
class TaskContract:
    task_id: str
    history_duration: str
    forecast_duration: str
    window_protocol: str
    split_protocol: str
    granularity: str


@dataclass(frozen=True)
class ExperimentFamilySpec:
    family_id: str
    display_name: str
    status: str
    summary: str
    model_family: str
    training_mode: str
    implementation_root: str
    readme_path: str
    task_contract: TaskContract
    dataset_scope: tuple[str, ...]
    supported_eval_protocols: tuple[str, ...]
    supported_result_splits: tuple[str, ...]
    supported_feature_protocols: tuple[str, ...]
    implementation_label_kind: str
    implementation_labels: tuple[str, ...]
    implementation_bindings: dict[str, str]
    runner_entrypoint: str | None = None
    orchestrator_entrypoint: str | None = None
    default_output_path: str | None = None
    default_output_schema_version: str | None = None
    model_variants: tuple[str, ...] = ()
    notes: tuple[str, ...] = ()


@dataclass(frozen=True)
class RegistrySnapshot:
    repo_root: Path
    registry_root: Path
    families: dict[str, ExperimentFamilySpec]


@dataclass(frozen=True)
class CoverageRow:
    dataset_id: str
    family_id: str
    feature_protocol_id: str
    family_status: str
    model_family: str
    training_mode: str
    implementation_label_kind: str
    implementation_labels: tuple[str, ...]
    supported_result_splits: tuple[str, ...]


def _registry_root_from(repo_root: str | Path | None = None) -> Path:
    if repo_root is not None:
        return Path(repo_root).resolve() / "experiment" / "infra" / "registry"
    return Path(__file__).resolve().parents[1] / "registry"


def _repo_root_from_registry_root(registry_root: Path) -> Path:
    return registry_root.parents[2]


def _load_toml(path: Path) -> dict[str, Any]:
    with path.open("rb") as handle:
        return tomllib.load(handle)


def _require_string(data: dict[str, Any], key: str, *, path: Path) -> str:
    value = data.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{path}: expected non-empty string field {key!r}.")
    return value.strip()


def _optional_string(data: dict[str, Any], key: str, *, path: Path) -> str | None:
    value = data.get(key)
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{path}: expected optional string field {key!r} to be non-empty when present.")
    return value.strip()


def _require_string_tuple(data: dict[str, Any], key: str, *, path: Path) -> tuple[str, ...]:
    value = data.get(key)
    if not isinstance(value, list) or not value:
        raise ValueError(f"{path}: expected non-empty array field {key!r}.")
    normalized: list[str] = []
    for item in value:
        if not isinstance(item, str) or not item.strip():
            raise ValueError(f"{path}: field {key!r} must contain non-empty strings.")
        normalized.append(item.strip())
    if len(normalized) != len(set(normalized)):
        raise ValueError(f"{path}: field {key!r} must not contain duplicates.")
    return tuple(normalized)


def _optional_string_tuple(data: dict[str, Any], key: str, *, path: Path) -> tuple[str, ...]:
    value = data.get(key)
    if value is None:
        return ()
    if not isinstance(value, list):
        raise ValueError(f"{path}: expected optional array field {key!r}.")
    normalized: list[str] = []
    for item in value:
        if not isinstance(item, str) or not item.strip():
            raise ValueError(f"{path}: field {key!r} must contain non-empty strings when present.")
        normalized.append(item.strip())
    if len(normalized) != len(set(normalized)):
        raise ValueError(f"{path}: field {key!r} must not contain duplicates.")
    return tuple(normalized)


def _require_string_map(data: dict[str, Any], key: str, *, path: Path) -> dict[str, str]:
    value = data.get(key)
    if not isinstance(value, dict) or not value:
        raise ValueError(f"{path}: expected non-empty table field {key!r}.")
    normalized: dict[str, str] = {}
    for raw_key, raw_value in value.items():
        if not isinstance(raw_key, str) or not raw_key.strip():
            raise ValueError(f"{path}: table {key!r} must have non-empty string keys.")
        if not isinstance(raw_value, str) or not raw_value.strip():
            raise ValueError(f"{path}: table {key!r} must have non-empty string values.")
        normalized[raw_key.strip()] = raw_value.strip()
    return normalized


def _load_family_spec(path: Path) -> ExperimentFamilySpec:
    data = _load_toml(path)
    schema_version = data.get("schema_version")
    if schema_version != 1:
        raise ValueError(f"{path}: unsupported schema_version {schema_version!r}.")
    family_id = _require_string(data, "family_id", path=path)
    if path.stem != family_id:
        raise ValueError(f"{path}: filename stem must match family_id {family_id!r}.")
    task_contract_data = data.get("task_contract")
    if not isinstance(task_contract_data, dict):
        raise ValueError(f"{path}: missing task_contract table.")
    status = _require_string(data, "status", path=path)
    if status not in ALLOWED_FAMILY_STATUSES:
        raise ValueError(f"{path}: unsupported family status {status!r}.")
    training_mode = _require_string(data, "training_mode", path=path)
    if training_mode not in ALLOWED_TRAINING_MODES:
        raise ValueError(f"{path}: unsupported training_mode {training_mode!r}.")
    dataset_scope = _require_string_tuple(data, "dataset_scope", path=path)
    unknown_datasets = sorted(set(dataset_scope) - set(ALLOWED_DATASET_IDS))
    if unknown_datasets:
        raise ValueError(f"{path}: unknown dataset ids {unknown_datasets!r}.")
    supported_feature_protocols = _require_string_tuple(data, "supported_feature_protocols", path=path)
    implementation_labels = _require_string_tuple(data, "implementation_labels", path=path)
    implementation_bindings = _require_string_map(data, "implementation_bindings", path=path)
    if set(implementation_bindings) != set(implementation_labels):
        raise ValueError(f"{path}: implementation_bindings keys must exactly match implementation_labels.")
    spec = ExperimentFamilySpec(
        family_id=family_id,
        display_name=_require_string(data, "display_name", path=path),
        status=status,
        summary=_require_string(data, "summary", path=path),
        model_family=_require_string(data, "model_family", path=path),
        training_mode=training_mode,
        implementation_root=_require_string(data, "implementation_root", path=path),
        readme_path=_require_string(data, "readme_path", path=path),
        runner_entrypoint=_optional_string(data, "runner_entrypoint", path=path),
        orchestrator_entrypoint=_optional_string(data, "orchestrator_entrypoint", path=path),
        default_output_path=_optional_string(data, "default_output_path", path=path),
        default_output_schema_version=_optional_string(data, "default_output_schema_version", path=path),
        task_contract=TaskContract(
            task_id=_require_string(task_contract_data, "task_id", path=path),
            history_duration=_require_string(task_contract_data, "history_duration", path=path),
            forecast_duration=_require_string(task_contract_data, "forecast_duration", path=path),
            window_protocol=_require_string(task_contract_data, "window_protocol", path=path),
            split_protocol=_require_string(task_contract_data, "split_protocol", path=path),
            granularity=_require_string(task_contract_data, "granularity", path=path),
        ),
        dataset_scope=dataset_scope,
        supported_eval_protocols=_require_string_tuple(data, "supported_eval_protocols", path=path),
        supported_result_splits=_require_string_tuple(data, "supported_result_splits", path=path),
        supported_feature_protocols=supported_feature_protocols,
        implementation_label_kind=_require_string(data, "implementation_label_kind", path=path),
        implementation_labels=implementation_labels,
        implementation_bindings=implementation_bindings,
        model_variants=_optional_string_tuple(data, "model_variants", path=path),
        notes=_optional_string_tuple(data, "notes", path=path),
    )
    if spec.default_output_schema_version is not None and spec.default_output_path is None:
        raise ValueError(f"{path}: default_output_schema_version requires default_output_path.")
    return spec


def load_registry_snapshot(repo_root: str | Path | None = None) -> RegistrySnapshot:
    registry_root = _registry_root_from(repo_root)
    repo_root_path = _repo_root_from_registry_root(registry_root)
    family_dir = registry_root / "families"
    if not family_dir.exists():
        raise FileNotFoundError(f"Registry family directory is missing: {family_dir}")

    families: dict[str, ExperimentFamilySpec] = {}
    for path in sorted(family_dir.glob("*.toml")):
        spec = _load_family_spec(path)
        if spec.family_id in families:
            raise ValueError(f"Duplicate family_id {spec.family_id!r}.")
        bound_protocols = tuple(spec.implementation_bindings[label] for label in spec.implementation_labels)
        if set(bound_protocols) != set(spec.supported_feature_protocols):
            raise ValueError(f"{path}: supported_feature_protocols must exactly match the protocols referenced by implementation_bindings.")
        _validate_repo_path(repo_root_path, spec.implementation_root, path=path, field_name="implementation_root", must_exist=True)
        _validate_repo_path(repo_root_path, spec.readme_path, path=path, field_name="readme_path", must_exist=True)
        if spec.runner_entrypoint is not None:
            _validate_repo_path(repo_root_path, spec.runner_entrypoint, path=path, field_name="runner_entrypoint", must_exist=True)
        if spec.orchestrator_entrypoint is not None:
            _validate_repo_path(repo_root_path, spec.orchestrator_entrypoint, path=path, field_name="orchestrator_entrypoint", must_exist=True)
        if spec.default_output_path is not None:
            _validate_repo_path(repo_root_path, spec.default_output_path, path=path, field_name="default_output_path", must_exist=False)
        families[spec.family_id] = spec

    return RegistrySnapshot(
        repo_root=repo_root_path,
        registry_root=registry_root,
        families=families,
    )


def _validate_repo_path(repo_root: Path, relative_path: str, *, path: Path, field_name: str, must_exist: bool) -> None:
    resolved_repo_root = repo_root.resolve()
    resolved = (repo_root / relative_path).resolve()
    if resolved != resolved_repo_root and resolved_repo_root not in resolved.parents:
        raise ValueError(f"{path}: {field_name} path {relative_path!r} escapes the repository root.")
    if must_exist and not resolved.exists():
        raise ValueError(f"{path}: {field_name} points to missing path {relative_path!r}.")
    if not must_exist:
        existing_parent = resolved.parent
        while existing_parent != resolved_repo_root and not existing_parent.exists():
            existing_parent = existing_parent.parent
        if not existing_parent.exists():
            raise ValueError(
                f"{path}: no existing repository ancestor was found for {field_name} path {relative_path!r}."
            )


def build_dataset_family_feature_matrix(
    snapshot: RegistrySnapshot | None = None,
    *,
    repo_root: str | Path | None = None,
) -> tuple[CoverageRow, ...]:
    resolved_snapshot = snapshot or load_registry_snapshot(repo_root=repo_root)
    rows: list[CoverageRow] = []
    for family in sorted(resolved_snapshot.families.values(), key=lambda item: item.family_id):
        labels_by_protocol: dict[str, list[str]] = {protocol_id: [] for protocol_id in family.supported_feature_protocols}
        for label in family.implementation_labels:
            protocol_id = family.implementation_bindings[label]
            labels_by_protocol[protocol_id].append(label)
        for dataset_id in family.dataset_scope:
            for protocol_id in family.supported_feature_protocols:
                rows.append(
                    CoverageRow(
                        dataset_id=dataset_id,
                        family_id=family.family_id,
                        feature_protocol_id=protocol_id,
                        family_status=family.status,
                        model_family=family.model_family,
                        training_mode=family.training_mode,
                        implementation_label_kind=family.implementation_label_kind,
                        implementation_labels=tuple(labels_by_protocol[protocol_id]),
                        supported_result_splits=family.supported_result_splits,
                    )
                )
    return tuple(rows)


def render_matrix_markdown(rows: tuple[CoverageRow, ...]) -> str:
    header = [
        "dataset_id",
        "family_id",
        "feature_protocol_id",
        "status",
        "training_mode",
        "implementation_labels",
        "result_splits",
    ]
    lines = [
        "| " + " | ".join(header) + " |",
        "| " + " | ".join(["---"] * len(header)) + " |",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    row.dataset_id,
                    row.family_id,
                    row.feature_protocol_id,
                    row.family_status,
                    row.training_mode,
                    ", ".join(row.implementation_labels),
                    ", ".join(row.supported_result_splits),
                ]
            )
            + " |"
        )
    return "\n".join(lines)


def coverage_rows_to_dicts(rows: tuple[CoverageRow, ...]) -> list[dict[str, object]]:
    return [
        {
            "dataset_id": row.dataset_id,
            "family_id": row.family_id,
            "feature_protocol_id": row.feature_protocol_id,
            "family_status": row.family_status,
            "model_family": row.model_family,
            "training_mode": row.training_mode,
            "implementation_label_kind": row.implementation_label_kind,
            "implementation_labels": list(row.implementation_labels),
            "supported_result_splits": list(row.supported_result_splits),
        }
        for row in rows
    ]


def render_matrix_csv(rows: tuple[CoverageRow, ...]) -> str:
    fieldnames = [
        "dataset_id",
        "family_id",
        "feature_protocol_id",
        "family_status",
        "model_family",
        "training_mode",
        "implementation_label_kind",
        "implementation_labels",
        "supported_result_splits",
    ]
    from io import StringIO

    buffer = StringIO()
    writer = csv.DictWriter(buffer, fieldnames=fieldnames)
    writer.writeheader()
    for row in rows:
        writer.writerow(
            {
                "dataset_id": row.dataset_id,
                "family_id": row.family_id,
                "feature_protocol_id": row.feature_protocol_id,
                "family_status": row.family_status,
                "model_family": row.model_family,
                "training_mode": row.training_mode,
                "implementation_label_kind": row.implementation_label_kind,
                "implementation_labels": ",".join(row.implementation_labels),
                "supported_result_splits": ",".join(row.supported_result_splits),
            }
        )
    return buffer.getvalue().rstrip()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Validate the experiment registry and print the dataset x family x feature-protocol matrix."
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=None,
        help="Optional repository root override. Defaults to the current repository.",
    )
    parser.add_argument(
        "--format",
        choices=("markdown", "csv", "json"),
        default="markdown",
        help="Output format for the coverage matrix.",
    )
    args = parser.parse_args(argv)

    snapshot = load_registry_snapshot(repo_root=args.repo_root)
    rows = build_dataset_family_feature_matrix(snapshot)
    if args.format == "markdown":
        print(render_matrix_markdown(rows))
    elif args.format == "csv":
        print(render_matrix_csv(rows))
    else:
        print(
            json.dumps(
                {
                    "families": sorted(snapshot.families),
                    "coverage_rows": coverage_rows_to_dicts(rows),
                },
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
