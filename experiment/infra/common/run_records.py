from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
import hashlib
import json
from pathlib import Path
import subprocess
from typing import Any, Mapping, Sequence

from experiment_registry import load_registry_snapshot


@dataclass(frozen=True)
class OutputArtifact:
    path: str
    exists: bool
    size_bytes: int | None
    sha256: str | None


def default_runs_root(*, repo_root: str | Path) -> Path:
    return Path(repo_root).resolve() / "experiment" / "artifacts" / "runs"


def default_scratch_root(*, repo_root: str | Path) -> Path:
    return Path(repo_root).resolve() / "experiment" / "artifacts" / "scratch"


def resolve_family_feature_protocol_ids(
    family_id: str,
    implementation_labels: Sequence[str],
    *,
    repo_root: str | Path,
) -> tuple[str, ...]:
    snapshot = load_registry_snapshot(repo_root=repo_root)
    try:
        family = snapshot.families[family_id]
    except KeyError as exc:
        raise ValueError(f"Unknown experiment family {family_id!r}.") from exc
    resolved: list[str] = []
    for label in implementation_labels:
        try:
            resolved.append(family.implementation_bindings[label])
        except KeyError as exc:
            raise ValueError(
                f"Family {family_id!r} does not define an implementation binding for label {label!r}."
            ) from exc
    return tuple(dict.fromkeys(resolved))


def record_cli_run(
    *,
    family_id: str,
    repo_root: str | Path,
    invocation_kind: str,
    entrypoint: str,
    args: Mapping[str, Any],
    output_path: str | Path,
    result_row_count: int | None,
    dataset_ids: Sequence[str],
    feature_protocol_ids: Sequence[str],
    model_variants: Sequence[str] = (),
    eval_protocols: Sequence[str] = (),
    result_splits: Sequence[str] = (),
    artifacts: Mapping[str, str | Path] | None = None,
    notes: Sequence[str] = (),
    run_label: str | None = None,
    runs_root: str | Path | None = None,
) -> Path:
    resolved_repo_root = Path(repo_root).resolve()
    snapshot = load_registry_snapshot(repo_root=resolved_repo_root)
    try:
        family = snapshot.families[family_id]
    except KeyError as exc:
        raise ValueError(f"Unknown experiment family {family_id!r}.") from exc

    resolved_protocol_ids = tuple(dict.fromkeys(feature_protocol_ids))
    unknown_protocols = sorted(set(resolved_protocol_ids) - set(family.supported_feature_protocols))
    if unknown_protocols:
        raise ValueError(
            f"Run record for family {family_id!r} references unsupported feature protocols {unknown_protocols!r}."
        )

    run_root = Path(runs_root).resolve() if runs_root is not None else default_runs_root(repo_root=resolved_repo_root)
    run_dir = _allocate_run_directory(run_root=run_root, family_id=family_id, run_label=run_label)
    output_artifact = describe_output_artifact(Path(output_path), repo_root=resolved_repo_root)
    artifact_payload = {
        key: {"path": _display_path(Path(value), repo_root=resolved_repo_root)}
        for key, value in (artifacts or {}).items()
    }

    payload = {
        "schema_version": 1,
        "created_at_utc": datetime.now(tz=UTC).isoformat(),
        "family": {
            "family_id": family.family_id,
            "display_name": family.display_name,
            "status": family.status,
            "model_family": family.model_family,
            "training_mode": family.training_mode,
            "implementation_root": family.implementation_root,
            "readme_path": family.readme_path,
            "runner_entrypoint": family.runner_entrypoint,
            "orchestrator_entrypoint": family.orchestrator_entrypoint,
            "default_output_path": family.default_output_path,
            "default_output_schema_version": family.default_output_schema_version,
        },
        "invocation": {
            "kind": invocation_kind,
            "entrypoint": entrypoint,
            "run_label": run_label,
        },
        "selection": {
            "dataset_ids": list(dict.fromkeys(dataset_ids)),
            "feature_protocol_ids": list(resolved_protocol_ids),
            "model_variants": list(dict.fromkeys(model_variants)),
            "eval_protocols": list(dict.fromkeys(eval_protocols)),
            "result_splits": list(dict.fromkeys(result_splits)),
        },
        "artifacts": {
            "primary_output": output_artifact.__dict__,
            **artifact_payload,
        },
        "result": {
            "row_count": result_row_count,
        },
        "args": _jsonable(args),
        "git": _capture_git_metadata(resolved_repo_root),
        "notes": list(notes),
    }
    manifest_path = run_dir / "manifest.json"
    manifest_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest_path


def describe_output_artifact(path: Path, *, repo_root: Path) -> OutputArtifact:
    resolved = path.resolve()
    if not resolved.exists():
        return OutputArtifact(
            path=_display_path(resolved, repo_root=repo_root),
            exists=False,
            size_bytes=None,
            sha256=None,
        )
    return OutputArtifact(
        path=_display_path(resolved, repo_root=repo_root),
        exists=True,
        size_bytes=resolved.stat().st_size,
        sha256=_sha256_file(resolved),
    )


def _allocate_run_directory(*, run_root: Path, family_id: str, run_label: str | None) -> Path:
    family_root = run_root / family_id
    family_root.mkdir(parents=True, exist_ok=True)
    stem = datetime.now(tz=UTC).strftime("%Y%m%d-%H%M%S")
    if run_label:
        stem = f"{stem}-{_sanitize_path_component(run_label)}"
    candidate = family_root / stem
    suffix = 1
    while candidate.exists():
        candidate = family_root / f"{stem}-{suffix:02d}"
        suffix += 1
    candidate.mkdir(parents=True, exist_ok=False)
    return candidate


def _sanitize_path_component(value: str) -> str:
    normalized = "".join(character if character.isalnum() or character in "-_." else "-" for character in value.strip())
    collapsed = "-".join(part for part in normalized.split("-") if part)
    return collapsed or "run"


def _display_path(path: Path, *, repo_root: Path) -> str:
    try:
        return str(path.resolve().relative_to(repo_root.resolve()))
    except ValueError:
        return str(path.resolve())


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, datetime):
        if value.tzinfo is None:
            return value.isoformat()
        return value.astimezone(UTC).isoformat()
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _capture_git_metadata(repo_root: Path) -> dict[str, Any]:
    return {
        "head": _run_git(repo_root, "rev-parse", "HEAD"),
        "branch": _run_git(repo_root, "rev-parse", "--abbrev-ref", "HEAD"),
        "is_dirty": bool(_run_git_lines(repo_root, "status", "--short")),
        "status_short": _run_git_lines(repo_root, "status", "--short"),
    }


def _run_git(repo_root: Path, *args: str) -> str | None:
    try:
        completed = subprocess.run(
            ["git", *args],
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None
    value = completed.stdout.strip()
    return value or None


def _run_git_lines(repo_root: Path, *args: str) -> list[str]:
    result = _run_git(repo_root, *args)
    if result is None:
        return []
    return [line for line in result.splitlines() if line]


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()
