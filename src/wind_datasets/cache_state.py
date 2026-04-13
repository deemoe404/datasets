from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Sequence

from .models import DatasetSpec, ResolvedTaskSpec
from .paths import DatasetCachePaths
from .utils import ensure_directory, read_json, write_json

CACHE_STATE_SCHEMA_VERSION = "v2"

_IGNORED_FILE_NAMES = {".DS_Store"}
_IGNORED_SUFFIXES = {".swp", ".webloc"}
_PACKAGE_ROOT = Path(__file__).resolve().parent
_HANDLER_FILES = {
    "greenbyte": "datasets/greenbyte.py",
    "hill_of_towie": "datasets/hill_of_towie.py",
    "sdwpf_kddcup": "datasets/sdwpf_kddcup.py",
}
_PACKAGED_LAYER_DEPENDENCIES = {
    ("silver", "hill_of_towie"): ("data/hill_of_towie_tuneup_2024.csv",),
}
_COMMON_FILES = (
    "datasets/base.py",
    "datasets/common.py",
    "feature_protocols.py",
    "models.py",
    "source_column_policy.py",
    "source_schema.py",
    "utils.py",
    "paths.py",
)


@dataclass(frozen=True)
class LayerBuildMeta:
    layer: str
    dataset_id: str
    fingerprint: str
    code_fingerprint: str
    parent_fingerprint: str | None
    spec_fingerprint: str
    params: dict[str, Any]
    built_at: str
    schema_version: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "LayerBuildMeta":
        return cls(
            layer=str(payload["layer"]),
            dataset_id=str(payload["dataset_id"]),
            fingerprint=str(payload["fingerprint"]),
            code_fingerprint=str(payload["code_fingerprint"]),
            parent_fingerprint=(
                str(payload["parent_fingerprint"])
                if payload.get("parent_fingerprint") is not None
                else None
            ),
            spec_fingerprint=str(payload["spec_fingerprint"]),
            params=dict(payload.get("params") or {}),
            built_at=str(payload["built_at"]),
            schema_version=str(payload["schema_version"]),
        )


@dataclass(frozen=True)
class LayerStatus:
    status: str
    reason: str | None
    fingerprint: str | None


def hash_json(payload: Any) -> str:
    text = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def hash_files(paths: Sequence[Path]) -> str:
    digest = hashlib.sha256()
    for path in sorted(paths):
        digest.update(str(path.relative_to(_PACKAGE_ROOT)).encode("utf-8"))
        digest.update(b"\0")
        with path.open("rb") as handle:
            while chunk := handle.read(1024 * 1024):
                digest.update(chunk)
        digest.update(b"\0")
    return digest.hexdigest()


def code_fingerprint_for(layer: str, handler: str) -> str:
    files: list[Path]
    if layer == "manifest":
        files = [
            _PACKAGE_ROOT / "manifest.py",
            _PACKAGE_ROOT / "registry.py",
            _PACKAGE_ROOT / "models.py",
            _PACKAGE_ROOT / "source_schema.py",
            _PACKAGE_ROOT / "utils.py",
            _PACKAGE_ROOT / "paths.py",
        ]
    elif layer in {"silver", "gold_base"}:
        files = [_PACKAGE_ROOT / relative for relative in _COMMON_FILES]
        files.append(_PACKAGE_ROOT / _HANDLER_FILES[handler])
    elif layer == "task":
        files = [_PACKAGE_ROOT / relative for relative in _COMMON_FILES]
    else:
        raise ValueError(f"Unsupported cache layer {layer!r}.")
    return hash_files(files)


def packaged_dependency_fingerprint_for(layer: str, spec: DatasetSpec) -> str | None:
    relative_paths = list(_PACKAGED_LAYER_DEPENDENCIES.get((layer, spec.handler), ()))
    if layer == "silver":
        relative_paths.append(f"data/source_column_policy/{spec.dataset_id}.csv")
    if layer == "gold_base":
        relative_paths.append(f"data/source_column_policy/{spec.dataset_id}.csv")
    if not relative_paths:
        return None
    files = [_PACKAGE_ROOT / relative for relative in relative_paths]
    return hash_files(files)


def spec_fingerprint_for(spec: DatasetSpec) -> str:
    payload = {
        "dataset_id": spec.dataset_id,
        "handler": spec.handler,
        "resolution_minutes": spec.resolution_minutes,
        "target_column": spec.target_column,
        "timezone_policy": spec.timezone_policy,
        "timestamp_convention": spec.timestamp_convention,
        "default_feature_groups": list(spec.default_feature_groups),
        "default_quality_profile": spec.default_quality_profile,
        "default_expected_release_id": spec.default_expected_release_id,
        "official_assets": list(spec.official_assets),
        "default_ingested_assets": list(spec.default_ingested_assets),
        "default_excluded_assets": list(spec.default_excluded_assets),
        "turbine_ids": list(spec.turbine_ids),
    }
    return hash_json(payload)


def source_snapshot_for(spec: DatasetSpec) -> list[dict[str, object]]:
    if not spec.source_root.exists():
        return []
    paths: list[Path] = []
    for path in sorted(spec.source_root.rglob("*")):
        if not path.is_file():
            continue
        if path.name in _IGNORED_FILE_NAMES:
            continue
        if path.suffix.lower() in _IGNORED_SUFFIXES:
            continue
        paths.append(path)
    return [source_file_record_for(path, spec.source_root) for path in paths]


def source_file_record_for(path: Path, root: Path) -> dict[str, object]:
    stat_result = path.stat()
    return {
        "relative_path": str(path.relative_to(root)),
        "size_bytes": stat_result.st_size,
        "created_ns": _created_ns_for_stat(stat_result),
    }


def _created_ns_for_stat(stat_result: os.stat_result) -> int:
    birthtime_ns = getattr(stat_result, "st_birthtime_ns", None)
    if birthtime_ns is not None:
        return int(birthtime_ns)
    birthtime = getattr(stat_result, "st_birthtime", None)
    if birthtime is not None:
        return int(float(birthtime) * 1_000_000_000)
    return int(stat_result.st_ctime_ns)


def build_layer_fingerprint(
    *,
    layer: str,
    dataset_id: str,
    code_fingerprint: str,
    parent_fingerprint: str | None,
    spec_fingerprint: str,
    params: dict[str, Any],
    schema_version: str = CACHE_STATE_SCHEMA_VERSION,
) -> str:
    return hash_json(
        {
            "schema_version": schema_version,
            "layer": layer,
            "dataset_id": dataset_id,
            "code_fingerprint": code_fingerprint,
            "parent_fingerprint": parent_fingerprint,
            "spec_fingerprint": spec_fingerprint,
            "params": params,
        }
    )


def build_meta_path_for(
    cache_paths: DatasetCachePaths,
    layer: str,
    *,
    task_id: str | None = None,
    feature_protocol_id: str | None = None,
) -> Path:
    if layer == "manifest":
        return cache_paths.manifest_build_meta_path
    if layer == "silver":
        return cache_paths.silver_build_meta_path
    if layer == "gold_base":
        return cache_paths.gold_base_build_meta_path
    if layer == "task":
        assert task_id is not None
        assert feature_protocol_id is not None
        return cache_paths.task_build_meta_path_for(task_id, feature_protocol_id)
    raise ValueError(f"Unsupported cache layer {layer!r}.")


def read_build_meta(path: Path) -> LayerBuildMeta | None:
    if not path.exists():
        return None
    try:
        payload = read_json(path)
    except (json.JSONDecodeError, OSError):
        return None
    if not isinstance(payload, dict):
        return None
    try:
        return LayerBuildMeta.from_dict(payload)
    except (KeyError, TypeError, ValueError):
        return None


def write_build_meta(path: Path, meta: LayerBuildMeta) -> Path:
    ensure_directory(path.parent)
    return write_json(path, meta.to_dict())


def expected_manifest_meta(spec: DatasetSpec) -> LayerBuildMeta:
    snapshot_fingerprint = hash_json(source_snapshot_for(spec))
    params = {"source_snapshot_fingerprint": snapshot_fingerprint}
    code_fingerprint = code_fingerprint_for("manifest", spec.handler)
    spec_fingerprint = spec_fingerprint_for(spec)
    return _make_meta(
        layer="manifest",
        dataset_id=spec.dataset_id,
        code_fingerprint=code_fingerprint,
        parent_fingerprint=None,
        spec_fingerprint=spec_fingerprint,
        params=params,
    )


def manifest_meta_from_payload(spec: DatasetSpec, payload: dict[str, Any]) -> LayerBuildMeta:
    snapshot = [
        {
            "relative_path": item["relative_path"],
            "size_bytes": item["size_bytes"],
            "created_ns": item.get("created_ns", item.get("mtime_ns")),
        }
        for item in payload.get("files", [])
    ]
    params = {"source_snapshot_fingerprint": hash_json(snapshot)}
    code_fingerprint = code_fingerprint_for("manifest", spec.handler)
    spec_fingerprint = spec_fingerprint_for(spec)
    built_at = str(payload.get("generated_at") or _now_iso())
    return _make_meta(
        layer="manifest",
        dataset_id=spec.dataset_id,
        code_fingerprint=code_fingerprint,
        parent_fingerprint=None,
        spec_fingerprint=spec_fingerprint,
        params=params,
        built_at=built_at,
    )


def expected_silver_meta(spec: DatasetSpec) -> LayerBuildMeta:
    manifest_meta = expected_manifest_meta(spec)
    params: dict[str, Any] = {}
    packaged_dependency_fingerprint = packaged_dependency_fingerprint_for("silver", spec)
    if packaged_dependency_fingerprint is not None:
        params["packaged_dependency_fingerprint"] = packaged_dependency_fingerprint
    return _make_meta(
        layer="silver",
        dataset_id=spec.dataset_id,
        code_fingerprint=code_fingerprint_for("silver", spec.handler),
        parent_fingerprint=manifest_meta.fingerprint,
        spec_fingerprint=spec_fingerprint_for(spec),
        params=params,
    )


def expected_gold_base_meta(
    spec: DatasetSpec,
) -> LayerBuildMeta:
    silver_meta = expected_silver_meta(spec)
    params: dict[str, Any] = {}
    packaged_dependency_fingerprint = packaged_dependency_fingerprint_for("gold_base", spec)
    if packaged_dependency_fingerprint is not None:
        params["packaged_dependency_fingerprint"] = packaged_dependency_fingerprint
    return _make_meta(
        layer="gold_base",
        dataset_id=spec.dataset_id,
        code_fingerprint=code_fingerprint_for("gold_base", spec.handler),
        parent_fingerprint=silver_meta.fingerprint,
        spec_fingerprint=spec_fingerprint_for(spec),
        params=params,
    )


def expected_task_meta(
    spec: DatasetSpec,
    *,
    task: ResolvedTaskSpec,
    feature_protocol_id: str,
) -> LayerBuildMeta:
    gold_meta = expected_gold_base_meta(spec)
    return _make_meta(
        layer="task",
        dataset_id=spec.dataset_id,
        code_fingerprint=code_fingerprint_for("task", spec.handler),
        parent_fingerprint=gold_meta.fingerprint,
        spec_fingerprint=spec_fingerprint_for(spec),
        params={
            "task": task.to_dict(),
            "feature_protocol_id": feature_protocol_id,
        },
    )


def check_manifest_status(spec: DatasetSpec, cache_paths: DatasetCachePaths) -> LayerStatus:
    return _check_layer_status(
        expected=expected_manifest_meta(spec),
        actual=read_build_meta(cache_paths.manifest_build_meta_path),
        required_outputs=(cache_paths.manifest_path,),
        layer="manifest",
    )


def check_silver_status(
    spec: DatasetSpec,
    cache_paths: DatasetCachePaths,
    *,
    required_outputs: Sequence[Path],
) -> LayerStatus:
    return _check_layer_status(
        expected=expected_silver_meta(spec),
        actual=read_build_meta(cache_paths.silver_build_meta_path),
        required_outputs=required_outputs,
        layer="silver",
    )


def check_gold_base_status(
    spec: DatasetSpec,
    cache_paths: DatasetCachePaths,
    *,
    blocked_reason: str | None = None,
) -> LayerStatus:
    expected = expected_gold_base_meta(spec)
    return _check_layer_status(
        expected=expected,
        actual=read_build_meta(cache_paths.gold_base_build_meta_path),
        required_outputs=(
            cache_paths.gold_base_series_path,
            cache_paths.gold_base_quality_path,
        ),
        layer="gold_base",
        blocked_reason=blocked_reason,
    )


def check_task_status(
    spec: DatasetSpec,
    cache_paths: DatasetCachePaths,
    *,
    task: ResolvedTaskSpec,
    feature_protocol_id: str,
    blocked_reason: str | None = None,
) -> LayerStatus:
    required_outputs = [
        cache_paths.task_series_path_for(task.task_id, feature_protocol_id),
        cache_paths.task_known_future_path_for(task.task_id, feature_protocol_id),
        cache_paths.task_turbine_static_path_for(task.task_id, feature_protocol_id),
        cache_paths.task_window_index_path_for(task.task_id, feature_protocol_id),
        cache_paths.task_context_path_for(task.task_id, feature_protocol_id),
    ]
    expected = expected_task_meta(
        spec,
        task=task,
        feature_protocol_id=feature_protocol_id,
    )
    return _check_layer_status(
        expected=expected,
        actual=read_build_meta(cache_paths.task_build_meta_path_for(task.task_id, feature_protocol_id)),
        required_outputs=tuple(required_outputs),
        layer="task",
        blocked_reason=blocked_reason,
    )


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _make_meta(
    *,
    layer: str,
    dataset_id: str,
    code_fingerprint: str,
    parent_fingerprint: str | None,
    spec_fingerprint: str,
    params: dict[str, Any],
    built_at: str | None = None,
) -> LayerBuildMeta:
    fingerprint = build_layer_fingerprint(
        layer=layer,
        dataset_id=dataset_id,
        code_fingerprint=code_fingerprint,
        parent_fingerprint=parent_fingerprint,
        spec_fingerprint=spec_fingerprint,
        params=params,
    )
    return LayerBuildMeta(
        layer=layer,
        dataset_id=dataset_id,
        fingerprint=fingerprint,
        code_fingerprint=code_fingerprint,
        parent_fingerprint=parent_fingerprint,
        spec_fingerprint=spec_fingerprint,
        params=params,
        built_at=built_at or _now_iso(),
        schema_version=CACHE_STATE_SCHEMA_VERSION,
    )


def _check_layer_status(
    *,
    expected: LayerBuildMeta,
    actual: LayerBuildMeta | None,
    required_outputs: Sequence[Path],
    layer: str,
    blocked_reason: str | None = None,
) -> LayerStatus:
    fingerprint = actual.fingerprint if actual is not None else expected.fingerprint
    if blocked_reason is not None:
        return LayerStatus(status="stale", reason=blocked_reason, fingerprint=fingerprint)
    if any(not path.exists() for path in required_outputs):
        return LayerStatus(status="missing", reason="missing_output", fingerprint=fingerprint)
    if actual is None:
        return LayerStatus(status="stale", reason="missing_build_meta", fingerprint=fingerprint)
    if actual.schema_version != expected.schema_version:
        return LayerStatus(status="stale", reason="params_changed", fingerprint=fingerprint)
    if actual.code_fingerprint != expected.code_fingerprint:
        return LayerStatus(status="stale", reason="code_fingerprint_changed", fingerprint=fingerprint)
    if actual.parent_fingerprint != expected.parent_fingerprint:
        return LayerStatus(status="stale", reason="parent_fingerprint_changed", fingerprint=fingerprint)
    if actual.spec_fingerprint != expected.spec_fingerprint:
        return LayerStatus(status="stale", reason="params_changed", fingerprint=fingerprint)
    if actual.params != expected.params:
        reason = "source_snapshot_changed" if layer == "manifest" else "params_changed"
        return LayerStatus(status="stale", reason=reason, fingerprint=fingerprint)
    if actual.fingerprint != expected.fingerprint:
        return LayerStatus(status="stale", reason="params_changed", fingerprint=fingerprint)
    return LayerStatus(status="fresh", reason=None, fingerprint=fingerprint)
