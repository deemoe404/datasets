from __future__ import annotations

from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path

from .models import DatasetSpec
from .paths import dataset_cache_paths
from .utils import ensure_directory, sha256_file, write_json

_IGNORED_FILE_NAMES = {".DS_Store"}
_IGNORED_SUFFIXES = {".swp", ".webloc"}
_BUILD_VERSION = "v1"


def _iter_source_files(root: Path) -> list[Path]:
    paths: list[Path] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        if path.name in _IGNORED_FILE_NAMES:
            continue
        if path.suffix.lower() in _IGNORED_SUFFIXES:
            continue
        paths.append(path)
    return paths


def build_manifest(spec: DatasetSpec, cache_root: Path) -> Path:
    cache_paths = dataset_cache_paths(cache_root, spec.dataset_id)
    ensure_directory(cache_paths.manifest_dir)
    source_files = _iter_source_files(spec.source_root)
    payload = {
        "dataset_id": spec.dataset_id,
        "build_version": _BUILD_VERSION,
        "source_root": str(spec.source_root),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "dependencies": {
            name: metadata.version(name)
            for name in ("polars", "pyarrow", "duckdb", "pytest")
        },
        "files": [
            {
                "relative_path": str(path.relative_to(spec.source_root)),
                "size_bytes": path.stat().st_size,
                "mtime_ns": path.stat().st_mtime_ns,
                "sha256": sha256_file(path),
            }
            for path in source_files
        ],
    }
    return write_json(cache_paths.manifest_path, payload)
