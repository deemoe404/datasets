from __future__ import annotations

import hashlib
import json
import re
from datetime import timedelta
from pathlib import Path
from typing import Any, Iterable

_DURATION_PATTERN = re.compile(r"(?P<value>\d+)(?P<unit>[smhd])")
_UNIT_SECONDS = {"s": 1, "m": 60, "h": 3600, "d": 86400}


def parse_duration(value: timedelta | str | int | float) -> timedelta:
    if isinstance(value, timedelta):
        return value
    if isinstance(value, (int, float)):
        return timedelta(seconds=float(value))
    if not isinstance(value, str):
        raise TypeError(f"Unsupported duration type: {type(value)!r}")

    text = value.strip().lower()
    if not text:
        raise ValueError("Duration text must not be empty.")

    total_seconds = 0
    position = 0
    for match in _DURATION_PATTERN.finditer(text):
        if match.start() != position:
            raise ValueError(f"Invalid duration segment in {value!r}.")
        total_seconds += int(match.group("value")) * _UNIT_SECONDS[match.group("unit")]
        position = match.end()

    if position != len(text):
        raise ValueError(f"Invalid duration value {value!r}.")
    return timedelta(seconds=total_seconds)


def format_duration(value: timedelta | str | int | float | None) -> str:
    if value is None:
        return "dataset_resolution"

    duration = parse_duration(value)
    seconds = int(duration.total_seconds())
    if seconds % 86400 == 0:
        return f"{seconds // 86400}d"
    if seconds % 3600 == 0:
        return f"{seconds // 3600}h"
    if seconds % 60 == 0:
        return f"{seconds // 60}m"
    return f"{seconds}s"


def ensure_directory(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def duration_to_steps(duration: timedelta, resolution_minutes: int) -> int:
    resolution_seconds = resolution_minutes * 60
    duration_seconds = int(duration.total_seconds())
    if duration_seconds % resolution_seconds != 0:
        raise ValueError(
            f"Duration {format_duration(duration)} is not divisible by "
            f"{resolution_minutes} minutes."
        )
    return duration_seconds // resolution_seconds


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: Any) -> Path:
    ensure_directory(path.parent)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n")
    return path


def read_json(path: Path) -> Any:
    return json.loads(path.read_text())


def join_flags(*values: str | None) -> str:
    ordered: list[str] = []
    seen: set[str] = set()
    for value in values:
        if not value:
            continue
        for flag in value.split("|"):
            if flag and flag not in seen:
                ordered.append(flag)
                seen.add(flag)
    return "|".join(ordered)


def maybe_float(value: str | None) -> float | None:
    if value is None:
        return None
    text = value.strip()
    if text in {"", "NaN", "-", "null", "None"}:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def stringify_path_list(paths: Iterable[Path], root: Path) -> list[str]:
    return [str(path.relative_to(root)) for path in sorted(paths)]
