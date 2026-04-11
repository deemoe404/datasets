from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import polars as pl

from .source_schema import normalize_source_column_name, normalize_source_frame

_PACKAGE_ROOT = Path(__file__).resolve().parent
_POLICY_DIR = _PACKAGE_ROOT / "data" / "source_column_policy"
_ALLOWED_DECISIONS = {"drop", "keep", "keep+mask"}
_LEGACY_DECISION_ALIASES = {"mask+keep": "keep+mask"}


@dataclass(frozen=True)
class SourceColumnPolicyEntry:
    source_asset: str
    source_table_or_file: str
    source_column: str
    decision: str
    mask_rule: str
    canonical_outputs: tuple[str, ...]
    required_for_protocols: tuple[str, ...]
    reason: str
    notes: str

    @property
    def lookup_key(self) -> tuple[str, str, str]:
        return (self.source_asset, self.source_table_or_file, self.source_column)


@dataclass(frozen=True)
class SourceColumnPolicy:
    dataset_id: str
    path: Path
    relative_path: str
    entries: tuple[SourceColumnPolicyEntry, ...]

    @property
    def decision_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for entry in self.entries:
            counts[entry.decision] = counts.get(entry.decision, 0) + 1
        return counts

    def entries_for_source(self, source_asset: str, source_table_or_file: str) -> tuple[SourceColumnPolicyEntry, ...]:
        return tuple(
            entry
            for entry in self.entries
            if entry.source_asset == source_asset and entry.source_table_or_file == source_table_or_file
        )

    def entry_map_for_source(self, source_asset: str, source_table_or_file: str) -> dict[str, SourceColumnPolicyEntry]:
        return {
            entry.source_column: entry
            for entry in self.entries_for_source(source_asset, source_table_or_file)
        }


def _split_csv_list(value: str | None) -> tuple[str, ...]:
    if value is None:
        return ()
    items = [item.strip() for item in value.split("|")]
    return tuple(item for item in items if item)


def normalize_decision(raw_value: str) -> str:
    value = raw_value.strip()
    normalized = _LEGACY_DECISION_ALIASES.get(value, value)
    if normalized not in _ALLOWED_DECISIONS:
        raise ValueError(
            f"Unsupported source column policy decision {raw_value!r}. Expected one of {_ALLOWED_DECISIONS!r}."
        )
    return normalized


def load_source_column_policy(dataset_id: str) -> SourceColumnPolicy:
    path = _POLICY_DIR / f"{dataset_id}.csv"
    if not path.exists():
        raise FileNotFoundError(f"Source column policy for dataset {dataset_id!r} is missing: {path}")
    frame = pl.read_csv(path, infer_schema_length=0)
    required_columns = {
        "source_asset",
        "source_table_or_file",
        "source_column",
        "decision",
        "mask_rule",
        "canonical_outputs",
        "required_for_protocols",
        "reason",
        "notes",
    }
    missing = sorted(required_columns - set(frame.columns))
    if missing:
        raise ValueError(f"{path}: missing required columns {missing!r}.")

    seen: set[tuple[str, str, str]] = set()
    entries: list[SourceColumnPolicyEntry] = []
    for row in frame.iter_rows(named=True):
        entry = SourceColumnPolicyEntry(
            source_asset=str(row["source_asset"]).strip(),
            source_table_or_file=str(row["source_table_or_file"]).strip(),
            source_column=str(row["source_column"]).strip(),
            decision=normalize_decision(str(row["decision"])),
            mask_rule=str(row["mask_rule"] or "").strip(),
            canonical_outputs=_split_csv_list(str(row["canonical_outputs"] or "")),
            required_for_protocols=_split_csv_list(str(row["required_for_protocols"] or "")),
            reason=str(row["reason"] or "").strip(),
            notes=str(row["notes"] or "").strip(),
        )
        if not entry.source_asset or not entry.source_table_or_file or not entry.source_column:
            raise ValueError(f"{path}: source policy rows must have non-empty source identity fields.")
        if entry.lookup_key in seen:
            raise ValueError(f"{path}: duplicate source policy row for {entry.lookup_key!r}.")
        seen.add(entry.lookup_key)
        entries.append(entry)
    return SourceColumnPolicy(
        dataset_id=dataset_id,
        path=path,
        relative_path=str(path.relative_to(_PACKAGE_ROOT)),
        entries=tuple(entries),
    )


def validate_policy_coverage(
    policy: SourceColumnPolicy,
    inventory_rows: Iterable[dict[str, object]],
) -> None:
    inventory_keys: set[tuple[str, str, str]] = set()
    for row in inventory_rows:
        source_column = normalize_source_column_name(row["source_column"])
        if not source_column:
            continue
        inventory_keys.add(
            (
                str(row["source_asset"]).strip(),
                str(row["source_table_or_file"]).strip(),
                source_column,
            )
        )
    missing = sorted(inventory_keys - {entry.lookup_key for entry in policy.entries})
    if missing:
        raise ValueError(
            f"Source column policy {policy.relative_path} does not cover inventory keys {missing[:10]!r}."
        )


def kept_source_columns(
    policy: SourceColumnPolicy,
    *,
    source_asset: str,
    source_table_or_file: str,
    always_keep: Iterable[str] = (),
) -> tuple[str, ...]:
    entry_map = policy.entry_map_for_source(source_asset, source_table_or_file)
    columns = [column for column in always_keep]
    for column, entry in entry_map.items():
        if entry.decision != "drop" and column not in columns:
            columns.append(column)
    return tuple(columns)


def filter_source_frame(
    frame: pl.DataFrame,
    *,
    policy: SourceColumnPolicy,
    source_asset: str,
    source_table_or_file: str,
    always_keep: Iterable[str] = (),
) -> pl.DataFrame:
    frame = normalize_source_frame(frame, drop_empty=True)
    keep = [column for column in kept_source_columns(
        policy,
        source_asset=source_asset,
        source_table_or_file=source_table_or_file,
        always_keep=always_keep,
    ) if column in frame.columns]
    return frame.select(keep)


def apply_keep_mask_rules(
    frame: pl.DataFrame,
    *,
    policy: SourceColumnPolicy,
    source_asset: str,
    source_table_or_file: str,
) -> pl.DataFrame:
    frame = normalize_source_frame(frame, drop_empty=True)
    entry_map = policy.entry_map_for_source(source_asset, source_table_or_file)
    expressions: list[pl.Expr] = []
    for column, entry in entry_map.items():
        if entry.decision != "keep+mask" or column not in frame.columns:
            continue
        if entry.mask_rule == "sdwpf_patv_non_positive_wspd_gt_2_5":
            expressions.append(
                pl.when(
                    pl.col(column).cast(pl.Float64, strict=False).le(0)
                    & pl.col("Wspd").cast(pl.Float64, strict=False).gt(2.5)
                )
                .then(pl.lit(None))
                .otherwise(pl.col(column))
                .alias(column)
            )
        elif entry.mask_rule == "sdwpf_wdir_out_of_range":
            expressions.append(
                pl.when(
                    pl.col(column).cast(pl.Float64, strict=False).lt(-180)
                    | pl.col(column).cast(pl.Float64, strict=False).gt(180)
                )
                .then(pl.lit(None))
                .otherwise(pl.col(column))
                .alias(column)
            )
        elif entry.mask_rule == "sdwpf_ndir_out_of_range":
            expressions.append(
                pl.when(
                    pl.col(column).cast(pl.Float64, strict=False).lt(-720)
                    | pl.col(column).cast(pl.Float64, strict=False).gt(720)
                )
                .then(pl.lit(None))
                .otherwise(pl.col(column))
                .alias(column)
            )
        elif entry.mask_rule == "sdwpf_pitch_gt_89":
            expressions.append(
                pl.when(pl.col(column).cast(pl.Float64, strict=False).gt(89))
                .then(pl.lit(None))
                .otherwise(pl.col(column))
                .alias(column)
            )
    if not expressions:
        return frame
    return frame.with_columns(expressions)


__all__ = [
    "SourceColumnPolicy",
    "SourceColumnPolicyEntry",
    "apply_keep_mask_rules",
    "filter_source_frame",
    "kept_source_columns",
    "load_source_column_policy",
    "normalize_decision",
    "validate_policy_coverage",
]
