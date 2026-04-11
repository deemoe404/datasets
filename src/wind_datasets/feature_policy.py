from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
import re
from typing import Any, Sequence

import polars as pl

_PACKAGE_ROOT = Path(__file__).resolve().parent
_FEATURE_POLICY_DIR = _PACKAGE_ROOT / "data" / "feature_policy"
_FEATURE_POLICY_COLUMNS = ("feature_name", "decision", "mask_rule", "reason", "notes")
_VALID_DECISIONS = frozenset({"keep", "mask+keep", "drop"})
_VALID_MASK_OPERATORS = frozenset({"eq", "lt", "le", "gt", "ge"})
_INTEGER_PATTERN = re.compile(r"^[+-]?\d+$")
_FLOAT_PATTERN = re.compile(r"^[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?$")

SERIES_NON_FEATURE_COLUMNS = frozenset(
    {
        "dataset",
        "turbine_id",
        "timestamp",
        "target_kw",
        "is_observed",
        "quality_flags",
        "feature_quality_flags",
        "farm_turbines_expected",
        "farm_turbines_observed",
        "farm_turbines_with_target",
        "farm_is_fully_synchronous",
        "farm_has_all_targets",
    }
)


@dataclass(frozen=True)
class MaskPredicate:
    op: str
    value: Any


@dataclass(frozen=True)
class FeaturePolicyEntry:
    feature_name: str
    decision: str
    mask_rule: str
    mask_predicates: tuple[MaskPredicate, ...]
    reason: str
    notes: str


@dataclass(frozen=True)
class FeaturePolicy:
    dataset_id: str
    path: Path
    relative_path: str
    entries: tuple[FeaturePolicyEntry, ...]

    @property
    def feature_names(self) -> tuple[str, ...]:
        return tuple(entry.feature_name for entry in self.entries)

    @property
    def decision_counts(self) -> dict[str, int]:
        counts = {decision: 0 for decision in _VALID_DECISIONS}
        for entry in self.entries:
            counts[entry.decision] += 1
        return counts


def feature_policy_path_for(dataset_id: str) -> Path:
    return _FEATURE_POLICY_DIR / f"{dataset_id}.csv"


def feature_policy_relative_path_for(dataset_id: str) -> str:
    path = feature_policy_path_for(dataset_id)
    try:
        return str(path.relative_to(_PACKAGE_ROOT))
    except ValueError:
        return str(path)


def series_candidate_feature_columns(columns: Sequence[str]) -> tuple[str, ...]:
    return tuple(column for column in columns if column not in SERIES_NON_FEATURE_COLUMNS)


def load_feature_policy(
    dataset_id: str,
    expected_features: Sequence[str],
    *,
    allow_extra_features: bool = False,
) -> FeaturePolicy:
    path = feature_policy_path_for(dataset_id)
    if not path.exists():
        raise FileNotFoundError(f"Feature policy file does not exist for dataset {dataset_id!r}: {path}")

    expected = tuple(expected_features)
    duplicate_expected = sorted({feature for feature in expected if expected.count(feature) > 1})
    if duplicate_expected:
        raise ValueError(f"Expected feature set for dataset {dataset_id!r} contains duplicates: {duplicate_expected!r}.")
    expected_set = set(expected)

    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != _FEATURE_POLICY_COLUMNS:
            raise ValueError(
                f"Feature policy {path} must have columns {_FEATURE_POLICY_COLUMNS!r}, "
                f"got {tuple(reader.fieldnames or ())!r}."
            )
        entries = tuple(_parse_policy_row(path, row_number, row) for row_number, row in enumerate(reader, start=2))

    seen: set[str] = set()
    duplicate_csv_features: list[str] = []
    for entry in entries:
        if entry.feature_name in seen:
            duplicate_csv_features.append(entry.feature_name)
        seen.add(entry.feature_name)
    if duplicate_csv_features:
        raise ValueError(
            f"Feature policy {path} for dataset {dataset_id!r} contains duplicate rows for "
            f"{sorted(set(duplicate_csv_features))!r}."
        )

    csv_features = set(seen)
    missing = sorted(expected_set - csv_features)
    extra = sorted(csv_features - expected_set) if not allow_extra_features else []
    if missing or extra:
        detail_parts: list[str] = []
        if missing:
            detail_parts.append(f"missing features {missing!r}")
        if extra:
            detail_parts.append(f"unknown features {extra!r}")
        raise ValueError(
            f"Feature policy {path} does not match the candidate feature universe for "
            f"dataset {dataset_id!r}: {', '.join(detail_parts)}."
        )

    return FeaturePolicy(
        dataset_id=dataset_id,
        path=path,
        relative_path=feature_policy_relative_path_for(dataset_id),
        entries=entries,
    )


def apply_feature_policy(frame: pl.DataFrame, policy: FeaturePolicy) -> pl.DataFrame:
    mask_expressions: list[pl.Expr] = []
    columns_to_drop: list[str] = []

    for entry in policy.entries:
        if entry.feature_name not in frame.columns:
            continue
        if entry.decision == "drop":
            columns_to_drop.append(entry.feature_name)
            continue
        if entry.decision == "mask+keep":
            mask_expressions.append(
                pl.when(_mask_expression(entry.feature_name, entry.mask_predicates))
                .then(pl.lit(None, dtype=frame.schema[entry.feature_name]))
                .otherwise(pl.col(entry.feature_name))
                .alias(entry.feature_name)
            )

    if mask_expressions:
        frame = frame.with_columns(mask_expressions)
    if columns_to_drop:
        frame = frame.drop(columns_to_drop)
    return frame


def _parse_policy_row(
    path: Path,
    row_number: int,
    row: dict[str, str | None],
) -> FeaturePolicyEntry:
    feature_name = str(row.get("feature_name") or "").strip()
    if not feature_name:
        raise ValueError(f"Feature policy {path} row {row_number} is missing feature_name.")

    decision = str(row.get("decision") or "").strip()
    if decision not in _VALID_DECISIONS:
        raise ValueError(
            f"Feature policy {path} row {row_number} has invalid decision {decision!r}. "
            f"Expected one of {sorted(_VALID_DECISIONS)!r}."
        )

    mask_rule = str(row.get("mask_rule") or "").strip()
    if decision == "mask+keep":
        if not mask_rule:
            raise ValueError(
                f"Feature policy {path} row {row_number} for feature {feature_name!r} requires a mask_rule."
            )
        mask_predicates = _parse_mask_rule(path, row_number, feature_name, mask_rule)
    else:
        if mask_rule:
            raise ValueError(
                f"Feature policy {path} row {row_number} for feature {feature_name!r} must leave mask_rule empty "
                f"when decision={decision!r}."
            )
        mask_predicates = ()

    reason = str(row.get("reason") or "").strip()
    if not reason:
        raise ValueError(f"Feature policy {path} row {row_number} for feature {feature_name!r} is missing reason.")
    notes = str(row.get("notes") or "").strip()

    return FeaturePolicyEntry(
        feature_name=feature_name,
        decision=decision,
        mask_rule=mask_rule,
        mask_predicates=mask_predicates,
        reason=reason,
        notes=notes,
    )


def _parse_mask_rule(
    path: Path,
    row_number: int,
    feature_name: str,
    mask_rule: str,
) -> tuple[MaskPredicate, ...]:
    predicates: list[MaskPredicate] = []
    for raw_clause in mask_rule.split("|"):
        clause = raw_clause.strip()
        if not clause:
            raise ValueError(
                f"Feature policy {path} row {row_number} for feature {feature_name!r} contains an empty mask clause."
            )
        op, separator, raw_value = clause.partition(":")
        if separator != ":":
            raise ValueError(
                f"Feature policy {path} row {row_number} for feature {feature_name!r} has invalid mask clause "
                f"{clause!r}. Expected op:value."
            )
        op = op.strip()
        if op not in _VALID_MASK_OPERATORS:
            raise ValueError(
                f"Feature policy {path} row {row_number} for feature {feature_name!r} has invalid mask operator "
                f"{op!r}. Expected one of {sorted(_VALID_MASK_OPERATORS)!r}."
            )
        raw_value = raw_value.strip()
        if not raw_value:
            raise ValueError(
                f"Feature policy {path} row {row_number} for feature {feature_name!r} has empty mask value."
            )
        value = _parse_mask_value(raw_value)
        if op != "eq" and not isinstance(value, (int, float)):
            raise ValueError(
                f"Feature policy {path} row {row_number} for feature {feature_name!r} uses operator {op!r} "
                f"with non-numeric value {raw_value!r}."
            )
        predicates.append(MaskPredicate(op=op, value=value))
    return tuple(predicates)


def _parse_mask_value(raw_value: str) -> Any:
    lowered = raw_value.lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    if _INTEGER_PATTERN.fullmatch(raw_value):
        return int(raw_value)
    if _FLOAT_PATTERN.fullmatch(raw_value):
        return float(raw_value)
    return raw_value


def _mask_expression(feature_name: str, predicates: Sequence[MaskPredicate]) -> pl.Expr:
    expression: pl.Expr | None = None
    for predicate in predicates:
        if predicate.op == "eq":
            current = pl.col(feature_name) == pl.lit(predicate.value)
        elif predicate.op == "lt":
            current = pl.col(feature_name) < pl.lit(predicate.value)
        elif predicate.op == "le":
            current = pl.col(feature_name) <= pl.lit(predicate.value)
        elif predicate.op == "gt":
            current = pl.col(feature_name) > pl.lit(predicate.value)
        elif predicate.op == "ge":
            current = pl.col(feature_name) >= pl.lit(predicate.value)
        else:  # pragma: no cover
            raise ValueError(f"Unsupported mask operator {predicate.op!r}.")
        expression = current if expression is None else (expression | current)
    if expression is None:
        raise ValueError(f"Mask expression for feature {feature_name!r} is empty.")
    return expression.fill_null(False)


__all__ = [
    "FeaturePolicy",
    "FeaturePolicyEntry",
    "MaskPredicate",
    "SERIES_NON_FEATURE_COLUMNS",
    "apply_feature_policy",
    "feature_policy_path_for",
    "feature_policy_relative_path_for",
    "load_feature_policy",
    "series_candidate_feature_columns",
]
