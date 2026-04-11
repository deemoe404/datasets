from __future__ import annotations

import csv
from pathlib import Path

import polars as pl
import pytest

from wind_datasets import feature_policy as feature_policy_module


def _write_policy(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["feature_name", "decision", "mask_rule", "reason", "notes"],
        )
        writer.writeheader()
        writer.writerows(rows)


def test_load_feature_policy_and_apply_drop_and_mask(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(feature_policy_module, "_FEATURE_POLICY_DIR", tmp_path)
    _write_policy(
        tmp_path / "toy.csv",
        [
            {
                "feature_name": "keep_feature",
                "decision": "keep",
                "mask_rule": "",
                "reason": "seed",
                "notes": "",
            },
            {
                "feature_name": "mask_feature",
                "decision": "mask+keep",
                "mask_rule": "eq:0|lt:-1",
                "reason": "invalid_zero",
                "notes": "",
            },
            {
                "feature_name": "drop_feature",
                "decision": "drop",
                "mask_rule": "",
                "reason": "constant",
                "notes": "",
            },
        ],
    )

    policy = feature_policy_module.load_feature_policy(
        "toy",
        ("keep_feature", "mask_feature", "drop_feature"),
    )
    frame = pl.DataFrame(
        {
            "keep_feature": [1.0, 2.0],
            "mask_feature": [0.0, 3.0],
            "drop_feature": [9.0, 9.0],
        }
    )

    transformed = feature_policy_module.apply_feature_policy(frame, policy)

    assert transformed.columns == ["keep_feature", "mask_feature"]
    assert transformed["keep_feature"].to_list() == [1.0, 2.0]
    assert transformed["mask_feature"].to_list() == [None, 3.0]
    assert policy.decision_counts == {"drop": 1, "keep": 1, "mask+keep": 1}


def test_load_feature_policy_validates_schema_and_mask_rules(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(feature_policy_module, "_FEATURE_POLICY_DIR", tmp_path)
    path = tmp_path / "toy.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "feature_name,decision,mask_rule,reason,notes\n"
        "mask_feature,mask+keep,,missing_rule,\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="requires a mask_rule"):
        feature_policy_module.load_feature_policy("toy", ("mask_feature",))


def test_load_feature_policy_is_strict_by_default_but_can_allow_policy_supersets(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(feature_policy_module, "_FEATURE_POLICY_DIR", tmp_path)
    _write_policy(
        tmp_path / "toy.csv",
        [
            {
                "feature_name": "feature_a",
                "decision": "keep",
                "mask_rule": "",
                "reason": "seed",
                "notes": "",
            },
            {
                "feature_name": "feature_b",
                "decision": "keep",
                "mask_rule": "",
                "reason": "seed",
                "notes": "",
            },
        ],
    )

    with pytest.raises(ValueError, match="unknown features"):
        feature_policy_module.load_feature_policy("toy", ("feature_a",))

    relaxed = feature_policy_module.load_feature_policy(
        "toy",
        ("feature_a",),
        allow_extra_features=True,
    )

    assert relaxed.feature_names == ("feature_a", "feature_b")
