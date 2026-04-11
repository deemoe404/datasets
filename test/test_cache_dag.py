from __future__ import annotations

import os

import polars as pl
import pytest

from wind_datasets import rebuild_cache as rebuild_module
from wind_datasets.cache_state import LayerStatus, read_build_meta
from wind_datasets.datasets.greenbyte import GreenbyteDatasetBuilder
from wind_datasets.datasets.hill_of_towie import HillOfTowieDatasetBuilder
from wind_datasets.datasets.sdwpf_kddcup import SDWPFKDDCupDatasetBuilder
from wind_datasets.manifest import build_manifest as build_manifest_for_spec
from wind_datasets.models import TaskSpec

from .helpers import build_greenbyte_fixture, build_hill_fixture, build_sdwpf_kddcup_fixture


def test_load_series_rebuilds_gold_when_build_meta_missing(tmp_path) -> None:
    spec = build_greenbyte_fixture(tmp_path / "raw" / "kelmarsh", "Kelmarsh", "Kelmarsh 1")
    builder = GreenbyteDatasetBuilder(spec=spec, cache_root=tmp_path / "cache")

    builder.build_gold_base()
    gold_meta_path = builder.cache_paths.gold_base_build_meta_path
    gold_meta_path.unlink()

    status = builder.gold_base_status()
    assert status.status == "stale"
    assert status.reason == "missing_build_meta"

    series = builder.load_series()

    assert series.height > 0
    assert builder.gold_base_status().status == "fresh"


def test_load_window_index_rebuilds_task_when_build_meta_missing(tmp_path) -> None:
    spec = build_greenbyte_fixture(tmp_path / "raw" / "kelmarsh", "Kelmarsh", "Kelmarsh 1")
    builder = GreenbyteDatasetBuilder(spec=spec, cache_root=tmp_path / "cache")
    task = TaskSpec(
        history_duration="30m",
        forecast_duration="20m",
        task_id="farm_short",
        granularity="farm",
    )

    builder.build_task_cache(task)
    task_meta_path = builder.cache_paths.task_build_meta_path_for("farm_short", "power_only")
    task_meta_path.unlink()

    status = builder.task_cache_status(task)
    assert status.status == "stale"
    assert status.reason == "missing_build_meta"

    window_index = builder.load_window_index(task)

    assert window_index.height > 0
    assert builder.task_cache_status(task).status == "fresh"


def test_missing_task_report_does_not_invalidate_task_cache(tmp_path) -> None:
    spec = build_greenbyte_fixture(tmp_path / "raw" / "kelmarsh", "Kelmarsh", "Kelmarsh 1")
    builder = GreenbyteDatasetBuilder(spec=spec, cache_root=tmp_path / "cache")
    task = TaskSpec(
        history_duration="30m",
        forecast_duration="20m",
        task_id="farm_short",
        granularity="farm",
    )

    builder.build_task_cache(task)
    report_path = builder.task_bundle_paths(task).task_report_path
    report_path.unlink()

    status = builder.task_cache_status(task)
    assert status.status == "fresh"
    assert status.reason is None

    bundle = builder.load_task_bundle(task)
    assert bundle.window_index.height > 0
    assert bundle.task_report is None


def test_source_snapshot_change_invalidates_manifest_and_descendants(tmp_path) -> None:
    spec = build_greenbyte_fixture(tmp_path / "raw" / "kelmarsh", "Kelmarsh", "Kelmarsh 1")
    builder = GreenbyteDatasetBuilder(spec=spec, cache_root=tmp_path / "cache")
    task = TaskSpec(
        history_duration="30m",
        forecast_duration="20m",
        task_id="farm_short",
        granularity="farm",
    )

    builder.build_task_cache(task)
    manifest_meta_before = read_build_meta(builder.cache_paths.manifest_build_meta_path)
    silver_meta_before = read_build_meta(builder.cache_paths.silver_build_meta_path)
    gold_meta_before = read_build_meta(builder.cache_paths.gold_base_build_meta_path)
    task_meta_before = read_build_meta(builder.cache_paths.task_build_meta_path_for("farm_short", "power_only"))
    assert manifest_meta_before is not None
    assert silver_meta_before is not None
    assert gold_meta_before is not None
    assert task_meta_before is not None

    source_path = sorted(spec.source_root.rglob("Turbine_Data_*.csv"))[0]
    updated_ns = source_path.stat().st_mtime_ns + 1_000_000_000
    os.utime(source_path, ns=(updated_ns, updated_ns))

    assert builder.manifest_status().status == "stale"
    assert builder.manifest_status().reason == "source_snapshot_changed"
    assert builder.silver_status().status == "stale"
    assert builder.silver_status().reason == "parent_fingerprint_changed"
    assert builder.gold_base_status().status == "stale"
    assert builder.gold_base_status().reason == "parent_fingerprint_changed"
    assert builder.task_cache_status(task).status == "stale"
    assert builder.task_cache_status(task).reason == "parent_fingerprint_changed"

    window_index = builder.load_window_index(task)

    assert window_index.height > 0
    assert builder.manifest_status().status == "fresh"
    assert builder.silver_status().status == "fresh"
    assert builder.gold_base_status().status == "fresh"
    assert builder.task_cache_status(task).status == "fresh"
    assert read_build_meta(builder.cache_paths.manifest_build_meta_path).fingerprint != manifest_meta_before.fingerprint
    assert read_build_meta(builder.cache_paths.silver_build_meta_path).fingerprint != silver_meta_before.fingerprint
    assert (
        read_build_meta(builder.cache_paths.gold_base_build_meta_path).fingerprint
        != gold_meta_before.fingerprint
    )
    assert (
        read_build_meta(builder.cache_paths.task_build_meta_path_for("farm_short", "power_only")).fingerprint
        != task_meta_before.fingerprint
    )


def test_build_gold_base_ensures_silver_fresh_when_code_fingerprint_changes(tmp_path, monkeypatch) -> None:
    from wind_datasets import cache_state

    spec = build_greenbyte_fixture(tmp_path / "raw" / "kelmarsh", "Kelmarsh", "Kelmarsh 1")
    builder = GreenbyteDatasetBuilder(spec=spec, cache_root=tmp_path / "cache")

    builder.build_gold_base()
    silver_meta_before = read_build_meta(builder.cache_paths.silver_build_meta_path)
    gold_meta_before = read_build_meta(builder.cache_paths.gold_base_build_meta_path)
    assert silver_meta_before is not None
    assert gold_meta_before is not None

    original = cache_state.code_fingerprint_for

    def _patched(layer: str, handler: str) -> str:
        fingerprint = original(layer, handler)
        if layer == "silver":
            return f"patched-{fingerprint}"
        return fingerprint

    monkeypatch.setattr(cache_state, "code_fingerprint_for", _patched)

    assert builder.silver_status().status == "stale"
    assert builder.silver_status().reason == "code_fingerprint_changed"
    assert builder.gold_base_status().status == "stale"
    assert builder.gold_base_status().reason == "parent_fingerprint_changed"

    builder.build_gold_base()

    assert builder.silver_status().status == "fresh"
    assert builder.gold_base_status().status == "fresh"
    assert read_build_meta(builder.cache_paths.silver_build_meta_path).fingerprint != silver_meta_before.fingerprint
    assert (
        read_build_meta(builder.cache_paths.gold_base_build_meta_path).fingerprint
        != gold_meta_before.fingerprint
    )


def test_hill_packaged_dependency_change_invalidates_silver_and_descendants(tmp_path, monkeypatch) -> None:
    from wind_datasets import cache_state

    spec = build_hill_fixture(tmp_path / "raw" / "hill_of_towie")
    builder = HillOfTowieDatasetBuilder(spec=spec, cache_root=tmp_path / "cache")
    task = TaskSpec(
        history_duration="30m",
        forecast_duration="20m",
        task_id="farm_short",
        granularity="farm",
    )

    builder.build_task_cache(task)

    silver_meta_before = read_build_meta(builder.cache_paths.silver_build_meta_path)
    gold_meta_before = read_build_meta(builder.cache_paths.gold_base_build_meta_path)
    task_meta_before = read_build_meta(builder.cache_paths.task_build_meta_path_for("farm_short", "power_only"))
    assert silver_meta_before is not None
    assert gold_meta_before is not None
    assert task_meta_before is not None
    assert "packaged_dependency_fingerprint" in silver_meta_before.params

    greenbyte_spec = build_greenbyte_fixture(tmp_path / "raw" / "kelmarsh", "Kelmarsh", "Kelmarsh 1")
    greenbyte_builder = GreenbyteDatasetBuilder(spec=greenbyte_spec, cache_root=tmp_path / "cache_greenbyte")
    greenbyte_builder.build_silver()
    greenbyte_silver_meta = read_build_meta(greenbyte_builder.cache_paths.silver_build_meta_path)
    assert greenbyte_silver_meta is not None
    assert "packaged_dependency_fingerprint" not in greenbyte_silver_meta.params

    original = cache_state.packaged_dependency_fingerprint_for

    def _patched(layer: str, spec) -> str | None:
        fingerprint = original(layer, spec)
        if layer == "silver" and spec.handler == "hill_of_towie":
            assert fingerprint is not None
            return f"patched-{fingerprint}"
        return fingerprint

    monkeypatch.setattr(cache_state, "packaged_dependency_fingerprint_for", _patched)

    assert builder.silver_status().status == "stale"
    assert builder.silver_status().reason == "params_changed"
    assert builder.gold_base_status().status == "stale"
    assert builder.gold_base_status().reason == "parent_fingerprint_changed"
    assert builder.task_cache_status(task).status == "stale"
    assert builder.task_cache_status(task).reason == "parent_fingerprint_changed"

    window_index = builder.load_window_index(task)

    assert window_index.height > 0
    assert builder.silver_status().status == "fresh"
    assert builder.gold_base_status().status == "fresh"
    assert builder.task_cache_status(task).status == "fresh"
    assert read_build_meta(builder.cache_paths.silver_build_meta_path).fingerprint != silver_meta_before.fingerprint
    assert (
        read_build_meta(builder.cache_paths.gold_base_build_meta_path).fingerprint
        != gold_meta_before.fingerprint
    )
    assert (
        read_build_meta(builder.cache_paths.task_build_meta_path_for("farm_short", "power_only")).fingerprint
        != task_meta_before.fingerprint
    )


def test_gold_base_policy_dependency_change_invalidates_gold_and_task(tmp_path, monkeypatch) -> None:
    from wind_datasets import cache_state

    spec = build_greenbyte_fixture(tmp_path / "raw" / "kelmarsh", "Kelmarsh", "Kelmarsh 1")
    builder = GreenbyteDatasetBuilder(spec=spec, cache_root=tmp_path / "cache")
    task = TaskSpec(
        history_duration="30m",
        forecast_duration="20m",
        task_id="farm_short",
        granularity="farm",
    )

    builder.build_task_cache(task)
    gold_meta_before = read_build_meta(builder.cache_paths.gold_base_build_meta_path)
    task_meta_before = read_build_meta(builder.cache_paths.task_build_meta_path_for("farm_short", "power_only"))
    assert gold_meta_before is not None
    assert task_meta_before is not None
    assert "packaged_dependency_fingerprint" in gold_meta_before.params

    original = cache_state.packaged_dependency_fingerprint_for

    def _patched(layer: str, spec) -> str | None:
        fingerprint = original(layer, spec)
        if layer == "gold_base" and spec.dataset_id == "kelmarsh":
            assert fingerprint is not None
            return f"patched-{fingerprint}"
        return fingerprint

    monkeypatch.setattr(cache_state, "packaged_dependency_fingerprint_for", _patched)

    assert builder.gold_base_status().status == "stale"
    assert builder.gold_base_status().reason == "params_changed"
    assert builder.task_cache_status(task).status == "stale"
    assert builder.task_cache_status(task).reason == "parent_fingerprint_changed"

    window_index = builder.load_window_index(task)

    assert window_index.height > 0
    assert builder.gold_base_status().status == "fresh"
    assert builder.task_cache_status(task).status == "fresh"
    assert (
        read_build_meta(builder.cache_paths.gold_base_build_meta_path).fingerprint
        != gold_meta_before.fingerprint
    )
    assert (
        read_build_meta(builder.cache_paths.task_build_meta_path_for("farm_short", "power_only")).fingerprint
        != task_meta_before.fingerprint
    )


def test_sdwpf_gold_and_task_status_blocked_when_manifest_time_semantics_invalid(tmp_path) -> None:
    spec = build_sdwpf_kddcup_fixture(tmp_path / "raw" / "sdwpf_bad")
    path = spec.source_root / "sdwpf_245days_v1.csv"
    frame = pl.read_csv(path).with_row_index("row_nr")
    frame = frame.with_columns(
        pl.when(pl.col("row_nr") == 3)
        .then(pl.lit("00:25"))
        .otherwise(pl.col("Tmstamp"))
        .alias("Tmstamp")
    ).drop("row_nr")
    frame.write_csv(path)

    build_manifest_for_spec(spec, tmp_path / "cache")
    builder = SDWPFKDDCupDatasetBuilder(spec=spec, cache_root=tmp_path / "cache")
    task = TaskSpec.next_6h_from_24h()

    assert builder.manifest_status().status == "fresh"
    assert builder.gold_base_status().status == "stale"
    assert builder.gold_base_status().reason == "blocked_by_manifest_time_semantics"
    assert builder.task_cache_status(task).status == "stale"
    assert builder.task_cache_status(task).reason == "blocked_by_manifest_time_semantics"


def test_rebuild_cli_check_reports_fresh_and_stale_layers(monkeypatch, capsys) -> None:
    class _FakeBuilder:
        def manifest_status(self) -> LayerStatus:
            return LayerStatus(status="fresh", reason=None, fingerprint="manifest")

        def silver_status(self) -> LayerStatus:
            return LayerStatus(status="stale", reason="missing_build_meta", fingerprint="silver")

        def gold_base_status(self, quality_profile=None, layout=None) -> LayerStatus:
            del quality_profile, layout
            return LayerStatus(status="fresh", reason=None, fingerprint="gold")

        def task_cache_status(self, task_spec, feature_protocol_id="power_only", quality_profile=None) -> LayerStatus:
            del task_spec, feature_protocol_id, quality_profile
            return LayerStatus(status="fresh", reason=None, fingerprint="task")

    monkeypatch.setattr(rebuild_module, "get_dataset_spec", lambda dataset: object())
    monkeypatch.setattr(rebuild_module, "get_builder", lambda spec, cache_root: _FakeBuilder())

    code = rebuild_module.main(["--check", "kelmarsh"])

    assert code == 1
    captured = capsys.readouterr()
    assert "layer=manifest status=fresh" in captured.out
    assert "layer=silver status=stale reason=missing_build_meta" in captured.out
    assert "turbine" not in captured.out
