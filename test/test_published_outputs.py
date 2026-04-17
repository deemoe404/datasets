from __future__ import annotations

from datetime import UTC, datetime
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import sys

import pytest


def _load_module():
    module_path = (
        Path(__file__).resolve().parents[1]
        / "experiment"
        / "infra"
        / "common"
        / "published_outputs.py"
    )
    spec = spec_from_file_location("published_outputs", module_path)
    assert spec is not None
    assert spec.loader is not None
    module = module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_generate_run_stem_uses_utc_timestamp_format() -> None:
    module = _load_module()

    assert module.generate_run_stem(now=datetime(2026, 4, 18, 7, 8, 9, tzinfo=UTC)) == "20260418-070809"


def test_default_family_output_path_uses_timestamped_publish_layout() -> None:
    module = _load_module()
    repo_root = Path("/tmp/datasets").resolve()
    run_stem = "20260418-070809"

    assert module.default_family_output_path(
        repo_root=repo_root,
        family_id="agcrn_official_aligned",
        run_stem=run_stem,
    ) == repo_root / "experiment" / "artifacts" / "published" / "agcrn_official_aligned" / "20260418-070809.csv"
    assert module.default_family_output_path(
        repo_root=repo_root,
        family_id="agcrn_masked",
        run_stem=run_stem,
    ) == repo_root / "experiment" / "artifacts" / "published" / "agcrn_masked" / "20260418-070809.csv"
    assert module.default_family_output_path(
        repo_root=repo_root,
        family_id="world_model_agcrn_v1",
        run_stem=run_stem,
    ) == repo_root / "experiment" / "artifacts" / "published" / "world_model_agcrn_v1" / "20260418-070809.csv"
    assert module.default_family_output_path(
        repo_root=repo_root,
        family_id="world_model_rollout_v1",
        run_stem=run_stem,
    ) == repo_root / "experiment" / "artifacts" / "published" / "world_model_rollout_v1" / "20260418-070809.csv"
    assert module.default_family_output_path(
        repo_root=repo_root,
        family_id="world_model_baselines_v1",
        run_stem=run_stem,
    ) == repo_root / "experiment" / "artifacts" / "published" / "world_model_baselines_v1" / "20260418-070809.csv"
    assert module.default_family_output_path(
        repo_root=repo_root,
        family_id="world_model_state_space_v1",
        run_stem=run_stem,
    ) == repo_root / "experiment" / "artifacts" / "published" / "world_model_state_space_v1" / "20260418-070809.csv"


def test_default_family_output_template_uses_run_timestamp_token() -> None:
    module = _load_module()
    repo_root = Path("/tmp/datasets").resolve()

    assert module.default_family_output_template(
        repo_root=repo_root,
        family_id="world_model_state_space_v1",
    ) == repo_root / "experiment" / "artifacts" / "published" / "world_model_state_space_v1" / "{run_timestamp}.csv"


def test_default_family_output_path_rejects_invalid_inputs() -> None:
    module = _load_module()

    with pytest.raises(ValueError, match="Run timestamp stem"):
        module.default_family_output_path(
            repo_root=Path("/tmp/datasets"),
            family_id="agcrn_official_aligned",
            run_stem="not-a-timestamp",
        )
    with pytest.raises(ValueError, match="single path component"):
        module.default_family_output_path(
            repo_root=Path("/tmp/datasets"),
            family_id="agcrn_official_aligned",
            filename="nested/output.csv",
        )
    with pytest.raises(ValueError, match="either run_stem or filename"):
        module.default_family_output_path(
            repo_root=Path("/tmp/datasets"),
            family_id="agcrn_official_aligned",
            run_stem="20260418-070809",
            filename="custom.csv",
        )


def test_family_id_for_experiment_name_resolves_active_family() -> None:
    module = _load_module()

    assert module.family_id_for_experiment_name("agcrn") == "agcrn_official_aligned"
    assert module.family_id_for_experiment_name("agcrn_masked") == "agcrn_masked"
    assert module.family_id_for_experiment_name("world_model_agcrn_v1") == "world_model_agcrn_v1"
    assert module.family_id_for_experiment_name("world_model_rollout_v1") == "world_model_rollout_v1"
    assert module.family_id_for_experiment_name("world_model_baselines_v1") == "world_model_baselines_v1"
    assert module.family_id_for_experiment_name("world_model_state_space_v1") == "world_model_state_space_v1"


def test_family_id_for_experiment_name_rejects_unknown_values() -> None:
    module = _load_module()

    with pytest.raises(ValueError, match="Unknown experiment name"):
        module.family_id_for_experiment_name("unknown-experiment")
