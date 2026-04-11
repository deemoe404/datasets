from __future__ import annotations

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


def test_default_family_output_path_uses_published_layout() -> None:
    module = _load_module()

    assert module.default_family_output_path(
        repo_root=Path("/tmp/datasets"),
        family_id="ltsf_linear_local",
    ) == (
        Path("/tmp/datasets").resolve()
        / "experiment"
        / "artifacts"
        / "published"
        / "ltsf_linear_local"
        / "latest.csv"
    )


def test_family_id_for_experiment_name_rejects_unknown_values() -> None:
    module = _load_module()

    with pytest.raises(ValueError, match="Unknown experiment name"):
        module.family_id_for_experiment_name("unknown-experiment")
