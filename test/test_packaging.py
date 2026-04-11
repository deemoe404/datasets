from __future__ import annotations

from pathlib import Path
import shutil
import subprocess
import sys
from zipfile import ZipFile


def test_wheel_includes_runtime_policy_csvs(tmp_path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    project_root = tmp_path / "project"
    wheel_dir = tmp_path / "wheelhouse"
    project_root.mkdir()
    wheel_dir.mkdir()

    shutil.copy2(repo_root / "pyproject.toml", project_root / "pyproject.toml")
    shutil.copytree(repo_root / "src", project_root / "src")

    subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "wheel",
            "--no-deps",
            "--no-build-isolation",
            str(project_root),
            "-w",
            str(wheel_dir),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    wheel_path = next(wheel_dir.glob("wind_datasets-*.whl"))
    with ZipFile(wheel_path) as wheel:
        names = set(wheel.namelist())

    assert "wind_datasets/data/hill_of_towie_tuneup_2024.csv" in names
    assert "wind_datasets/data/feature_policy/kelmarsh.csv" in names
    assert "wind_datasets/data/source_column_policy/kelmarsh.csv" in names
