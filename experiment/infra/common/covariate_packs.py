from __future__ import annotations

from pathlib import Path
import sys

_REPO_ROOT = Path(__file__).resolve().parents[3]
_SRC_ROOT = _REPO_ROOT / "src"
if str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))

from wind_datasets.feature_protocols import (  # noqa: E402
    DEFAULT_COVARIATE_STAGES,
    CovariatePackSpec,
    iter_covariate_packs,
    reference_pack_for,
    resolve_covariate_pack,
)

__all__ = [
    "DEFAULT_COVARIATE_STAGES",
    "CovariatePackSpec",
    "iter_covariate_packs",
    "reference_pack_for",
    "resolve_covariate_pack",
]
