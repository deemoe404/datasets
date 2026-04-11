from __future__ import annotations

from pathlib import Path
import sys


EXPERIMENT_ROOT = Path(__file__).resolve().parents[2]
COMMON_DIR = EXPERIMENT_ROOT / "infra" / "common"
if str(COMMON_DIR) not in sys.path:
    sys.path.insert(0, str(COMMON_DIR))

from covariate_packs import (  # noqa: E402
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
