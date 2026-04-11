from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _default_source_policy_behavior() -> None:
    return None
