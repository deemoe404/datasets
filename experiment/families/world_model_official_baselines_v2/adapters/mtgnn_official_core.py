from __future__ import annotations

from pathlib import Path
import sys


class MTGNNOfficialCoreAdapter:
    def __init__(self, repo_root: Path) -> None:
        self.source_root = repo_root / "experiment" / "official_baselines" / "mtgnn" / "source"
        self.source_file = str(self.source_root / "net.py")

    def load_model_class(self):
        if str(self.source_root) not in sys.path:
            sys.path.insert(0, str(self.source_root))
        from net import gtnet  # type: ignore

        return gtnet
