from __future__ import annotations


class Chronos2OfficialAdapter:
    source_file = "site-packages/chronos"
    model_id = "amazon/chronos-2"

    def load_pipeline_class(self):
        from chronos import Chronos2Pipeline  # type: ignore

        return Chronos2Pipeline
