from __future__ import annotations


class TFTPytorchForecastingAdapter:
    source_file = "site-packages/pytorch_forecasting/models/temporal_fusion_transformer/_tft.py"

    def load_model_class(self):
        from pytorch_forecasting.models import TemporalFusionTransformer  # type: ignore

        return TemporalFusionTransformer
