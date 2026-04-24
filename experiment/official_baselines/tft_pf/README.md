# PyTorch Forecasting TFT Wrapper

- Source: PyTorch Forecasting package `pytorch-forecasting`
- Source docs:
  `https://pytorch-forecasting.readthedocs.io/en/stable/api/pytorch_forecasting.models.temporal_fusion_transformer._tft.TemporalFusionTransformer.html`
- License: governed by the installed package metadata; verify current package
  terms before publication or redistribution.
- Adapter target: future `world_model_tft_pf_hardened_v1_farm_sync`

This wrapper reserves the standard TemporalFusionTransformer path for the second
tranche of hardened baselines. It intentionally has no `source/` submodule.

Run or refresh the wrapper environment with:

```shell
cd experiment/official_baselines/tft_pf
./create_env.sh
```
