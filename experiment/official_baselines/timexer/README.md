# TimeXer Official Wrapper

- Source: `https://github.com/thuml/TimeXer.git`
- Pinned commit: `76011909357972bd55a27adba2e1be994d81b327`
- Source path: `source/`
- License: no root license file found in the pinned repository snapshot; verify
  license terms before publication or redistribution.
- Adapter target: `world_model_timexer_official_v1_farm_sync`

The TimeXer source repository is retained as the source-of-truth reference for
the exogenous-transformer baseline. The hardened family adapts the existing
`world_model_v1` history and known-future calendar tensors to the official
exogenous forecasting contract.

Run or refresh the wrapper environment with:

```shell
cd experiment/official_baselines/timexer
./create_env.sh
```
