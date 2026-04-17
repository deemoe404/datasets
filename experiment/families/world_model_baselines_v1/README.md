# world_model_baselines_v1

`world_model_baselines_v1` is a Kelmarsh-only baseline family for comparing
against `world_model_state_space_v1` on the active `next_6h_from_24h` task and
the existing `world_model_v1` task bundle. It does not modify source data or
dataset-side protocol semantics.

The first implementation includes:

- `world_model_persistence_last_value_v1_farm_sync`: analytic last-value
  persistence over the 24-hour history window, with train-only target means as
  fallback for fully missing histories.
- `world_model_shared_weight_tft_no_graph_v1_farm_sync`: repo-local PyTorch
  shared-weight TFT-style model with no graph, no pairwise features, no global
  latent state, no future observations, and no turbine id embedding.
- `world_model_shared_weight_timexer_no_graph_v1_farm_sync`: repo-local
  shared-weight TimeXer-style exogenous forecaster with endogenous
  `target_pu` history patches, historical local/global exogenous channels, and
  historical calendar marks only; it does not consume `context_future`,
  `static`, or `pairwise`.
- `world_model_dgcrn_v1_farm_sync`: repo-local DGCRN-style dynamic-graph
  recurrent baseline that consumes the same `world_model_v1` bundle contract
  through broadcast history context, known-future calendar inputs, static node
  metadata, and pairwise geometry bias terms.
- `world_model_chronos_2_zero_shot_v1_farm_sync`: Chronos-2 zero-shot baseline
  over the same `world_model_v1` history/future tensor contract, with NaN-restored
  target history, local/global historical covariates, and calendar covariates
  adapted into Chronos batch payloads without fine-tuning.
- `world_model_itransformer_no_graph_v1_farm_sync`: repo-local iTransformer
  baseline that consumes full farm windows without graph inputs and predicts
  all turbines jointly from local history, global history context, future
  calendar covariates, and static turbine metadata.
- `world_model_mtgnn_calendar_graph_v1_farm_sync`: repo-local MTGNN calendar-
  graph baseline that learns an adaptive turbine graph from the same history
  and calendar tensors and predicts all turbines jointly without adding new
  dataset-side features.

## Run

```shell
cd experiment/families/world_model_baselines_v1
./create_env.sh
./.conda/bin/python run_world_model_baselines_v1.py
```

Run only persistence:

```shell
./.conda/bin/python run_world_model_baselines_v1.py \
  --variant world_model_persistence_last_value_v1_farm_sync
```

Run only DGCRN:

```shell
./.conda/bin/python run_world_model_baselines_v1.py \
  --variant world_model_dgcrn_v1_farm_sync
```

Run only Chronos-2 zero-shot:

```shell
./.conda/bin/python run_world_model_baselines_v1.py \
  --variant world_model_chronos_2_zero_shot_v1_farm_sync
```

Run only iTransformer:

```shell
./.conda/bin/python run_world_model_baselines_v1.py \
  --variant world_model_itransformer_no_graph_v1_farm_sync
```

Run only MTGNN:

```shell
./.conda/bin/python run_world_model_baselines_v1.py \
  --variant world_model_mtgnn_calendar_graph_v1_farm_sync
```

Smoke run:

```shell
./.conda/bin/python run_world_model_baselines_v1.py \
  --epochs 1 \
  --device cpu \
  --max-train-origins 64 \
  --max-eval-origins 32 \
  --output-path ../../artifacts/scratch/world_model_baselines_v1/kelmarsh_smoke.csv \
  --no-record-run
```

Default formal output:

- `experiment/artifacts/published/world_model_baselines_v1/<run_timestamp>.csv`
- `experiment/artifacts/published/world_model_baselines_v1/<run_timestamp>.training_history.csv`
- `experiment/artifacts/runs/world_model_baselines_v1/<timestamp>/manifest.json`

Default `--resume` and `--force-rerun` flows must pass the exact historical
`--output-path` because the formal publish path is timestamped per run.

TensorBoard is enabled by default when the family environment includes the
`tensorboard` package. The default log root is tied to the output-path hash:

- `experiment/families/world_model_baselines_v1/.work/run_world_model_baselines_v1/<output-hash>/tensorboard/`

View logs with:

```shell
tensorboard --logdir experiment/families/world_model_baselines_v1/.work/run_world_model_baselines_v1
```

Or override the log root explicitly:

```shell
./.conda/bin/python run_world_model_baselines_v1.py \
  --tensorboard-log-dir ../../artifacts/scratch/world_model_baselines_v1/tensorboard
```

## Scope

- Dataset scope: `kelmarsh` only.
- Feature protocol: `world_model_v1`.
- Task: `next_6h_from_24h`, `history_steps=144`, `forecast_steps=36`.
- Registry `training_mode` is `trainable` because the family includes TFT,
  TimeXer, DGCRN, iTransformer, and MTGNN, even though persistence and
  Chronos-2 are analytic / zero-shot baselines.
- Deferred: state-space ablations such as no-dynamic-graph, no-global-state,
  single-state, linear-head, and no-met-loss variants.
