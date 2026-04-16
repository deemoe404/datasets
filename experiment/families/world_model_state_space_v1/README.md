# world_model_state_space_v1

`world_model_state_space_v1` is a Kelmarsh-only state-space world-model family
for the active `next_6h_from_24h` task and `world_model_v1` feature protocol.
It consumes the existing task bundle (`series`, `window_index`, `known_future`,
`static`, `pairwise`) and does not modify source data or dataset-side protocol
semantics.

The first implementation is an MVP of the state-space/filtering/rollout network:
history steps run transition plus correction with real historical observations,
future steps roll out open-loop from latent state and calendar inputs, and future
SCADA/PMU/global observations are used only as supervised targets.

## Run

```shell
cd experiment/families/world_model_state_space_v1
./create_env.sh
./.conda/bin/python run_world_model_state_space_v1.py
```

Smoke run:

```shell
./.conda/bin/python run_world_model_state_space_v1.py \
  --epochs 1 \
  --device cpu \
  --max-train-origins 64 \
  --max-eval-origins 32 \
  --output-path ../../artifacts/scratch/world_model_state_space_v1/kelmarsh_smoke.csv
```

Default formal output:

- `experiment/artifacts/published/world_model_state_space_v1/latest.csv`
- `experiment/artifacts/published/world_model_state_space_v1/latest.training_history.csv`
- `experiment/artifacts/runs/world_model_state_space_v1/<timestamp>/manifest.json`

## Scope

- Dataset scope: `kelmarsh` only.
- Feature protocol: `world_model_v1`.
- Task: `next_6h_from_24h`, `history_steps=144`, `forecast_steps=36`.
- Deferred: search harness, multi-seed orchestration, staged training, robust
  augmentation, learned residual edges, stratified evaluation, and full ablation
  matrix.
