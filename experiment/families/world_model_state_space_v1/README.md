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

Active variants:

- `world_model_state_space_v1_farm_sync`: canonical model.
- `world_model_state_space_v1_wake_off_farm_sync`: disables only dynamic wake features.
- `world_model_state_space_v1_graph_off_farm_sync`: bypasses graph aggregation.
- `world_model_state_space_v1_no_farm_aux_farm_sync`: sets farm auxiliary loss weight to `0.0`.
- `world_model_state_space_v1_no_met_aux_farm_sync`: sets met auxiliary loss weight to `0.0`.

The default runner behavior remains canonical-only; ablations must be selected
explicitly with repeated `--variant` flags.

## Run

```shell
cd experiment/families/world_model_state_space_v1
./create_env.sh
./.conda/bin/python run_world_model_state_space_v1.py
tensorboard --logdir ./.work/run_world_model_state_space_v1
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

Farm-loss sweep:

```shell
./.conda/bin/python search_world_model_state_space_v1.py
```

The search harness keeps the canonical variant fixed and screens
`farm_loss_weight ∈ {0.0, 0.02, 0.05, 0.1}` with a short full-window budget
before re-running the top 2 candidates with the formal default budget. It
writes `screen_summary.csv`, `final_summary.csv`, `final_detailed_rows.csv`,
`search_plan.json`, and `selected_defaults.json` under
`experiment/artifacts/scratch/world_model_state_space_v1/search_<date>/`.

Default formal output:

- `experiment/artifacts/published/world_model_state_space_v1/<run_timestamp>.csv`
- `experiment/artifacts/published/world_model_state_space_v1/<run_timestamp>.training_history.csv`
- `experiment/artifacts/runs/world_model_state_space_v1/<timestamp>/manifest.json`

Default `--resume` and `--force-rerun` flows must pass the exact historical
`--output-path` because the formal publish path is timestamped per run.

Default TensorBoard output:

- `experiment/families/world_model_state_space_v1/.work/run_world_model_state_space_v1/<output-hash>/tensorboard/<dataset>/<variant>/`

## Scope

- Dataset scope: `kelmarsh` only.
- Feature protocol: `world_model_v1`.
- Task: `next_6h_from_24h`, `history_steps=144`, `forecast_steps=36`.
- Default variant: `world_model_state_space_v1_farm_sync`.
- Deferred: multi-seed orchestration, staged training, robust augmentation,
  learned residual edges, stratified evaluation, and full ablation matrix.
