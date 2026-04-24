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
- `world_model_state_space_v1_residual_persistence_farm_sync`: predicts an additive residual over the per-turbine last-value persistence anchor.
- `world_model_state_space_v1_residual_persistence_gated_sum_farm_sync`: keeps the residual head but changes graph aggregation from normalized gated average to gated sum.
- `world_model_state_space_v1_residual_persistence_rotor_units_wake_farm_sync`: keeps the residual head and normalized aggregation, but interprets wake `dx/dy` in rotor-diameter units before computing `d_parallel` and `d_cross`.
- `world_model_state_space_v1_residual_persistence_gated_sum_rotor_units_wake_farm_sync`: combines the residual-head gated-sum aggregator with the rotor-diameter wake-unit interpretation.
- `world_model_state_space_v1_global_local_residual_farm_sync`: decomposes the anchor-relative forecast into one farm-shared residual plus a hard zero-mean local correction.
- `world_model_state_space_v1_global_local_increment_farm_sync`: predicts one farm-shared increment plus a hard zero-mean local increment correction, then accumulates them from the persistence anchor back into level space.
- `world_model_state_space_v1_wake_off_farm_sync`: disables only dynamic wake features.
- `world_model_state_space_v1_graph_off_farm_sync`: bypasses graph aggregation.
- `world_model_state_space_v1_no_farm_aux_farm_sync`: sets farm auxiliary loss weight to `0.0`.
- `world_model_state_space_v1_no_met_aux_farm_sync`: sets met auxiliary loss weight to `0.0`.

Main-paper candidate freeze as of `2026-04-24`: do not add new proposed
architecture variants until the official baseline v2 hardening is complete.
Treat canonical, residual persistence with `farm_loss_weight=0.0`, ramp v2
residual `0.02`, and ramp v2 residual `0.05` as the main candidate set.
Global/local, edge-gating, wake-unit, and related variants are diagnostics or
appendix material unless a later explicit decision reopens proposed-model
optimization.

The default runner behavior remains canonical-only; ablations must be selected
explicitly with repeated `--variant` flags.
The search harness also remains canonical-only in this revision.
The residual edge-gating 2x2 stays residual-only on purpose: the base residual
variant is the control, while the three extra residual variants isolate
`gated_sum`, `rotor_diameter_units`, or both without turning these axes into
family-wide CLI knobs.
The two global/local variants use `global = whole farm`, fix `alpha = 1`, and
enforce a hard weighted zero-mean constraint on the local branch; they do not
introduce cluster state or extra CLI knobs in this first pass.

The family also exposes an optional derived-ramp auxiliary loss via
`--ramp-loss-weight` and `--ramp-huber-delta`. Both knobs default to disabled
behavior (`ramp_loss_weight=0.0`) and apply uniformly to every variant without
changing the published result schema or checkpoint-selection metric. The current
`ramp v2` implementation fixes the auxiliary to multiscale first-order
differences with `K={3,6}`; those scales are not exposed as CLI knobs in v2.

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

Residual-head ramp ablation example:

```shell
./.conda/bin/python run_world_model_state_space_v1.py \
  --variant world_model_state_space_v1_residual_persistence_farm_sync \
  --farm-loss-weight 0.0 \
  --ramp-loss-weight 0.02 \
  --ramp-huber-delta 0.015 \
  --selection-metric val_rmse_pu
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

Scratch checkpoint evaluation (`--load-best-checkpoint`) continues to write
`*.diagnostics.csv` and `*.summary.csv`, and now includes ramp-side diagnostics
such as `ramp_mae_pu`, `ramp_rmse_pu`, and `sign_agreement_rate` for both
`rolling_origin_no_refit` and optional `rolling_origin_carry_over`. The
aggregate ramp metrics now summarize the active `K={3,6}` scales, and the
scratch outputs also expose per-scale columns:

- `ramp_mae_pu_k3`, `ramp_rmse_pu_k3`, `sign_agreement_rate_k3`
- `ramp_mae_pu_k6`, `ramp_rmse_pu_k6`, `sign_agreement_rate_k6`

`rolling_origin_no_refit` scratch evaluation now runs through a batched path
instead of the old per-window sequential loop; `rolling_origin_carry_over`
keeps its sequential state-persistence semantics.

When checkpoint scratch eval is run with `--include-edge-diagnostics`, the same
scratch directory also gains a `*.edge_diagnostics.csv` file with one row per
`(split_name, eval_protocol, phase)` where `phase ∈ {transition, update}`.
Those rows summarize wake geometry and learned gate distributions
(`d_parallel/d_cross`, `wake_gate`, `learned_gate`, `gate_sum_per_dst`,
`gated_message_norm`, `aggregated_message_norm`) without changing the formal
publish schema or the default training path.

Default TensorBoard output:

- `experiment/families/world_model_state_space_v1/.work/run_world_model_state_space_v1/<output-hash>/tensorboard/<dataset>/<variant>/`

## Scope

- Dataset scope: `kelmarsh` only.
- Feature protocol: `world_model_v1`.
- Task: `next_6h_from_24h`, `history_steps=144`, `forecast_steps=36`.
- Default variant: `world_model_state_space_v1_farm_sync`.
- Deferred: multi-seed orchestration, staged training, robust augmentation,
  learned residual edges, stratified evaluation, and full ablation matrix.
