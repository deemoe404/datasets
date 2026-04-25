# world_model_official_baselines_v2

This family supersedes `world_model_hardened_baselines_v1` for paper-grade
official / official-core baseline comparisons. The previous family recorded
official source provenance but delegated trainable execution to repo-local
backend variants; those outputs are retained as phase-1 adapter sanity checks
only.

`world_model_official_baselines_v2` separates data/evaluation utilities from
model implementation. Trainable baselines instantiate official or official-core
model classes from `experiment/official_baselines/<model>/source` or standard
external packages. Runs fail if the model implementation resolves to repo-local
baseline backend classes.

## Variants

- `baseline_last_value_persistence_v2`
- `baseline_seasonal_persistence_v2`
- `baseline_ridge_residual_persistence_b0_v2`
- `baseline_mlp_residual_persistence_b0_v2`
- `baseline_gru_residual_persistence_b0_v2`
- `baseline_tcn_residual_persistence_b0_v2`
- `dgcrn_official_core_direct_b2_v2`
- `dgcrn_official_core_residual_b2_v2`
- `timexer_official_target_only_direct_b0_v2`
- `timexer_official_target_only_residual_b0_v2`
- `timexer_official_full_exog_residual_b2_v2`
- `itransformer_official_target_only_direct_b0_v2`
- `itransformer_official_target_only_residual_b0_v2`
- `itransformer_official_target_plus_exog_residual_b2_v2`
- `tft_pf_per_turbine_direct_b2_v2`
- `tft_pf_per_turbine_residual_b2_v2`
- `mtgnn_official_core_target_only_b0_v2`
- `mtgnn_official_core_calendar_residual_b1_v2`
- `chronos2_official_zero_shot_b2_v2`

## Feature Budgets

- `B0`: target history only.
- `B1`: target history plus future calendar.
- `B2`: target, historical local/global exogenous variables, future calendar,
  and static features.
- `B3`: B2 plus pairwise geometry for graph/state-space models.

Every result row records the feature-budget booleans and
`uses_future_target=false`.

## Gates

Before formal tuning, each trainable baseline must pass:

- Gate A: shape, horizon, and leakage snapshot.
- Gate B: 64-window overfit.
- Gate C: 10-minute continuity against persistence.
- Gate D: validation-only selection.
- Gate E: frozen test-once manifest.

The current runner writes the v2 artifact/provenance/gate surface needed for
debug execution. Formal CUDA search should be launched on `Ubuntu:/home/sam/datasets`
after local tests pass and the branch is synced through git.

## Run

```shell
cd experiment/families/world_model_official_baselines_v2
./.conda/bin/python run_world_model_official_baselines_v2.py \
  --output-path ../../artifacts/scratch/world_model_official_baselines_v2/debug_matrix.csv \
  --no-record-run
```

Formal tuning is fail-closed while the official trainable adapters are being
ported. The launcher runs only executable baselines and records all missing
official trainable adapters as blocked rows, not performance results:

```shell
./.conda/bin/python run_world_model_official_baselines_v2_formal_tuning.py \
  --output-path ../../artifacts/published/world_model_official_baselines_v2/20260424-formal-tuning-start.csv \
  --run-label official_baselines_v2_formal_tuning_start_20260424
```

For neural adapter bring-up, keep checkpoint selection validation-only but bound
the validation surface while debugging:

```shell
./.conda/bin/python run_world_model_official_baselines_v2_formal_tuning.py \
  --variant dgcrn_official_core_residual_b2_v2 \
  --eval-protocol non_overlap \
  --checkpoint-eval-protocol non_overlap \
  --max-checkpoint-origins 1310 \
  --gate-origin-count 64 \
  --residual-anchor-steps 1 \
  --dgcrn-hidden-dim 64 \
  --dgcrn-dropout 0.1 \
  --dgcrn-gcn-depth 2 \
  --output-path ../../artifacts/scratch/world_model_official_baselines_v2/dgcrn_debug.csv \
  --no-record-run
```

`--residual-anchor-steps 1` is a declared output parameterization for residual
neural adapters: the first 10-minute residual is fixed to zero, so the point
forecast at lead 1 is exactly the last-value persistence anchor. The anchored
lead is excluded from training loss because the model is not allowed to change
it.

Current executable formal rows include analytic persistence, the closed-form
Ridge residual control, Chronos-2 zero-shot, DGCRN official-core
direct/residual, TimeXer official target-only direct/residual/full-exog
residual, and iTransformer official target-only direct/residual plus
target-plus-exog residual, TFT-PF per-turbine direct/residual, and MTGNN
official-core target-only/calendar-residual. Neural residual controls must not
be interpreted as tuned until their v2 adapters implement real training.

DGCRN official-core debug search should vary the declared CLI knobs
`--dgcrn-hidden-dim`, `--dgcrn-dropout`, `--dgcrn-gcn-depth`,
`--learning-rate`, and `--residual-anchor-steps`; these values are recorded in
the summary, manifest, trial id, and formal search config id.

iTransformer official debug search should vary the declared CLI knobs
`--itransformer-d-model`, `--itransformer-n-heads`,
`--itransformer-e-layers`, `--itransformer-dropout`, `--learning-rate`, and
`--residual-anchor-steps`; these values are recorded in the summary, manifest,
trial id, and formal search config id.

TFT-PF debug search should vary the declared CLI knobs `--tft-hidden-size`,
`--tft-lstm-layers`, `--tft-attention-head-size`,
`--tft-hidden-continuous-size`, `--tft-dropout`, `--learning-rate`, and
`--residual-anchor-steps`; these values are recorded in the summary, manifest,
trial id, and formal search config id.

MTGNN official-core debug search should vary the declared CLI knobs
`--mtgnn-gcn-depth`, `--mtgnn-subgraph-size`, `--mtgnn-node-dim`,
`--mtgnn-residual-channels`, `--mtgnn-skip-channels`,
`--mtgnn-end-channels`, `--mtgnn-layers`, `--mtgnn-dropout`,
`--learning-rate`, and `--residual-anchor-steps`; these values are recorded in
the summary, manifest, trial id, and formal search config id. The B0
target-only variant uses official `gtnet` directly. The B1 residual variant
keeps official `gtnet` as the temporal graph core and adds only a small
task-adapter future-calendar bias head before residual re-anchoring.

TFT-PF full rolling evaluation is chunked by forecast origins via
`--tft-eval-window-chunk-size` (default `1024`). This preserves the same test
window set and metrics while avoiding one-shot construction of the full
per-turbine PyTorch Forecasting prediction frame. Evaluation uses the official
TFT forward pass over a PyTorch dataloader rather than repeated
`TemporalFusionTransformer.predict()` calls, which keeps the model
implementation official while avoiding repeated Lightning predictor teardown.

Current blocker: TFT-PF full rolling test-once is blocked as of 2026-04-25.
The full rolling test expands 94,458 forecast origins into per-turbine
PyTorch Forecasting frames and repeatedly destabilized the Ubuntu CUDA host:
the one-shot path exhausted memory and I/O, chunked predictor calls hit native
NumPy/PyTorch Forecasting crashes, and the manual full rolling attempt made
the host unreachable. The existing TFT-PF smoke, overfit64, validation, and
bounded-test diagnostics are retained as adapter evidence, but TFT-PF must not
be used as a paper-grade full rolling test row until a redesigned streaming
evaluator passes the bounded-to-full recovery ladder.

For DGCRN formal search, `gate_b_passed` may be sourced from a declared
64-window overfit preflight via `--gate-b-overfit64-passed`; the full-fit
train-window diagnostic is recorded separately as
`train_gate_after_fit_passed`, `train_gate_after_fit_rmse_pu`, and
`train_gate_after_fit_mae_pu`. This keeps the paper gate contract distinct from
post-search training-set diagnostics.
