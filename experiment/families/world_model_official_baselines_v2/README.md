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
- `dgcrn_official_core_residual_b3_geometry_v2`
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
The B3 geometry residual variant uses the task pairwise feature
`distance_in_rotor_diameters`: off-diagonal edge weights are
`exp(-distance_in_rotor_diameters / 4)`, diagonal self-loops are `1.0`, and the
official DGCRN core receives `[A, A.T]`.

iTransformer official debug search should vary the declared CLI knobs
`--itransformer-d-model`, `--itransformer-n-heads`,
`--itransformer-e-layers`, `--itransformer-dropout`, `--learning-rate`, and
`--residual-anchor-steps`; these values are recorded in the summary, manifest,
trial id, and formal search config id. As of 2026-04-25, the validation-frozen
residual variants center target-history input channels by the last-value anchor
and disable the official internal normalization (`use_norm=False`). Direct
variants keep absolute target-history inputs and `use_norm=True`. Result rows
record this as `residual_input_mode` and `official_internal_norm`. The prior
validation-frozen target-plus-exog residual trial24 config
(`itransformer_target_plus_exog_d64_h4_e2_dropout0.1_lr0.0003_anchor1`) has
five-seed test mean/std and paired persistence bootstrap artifacts at
`experiment/artifacts/published/world_model_official_baselines_v2/20260425-itransformer-trial24-multiseed-origin-errors-*`.
The bootstrap direction is `baseline_abs_error_pu - proposed_abs_error_pu`;
rolling and non-overlap deltas are both negative with
`prob_delta_gt_zero=0.0`, so this official iTransformer residual baseline is
weak relative to last-value persistence.

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

Chronos-2 zero-shot full rolling validation and full rolling test-once are
closed through the recoverable shard evaluator at
`experiment/families/world_model_official_baselines_v2/diagnostics/chronos2_rolling_shards.py`.
The frozen zero-shot B2 rolling test aggregate is
`experiment/artifacts/scratch/world_model_official_baselines_v2/chronos2_test_rolling_shards_20260425/chronos2_kelmarsh_test_rolling_origin_no_refit_aggregate.{json,csv}`.
The shard evaluator stores flat exact absolute-error arrays rather than
per-origin CSVs, so
`experiment/families/world_model_official_baselines_v2/diagnostics/enrich_chronos_origin_errors.py`
reconstructs per-origin proposed AE from those arrays using the prepared
valid-count mask, verifies shard/window metadata, and writes the published
paired persistence comparison and bootstrap/status artifacts at
`experiment/artifacts/published/world_model_official_baselines_v2/20260425-chronos2-rolling-test-origin-errors-*`.
The bootstrap direction is `baseline_abs_error_pu - proposed_abs_error_pu`;
the rolling paired/block deltas are positive, so the zero-shot Chronos-2 row
is slightly favorable to Chronos versus last-value persistence on origin-level
AE. Chronos-2 remains zero-shot only, not a trainable multiseed baseline.

TFT-PF full rolling evaluation is chunked by forecast origins via
`--tft-eval-window-chunk-size` (default `1024`). This preserves the same test
window set and metrics while avoiding one-shot construction of the full
per-turbine PyTorch Forecasting prediction frame. Evaluation uses the official
TFT forward pass over a PyTorch dataloader rather than repeated
`TemporalFusionTransformer.predict()` calls, which keeps the model
implementation official while avoiding repeated Lightning predictor teardown.

Current status as of 2026-04-25: TFT-PF full rolling validation and full
rolling test-once are closed only through the recoverable shard evaluator,
not through a monolithic run. The full rolling test expands 94,458 forecast
origins into per-turbine PyTorch Forecasting frames; the one-shot path
exhausted memory and I/O, and larger chunked predictor calls hit native
NumPy/PyTorch Forecasting crashes. The shard evaluator isolates the lifecycle
in one subprocess per segment, retries native failures, splits persistent
failures, checks continuous coverage, and aggregates only complete shard
artifacts. The frozen residual B2 config now has full rolling test-once
evidence at
`experiment/artifacts/scratch/world_model_official_baselines_v2/tft_pf_test_rolling_shards_full_20260425/tft_pf_residual_kelmarsh_test_rolling_origin_no_refit_aggregate.{json,csv}`;
published origin-level persistence comparison and bootstrap/status artifacts
were generated at
`experiment/artifacts/published/world_model_official_baselines_v2/20260425-tft-pf-trial02-rolling-test-origin-errors-*`.
The bootstrap direction is `baseline_abs_error_pu - proposed_abs_error_pu`;
the rolling paired/block deltas are both negative, so this single-seed result
is unfavorable to TFT-PF versus last-value persistence. The result remains
runtime-fragile and should not be used as a final paper-table row until
multiseed mean/std artifacts are generated.

For DGCRN formal search, `gate_b_passed` may be sourced from a declared
64-window overfit preflight via `--gate-b-overfit64-passed`; the full-fit
train-window diagnostic is recorded separately as
`train_gate_after_fit_passed`, `train_gate_after_fit_rmse_pu`, and
`train_gate_after_fit_mae_pu`. This keeps the paper gate contract distinct from
post-search training-set diagnostics.

## Paper-Grade Long Run Driver

The 2026-04-25 paper-grade queue is generated by:

```shell
./.conda/bin/python experiment/families/world_model_official_baselines_v2/diagnostics/long_run_driver.py \
  plan \
  --run-root experiment/artifacts/scratch/world_model_official_baselines_v2/long_run_20260425_paper_grade
```

Execute or resume the queue one item at a time with:

```shell
./.conda/bin/python experiment/families/world_model_official_baselines_v2/diagnostics/long_run_driver.py \
  run \
  --run-root experiment/artifacts/scratch/world_model_official_baselines_v2/long_run_20260425_paper_grade
```

The driver writes command logs and exit records under the run root, skips items
whose expected artifacts already exist, and keeps selection validation-only.
After phase-2 search finishes, run the generated `select` item, then run the
materialized `phase3_full_validation_queue.json`. After selecting from the full
validation outputs, materialize and run `phase4_test_multiseed_queue.json`.
