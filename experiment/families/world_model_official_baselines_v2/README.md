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
