# Official-Aligned AGCRN

This experiment runs an official-aligned AGCRN baseline on:

- `kelmarsh`
- `penmanshiel`

The scope is intentionally narrow:

- `24h look back -> 6h ahead`
- `power_only`, `power_ws_hist`, `power_atemp_hist`, `power_itemp_hist`, `power_wd_hist_sincos`, `power_wd_yaw_hist_sincos`, `power_wd_yaw_pitchmean_hist_sincos`, `power_wd_yaw_lrpm_hist_sincos`, and `power_ws_wd_hist_sincos`
- `farm-synchronous` turbine panel with stable task-local `turbine_index`
- official AGCRN core architecture (`AVWGCN` + `AGCRNCell` + encoder + `end_conv`)
- train on `train`
- report both `val` and `test`
- report both `rolling_origin_no_refit` and `non_overlap`
- output one `overall` row and `36` horizon rows per eval view

Current exclusions:

- `sdwpf_kddcup`: the strict `24h -> 6h` farm-synchronous task currently has `0` usable windows
- `hill_of_towie`: the direction-aware variants currently fail during split selection because the resulting `train` split is empty

The alignment target is the model core only. Data loading, window filtering, split logic, and metrics remain in this repository's unified wind-farm evaluation framework.

## Data Contract

The runner loads the public task bundle through `wind_datasets.load_task_bundle(...)`.

For each supported dataset and active feature protocol, bundles are materialized under:

```text
cache/<dataset_id>/tasks/next_6h_from_24h/<feature_protocol_id>/series.parquet
cache/<dataset_id>/tasks/next_6h_from_24h/<feature_protocol_id>/window_index.parquet
cache/<dataset_id>/tasks/next_6h_from_24h/<feature_protocol_id>/static.parquet
cache/<dataset_id>/tasks/next_6h_from_24h/<feature_protocol_id>/task_context.json
```

The current supported `dataset_id` values are `kelmarsh` and `penmanshiel`.

`static.parquet` is treated as the complete experiment-facing static sidecar; the
runner does not read `silver` or `gold_base` artifacts directly.

Angle semantics remain dataset-side. In particular, `sdwpf_kddcup` now treats
`Wdir` as the documented relative yaw-error angle, reconstructs absolute wind
direction as `Ndir + Wdir` for the wind-direction protocols, and records the
exact transform used for each protocol in task bundle `task_context.json`.

Only windows satisfying all of the following are kept:

- `quality_flags == ""`
- `is_complete_input == true`
- `is_complete_output == true`
- `is_fully_synchronous_input == true`
- `is_fully_synchronous_output == true`

This strict filtering still happens inside the AGCRN family code rather than in
the dataset task cache.

When one CLI run includes multiple variants, the runner first computes each
variant's strict `train`/`val`/`test` rolling windows independently, then
intersects those splits on `(output_start_ts, output_end_ts)` before applying
`--max-train-origins` / `--max-eval-origins` and before deriving
`non_overlap`.

Running a single variant keeps that variant's own strict windows without any
cross-protocol shrinkage.

## Environment

Create or update the isolated experiment environment:

```bash
./create_env.sh
```

## Run

From this directory:

```bash
./.conda/bin/python run_agcrn.py
```

The default invocation now runs all nine active variants on both supported
datasets (`kelmarsh` and `penmanshiel`) with the tuned 2026-04-13 defaults:

- `official_aligned_power_only_farm_sync`
  - `batch_size=512`
  - `learning_rate=1e-3`
  - `max_epochs=20`
  - `early_stopping_patience=5`
  - `hidden_dim=64`
  - `embed_dim=10`
  - `num_layers=2`
  - `cheb_k=2`
- `official_aligned_power_ws_hist_farm_sync`
  - `batch_size=512`
  - `learning_rate=5e-4`
  - `max_epochs=20`
  - `early_stopping_patience=5`
  - `hidden_dim=64`
  - `embed_dim=16`
  - `num_layers=2`
  - `cheb_k=3`
- `official_aligned_power_atemp_hist_farm_sync`
  - `batch_size=512`
  - `learning_rate=1e-3`
  - `max_epochs=20`
  - `early_stopping_patience=5`
  - `hidden_dim=64`
  - `embed_dim=10`
  - `num_layers=2`
  - `cheb_k=2`
- `official_aligned_power_itemp_hist_farm_sync`
  - `batch_size=512`
  - `learning_rate=1e-3`
  - `max_epochs=20`
  - `early_stopping_patience=5`
  - `hidden_dim=64`
  - `embed_dim=10`
  - `num_layers=2`
  - `cheb_k=2`
- `official_aligned_power_wd_hist_sincos_farm_sync`
  - `batch_size=512`
  - `learning_rate=5e-4`
  - `max_epochs=20`
  - `early_stopping_patience=5`
  - `hidden_dim=64`
  - `embed_dim=10`
  - `num_layers=2`
  - `cheb_k=2`
- `official_aligned_power_wd_yaw_hist_sincos_farm_sync`
  - `batch_size=512`
  - `learning_rate=5e-4`
  - `max_epochs=20`
  - `early_stopping_patience=5`
  - `hidden_dim=64`
  - `embed_dim=10`
  - `num_layers=2`
  - `cheb_k=2`
- `official_aligned_power_wd_yaw_pitchmean_hist_sincos_farm_sync`
  - `batch_size=512`
  - `learning_rate=1e-3`
  - `max_epochs=20`
  - `early_stopping_patience=5`
  - `hidden_dim=64`
  - `embed_dim=10`
  - `num_layers=2`
  - `cheb_k=2`
- `official_aligned_power_wd_yaw_lrpm_hist_sincos_farm_sync`
  - `batch_size=512`
  - `learning_rate=1e-3`
  - `max_epochs=20`
  - `early_stopping_patience=5`
  - `hidden_dim=64`
  - `embed_dim=10`
  - `num_layers=2`
  - `cheb_k=2`
- `official_aligned_power_ws_wd_hist_sincos_farm_sync`
  - `batch_size=512`
  - `learning_rate=1e-3`
  - `max_epochs=20`
  - `early_stopping_patience=5`
  - `hidden_dim=64`
  - `embed_dim=10`
  - `num_layers=2`
  - `cheb_k=2`

Explicit CLI flags such as `--batch-size`, `--learning-rate`, `--epochs`,
`--patience`, `--embed-dim`, or `--cheb-k` still override these tuned defaults
for all selected variants.

This writes:

```text
../../artifacts/published/agcrn_official_aligned/latest.csv
../../artifacts/published/agcrn_official_aligned/latest.training_history.csv
```

For ad hoc smoke/debug runs, prefer an explicit `--output-path` under
`../../artifacts/scratch/agcrn_official_aligned/`.

Useful smoke-test options:

```bash
./.conda/bin/python run_agcrn.py --epochs 1 --device cpu --max-train-origins 64 --max-eval-origins 32
./.conda/bin/python run_agcrn.py --dataset penmanshiel --epochs 1 --device cpu --max-train-origins 64 --max-eval-origins 32
./.conda/bin/python run_agcrn.py --variant official_aligned_power_ws_hist_farm_sync --epochs 1 --device cpu --max-train-origins 64 --max-eval-origins 32
./.conda/bin/python run_agcrn.py --dataset penmanshiel --variant official_aligned_power_ws_hist_farm_sync --epochs 1 --device cpu --max-train-origins 64 --max-eval-origins 32
./.conda/bin/python run_agcrn.py --variant official_aligned_power_atemp_hist_farm_sync --epochs 1 --device cpu --max-train-origins 64 --max-eval-origins 32
./.conda/bin/python run_agcrn.py --variant official_aligned_power_itemp_hist_farm_sync --epochs 1 --device cpu --max-train-origins 64 --max-eval-origins 32
./.conda/bin/python run_agcrn.py --variant official_aligned_power_wd_hist_sincos_farm_sync --epochs 1 --device cpu --max-train-origins 64 --max-eval-origins 32
./.conda/bin/python run_agcrn.py --variant official_aligned_power_wd_yaw_hist_sincos_farm_sync --epochs 1 --device cpu --max-train-origins 64 --max-eval-origins 32
./.conda/bin/python run_agcrn.py --variant official_aligned_power_wd_yaw_pitchmean_hist_sincos_farm_sync --epochs 1 --device cpu --max-train-origins 64 --max-eval-origins 32
./.conda/bin/python run_agcrn.py --variant official_aligned_power_wd_yaw_lrpm_hist_sincos_farm_sync --epochs 1 --device cpu --max-train-origins 64 --max-eval-origins 32
./.conda/bin/python run_agcrn.py --variant official_aligned_power_ws_wd_hist_sincos_farm_sync --epochs 1 --device cpu --max-train-origins 64 --max-eval-origins 32
```

### Resume

`run_agcrn.py` now supports explicit resume for interrupted family runs:

```bash
./.conda/bin/python run_agcrn.py --resume
```

Resume state is family-local and keyed by the resolved `--output-path`:

```text
./.work/run_agcrn/<sha256(output_path)>/
  run_state.json
  partial_results.csv
  training_history.csv
  checkpoints/<dataset_id>__<model_variant>.pt
```

Behavior:

- `--resume` is required to continue an interrupted run; the default invocation still starts a fresh run
- completed `(dataset_id, model_variant)` jobs are skipped based on `partial_results.csv`
- completed epochs are recorded in `training_history.csv` and republished beside
  the result CSV as `<output-stem>.training_history.csv`
- the interrupted active job resumes from the latest epoch checkpoint when the matching `.pt` file exists
- if the active job has no checkpoint file, that job restarts from epoch `0`
- only epoch-boundary recovery is supported; there is no batch-level recovery

Resume is strict about configuration identity. The stored state must match:

- ordered `dataset_ids`
- ordered `variant_names`
- `seed`
- `max_train_origins`
- `max_eval_origins`
- each selected variant's resolved hyperparameter profile

Flags such as `--run-label` and `--no-record-run` are intentionally ignored for
resume matching because they do not change the training definition.

Typical interrupted-run flow:

```bash
./.conda/bin/python run_agcrn.py --output-path ../../artifacts/scratch/agcrn_official_aligned/debug.csv
./.conda/bin/python run_agcrn.py --output-path ../../artifacts/scratch/agcrn_official_aligned/debug.csv --resume
./.conda/bin/python run_agcrn.py --output-path ../../artifacts/scratch/agcrn_official_aligned/debug.csv --force-rerun
```

If a previous run for the same `--output-path` is still marked `running`, a
fresh invocation without `--resume` is rejected so the interrupted state is not
silently overwritten. Use `--resume` to continue that run, or `--force-rerun`
to discard the stored resume slot and start over from epoch `0`.

## Hyperparameter Search

Use the aligned search harness when you want to tune any subset of the active
variants without falling back to mismatched window sets.

The search script always prepares all nine active variants together,
intersects their strict `train`/`val`/`test` windows, and only then tunes the
requested variants on that shared split surface. This keeps search-time window
counts aligned with the default full-family rerun.

### Search Setup

The recorded 2026-04-13 pre-temperature full-family tuning pass on `kelmarsh`
used these shared split sizes:

- `train=275587`
- `val_rolling=43287`
- `val_non_overlap=1203`
- `test_rolling=82024`
- `test_non_overlap=2279`

The quick screening budget used for the recorded seven-variant pre-temperature pass was:

- `train_origins=32768`
- `eval_origins=4096`
- `epochs=4`
- `patience=2`

The 2026-04-13 screen compared these three high-value candidates for every
variant in that pass:

- `baseline_bs512_h64_e10_l2_k2_lr1e-3`
- `baseline_bs512_h64_e10_l2_k2_lr5e-4`
- `graph_bs512_h64_e16_l2_k3_lr5e-4`

### Search Outcome

The practical outcome was:

- `power_only`: `baseline_bs512_h64_e10_l2_k2_lr1e-3`
- `power_ws_hist`: kept the previously confirmed 2026-04-12 full-window graph profile `graph_bs512_h64_e16_l2_k3_lr5e-4`
- `power_atemp_hist`: `baseline_bs512_h64_e10_l2_k2_lr1e-3`
- `power_itemp_hist`: `baseline_bs512_h64_e10_l2_k2_lr1e-3`
- `power_wd_hist_sincos`: `baseline_bs512_h64_e10_l2_k2_lr5e-4`
- `power_wd_yaw_hist_sincos`: `baseline_bs512_h64_e10_l2_k2_lr5e-4`
- `power_wd_yaw_pitchmean_hist_sincos`: `baseline_bs512_h64_e10_l2_k2_lr1e-3`
- `power_wd_yaw_lrpm_hist_sincos`: `baseline_bs512_h64_e10_l2_k2_lr1e-3`
- `power_ws_wd_hist_sincos`: `baseline_bs512_h64_e10_l2_k2_lr1e-3`

The two temperature-history variants are new additions and currently inherit the
baseline profile pending a dedicated nine-variant rerun. In the recorded
seven-variant aligned screen, the graph-style candidate did not beat the
baseline-style candidate for the five newly tuned protocols. The one retained
graph exception is `power_ws_hist`, where the earlier 2026-04-12 full-window
confirmation remained stronger evidence than a short 4-epoch screen.

### Search Artifacts

The most relevant outputs now live under:

```text
../../artifacts/scratch/agcrn_official_aligned/search_20260413/
```

Useful result files:

- `full_family_screen_v1/screen_summary.csv`
  - seven-variant aligned screen results for the 2026-04-13 pass
- `full_family_screen_v1/search_plan.json`
  - exact screen budget and candidate list used for the seven-variant pass
- `search_20260412/po_bs512_full_v1/final_summary.csv`
  - full-window confirmation for the `power_only` default
- `search_20260412/ws_remaining_screen_v1/final_summary.csv`
  - full-window confirmation for the retained `power_ws_hist` graph default

Typical invocation:

```bash
./.conda/bin/python search_agcrn.py --device cuda
./.conda/bin/python search_agcrn.py --dataset penmanshiel --device cuda
```

This writes stage summaries under:

```text
../../artifacts/scratch/agcrn_official_aligned/search_20260413/
```

Key outputs:

- `screen_summary.csv`: validation-only screening results used for shortlist selection
- `final_summary.csv`: full-window confirmatory runs for the top screened configs
- `final_detailed_rows.csv`: long result rows matching the family output schema
- `search_plan.json`: search budget, candidate list, and stage settings

For faster iterations, use `--skip-final` or lower `--screen-train-origins` /
`--screen-eval-origins`.

## Output Schema

`../../artifacts/published/agcrn_official_aligned/latest.csv` is a long result file with:

- `split_name in {val, test}`
- `eval_protocol in {rolling_origin_no_refit, non_overlap}`
- `metric_scope in {overall, horizon}`

For one dataset and one eval view, runs that include multiple variants should
report matching window counts across all selected feature protocols after
alignment.

That yields `2 * 2 * (1 + 36) = 148` rows for one dataset/variant job.
Running all nine active variants on one supported dataset yields `9 * 148 = 1332`
rows. Running the default two-dataset invocation yields `2 * 9 * 148 = 2664`
rows.
