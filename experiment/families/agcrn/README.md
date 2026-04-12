# Official-Aligned AGCRN

This experiment runs an official-aligned AGCRN baseline on:

- `kelmarsh`

The scope is intentionally narrow:

- `24h look back -> 6h ahead`
- `power_only`, `power_ws_hist`, `power_wd_hist_sincos`, `power_wd_yaw_hist_sincos`, and `power_ws_wd_hist_sincos`
- `farm-synchronous` turbine panel with stable task-local `turbine_index`
- official AGCRN core architecture (`AVWGCN` + `AGCRNCell` + encoder + `end_conv`)
- train on `train`
- report both `val` and `test`
- report both `rolling_origin_no_refit` and `non_overlap`
- output one `overall` row and `36` horizon rows per eval view

The alignment target is the model core only. Data loading, window filtering, split logic, and metrics remain in this repository's unified wind-farm evaluation framework.

## Data Contract

The runner loads the public task bundle through `wind_datasets.load_task_bundle(...)`.

For the current Kelmarsh run, those bundles are materialized under:

```text
cache/kelmarsh/tasks/next_6h_from_24h/power_only/series.parquet
cache/kelmarsh/tasks/next_6h_from_24h/power_only/window_index.parquet
cache/kelmarsh/tasks/next_6h_from_24h/power_only/static.parquet
cache/kelmarsh/tasks/next_6h_from_24h/power_only/task_context.json
cache/kelmarsh/tasks/next_6h_from_24h/power_ws_hist/series.parquet
cache/kelmarsh/tasks/next_6h_from_24h/power_ws_hist/window_index.parquet
cache/kelmarsh/tasks/next_6h_from_24h/power_ws_hist/static.parquet
cache/kelmarsh/tasks/next_6h_from_24h/power_ws_hist/task_context.json
cache/kelmarsh/tasks/next_6h_from_24h/power_wd_hist_sincos/series.parquet
cache/kelmarsh/tasks/next_6h_from_24h/power_wd_hist_sincos/window_index.parquet
cache/kelmarsh/tasks/next_6h_from_24h/power_wd_hist_sincos/static.parquet
cache/kelmarsh/tasks/next_6h_from_24h/power_wd_hist_sincos/task_context.json
cache/kelmarsh/tasks/next_6h_from_24h/power_wd_yaw_hist_sincos/series.parquet
cache/kelmarsh/tasks/next_6h_from_24h/power_wd_yaw_hist_sincos/window_index.parquet
cache/kelmarsh/tasks/next_6h_from_24h/power_wd_yaw_hist_sincos/static.parquet
cache/kelmarsh/tasks/next_6h_from_24h/power_wd_yaw_hist_sincos/task_context.json
cache/kelmarsh/tasks/next_6h_from_24h/power_ws_wd_hist_sincos/series.parquet
cache/kelmarsh/tasks/next_6h_from_24h/power_ws_wd_hist_sincos/window_index.parquet
cache/kelmarsh/tasks/next_6h_from_24h/power_ws_wd_hist_sincos/static.parquet
cache/kelmarsh/tasks/next_6h_from_24h/power_ws_wd_hist_sincos/task_context.json
```

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

The default invocation now runs all five active variants. The tuned 2026-04-12
full-run profiles are still used for the original two searched variants, and
the three wind-direction variants currently reuse the `power_ws_hist`
hyperparameter profile until they are tuned separately:

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
- `official_aligned_power_wd_hist_sincos_farm_sync`
  - currently reuses the `official_aligned_power_ws_hist_farm_sync` profile
- `official_aligned_power_wd_yaw_hist_sincos_farm_sync`
  - currently reuses the `official_aligned_power_ws_hist_farm_sync` profile
- `official_aligned_power_ws_wd_hist_sincos_farm_sync`
  - currently reuses the `official_aligned_power_ws_hist_farm_sync` profile

Explicit CLI flags such as `--batch-size`, `--learning-rate`, `--epochs`,
`--patience`, `--embed-dim`, or `--cheb-k` still override these tuned defaults
for all selected variants.

This writes:

```text
../../artifacts/published/agcrn_official_aligned/latest.csv
```

For ad hoc smoke/debug runs, prefer an explicit `--output-path` under
`../../artifacts/scratch/agcrn_official_aligned/`.

This repository change only updates code, docs, and tests. The checked-in
`../../artifacts/published/agcrn_official_aligned/latest.csv` has not been
refreshed for the new variant yet.

Useful smoke-test options:

```bash
./.conda/bin/python run_agcrn.py --epochs 1 --device cpu --max-train-origins 64 --max-eval-origins 32
./.conda/bin/python run_agcrn.py --variant official_aligned_power_ws_hist_farm_sync --epochs 1 --device cpu --max-train-origins 64 --max-eval-origins 32
./.conda/bin/python run_agcrn.py --variant official_aligned_power_wd_hist_sincos_farm_sync --epochs 1 --device cpu --max-train-origins 64 --max-eval-origins 32
./.conda/bin/python run_agcrn.py --variant official_aligned_power_wd_yaw_hist_sincos_farm_sync --epochs 1 --device cpu --max-train-origins 64 --max-eval-origins 32
```

## Hyperparameter Search

Use the aligned search harness when you want to tune the two already-searched
variants separately without falling back to mismatched window sets.

The search script always prepares `power_only` and `power_ws_hist` together,
intersects their strict `train`/`val`/`test` windows, and only then tunes each
variant on that shared split surface. It does not yet tune
`power_wd_hist_sincos`, `power_wd_yaw_hist_sincos`, or `power_ws_wd_hist_sincos`.

### Search Setup

The 2026-04-12 search used the aligned window fix now present on `HEAD`. With
that fix, both feature protocols share exactly the same full-window split sizes
for Kelmarsh:

- `train=292161`
- `val_rolling=43473`
- `val_non_overlap=1208`
- `test_rolling=82597`
- `test_non_overlap=2295`

The screening and confirmation budgets were:

- screen: `train_origins=65536`, `eval_origins=8192`, `epochs=10`, `patience=3`
- final confirm: `epochs=20`, `patience=5`

The search covered these candidate families:

- `power_only`
  - `baseline_bs1024_h64_e10_l2_k2_lr1e-3`
  - `baseline_bs512_h64_e10_l2_k2_lr1e-3`
  - `compact_bs1024_h48_e8_l1_k2_lr2e-3`
  - `larger_bs512_h96_e16_l2_k2_lr1e-3`
  - `graph_bs512_h64_e16_l2_k3_lr5e-4`
- `power_ws_hist`
  - `baseline_bs1024_h64_e10_l2_k2_lr1e-3`
  - `baseline_bs512_h64_e10_l2_k2_lr1e-3`
  - `baseline_bs512_h64_e10_l2_k2_lr5e-4`
  - `compact_bs1024_h48_e8_l1_k2_lr2e-3`
  - `compact_bs512_h48_e8_l1_k2_lr1e-3`
  - `larger_bs512_h96_e16_l2_k2_lr1e-3`
  - `larger_bs512_h96_e16_l2_k2_lr5e-4`
  - `graph_bs512_h64_e16_l2_k3_lr5e-4`

### Search Outcome

The alignment fix alone removed the earlier anomaly where adding wind-speed
history looked harmful. Under the same default baseline hyperparameters,
`power_ws_hist` already beats `power_only` on the full aligned windows:

| Variant | Config | Test Rolling RMSE PU | Test Non-Overlap RMSE PU |
| --- | --- | ---: | ---: |
| `power_only` | `baseline_bs1024_h64_e10_l2_k2_lr1e-3` | `0.170926449` | `0.172388097` |
| `power_ws_hist` | `baseline_bs1024_h64_e10_l2_k2_lr1e-3` | `0.170529890` | `0.171792055` |

After tuning, the strongest confirmed full-window configs were:

| Variant | Best Config | Test Rolling RMSE PU | Test Non-Overlap RMSE PU |
| --- | --- | ---: | ---: |
| `power_only` | `baseline_bs512_h64_e10_l2_k2_lr1e-3` | `0.170218769` | `0.171469075` |
| `power_ws_hist` | `graph_bs512_h64_e16_l2_k3_lr5e-4` | `0.169534975` | `0.171100681` |

Relative to the best tuned `power_only` run, the best tuned `power_ws_hist` run
improves:

- rolling RMSE PU by `0.000683795`
- non-overlap RMSE PU by `0.000368394`

The main practical conclusions from this search are:

- the earlier regression was primarily a window-alignment artifact
- `power_ws_hist` benefits from a different hyperparameter profile than `power_only`
- the strongest `power_ws_hist` setting uses `cheb_k=3` and `embed_dim=16`
- simply increasing hidden size to `96` did not help either protocol

### Search Artifacts

All outputs live under:

```text
../../artifacts/scratch/agcrn_official_aligned/search_20260412/
```

The most useful result files from this search are:

- `full_baseline_compare_v1/final_summary.csv`
  - aligned full-window comparison for the old shared baseline config
- `po_bs512_full_v1/final_summary.csv`
  - full-window confirmation for the best `power_only` config
- `po_remaining_screen_v1/screen_summary.csv`
  - remaining `power_only` screening candidates
- `ws_bs512_full_v1/final_summary.csv`
  - full-window confirmations for the strongest baseline-style `power_ws_hist` configs
- `ws_focus_screen_v1/screen_summary.csv`
  - focused `power_ws_hist` baseline-style screening results
- `ws_remaining_screen_v1/screen_summary.csv`
  - remaining `power_ws_hist` screening candidates, including the winning graph config
- `ws_remaining_screen_v1/final_summary.csv`
  - full-window confirmation for the winning `power_ws_hist` graph config

Typical invocation:

```bash
./.conda/bin/python search_agcrn.py --device cuda
```

This writes stage summaries under:

```text
../../artifacts/scratch/agcrn_official_aligned/search_20260412/
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

That yields `2 * 2 * (1 + 36) = 148` rows for the Kelmarsh official-aligned job.
Running all five active variants by default yields `5 * 148 = 740` rows.
