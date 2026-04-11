# Official-Aligned AGCRN

This experiment runs an official-aligned AGCRN baseline on:

- `kelmarsh`

The scope is intentionally narrow:

- `24h look back -> 6h ahead`
- `power_only` and `power_ws_hist`
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
```

`static.parquet` is treated as the complete experiment-facing static sidecar; the
runner does not read `silver` or `gold_base` artifacts directly.

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

Current tuned defaults favor faster full runs on this workspace:

- `--device auto` (prefers CUDA when available)
- `--batch-size 1024`
- `--epochs 15`
- `--patience 4`

This writes:

```text
../../artifacts/published/agcrn_official_aligned/latest.csv
```

For ad hoc smoke/debug runs, prefer an explicit `--output-path` under
`../../artifacts/scratch/agcrn_official_aligned/`.

Useful smoke-test options:

```bash
./.conda/bin/python run_agcrn.py --epochs 1 --device cpu --max-train-origins 64 --max-eval-origins 32
./.conda/bin/python run_agcrn.py --variant official_aligned_power_ws_hist_farm_sync --epochs 1 --device cpu --max-train-origins 64 --max-eval-origins 32
```

## Output Schema

`../../artifacts/published/agcrn_official_aligned/latest.csv` is a long result file with:

- `split_name in {val, test}`
- `eval_protocol in {rolling_origin_no_refit, non_overlap}`
- `metric_scope in {overall, horizon}`

For one dataset and one eval view, runs that include both active variants should
now report matching window counts across `official_aligned_power_only_farm_sync`
and `official_aligned_power_ws_hist_farm_sync`.

That yields `2 * 2 * (1 + 36) = 148` rows for the Kelmarsh official-aligned job.
Running both active variants by default yields `2 * 148 = 296` rows.
