# Official-Aligned AGCRN

This experiment runs an official-aligned AGCRN baseline on:

- `kelmarsh`

The scope is intentionally narrow:

- `24h look back -> 6h ahead`
- `power-only`
- `farm-synchronous` turbine panel with stable task-local `turbine_index`
- official AGCRN core architecture (`AVWGCN` + `AGCRNCell` + encoder + `end_conv`)
- train on `train`
- report both `val` and `test`
- report both `rolling_origin_no_refit` and `non_overlap`
- output one `overall` row and `36` horizon rows per eval view

The alignment target is the model core only. Data loading, window filtering, split logic, and metrics remain in this repository's unified wind-farm evaluation framework.

## Data Contract

The runner reads:

```text
cache/kelmarsh/gold_base/default/farm/default/series.parquet
cache/kelmarsh/tasks/default/farm/next_6h_from_24h/window_index.parquet
cache/kelmarsh/tasks/default/farm/next_6h_from_24h/turbine_static.parquet
```

Only windows satisfying all of the following are kept:

- `quality_flags == ""`
- `is_complete_input == true`
- `is_complete_output == true`
- `is_fully_synchronous_input == true`
- `is_fully_synchronous_output == true`

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

This writes:

```text
../agcrn-official-aligned.csv
```

Useful smoke-test options:

```bash
./.conda/bin/python run_agcrn.py --epochs 1 --device cpu --max-train-origins 64 --max-eval-origins 32
```

## Output Schema

`../agcrn-official-aligned.csv` is a long result file with:

- `split_name in {val, test}`
- `eval_protocol in {rolling_origin_no_refit, non_overlap}`
- `metric_scope in {overall, horizon}`

That yields `2 * 2 * (1 + 36) = 148` rows for the Kelmarsh official-aligned job.
