# LTSF-Linear V1

This experiment trains local `NLinear` and `DLinear` baselines on the repository
task `24h history -> 6h forecast -> 6h stride` using:

- `power-only`
- `turbine` granularity
- per-dataset pooled windows
- chronological `70/10/20` split on unique `output_start_ts`

It reads the existing turbine `gold_base` and turbine task cache under:

```text
cache/<dataset>/gold_base/default/turbine/default/
cache/<dataset>/tasks/default/turbine/next_6h_from_24h_stride_6h/
```

If the stride-`6h` task cache is missing, the runner attempts to build it through
`wind_datasets.build_task_cache(...)`. That path requires a valid
`wind_datasets.local.toml`.

## Environment

Create or update the isolated experiment environment:

```bash
./create_env.sh
```

## Run

From this directory:

```bash
./.conda/bin/python run_ltsf_linear.py
```

This overwrites:

```text
../ltsf-linear.csv
```

Run one dataset/model pair:

```bash
./.conda/bin/python run_ltsf_linear.py --dataset kelmarsh --model nlinear
```

Useful smoke-test options:

```bash
./.conda/bin/python run_ltsf_linear.py --dataset hill_of_towie --model dlinear --epochs 3 --max-windows-per-split 128
```

Run the full 8-row benchmark grid:

```bash
./.conda/bin/python run_ltsf_linear_full.py
```

Pin the full run to a device:

```bash
./.conda/bin/python run_ltsf_linear_full.py --device cuda
```

When a non-CPU full-run job fails, the orchestrator retries the same job once on
`cpu` and keeps per-job logs under `./.work/`.
