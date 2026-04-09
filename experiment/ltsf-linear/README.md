# LTSF-Linear Exogenous V1

This experiment trains local `NLinear` and `DLinear` baselines on the repository
task `24h history -> 6h forecast -> 6h stride` using:

- `turbine` granularity
- per-dataset pooled windows
- chronological `70/10/20` split on unique `output_start_ts`
- one shared result CSV that contains both `power_only` reference rows and
  staged exogenous rows

The exogenous input protocol matches `experiment/chronos-2-exogenous`:

- `stage1_core`
- `stage2_ops`
- `stage3_regime`
- past-only covariates only
- no future covariates

Greenbyte datasets use `feature_set="lightweight"`. `hill_of_towie` and
`sdwpf_kddcup` use `feature_set="default"`.

The experiment reads turbine `gold_base` and the turbine task cache under:

```text
cache/<dataset>/gold_base/default/turbine/<feature_set>/
cache/<dataset>/tasks/default/turbine/next_6h_from_24h_stride_6h/
```

If a required turbine `gold_base` feature set or the stride-`6h` task cache is
missing, the runner attempts to build it through `wind_datasets`.

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

Run one dataset/model pair with a single exogenous stage:

```bash
./.conda/bin/python run_ltsf_linear.py --dataset kelmarsh --model nlinear --covariate-stage stage1_core --no-power-only-reference
```

Run only the local power-only reference row:

```bash
./.conda/bin/python run_ltsf_linear.py --dataset kelmarsh --model nlinear --reference-only
```

Useful smoke-test options:

```bash
./.conda/bin/python run_ltsf_linear.py --dataset hill_of_towie --model dlinear --covariate-stage stage3_regime --epochs 3 --max-windows-per-split 128
```

Run the full 32-row benchmark grid:

```bash
./.conda/bin/python run_ltsf_linear_full.py
```

Pin the full run to a device:

```bash
./.conda/bin/python run_ltsf_linear_full.py --device cuda
```

When a non-CPU full-run job fails, the orchestrator retries the same job once on
`cpu` and keeps per-job logs under `./.work/`.
