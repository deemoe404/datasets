# LTSF-Linear Exogenous V2

This experiment trains local `NLinear` and `DLinear` baselines on the dense
repository task:

- `24h history -> 6h forecast`
- `turbine` granularity
- dense sliding windows at the raw turbine timestep
- per-dataset pooled windows

The runner is dense-only in this phase. It no longer keeps a legacy `6h stride`
mode.

## Split And Evaluation Protocol

The raw turbine timeline is split first, before window construction:

- chronological `70/10/20` on unique raw timestamps
- floor rounding on split boundaries
- strict-contained windows only

`strict-contained` means the full `24h` history and full `6h` forecast must both
stay inside the same split. `val` and `test` windows are not allowed to borrow
history from the preceding split.

Training and evaluation then differ:

- `train`: dense windows, stride `1`
- `val` / `test`: two evaluation protocols from the same trained checkpoint
- `rolling_origin_no_refit`: dense stride `1`, train once, no rolling retrain
- `non_overlap`: every `36`th eligible origin, aligned from the first eligible
  dense origin inside the split

Early stopping uses `val` `rolling_origin_no_refit` `overall RMSE_pu`.

## Covariates

The exogenous input protocol matches `experiment/chronos-2-exogenous`:

- `stage1_core`
- `stage2_ops`
- `stage3_regime`
- past-only covariates only
- no future covariates

Greenbyte datasets use `feature_set="lightweight"`. `hill_of_towie` and
`sdwpf_kddcup` use `feature_set="default"`.

Past covariates use train-only z-score normalization with missing-value fill `0`
and appended missing masks.

## Cache Inputs

The experiment reads turbine `gold_base` and the dense turbine task cache under:

```text
cache/<dataset>/gold_base/default/turbine/<feature_set>/
cache/<dataset>/tasks/default/turbine/next_6h_from_24h/
```

If a required turbine `gold_base` feature set or dense task cache is missing,
the runner attempts to build it through `wind_datasets`.

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

Run only the local power-only reference rows:

```bash
./.conda/bin/python run_ltsf_linear.py --dataset kelmarsh --model nlinear --reference-only
```

Useful smoke-test options:

```bash
./.conda/bin/python run_ltsf_linear.py --dataset hill_of_towie --model dlinear --covariate-stage stage3_regime --epochs 3 --max-windows-per-split 128
```

`--max-windows-per-split` now clips each split's dense base eligible origins
before the `non_overlap` thinning step.

Run the full benchmark grid:

```bash
./.conda/bin/python run_ltsf_linear_full.py
```

The full-run orchestrator automatically selects `cuda -> mps -> cpu`.
Pin the full run to a device:

```bash
./.conda/bin/python run_ltsf_linear_full.py --device cuda
```

By default the full run reruns jobs and overwrites any old chunk CSVs under
`./.work/chunks/`. Use explicit resume mode only when you want to reuse
compatible chunk outputs:

```bash
./.conda/bin/python run_ltsf_linear_full.py --reuse-existing-chunks
```

When a non-CPU full-run job fails, the orchestrator retries the same job once on
`cpu` and keeps per-job logs under `./.work/`.

## Output Schema

`../ltsf-linear.csv` is now a long result file. Each dataset / covariate pack /
model job emits:

- `split_name in {val, test}`
- `eval_protocol in {rolling_origin_no_refit, non_overlap}`
- `metric_scope in {overall, horizon}`

That yields `2 * 2 * (1 + 36) = 148` rows per job and `4736` rows for the full
32-job grid.
