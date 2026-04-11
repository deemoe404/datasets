# Chronos-2 `exogenous`

This experiment evaluates Chronos-2 zero-shot forecasting on:

- `kelmarsh`
- `penmanshiel`
- `hill_of_towie`
- `sdwpf_kddcup`

The task is fixed to `24h look back -> 6h ahead` and uses only univariate
turbine targets.

Evaluation protocol:

- dense sliding windows on the raw turbine timestep
- raw unique timestamp chronological split `70/10/20`
- strict-contained windows within each split
- zero-shot scoring on `test` only
- both `rolling_origin_no_refit` and `non_overlap`
- long-form output with one `overall` row and `36` horizon rows per eval view

Compared with [`experiment/families/chronos-2`](../chronos-2/README.md), this runner:

- keeps the target on `target_kw` only
- does not use target-derived `power_stats`
- adds dataset-native past-only covariates through staged packs
- records pack metadata in the result CSV

Current scope:

- `layout=univariate`
- past-only covariates only
- no future covariates
- no `cross_learning`
- no multivariate `knn6`
- no high-cardinality status/alarm code one-hot packs

## Covariate Stages

Each dataset runs three staged packs:

- `stage1_core`
- `stage2_ops`
- `stage3_regime`

All packs now read the canonical turbine `gold_base` series and select their
required columns directly, without a separate `feature_set` cache layer.

Optional flag:

- `--include-power-only-reference` adds one `covariate_stage=reference` row per dataset.

## Environment

Create or update the isolated experiment environment:

```bash
./create_env.sh
```

The environment is stored under `./.conda/` and is ignored by git.

## Run

From this directory:

```bash
./.conda/bin/python run_exogenous.py
```

This writes:

```text
../../artifacts/published/chronos2_exogenous/latest.csv
```

For ad hoc smoke/debug runs, prefer an explicit `--output-path` under
`../../artifacts/scratch/chronos2_exogenous/`.

Useful smoke-test options:

```bash
./.conda/bin/python run_exogenous.py --dataset kelmarsh --covariate-stage stage1_core --max-windows-per-split 64
```

```bash
./.conda/bin/python run_exogenous.py --dataset hill_of_towie --covariate-stage stage3_regime --series-budget 256 --max-windows-per-split 32
```

Include a local power-only reference row:

```bash
./.conda/bin/python run_exogenous.py --dataset sdwpf_kddcup --include-power-only-reference --max-windows-per-split 32
```

Run the full staged benchmark serially:

```bash
./.conda/bin/python run_exogenous_full.py
```

The default full-run output is an `888`-row long CSV:

- `12` dataset/stage-pack jobs
- `2` test eval views
- `1 + 36` metric rows per eval view

With `--include-power-only-reference`, it expands to `1184` rows.

The full-run orchestrator automatically selects `cuda -> mps -> cpu`. When a
non-CPU chunk fails, it retries the same chunk with fixed series budgets
`1024 -> 768 -> 512` on the selected accelerator before falling back to
`cpu + 1024`. Pin the device explicitly when needed:

```bash
./.conda/bin/python run_exogenous_full.py --device cuda
```

Mini-bench note: on the `RTX 4090 D 24GB` check from `2026-04-09`, increasing
the non-CPU `series_budget` above `1024` made both light and heavy packs slower,
and `stage3_regime` started to hit CUDA OOM at higher budgets. The repo keeps
the fixed retry ladder `1024 -> 768 -> 512 -> cpu 1024` intentionally.

`--series-budget` remains available for backward compatibility, but it is only
honored when the full run resolves to `cpu`. Non-CPU full runs always use the
fixed retry ladder above.
