# Chronos-2 `power_only`

This experiment evaluates Chronos-2 zero-shot forecasting on:

- `kelmarsh`
- `penmanshiel`
- `hill_of_towie`
- `sdwpf_kddcup`

The task is fixed to `24h look back -> 6h ahead` and uses only turbine power
histories.

Evaluation protocol:

- dense sliding windows on the raw turbine timestep
- raw unique timestamp chronological split `70/10/20`
- strict-contained windows within each split
- zero-shot scoring on `test` only
- both `rolling_origin_no_refit` and `non_overlap`
- long-form output with one `overall` row and `36` horizon rows per eval view

It supports three evaluation layouts:

- `univariate`: each turbine is forecast independently and aggregated at the farm level
- `univariate_power_stats`: univariate target forecasting with same-turbine historical power-stat covariates on the three supported datasets
- `multivariate_knn6`: each target turbine is forecast with `self + 5 nearest turbines`,
  and only the target turbine forecast is scored
- `multivariate_knn6_power_stats`: the same `self + 5 nearest turbines` layout, with
  historical power-stat covariates for the three supported datasets

For `multivariate_knn6`, each local 6-turbine neighborhood is reindexed onto its
own full `10m` grid before Chronos-2 inference, so partially asynchronous turbine
histories are aligned with `NaN` gaps instead of being rejected.

For `multivariate_knn6_power_stats`, each local neighborhood is encoded as a
6-target multivariate panel plus flattened per-neighbor historical power-stat
covariates:

- `kelmarsh`: `min/max/stddev`
- `penmanshiel`: `min/max/stddev`
- `hill_of_towie`: `min/max/stddev/endvalue`

Invalid target points are defined as:

- `target_kw` is null
- `quality_flags != ""`

Those invalid points are passed to Chronos-2 as `NaN`, excluded from future
scoring, and all remaining valid targets are clipped to `[0, rated_power_kw]`.

## Environment

Create or update the isolated experiment environment:

```bash
./create_env.sh
```

The environment is stored under `./.conda/` and is ignored by git.

## Run

From this directory:

```bash
./.conda/bin/python run_power_only.py
```

This writes:

```text
../../artifacts/published/chronos2_power_only/latest.csv
```

For ad hoc smoke/debug runs, prefer an explicit `--output-path` under
`../../artifacts/scratch/chronos2_power_only/`.

Useful smoke-test options:

```bash
./.conda/bin/python run_power_only.py --dataset kelmarsh --mode all --max-windows-per-split 64 --batch-size 32
```

Run only one layout:

```bash
./.conda/bin/python run_power_only.py --dataset sdwpf_kddcup --mode multivariate_knn6 --batch-size 16
```

For the three datasets with power-stats support, `--mode multivariate_knn6`
emits both the plain `*_multivariate_knn6` rows and the added
`*_multivariate_knn6_power_stats` rows.

Run only the added power-stat covariate variant:

```bash
./.conda/bin/python run_power_only.py --mode univariate_power_stats --dataset hill_of_towie --batch-size 16
```

`univariate_power_stats` is supported only for `kelmarsh`, `penmanshiel`, and `hill_of_towie`.

To run the full aligned benchmark safely, use the repo-tracked orchestrator
instead of ad hoc shell loops. It runs serially, chunks the heavy datasets,
automatically selects `cuda -> mps -> cpu`, falls back to CPU when a non-CPU
chunk fails, can split heavy `multivariate_knn6` target groups into smaller
chunks, and writes chunk logs under a fresh `./.work/full-run-<timestamp>/`:

```bash
./.conda/bin/python run_power_only_full.py
```

The full-run output is a `1036`-row long CSV:

- `14` dataset/layout jobs
- `2` test eval views
- `1 + 36` metric rows per eval view

Pin the full run to a specific device when needed:

```bash
./.conda/bin/python run_power_only_full.py --device cuda
```

CUDA full-run defaults are tuned per chunk family rather than using the
runner-wide `--batch-size` default. On the `RTX 4090 D 24GB` mini-bench from
`2026-04-09`, the best stable primary chunk sizes were:

| full-run chunk family                            | datasets                                  | cuda batch size |
| ----------------------------------------------- | ----------------------------------------- | --------------: |
| `univariate`                                    | `kelmarsh`, `penmanshiel`                 |             128 |
| `univariate`                                    | `hill_of_towie` 3-turbine chunks          |             128 |
| `univariate`                                    | `sdwpf_kddcup` 16-turbine chunks          |             128 |
| `univariate_power_stats`                        | `kelmarsh`, `penmanshiel`                 |              32 |
| `univariate_power_stats`                        | `hill_of_towie` 3-turbine chunks          |              32 |
| `multivariate_knn6` and added power-stats rows  | `kelmarsh`, `penmanshiel`                 |              32 |
| `multivariate_knn6` and added power-stats rows  | `hill_of_towie` single-target groups      |              32 |
| `multivariate_knn6`                             | `sdwpf_kddcup` 8-target groups            |              32 |

`cpu` and `mps` still keep the older conservative chunk sizes because only the
CUDA profile was re-tuned.

Debug-only options:

- `--turbine-id` limits `univariate` runs to a turbine subset
- `--window-offset` skips retained windows before scoring
- `--max-windows-per-split` limits retained `test` windows per eval slice for smoke tests
