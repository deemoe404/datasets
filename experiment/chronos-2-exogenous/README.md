# Chronos-2 `exogenous`

This experiment evaluates Chronos-2 zero-shot forecasting on:

- `kelmarsh`
- `penmanshiel`
- `hill_of_towie`
- `sdwpf_kddcup`

The task is fixed to `24h look back -> 6h ahead` with a `6h` stride and uses
only univariate turbine targets.

Compared with `experiment/chronos-2`, this runner:

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

Greenbyte (`kelmarsh`, `penmanshiel`) uses `feature_set="lightweight"`.
`hill_of_towie` and `sdwpf_kddcup` use `feature_set="default"`.

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

This overwrites:

```text
/Users/sam/Developer/Code/Wind Power Forecasting/datasets/experiment/chronos-2-exogenous.csv
```

Useful smoke-test options:

```bash
./.conda/bin/python run_exogenous.py --dataset kelmarsh --covariate-stage stage1_core --max-windows-per-dataset 64
```

```bash
./.conda/bin/python run_exogenous.py --dataset hill_of_towie --covariate-stage stage3_regime --series-budget 256 --max-windows-per-dataset 32
```

Include a local power-only reference row:

```bash
./.conda/bin/python run_exogenous.py --dataset sdwpf_kddcup --include-power-only-reference --max-windows-per-dataset 32
```

Run the full staged benchmark serially:

```bash
./.conda/bin/python run_exogenous_full.py
```
