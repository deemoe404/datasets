# Chronos-2 `power_only`

This experiment evaluates Chronos-2 zero-shot forecasting on:

- `kelmarsh`
- `penmanshiel`
- `hill_of_towie`
- `sdwpf_kddcup`

The task is fixed to `24h look back -> 6h ahead` with a `6h` stride and uses
only turbine power histories.

It supports two evaluation layouts:

- `univariate`: each turbine is forecast independently and aggregated at the farm level
- `multivariate_knn6`: each target turbine is forecast with `self + 5 nearest turbines`,
  and only the target turbine forecast is scored

For `multivariate_knn6`, each local 6-turbine neighborhood is reindexed onto its
own full `10m` grid before Chronos-2 inference, so partially asynchronous turbine
histories are aligned with `NaN` gaps instead of being rejected.

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

This overwrites:

```text
/Users/sam/Developer/Code/Wind Power Forecasting/datasets/experiment/chronos-2.csv
```

Useful smoke-test options:

```bash
./.conda/bin/python run_power_only.py --dataset kelmarsh --mode all --max-windows-per-dataset 64 --batch-size 32
```

Run only one layout:

```bash
./.conda/bin/python run_power_only.py --dataset sdwpf_kddcup --mode multivariate_knn6 --batch-size 16
```

To run the full 8-row benchmark safely, use the repo-tracked orchestrator instead
of ad hoc shell loops. It runs serially, chunks the heavy datasets, falls back to
CPU when an MPS chunk fails, can split heavy `multivariate_knn6` target groups
into smaller chunks, and writes chunk logs under `./.work/`:

```bash
./.conda/bin/python run_power_only_full.py
```

Debug-only options:

- `--turbine-id` limits `univariate` runs to a turbine subset
- `--window-offset` skips retained windows before scoring
