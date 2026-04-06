# Chronos-2 `power_only`

This experiment evaluates Chronos-2 zero-shot forecasting on:

- `kelmarsh`
- `penmanshiel`
- `hill_of_towie`

The task is fixed to `24h look back -> 6h ahead` with a `6h` stride and uses
only clipped turbine power histories.

It supports two evaluation layouts:

- `univariate`: each turbine is forecast independently
- `multivariate`: each wind farm is forecast as a single multivariate panel

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
./.conda/bin/python run_power_only.py --dataset kelmarsh --max-windows-per-dataset 64 --batch-size 32
```

Append multivariate farm-level rows into the shared CSV without rerunning the
existing univariate rows:

```bash
./.conda/bin/python run_power_only.py --mode append_multivariate --device mps
```
