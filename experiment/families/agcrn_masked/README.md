# AGCRN Masked

This family runs a masked AGCRN smoke-test on:

- `kelmarsh`
- `penmanshiel`

Its scope is intentionally narrow:

- `24h look back -> 6h ahead`
- single feature protocol: `power_wd_yaw_pmean_hist_sincos_masked`
- single model variant: `masked_power_wd_yaw_pmean_hist_sincos_farm_sync`
- `farm-synchronous` turbine panel with stable task-local `turbine_index`
- official AGCRN core architecture (`AVWGCN` + `AGCRNCell` + encoder + `end_conv`)
- train on `train`
- report both `val` and `test`
- report both `rolling_origin_no_refit` and `non_overlap`

This family exists to validate that masked task bundles can pass through the
full experiment runner without changing the dataset generation layer. It is not
intended to remain comparable to `agcrn_official_aligned`.

## Masked Contract

The runner still loads the public task bundle through
`wind_datasets.load_task_bundle(...)`, but it rebuilds the masked training
contract inside the experiment code:

- source value channels are normalized on finite train-history values only
- source value-channel nulls are replaced with `0`
- mask channels remain raw `0/1` inputs and are not normalized
- output validity is reconstructed as `isfinite(target_pu)`
- loss and metrics are accumulated only on valid target positions
- windows are kept when the forecast horizon contains at least one valid target

This family therefore does not use the baseline AGCRN strict-window filter.

## Environment

Create or update the isolated experiment environment:

```bash
./create_env.sh
```

## Run

From this directory:

```bash
./.conda/bin/python run_agcrn_masked.py
```

The default invocation runs the only active variant on both supported datasets
and writes:

```text
../../artifacts/published/agcrn_masked/latest.csv
../../artifacts/published/agcrn_masked/latest.training_history.csv
```

For ad hoc smoke runs, prefer an explicit output path under
`../../artifacts/scratch/agcrn_masked/`.

Useful smoke-test options:

```bash
./.conda/bin/python run_agcrn_masked.py --epochs 1 --device cpu --max-train-origins 64 --max-eval-origins 32
./.conda/bin/python run_agcrn_masked.py --dataset penmanshiel --epochs 1 --device cpu --max-train-origins 64 --max-eval-origins 32
./.conda/bin/python run_agcrn_masked.py --variant masked_power_wd_yaw_pmean_hist_sincos_farm_sync --epochs 1 --device cpu --max-train-origins 64 --max-eval-origins 32
```

## Resume

`run_agcrn_masked.py` supports explicit resume for interrupted family runs:

```bash
./.conda/bin/python run_agcrn_masked.py --resume
```

Resume state is family-local and keyed by the resolved `--output-path`:

```text
./.work/run_agcrn_masked/<sha256(output_path)>/
  run_state.json
  partial_results.csv
  training_history.csv
  checkpoints/<dataset_id>__<model_variant>.pt
```

`training_history.csv` records one row per completed epoch and is republished
beside the selected output CSV as `<output-stem>.training_history.csv`.

Formal run records are written under
`experiment/artifacts/runs/agcrn_masked/<timestamp>/manifest.json` unless
`--no-record-run` is set.
