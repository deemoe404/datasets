# TFT Pilot

This experiment evaluates a first-phase Temporal Fusion Transformer pilot on:

- `kelmarsh`

The task is fixed to `24h look back -> 6h ahead` at turbine granularity and
keeps the repository's strict split/evaluation contract:

- raw unique timestamp chronological split `70/10/20`
- strict-contained windows within each split
- `val` and `test` both report `rolling_origin_no_refit` and `non_overlap`

The pilot intentionally changes only the training sampler:

- `train`: keep every `12`th strict origin per turbine (`2h` stride)
- `val/test`: unchanged repository protocol

## Input Packs

- `reference`
- `known_static`
- `hist_stage1`
- `hist_stage2`
- `mixed_stage1`
- `mixed_stage2`

Inputs are constrained on purpose:

- `static`: only normalized spatial coordinates
- `known future`: `relative_time_idx`, `tod_sin`, `tod_cos`, `dow_sin`, `dow_cos`
- `historical exogenous`: staged `stage1_core` or `stage2_ops`
- no `stage3_regime`
- no future weather or dispatch inputs

## Environment

Create or update the isolated experiment environment:

```bash
./create_env.sh
```

## Run

From this directory:

```bash
./.conda/bin/python run_tft.py
```

This writes:

```text
../tft-pilot.csv
```

Useful smoke-test examples:

```bash
./.conda/bin/python run_tft.py --input-pack reference --epochs 1 --max-train-origins 64
```

```bash
./.conda/bin/python run_tft.py --input-pack mixed_stage1 --epochs 1 --max-train-origins 64
```

The default all-pack run emits `888` rows:

- `6` input packs
- `2` splits
- `2` eval protocols
- `1 + 36` metric rows per eval view
