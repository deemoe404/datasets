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

Explicit `input_pack` mapping:

| `input_pack`    | target history | `static` | `known future` | staged historical exogenous |
| --------------- | -------------: | -------: | -------------: | --------------------------: |
| `reference`     |            yes |       no |             no |                        none |
| `known_static`  |            yes |      yes |            yes |                        none |
| `hist_stage1`   |            yes |       no |             no |               `stage1_core` |
| `hist_stage2`   |            yes |       no |             no |                `stage2_ops` |
| `mixed_stage1`  |            yes |      yes |            yes |               `stage1_core` |
| `mixed_stage2`  |            yes |      yes |            yes |                `stage2_ops` |

Relationship to the repository-wide staged covariate packs:

- `stage1_core`, `stage2_ops`, and `stage3_regime` are defined centrally in `experiment/common/covariate_packs.py`.
- TFT uses only `stage1_core` and `stage2_ops` for historical exogenous inputs.
- TFT does not currently include `stage3_regime`.

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

After an interruption, rerun with:

```bash
./.conda/bin/python run_tft.py --resume
```

`--resume` does two things:

- skips `input_pack` jobs that are already complete in `../tft-pilot.csv`
- resumes an unfinished training job from `./.work/jobs/<dataset>/<input_pack>/checkpoints/last.ckpt` when available

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

## Tuning Notes

The runner now exposes the main throughput-sensitive knobs directly:

- `--batch-size`
- `--num-workers`
- `--trainer-precision`
- `--matmul-precision`
- `--hidden-size`
- `--attention-head-size`
- `--hidden-continuous-size`
- `--dropout`
- `--gradient-clip-val`

On CUDA, `auto` now resolves to:

- `batch_size=256`
- `trainer_precision=bf16-mixed`
- `matmul_precision=high`
- `num_workers=0`

The default command now uses the safety-first CUDA starting point:

```bash
./.conda/bin/python run_tft.py
```

This resolves to:

- `batch_size=256`
- `num_workers=0`
- `trainer_precision=bf16-mixed`
- `matmul_precision=high`

The equivalent explicit command is:

```bash
./.conda/bin/python run_tft.py \
  --input-pack mixed_stage2 \
  --batch-size 256 \
  --num-workers 0 \
  --trainer-precision bf16-mixed \
  --matmul-precision high
```

This profile is intentionally conservative on host memory. If you want to push
throughput after confirming the machine stays stable, increase `--batch-size`
first and only then try `--num-workers 2`.
