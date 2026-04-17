# world_model_agcrn_v1

This family runs a geometry-aware seq2seq AGCRN on:

- `kelmarsh`
- `penmanshiel`

Its active scope is intentionally narrow:

- `24h look back -> 6h ahead`
- single feature protocol: `world_model_v1`
- single model variant: `world_model_v1_seq2seq_farm_sync`
- `farm-synchronous` turbine panel with stable task-local `turbine_index`
- train on `train`
- report both `val` and `test`
- report both `rolling_origin_no_refit` and `non_overlap`

This family exists to validate that the full `world_model_v1` task-bundle
surface can pass through a real trainable AGCRN family without modifying the
dataset generation layer.

## Protocol Contract

The runner consumes the full `world_model_v1` bundle:

- `series`
- `window_index`
- `known_future`
- `static`
- `pairwise`

Experiment-side contract:

- history input is fixed to 52 channels: `target_pu` + `target_kw__mask` + 25 value columns + 25 companion mask columns
- source value channels are normalized on finite train-history values only
- source value-channel nulls are replaced with `0`
- mask channels remain raw `0/1` inputs and are not normalized
- `known_future` uses the 7 protocol-defined calendar features directly in the decoder
- `static` is reduced to numeric node geometry/turbine fields and normalized per feature
- `pairwise` is reduced to numeric directed geometry features, with `bearing_deg` expanded to `sin/cos`
- output validity is reconstructed as `isfinite(target_pu)`
- loss and metrics are accumulated only on valid target positions
- windows are kept when the forecast horizon contains at least one valid target

This family therefore does not use the baseline AGCRN strict-window filter.

## Environment

Create or update the isolated experiment environment:

```bash
./create_env.sh
```

`create_env.sh` now runs a post-install CUDA probe and fails if the family env
ends up with a CPU-only torch build or if a visible NVIDIA driver exists but
PyTorch still cannot see a CUDA device.

Important:

- the repository root `./.conda` is the dataset-processing/test environment and may remain CPU-only
- the GPU-capable experiment environment for this family is `experiment/families/world_model_agcrn_v1/.conda`

Manual verification:

```bash
./.conda/bin/python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available(), torch.cuda.device_count())"
```

## Run

From this directory:

```bash
./.conda/bin/python run_world_model_agcrn_v1.py
```

The default invocation runs the only active variant on both supported datasets
and writes:

```text
../../artifacts/published/world_model_agcrn_v1/<run_timestamp>.csv
../../artifacts/published/world_model_agcrn_v1/<run_timestamp>.training_history.csv
```

For ad hoc smoke runs, prefer an explicit output path under
`../../artifacts/scratch/world_model_agcrn_v1/`.

Useful smoke-test options:

```bash
./.conda/bin/python run_world_model_agcrn_v1.py --epochs 1 --device cpu --max-train-origins 64 --max-eval-origins 32
./.conda/bin/python run_world_model_agcrn_v1.py --dataset penmanshiel --epochs 1 --device cpu --max-train-origins 64 --max-eval-origins 32
./.conda/bin/python run_world_model_agcrn_v1.py --variant world_model_v1_seq2seq_farm_sync --epochs 1 --device cpu --max-train-origins 64 --max-eval-origins 32
```

## Search

The family-local search harness writes artifacts under:

```text
../../artifacts/scratch/world_model_agcrn_v1/search_<YYYYMMDD>/
```

Invoke it with:

```bash
./.conda/bin/python search_world_model_agcrn_v1.py
```

The checked-in dataset-specific defaults are currently:

- `kelmarsh`: `batch_size=512`, `learning_rate=1e-3`, `hidden_dim=64`, `embed_dim=16`, `num_layers=2`, `cheb_k=2`
- `penmanshiel`: `batch_size=512`, `learning_rate=5e-4`, `hidden_dim=64`, `embed_dim=24`, `num_layers=2`, `cheb_k=3`

For GPU throughput, the runner now also:

- auto-scales eval batches above the train batch unless you pass `--eval-batch-size`
- enables CUDA `pin_memory` and a non-zero worker count by default unless you pass `--num-workers`
- enables TF32 matmul/cudnn fast paths on CUDA

Latest formal rerun label: not recorded yet in this changeset.
Latest formal publish timestamp: not recorded yet in this changeset.

## Resume

`run_world_model_agcrn_v1.py` supports explicit resume for interrupted family
runs:

```bash
./.conda/bin/python run_world_model_agcrn_v1.py --output-path ../../artifacts/published/world_model_agcrn_v1/<run_timestamp>.csv --resume
```

Resume state is family-local and keyed by the resolved `--output-path`:

```text
./.work/run_world_model_agcrn_v1/<sha256(output_path)>/
  run_state.json
  partial_results.csv
  training_history.csv
  checkpoints/<dataset_id>__<model_variant>.pt
```

`training_history.csv` records one row per completed epoch and is republished
beside the selected output CSV as `<output-stem>.training_history.csv`.
For default timestamped publish outputs, both `--resume` and `--force-rerun`
must pass the exact prior `--output-path`.

Formal run records are written under
`experiment/artifacts/runs/world_model_agcrn_v1/<timestamp>/manifest.json`
unless `--no-record-run` is set.
