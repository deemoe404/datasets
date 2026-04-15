# world_model_rollout_v1

This family runs a deterministic filtering-plus-rollout world model on:

- `kelmarsh`
- `penmanshiel`

Its current scope is intentionally narrow:

- `24h look back -> 6h ahead`
- single feature protocol: `world_model_v1`
- single model variant: `world_model_rollout_v1_farm_sync`
- `farm-synchronous` turbine panel with stable task-local `turbine_index`
- train on `train`
- report both `val` and `test`
- report both `rolling_origin_no_refit` and `non_overlap`

The runner consumes the full `world_model_v1` task bundle:

- `series`
- `window_index`
- `known_future`
- `static`
- `pairwise`

Experiment-side contract:

- local history input is rebuilt into mask-aware per-turbine observations
- history context uses real farm-level observations plus calendar features
- future context uses zero-filled farm observations with mask=`1` plus calendar features
- static input uses `latitude`, `longitude`, `elevation_m`, `rated_power_kw`, `hub_height_m`, `rotor_diameter_m`
- pairwise input uses directed geometry with `edge_attr[dst, src] = src -> dst`
- loss and metrics are accumulated only on valid target positions
- windows are kept when the forecast horizon contains at least one valid target

## Environment

Create or update the isolated experiment environment:

```bash
./create_env.sh
```

## Run

From this directory:

```bash
./.conda/bin/python run_world_model_rollout_v1.py
```

The default invocation runs the only active variant on both supported datasets
and writes:

```text
../../artifacts/published/world_model_rollout_v1/latest.csv
../../artifacts/published/world_model_rollout_v1/latest.training_history.csv
```

Default CUDA-oriented profiles are dataset-specific because the dense message
passing memory scales with `batch_size * node_count^2`:

- `kelmarsh`: `batch_size=256`
- `penmanshiel`: `batch_size=64`
- both datasets: `num_workers=0`

For ad hoc smoke runs, prefer an explicit output path under
`../../artifacts/scratch/world_model_rollout_v1/`.

Useful smoke-test options:

```bash
./.conda/bin/python run_world_model_rollout_v1.py --epochs 1 --device cpu --max-train-origins 64 --max-eval-origins 32
./.conda/bin/python run_world_model_rollout_v1.py --dataset penmanshiel --epochs 1 --device cpu --max-train-origins 64 --max-eval-origins 32
```

## Resume

`run_world_model_rollout_v1.py` supports explicit resume for interrupted family
runs:

```bash
./.conda/bin/python run_world_model_rollout_v1.py --resume
```

Resume state is family-local and keyed by the resolved `--output-path`:

```text
./.work/run_world_model_rollout_v1/<sha256(output_path)>/
  run_state.json
  partial_results.csv
  training_history.csv
  checkpoints/<dataset_id>__<model_variant>.pt
```

`training_history.csv` records one row per completed epoch and is republished
beside the selected output CSV as `<output-stem>.training_history.csv`. In this
family it includes total, future, history-prior, auxiliary, and consistency
loss columns.

Formal run records are written under
`experiment/artifacts/runs/world_model_rollout_v1/<timestamp>/manifest.json`
unless `--no-record-run` is set.
