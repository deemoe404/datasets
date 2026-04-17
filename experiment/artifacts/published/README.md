# Published Outputs

This directory is the canonical home for formal result tables produced by
experiment families.

Versioned runtime contents:

- `experiment/artifacts/published/<family_id>/<run_timestamp>.csv`
- `experiment/artifacts/published/<family_id>/<run_timestamp>.training_history.csv`

Rules:

- `<run_timestamp>` uses the UTC run-start stem `%Y%m%d-%H%M%S`.
- every formal default invocation publishes a new timestamped artifact pair
  instead of overwriting a `latest.csv`.
- `<run_timestamp>.training_history.csv` is the matching epoch-level train-loss
  and validation-RMSE history for that published result.
- provenance lives in `experiment/artifacts/runs/<family_id>/<timestamp>/manifest.json`
- ad hoc smoke/debug outputs should go under `experiment/artifacts/scratch/`
- resume and chunk artifacts stay under `experiment/families/<family>/.work/`
- default `--resume` and `--force-rerun` flows must pass the exact
  `--output-path` because the default publish path is timestamped

For dissertation-grade reruns, keep the default `published/` location and use
`--run-label` so the matching manifest is easy to audit.
