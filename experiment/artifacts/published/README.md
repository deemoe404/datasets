# Published Outputs

This directory is the canonical home for current formal result tables produced
by experiment families.

Versioned runtime contents:

- `experiment/artifacts/published/<family_id>/latest.csv`
- `experiment/artifacts/published/<family_id>/latest.training_history.csv`

Rules:

- `latest.csv` is the current family-level published result artifact, not a
  historical archive.
- `latest.training_history.csv` is the matching epoch-level train-loss and
  validation-RMSE history for the current published result.
- provenance lives in `experiment/artifacts/runs/<family_id>/<timestamp>/manifest.json`
- ad hoc smoke/debug outputs should go under `experiment/artifacts/scratch/`
- resume and chunk artifacts stay under `experiment/families/<family>/.work/`

For dissertation-grade reruns, keep the default `published/` location and use
`--run-label` so the matching manifest is easy to audit.
