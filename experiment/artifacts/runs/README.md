# Formal Run Records

This directory is reserved for canonical run records created by formal
experiment CLI entrypoints.

Versioned runtime contents:

- `experiment/artifacts/runs/<family_id>/<timestamp>/manifest.json`

Each run record is intended to capture:

- experiment family identity
- selected datasets and feature protocols
- output artifact path and checksum
- matching training-history artifact path
- CLI arguments
- git commit and dirty state

Run records are separate from:

- `experiment/artifacts/published/<family_id>/latest.csv`: family-level published result files
- `experiment/artifacts/published/<family_id>/latest.training_history.csv`: epoch-level train history for the current result files
- `experiment/families/*/.work/`: resume and chunk artifacts
- `experiment/artifacts/scratch/`: ad hoc smoke/debug outputs
