# Experiment Overview

This directory contains the repository-tracked experiment runner, machine-
readable registry, and runtime artifact contracts for the forecasting task.

## Active Family

- [agcrn](./families/agcrn/README.md): Kelmarsh-only, farm-synchronous, `power_only` AGCRN pilot

Archived experiment families have been removed from the active tree. The
current experiment surface is intentionally narrow while the dataset/task
architecture stabilizes around `tasks/<task_id>/<feature_protocol_id>/...`.

## Layout

- `families/`: concrete experiment-family code, environments, and family-local `.work/`
- `infra/`: shared experiment infrastructure such as registry and common protocol helpers
- `artifacts/`: published outputs, formal run records, and scratch outputs

Canonical runtime artifacts:

- `./artifacts/published/<family_id>/latest.csv`: current formal family-level result CSV
- `./artifacts/runs/<family_id>/<timestamp>/manifest.json`: immutable run record for one CLI invocation
- `./artifacts/scratch/<family_id>/...`: ad hoc smoke/debug outputs
- `./families/<family>/.work/`: family-local resume, chunk, and checkpoint artifacts

## Notes

- `agcrn_official_aligned` is the only registry-backed family currently intended for use.
- The active task contract is `24h -> 6h`, dense sliding windows, `farm` granularity.
- Dataset-side feature protocol definitions remain in `src/wind_datasets/feature_protocols.py`.
