# Persistence Turbine Baseline

This directory records the archived persistence turbine baseline family.

Current state:

- implementation logic exists in [`src/wind_datasets/persistence.py`](../../src/wind_datasets/persistence.py)
- the family is registered under `persistence_turbine_baseline` in
  [`experiment/infra/registry/families/persistence_turbine_baseline.toml`](../../infra/registry/families/persistence_turbine_baseline.toml)
- no repo-tracked experiment runner or notebook entry remains

The family stays in the registry as an archived analytic baseline so published
artifacts and coverage metadata can still refer to it without keeping the old
notebook around.
