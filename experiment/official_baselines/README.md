# Official Baseline Wrappers

This directory pins external baseline sources and package-level references used
by hardened world-model baseline experiments. Each wrapper owns its source
reference and its own sibling `.conda/` environment so official dependencies do
not pollute dataset processing or family-local training environments.

Wrapper contract:

- `source/`: git submodule for source-based official or official-core baselines.
- `.conda/`: ignored wrapper-local environment.
- `environment.yml`: dependency seed for the wrapper environment.
- `create_env.sh`: idempotent conda create/update helper.
- `README.md`: source URL, pinned commit, license status, and adapter notes.

The experiment family consumes the existing `world_model_v1` task bundle. These
wrappers must not mutate raw data, cache bundle semantics, source keys, or
feature protocols.
