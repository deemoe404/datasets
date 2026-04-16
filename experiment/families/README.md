# Experiment Families

This directory contains concrete experiment-family implementations.

Each family owns:

- its runnable CLI entrypoints
- its isolated `./.conda/` environment
- its family-local `./.work/` resume and chunk artifacts
- its family README

Current families:

- `agcrn`
- `agcrn_masked`
- `world_model_agcrn_v1`
- `world_model_baselines_v1`
- `world_model_rollout_v1`
- `world_model_state_space_v1`
