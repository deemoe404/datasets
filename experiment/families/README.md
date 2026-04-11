# Experiment Families

This directory contains concrete experiment-family implementations.

Each family owns:

- its runnable CLI entrypoints
- its isolated `./.conda/` environment
- its family-local `./.work/` resume and chunk artifacts
- its family README

Current families:

- `chronos-2`
- `chronos-2-exogenous`
- `ltsf-linear`
- `tft`
- `agcrn`
- `persistence` (registered placeholder, not yet a first-class runner)
