# Experiment Registry

This directory is the repository's machine-readable registry for experiment
families and canonical feature protocols.

The registry is the first step toward a single source of truth for questions
such as:

- which experiment families are formal benchmark lines vs pilots or prototypes
- which datasets a family officially covers
- which canonical feature protocols a family supports
- where the family implementation and default output paths live

## Layout

```text
experiment/infra/registry/
  families/
  feature_protocols/
  published/
```

- `families/`: one TOML file per canonical experiment family
- `feature_protocols/`: one TOML file per canonical feature protocol
- `published/`: reserved for future benchmark indexes derived from formal run
  manifests

Formal result CSVs do not live under `experiment/infra/registry/`; they live under
`experiment/artifacts/published/`.

## Family Schema

Each family TOML declares:

- identity: `family_id`, `display_name`, `status`
- implementation: `implementation_root`, `readme_path`,
  `runner_entrypoint`, `orchestrator_entrypoint`
- experimental contract: `model_family`, `training_mode`,
  `dataset_scope`, `supported_feature_protocols`,
  `supported_eval_protocols`, `supported_result_splits`
- task contract: `task_contract.*`
- local-to-canonical mapping: `implementation_label_kind`,
  `implementation_labels`, `implementation_bindings`

The `implementation_bindings` table maps current family-local labels
(`input_pack`, `covariate_stage`, `variant`, and so on) onto canonical feature
protocol ids.

## Feature Protocol Schema

Each feature protocol TOML declares:

- identity: `feature_protocol_id`, `display_name`
- semantics: `protocol_kind`, `summary`
- covariate usage flags
- optional aliases for current implementation labels

The canonical feature protocol is intentionally separate from model family,
layout, and training strategy. `stage1/2/3` therefore lives here as a feature
protocol concept, not as the top-level experiment lifecycle.

## Validation

Use the lightweight loader and validator in
[`experiment/infra/common/experiment_registry.py`](../common/experiment_registry.py).

Example:

```bash
./.conda/bin/python experiment/infra/common/experiment_registry.py --format markdown
```

That command validates the registry and prints the current dataset x family x
feature-protocol matrix.
