# Experiment Registry

This directory is the repository's machine-readable registry for experiment
families and published-result indexes.

The registry is the first step toward a single source of truth for questions
such as:

- which experiment families are formal benchmark lines vs pilots or prototypes
- which datasets a family officially covers
- which dataset-side `feature_protocol_id` values a family supports
- where the family implementation and default output paths live

## Layout

```text
experiment/infra/registry/
  families/
  published/
```

- `families/`: one TOML file per canonical experiment family
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
(`input_pack`, `variant`, and so on) onto dataset-side `feature_protocol_id`
strings. Feature-protocol semantics and task-bundle column selection are owned
by `src/wind_datasets/feature_protocols.py`, not by the experiment registry.

## Validation

Use the lightweight loader and validator in
[`experiment/infra/common/experiment_registry.py`](../common/experiment_registry.py).

Example:

```bash
./.conda/bin/python experiment/infra/common/experiment_registry.py --format markdown
```

That command validates the family registry and prints the current dataset x
family x feature-protocol matrix.
