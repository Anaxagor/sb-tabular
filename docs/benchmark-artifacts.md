# Benchmark artifact format

Status: implemented for cross-validation generation, MSBM tuning, and final
evaluation runs.

`write_cross_validation_artifacts(result, output_dir)` creates a new local
directory. It refuses to overwrite an existing path. The manifest is written
last, so its presence means that every declared CSV was written successfully.

```text
<output_dir>/
├── manifest.json
├── real-post-policy.csv
├── fold-0/
│   └── synthetic.csv
└── fold-N/
    └── synthetic.csv
```

`real-post-policy.csv` stores the raw dataset after the one global missing
policy. It includes a declared identifier when present. Each synthetic table
contains only modeled columns in canonical order.

`manifest.json` version 1 records:

- adapter and dataset identity;
- semantic column declarations, target, task, and identifier;
- split strategy and seed;
- training/sample base seeds and device;
- the complete missing-data report;
- train/test positional indices for every fold;
- real and synthetic row counts;
- native adapter fit/sample timings;
- relative paths to all stored tables;
- SHA-256 digests for the stored real table and every synthetic table.

Real train and test tables are not duplicated per fold. A reviewer reconstructs
them exactly with positional indexing:

```python
train_raw = real.iloc[manifest_fold["train_positions"]]
test_raw = real.iloc[manifest_fold["test_positions"]]
```

This is a review and evaluation handoff format, not a pandas dtype-preserving
archive. Semantic dtypes come from the manifest's column declarations; CSV
storage must not be treated as a replacement dataset contract.

## MSBM tuning study

`write_msbm_tuning_artifacts(result, output_dir)` creates:

```text
<output_dir>/
├── manifest.json
├── best-config.json
└── trials.json
```

The version 1 manifest records dataset semantics, reference holdout controls,
all seeds, study direction and sampler type, best trial and score, and paths to
the complete best native config and trial evidence. `trials.json` retains every
trial's state, value, suggested parameters, component scores, column scores,
and fit/sample timings.

The Optuna storage URI is deliberately not written because it may contain
credentials. The manifest records only whether persistent storage was
configured.

Final metrics use a separate create-only `evaluation/` artifact. Its
`metrics.json` preserves every fold, per-column value, and population summary.
Its `manifest.json` records both a relative path and SHA-256 digest for the
exact generation manifest. That source manifest in turn records every table
digest, so the linkage reaches the exact CSV bytes. Evaluation does not
rewrite generated tables.

## Online Shoppers pilot root

The MSBM Online Shoppers entrypoint creates `pilot-manifest.json` only after
the tuning, five-fold generation, and final-evaluation manifests all exist.
Its status is `complete`; the file links the three child artifact roots and
records the canonical UCI ID, target, best trial, and tuning score. A missing
root manifest means the pilot stopped before completing the benchmark.
