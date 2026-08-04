# Unified benchmark data contract

Status: initial reviewable foundation. This document covers only the public
data boundary implemented in `sbtab/benchmark/contracts.py` and its validation.
Splitting, preprocessing, model adapters, runners, and evaluation are separate
changes and are intentionally not implemented by this contract PR.

## Purpose

The repository currently has model-specific experiment paths with different
assumptions about column types, preprocessing, targets, and native input
layouts. The unified benchmark needs one semantic data boundary before those
paths can be migrated safely.

The boundary separates two questions:

1. What does each raw column mean?
2. In which semantic representation does a model consume that column group?

It deliberately does not standardize a model's tensor layout, backend dtype,
device, loader, or temporary `X`/`y` call signature. Those are adapter-owned
integration details, not dataset semantics.

## Contract objects

### `ColumnSpec`

`ColumnSpec` declares one modeled raw column:

- `name` is its exact DataFrame label;
- `kind` is `CONTINUOUS`, `DISCRETE`, or `CATEGORICAL`;
- `ordered_values` optionally declares a complete ordinal domain.

Column kind is explicit. Shared code and adapters must not infer it from pandas
dtype, observed cardinality, or a dataset name.

Continuous columns have real-valued support. Discrete columns have finite
numeric support with meaningful order. Categorical columns are nominal unless
`ordered_values` gives an explicit semantic order.

### `TabularDataset`

`TabularDataset` is the only public raw-dataset object in the new benchmark. It
contains:

- the raw DataFrame;
- ordered `ColumnSpec` entries for every modeled column;
- an optional target and its task;
- an optional identifier.

The `columns` tuple defines canonical generated-table order. The raw frame must
contain exactly those modeled columns plus, optionally, the declared
identifier. Undeclared columns are rejected instead of silently dropped.

The target remains an ordinary modeled column. It is not removed from the
table or encoded as a separate input mode. If a native model API requires a
separate `y`, its adapter may extract and later reassemble it.

An identifier is never modeled. A future runner may retain it for audit or
reconstruct output identifiers after generation, but it must not pass training
identifiers to a generative model.

### `InputSpec`

`InputSpec` is the complete shared request made by one model family. It has
exactly three fields:

| Semantic group | Supported views |
| --- | --- |
| continuous | `RAW`, `STANDARD`, `UNSUPPORTED` |
| discrete | `RAW_VALUES`, `FINITE_STATE_CODES`, `UNSUPPORTED` |
| categorical | `RAW_VALUES`, `FINITE_STATE_CODES`, `UNSUPPORTED` |

These fields describe semantic values prepared by a future fold-local codec.
They do not describe DataFrames versus arrays, tensor dtypes, devices, target
extraction, missing-value support, or model hyperparameters.

`UNSUPPORTED` means that a model rejects datasets with a non-empty group of
that kind. It is an explicit limitation, not a request for an automatic
fallback.

### `PreparedSchema`

`PreparedSchema` describes a canonical prepared table:

- `column_order` includes every modeled column and the target;
- the three semantic column tuples partition `column_order` exactly;
- `target_col` and `task_type` preserve evaluation semantics;
- `state_columns` records cardinality and ordering for named finite-state
  columns.

State metadata is keyed by column name. Cardinality and ordering must never be
inferred from a model-specific array position or replaced with a shared maximum
cardinality.

The `state_columns` mapping is copied and made read-only during construction.
This prevents later mutation of a caller-owned dictionary from changing an
already constructed schema.

### `PreparedTable`

`PreparedTable` is the future adapter input and output. It consists of a
DataFrame and its `PreparedSchema`.

The frame:

- follows `schema.column_order` exactly;
- contains the target when declared;
- excludes the raw identifier;
- contains no missing or non-finite numeric values;
- uses integer codes in `[0, cardinality)` for encoded state columns.

Generated invalid states are rejected. Shared code must not clip, round, pad,
replace, or silently drop them to make a model run succeed.

## Validation ownership

`sbtab/benchmark/validation.py` validates the boundary without changing data:

- dataset validation checks raw declarations and observed value types;
- input validation accepts only the defined semantic enums;
- prepared-schema validation checks order, partitioning, target semantics, and
  complete finite-state metadata;
- prepared-table validation checks physical DataFrame values against the
  attached schema.

Missing-value filtering and learned transforms are not validation operations.
They will be owned by later benchmark components and must be applied uniformly
across models.

## Dependency boundary

The new `sbtab.benchmark` package must not import the legacy orchestration APIs
under:

- `sbtab.data`;
- `sbtab.transforms`;
- `sbtab.experiments`.

Those modules remain useful migration evidence, but extending them would bind
the new contract to the assumptions it is intended to replace. An AST-based
test checks both absolute and relative imports without importing their targets.

## Concrete mixed-table declaration

`sbtab.benchmark.datasets.online_shoppers` supplies one explicit declaration
for UCI Online Shoppers (dataset 468). It contains no download logic and does
not infer semantics from pandas dtypes.

The three page-count columns are numeric discrete states. Duration and rate
columns are continuous. Remaining finite columns, including `Revenue`, are
categorical. `Month` stays nominal because a linear ordered-state model would
not represent its cyclic neighbourhood correctly.

This declaration is included to exercise the contract against a real mixed
schema. It does not introduce an MSBM adapter or benchmark runner.

## Verification

From the repository root, run the tests owned by this contract:

```bash
python -m unittest \
  tests.benchmark.test_contracts \
  tests.benchmark.test_import_boundaries \
  tests.benchmark.test_online_shoppers
```

The tests cover malformed declarations, target and identifier rules, semantic
partitions, finite-state ranges, timestamp category identity, canonical order,
and the legacy import boundary.
