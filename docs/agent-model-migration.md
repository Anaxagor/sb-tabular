# Agent workflow for migrating one model

Status: draft. Use this workflow only after the model's three-view `InputSpec`
has been reviewed against `docs/benchmark-contract.md`.

All implementation and review work follows `docs/coding-standards.md`.

## Objective

Add one thin adapter that translates between the canonical `PreparedTable` and
one existing model's native API. Preserve the approved benchmark protocol and
the model's mathematics.

One migration task owns:

- one explicitly named model or solver family;
- one adapter module;
- one typed adapter config when model-specific controls are needed;
- contract, codec-boundary, smoke, and characterization tests for that model;
- review evidence for the model owner.

It does not own shared contract redesign, unrelated cleanup, metric redesign,
removal of legacy entrypoints, or model-internal fixes.

The adapter may call the existing native model or solver, but neither it nor
the new runner may depend on old DataModules, splitters, transform pipelines,
experiment scripts, or dataset-name metadata. Those files are evidence during
archaeology, not building blocks of the new benchmark.

## Required task input

The coordinating agent must provide:

- model and adapter name;
- native implementation and configuration paths;
- relevant tuning and evaluation paths;
- an approved three-field `InputSpec`;
- model owner or intended reviewer;
- explicit file allowlist;
- a native characterization command or fixture;
- optional legacy evidence and known defects when a reproducible old path
  exists; absence of such a path does not change the new architecture;
- expected supported and unsupported dataset kinds;
- completion criteria and commands available in the environment.

If any item is unknown, assign a read-only investigation first. Do not ask an
implementation agent to invent the contract while writing the adapter.

## Phase 1: model archaeology

Work read-only and cite concrete files and lines.

1. Read the complete native model/solver and its configuration.
2. Read its tuning path. Record hyperparameters, preprocessing, reference
   construction, training order, and sampling mode.
3. Read evaluation scripts and examples. Record what actually reaches native
   `fit` and `sample`.
4. Compare native code, tuning code, and evaluation code. Legacy is evidence,
   not automatically the oracle. Record disagreements rather than choosing the
   most convenient path.
5. Separate semantic requirements from integration details:
   - semantic: raw/standard continuous values, raw/code discrete values,
     raw/code categorical values, real state cardinalities, and order meaning;
   - integration-only: DataFrame versus array, block splitting, dtype, device,
     data loader, native `X`/`y` signature, and argument names.
6. Record model invariants that the adapter must not change: loss, prior,
   reference process, cardinalities, schedules, direction sequences, training
   steps, and sampling algorithm.
7. Compare the semantic evidence with the approved `InputSpec`. Stop if they
   disagree. Do not add layout, dtype, target mode, or missing capability to
   the shared spec.

Required archaeology report:

```text
Native implementation:
Native fit/sample API:
Approved InputSpec:
Supported dataset semantics:
Unsupported dataset semantics:
State metadata required by the model:
Adapter-local conversions:
Algorithmic invariants:
Tuning/evaluation discrepancies:
Characterization path:
Open decisions:
```

## Phase 2: adapter design

Planned location:

```text
sbtab/benchmark/adapters/<model_name>.py
```

Before editing, write a short mapping from canonical data to native arguments.
For example:

```text
PreparedTable.continuous columns -> float32 tensor x_cont
PreparedTable discrete+categorical state columns -> int64 tensor x_state
PreparedSchema.state_columns -> native cardinalities and ordered_mask
native numeric/state sample -> one DataFrame in PreparedSchema.column_order
```

This mapping belongs in the adapter's docstring or model migration note. It
must not become fields in the shared `InputSpec`.

Target is never removed from the benchmark table. If a native API requires
`fit(X, y)`, the mapping may mechanically select `y` by
`PreparedSchema.target_col`. `sample()` must reassemble target with the sampled
features and return the complete prepared schema.

## Phase 3: adapter implementation

Implementation rules:

1. Accept only the canonical `PreparedTable`.
2. Assume the codec/runner has validated the canonical `PreparedTable`; do not
   duplicate generic schema, missing-value, support, or row checks.
3. Select columns exclusively through `PreparedSchema`; do not guess from
   pandas dtype or dataset name.
4. Perform native layout and dtype conversion locally in the adapter.
5. For state models, construct native cardinality/order arrays by iterating the
   named state columns in the exact native column order.
6. Construct the native model exactly as the approved configuration requires.
7. Call the existing native training and sampling implementation.
8. Reassemble native output into a DataFrame with exactly
   `PreparedSchema.column_order`.
9. Return a `PreparedTable` carrying the same prepared schema received by
   `fit`.
10. Keep sampling variants in a typed adapter config, not common `**kwargs`.

The shared runner validates requested row count and the returned
`PreparedTable` before codec decoding. The codec rejects invalid generated state
codes. Native mathematical restrictions belong to the native model, not to an
adapter compatibility layer.

Allowed adapter-local work includes:

- DataFrame selection and column reordering;
- temporary native `X`/`y` extraction;
- NumPy/Torch conversion and device placement;
- block concatenation or splitting;
- native data-loader and reference-process construction;
- packaging the native sample back into the canonical frame.

Do not:

- copy a solver algorithm into the adapter;
- split train/test data;
- fit a scaler, imputer, quantile transform, or category encoder;
- inverse-transform a sample to raw values;
- inspect fitted codec internals;
- infer target kind, task type, ordinal order, or supports from dataset names;
- accept several native layouts "for convenience";
- cast unsupported categorical codes to floats and claim semantic support;
- replace per-column cardinalities with `max(cardinality)`;
- clip, round, pad, or replace invalid state output;
- add `if model_name == ...` outside the adapter;
- catch broad exceptions and continue with guessed defaults;
- change model mathematics in the adapter PR.

If correct wrapping requires a model-internal fix, stop and propose a separate
model-fix PR with its own characterization evidence.

## Phase 4: required tests

Tests should be narrow, fast, and explicit about what they prove.

### Shared contract and codec tests

These belong with shared benchmark code and should normally be written by the
coordinating agent, not duplicated by every adapter:

- only the three approved views exist in MVP `InputSpec`;
- a non-empty semantic group declared `UNSUPPORTED` fails before adapter
  construction;
- every modeled column, including target, follows its declared
  `ColumnSpec.kind` and remains in canonical order;
- identifiers never enter `PreparedTable`;
- learned preprocessing and state codebooks are fitted only on train;
- held-out test never enters the model codec or adapter and remains raw for
  evaluation;
- a state seen only in raw test does not change train cardinality and is not
  mapped to a generator `UNKNOWN` state;
- encode/decode restores raw column order and semantic output dtypes: prepared
  states are integer-coded, decoded states recover exact raw values, and
  decoded standardized continuous values remain real-valued rather than being
  rounded to an original pandas storage dtype;
- finite-state codes round-trip through the train-fold mapping;
- invalid finite-state codes fail with column/value evidence;
- a fresh codec and adapter are created per fold;
- omitted missing policy defaults to `ERROR` and fails before split with
  per-column evidence;
- explicit v1 `COMPLETE_CASE` filtering happens once before split across all
  modeled columns, ignores identifier, and produces common rows for all models;
- the run records rows before/after, dropped count/fraction, per-column missing
  counts, and applicable class-distribution changes;

### Adapter boundary tests

Every adapter must test:

- its declared `InputSpec` exactly matches the approved values;
- the adapter selects native columns in deterministic order;
- native arrays/tensors receive the required shape and dtype;
- temporary `X`/`y` extraction, when required, is reassembled into the full
  sampled table;
- native sample output is returned in `PreparedSchema.column_order`;
- the returned prepared schema is unchanged.

Malformed prepared input, requested/returned row-count mismatches, and invalid
generated states are shared codec/runner tests. Do not duplicate those cases in
every adapter suite.

### Model smoke tests

- a tiny supported configuration calls `fit` and `sample` on CPU when the
  dependency supports CPU;
- requested and returned row counts match;
- native block dimensions match selected columns;
- real per-column cardinalities and order flags reach native construction
  unchanged;
- sampling configuration reaches the native call unchanged;
- seed behavior matches what the native implementation actually promises.

Use fakes only to verify boundary translation. At least one smoke or
characterization path must exercise the real native implementation when the
dependency and runtime make that practical. Never claim an unavailable heavy
dependency passed.

### Characterization evidence

Characterize the adapter against the native model or solver using the same
prepared dataset slice, approved configuration, and seed. When a reproducible
legacy path exists, it may provide additional evidence using the same split,
but the adapter must not import or call that path. Stochastic generated values
need not be identical unless the native model promises determinism.

`test` in this section means held-out real data used by evaluation, not an
input passed to the generator. Automated unit/contract tests may use tiny
inline DataFrames to exercise boundary cases; the pilot integration and
quality run use a real benchmark dataset.

The comparison must cover:

- semantic representation before the native call;
- native column order, shapes, and construction arguments;
- target presence in the input and output table;
- real cardinalities and order information;
- algorithmic invariants;
- sampled prepared schema and decoded raw schema;
- confirmation that raw held-out test bypasses the model codec and adapter;
- the common complete-case row set and recorded missing-data report;
- known differences introduced by the new common benchmark policy.

Do not use characterization to preserve a known methodological defect. If the
legacy path clips states, pads cardinalities, resamples IDs, leaks preprocessing
across folds, or guesses target metadata, record that as an intentional
benchmark correction for maintainer review.

## Phase 5: independent review

Use two read-only reviews after implementation:

1. **Method review:** inspect mathematical and experimental semantics,
   especially state spaces, reference processes, target handling, training
   schedules, and sampling direction.
2. **Contract/test review:** inspect leakage, silent coercion, model branches in
   shared code, adapter-owned generic preprocessing, weak tests, and missing
   reproduction evidence.

The implementation agent must not be the only reviewer of its own work. Review
agents report findings with file references and severity; they do not silently
edit the implementation they review.

## Human-owned pull-request handoff

Agents do not open, update, comment on, approve, or merge pull requests and do
not push branches. After local implementation and review, an agent may prepare
the following description for the human who owns publication.

Use this structure in the PR description:

```markdown
## Why

## Scope

## Approved semantic InputSpec
| Column kind | View | Repository evidence |
| --- | --- | --- |

## Canonical PreparedTable -> native API mapping
| Canonical source | Native argument | Local conversion |
| --- | --- | --- |

## Native API -> PreparedTable sample mapping

## Target handling

## Algorithmic invariants
| Invariant | Legacy/native | Adapter | Evidence |
| --- | --- | --- | --- |

## Intentional corrections to legacy behavior

## Tests and exact commands run

## Characterization result

## Supported and unsupported datasets

## Out of scope

## Questions for the model owner
```

State whether code was agent-assisted. Reviewability comes from small diffs,
explicit mappings, tests, and evidence rather than from hiding who typed it.

## Stop conditions

Stop and request a coordinating decision when:

- the model needs a semantic representation absent from the approved contract;
- native, tuning, and evaluation paths use materially different algorithms;
- a modeled column lacks `ColumnSpec` semantics, or target task/category order
  metadata is missing;
- the adapter would need generic learned preprocessing;
- correct wrapping requires a model-internal change;
- a test passes only after clipping, padding, dropping, rounding, or replacing
  values;
- shared runner, codec, or evaluator code needs a model-name branch;
- a native dependency or characterization path cannot be reproduced.

Do not work around a stop condition locally. Return the evidence, the smallest
decision required, and the affected files.

## Ready-to-use task prompt

```text
Goal:
Migrate <MODEL> to the unified benchmark through one thin adapter.

Read first:
- AGENTS.md
- docs/benchmark-contract.md
- docs/agent-model-migration.md
- docs/coding-standards.md

Repository evidence:
- Native implementation: <PATHS>
- Native config: <PATHS>
- Tuning/evaluation paths: <PATHS>
- Characterization command or fixture: <COMMAND OR PATH>

Approved contract:
- continuous_view: <RAW | STANDARD | QUANTILE_NORMAL | UNSUPPORTED>
- discrete_view: <RAW_VALUES | FINITE_STATE_CODES | UNSUPPORTED>
- categorical_view: <RAW_VALUES | FINITE_STATE_CODES | UNSUPPORTED>
- benchmark missing_policy: COMPLETE_CASE
- Supported dataset semantics: <LIST>
- Known unsupported semantics: <LIST>

Ownership:
- Model owner/reviewer: <NAME>
- You may edit only: <ALLOWLIST>
- Work locally only. Commit the completed task atomically and report its hash.
  Never push or create/update/comment on/merge a pull request.

Constraints:
- Accept and return only PreparedTable.
- Keep native layout, dtype, device, data loaders, and any temporary X/y split
  inside the adapter.
- Do not filter, impute, or otherwise handle missing values inside the adapter;
  the runner has already applied the common COMPLETE_CASE policy before split.
- Do not fit generic preprocessing or inverse-transform samples.
- Do not change model/solver mathematics.
- Use real named state metadata; do not pad, clip, or guess.
- Do not edit shared contracts or add model-name branches.
- Stop with evidence if the approved contract is insufficient.

Done when:
- The adapter implements the approved three-view InputSpec.
- Boundary, smoke, and characterization checks pass.
- Exact commands and results are reported.
- The returned PreparedTable has the complete canonical schema, including
  target when present.
- Independent method and contract/test reviews have no unresolved blocking
  findings.
- The final summary tells the model owner what changed, what did not change,
  and what requires close review.
```
