# Repository guidance for coding agents

## Purpose

This repository is migrating from model-specific experiment scripts to one
benchmark pipeline with thin model adapters. Preserve model mathematics while
moving splitting, generic preprocessing, decoding, validation, and evaluation
into shared benchmark code.

## Read first

- `README.md` mixes current and proposed structure. Confirm that every path and
  command exists before relying on it.
- `docs/benchmark-contract.md` is the draft source of truth for the benchmark
  boundary.
- `docs/benchmark-metrics.md` fixes the mathematical conventions for final
  quality and TSTR numbers.
- `docs/agent-model-migration.md` defines the evidence, implementation, testing,
  and review workflow for one model migration.
- `docs/coding-standards.md` defines mandatory code, documentation, test, and
  local-only Git rules for migration work.

Do not implement an adapter whose semantic `InputSpec` is still marked
unresolved in the contract.

## Current repository map

- `sbtab/data/`: current schema, splits, and fold-local preparation. Treat this
  pipeline as migration evidence, not as the implementation base for the new
  benchmark core.
- `sbtab/transforms/`: current transform implementations. They may inform the
  new codec, but the new runner and codec must not depend on their legacy
  orchestration APIs.
- `sbtab/models/`: trainable components; these are not benchmark adapters.
- `sbtab/solvers/`: SB algorithms with several native APIs.
- `sbtab/baselines/`: baseline generators and current wrappers.
- `sbtab/experiments/`: legacy tuning and evaluation entrypoints. Treat these
  as behavioral evidence, not automatically correct specifications.
- `sbtab/evaluation/`: model-independent raw-space tuning, final quality,
  TSTR, cross-fold aggregation, and linked evaluation artifacts.
- `sbtab/benchmark/`: greenfield benchmark core. Contracts, dataset
  declarations, missing policy, splitting, fold-local codec, the MSBM adapter,
  and fixed-configuration holdout/cross-validation runners are implemented.
  The common tuning objective and model-owned MSBM Optuna study are implemented.
  Create-only cross-validation generation and MSBM tuning artifacts are
  implemented. The Online Shoppers pilot entrypoint connects tuning, final
  generation, evaluation, and linked artifacts. Other tuners and further
  adapters are migrating incrementally.

## Target architecture

```text
TabularDataset -> COMPLETE_CASE -> split
                                  |-- train raw -> fold-local model codec
                                  |                -> PreparedTable -> adapter
                                  |                -> PreparedTable sample
                                  |                -> decode -> synthetic raw
                                  `-- test raw ---------------------> evaluation
```

Dependency direction:

```text
benchmark core -> benchmark adapters -> existing models/solvers
benchmark core -> evaluation
```

Models and solvers must not import benchmark or evaluation code.

The new contracts, dataset declarations, missing policy, splitting, codec, and
runner are implemented together under `sbtab/benchmark/`. New code must not
extend the old `sbtab.data` splitter/DataModule or import experiment-level
target mappings. Existing data, transform, and experiment code is read-only
evidence unless a separate, explicitly scoped migration task says otherwise.

## Contract invariants

- `InputSpec` contains only `continuous_view`, `discrete_view`, and
  `categorical_view`.
- `TabularDataset` is the only public raw-dataset object. It contains the raw
  frame and ordered `ColumnSpec` entries for every modeled column, including
  target.
- The existing `sbtab.data.TabularSchema` is legacy-only. New runner, codec,
  adapter, and evaluator code must not depend on it; migration access goes
  through one removable bridge.
- Every adapter accepts and returns one canonical `PreparedTable`: a DataFrame
  plus semantic prepared schema.
- Native layout, array/tensor dtype, device, data loaders, and temporary `X`/`y`
  extraction belong inside the adapter. They are not shared contract fields.
- Target remains a normal modeled `ColumnSpec`. Its kind is not stored or
  inferred separately. If a native API requires `y`, the adapter reassembles it
  into the complete sampled table.
- Identifier columns never enter `PreparedTable`. Generate new IDs only after
  decoding when output requires them; never resample training IDs.
- `MissingPolicy` is selected once in `BenchmarkConfig`. `ERROR` is the safe
  default; the official v1 comparison explicitly uses `COMPLETE_CASE` once
  before split across all modeled columns. The identifier is ignored.
- Record rows before/after filtering, dropped count/fraction, per-column
  missing counts, and applicable class-distribution changes. Do not add a
  per-model missing capability or fallback to `InputSpec` or an adapter.
- Any learned transform is fitted on the training partition only. Train, test,
  and generated data must never cause model-codec state to be refitted.
- The model codec transforms train and inversely decodes model samples. It does
  not transform held-out test data. Evaluation receives raw test and raw
  synthetic tables; utility evaluators own any downstream predictive pipeline.
- The codec, not an adapter, owns generic scalers, imputers, encoders, inverse
  transforms, category maps, and train-observed finite-state supports. Physical
  pandas storage dtype is not restored by rounding generated values; decoded
  output follows the semantic dtype rules in `docs/benchmark-contract.md`.
- The codec validates its prepared train output. Adapters consume that trusted
  boundary without repeating generic schema, support, missing, or row checks.
- Finite-state metadata is keyed by column name. Use real cardinality and order
  meaning for each column; never replace them with a shared maximum.
- Invalid generated states fail at the shared runner/codec decoding boundary.
  Do not clip, round, pad, or replace them to make a run succeed.
- A fresh codec and adapter instance are created for every fold.
- MSBM is the approved pilot adapter with `STANDARD` continuous values and
  `FINITE_STATE_CODES` for discrete and categorical values.

## Adapter scope

An adapter may select/reorder columns, split/concatenate native blocks, convert
to backend tensors and dtypes, construct native loaders/reference processes,
call existing `fit`/`sample`, and reassemble native output in canonical order.
It does not duplicate validation already owned by the codec, runner, or native
model.

An adapter must not split the dataset, fit generic preprocessing, decode raw
values, calculate metrics, infer dataset semantics by name, or change model
mathematics.

If a correct adapter requires a model-internal fix, propose it as a separate PR
with separate characterization evidence.

## Working rules

1. Explore the complete model, config, tuning path, and evaluation path before
   editing.
2. Record semantic requirements separately from native API details.
3. Treat legacy scripts as evidence. Record contradictions and known defects;
   do not preserve them silently.
4. Keep shared contract changes separate from model-adapter changes.
5. Implement one model per branch and worktree.
6. Do not add model-name branches to runner, codec, or evaluation code.
7. Do not edit old splitters, DataModules, transform pipelines, or experiment
   scripts to make the new benchmark work. Implement the required shared
   behavior in `sbtab/benchmark/`; migrate or remove legacy entrypoints only in
   separately reviewed tasks.
8. Let the codec or native model raise a precise error for unsupported
   semantics instead of duplicating compatibility logic in every adapter.
9. Keep model-specific training and sampling controls in typed adapter config,
   not unrestricted `**kwargs`.
10. Preserve unrelated user changes and keep generated results out of source
   changes unless the task explicitly requires fixtures.
11. Report only commands and checks that actually ran.

## Code, documentation, and Git policy

Follow `docs/coding-standards.md` for every new benchmark module and adapter.
Public contracts and configuration fields require semantic documentation;
adapter documentation must show both canonical/native mappings and state which
model mathematics remains unchanged. Comments explain invariants and reasons,
not obvious syntax.

All agent work is local. Agents must not push, create or modify pull requests,
comment on repository hosting, merge, tag, or otherwise mutate a remote. Make
atomic local commits at coherent checkpoints, stage only explicit task files,
and report each commit hash. Do not rewrite local history unless the user asks.
Agents may prepare draft handoff text for a human.

## Multi-agent work

- The coordinating agent owns the shared contract, task decomposition, and
  final integration.
- Use read-only agents for model archaeology and independent method review.
- Give an implementation agent one adapter, one isolated worktree, and an
  explicit file allowlist.
- Only the coordinating agent edits shared benchmark contracts during a
  migration wave.
- Review agents report findings with file references and severity; they do not
  silently fix the work they review.
- Do not begin a migration wave until one pilot and a second model with
  different semantic views work without model-name branches in shared code.

## Verification

The repository currently has no committed standard test or packaging command.
Do not claim that a nonexistent suite passed.

For documentation-only changes, check links, referenced paths, stale contract
terms, whitespace, and the final diff. For implementation, add the narrow tests
required by `docs/agent-model-migration.md` and record exact commands.

A model-adapter task is complete only when:

- its three semantic views are justified from repository evidence and approved;
- supported and unsupported dataset semantics are tested;
- fold-local encode/decode and output-schema checks pass;
- native construction and model invariants are shown unchanged;
- known corrections to legacy behavior are explicit;
- independent method and contract/test reviews have no unresolved blockers;
- the PR explains what changed, what did not change, and how to reproduce the
  verification.
