# Coding and documentation standard

Status: normative for new benchmark code and model-adapter migrations.

## Scope

These rules apply to new or materially changed code in `sbtab/benchmark/`, new
model adapters, their tests, and their documentation. Do not reformat native
models or unrelated legacy files merely to make them match this document.

The repository currently has no committed formatter, linter, type checker,
packaging configuration, or standard test command. `README.md` mentions some
of these as recommendations, not as installed tooling. Agents must not claim
that Ruff, Black, mypy, pre-commit, or a repository-wide test suite passed
until the corresponding configuration and command actually exist.

## Design before style

- Prefer a small explicit implementation over a speculative abstraction.
- Keep dependency direction visible: benchmark core -> adapter -> native
  model/solver.
- Shared code must describe dataset or experiment semantics, never a model
  name. Model-specific layout and calls stay in that model's adapter.
- Reject invalid or unsupported input with a precise exception. Do not add a
  silent fallback, guessed default, coercion, clipping, or data dropping.
- Do not preserve a legacy behavior unless the approved contract requires it.
  Record intentional corrections instead of hiding them in compatibility code.
- Avoid duplicate sources of truth. Derive column groups from `ColumnSpec` and
  native arrays from named prepared metadata.

## Python style

- Use four spaces for indentation and UTF-8 files with a final newline.
- Keep new code readable at approximately 88 characters per line. A longer
  mathematical expression is acceptable when splitting it would obscure the
  formula.
- Group imports as standard library, third-party, then local `sbtab` imports.
  Do not use wildcard imports.
- Add `from __future__ import annotations` to new Python modules while the
  repository has no declared minimum Python version.
- Type every public function, method, property, dataclass field, and protocol
  member. Type private helpers when the accepted or returned shape is not
  obvious. Avoid `Any` at contract boundaries.
- Use frozen dataclasses for immutable contracts and run metadata. Do not use a
  dictionary when a fixed set of fields has defined semantics.
- Prefer enums or explicit strategy objects for closed behavioral choices.
  Do not encode them as loosely interpreted strings or booleans.
- Validate at the boundary that owns an invariant and raise a specific
  `ValueError`, `TypeError`, or domain exception with column, value, fold, or
  model context. Do not catch broad exceptions and continue.
- Keep functions focused. Extract a helper when it gives one invariant or
  conversion a name; do not split linear code into one-line wrappers.
- Keep public mutation explicit. A fresh codec and adapter are constructed per
  fold; hidden module-level state and mutable default arguments are forbidden.

## Documentation and comments

Documentation is part of the implementation, not optional polish.

- Every public module, class, protocol, enum, and non-trivial public function
  has a docstring describing purpose, ownership, and important invariants.
- Every public contract or configuration field documents what it means, who
  consumes it, valid values or units when relevant, and what it must not be
  used to infer. Do not merely repeat the field name.
- An adapter module documents both mappings:
  `PreparedTable -> native arguments` and
  `native sample -> PreparedTable`. It also lists supported semantics and the
  model mathematics that the adapter intentionally leaves unchanged.
- A non-obvious mathematical choice includes the reason and a stable source or
  native implementation reference. Comments must distinguish mathematical
  requirements from integration mechanics.
- Inline comments explain why an invariant or unusual operation exists. Do not
  narrate obvious syntax line by line.
- When behavior changes, update the contract or migration note in the same
  local change. Examples and commands must refer to paths and commands that
  actually exist.
- Do not leave placeholder prose, invented benchmark results, or unsupported
  claims. A TODO must name the missing decision and belongs only where work is
  deliberately deferred by the task.

## Tests

- Test observable contracts and failure modes, not private implementation
  details.
- Test names state the behavior and expected result.
- Every bug fix first gains a narrow regression test when the runtime permits.
- Shared contract and codec tests cover malformed prepared data, state support,
  row counts, and train-only fitting. Adapter tests focus on column order,
  target preservation, cardinality/order metadata, and native conversion.
- Fakes may verify adapter translation, but they cannot be the only evidence
  that a native model can be constructed and sampled when its dependency is
  available.
- Stochastic tests use explicit seeds and assert promised invariants rather
  than fragile equality the native model does not guarantee.

## Change and verification hygiene

- Modify only the assigned allowlist and preserve unrelated user changes.
- Do not perform drive-by cleanup, repository-wide formatting, or generated
  result updates in a model-adapter change.
- Inspect the final diff for duplicated logic, stale terminology, accidental
  legacy dependencies, debug output, and secrets.
- Report only checks that actually ran, including the exact command and any
  skipped verification with its reason.

## Git and external actions

Agents preserve a reviewable local history. Create a local commit at every
coherent checkpoint: one shared-core stage, one adapter implementation, one
review-driven correction, or one documentation decision set. A commit must:

- contain one logical change and only files inside the task allowlist;
- be preceded by inspection of `git status` and the staged diff;
- stage explicit paths rather than unrelated workspace changes;
- use an imperative message such as `feat(benchmark): add core contracts`,
  `test(msbm): cover adapter state mapping`, or
  `docs(benchmark): record pilot protocol`;
- report its hash in the handoff.

Do not amend, squash, rebase, or delete local commits unless the user explicitly
requests history rewriting. Agents must not:

- push commits or branches to any remote;
- create, edit, comment on, close, approve, or merge a pull request;
- create or edit issues, releases, tags, or other remote repository state;
- invoke `gh`, a GitHub API, or a connector for those actions.

Local commits are required by the project workflow, but they are never pushed
by an agent. The agent may also draft a pull-request description as handoff
text; a human owns all remote publication and review actions.
