# PR handoff: Gaussian routing for numeric discrete TabDDPM columns

## Why

The earlier benchmark adapter routed both discrete and categorical columns to
multinomial diffusion. The approved experiment instead requires continuous and
numeric discrete columns in Gaussian diffusion, with only categorical columns
in multinomial diffusion. This implementation was agent-assisted.

## Scope and branch

Local branch: `fix/tabddpm-gaussian-discrete`, based on
`feat/tabddpm-full-pilot` at `7a4a511`. Review the focused correction with:

```bash
git diff feat/tabddpm-full-pilot...fix/tabddpm-gaussian-discrete
```

The shared codec change is a separate commit, `ee81674`. The subsequent commit
contains adapter routing, protocol isolation, tests, and documentation.
No native model/solver, metric formulas, search space, or dataset declarations
are changed by this correction.

## Approved semantic InputSpec

| Column kind | View | Evidence |
| --- | --- | --- |
| Continuous | `STANDARD` | Project benchmark preprocessing specification |
| Discrete | `RAW_VALUES` | Approved correction; legacy wrapper numeric block |
| Categorical | `FINITE_STATE_CODES` | Native multinomial input and cardinalities |

## Canonical PreparedTable -> native API mapping

| Canonical source | Native argument | Conversion |
| --- | --- | --- |
| Continuous + discrete in canonical order | `train_num` | float32 |
| Categorical in canonical order | `train_cat` | int64 |
| Categorical state metadata | `cardinalities` | Each column's own cardinality |

```python
from sbtab.baselines.tabddpm.native import TabDDPMConfig
from sbtab.benchmark.adapters import TabDDPMAdapter

adapter = TabDDPMAdapter(TabDDPMConfig(steps=10_000, num_timesteps=100))
# Shared runner owns preparation and creates a fresh adapter for every fold.
# Continuous + discrete -> Gaussian; categorical -> multinomial.
```

## Native API -> PreparedTable sample mapping

Numeric and categorical tensors are reassembled in canonical order. Only
numeric discrete outputs receive `np.rint`: nearest integer, ties to even.
No clipping, range correction, or projection onto train support is applied.
Non-finite values and invalid categorical codes still fail shared validation.
Explicit `ColumnSpec.ordered_values` remains a domain constraint.

## Target handling

Target stays a modeled column. Its declared kind determines its block; numeric
discrete targets are also rounded on output. No conditional-label extraction.

## Algorithmic invariants

| Invariant | Native and corrected adapter | Evidence |
| --- | --- | --- |
| Loss, beta schedule, denoiser, sampler | Native formulas unchanged | No native file changes |
| Optimizer, learning-rate schedule, EMA | Native behavior unchanged | Direct native characterization test |
| Block width and loss assignment | Corrected for discrete columns | Adapter mapping tests |
| Fold isolation | New codec/adapter; train-only transforms | Shared runner and codec tests |

## Intentional corrections to legacy behavior

Legacy discrete JS rounded values only inside that metric. The approved new
output convention rounds once before every metric and TSTR calculation.
This is a protocol change, not a claim of exact legacy-output equivalence.
Numeric RAW_VALUES no longer acquires a train-support restriction in the codec;
categorical RAW_VALUES and finite-state codes retain their support checks.

Tuning protocol and collection manifest are now v3. Old studies, rerank caches,
and collection output roots cannot be resumed into the corrected experiment.
Keep them for comparison and start a new output directory.

## Tests and exact commands run

```bash
conda run -n lightning11 python -m unittest discover -s tests/benchmark -q
git diff --check
```

Result: 193 tests passed. Existing NumPy deprecation warnings remain.
Failure-path tests intentionally emit Optuna error/pruning logs.

## Characterization result

CPU tests compare the adapter with direct native fit/sample using identical
prepared inputs, configuration, and seeds, accounting explicitly for output
quantization. Routing, targets, ties-to-even, novel numeric output, fractional
input rejection, declared-domain validation, and old-protocol resume guards
have regression coverage.

An additional one-step CPU smoke on the first 256 real Online Shoppers rows
produced 204 train rows, 52 validation rows, and 52 generated rows. All three
discrete columns were finite integers and the tuning score was finite.
This smoke is not a quality benchmark. No full retraining or GPU validation
was performed for this correction.

Read-only method and contract/test reviews found no blocking issues. The
contract reviewer requested the declared-domain qualification and regression;
both are included.

## Supported and unsupported datasets

Supports integer-valued numeric discrete columns and any combination of empty
or nonempty Gaussian/categorical blocks. The local 14-dataset bundle contains
33 integer-valued discrete columns. Fractional discrete training values are
explicitly unsupported by this integer-output convention. They are not
silently renumbered, since that changes numerical distances.

## Out of scope

New tuning results, quality claims, conditional generation, a new Optuna
abstraction, unrelated migrations, and native algorithm changes.

## Questions for the model owner

Please review the corrected block mapping and the explicit once-per-sample
quantization convention. The implementation intentionally exposes negative
or otherwise out-of-range integer samples rather than repairing them.

## Human-owned publication

Do not accidentally select upstream `main` expecting a two-commit diff: this
branch includes the full-pilot base. A focused review requires a PR targeting
`feat/tabddpm-full-pilot` in a repository where that base branch exists; targeting
upstream `main` may produce a cumulative PR. No remote action was performed.

Fresh benchmark command, after selecting this branch and the intended device:

```bash
python -m sbtab.benchmark.pilots.tabddpm_mixed_benchmark \
  --output-dir artifacts/tabddpm-mixed-optuna-v3 \
  --device cuda \
  --target-complete-trials 30 \
  --max-total-trials 45 \
  --rerank-candidates 3 \
  --rerank-seed-pairs 2
```

Without `--dataset-pickle`, acquisition uses the configured dataset download
path and may require network access. For an offline run, supply an existing
trusted, pandas-compatible bundle using `--dataset-pickle /path/to/bundle.pkl`.
