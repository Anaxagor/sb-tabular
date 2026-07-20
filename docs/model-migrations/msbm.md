# MSBM migration note

Status: adapter implemented with direct imports of `torch`, `MixedSBMConfig`,
and `MixedSBMSolver`; canonical/native boundary tests and a native CPU smoke
pass in the `lightning11` environment.

## Scope and implementation under review

This migration wraps the MSBM implementation currently present in the working
tree:

- `sbtab/solvers/msbm/config.py`;
- `sbtab/solvers/msbm/solver.py`;
- `sbtab/solvers/msbm/updater.py`;
- `sbtab/models/neural/MixedMLP.py`;
- the mixed references, loss, path sampler, SDE, and time grid under
  `sbtab/bridge/`.

The current config, solver, and updater match repository commit `09d2592`.
Deleted tuning and evaluation paths from later commits targeted a materially
different solver and are not characterization evidence for this implementation.

## Approved semantic input

```python
InputSpec(
    continuous_view=ContinuousView.STANDARD,
    discrete_view=DiscreteView.FINITE_STATE_CODES,
    categorical_view=CategoricalView.FINITE_STATE_CODES,
)
```

The fold-local shared codec owns standardization and reversible state mappings.
The adapter neither fits preprocessing nor reads held-out rows.

## Canonical table to native API

The current native constructor and lifecycle are:

```python
solver = MixedSBMSolver(
    continuous_dim: int,
    cardinalities: list[int],
    is_ordered: torch.Tensor,
    cfg: MixedSBMConfig,
)
solver.fit(train_num, train_cat)
gen_num, gen_cat = solver.sample(n_samples, seed)
```

The adapter mapping is fixed:

| Canonical source | Native value | Shape | dtype |
| --- | --- | --- | --- |
| `PreparedSchema.continuous_columns` | `train_num` | `(N, D_cont)` | `torch.float32` |
| `column_order` filtered by `state_columns` | `train_cat` | `(N, D_state)` | `torch.int64` |
| same state names | `cardinalities` | `D_state` Python values | positive `int` |
| same state names | `is_ordered` | `(D_state,)` | `torch.bool` |

Every tensor is placed on `RunContext.device` before the native call. The
solver moves its model and references but does not move train tensors itself.

Before native conversion, the adapter rejects continuous values that become
non-finite in `float32`, state codes or cardinalities that cannot be represented
by `torch.int64`, and train metadata whose declared cardinality is not realized
by dense observed codes `0..K-1`. These checks prevent silent cast corruption
and fictitious embedding states; they do not repair the prepared table.

State columns follow canonical table order. They are not regrouped as
"categorical then discrete". Data, cardinalities, and ordered flags always use
the same state-name sequence.

## Native sample to canonical table

Native output is expected to contain:

- `gen_num`: `(n, D_cont)`, `torch.float32`;
- `gen_cat`: `(n, D_state)`, `torch.int64`.

The adapter moves both blocks to CPU, labels them with their selected canonical
column names, and reassembles one DataFrame in
`PreparedSchema.column_order`. The returned `PreparedTable` carries the exact
schema object received by `fit`. The adapter checks native block ranks, widths,
row counts, and dtypes before DataFrame assembly. Shared prepared-table
validation then rejects missing/non-finite continuous values and out-of-range
state codes. Neither layer clips, rounds, pads, or replaces output.

The native Gaussian reference rejects `n=0`. The shared adapter contract allows
a zero-row request, so MSBM returns an empty valid prepared table without
calling the native solver in that case.

## Target handling

MSBM has no native `X`/`y` split. A target such as Online Shoppers `Revenue`
remains an ordinary state column throughout fit and sampling:

```text
categorical target -> shared state code -> train_cat -> gen_cat
                   -> shared inverse code -> raw generated target
```

The adapter never separates or conditions on `y`.

## Supported and rejected prepared schemas

The current native implementation requires both blocks to be non-empty:

- an empty state block fails inside `CategoricalReference`;
- an empty continuous block can produce a `NaN` numeric loss.

The adapter therefore supports mixed prepared tables only and reports a precise
compatibility error for pure-continuous or pure-state schemas. This is
adapter-local capability evidence, not a new field in `InputSpec`.

A nominal state with train cardinality one is also rejected under the current
default reference profile. With `alpha=0.01`, the uniform reference obtains
`b=10000`; precomputing 100 transition powers overflows at power 78. A shorter
time grid may avoid that constructor failure, so this is a current-profile
compatibility rule rather than a universal statement about singleton states.
The old pipeline silently removed constant columns; the new path must not
change the dataset without review. Ordered singleton support is not rejected
by this specific check.

The updater uses `DataLoader(..., drop_last=True)`. When `N < batch_size`, it
performs no optimizer steps but still marks the solver fitted. The adapter
rejects that configuration instead of silently reporting an untrained run, and
it never changes `batch_size` to make the call succeed.

Online Shoppers satisfies the structural requirements: its approved
declaration has seven continuous and eleven state columns. Actual fold support
and row-count checks still run before every native fit.

## State order semantics

MSBM uses real per-column cardinalities. Its internal maximum cardinality is
only a padded logits dimension; adapters must not replace the list with a
shared maximum.

For ordered states, the reference uses squared distance between integer codes.
The codec therefore expresses rank adjacency, not the numerical gaps between
raw values:

- numeric discrete columns are ordered;
- categorical columns are ordered only with explicit `ordered_values`;
- nominal categories are unordered;
- Online Shoppers `Month` is nominal because a calendar cycle is not a line.

The native backbone clamps state inputs before embedding. This is defensive
native behavior, not an adapter repair policy: shared prepared-table validation
must reject invalid generated codes before they can be accepted as output.

## Native configuration used by the pilot

The pilot uses current native defaults, with only device and training seed
supplied by `RunContext`:

```python
MixedSBMConfig(
    fb_sequence=("b", "f", "b", "f", "b"),
    cat_emb_dim=16,
    hidden_dim=512,
    time_dim=128,
    n_layers=5,
    dropout=0.1,
    num_steps=100,
    sigma=0.1,
    lambda_num=0.8,
    lambda_cat=0.2,
    eps=1e-3,
    lr=1e-4,
    batch_size=256,
    epochs_per_direction=5,
    grad_clip=1.0,
    device=context.device,
    seed=context.seed,
)
```

`eps` is declared but not consumed by the current solver. `AdamW` receives only
`lr`, so the installed Torch backend's default weight decay remains active.
The repository does not pin a Torch version, so the migration does not claim a
numeric default it cannot verify. The adapter must not reinterpret either
behavior.

Before native construction, the adapter validates the instantiated config
instead of relying on downstream tensor errors:

- `time_dim` is positive and even;
- `num_steps >= 2` and `batch_size > 0`;
- embedding, hidden, layer, and continuous dimensions are positive where the
  mixed solver requires them;
- `epochs_per_direction` is non-negative;
- `fb_sequence` is non-empty, contains only `"f"`/`"b"`, and includes at least
  one backward stage required by `sample`;
- when epochs are positive, `N >= batch_size` so `drop_last=True` cannot create
  a zero-step fit.

## Algorithmic invariants left unchanged

The adapter preserves:

- the standard Gaussian continuous prior;
- one independent uniform prior per state column;
- categorical reference `alpha=0.01`;
- Gaussian/rank transitions for ordered states and uniform transitions for
  unordered states;
- the native geometric time grid;
- noisy Euler--Maruyama integration with `cfg.sigma`;
- `fb_sequence`, coupling order, epochs per direction, and one snapshot after
  every direction;
- selection of the last backward snapshot for sampling;
- numeric MSE, categorical CSBM loss, their weights, and native categorical
  normalization;
- AdamW, native batch shuffling, gradient clipping, and backward sampling.

No solver, loss, reference, schedule, or sampling algorithm is copied into or
modified by the adapter.

## Seed behavior

The adapter passes `RunContext.seed` and the requested sample seed unchanged.
The current solver does not make full mixed sampling deterministic:

- Gaussian prior and continuous Euler noise consume explicit seeds;
- categorical `torch.randint`, categorical `torch.multinomial`, training
  shuffle, training timesteps, and training noise also depend on global Torch
  RNG state.

Tests must therefore assert shapes, schema, dtype, support, and finiteness, not
repeat-call equality. Changing native RNG behavior is a separate model fix, not
adapter integration.

## Intentional corrections to legacy behavior

The new benchmark deliberately does not preserve the following historical
choices:

- dataset-name-based semantic inference is replaced with `ColumnSpec`;
- `Month` is not incorrectly represented as linearly ordered;
- constant columns are not silently removed;
- invalid generated states are not replaced with sentinel/unknown values;
- missing rows are selected once before splitting, not differently inside
  model folds;
- target is generated jointly without legacy target-list reconstruction.

## Characterization status

Static archaeology completed:

- native config, solver, updater, backbone, references, loss, sampler, SDE, and
  time grid were inspected;
- current experiment scripts contain no `MixedSBMSolver` call;
- historical tuning/evaluation code was inspected through Git history;
- current config/solver/updater were compared with `09d2592`;
- consumed config fields and the unused `eps` field were traced.

Adapter boundary tests use real Torch tensors and the real `MixedSBMConfig`.
The real `MixedSBMSolver` signature is autospecced so unit tests can inspect the
boundary without performing model training. They cover canonical block order,
target preservation, per-column state metadata, device and dtype conversion,
context seed forwarding, schema identity, output shape/row/dtype/support
validation, zero-row bypass, lifecycle failures, malformed prepared input, and
float/state cast overflow.

A CPU smoke in `lightning11` exercised the adapter with the real
`MixedSBMSolver`: construction, one backward training stage, sampling, and
canonical table assembly completed successfully. The generated continuous and
state blocks had the required `torch.float32` and `torch.int64` dtypes.

## Questions for the model owner

1. Should the benchmark wrap the current `09d2592`-equivalent solver, or should
   the newer historical solver be restored first? Historical tuned parameters
   and metrics are not reproducible by the current implementation.
2. Is full `sample(seed)` determinism intended? If yes, native categorical and
   training RNG need a model-owned fix.
3. Should singleton nominal states become mathematically supported natively,
   or remain an explicit compatibility rejection?
4. Is `drop_last=True` intentional when a fold has fewer than `batch_size`
   rows? The adapter will reject zero-step training rather than change it.
5. Is the unused `eps` field intentional in the current config?

## Required native smoke after Torch is available

Use CPU, one backward stage, small even `time_dim`, `num_steps=2`,
`batch_size=2`, and two mixed train rows. Verify:

- native fit performs at least one optimizer step;
- sample shapes match `(2, 1)` and `(2, 2)`;
- output dtypes are `float32` and `int64`;
- output device is CPU;
- continuous output is finite;
- state codes stay within per-column cardinalities.
