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

The codec has already produced validated, dense train-observed state codes.
The adapter does not revalidate that shared contract. After conversion, native
`MixedSBMSolver.fit` rejects non-finite continuous tensors.

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
schema object received by `fit`. The shared runner checks requested row count,
and codec decoding rejects missing/non-finite values or out-of-range state
codes. The adapter does not duplicate those checks or repair native output.

Official benchmark sampling is always positive. The native Gaussian reference
continues to reject `n=0`; the adapter does not add a separate empty-sample
path.

## Target handling

MSBM has no native `X`/`y` split. A target such as Online Shoppers `Revenue`
remains an ordinary state column throughout fit and sampling:

```text
categorical target -> shared state code -> train_cat -> gen_cat
                   -> shared inverse code -> raw generated target
```

The adapter never separates or conditions on `y`.

## Supported and rejected prepared schemas

The current native implementation requires both blocks to be non-empty and now
owns those errors directly:

- `CategoricalReference` rejects an empty state block;
- `MixedSBMSolver` rejects an empty continuous block.

The adapter therefore remains a mechanical mapping for mixed prepared tables;
it does not restate model capabilities as compatibility checks or new
`InputSpec` fields.

A nominal state with train cardinality one is rejected directly by
`CategoricalReference`. With `alpha=0.01`, the uniform reference obtains
`b=10000`; precomputing 100 transition powers overflows at power 78. The old
pipeline silently removed constant columns; the new path does not change the
dataset to make the model run. Ordered singleton support remains allowed.

The updater uses `DataLoader(..., drop_last=True)`. When `N < batch_size`, it
performs no optimizer steps but still marks the solver fitted. This is a known
native-model issue; the adapter neither changes `batch_size` nor adds a second
configuration validator. Official pilot folds exceed the default batch size.

Online Shoppers satisfies the structural requirements: its approved
declaration has seven continuous and eleven state columns. The codec derives
actual fold support before every native fit.

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

`MSBMAdapter` accepts one fixed native `MixedSBMConfig`; omitting it uses the
current native defaults. Model-owned tuning may construct the fixed config
without duplicating its fields in a benchmark contract. On every fold the
adapter copies that config and replaces only device and training seed from
`RunContext`:

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
    alpha=0.01,
    lambda_num=0.8,
    lambda_cat=0.2,
    categorical_loss_normalization=(
        CategoricalLossNormalization.BY_NUM_COLUMNS
    ),
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

The adapter does not validate native hyperparameters. `MixedSBMConfig` and the
solver own their configuration semantics; duplicating those fields in a second
adapter config would create another source of truth.

## Model-owned tuning

`sbtab/benchmark/adapters/msbm_tuning.py` owns the Optuna dependency and MSBM
search space. Shared runner, codec, contracts, and evaluation modules do not
import Optuna. Every trial constructs a real `MixedSBMConfig`, passes it to a
fresh `MSBMAdapter`, executes the common reference holdout, and minimizes the
common raw-space tuning score. The complete native config and per-group,
per-column score evidence are stored as Optuna trial attributes.
`write_msbm_tuning_artifacts` writes those trials, the best native config,
reference holdout controls, missing report, seeds, and timings into a
create-only local handoff directory without exposing the Optuna storage URI.
Artifact version 2 records explicit `alpha` and categorical-loss normalization;
the loader assigns the preserved `0.01`/`BY_NUM_COLUMNS` defaults when reading
an earlier version-1 `best-config.json`.

The human-owned Online Shoppers entrypoint is:

```bash
python -m sbtab.benchmark.pilots.msbm_online_shoppers \
  --output-dir artifacts/msbm-online-shoppers \
  --n-trials 50 \
  --device cpu
```

Omitting `--csv` fetches canonical UCI dataset 468 through `ucimlrepo`; passing
`--csv path/to/raw.csv` performs no network acquisition. The output root must
not already exist. The entrypoint writes tuning and five-fold generation
artifacts, evaluates decoded folds with the shared statistical/TSTR protocol,
and writes a root `pilot-manifest.json` with status `complete` only after all
three child manifests succeed.

The current provisional search space is:

| Native field | Optuna domain |
| --- | --- |
| `fb_sequence` | alternating backward/forward sequence of length 3, 5, 7, or 9 |
| `cat_emb_dim` | integer 8 through 32 |
| `hidden_dim` | 128, 256, or 512 |
| `time_dim` | 32, 64, 96, or 128 |
| `n_layers` | integer 2 through 6 |
| `dropout` | 0.0 through 0.3 |
| `num_steps` | 20 through 100 in increments of 10 |
| `sigma` | log-scaled 0.01 through 1.0 |
| `lambda_num`, `lambda_cat` | independently 0.1 through 1.0 |
| `lr` | log-scaled `1e-4` through `2e-3` |
| `batch_size` | 128, 256, or 512 |
| `epochs_per_direction` | integer 5 through 20 |
| `grad_clip` | 0.1 through 1.0 |

This space still requires model-owner review. It uses only fields consumed by
the current native solver. `alpha` and categorical-loss normalization are
explicit native config fields, but they remain fixed at the current defaults
during Optuna tuning. Their effect is isolated by the factorial ablation below
instead of being confounded with the main hyperparameter search. `eps` exists
in the current config but remains unused by the solver, so it is intentionally
not tuned. Trial `device` and `seed` placeholders are replaced by the common
`RunContext`.

## Categorical-mechanics ablation

`sbtab.benchmark.pilots.msbm_online_shoppers_ablation` runs a predeclared 2x2
full factorial:

| Factor | Values |
| --- | --- |
| categorical reference `alpha` | `0.01`, `0.798` |
| divide categorical loss by state-column count `C` | off, on |

Every other `MixedSBMConfig` field is loaded from one frozen tuning
`best-config.json`. All four cells use the same complete-case rows,
target-stratified five folds, split seed 42, fold training seeds, sample seeds,
sample sizes, and raw evaluation. The ablation does not run Optuna. This makes
the main effects and their interaction reviewable without changing the
benchmark contract or adding model branches to shared code.

The categorical CSBM implementation flattens batch and state-column axes before
its `batchmean` reduction. The optional `BY_NUM_COLUMNS` mode then performs the
additional historical division by `C`; `NONE` leaves the already reduced CSBM
loss unchanged. The choice is part of native model mathematics and therefore
lives in `MixedSBMConfig`, not `InputSpec` or the adapter.

Run from an existing pilot's frozen configuration:

```bash
python -m sbtab.benchmark.pilots.msbm_online_shoppers_ablation \
  --csv path/to/online_shoppers.csv \
  --base-config artifacts/msbm-online-shoppers/tuning/best-config.json \
  --output-dir artifacts/msbm-online-shoppers-ablation \
  --device mps
```

The create-only root `ablation-manifest.json` records the base-config SHA-256,
the factorial design, common protocol, four variant manifests, and each
cross-fold metric summary.

## Sampling time-scale correction

MSBM training samples an integer bridge state `n` and conditions the mixed MLP
on normalized time `t = n / K`. The previous `MixedPathSampler` instead
conditioned that same model on `TimeGrid.times()`, the cumulative geometric
integration step sizes. These are different quantities: cumulative `gamma`
does not generally begin at zero, end at one, or match the training values.

Sampling now uses the bridge-state convention at the current state:

- forward calls use `0/K, 1/K, ..., (K-1)/K`;
- backward calls use `K/K, (K-1)/K, ..., 1/K`.

Only the MLP time-conditioning input changed. Euler--Maruyama still receives
the original geometric `gamma[k]`, and categorical transition methods still
receive the same integer `k`. The shared `TimeGrid.times()` API and other
solver families remain unchanged because their training parameterizations
require separate evidence.

`tests/benchmark/test_msbm_sampling_time.py` uses a recording oracle model to
assert every forward and backward time value independently of learned weights.
Artifacts generated before this correction use the legacy sampling scale and
must not be presented as corrected benchmark or ablation results.

## Algorithmic invariants left unchanged

The adapter preserves:

- the standard Gaussian continuous prior;
- one independent uniform prior per state column;
- configured categorical reference `alpha` (`0.01` by default);
- Gaussian/rank transitions for ordered states and uniform transitions for
  unordered states;
- the native geometric time grid;
- noisy Euler--Maruyama integration with `cfg.sigma`;
- `fb_sequence`, coupling order, epochs per direction, and one snapshot after
  every direction;
- selection of the last backward snapshot for sampling;
- numeric MSE, categorical CSBM loss, their weights, and the configured
  categorical column normalization (`BY_NUM_COLUMNS` by default);
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
boundary without performing model training. They cover the approved
`InputSpec`, canonical block order, target preservation, per-column state
metadata, device and dtype conversion, context seed forwarding, schema identity,
and native output reassembly. Shared codec tests cover invalid generated states.
Native tests cover the three model-owned input restrictions.

A CPU smoke in `lightning11` exercised the adapter with the real
`MixedSBMSolver`: construction, one backward training stage, sampling, and
canonical table assembly completed successfully. The generated continuous and
state blocks had the required `torch.float32` and `torch.int64` dtypes.

`tests/benchmark/test_runner_msbm.py` additionally exercises the complete
pre-evaluation path for two folds: shared split, fresh codec and adapter, real
native solver construction and sampling, shared output validation, and raw
decoding. It supplies the adapter with a typed lightweight native config (one
backward stage, two integration steps, one training epoch); it does not replace
the solver or its sampling implementation.

## Questions for the model owner

1. Should the benchmark wrap the current `09d2592`-equivalent solver, or should
   the newer historical solver be restored first? Historical tuned parameters
   and metrics are not reproducible by the current implementation.
2. Is full `sample(seed)` determinism intended? If yes, native categorical and
   training RNG need a model-owned fix.
3. Should singleton nominal states become mathematically supported natively,
   or remain an explicit native rejection?
4. Is `drop_last=True` intentional when a fold has fewer than `batch_size`
   rows? The native solver currently permits a zero-step fit.
5. Is the unused `eps` field intentional in the current config?

## Native smoke profile

Use CPU, one backward stage, small even `time_dim`, `num_steps=2`,
`batch_size=2`, and two mixed train rows. Verify:

- native fit performs at least one optimizer step;
- sample shapes match `(2, 1)` and `(2, 2)`;
- output dtypes are `float32` and `int64`;
- output device is CPU;
- continuous output is finite;
- state codes stay within per-column cardinalities.
