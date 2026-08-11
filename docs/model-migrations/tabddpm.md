# TabDDPM migration note

Status: semantic input approved; native tensor boundary and benchmark adapter
are implemented locally. Independent method and contract reviews are pending;
maintainer/model-owner review remains required before merge.

## Scope

This migration wraps the TabDDPM implementation already present under
`sbtab/baselines/tabddpm/`. It does not copy the diffusion algorithm into the
benchmark package and does not import the legacy data or transform pipeline
from an adapter.

The implementation is deliberately split into two boundaries:

- a native TabDDPM solver accepts model-ready numeric and state tensors and
  owns training and sampling;
- a benchmark adapter translates the canonical `PreparedTable` to and from
  that native API.

The existing `TabDDPMWrapper` remains a legacy-facing compatibility shell. Its
schema inference, transform inspection, raw category reconstruction, identifier
handling, and output repair are not part of the new benchmark path.

## Approved semantic input

```python
InputSpec(
    continuous_view=ContinuousView.STANDARD,
    discrete_view=DiscreteView.FINITE_STATE_CODES,
    categorical_view=CategoricalView.FINITE_STATE_CODES,
)
```

The fold-local shared codec owns continuous standardization and every
reversible train-observed codebook. The adapter does not fit preprocessing or
read held-out rows.

This is an intentional correction to the legacy wrapper. That wrapper places
numeric discrete columns in the Gaussian block, which can generate arbitrary
real values, while its mixed-data tuning metric rounds those values before
counting states. The unified benchmark does not repair model output. Declared
finite supports are therefore represented as dense codes and modeled by
TabDDPM's multinomial diffusion.

## Canonical table to native API

The mapping is fixed:

| Canonical source | Native value | Shape | dtype |
| --- | --- | --- | --- |
| `PreparedSchema.continuous_columns` | numeric train block | `(N, D_num)` | `torch.float32` |
| `column_order` filtered by `state_columns` | state train block | `(N, D_state)` | `torch.int64` |
| same state names | per-column cardinalities | `D_state` Python values | positive `int` |

State columns remain in canonical table order. Discrete and categorical names
are not independently regrouped, so data columns and cardinalities always use
the same sequence. TabDDPM does not consume the `ordered` flag: its
multinomial transitions treat the states symmetrically even when the raw
column is numeric discrete or explicitly ordinal.

The codec has already validated dense state codes and train-observed
cardinalities. The adapter converts containers and dtypes but does not repeat
generic schema, support, missing-value, or row validation.

## Native sample to canonical table

The native sampler returns separate numeric and state blocks. The adapter
moves them to CPU, labels them with the fitted block names, and reassembles one
DataFrame in `PreparedSchema.column_order`. The returned `PreparedTable`
carries the exact schema object received by `fit`.

The adapter never clips, rounds, pads, or replaces generated states. Shared
runner validation checks the returned row count and prepared table, and the
codec rejects invalid state codes before raw decoding.

## Target handling

The migrated model remains an unconditional joint generator. Target is not a
separate conditioning label:

- a continuous target is standardized by the codec and enters the Gaussian
  block;
- a discrete or categorical target is encoded by the codec and enters the
  multinomial block;
- every sample contains the target in canonical table order.

This intentionally differs from the TabDDPM paper's conditional
classification configuration and preserves the joint-generation behavior of
the repository wrapper. A future conditional variant would require a separate
model contract and review; it is not inferred from `TaskType`.

## Native mathematics preserved

The extraction keeps the current repository behavior for:

- the `MLPDiffusion` denoiser and time embedding;
- Gaussian epsilon prediction and configured Gaussian loss;
- per-column multinomial diffusion with real cardinalities;
- the shared beta schedule and number of diffusion timesteps;
- the sum of mean Gaussian loss and multinomial variational loss normalized by
  the number of state columns;
- AdamW, fixed optimizer-step training, linear learning-rate annealing, and
  EMA update after every optimizer step;
- ancestral reverse diffusion and optional EMA sampling.

Native container extraction does not justify changes to these choices.

## Explicit native corrections

The extraction also makes three previously declared behaviors effective. They
are model-internal corrections, not adapter compatibility logic:

- `TabDDPMConfig.seed` is applied before denoiser construction, DataLoader
  shuffling, diffusion timestep selection, and training-noise generation;
- `TabDDPMConfig.gaussian_loss_type` is passed to
  `GaussianMultinomialDiffusion` instead of being silently ignored;
- the zero-valued loss for an absent Gaussian or multinomial block is created
  on the input tensor's device, so numerical-only and state-only training do
  not introduce a CPU/CUDA/MPS device mismatch.

The default configuration still uses Gaussian MSE, so forwarding that default
does not change its loss. Applying the documented training seed makes native
training reproducible and intentionally corrects the legacy wrapper, where the
field previously had no effect.

## Legacy discrepancies recorded for review

- The legacy wrapper depends on `TabularSchema` and fitted transform metadata;
  the new adapter does not.
- Legacy discrete numeric output is evaluated after rounding. The new path
  instead models declared finite supports as multinomial states.
- Legacy sampling clips categorical codes and can resample training IDs. The
  new adapter returns native state samples unchanged, and identifiers never
  enter `PreparedTable`.
- Legacy target preprocessing is inconsistent because target is excluded from
  schema feature groups and then classified again inside the wrapper. The new
  codec applies the target's declared `ColumnKind` exactly like every other
  modeled column.
- Older tuning varies `n_epochs` while the non-null default `steps` takes
  precedence, so those epoch values are not reliable configuration evidence.
- The mixed-data tuner's explicit optimizer-step search is the relevant
  search-space evidence, but model-owned tuning is outside this adapter change.

## Supported semantics

The adapter supports canonical tables containing any combination of
standardized continuous columns and encoded finite-state columns supported by
the native solver. It uses per-column train cardinalities and ignores ordinal
adjacency because TabDDPM has no ordered transition kernel.

Raw categorical values, raw numeric discrete values, missing-value handling,
identifiers, generic preprocessing, train/test splitting, tuning, and metrics
are outside the adapter boundary.

## Local characterization

The following checks exercise the real native implementation on CPU with one
optimizer step and two diffusion timesteps; this profile tests boundaries and
is not a quality benchmark:

```bash
conda run -n lightning11 python -W ignore -m unittest -q \
  tests.benchmark.test_tabddpm_native \
  tests.benchmark.test_tabddpm_adapter \
  tests.benchmark.test_runner_tabddpm \
  tests.benchmark.test_import_boundaries
```

This focused command passes 15 tests. The complete benchmark test discovery
also passes 144 tests:

```bash
conda run -n lightning11 python -m unittest discover \
  -s tests/benchmark -v
```

Both commands were run in the `lightning11` environment. The native smoke
tests emit the implementation's existing diffusion-timestep progress output.
The full discovery also emits NumPy 2 deprecation warnings; neither changes
the assertions. PyTorch 2.12.0 reports MPS as unavailable in the current
execution environment, so the device-safe absent loss is covered structurally
and on CPU but still needs a real MPS/CUDA run.

The native tests cover mixed, numerical-only, and state-only layouts, exact
multinomial state output, effective training seeds, configured Gaussian loss,
and compatibility of the legacy wrapper. Adapter tests cover canonical block
order, dtypes, per-column cardinalities, target preservation, config copying,
and schema identity. The runner smoke creates two real folds, fits fresh codecs
and native models, and decodes finite states without output repair.

Model-owned Optuna tuning, a frozen production configuration, a full real
dataset quality run, and comparison with published TabDDPM results remain
separate follow-up work.
