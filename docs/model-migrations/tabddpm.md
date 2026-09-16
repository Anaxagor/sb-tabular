# TabDDPM migration note

Status: corrected routing uses Gaussian diffusion for continuous and numeric
discrete columns, and multinomial diffusion only for categorical columns.
Discrete Gaussian output uses the approved integer-rounding convention below.
Previous finite-state discrete pilot results describe a different variant.

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

## Implemented semantic input decision

```python
InputSpec(
    continuous_view=ContinuousView.STANDARD,
    discrete_view=DiscreteView.RAW_VALUES,
    categorical_view=CategoricalView.FINITE_STATE_CODES,
)
```

The fold-local shared codec owns population mean/standard-deviation scaling for
each continuous column and every reversible train-observed codebook. The
adapter does not fit preprocessing or read held-out rows. This follows the
project's current experiment specification: continuous columns use
StandardScaler semantics, while generated values are decoded back to raw units
before final evaluation.

Numeric discrete input retains its raw values and distances, matching the
legacy wrapper's Gaussian routing. It is neither standardized nor renumbered.
The previous adapter's multinomial treatment of discrete input was a protocol
error. Correcting it changes network width and which loss models those columns;
old Optuna trials and final samples cannot be reused for the corrected run.

## Canonical table to native API

The mapping is fixed:

| Canonical source | Native value | Shape | dtype |
| --- | --- | --- | --- |
| `column_order` filtered by continuous + discrete (standard continuous, raw discrete) | numeric train block | `(N, D_num)` | `torch.float32` |
| `PreparedSchema.categorical_columns` | state train block | `(N, D_state)` | `torch.int64` |
| same state names | per-column cardinalities | `D_state` Python values | positive `int` |

Both native blocks follow canonical order. Only categorical names supply
cardinalities. TabDDPM does not consume the categorical `ordered` flag: its
multinomial transitions treat even ordinal categories symmetrically.

The codec has already validated dense state codes and train-observed
cardinalities. The adapter converts containers and dtypes but does not repeat
generic schema, support, missing-value, or row validation.

## Native sample to canonical table

The native sampler returns separate numeric and state blocks. The adapter
moves them to CPU, labels them with the fitted block names, and reassembles one
DataFrame in `PreparedSchema.column_order`. The returned `PreparedTable`
carries the exact schema object received by `fit`.

Only discrete Gaussian outputs are rounded by `np.rint`, with ties to even:
`1.5 -> 2`, `2.5 -> 2`, `-1.5 -> -2`. Rounded output remains floating-point to
avoid converting NaN/Inf into integers. There is no clipping or nearest-support
projection: values outside the train range remain visible as model error.
Categorical state codes and continuous values are untouched. Shared validation
still rejects non-finite output and invalid categorical codes.

This output convention is explicit and applies to the one synthetic table
used by tuning, quality metrics, and TSTR. Legacy code rounded only inside JS;
moving quantization before evaluation is a documented protocol change.

## Target handling

The migrated model remains an unconditional joint generator. Target is not a
separate conditioning label:

- a continuous target is standardized by the codec and enters the Gaussian
  block;
- a discrete target retains its raw numeric values, enters the Gaussian block,
  and is rounded on output;
- a categorical target is encoded and enters the multinomial block;
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

Native container extraction does not justify changes to these choices. Exact
repository sources for these claims are linked below for model-owner review.

## Explicit native corrections

The extraction also makes six previously declared behaviors effective. They
are model-internal corrections, not adapter compatibility logic:

- `TabDDPMConfig.seed` is applied before denoiser construction, DataLoader
  shuffling, diffusion timestep selection, and training-noise generation;
- `TabDDPMConfig.gaussian_loss_type` is passed to
  `GaussianMultinomialDiffusion` instead of being silently ignored;
- the zero-valued loss for an absent Gaussian or multinomial block is created
  on the input tensor's device, so numerical-only and state-only training do
  not introduce a CPU/CUDA/MPS device mismatch;
- a linear schedule with at most 20 timesteps fails before model construction,
  because the inherited scaled schedule would otherwise produce a beta greater
  than or equal to one and invalid diffusion probabilities;
- Gaussian posterior coefficients are registered as device-local `float32`
  buffers. Their formulas are unchanged, but sampling no longer attempts to
  convert a retained `float64` tensor on MPS, which that backend rejects.
- the native fit inspects its scalar loss about 100 times and raises a typed
  numerical error when it observes NaN or infinity. This stops an already
  invalid fit before sampling without changing any finite-loss optimizer or
  EMA update.

The default configuration still uses Gaussian MSE, so forwarding that default
does not change its loss. Applying the documented training seed makes native
training repeatable under the deterministic behavior available from the chosen
Torch backend; it does not promise bitwise equality across devices. This
intentionally corrects the legacy wrapper, where the field previously had no
effect.

## Repository evidence for model-owner review

| Reviewed claim | Repository evidence |
| --- | --- |
| Legacy numeric/state selection and target reclassification | [`TabDDPMWrapper._numeric_block_cols`, `_categorical_block_specs`, and `_preprocess_data`](../../sbtab/baselines/tabddpm/model.py) |
| Legacy reconstruction, clipping, and identifier behavior | [`TabDDPMWrapper._reconstruct_output_df` and `sample`](../../sbtab/baselines/tabddpm/model.py) |
| Gaussian plus multinomial loss and ancestral sampler | [`GaussianMultinomialDiffusion.mixed_loss`, `sample`, and `sample_all`](../../sbtab/baselines/tabddpm/gaussian_multinomial_diffsuion.py) |
| Denoiser architecture | [`MLPDiffusion`](../../sbtab/baselines/tabddpm/modules.py) |
| Mixed-data tuning and rounding of discrete values | [`_js_for_discrete_numeric`, `compute_composite_metric`, and `make_objective_for_dataset`](../../sbtab/experiments/tuning_script/tabddpm_mixed_data_tuning.py) |
| Older `n_epochs` search whose default `steps` takes precedence | [`make_objective_for_dataset`](../../sbtab/experiments/tuning_script/tabddpm_tuning.py) and [`TabDDPMConfig`](../../sbtab/baselines/tabddpm/native.py) |
| Legacy final K-fold config and processed-space evaluation | [`build_tabddpm_config_from_best` and `main`](../../sbtab/experiments/calculating_metrics/tabddpm_metrics.py) |
| Published model uses quantile-normal preprocessing, while this benchmark specification requires train-fold StandardScaler semantics | [Published TabDDPM implementation](https://github.com/yandex-research/tab-ddpm/blob/main/lib/data.py) and the project experiment table |
| Conditional classification is a distinct published variant | [TabDDPM paper](https://arxiv.org/abs/2209.15421) |

## Legacy discrepancies recorded for review

- The legacy wrapper depends on `TabularSchema` and fitted transform metadata;
  the new adapter does not.
- Legacy discrete JS rounds real and generated numeric values inside the
  metric. The corrected path accepts integer discrete training values and
  rounds generated discrete values once before any metric or TSTR pipeline.
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

The adapter supports combinations of standardized continuous, raw integer
discrete, and encoded categorical columns, including empty Gaussian or
multinomial blocks. Fractional discrete training values fail explicitly before
native construction because integer rounding would change their domain. The
14-dataset bundle has 33 declared discrete columns, all integer-valued.

An explicit `ColumnSpec.ordered_values` still declares a closed raw domain:
shared decoding rejects generated values outside it, without projection. The
14-dataset declarations do not impose such a domain on numeric discrete columns.

Raw categorical values, fractional discrete values, missing-value handling,
identifiers, generic preprocessing, train/test splitting, tuning, and metrics
are outside the adapter boundary.

## Local characterization

The following checks exercise the real native implementation on CPU with one
optimizer step and two diffusion timesteps; this profile tests boundaries and
is not a quality benchmark:

```bash
conda run -n lightning11 python -m unittest \
  tests.benchmark.test_tabddpm_native \
  tests.benchmark.test_tabddpm_adapter \
  tests.benchmark.test_runner_tabddpm \
  tests.benchmark.test_import_boundaries
```

Run the complete benchmark suite with:

```bash
conda run -n lightning11 python -m unittest discover \
  -s tests/benchmark -p 'test_*.py'
```

Native smoke tests exercise CPU boundaries. They are not full quality runs
or accelerator performance measurements. Exact verification counts and
commands for this correction are recorded in the PR handoff.

The native tests cover mixed, numerical-only, and state-only layouts, exact
multinomial state output, effective training seeds, configured Gaussian loss,
repeatable sample seeds, typed EMA selection, and compatibility of the legacy
wrapper. Adapter tests cover canonical block order, dtypes, per-column
cardinalities, target preservation, config copying, schema identity, and exact
agreement with a direct native call. The runner smoke creates mixed and pure
Gaussian folds, fits fresh codecs and native models, preserves categorical and
continuous targets, and decodes categorical states unchanged. Discrete tests
exercise integer quantization, unsupported fractional input, and new numeric
outputs outside train support.
Import-boundary tests cover both the legacy-free adapter import and the lazy
public compatibility export.

## Review and publication boundary

The routing correction is based on `feat/tabddpm-full-pilot`. Review its
shared raw-discrete codec correction separately from its TabDDPM routing,
quantization, and protocol-version change. No native solver formula is edited.
Agents prepare local commits and handoff text; a human publishes the PR.

## Fixed-configuration holdout score

The follow-up command below downloads UCI 468, applies the same 80/20
stratified holdout and raw-space tuning objective used by model-owned studies,
and writes one create-only JSON artifact:

```bash
python -m sbtab.benchmark.pilots.tabddpm_online_shoppers_score \
  --output-json artifacts/tabddpm-online-shoppers-score.json \
  --device cuda
```

Pass `--csv path/to/table.csv` to avoid network acquisition. The printed
`total_score` is minimized and belongs only to the exact native configuration
recorded in the artifact. Interactive CLI runs show separate training and
sampling progress bars by default; pass `--no-progress` for redirected logs or
automation. This command does not search hyperparameters or run final K-fold
quality/TSTR evaluation.

## Staged Optuna and complete metric run

The full Online Shoppers entrypoint searches model-owned hyperparameters on the
reference 80/20 stratified holdout, reranks three distinct leaders at 30,000
steps on two seed pairs, freezes one configuration, and runs the common final
five-fold protocol:

```bash
python -m sbtab.benchmark.pilots.tabddpm_online_shoppers \
  --output-dir artifacts/tabddpm-online-shoppers-optuna-v3 \
  --device mps
```

The command creates a local SQLite study automatically. If it is stopped after
a completed Phase-A trial or completed Phase-B seed run, repeat the same
command with `--resume`. See `docs/benchmark-artifacts.md` for the exact
recovery boundary and fingerprint checks.

Start v3 in a new output directory and study. Do not resume v1/v2 trials,
rerank state, or final generation: they used a different discrete routing.

Final evaluation reports decoded raw-space WD, 50-bin continuous KL,
exact-support discrete/categorical KL, Pearson/Spearman/NMI association
distances, train-standardized continuous RBF MMD, and CatBoost TSTR macro-F1.
None of those report metrics is optimized per trial.

Comparison with the published conditional TabDDPM results remains a separate
study: this repository's migrated model is an unconditional joint generator.

### Preliminary fixed-config report while Optuna runs

To produce a clearly labelled comparison row before the Optuna study finishes,
run the already characterized `10k / T=100 / batch=512 / [128,256,128]`
configuration directly through the final five-fold protocol:

```bash
python -m sbtab.benchmark.pilots.tabddpm_online_shoppers_fixed \
  --output-dir artifacts/tabddpm-online-shoppers-fixed-10k \
  --device mps
```

This command never reads or creates an Optuna study. `report.md` contains a
shareable `mean ± population-std` row for continuous Mean KL, train-standardized
Mean WD, Pearson correlation distance, `% F1_real - F1_synth`, applicable R²
degradation, and continuous MMD. `comparison-metrics.json` retains the exact
conventions and per-fold values; `evaluation/metrics.json` additionally retains
raw-unit WD plus every discrete and categorical metric.
