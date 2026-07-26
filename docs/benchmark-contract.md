# Unified benchmark contract

Status: draft for maintainer and model-owner review.

Evidence base: repository `main` at `52e86e0`. This document defines the target
boundary and records its incremental implementation. The greenfield contracts,
dataset declaration, missing policy, splitting, codec, adapter protocol, MSBM
adapter, and fixed-configuration holdout/cross-validation runners now exist
under `sbtab.benchmark`; MSBM-owned tuning and the common tuning objective are
also implemented. The create-only cross-validation generation artifact is
implemented, as is the create-only MSBM tuning artifact. Final evaluation
integration, other model-owned tuners, and further model adapters remain
migration work.

## Goal

Implement the dataset-to-report workflow once while preserving each model's
training and sampling semantics:

```text
TabularDataset -> COMPLETE_CASE -> split
                                  |-- train raw -> fold-local model codec
                                  |                -> PreparedTable
                                  |                -> model adapter
                                  |                -> PreparedTable sample
                                  |                -> decode -> synthetic raw
                                  `-- test raw ---------------------> evaluate
```

The benchmark owns the experiment protocol. An adapter is only a translation
layer between one canonical prepared table and one model's native API.

## Dependency boundary

```text
benchmark core -> benchmark adapters -> existing models/solvers
benchmark core -> evaluation
```

Models and solvers must not import benchmark or evaluation code. The benchmark
may wrap existing implementations, but it must not copy their algorithms.

### Greenfield benchmark core

The benchmark orchestration is a new subsystem under `sbtab/benchmark/`. It
owns its contracts, dataset declarations, global policies, splitting, codec,
runner, artifacts, and adapters. In particular, implementing the new runner
does not mean extending the current `sbtab.data` splitter or `DataModule`,
calling the existing transform pipelines, or importing target mappings from
`sbtab.experiments`.

Legacy modules are inspected to understand native model behavior and to find
methodological defects. They are not runtime dependencies of the new core and
do not define its public API. If an old entrypoint is later moved onto the new
runner, that is a separate migration task after the new path works directly.
The intentional reuse boundary is the existing model or solver behind a thin
adapter; model mathematics is not rewritten in the benchmark package.

The boundary deliberately separates three kinds of information:

- `TabularDataset` is the single public object containing raw rows and their
  column semantics.
- `InputSpec` declares only the train-fitted representations requested by a
  model.
- `PreparedTable` is the single canonical object exchanged with every adapter.

Native containers, tensor dtypes, devices, data loaders, and an `X`/`y` split
are not part of the shared contract. They stop at the adapter boundary.

## Existing native APIs are evidence, not the contract

The repository currently contains several native input shapes. They explain
what individual adapters must translate, but they do not justify adding
layouts or dtypes to `InputSpec`.

The names `continuous_time` and `discrete_time` in solver paths describe time
discretization, not support for continuous or categorical table columns.

| Family | Existing implementation | Observed native input | Adapter-local translation |
| --- | --- | --- | --- |
| Raw mixed table | `CTGANWrapper` | Raw-like `pandas.DataFrame` and schema | Select modeled columns and call the native wrapper/library |
| Mixed diffusion | `TabDDPMWrapper` | Numeric values plus categorical state codes | Split canonical columns into native numeric/state blocks |
| Supervised generator | `TabPFGenGenerative` | Numeric `X` and separate `y` | Mechanically extract target for the native call and reattach it to the sample |
| Flat numeric | `STaSyGenerative`, `LightSBSolver`, joint MLP IPF/DSBM solvers | Two-dimensional float matrix | Select ordered columns and convert to the required array/tensor |
| Named numeric frame | Joint and feature-wise CatBoost solvers | Numeric `pandas.DataFrame` | Preserve names and order expected by the native solver |
| Mixed continuous/state | `MixedSBMSolver` | `float32` continuous tensor and `int64` state tensor | Build both tensors and native state metadata inside the adapter |
| Finite-state only | `CSBMSolver` | `int64` state tensor and data loaders | Build the tensor, loaders, and reference-process arguments inside the adapter |

Do not reproduce a questionable legacy choice merely because it appears in an
evaluation script. The native model, tuning path, evaluation path, and model
owner must be compared before approving an adapter specification.

## `TabularDataset`: one public dataset object

The new benchmark API does not expose the existing `sbtab.data.TabularSchema`
and does not add a `DatasetSpec` wrapper around it. Both choices would make the
target architecture inherit legacy assumptions, especially parallel feature
lists and a target excluded from those lists.

Callers and the runner use one self-contained object:

```python
@dataclass
class TabularDataset:
    # Stable label used only in logs, reports, and artifact paths.
    # Shared code and adapters must never dispatch on this value.
    name: str

    # Raw table. It contains modeled columns and may contain one identifier.
    frame: pd.DataFrame

    # All modeled columns in canonical table order. Target, when present, is
    # an ordinary member of this sequence.
    columns: tuple[ColumnSpec, ...]

    # Optional column used by utility evaluation. This is a label, not an
    # instruction to remove the column or preprocess it separately.
    target: str | None = None

    # Classification or regression semantics for the declared target.
    task: TaskType | None = None

    # Optional raw identifier. It is present in frame but absent from columns
    # and therefore never reaches the codec's modeled table or an adapter.
    identifier: str | None = None
```

Every modeled column describes its own semantics instead of participating in
parallel name lists:

```python
@dataclass(frozen=True)
class ColumnSpec:
    # Column name in TabularDataset.frame.
    name: str

    # Continuous, discrete, or categorical.
    kind: ColumnKind

    # Explicit semantic order for an ordinal finite-state column. None means
    # no explicit ordinal domain. Continuous columns must use None.
    ordered_values: tuple[object, ...] | None = None
```

```python
class ColumnKind(Enum):
    CONTINUOUS = "continuous"
    DISCRETE = "discrete"
    CATEGORICAL = "categorical"
```

The continuous, discrete, and categorical column groups are computed
properties of `columns`. They are not stored as additional lists. The kind of
target is obtained from its `ColumnSpec`; there is no separate `target_kind`.

Validation rules:

- `ColumnSpec.name` values are unique and all exist in `frame`;
- `columns` preserves the canonical modeled output order;
- `target`, when present, names one of `columns`;
- `task` and `target` are either both present or both absent;
- `identifier`, when present, exists in `frame` and is absent from `columns`;
- continuous columns cannot declare `ordered_values`;
- values in `ordered_values` are unique and cover observed non-null values;
- categorical order is never inferred from a dataset name, lexicographic
  sorting, or encoder-assigned integer codes;
- no adapter reclassifies a column from pandas dtype or observed cardinality.

Example:

```python
dataset = TabularDataset(
    name="example",
    frame=df,
    columns=(
        ColumnSpec(name="age", kind=ColumnKind.CONTINUOUS),
        ColumnSpec(
            name="education",
            kind=ColumnKind.CATEGORICAL,
            ordered_values=("school", "bachelor", "master", "phd"),
        ),
        ColumnSpec(name="city", kind=ColumnKind.CATEGORICAL),
        ColumnSpec(name="income", kind=ColumnKind.CATEGORICAL),
    ),
    target="income",
    task=TaskType.CLASSIFICATION,
    identifier="row_id",
)
```

### Legacy boundary

The current `sbtab.data.TabularSchema` may be read only by a temporary migration
bridge:

```text
raw DataFrame + legacy TabularSchema
                 -> from_legacy(...)
                 -> TabularDataset
                 -> all new benchmark code
```

The bridge must materialize explicit `ColumnSpec` values and task semantics.
It must not be imported by the runner, codec, adapters, or evaluator. New
dataset loaders return `TabularDataset` directly. Remove the bridge after the
last old loader is migrated. The bridge is an optional compatibility shell
outside the new core, not a reason for new code to depend on `sbtab.data`.

## `InputSpec`: only semantic representations

The public model specification contains exactly three fields:

```python
@dataclass(frozen=True)
class InputSpec:
    # Representation requested for continuous modeled columns.
    continuous_view: ContinuousView

    # Representation requested for integer-like discrete modeled columns.
    discrete_view: DiscreteView

    # Representation requested for nominal/ordinal categorical modeled columns.
    categorical_view: CategoricalView
```

MVP values are intentionally limited to representations justified by current
models and existing transforms:

```python
class ContinuousView(Enum):
    RAW = "raw"
    STANDARD = "standard"
    UNSUPPORTED = "unsupported"


class DiscreteView(Enum):
    RAW_VALUES = "raw_values"
    FINITE_STATE_CODES = "finite_state_codes"
    UNSUPPORTED = "unsupported"


class CategoricalView(Enum):
    RAW_VALUES = "raw_values"
    FINITE_STATE_CODES = "finite_state_codes"
    UNSUPPORTED = "unsupported"
```

Meaning:

- `RAW` and `RAW_VALUES` preserve the raw values seen by the benchmark codec.
- `STANDARD` fits location and scale on the training partition only.
- `FINITE_STATE_CODES` fits a reversible mapping to integer codes `0..K-1` on
  the training partition only.
- `UNSUPPORTED` rejects a dataset when the corresponding modeled group is
  non-empty. It must not silently drop or cast that group.

The target remains a modeled table column. Its `ColumnSpec.kind` selects the
same view as every other column of that kind. Marking it in
`TabularDataset.target` does not imply an `X`/`y` split. An adapter may
temporarily split it only when a native API requires that call shape, and must
return it in the sampled table.

`ONE_HOT` and `QUANTILE_NORMAL` are deliberately absent from the MVP. Add a new
view only in a shared contract PR that includes a real model requirement, a
fold-local implementation, inverse transformation, and tests. Do not add enum
values for hypothetical future use.

The following do not belong in `InputSpec`:

- DataFrame/matrix/split-block layout;
- NumPy or Torch dtype;
- target mode;
- missing-value capability;
- identifier policy;
- required payload blocks;
- device, data-loader, or sampling settings;
- model hyperparameters.

## `PreparedTable`: one canonical adapter input

Every adapter receives and returns the same shape of object:

```python
@dataclass(frozen=True)
class PreparedTable:
    # Prepared modeled columns in canonical order. ID is never present.
    frame: pd.DataFrame

    # Semantic information needed to select columns and validate state spaces.
    schema: PreparedSchema
```

There is no union of payload layouts. A DataFrame is the canonical interchange
format because it preserves names, order, mixed dtypes, and the target column.
An adapter converts it to its native format locally.

The MVP representations do not expand a column, so prepared and decoded tables
retain a one-to-one column mapping.

```python
@dataclass(frozen=True)
class PreparedSchema:
    # Exact order expected from sample(). Includes target, excludes ID.
    column_order: tuple[str, ...]

    # Prepared semantic groups derived from ColumnSpec.kind. If a target exists,
    # it appears in exactly one of these groups like any other modeled column.
    continuous_columns: tuple[str, ...]
    discrete_columns: tuple[str, ...]
    categorical_columns: tuple[str, ...]

    # Label used by evaluation and by native APIs that require a separate y.
    # It remains a normal column in frame.
    target_col: str | None

    # Dataset task semantics needed by evaluation and supervised native APIs.
    task_type: TaskType | None

    # Metadata only for columns represented as finite-state codes.
    state_columns: Mapping[str, StateColumn]
```

Finite-state metadata is keyed by column name rather than stored in parallel
arrays:

```python
@dataclass(frozen=True)
class StateColumn:
    # Valid prepared codes satisfy 0 <= code < cardinality.
    cardinality: int

    # True only when transitions between neighbouring codes have ordered
    # meaning. Numeric discrete columns are ordered; nominal categories are not.
    ordered: bool
```

An MSBM or CSBM adapter may construct native `cardinalities` and
`ordered_mask` arrays from this mapping in its selected column order. Those
arrays are native model arguments, not common batch fields.

The fitted codec, not `PreparedSchema`, owns reversible implementation state:

- means and scales;
- category-to-code and code-to-category mappings;
- raw discrete and categorical supports;

Physical pandas dtypes are not a model capability and are therefore not part
of `InputSpec` or `PreparedSchema`. Decoding restores exact finite-state raw
values and canonical modeled-column order. A standardized continuous column is
decoded as real numeric values even when the source happened to use an integer
storage dtype; the codec never rounds generated values merely to reproduce a
pandas dtype. Missing-value evidence belongs to `MissingReport`, and any new
identifier is added later by the runner, not by the codec.

## `ModelAdapter`: thin native translation

```python
class ModelAdapter(Protocol):
    @property
    def name(self) -> str: ...

    @property
    def input_spec(self) -> InputSpec: ...

    def fit(self, train: PreparedTable, context: RunContext) -> None: ...

    def sample(self, n: int, seed: int) -> PreparedTable: ...
```

One adapter instance belongs to one fitted fold. It stores the prepared schema
received by `fit` and returns the same schema with every sample. The runner
creates a fresh codec and adapter for every fold; it must not reuse a fitted
adapter with another fold's codec. This lifecycle removes the need for a public
`codec_id` field.

Adapter metadata can be checked without fitting through
`validate_adapter_definition`. Official benchmark sampling uses a positive
integer row count and a non-negative 32-bit seed; the runner uses
`validate_sample_request` before entering native code. Runtime structural
validation confirms that `fit` and `sample` are callable, but Python protocols
do not inspect their signatures; static annotations and adapter boundary tests
remain required.

The codec validates prepared train data before returning it to the runner.
Adapters trust that boundary and do not repeat generic schema, missing-value,
support, or row validation. After sampling, the runner validates the requested
row count and the returned prepared table before codec decoding.

An adapter may:

- select and order columns using `PreparedSchema`;
- mechanically extract target as `y` if the native API requires it;
- concatenate or split continuous and state columns;
- convert the canonical frame to native NumPy/Torch dtypes and device;
- construct model-specific data loaders, reference processes, and native model
  configuration;
- call the existing native `fit` and `sample` methods;
- assemble native samples back into a `PreparedTable` with the original
  prepared column order.

An adapter may not:

- create a train/test split;
- fit generic scalers, imputers, or category encoders;
- infer target, task, ordinal order, or category semantics from dataset name;
- inverse-transform a prepared sample to raw values;
- compute benchmark metrics;
- silently drop unsupported columns;
- clip, round, pad, or replace invalid finite-state output;
- change losses, priors, reference processes, schedules, direction sequences,
  training steps, or sampling algorithms to make a test pass.

Native layout and dtype conversion belong in the adapter. Generic learned
preprocessing and inverse preprocessing do not.

## Runtime context and model configuration

Experiment-wide controls are selected once for every model in a comparison:

```python
class MissingPolicy(Enum):
    # Safe default: fail with per-column counts until a policy is explicit.
    ERROR = "error"

    # Remove rows missing any modeled column once, before creating splits.
    COMPLETE_CASE = "complete_case"


@dataclass(frozen=True)
class BenchmarkConfig:
    split: SplitConfig
    missing_policy: MissingPolicy = MissingPolicy.ERROR
    run_id: str = "benchmark"
    training_seed: int = 42
    sample_seed: int = 10_042
    device: str = "cpu"
    artifact_dir: Path = Path("artifacts")
```

`MissingPolicy` belongs to `BenchmarkConfig`, not `TabularDataset`,
`ColumnSpec`, `InputSpec`, or an adapter. The same dataset may be run under a
different named profile, while every model inside one comparison must receive
the same policy and resulting rows. `split` selects the final cross-validation
protocol. The remaining fields provide fold-local runtime controls; the runner
derives fold seeds by adding `fold_id` to each base seed.

Per-fold runtime controls are separate from both the dataset and model config:

```python
@dataclass(frozen=True)
class RunContext:
    run_id: str
    fold_id: int
    seed: int
    device: str
    artifact_dir: Path
```

`RunContext` validates non-empty run/device labels, a non-negative fold index,
a non-negative 32-bit training seed, and a `Path` artifact destination. Merely
constructing it never creates the directory; artifact lifecycle belongs to the
runner.

Sampling variants such as TabDDPM EMA, LightSB SDE mode, or integration steps
belong in a typed config for that adapter. They must not leak into the shared
API as unrestricted `**kwargs`.

## Codec and global benchmark policies

The v1 runner applies the missing policy once, creates common splits, and then
compiles one fold-local model codec per model/fold:

```text
missing_result = apply_missing_policy(
    dataset,
    config.missing_policy,
)
run_dataset = missing_result.dataset
missing_report = missing_result.report
splits = make_splits(run_dataset, config.split)

codec = compile_codec(run_dataset, adapter.input_spec)
train_prepared = codec.fit_transform(train_raw)
adapter.fit(train_prepared, run_context)
validate_sample_request(n, sample_seed)
sample_prepared = adapter.sample(n, sample_seed)
validate_prepared_table(sample_prepared, expected_rows=n)
sample_raw = codec.inverse_transform(sample_prepared)

report = evaluate(
    real_train=train_raw,
    real_test=test_raw,
    synthetic=sample_raw,
)
```

The model codec never transforms held-out test data. A generator has no model
inference step on test: it trains on prepared train data and produces a new
table. Evaluation operates on decoded synthetic and raw real tables. A TSTR or
other utility evaluator owns its own downstream predictive preprocessing; it
must not reuse the generator's codec.

The codec must:

1. exclude `TabularDataset.identifier` from modeled columns;
2. use each declared `ColumnSpec.kind`, including target, without guessing;
3. reject non-empty groups declared `UNSUPPORTED`;
4. fit all learned transformations and finite-state codebooks on train only;
5. transform train for the adapter and inversely transform only model samples;
6. emit one validated `PreparedTable` in canonical order;
7. invert the prepared sample to raw semantic values and canonical modeled
   column order;
8. reject invalid state codes instead of clipping them.

Finite-state cardinality is the number of states observed in train. A category
seen only in raw test is not added to the model state space, mapped to a shared
`UNKNOWN`, or treated as a codec failure. The raw evaluator includes it in the
real support, where its absence from synthetic data is a legitimate quality
signal.

The v1 transforms are deterministic:

- `STANDARD` uses the train population mean and standard deviation
  (`ddof=0`); a constant train column uses scale `1.0`;
- numeric discrete state codes follow ascending train-observed raw values;
- explicitly ordinal categorical codes follow `ordered_values`, filtered to
  values observed in train;
- nominal categorical codes follow first appearance in train row order;
- a generated `RAW_VALUES` discrete or categorical value must belong to that
  column's train support.

`ModelCodec` is single-use and intentionally exposes no transform operation for
held-out data. `compile_codec` validates dataset declarations, while learned
means, scales, supports, and mappings are created only by
`fit_transform(train_raw)`. Inverse decoding also supports a valid zero-row
`PreparedTable`; this keeps the data contract total even though official
benchmark folds and pilot samples are non-empty.

V1 missing semantics are fixed:

- `ERROR` is the safe default. It reports missing counts by modeled column and
  stops before splitting.
- The official v1 comparison explicitly selects `COMPLETE_CASE`.
- `COMPLETE_CASE` drops a row when any name in `TabularDataset.columns` is
  missing. The optional identifier is ignored.
- Filtering occurs exactly once before split; every model receives the same
  retained rows and split indices.
- Metrics compare synthetic data with the filtered real train/test data, never
  with the unfiltered source table.
- Run artifacts record rows before/after, dropped count and fraction, missing
  count per column, and task-class distribution before/after when applicable.
- If filtering leaves an invalid dataset or split, the dataset fails with an
  explicit reason instead of falling back to imputation.

`IMPUTE`, reversible missingness preservation, and model-native missing values
are not v1 enum values. They require separate future profiles, contract review,
and result labels; adapters must not implement a private fallback.

`apply_missing_policy` returns one immutable `MissingPolicyResult` containing
the post-policy `TabularDataset` and its `MissingReport`. Under `ERROR`, modeled
missing values raise `MissingValuesError` carrying that same report and no rows
are removed. Reports snapshot modeled-column missing counts and applicable raw
class counts; they never include the optional identifier.

V1 split strategies are new benchmark-owned objects:

- `HoldoutConfig(validation_fraction=0.2, seed=5)` creates the shuffled
  train/validation split used to tune a fixed model family;
- `StratifiedHoldoutConfig(validation_fraction=0.2, seed=5)` creates the same
  tuning split while preserving classification-target proportions;
- `KFoldConfig(n_splits, seed)` creates deterministic shuffled positional
  folds without reading target values;
- `StratifiedKFoldConfig(n_splits, seed)` requires a finite-state
  classification target and preserves its proportions in every held-out fold;
- stratification requires at least two observed classes. Raw finite-state
  labels may be strings, numbers, or mixed hashable values: the splitter
  factorizes them to temporary integer labels for scikit-learn without changing
  the target column in `TabularDataset`;
- every target class must contain at least `n_splits` post-policy rows;
- split seeds are explicit non-negative 32-bit integers;
- `FoldSplit` stores immutable train/test positions into the post-policy raw
  frame, not pandas index labels.

Splitters reject modeled missing values and instruct the caller to apply the
global policy first. They do not import or extend the legacy `sbtab.data`
splitter and do not preprocess train or held-out rows.

Identifiers follow one global rule: they never enter the model. If a decoded
output requires an identifier, the runner generates new identifiers after
sampling. It never resamples training identifiers.

## Reference experimental lifecycle

Tuning and final comparison are separate phases. They must not share fitted
codec or model state:

1. Apply the experiment's `MissingPolicy` once to the declared dataset.
2. For model-family tuning, create one shuffled 80/20 train/validation holdout
   with seed 5. Classification uses the stratified variant.
3. For every tuning trial, create a fresh codec and adapter, fit the codec only
   on the holdout train partition, train the generator, and generate exactly
   `len(validation_raw)` rows. The validation table remains raw and is used
   only by the tuning evaluator.
4. Select and freeze one typed adapter configuration. The selection procedure,
   search space, objective, and resulting configuration belong to tuning
   artifacts, not `InputSpec`.
5. For final comparison, create five shuffled folds with seed 42.
   Classification uses target stratification.
6. For every final fold, create a fresh codec and adapter, fit only on that
   fold's train partition, and generate exactly `len(train_raw)` rows with the
   frozen adapter configuration.
7. Decode the generated table and give raw train, raw held-out test, and raw
   synthetic tables to model-independent evaluation.

The shared runners implement the data lifecycle in step 3 and steps 5--7 for
an already fixed adapter configuration. They deliberately have no Optuna
dependency, metric selection, or hidden tuning behavior. A model-owned tuning
entrypoint defines its search space and creates one fresh fixed-config adapter
per trial; it calls the common holdout runner and common semantic objective.
The native model and adapter `fit()` remain unaware of Optuna.

The reference tuning objective is model-independent and operates after raw
decoding:

- each continuous modeled column contributes one-dimensional Wasserstein
  distance in its raw scale, without another normalization;
- each discrete or categorical modeled column contributes empirical
  Jensen--Shannon divergence using exact decoded values and natural logarithms;
- target participates according to its declared `ColumnKind`;
- `mean_wasserstein` averages continuous columns;
- `mean_jensen_shannon` averages discrete and categorical columns together;
- the minimized total is the sum of the group means that exist for the
  dataset.

This definition uses Jensen--Shannon *divergence*, not its square-root distance.
It never rounds decoded discrete values to repair preprocessing noise.
Per-column contributions are retained for review instead of reporting only the
composite scalar.

## Provisional family specifications

These rows are hypotheses for model-owner review, not compatibility claims.
Only the three semantic views are shared; the native input column above remains
adapter-local evidence.

| Family | Continuous | Discrete | Categorical | Review note |
| --- | --- | --- | --- | --- |
| CTGAN | `RAW` | `RAW_VALUES` | `RAW_VALUES` | Confirm that common missing and ID policies replace wrapper-owned behavior |
| TabDDPM | `STANDARD` | `RAW_VALUES` | `FINITE_STATE_CODES` | Confirm treatment of discrete numeric support and target column |
| TabPFGen | `STANDARD` | `RAW_VALUES` | unresolved | Decide whether encoded categorical features are mathematically supported or datasets must be restricted; do not add target mode to solve this |
| STaSy / LightSB / numeric SB solvers | `STANDARD` | `UNSUPPORTED` | `UNSUPPORTED` | Do not claim categorical support merely because codes can be cast to float |
| MSBM | `STANDARD` | `FINITE_STATE_CODES` | `FINITE_STATE_CODES` | **Approved pilot**; use train-observed cardinalities and explicit order semantics |
| CSBM | `UNSUPPORTED` | `FINITE_STATE_CODES` | `FINITE_STATE_CODES` | Use real per-column cardinalities; never replace them with a shared maximum |

An adapter task cannot start until its row is resolved to actual enum values
and approved with repository evidence.

## Approved pilot: MSBM

MSBM is the first adapter used to validate the contract:

```python
InputSpec(
    continuous_view=ContinuousView.STANDARD,
    discrete_view=DiscreteView.FINITE_STATE_CODES,
    categorical_view=CategoricalView.FINITE_STATE_CODES,
)
```

The codec fits continuous scaling and state codebooks on train only. The MSBM
adapter selects columns in canonical order, converts continuous values to the
native float tensor, converts finite-state codes to the native integer tensor,
and derives native cardinality/order arrays from named prepared metadata.

Order semantics are fixed:

- numeric discrete columns are ordered;
- categorical columns with explicit `ColumnSpec.ordered_values` are ordered;
- categorical columns without `ordered_values` are nominal and unordered.

MSBM's ordered reference uses rank adjacency, not distances between original
numeric values. For an ordered support such as `(0, 1, 10)`, codes preserve the
order but not the unequal numeric gaps.

The pilot integration run uses a real mixed dataset from the benchmark set,
not a synthetic quality benchmark. Small inline DataFrames remain appropriate
only for deterministic automated boundary tests such as invalid codes and
column-order failures.

The pilot dataset declaration and its split strategy belong to the new
benchmark core. Do not patch the current `sbtab.data` splitter and do not reuse
dataset-name/target mappings from legacy experiment scripts to make the pilot
run. A classification pilot may require a new target-aware stratified splitter;
that is shared benchmark protocol, not adapter behavior and not a legacy
extension.

### Pilot execution profile: Online Shoppers

The approved pilot dataset is UCI Online Shoppers Purchasing Intention,
dataset ID 468. The new dataset declaration uses `Revenue` as a categorical
modeled target and `TaskType.CLASSIFICATION`. It does not read target metadata
from an experiment script.

The canonical column semantics are:

| Kind | Columns | MSBM view | Ordered |
| --- | --- | --- | --- |
| Continuous | `Administrative_Duration`, `Informational_Duration`, `ProductRelated_Duration`, `BounceRates`, `ExitRates`, `PageValues`, `SpecialDay` | `STANDARD` | not applicable |
| Discrete | `Administrative`, `Informational`, `ProductRelated` | `FINITE_STATE_CODES` | yes |
| Categorical | `Month`, `OperatingSystems`, `Browser`, `Region`, `TrafficType`, `VisitorType`, `Weekend`, `Revenue` | `FINITE_STATE_CODES` | no |

`Month` is nominal for this pilot. MSBM's ordered reference is a line, whereas
calendar months are cyclic; declaring a linear order would encode the wrong
neighbourhood. `Revenue` remains in the generated table like every other
modeled column.

The benchmark protocol is one reproducible five-fold comparison:

- apply `MissingPolicy.COMPLETE_CASE` once before splitting;
- create a new target-stratified five-fold split with shuffle and base seed 42;
- create a fresh codec and MSBM adapter for every fold;
- use fold run seed `42 + fold_id`;
- generate exactly `len(train_raw)` rows with sample seed `10_042 + fold_id`;
- decode the sample to the raw schema and pass raw train, raw held-out test, and
  raw synthetic tables to evaluation;
- never transform or pass held-out test to MSBM.

The current vertical-slice runner does not tune MSBM: it accepts one fixed
typed adapter configuration. Its first smoke run records the native defaults
currently under model-owner review:

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

`device` and `seed` are supplied by `RunContext`; the adapter only translates
them into the existing native configuration. The current native config exposes
`eps`, but the MSBM implementation does not consume it. The run artifact must
not claim that `eps` affects training; whether to remove or implement it is a
separate model-owner decision.

Before an official benchmark result is reported, the fixed MSBM configuration
must come from the reference 80/20 tuning lifecycle above (or be explicitly
declared as an untuned baseline). A smoke run with native defaults validates
integration only; it is not a quality result.

## Resolved architectural decisions

1. **Canonical dataset:** new code uses `TabularDataset` with column-centric
   `ColumnSpec` metadata. There is no new `DatasetSpec`, and the existing
   `TabularSchema` is confined to a removable legacy bridge.
2. **Target:** target is one of the modeled `ColumnSpec` entries and follows the
   view for its declared kind. The target label exists for utility evaluation
   and native APIs, not for separate common preprocessing.
3. **Missing values:** `MissingPolicy` is selected once in `BenchmarkConfig`.
   `ERROR` is the safe default and the official v1 comparison explicitly uses
   `COMPLETE_CASE` before split for all modeled columns and all models.
4. **Held-out data:** the model codec fits/transforms train and inversely
   transforms generated samples. Held-out test stays raw and is used only by
   evaluators; unseen test categories do not enter train cardinalities.
5. **Pilot:** MSBM is selected with `STANDARD` continuous values and
   `FINITE_STATE_CODES` for both discrete and categorical values. UCI Online
   Shoppers (ID 468), target `Revenue`, is the approved five-fold pilot dataset.
6. **Implementation boundary:** the benchmark core is implemented as a new
   subsystem under `sbtab/benchmark/`. It reuses native models/solvers only
   through adapters and does not depend on old DataModules, splitters,
   transform orchestration, or experiment-level metadata.

## Decisions required before implementation

1. **Second family:** validate the contract with a model that has different
   semantic views. Native layout differences alone do not justify shared API
   changes.

The contract becomes stable only after a pilot and a second model work without
model-name branches in the codec, runner, or evaluator. If an adapter needs a
new semantic representation, propose a shared contract change with evidence;
do not smuggle native layout fields back into `InputSpec`.
