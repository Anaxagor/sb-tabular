# TabDDPM benchmark on fourteen mixed datasets

This entrypoint runs one reproducible experiment collection for the fourteen
datasets declared in [`benchmark-datasets.md`](benchmark-datasets.md). Each
dataset is tuned independently. Hyperparameters selected for one dataset are
never reused for another dataset.

## Protocol

For each dataset the command performs these stages in order:

1. apply `COMPLETE_CASE` once across all modeled columns;
2. run an 80/20 tuning holdout, stratified for classification;
3. run a persistent Optuna TPE study to 30 successful 10,000-step trials;
4. rerun the best three distinct configurations at 30,000 steps with two seed
   pairs and freeze the configuration with the lowest mean tuning score;
5. generate synthetic train-sized data independently in five folds, using
   stratified folds for classification;
6. evaluate every decoded synthetic fold and aggregate population mean and
   standard deviation.

The tuning score follows the experiment specification: train-standardized
mean Wasserstein distance for continuous columns plus exact-support mean
Jensen–Shannon distance for finite-state discrete and categorical columns.
Final report metrics do not influence Optuna selection.

The shared fold-local codec applies population `StandardScaler` semantics to
continuous columns. Categorical values use reversible train-observed state
codes and enter multinomial diffusion. Numeric discrete values remain raw and
join continuous values in Gaussian diffusion. The adapter applies `np.rint`
only to generated discrete values (ties to even), without clipping or
projection onto train support. All metrics and TSTR receive the same rounded
table. Fractional discrete training values are unsupported by this convention;
all 33 declared discrete columns in the current 14-dataset bundle are integral.

## Full command

Run from the repository root in the `lightning11` environment. The checked-in
pickle is the exact published collection; its legacy `DataFrame.attrs` are
ignored and every frame is reconstructed through the explicit new schemas.

```bash
conda activate lightning11
python -m sbtab.benchmark.pilots.tabddpm_mixed_benchmark \
  --output-dir artifacts/tabddpm-mixed-optuna-v3 \
  --dataset-pickle sbtab/data/datasets/datasets_mixed.pkl \
  --device mps \
  --target-complete-trials 30 \
  --max-total-trials 45
```

`--device cuda` is the corresponding accelerator choice on a CUDA machine.
Without `--dataset-pickle`, UCI/OpenML/Kaggle sources are fetched lazily; that
path additionally needs `ucimlrepo`, `kagglehub`, and any required Kaggle
authentication.

Before committing several days of compute, verify the setup on two datasets:

```bash
python -m sbtab.benchmark.pilots.tabddpm_mixed_benchmark \
  --output-dir artifacts/tabddpm-mixed-calibration \
  --dataset-pickle sbtab/data/datasets/datasets_mixed.pkl \
  --datasets credit_approval forest_fires \
  --device mps \
  --target-complete-trials 2 \
  --max-total-trials 3 \
  --rerank-candidates 1 \
  --rerank-seed-pairs 1
```

The calibration is a separate experiment and must not be resumed as the full
run because its immutable run specification and search budget differ.

## Progress and restart

The root `progress.json` lists completed and pending datasets. Each dataset
owns a separate `study.sqlite3`, so an interrupted Optuna phase can resume at
the next trial boundary:

```bash
python -m sbtab.benchmark.pilots.tabddpm_mixed_benchmark \
  --output-dir artifacts/tabddpm-mixed-optuna-v3 \
  --dataset-pickle sbtab/data/datasets/datasets_mixed.pkl \
  --device mps \
  --target-complete-trials 30 \
  --max-total-trials 60 \
  --resume
```

The native model has no mid-fit checkpoint, so interruption restarts only the
current fit. Completed rerank seed runs and completed datasets are reused.
The successful-trial target is immutable. ``max_total_trials`` is only an
invocation safety ceiling and may be increased after inspecting numerical
failures; changing it does not change the study fingerprint or result meaning.

If native training or sampling detects a non-finite trajectory during Phase A,
only that configuration is recorded as a pruned Optuna trial. Pruned trials do
not count toward the successful target, but TPE uses them as unfavorable
observations and therefore learns to avoid the unstable region. Native
training checks loss about 100 times per fit, aborting a diverged 10,000-step
trial within one percent of its budget without changing finite-loss optimizer
updates. Contract and unexpected model errors still stop the study.

This is tuning protocol v3. Both v1 and v2 sent discrete columns to multinomial
diffusion. Keep their results as evidence for that experimental variant;
start a new v3 output root and study. Resume rejects old run specifications,
studies, and rerank fingerprints. Tuning and final training must be repeated
because routing changes the network dimensions and loss assignment.
Five-fold final generation is currently create-only; interruption during that
stage restarts the current dataset's final generation. An interruption while
artifact files themselves are being finalized is reported explicitly rather
than silently overwriting partial evidence.

## Results

The collection root contains:

- `run-spec.json`: immutable dataset order, source digest, seeds,
  preprocessing, successful-trial target, and tuning protocol;
- `progress.json`: currently completed and pending datasets;
- `<dataset>/study.sqlite3`: resumable Optuna Phase-A state;
- `<dataset>/tuning/`: trials, rerank evidence, and frozen native config;
- `<dataset>/generation/`: real tables, decoded synthetic folds, and timings;
- `<dataset>/evaluation/`: per-column/per-fold metrics and aggregate metrics;
- `summary.json`: complete structured summaries for all datasets;
- `summary.csv`: one flat machine-readable row per dataset;
- `summary.md`: a review-friendly table;
- `manifest.json`: final relative paths and SHA-256 digests.

The summaries include continuous 50-bin mean KL, raw and standardized mean WD,
Pearson correlation distance and MMD; discrete exact-support mean KL and
Spearman distance; categorical exact-support mean KL and NMI distance; exact
state JS; CatBoost TSTR real/synthetic scores and `% real - synthetic` F1 or
R2 gaps; and native fit/sample time. Inapplicable metric groups are `null`
rather than zero.

## Expected duration

The measured Online Shoppers MPS pilot needed about 3.4 minutes for one small
10,000-step fit and sample. Thirty Phase-A trials alone therefore have a hard
lower bound near 1 hour 45 minutes per dataset; wider networks, the high-budget
rerank, five final folds, larger datasets, and 1,000 diffusion timesteps make
the real total substantially longer. On the current MPS evidence, budget
roughly four to seven uninterrupted days for all fourteen datasets. Use the
two-dataset calibration above to obtain a machine-specific estimate before the
full run.
