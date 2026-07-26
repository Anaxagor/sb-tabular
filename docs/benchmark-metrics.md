# Final benchmark metrics

Status: normative v1 definition for implementation and maintainer review.

This document makes the mathematical conventions behind final benchmark
numbers explicit. The formulas are model-independent and operate only on
decoded raw tables produced by the shared benchmark runner.

## Evaluation boundary

For every final cross-validation fold, evaluation receives:

- `real_train`: the post-missing-policy raw training partition;
- `real_test`: the post-missing-policy raw held-out partition;
- `synthetic`: a decoded sample with `len(real_train)` rows.

Marginal and association metrics compare `real_test` with `synthetic`. Unequal
row counts are expected and valid for distributional metrics. Utility compares
a predictor trained on `real_train` with the same predictor trained on
`synthetic`; both are evaluated on the same `real_test`.

All three tables contain every modeled column in canonical order, including
the target, and exclude the identifier. The target participates in marginal
and association metrics according to its declared `ColumnKind`. Utility alone
temporarily separates the declared target from feature columns.

Evaluation never sees prepared model values and never reuses the generator's
codec. CatBoost receives raw feature tables and owns its downstream predictive
handling.

## Marginal metrics

All divergences use natural logarithms. For a real probability vector `p` and
synthetic probability vector `q`, the reported direction is:

```text
KL(real || synthetic) = sum_i p_i * log(p_i / q_i)
```

Before normalization, a fixed pseudocount `1e-12` is added to every real and
synthetic count. This makes the empirical estimate finite when a state or bin
is absent from one sample. It is a benchmark constant, not a model or run
parameter.

### Continuous columns

For each continuous column:

1. Report the one-dimensional Wasserstein distance in decoded raw units.
2. Form 50 equal-width histogram bins from the minimum and maximum over the
   union of real and synthetic values.
3. Report histogram `KL(real || synthetic)` using the shared bins.

When both samples have the same single constant value, histogram KL is `0`.
When their constant values differ, the union has non-zero width and the normal
50-bin calculation applies.

The fold-level continuous marginal values are arithmetic means across all
continuous modeled columns. Per-column values are retained as review evidence.

### Discrete and categorical columns

For each discrete or categorical column, form one support from the union of
exact decoded real and synthetic values. Do not round, coerce, ordinally space,
or reuse codec state codes. Report empirical `KL(real || synthetic)` over that
support.

Discrete and categorical KL values are averaged separately. Per-column values
are retained.

## Association metrics

Association metrics compare relationships within a table. They must not call
an association score directly between real and synthetic column vectors:
rows in the two tables are unrelated and may have different lengths.

For each semantic group, construct a square association matrix for `real_test`
and another for `synthetic`, set their diagonals to zero, and report the
unnormalized Frobenius norm of their difference:

```text
distance(A_real, A_synthetic) = ||A_real - A_synthetic||_F
```

The group-specific association is:

- continuous columns: Pearson correlation;
- discrete columns: Spearman rank correlation;
- categorical columns: pairwise normalized mutual information using
  arithmetic normalization.

An undefined Pearson or Spearman value caused by a constant column is
represented as zero in both association matrices. Marginal metrics still
measure the distributional mismatch of that column. A group with fewer than
two columns has no pairwise association metric and returns `None`, not a
fabricated zero.

## TSTR utility

Utility is evaluated only when `TabularDataset` declares both `target` and
`task`.

For each fold:

1. Train one CatBoost predictor on feature columns from `real_train`.
2. Train a second predictor on the same feature columns from `synthetic`.
3. Evaluate both predictors on feature columns and target from `real_test`.

Both predictors use the same fold seed. CatBoost model hyperparameters remain
at library defaults; only reproducibility and side-effect controls are set:
`random_seed`, disabled verbose output, disabled file writing, and one worker
thread. The classifier additionally enables `allow_const_label`: a synthetic
sample containing one target class remains a valid but poor generator output
and therefore produces a constant downstream prediction instead of aborting
the benchmark. Categorical feature names come from categorical `ColumnSpec`
entries other than the target. Discrete columns remain numeric features.

Classification reports macro-F1 with `zero_division=0`. Regression reports
R². The artifact retains:

```text
real_score
synthetic_score
absolute_change = synthetic_score - real_score
relative_change_percent =
    100 * absolute_change / abs(real_score)
```

A negative change means worse synthetic utility. When `real_score == 0`, the
relative change is mathematically undefined and is stored as `None`; the two
scores and absolute change remain available. No epsilon is added to manufacture
a percentage. If any fold has an undefined relative change, the cross-fold
relative-change summary is also `None`; folds with defined values are not
silently averaged as a smaller subset.

The experiment prose also mentions regression MAPE, while the mixed-data
metric table selects delta R²/F1. V1 follows the table. Adding MAPE requires a
separate convention for zero and near-zero targets.

## Cross-fold aggregation

For every applicable scalar metric, report the arithmetic mean and population
standard deviation over all completed folds (`ddof=0`). Preserve the complete
per-fold and per-column values alongside the summary.

An inapplicable semantic group is `None` for every fold and in the summary.
Evaluation fails on a non-finite applicable metric instead of silently
dropping a fold.

## Deliberate corrections to legacy scripts

- Metrics use decoded raw tables, never one model's prepared representation.
- Column groups and target semantics come only from `TabularDataset`; there
  are no dataset-name target maps.
- Every model receives the same real folds and metric formulas.
- There is no fallback from CatBoost to a different estimator.
- Real and synthetic utility predictors use the same seed.
- Correlation diagonals do not turn constant-column `NaN` values into an
  artificial variance penalty.
- NMI compares within-table association matrices, not unrelated row vectors.
