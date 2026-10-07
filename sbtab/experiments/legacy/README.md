# `sbtab/experiments/legacy` — frozen historical scripts and result artifacts

## Imported `feature/tuning` artifacts

The merge imports `csbm_tuning.py`, `msbm_tuning.py`, 5 CSBM result JSON files and
28 MSBM result JSON files from `0b9f15f`. The JSON files are byte-identical to that
commit. Their historical protocol uses seed 5, a default 80/20 holdout and 60 trials;
the older seed-42/50-trial description below applies to the original scripts only.
The missing `Metrics` facade is restored verbatim from
`45e77a5:sbtab/evaluation/metrics.py` inside `legacy_metrics.py`, with explicit
provenance; this does not establish how the archived numbers were produced.

Both new scripts import and answer `--help`. Neither has been run end-to-end.
The CSBM script retains its historical solver API and its invalid epoch expression
(`inner_iters * batch_size / train_cat_t`); it cannot train against the refactored
CSBM API without a separate port. MSBM uses the imported solver, but historical
metric calls and dataset preprocessing have not been validated end-to-end.
Use the canonical stages for new experiments.

**Status: LEGACY. Metric definitions: `legacy/0`. Not the canonical protocol.**

Everything in this directory is superseded by the canonical stages

```
python -m sbtab.experiments.{prepare_splits,tune,cross_validate,calculate_metrics,aggregate_results}
```

and by the single canonical metric package `sbtab.evaluation` (metric version
`sbtab.metrics/1`). The files here are kept **only so that historical numbers remain
interpretable**. Do not extend them, do not port them, and never put a number produced
here in the same table, plot or ranking as a `sbtab.metrics/1` number.

How the legacy protocol differs from the canonical one: tuning on a single 80/20
holdout with seed 42 and 50 Optuna trials; evaluation with `KFold` on 100 % of the rows
(the tuning holdout is inside the evaluation folds); only `len(test fold)` synthetic rows
are sampled; metrics are the `legacy/0` definitions documented, defect by defect, in the
module docstring of [`legacy_metrics.py`](legacy_metrics.py).

Legacy code imports nothing from `sbtab.evaluation` or from the canonical stages, and
the canonical code must not import anything from here
(`tests/experiments/test_legacy_namespace.py` enforces the first half).

## Layout

| Path | What it is |
|---|---|
| `legacy_metrics.py` | The ONE copy of every helper that used to be pasted into the scripts (see below). `LEGACY_METRIC_VERSION = "legacy/0"`. |
| `calculating_metrics/*_metrics.py` | 7 per-model 5-fold evaluation scripts (ctgan, dsb, dsbm, light_sb, stasy, tabddpm, tapfn). |
| `{joint,structural}_{continuous,discrete}_metrics.py` | 4 evaluation scripts for the CatBoost ("boosted") IPF-DSB solvers. |
| `tuning_script/*.py` | 6 Optuna tuning scripts. |
| `calculating_metrics/{dsbm,tabpfgen}_kfold_eval/`, `tuning_script/dsbm_optuna_results/`, `tuning_results/best_params/`, `visualization/` | Tracked historical result files. **Never edit or delete** — see PROVENANCE. |

## What was consolidated

Before consolidation 102 helper definitions were pasted across the 17 scripts. They now
live once in `legacy_metrics.py`. Where historical copies differed **in behaviour** each
behaviour kept its own name and each script imports (under the old local name) the variant
it historically used. Evidence for every split is an AST diff of the historical copies;
equality of every consolidated helper with the historical copy it replaced is asserted by
`tests/experiments/test_legacy_namespace.py` against git commit `6e2d887`.

| Historical name (copies) | Now | Why more than one name |
|---|---|---|
| `avg_wd` (11), `average_wd` (3) | `avg_wd` (= `average_wd`), `avg_wd_autocols` | ctgan/tabddpm copies take `cols=None` → auto-select numeric columns and **raise** on an empty selection; the other 9 (+3) require `cols` and return **NaN** on an empty list. |
| `avg_kl_hist` (11) | `avg_kl_hist`, `avg_kl_hist_autocols` | same split as above; the other 4 "variants" differed only in docstring / line layout. |
| `corr_frobenius` (11) | `corr_frobenius_fillna0` (5), `corr_frobenius_raw` (4), `corr_frobenius_fillna0_autocols` (2) | `raw` has **no `fillna`**: one constant column makes the value NaN. Used by dsbm, light_sb, stasy, tapfn. |
| `make_regressor` (11) | `make_regressor_broad_fallback` (7), `make_regressor_importerror_fallback` (4) | `except Exception` vs `except ImportError` around the CatBoost import+constructor. |
| `utility_delta_r2_percent` (11) | `..._raw_target` (5), `..._raw_target_importerror_fallback` (4), `..._numeric_target` (2) | ctgan/tabddpm coerce the target with `pd.to_numeric(errors="raise")`; the 4 boosted scripts use the ImportError-only regressor fallback. |
| `average_wd_processed` (2) | `average_wd_processed`, `average_wd_processed_echo_columns` | the tabddpm_tuning copy has no `exclude_cols` and a left-over `print(c)` per column per trial (kept). |
| `resolve_target_col` (5) | `resolve_target_col_whitespace_tolerant(df, ds_name)` (2), `resolve_target_col_last_column_fallback(ds_name, df, strict)` (3) | different signatures **and** semantics: strip-tolerant match + `KeyError` for unmapped datasets, versus exact match + silent fall-back to the **last column** for unmapped datasets. |
| `build_transforms` (5) | `build_transforms` (3), `build_transforms_tabddpm_tuning` (1), `build_transforms_mixed` (1) | `_tabddpm_tuning` differs only in the text of one `ValueError`; `_mixed` takes `cat_encoding`. |
| `export_trials_csv` (6) | `export_trials_csv` (5), `export_trials_csv_composite` (1) | the composite variant adds two user-attribute columns. |
| `_common_numeric_cols` (3) | `common_numeric_cols` | one behaviour (the tabddpm copy is the `exclude_cols=None` case). |
| `load_best_params` (8) | `load_best_params` | identical. |
| `TARGET_COL_BY_DATASET` (15) | `TARGET_COL_BY_DATASET` + `TAPFN_DEFAULT_DATASET_ORDER` | same 9 pairs everywhere, but `tapfn_metrics` had a different key order and every script derives its `--datasets` default from `.keys()`, so that order is observable and was preserved. |
| `sliced_wasserstein` (import of a non-existent module, 4 scripts) | `sliced_wasserstein` | restored byte-for-byte from `de5acc9:sbtab/evaluation/metrics/statistical.py`. |

Not consolidated on purpose: `DEFAULT_DATASETS` (each tuner's own CLI default) and the
WD+JS composite objective in `tuning_script/tabddpm_mixed_data_tuning.py`
(`compute_composite_metric` and its `_js_*` helpers exist only there and were never duplicated).

Other edits made to the scripts, and nothing else (checked by an AST diff of every kept
function against `6e2d887`): a LEGACY banner docstring; `print_legacy_warning(...)` as the
first statement of `main()` (one line on stderr, also shown with `--help`); the stale
import `sbtab.evaluation.metrics.statistical` replaced; four `--best_json_dir` defaults
re-pointed from `sbtab/experiments/tuning_script/...` to `sbtab/experiments/legacy/tuning_script/...`;
the two developer-machine Windows `--pickle` defaults (`dsbm_tuning.py`, `tabddpm_tuning.py`)
replaced by `sbtab/data/datasets/datasets_continuous_only.pkl`.

## STATUS — what actually works on this branch

**NONE of these scripts has been run end-to-end on this branch.** The solver and baseline
classes they drive changed API on this branch (DSBM `noise` handling and default coupling,
the IPF solvers, the TabDDPM training budget, STaSy, `CategoricalReference`, ...). The
scripts were deliberately **not** ported. The work done here is import-level hygiene and
de-duplication only. "Imports cleanly and `--help` works" means exactly that and nothing
more: it says nothing about whether training, sampling or the numbers are right.

Verified on 2026-09-20 in the working tree of `refactor/sbtab-8515-spec-v1`
(HEAD `6e2d887` plus uncommitted work of several people — re-verify before relying on it)
with `python -c "import importlib; importlib.import_module('<module>')"` and
`python -m <module> --help`, interpreter `/opt/anaconda3/envs/synth/bin/python`:

| Module (`sbtab.experiments.legacy.`…) | Import | `--help` | Known dead code reached at run time (static check against the current signatures; not executed) |
|---|---|---|---|
| `calculating_metrics.ctgan_metrics` | ok | ok | none found statically |
| `calculating_metrics.dsb_metrics` | ok | ok | reads best-params key `"N"` but the tuner writes `"num_steps"` → the tuned value is silently ignored and 48 is used; the tuned `cache_batches` and `steps_per_phase_multiplier` are never read either |
| `calculating_metrics.dsbm_metrics` | ok | ok | none found statically |
| `calculating_metrics.light_sb_metrics` | ok | ok | **(1)** + **(2)** in the first fold |
| `calculating_metrics.stasy_metrics` | ok | ok | **(1)** + **(2)** in the first fold |
| `calculating_metrics.tabddpm_metrics` | ok | ok | none found statically |
| `calculating_metrics.tapfn_metrics` | ok | ok | none found statically (see the un-seeded sub-sampling note below) |
| `joint_continuous_metrics` | ok | ok | **(1)** + **(2)** in the first fold |
| `joint_discrete_metrics` | ok | ok | **(1)** + **(2)** in the first fold |
| `structural_continuous_metrics` | ok | ok | **(1)** + **(2)** in the first fold |
| `structural_discrete_metrics` | ok | ok | **(1)** + **(2)** in the first fold |
| `tuning_script.ctgan_tuning` | ok | ok | none found statically |
| `tuning_script.dsbm_tuning` | ok | ok | none found statically |
| `tuning_script.ipf_dsb_tuning` | ok | ok | none found statically |
| `tuning_script.lightsb_optuna_tune` | ok | ok | none found statically |
| `tuning_script.tabddpm_tuning` | ok | ok | none found statically |
| `tuning_script.tabddpm_mixed_data_tuning` | **FAILS** | **FAILS** | import error: `ModuleNotFoundError: No module named 'ucimlrepo'` (third-party package absent from the environment; not installed on purpose) |
| `legacy_metrics` | ok | n/a | — |

Summary: 16 of 17 scripts import and answer `--help`; 1 fails at import; 0 of 17 have been run.

Pre-existing dead calls (present before the move, reported, **not** ported):

* **(1)** `TabularSchema(feature_cols=cols)` →
  `TypeError: TabularSchema.__init__() got an unexpected keyword argument 'feature_cols'`
* **(2)** `TransformPipeline.default_continuous_dropna()` →
  `AttributeError: type object 'TransformPipeline' has no attribute 'default_continuous_dropna'`

  Six scripts contain both, inside the fold loop, so they cannot complete a single fold.
* `dsb_metrics.build_dsb_config_from_best` reads `"N"`; `ipf_dsb_tuning` writes `"num_steps"`.
* The four `joint_*`/`structural_*` scripts imported `sliced_wasserstein` from
  `sbtab.evaluation.metrics.statistical`. That module **never existed in the ancestry of
  this branch**: the four commits that touch it are reachable only from unmerged remote
  branches (`2f3325b`, `0190138` from `origin/feat/igor`; `de5acc9`, `17ee127` from
  `origin/feat/evaluation-metrics`). As committed at `de5acc9` it could not even be imported
  (`NameError: name 'pd' is not defined`); `sliced_wasserstein` itself is identical in
  `0190138`, `de5acc9` and `17ee127`. So these four scripts cannot have produced any number
  from a commit of this branch.
* "None found statically" means only that every keyword passed to an `sbtab` constructor or
  method still exists by name. Semantics were not checked, and several are known to have changed.

Further facts about the scripts that matter when reading old numbers:

* `tapfn_metrics` sub-samples every fold larger than 8000 rows to 5000 rows with an
  **un-seeded** `DataFrame.sample`, and its `reset_index` (without `drop=True`) adds an
  `index` column to the frame. The tracked TabPFGen numbers for the large datasets are
  therefore not reproducible even in principle.
* `sliced_wasserstein` draws its projections from the global, un-seeded torch RNG.
* No tracked tuner exists for the four boosted solvers, for STaSy or for TabPFGen;
  `light_sb_metrics` takes its hyper-parameters from CLI flags and never reads the output
  of `lightsb_optuna_tune`.

## PROVENANCE of the tracked result files

These files are byte-identical to what was committed (asserted by the test against
`6e2d887`). They were not edited, re-generated or re-formatted.

| Directory | Holds |
|---|---|
| `calculating_metrics/dsbm_kfold_eval/` | 9 `<dataset>_kfold_summary.json`, 9 `<dataset>_fold_metrics.csv`, 1 `kfold_summary_all_datasets.csv` — output format of `dsbm_metrics.py` (continuous-time MLP IMF-DSBM, 9 datasets × 5 folds, `seed=42`, `device=cuda`). |
| `calculating_metrics/tabpfgen_kfold_eval/` | same 9 + 9 + 1 layout — output format of `tapfn_metrics.py` (TabPFGen). |
| `tuning_script/dsbm_optuna_results/` | 9 `<dataset>_best.json` + `dsbm_optuna_summary.csv` — output format of `dsbm_tuning.py` (`n_trials=50` each). |
| `tuning_results/best_params/` | 6 JSON files (`ctgan`, `dsb`, `dsbm`, `lightsbm`, `tabbyflow`, `tabddpm`), each keyed by a **different, 14-dataset suite** (`Adult`, `Credit Approval`, `Eucalyptus`, …). Added by commit `1698b17` (2026-09-11). |
| `visualization/` | `SB results - Лист1 (1).csv` (a hand-typed spreadsheet export, 63 rows = 9 datasets × 7 models, cells are strings such as `0.044+-0.007`), two PNG radar charts, and the one-cell notebook that draws them from that CSV. |

Verified facts (each re-checked for this document; the counts are asserted by the test):

1. **No file records where it came from.** A case-insensitive search for
   `commit|sha|version|timestamp|hostname` over every tracked JSON/CSV returns 5 hits, all
   of them the substring `sha` inside the column name `" shares"`. An enumeration of every
   JSON key path confirms it: there is no git commit, code version, metric version,
   timestamp, host name, OS, Python or package version anywhere. The only time-like field is
   `elapsed_sec`.
2. **The 9 DSBM summaries embed a developer-machine Windows absolute path** in
   `best_params_path` (drive `C:`, profile `Anaxagor`). It points at
   `...\sbtab\experiments\dsbm_optuna_results\<dataset>_best.json`, a directory that was never
   tracked (the tracked copies lived in `experiments/tuning_script/dsbm_optuna_results/`).
   The `best_params` embedded in the summaries are equal to the tracked
   `dsbm_optuna_results/*_best.json` in 9 of 9 cases. These are the only files in this
   directory that contain such a path.
3. **`tuning_results/best_params/*.json` were not produced by any tuning code in this
   repository.** No file in the repository reads them. They are keyed by a 14-dataset suite
   that no legacy script knows (the legacy scripts know 9 pickle keys), and they contain
   parameters outside every tracked search space:
   * `dsbm`: `imf_len = 9` in 5 of 14 entries (tracked space: 3, 5, 7) and a tuned
     `grad_clip` in 14 of 14 (tracked code fixes `grad_clip = 1.0`).
   * `dsb`: keys `schedule`, `noise`, `grad_clip`, `steps_per_phase` never suggested by
     tracked code; `ipf_iters` outside {1, 2, 4, 8} in 5 of 13; `num_steps` outside
     {16, 24, 32, 48} in 9 of 13; `cache_batches` ∈ {32, 64, 128, 200} (tracked: 1, 4, 8);
     one entry (`Stroke Prediction`) is `null`.
   * `ctgan`: `batch_pac`, `generator_arch`, `discriminator_arch`, `weight_decay`,
     `discriminator_steps`, `log_frequency`; none of the tracked `batch_size`,
     `gen_disc_width`, `pac`.
   * `tabddpm`: `architecture`, `weight_decay`, `dropout`, `scheduler`; `n_epochs` ∈
     {100, 200, 300} (tracked: 5000, 10000, 20000); `num_timesteps` includes 250 and 500
     (tracked: 100, 1000).
4. **`tabbyflow_best_params.json` is an orphan**: there is no TabbyFlow implementation,
   wrapper or import anywhere in the tree.
5. **`lightsbm_best_params.json` holds LightSB-style parameters** (`n_potentials`,
   `epsilon`, `S_diagonal_init`, `sampling_batch_size`, `max_iter`, `init_r_from_data`),
   not LightSB-M ones, uses none of the key names the tracked LightSB tuner writes
   (`potential_n_potentials`, …), and is read by no code.
6. **`noise = false` was selected in 3 of 9 `dsbm_optuna_results`**
   (`covertype`, `german_credit`, `online_news_popularity`) **and in 5 of 14
   `dsbm_best_params` entries** (`Adult`, `Auto MPG`, `Churn Modelling`, `Diamonds`,
   `House Sales`). `first_coupling = "ind"` was selected in 9 of 9 and 13 of 14.
7. **The spreadsheet can only be checked for 18 of its 63 rows.** Its DSBM and TabPFN rows
   agree with the tracked summaries (to the printed precision) in 9 of 9 datasets each. The
   other 45 rows (CTGAN, TabDDPM, Stasy, DSB, LightSB) have **no tracked fold or summary
   file at all**. Its utility column is headed `% R2_real - R2_synth` but contains no
   negative cell (0 of 63): the sign was discarded. In 5 of the 9 TabPFN rows the tracked
   `delta_r2_percent` is **positive** (the synthetic-data model scored higher than the
   real-data model, e.g. `german_credit` +77.5), which the sheet shows exactly like a loss.
8. Circumstantial, from an **untracked, git-ignored** file that will not survive a fresh
   clone: `tuning_script/__pycache__/forest_diffusion_tuning.cpython-311.pyc`, compiled on
   2026-09-09 from `sbtab/experiments/tuning_script/forest_diffusion_tuning.py`. That source
   file was never committed. It shows that tuning code outside the repository existed on the
   developer machine two days before `1698b17`.

**Conclusion.** The provenance of every historical number in this directory is
**UNRESOLVED**. For no file can the producing commit, solver version, metric definition
(e.g. which `--n-bins-kl`, which `corr_frobenius` variant, CatBoost or the silent
fall-back regressor), environment or random state be established from the repository, and
part of the record demonstrably came from code that is not in it. Consequently **none of
these numbers can be attributed to a specific defect** (for example "DSBM was bad because
`noise` was mishandled"): 6 of 9 tracked DSBM tunings selected `noise = true` and 3 selected
`noise = false`, and nothing ties either group to a known solver revision. Treat the files
as a record of what was once reported, not as evidence about the current code.
