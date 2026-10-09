# SB-tabular — implementation report

**Historical MSBM scope:** this report describes the two-network, semigroup-reference
MSBM before the merge of `feature/tuning` (`0b9f15f`). The active MSBM now follows that
branch's shared-network, per-step `alpha` implementation, with current adapter,
checkpoint and reproducibility support. MSBM-specific correctness claims below
do not describe the imported historical categorical kernel. CSBM and the common
bridge primitives retain the refactored implementation. See the repository README.

Follow-up: [review and corrections, 2026-09-23](REVIEW_2026-09-23.md). That report records the latest
787-pass test run, additional defects fixed after this report, and remaining specification gaps.
The execution counts and artifact descriptions below describe the original refactor validation.

Specification: *SB-tabular: correctness and experiment implementation specification* v1.0 (19 Sep 2026),
protocol `sbtab_8515_hpo100_cv5_v1`; **default protocol since 2026-09-20: `sbtab_8515_hpo100_cv5_v2`** (= v1 + a
dataset-eligibility rule requested by the repository owner, §4.1). Branch `refactor/sbtab-8515-spec-v1`.

**Checkout.** Work started from `main` at `6e2d887e852e9c74ed6341f5036d47c5f6d737c5`, which is exactly the
revision the specification reviewed; every cited defect was re-checked against that code before it was changed.
Nothing has been committed. Old scripts and tracked result files were moved (history preserved) into
`sbtab/experiments/legacy/`.

**What this report does and does not claim.** Contracts were checked by derivation, by independent-oracle
regression tests and by bounded end-to-end runs. It does **not** claim that "all algorithms are correct", and
**no production benchmark was run**: the only end-to-end runs (six, on five real datasets) used the separate
*smoke* protocol (3 trials, tiny budgets). Smoke numbers say nothing about model quality and are not evidence that the 100-trial benchmark was
completed. Section 6 separates what was executed from what was skipped or not exercised.

Two things a reader should know first:

1. **A local, unpushed branch stack overlaps this work.** `pr/benchmark-*` (author Mikhail-Galkin, Aug 2026,
   ~9.8k lines, `sbtab/benchmark/` + `sbtab/evaluation/`) implements a different protocol (80/20 holdout, raw-space
   tuning score, histogram edges from the compared samples, no MMD / conditional metrics). It was left untouched.
   Both lines of work populate `sbtab/evaluation/`; they will conflict and need a deliberate reconciliation.
2. **The tracked dataset bundles cannot be unpickled in the documented environment** (written with numpy 2 /
   pandas 3; `requirements.txt` allows numpy 1.26 / pandas 2.2). At `6e2d887` *none* of the repository's experiment
   scripts could load data here. `sbtab.data.loading.load_bundle` now falls back to a cross-version unpickler and
   `prepare_splits` re-materialises each dataset as Parquet + JSON schema with a value-based fingerprint.

---

## 1. Registry and tested support matrix

`sbtab/solvers/registry.py` (`solver_registry`) — 18 supported, 1 heuristic, 5 unavailable. "Tested" means the entry
passed `tests/experiments/test_stages.py::test_every_supported_entry_…`: fit → sample (n = 1, 64, 65) → checkpoint →
reload without refit → identical samples → validity + tuning objective, in **every** compatible regime, and refusal
of every incompatible regime.

| id | family | time parameterisation (as implemented) | structure | backend | native / adapted regimes | tested |
|---|---|---|---|---|---|---|
| `dsb_ct_joint_mlp` | IPF-DSB | time-conditioned networks on a γ grid | joint | torch | continuous / discrete, mixed | ✅ |
| `dsb_dt_joint_mlp` | IPF-DSB | **per-step** (one MLP per edge and direction) — was a copy of the CT solver | joint | torch | continuous / discrete, mixed | ✅ |
| `dsb_ct_joint_gbt` | IPF-DSB | one time-conditioned regressor per direction | joint | CatBoost | continuous / discrete, mixed | ✅ |
| `dsb_dt_joint_gbt` | IPF-DSB | one model per edge and direction | joint | CatBoost | continuous / discrete, mixed | ✅ |
| `dsb_ct_structural_gbt` | IPF-DSB | time-conditioned scalar regressor per column | DAG learned on fit rows | CatBoost + pgmpy | continuous / discrete, mixed | ✅ |
| `dsb_dt_structural_gbt` | IPF-DSB | per column, edge and direction | DAG learned on fit rows | CatBoost + pgmpy | continuous / discrete, mixed | ✅ |
| `dsbm_ct_joint_mlp` | IMF-DSBM | time-conditioned drift, t ~ U[ε, 1−ε] | joint | torch | continuous / discrete, mixed | ✅ |
| `dsbm_ct_joint_gbt` | IMF-DSBM | one time-conditioned regressor per direction | joint | CatBoost | continuous / discrete, mixed | ✅ |
| `dsbm_dt_joint_mlp` | IMF-DSBM | per edge, at the **state** time | joint | torch | continuous / discrete, mixed | ✅ |
| `dsbm_dt_joint_gbt` | IMF-DSBM | per edge, at the **state** time | joint | CatBoost | continuous / discrete, mixed | ✅ |
| `dsbm_dt_structural_gbt` | IMF-DSBM | per column and edge | autoregressive chain (default); optional map / learned DAG | CatBoost | continuous / discrete, mixed | ✅ |
| `lightsb` | LightSB | static potential, exact conditional sampler | joint | torch GMM | continuous / discrete, mixed | ✅ (diagonal) |
| `csbm` | CSBM / D-IMF | time-conditioned, uniform unit grid | joint input, factorised head | torch | discrete / – | ✅ |
| `csbm_annealed` | CSBM (**heuristic**) | as `csbm`; reference annealed | as `csbm` | torch | discrete / – | ✅ |
| `mixedsbm` | MixedSBM | time-conditioned, one uniform unit grid | joint | torch | continuous, discrete, mixed / – | ✅ |
| `tabddpm` | diffusion baseline | diffusion timesteps | joint row (X, y) | torch | all / – | ✅ |
| `ve_score_sde_simplified` | score-SDE baseline | VE SDE, predictor–corrector | joint | torch | continuous / discrete, mixed | ✅ |
| `ctgan` | GAN baseline | – | joint | sdv | all / – | ⛔ **not validated** (sdv missing) |
| `tabpfgen` | pretrained prior | – | SGLD + TabPFN | tabpfgen, tabpfn | all / – | ⛔ **not validated** (libs missing) |

"Adapted" regimes go through a declared, serialised representation (`sbtab/adapters/representation.py`): nominal
columns one-hot (argmax decoding), discrete columns standardised and decoded to the **nearest training-support value**
with the pre-decoding invalidity rate and rounding distance recorded. No continuous output is ever clipped.

**Unavailable — named elsewhere, not implemented; never substituted** (`get_adapter_class` raises):

| id | finding | missing work |
|---|---|---|
| `stasy` | `sbtab/baselines/stasy` is a simplified VE score-SDE, now registered honestly as `ve_score_sde_simplified` | per-sample SPL weights (α₀/β₀ thresholds), fine-tuning stage, VP/sub-VP SDEs, probability-flow ODE sampler, ncsnpp-tabular network |
| `lightsb_m` | a class alias `LightSBM = LightSBPotential` and `lightsbm_best_params.json` existed; the code is LightSB | the bridge-matching objective of Gushchin et al. with a differentiable drift |
| `tabbyflow` | **orphaned configuration**: `tabbyflow_best_params.json`, no code or import anywhere | the whole wrapper |
| `tabsyn` | no code in the tracked tree | the whole wrapper |
| `forestdiffusion` | not on this branch (only a stale `__pycache__`); an implementation exists on branch `forest_diffusion`, **not audited here** | audit + adapter |

Other orphans / stale imports found: four top-level metric scripts imported `sliced_wasserstein` from the non-existent
`sbtab.evaluation.metrics.statistical`; `examples/California_Housing_example.py` imported the non-existent
`sbtab.solvers.ipf_dsb.solver`; 11 files called the non-existent `TransformPipeline.default_continuous_dropna()` and
`TabularSchema(feature_cols=…)`; `sbtab/solvers/base.py`, `sbtab/models/base.py` were empty; `get_datasets.py` imports
`kagglehub`, which `requirements.txt` did not list.

---

## 2. Confirmed defects, corrections and regression tests

Every row was confirmed against `6e2d887` before the fix: **by running the old code wherever a number is quoted**,
otherwise by reading it (this applies to several baseline items — ignored `gaussian_loss_type`, copied ids, CTGAN
metadata/seeding, TabPFGen defaults). One exception: TabPFGen's *upstream* behaviour (B-6: label handling, output size
10·⌊n/10⌋) could not be executed because the library is not installed; it was taken from the upstream source and the
wrapper-side fix is tested with a fake generator only. *Before → after* is the recorded behaviour. Test files are
under `tests/`.

### 2.1 Shared bridge primitives, MixedSBM, CSBM

| # | defect (component) | before → after | regression test |
|---|---|---|---|
| P0-1 | **Mixed time grid**: training used `n/K` on a unit horizon, the sampler integrated the geometric γ grid and fed the network those times (`bridge/timegrid.py`, `pathsampler.py`, `msbm/*`) | horizon **0.217914** at N = 100 (0.110 / 0.433 at N = 50 / 200), network saw t ∈ [1e-4, 0.218] → one explicit grid `t[0]=0<…<t[N]=T`; MixedSBM asserts T = 1; sampler integrates exactly T | `bridge/test_timegrid_and_reference.py`, `bridge/test_losses_and_clocks.py::test_network_continuous_step_and_categorical_index_share_one_clock` |
| P0-2 | **Categorical reference**: ordered kernel used true powers for k < 30 and separately normalised Gaussian matrices from k = 30 (`bridge/reference.py`) | `max|P[29]·Q₁ − P[30]|` = **0.384**; "bridge probabilities" summed to **0.012 … 1.24** → semigroup `Q(s)=exp(sR)`: matches `scipy.linalg.expm` to 2.7e-15, composes on uniform / non-uniform grids, bridges sum to 1 incl. the rarest pair 0→S−1 | `…::test_transitions_match_independent_matrix_exponential`, `…across_old_k29_30_boundary`, `…rare_pairs`, `…brute_force_conditionals` |
| P0-2b | reference law depended on the step count | P(stay over horizon) 0.633 / 0.446 / 0.301 for K = 50 / 100 / 200 → identical to 1e-12 (rate per unit time) | `…::test_reference_law_does_not_depend_on_the_number_of_steps` |
| P0-2c | zero-probability bridges hidden: `+1e-12` then sampling ⇒ effectively uniform fallback (with the hard-coded α = 0.01 the ordered kernel was ≈ identity, so this path was the norm) | → no epsilon anywhere; `IncompatibleBridgeError`; malformed probabilities rejected; padded logits masked to −∞ | `…::test_unreachable_bridge_is_reported_not_smoothed`, `…never_selects_a_padded_category…` |
| P0-3 | **Loss normalisation**: categorical loss averaged over batch×columns, then divided by the column count again (`bridge/losses.py`) | duplicating a column halved the term → normalised once; both blocks are per-feature means (documented) | `bridge/test_losses_and_clocks.py::test_duplicating_a_categorical_column…`, `…equals_scalar_loop…` |
| P0-4 | **Absent blocks**: `max`/`stack` of empty lists, mean of an empty tensor → NaN | → branch before computing; pure regimes finite; loss parity with `CSBMLoss` / plain MSE | `…::test_absent_blocks_give_finite_losses…`, `solvers/test_mixedsbm_csbm.py::test_regimes_small_sample…` |
| P0-5 | **Small datasets**: `drop_last=True` ⇒ zero batches, stage silently trained nothing | N = 60 < 256: 0 updates → positive updates asserted per stage | same |
| P0-6 | **Orientation / clocks**: training drew n ∈ [1, K−1] for *both* directions, but forward sampling starts at n = 0 and backward at n = K (never-trained times); one network shared by both directions | → forward n ∈ [0, N−1], backward n ∈ [1, N]; one network per direction; alternation enforced; snapshot = last `b` | `…::test_noise_corrected_target_equals_endpoint_conditioned_target`, `…::test_imf_stages_use_the_previous_learned_coupling…` |
| P1-1 | **CSBM reference changed** during outer iterations (`csbm/solver.py`) | → fixed in canonical `csbm`; annealing only in `csbm_annealed` (own config, reload restores the *final* reference) | `…::test_csbm_reference_is_fixed…`, `…::test_annealing_exists_only_as_the_named_variant` |
| P1-2 | **CSBM had no sampling API**; shared sampler passed geometric times while the updater trained at n/K; size mismatch when \|p0\| < \|p1\| | → `CSBMSolver.sample(n, seed, batch_size)`: exact n, masked support, current-state clock | `…::test_csbm_updater_feeds_the_grid_time_of_the_current_state` |
| – | `GaussianReference.sample(seed=None)` built a fresh `torch.Generator` (fixed default seed) ⇒ **identical "random" draws on every unseeded call** | → global RNG / explicit generator | `bridge/…::test_gaussian_reference_unseeded_calls_differ` |

Preserved because it was already correct: MixedSBM's forward noise-corrected target and reverse-clock target; the
uniform kernel's closed form; the coupling anchoring logic.

### 2.2 IPF-DSB

| # | defect | before → after | test |
|---|---|---|---|
| D-1 | **DT boosted index bug** (joint + structural): B fitted at `min(k+1,K−1)`, F at `max(k−1,0)` | **`fit()` crashed**: `RuntimeError: Model for time step 0 is not trained` (B trained `[F,T,T,T]`) — this code could not have produced any result → edge k ≡ (X_k, X_{k+1}, γ_k) in both directions; all edges asserted | `solvers/test_boosted_ipf.py::test_every_edge_model_is_fitted_and_called_under_its_own_index` |
| D-2 | CT boosted: edge labelled `times[k±1]` (clamped) — one label duplicated, one never trained; CT joint evaluated the second mean at the *wrong* time | → label `times[k]` everywhere; both evaluations on the same edge | `…::test_continuous_time_labels_are_one_to_one…` |
| D-3 | DT joint OU init used cumulative time | contraction 0.9677 instead of 0.9930 at k = 19 (α = 0.7) → step interval γ_k | `…::test_ou_reference_uses_the_step_interval` |
| D-4 | reference was a CatBoost *fit* of x(1−αγ) whose RMSE (0.047) was ~13× the OU signal (0.0035) | → iteration 0 uses the **analytic** OU mean; mean-matching maps fitted as residuals (`residual=True`, enforced by the IPF solvers only) | `…::test_forward_cache_holds_actual_reference_trajectory_states`, `…::test_residual_parameterisation…` |
| D-5 | **MLP IPF caches**: `x` never advanced — every cache input was an endpoint + one step | data with mean (3, −3) → samples with mean **(0.00, 0.01)**, std 1.12: the solver ignored the data → full trajectories of the opposite process | `solvers/test_ipf_mlp_caches.py`, `test_ipf_mlp_learning.py::test_short_fit_moves_the_samples_onto_the_data` |
| D-6 | **units**: displacement target re-multiplied by γ in the sampler | → displacement added as is; derivation in §3 | `test_ipf_mlp_sampling.py::test_sampler_step_adds_the_displacement_exactly` |
| D-7 | iteration 0 simulated with an **untrained random network**; `alpha_ou` only rescaled γ | → declared OU/Brownian reference; `alpha_ou` is the mean-reversion rate | `…::test_iteration_zero_simulates_the_reference…` |
| D-8 | `noise=False` made the *caches* deterministic; same seed for start draw and path noise (first increment ≡ x_T) | → caches always stochastic; one generator | `…::test_caches_stay_stochastic…`, `…first_increment_is_not_a_replay…` |
| D-9 | `dim = 1` broadcast `(n,1)+(n,)→(n,n)`; `feature_mode` and `x0` accepted and ignored | → fixed / rejected | `test_boosted_ipf.py::test_ignored_options_are_rejected` |

### 2.3 IMF-DSBM, LightSB

Targets, time/noise consistency, coupling refresh and orientation were **correct and preserved**.

| # | defect | before → after | test |
|---|---|---|---|
| M-1 | DT solvers trained model k at the **midpoint** (k+½)/N but applied it at the step start with a full dt | exact drift to a point mass at 5 lands on 5 + 5/(2N−1): **5.263158** (N = 10) → state-time grids k/N and (k+1)/N: **5.000000** | `solvers/test_dsbm_math.py::test_dt_exact_drift_hits_point_mass` |
| M-2 | `noise=False` in every solver **and** in the fit-time coupling simulation **and** in the search space | not a probability-flow ODE (std 0.253 vs 0.284 data) → couplings always noisy; `noise=False` ⇒ `variant_id=*_noiseless_heuristic`; not an adapter option; rejected in search spaces | `test_dsbm_solvers.py::test_noise_flag…`, `experiments/test_stages.py` |
| M-3 | default `first_coupling="ref"`: t = 1 marginal is data∗N(0, σ²), not the N(0, I) the generator starts from | avg WD 0.48 vs 0.09 → default `"ind"`; `"ref"` kept as the declared IPF-like variant; unknown values raise | `…::test_default_first_coupling_is_independent` |
| M-4 | `sample(seed)` not reproducible (path noise from a stateful RNG); CT-MLP first increment ≡ start sample; `drop_last` crash (`StopIteration`) for N < batch; network fed t outside [ε, 1−ε]; hidden architecture; ignored `steps`; `x0` feature modes leaked the regression answer | → all fixed | `test_dsbm_solvers.py` (243 tests) |
| L-1 | LightSB seeded SDE path reused x₀ as the first Brownian increment; `transport(DataFrame)` overwrote training columns; global-RNG clobbering; `LightSBM` alias; `set_epsilon` silently rescaled mixture weights | → fixed / removed. **Objective, log C, log-potential (log-det kept), conditional sampler and drift were verified correct and left unchanged** | `solvers/test_lightsb_oracles.py`, `test_lightsb_solver.py` |

### 2.4 Baselines, data, protocol

| # | defect | before → after | test |
|---|---|---|---|
| B-1 | TabDDPM `steps=10000` default silently overrode a tuned `n_epochs` | every "tuned" run trained exactly 10 000 steps → exactly one budget key; resolved `total_steps_` persisted | `baselines/test_tabddpm_wrapper.py::test_budget_*` |
| B-2 | TabDDPM Gaussian block unnormalised | means 323/400/411 vs 0.07/−0.10/4975 → z-scored on fit rows, inverted, persisted | `…::test_numeric_block_is_scale_invariant` |
| B-3 | dtype/cardinality inference: integer class labels and nominal codes diffused as continuous (returned −293.69) | → explicit roles; categorical incl. target through the multinomial block | `…::test_integer_coded_categoricals…` |
| B-4 | unseeded training; `FoundNANsError(BaseException)` aborted whole studies; `dropout=0` asserted; real training ids copied into output | → fixed | `baselines/*` |
| B-5 | "STaSy" is a simplified VE score-SDE; its time curriculum starved the noise levels sampling starts from (std 3.2 vs 0.98) | → renamed, `FAITHFULNESS` dict, deprecated aliases warn, curriculum off by default | `baselines/test_ve_score_sde.py` |
| B-6 | TabPFGen: `balance_classes=True`; neither upstream mode preserves the label prior; output size 10·⌊n/10⌋; raw labels | → prior-preserving resampling to the *training* frequencies, exact n, `LabelCodec` (**fake-generator tests only**) | `baselines/test_tabpfgen_wrapper.py` |
| B-7 | CTGAN: integer codes declared numerical; `enforce_min_max_values=True` clips to the training range | → metadata from explicit roles; clipping off (**stub tests only**) | `baselines/test_ctgan_wrapper.py` |
| X-1 | **target never scaled** (old schema excluded it from every group); schema inferred from dtype on the full table | → target in exactly one declared group; explicit YAML metadata | `experiments/test_protocol_and_splits.py::test_schema_rules` |
| X-2 | two different KFold implementations under the same "seed 42" (sklearn vs numpy permutation): test-fold overlap at chance level | → actual `sklearn.KFold` membership, asserted | `…::test_split_membership_is_identical…` |
| X-3 | TSTR: `CatBoostRegressor`+R² for every task, no `cat_features`, different seeds for real/synthetic, synthetic model trained on len(test) rows, silent `HistGradientBoosting` fallback | → §5 of the spec | `evaluation/test_eval_utility.py` |
| X-4 | tuning: 80/20 seed 42, 50 trials, `MedianPruner` on finished trials, re-running **added** another `n_trials`, shared sampler across datasets, no checkpoints | → `tune.py` | `experiments/test_stages.py` |

---

## 3. Mathematical contracts

Common to all continuous-state SB entries: state space ℝ^d of the adapter representation; **x₀ = data, x₁ = N(0, I)
prior; generation runs the backward model from the prior**. Dynamics noise is always on in canonical entries.

**IPF-DSB** (De Bortoli et al., Alg. 1 / Prop. 3, mean-matching). Reference OU `dX = −αX dt + σ dW` on a grid
γ₀…γ_{K−1}, T = Σγ (a property of the reference, *not* forced to 1). Edge k joins X_k and X_{k+1} with γ_k in **both**
directions. With F_k(x)=x+γ_k f_k(x), B_k(x)=x+γ_k b_k(x):
`B_k(X_{k+1}) ← X_{k+1} + F_k(X_k) − F_k(X_{k+1})` on forward paths, `F_k(X_k) ← X_k + B_k(X_{k+1}) − B_k(X_k)` on
backward paths; both evaluations of the opposite map use the same edge. Iteration 0 uses the analytic reference.
Sampler `X_k = B_k(X_{k+1}) + σ√γ_k Z`. Check: `F_k(X_k)=X_{k+1}−σ√γ_k Z` so the target is `−γ_k f_k(X_{k+1}) − σ√γ_k Z`,
and `E[Z|X_{k+1}] = −σ√γ_k ∇log p_{k+1}` gives conditional mean `γ_k(−f + σ²∇log p_{k+1})`, the time-reversal drift × γ_k.

| entry | parameter sharing | σ, units | independent test |
|---|---|---|---|
| `dsb_dt_joint_gbt` | 2K CatBoost models | σ = √2; target = next-state **mean** (units of x), fitted as residual | index spy; units stub (+1 per step ⇒ +K); OU moment recursion of cached states |
| `dsb_ct_joint_gbt` | 2 time-conditioned models, label `times[k]` | same | one-to-one labels; paired evaluations at equal t |
| `dsb_*_structural_gbt` | per column; parents fixed along a path | same, dim = 1 | DAG learned on fit rows only (index spy); same-row parents; checkpoint restores ordered parent lists |
| `dsb_ct_joint_mlp` | 2 networks, clock 1000·t/T | σ config (default √2); target = **displacement** | cached states follow the reference variance; sampler adds the displacement exactly |
| `dsb_dt_joint_mlp` | 2K networks, no time input | same | every edge trained and called with its index, both directions |

**IMF-DSBM** (Shi et al.). Unit-time Brownian reference, σ, T = 1: `X_t=(1−t)x₀+t x₁+σ√(t(1−t)) ε`,
`u_f=(x₁−X_t)/(1−t)=x₁−x₀−σ√(t/(1−t)) ε`, reverse clock s = 1−t: `u_b=(x₀−X_t)/t=−(x₁−x₀)−σ√((1−t)/t) ε` (drifts, units
x/time; MSE per coordinate). Stage 1: independent coupling; stage ≥ 2: re-simulated by the latest **opposite** model,
anchored on its real start marginal, under `no_grad`. Snapshot: last `b`. Sampler: Euler–Maruyama, `+σ√dt Z`.

| entry | time | test |
|---|---|---|
| `dsbm_ct_joint_mlp` / `_gbt` | t ~ U[ε, 1−ε]; network time clamped to that range, state uses true dt | float64 target oracle; analytic OU (σ = √2 ⇒ drift −x): moments within 5 SE of the **closed-form Euler variance**, bias shrinks N = 4 → 8 |
| `dsbm_dt_joint_mlp` / `_gbt` | forward edge k at k/N, backward at (k+1)/N; no clipping needed | exact-drift oracle hits the point mass to 1e-6 for N ∈ {4, 10, 32} |
| `dsbm_dt_structural_gbt` | same, per column | chain factorisation exact; same-row parents |

**MixedSBM.** ℝ^{d_c} × ∏ finite states; one uniform unit grid drives the Brownian block (formulas above, forward
n ∈ [0, N−1], backward n ∈ [1, N]), the categorical reference, the network clock and the sampler. Categorical block:
`Q_d(s)=exp(sR_d)`, uniform `R=λ(11ᵀ/S−I)` (closed form) or ordered `R=λ(P−I)`, P the symmetric discretised Gaussian,
evaluated by uniformisation (all-positive series ⇒ relative accuracy on tiny entries). Ordered kernel only for columns
whose schema declares an order. Loss `λ_num·MSE + λ_cat·mean_{b,d}[KL(bridge step ‖ model-induced step) + λ_ce CE]`.
**`csbm`**: the categorical block alone; factorised endpoint head (the paper's approximation — a single step need not
reproduce same-step cross-column correlation). **`lightsb`**: `v(y)=Σα_k N(y|r_k, εS_k)`,
`L = E log c(x₀) − E log v(x₁)`, x₀ ~ N(0, I); oracles: conditional moments (N = 2·10⁵, 5σ), quadrature identity for
log c in 1-D/2-D ≤ 1e-8, drift at t = 0 to 1e-10, SDE covariance error ≥ halving per refinement.

---

## 4. Production protocol and category-support feasibility

`configs/protocols/sbtab_8515_hpo100_cv5_v1.yaml` (hash `d661d3713a5f…`), verified by `--dry-run` and by test:
stratified `train_test_split(test_size=0.15, random_state=5)` (regression: 10 target-quantile bins, repeated edges
dropped, reduced until feasible; persisted) → T / V; 100 **allocated** Optuna trials, TPE seed 5, `n_jobs=1`, no pruning,
minimise; `KFold(5, shuffle=True, random_state=42)` on **T only** (pool order: sorted row id); fresh preprocessing and
model per fold; len(V) rows for tuning, len(T_k) for CV; fixed default-derived CatBoost utility. A `kind: production`
protocol with any other constant is rejected; `--smoke` selects a separate protocol with its own id, hash and artifact
root, and refuses production search spaces (and vice versa). There is no `--n-trials` / `--seed` / `--test-size` flag.

*Statistical interpretation (recorded in every manifest):* fixed-hyperparameter CV after dataset-level tuning, **not
nested CV** — every tuning candidate was trained on all of T, which contains every E_k.

**Feasibility under the canonical split WITHOUT row filtering (protocol v1): 18 of 28 configured datasets.** Support is validated for every categorical and
discrete column and the classification target: support(V) ⊆ support(T) and support(E_k) ⊆ support(T_k) ∀k (6 checks).
Blocked datasets are stopped before tuning with `support_report.json` (column/value counts, affected row ids, folds).
No alternative seed, pinned row, merged category, changed K, dropped row or global encoder is used.

| status | datasets |
|---|---|
| **ok** (18) | `auto_mpg`, `bank_loan`, `bank_marketing`, `california_housing`, `car_evaluation`, `cardiovascular_disease`, `churn_modelling`, `covertype`, `diabetes`, `diamonds`, `german_credit`, `insurance`, `king_county_housing`, `lymphography`, `mushroom`, `online_news_popularity`, `online_shoppers`, `real_estate` |
| **blocked_support** (9) | `adult` (native-country), `breast_cancer` (tumor-size, breast-quad, inv-nodes, age), `credit_approval` (A7 ×2, A5, A4), `eucalyptus` (Rep), `forest_fires` (Y, month), `house_sales` (date ×6, bedrooms ×2, grade), `online_shoppers_mixed` (Informational ×4, Browser, TrafficType ×2), `palmer_penguins` (Date Egg ×3, Sex), `stroke_prediction` (gender) |
| **stratification infeasible** (1) | `student_perf`: target `G3` has a class with one row |

Many blockers are single-row artefacts (`gender='Other'`, `Sex='.'`, one-row countries, a 33-bedroom house). Resolving
them is a **schema decision** (new schema hash ⇒ new artifact namespace), deliberately not taken here. Column typing is
explicit in `configs/datasets/*.yaml`, drafted by a fixed rule plus named overrides (`sbtab/data/make_dataset_configs.py`)
and meant to be reviewed; the rule was **not** adjusted to make datasets pass. The only row/column interventions are
dropping `bank_loan.ID` and normalising Adult's `'>50K.'` labels.

### 4.1 Protocol v2 — dataset-eligibility rule (owner decision, 2026-09-20)

The specification forbids discarding rows under `v1` and says that a support-constrained variant "is a separately
versioned protocol". The repository owner decided to remove rows carrying rare category values; this is therefore
implemented as **`sbtab_8515_hpo100_cv5_v2`** (hash `18b2b015dafc…`, now the default; `sbtab_smoke_v2` mirrors it).
`v1` is unchanged, keeps its hash `d661d3713a5f…`, never removes a row, and remains runnable via `--protocol`. The two
protocols cannot share an artifact root.

*Rule* (`sbtab/data/eligibility.py`, `eligibility:` block of the protocol): **before any split**, a row is removed when
its value in a finite-support column — categorical or discrete, including the classification target, i.e. exactly the
columns the support check covers — occurs in **fewer than 3 rows** of the source table. A missing categorical value
counts as the level `__missing__`. The rule is iterated to a fixed point (needed in practice: `lymphography` takes two
passes) and evaluates all columns on the same snapshot, so it does not depend on column order. It uses value counts of
the whole table only — no split, seed or model — and defines D, the "complete eligible dataset". Original row ids are
kept. Each dataset gets an `eligibility_report.json` (rule, removed row ids per column/value and pass, totals) and the
manifest records `n_source_rows`, `source_fingerprint` and the new fingerprint.

*Measured effect* on the 28 configured datasets under the fixed seeds (datasets that pass):

| `min_value_count` | 1 (= v1) | 2 | **3 (v2)** | 4 | 5 | 6–10 | 15 | 20 |
|---|---|---|---|---|---|---|---|---|
| columns = categorical + discrete | 18 | 23 | **25** | 26 | 28 | 28 | 26 | 24 |
| columns = categorical only | 18 | 21 | 23 | 23 | 23 | 23 | 22 | 21 |

| dataset | v1 | v2 | rows removed | note |
|---|---|---|---|---|
| `adult` | blocked | **ok** | 1 / 48 842 | |
| `stroke_prediction` | blocked | **ok** | 1 / 5 110 | `gender='Other'` (row 3116) |
| `credit_approval` | blocked | **ok** | 3 / 690 | |
| `eucalyptus` | blocked | **ok** | 3 / 736 | |
| `forest_fires` | blocked | **ok** | 6 / 517 (1.2 %) | |
| `online_shoppers_mixed` | blocked | **ok** | 9 / 12 330 | |
| `palmer_penguins` | blocked | **ok** | 19 / 344 (**5.5 %**) | mostly rare `Date Egg` values |
| `lymphography` | ok | ok | 10 / 148 (6.8 %) | **target class `normal` (2 rows) removed → 3-class task** (`task_changed`) |
| `breast_cancer` | blocked | **still blocked** | 3 / 286 | `inv-nodes='14-Dec'` has exactly 3 rows; two are in the same test fold. Passes at threshold 4 |
| `house_sales` | blocked | **still blocked** | 43 / 21 613 | `bedrooms`, `date`. Passes at threshold 5 (157 rows) |
| `student_perf` | not stratifiable | **still blocked** | 10 / 649 | target `G3`; classes 19, 1, 5 removed. Passes at 5 (23 rows, 4 classes removed) |
| the other 17 | ok | ok | 0 | |

**What the rule does not do — stated plainly.**
1. **It does not guarantee coverage.** The motivation given was "at least one row in each split", but coverage is a
   different condition: a value present in E_k must be present in T_k (and V ⊆ T). Three rows can fall 1 into V and 2
   into the same test fold. That is exactly why three datasets remain blocked, and it is pinned by
   `test_eligibility.py::test_three_rows_of_a_value_do_not_guarantee_support_coverage`. The support validation
   therefore still runs after the rule and still stops a dataset. Threshold 5 happens to pass all 28 under these seeds;
   no threshold is a guarantee, and large thresholds backfire (whole columns become "rare" and datasets are emptied).
2. **It is outcome-dependent for classification targets** (rare classes are dropped), which `v1`'s spec wording
   excluded. It changes the task for `lymphography` and `student_perf`; every such case is flagged `task_changed`.
3. It filters on the whole table, including rows that later become held-out rows. No statistic is learned from them,
   but D itself now depends on them — results under `v1` and `v2` are **not comparable** and are kept in separate
   namespaces by the protocol hash.

Tests: `tests/experiments/test_eligibility.py` (10 tests: rule, fixed point, order independence, missingness, scope,
target classes, frozen v1, bad rules, the non-guarantee, stage integration, three documented real-data facts).

---

## 5. Entry points, outputs, checkpoints, resume

```bash
python -m sbtab.experiments.prepare_splits --dataset insurance --output-root artifacts/sbtab_8515_hpo100_cv5_v1
python -m sbtab.experiments.tune --dataset insurance --model mixedsbm \
    --splits artifacts/sbtab_8515_hpo100_cv5_v1/insurance/splits.json \
    --search-space configs/search_spaces/mixedsbm.yaml --resume
python -m sbtab.experiments.cross_validate --dataset insurance --model mixedsbm \
    --selected-config <run>/tuning/selected_config.json --splits <splits.json> --output-root <root>
python -m sbtab.experiments.calculate_metrics --cv-run <run>/cv/cv_run_manifest.json --metrics-config configs/metrics/metrics_v1.yaml
python -m sbtab.experiments.aggregate_results --output-root <root> --ranks-on marginal.groups.continuous.mean_wd
```
All accept `--dry-run`; `--smoke` selects `sbtab_smoke_v1` + `configs/search_spaces/smoke/*.yaml`.

Layout: `<root>/<dataset>/{dataset_manifest,schema,splits,support_report,utility_config}.json, data.parquet,
utility_reference/`, then `<model>/<run-id>/{run_manifest.json, tuning/{study.sqlite3, sampler.pkl, trials.csv, best.json,
selected_config.json, trial-NNN/…}, cv/fold-k/{manifest.json, preprocessor/, checkpoint/, synthetic.parquet, timing.json,
training_log.jsonl}, evaluation/<metric-version>-<config-hash>/{fold-k/…, per_fold.csv, summary.json, summary.csv}}`.
`<root>` is the protocol-level directory and is pinned to one protocol hash by `protocol.json`.

* **Resume.** `run-id` derives from a compatibility fingerprint (data, split membership, schema, search space, metric
  config/version, checkpoint/adapter format, protocol, commit + dirty-diff hash); a mismatch refuses to resume. Budget =
  allocated records; stale `RUNNING` trials are reconciled to `FAIL` and counted; the sampler object is pickled after
  every trial (the study DB alone does not hold its RNG state). `best.json` is `provisional` until the budget is spent
  **and** the selected checkpoint reloads, then `final`; `failed` if no trial completes.
* **Checkpoints** are *inference-complete* (config, schema, representation, all networks/trees/potentials, both
  directions' snapshots, reference kernel and grid, EMA, fit status, seeds) and reload without refitting; CPU reload is
  bit-exact and verified on every fit. **They are not resumable** (no optimiser/RNG/cache state): the safe resume
  interval is a trial or a fold. GPU determinism was not measured.
* **Metrics** read saved tables and fold transforms only. Verified on real data: all 36 generator files byte-identical
  after metric calculation; with every checkpoint removed and `fit`/`load_checkpoint` poisoned, 1 925 metrics recompute
  with max |Δ| = 0. The real-data TSTR reference and frozen CatBoost parameters are cached per dataset/split/schema/
  evaluator config, not per generator.

**Metric conventions chosen where the specification was silent** (all versioned under `sbtab.metrics/1`; changing one
requires a new metric version):

* A categorical label outside the training support is an invalid native category ⇒ `invalid_generated_data`. An
  out-of-support value of a *discrete* column is a legal number: status stays `ok` and it is scored through the
  unexpected-value bin (KL) and the label union (JS).
* Interior histogram bins are `[e_i, e_{i+1})`, the last one closed; under/overflow are strict. The same smoothing mass
  applies to continuous-histogram KL. A constant training column uses 48 bins over `[v−½, v+½]` and is flagged.
* Labels align by value (1 and 1.0 are the same label; 1 and "1" are not). Ordinal categoricals are nominal for
  NMI / the Hamming kernel; Spearman runs on `discrete` columns only.
* A conditioner is `incomplete_conditional_coverage` only when an E_k level is absent from the generated table. Levels
  with < 10 rows on either side are kept as `insufficient_data` rows and reported as `insufficient_mass`.
* MMD bandwidth rows are evenly spaced positions (no RNG). The matched real–real floor uses
  `min(⌊|E_k|/2⌋, |G_k|, 2048)`; the `|G_k|` term (added so both sides always have the same size) never binds in CV, and
  `floor_size_limited_by_synth` flags it if it does. The biased estimate is clipped at 0; the unbiased one never is.
* Classification utility trains on integer codes of the fixed training label universe; synthetic labels outside it stay
  in the fit as one always-wrong class and are counted. Resolved CatBoost parameters are every `get_all_params()` key
  the constructor accepts minus `class_names, classes_count, eval_metric, eval_fraction, use_best_model,
  best_model_min_trees`; a refit with them reproduces the default model exactly (tested).

---

## 6. Executed results, timing, unavailable dependencies, remaining issues

**Tests** (`python -m pytest tests`, CPU, macOS arm64, Python 3.11.8, torch 2.6.0, catboost 1.2.7, optuna 4.2.1,
numpy 1.26.4, pandas 2.2.3): **739 passed, 9 skipped, 0 failed** in ≈ 30–55 s.

| suite | scope |
|---|---|
| `tests/bridge` (38) | grid/clock, reference semigroup vs `scipy.linalg.expm`, bridges, loss weights (scalar-loop oracle), absent blocks |
| `tests/solvers` (360, 1 skipped) | MixedSBM / CSBM, boosted IPF-DSB, MLP IPF-DSB, IMF-DSBM, LightSB oracles |
| `tests/baselines` (123, 4 skipped) | TabDDPM, VE score-SDE; CTGAN / TabPFGen library-free logic |
| `tests/evaluation` (73) | marginal, dependence, conditional, MMD, TSTR, JSON safety |
| `tests/experiments` (154, 4 skipped) | protocol constants, split identity, support handling, train-only fitting, Optuna budget / resume / stale trials, sample sizes, persistence and timing, ranks, registry-wide fit→sample→serialise→evaluate, examples, legacy namespace |

Several suites were **mutation-checked**: known defects were re-injected one at a time (9 IMF-DSBM, 21 IPF-MLP + LightSB,
23 baselines, 38 metrics) and every mutant was caught — evidence that the tests can fail for a wrong implementation
rather than restating it. Four of my own tests initially failed for reasons that were *test* errors, not
implementation errors (a hard-coded 0.990 that assumed α = 1; asserting a DAG edge direction that is not identifiable
between Markov-equivalent graphs; resolving only the `fixed` block of a search space; varying production ranges against a
smoke base, which violated the OU stability bound — the production spaces themselves were then verified valid everywhere,
worst case α·γ_max = 0.799). Each was corrected by deriving the expectation, never by loosening a tolerance. One failure
*was* an implementation defect of mine (rank aggregation, below).

**Skipped — not validated (9):** CTGAN adapter (`sdv` missing) ×4; TabPFGen adapter (`tabpfgen`, `tabpfn` missing) ×4;
LightSB full covariance (`geotorch` missing) ×1. Every skip message states that a skip is not a validation. Only library-free logic of CTGAN/TabPFGen was
tested, with fakes. **Not exercised at all:** any GPU path; the production 100-trial budget and production search spaces (validated
against their adapters, never run); end-to-end runs on datasets other than `insurance`, `car_evaluation`,
`diabetes` and `auto_mpg`, and for models other than `mixedsbm`, `csbm`, `lightsb`, `dsbm_dt_joint_gbt` (the other 13 runnable
entries were exercised through the adapter-level registry test, not through the four CLI stages); `covertype`-scale
(581k rows) runtime/memory of any stage on the real dataset; the legacy scripts end-to-end.

**Bounded integration runs** (real data, smoke protocol, all stages through the CLI, metrics in a separate process):

| dataset (regime) | model | trials | folds | metrics / fold |
|---|---|---|---|---|
| `insurance` (mixed, 1 338 rows) | `mixedsbm` | 3/3 COMPLETE | 5/5 ok | 395 |
| `car_evaluation` (discrete, 1 728) | `csbm` | 3/3 | 5/5 | 506 |
| `diabetes` (continuous, 442) | `lightsb` | 3/3 | 5/5 | 210 |
| `diabetes` (continuous) | `dsbm_dt_joint_gbt` | 3/3 | 5/5 | 210 |
| `auto_mpg` (mixed, 398, **6 missing values**, policy `impute`) | `mixedsbm` | 3/3 | 5/5 | – |
| `stroke_prediction` (mixed, 5 110 → 5 109; **blocked under v1**, run under `sbtab_smoke_v2`) | `mixedsbm` | 3/3 | 5/5 | 719 |

On `insurance` the study was interrupted after 1 trial (`provisional`), `--resume` allocated exactly 2 more, a further
resume allocated 0; a start without `--resume` and a production search space under the smoke protocol were both refused.
Metrics were then recomputed with every checkpoint removed (see §5). On `diabetes` the real-data utility reference was
**numerically identical in both model tables** (R² per fold 0.4046, 0.4038, 0.6393, 0.2874, 0.4745). Observed per-fold
means for `mixedsbm` (smoke budget: 32-unit MLP, 2 epochs — **not indicative of production cost or quality**):
preprocessing 0.002 s, init 0.07 s, generator fit 0.10 s, checkpoint I/O 0.015 s, generation 0.012 s, inverse transform
0.001 s, fidelity metrics 0.03 s, utility 0.37 s.

The first five runs used `sbtab_smoke_v1` (before the eligibility rule existed); `stroke_prediction` used
`sbtab_smoke_v2`: the rule removed row 3116, that row appears in no split, and `v1` refuses to write into the `v2` root.
On `auto_mpg` no row was dropped (D = 398), imputed counts are recorded per fold (4, 5, 4, 3, 4 in T_k), and the
train-fitted imputation value **differs per fold** (`horsepower` median 95, 90, 92, 95, 92) — it is fitted on each T_k,
never globally.

*A defect found by this run and fixed:* the first version of `aggregate_results` required a (dataset, fold) cell shared
by **every** model under the root, so an unrelated run on another dataset emptied the intersection and two models that
shared all five `diabetes` folds got no rank. Ranks are now computed per dataset on identical fold sets (models missing
a fold are listed as excluded) and overall only for a model set with common cells
(`test_stages.py::test_ranks_use_identical_fold_sets_and_never_compare_across_datasets`).

**README and examples.** The README was rewritten around the canonical stages; its quickstart block was executed
verbatim (12.5 s). The ten example scripts are thin wrappers over one helper (`examples/_common.py`: dataset registry →
train-only `CommonPreprocessor` → adapter → `sbtab.evaluation`); no metric formula, pickle access or dead API remains
under `examples/` (AST-checked). Each script keeps the solver it demonstrated at `6e2d887` (two of them use the
IMF-DSBM boosted solvers; their IPF-DSB counterparts are separate, accurately named files). Every `--quick` path runs
in the test-suite (≈ 1 s each). Every non-quick "moderate" configuration is strict-key valid, constructible, and was
**executed once** (single run, seed 0, CPU, 7–85 s, validity `ok`) — enough to show it runs, not a quality claim.

**Deviations from the specification** (requested, not fully delivered)

* **No resumable mid-fit checkpoints** (§7 "save resumable state at safe intervals for long trials"; optimiser /
  scheduler / RNG / cache state). All checkpoints are declared `checkpoint_kind: "inference"`. The safe resume interval
  is one trial or one fold, so a long trial that is killed is lost (and recorded as `FAIL`). Consequently the
  "interrupted/resumed fit time is cumulative" acceptance test has no mid-fit case to exercise: `timing.json` records
  `fit_segments: 1`, and the `Timer` accumulates segments by construction, but that path is untested.
* **Peak memory** is the process-level `ru_maxrss`, not a per-stage measurement; GPU nondeterminism was not measured
  because no GPU path was run.
* **Train-only fitting spies** cover the scaler / encoder / imputer (fit-row hash), the learned graph (index spy) and
  the adapter (hash equality enforced at run time); baseline-internal transforms are covered by the baseline tests, but
  SDV's internal transformers (CTGAN) could not be observed because the library is missing.
* The old pipeline (`sbtab/data/{schema,splits,datamodule}.py`, `sbtab/transforms/`) was left in place for the legacy
  scripts rather than removed; the experiment stages do not use it.

**Residual limitations**

1. **IPF-DSB short default horizon — resolved 2026-09-26.** With the legacy grid (γ ∈ [1e-4, 1e-2]) T ≈ 0.046 for K = 20: found twice
   independently — boosted: corr 0.38 → 0.70 (real 0.90) for 1 → 3 IPF iterations vs **0.85 in one iteration at T ≈ 0.8**;
   MLP: W1 0.30 vs 0.33 for the untouched prior, 0.088 at T = 2. Per the spec the solvers were *not* forced to unit
   time; `horizon` / `gamma_max` are exposed and searched. All six IPF solver variants and their adapters now
   default to `horizon=2.0`; the gamma profile is rescaled independently of K. Explicit `horizon=None`
   retains raw increments, and old checkpoints restore their original grids. Existing boosted search
   spaces pin `horizon: null` to preserve the meaning of their `gamma_max` search; these variants remain
   excluded from the experiment pipeline. Regression checks cover OU contraction/noise scale, explicit
   overrides, stability rejection, adapter configuration, and fitted old/new checkpoint reloads in
   `tests/solvers/test_ipf_default_horizon.py`. This corrects the default, not finite-time or Euler error.
2. TabDDPM with tiny budgets is unusable on mixed data (sane from ≈ 5k steps). The VE score-SDE stays weak at small budgets.
3. Per-step MLP IPF produces ~0.1–0.3 % far outliers in short fits.
4. Support typing (heavy-tailed counts typed `discrete`) blocks datasets; see §4.
5. CatBoost reproducibility across thread counts is not guaranteed; `cb_thread_count` is part of the config.
6. TabPFGen upstream internals (possible clipping, SGLD initialised near training rows, checkpoint choice) are outside
   the wrapper; its checkpoints contain the training rows.
7. Boosted-solver checkpoints are pickles — do not load untrusted files.
8. **Metric-stage cost at scale.** On a synthetic 500k × 40 table (E_k = 125k rows): context fit 0.1 s, marginal 5 s,
   association 3 s, MMD 3.5 s, **conditional ≈ 65 s**, ~96 % of it inside `scipy.stats.wasserstein_distance`, which is kept
   as the canonical WD. Relevant for `covertype`; not measured on the real dataset.
9. **Tests against the mixed-XOR MMD oracle** use a two-point continuous coordinate (exact enumeration); a Gaussian
   version was not added because its rigorous tolerance is too loose to discriminate.

---

## 7. Affected results

**Provenance of every tracked historical number is unresolved**: no result file records a commit, code version,
timestamp, host or package versions, so *no number can be assigned to a specific defect*. What can be said:

| tracked artefact (now under `sbtab/experiments/legacy/`) | finding | action |
|---|---|---|
| `calculating_metrics/dsbm_kfold_eval/*`, `tabpfgen_kfold_eval/*` | **inconsistent with the code they were committed with**: under that code the unscaled target gives a best-case `avg_wd` floor of 0.214 on `bank_loan` (964 on `king_county_housing`), yet 0.105 / 0.0866 (0.0897) are recorded | regenerate; provenance unresolved |
| `tuning_script/dsbm_optuna_results/*` | same floor problem (`best_avg_wd` 0.089); several selected `noise=false` | regenerate |
| `tuning_results/best_params/*.json` | parameters **outside every tracked search space** (`imf_len=9`, tuned `grad_clip`, `schedule`, `batch_pac`, …): produced by code not in this repository; `tabbyflow` orphan; `lightsbm` = LightSB | do not reuse; re-tune |
| any DT boosted IPF-DSB result | the committed code **crashes in `fit()`** | cannot originate from `6e2d887` |
| any result of the four boosted metric scripts (`joint_*`, `structural_*_metrics.py`) | they import `sbtab.evaluation.metrics.statistical`, which **never existed in `main`'s history** (its commits `de5acc9`, `2f3325b` are reachable only from unmerged remote branches) and which, as committed there, does not even import (`pd.DataFrame` annotated without importing pandas) | cannot originate from any commit of this branch |
| `tabpfgen_kfold_eval/*` for datasets > 8 000 rows | the script sub-sampled folds with an **unseeded** `.sample(5000)` and `reset_index()` added a row-number column that was fed to the generator as a feature | irreproducible in principle; regenerate |
| `visualization/*.csv` — 45 of 63 rows (CTGAN, TabDDPM, STaSy, DSB, LightSB) | **no tracked fold or summary file exists** for them | source unknown |
| any MLP IPF-DSB result | solver ignored the data | regenerate |
| any TabDDPM "tuned `n_epochs`" | all runs trained 10 000 steps | re-tune with `steps` |
| `visualization/*.csv` | hand-typed `mean+-std` strings in five formats, dropped utility sign, no raw source for 5 of 7 models | do not parse |

All historical metrics used definitions `legacy/0` (KL edges from real ∪ synthetic, regressor-R² utility for every task,
KFold on 100 % of rows, len(test) synthetic rows). They are **not comparable** with `sbtab.metrics/1` and must never be
merged into the same table.

**Legacy namespace** (`sbtab/experiments/legacy/`, see its `README.md`). The 17 old scripts were moved with history and
made thin: 102 copy-pasted helper definitions were replaced by one frozen module `legacy_metrics.py`
(`LEGACY_METRIC_VERSION = "legacy/0"`). The historical definitions are preserved deliberately — where copies differed in
behaviour they are kept as separately named variants (e.g. `corr_frobenius_fillna0` vs `corr_frobenius_raw`, two
`resolve_target_col` semantics) rather than silently unified; equivalence with the copies at `6e2d887` was checked (283
comparisons, 0 mismatches). Tracked result files are byte-identical to the pre-move commit (tested). Status: 16 of 17
scripts import and serve `--help` (each prints a legacy warning); `tabddpm_mixed_data_tuning` needs the uninstalled
`ucimlrepo`; six still contain the pre-existing dead calls; **none was ported to the changed solver APIs and none was
run end-to-end** — they exist so that historical numbers stay interpretable, not to be used.
