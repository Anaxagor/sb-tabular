# Generative algorithm audit — 2026-10-09

This report covers the active generative implementations in `refactor/sbtab-8515-spec-v1`, starting from `1cde6bf`, together with the ForestDiffusion import and corrections made during this audit. It supersedes the model-availability and MSBM-specific statements in the older implementation reports where they disagree.

The review checked model objectives, reference processes, forward/reverse orientation, training targets, time indices, sampling equations, representations, and inference checkpoints. Static inspection was combined with independent mathematical oracles and bounded executable tests. It is **not** a proof of convergence or evidence that every configuration, optional backend, or production benchmark has been validated. Test totals belong to the final task validation output; the family-level coverage below identifies what those tests actually exercise.

**Coverage: 21 implemented model entries.** ForestDiffusion is integrated into the experiment registry. TabbyFlow has a standalone API and remains outside the common experiment adapter. Names for unimplemented algorithms are listed separately below.

| Model ID / standalone name | Implementation reviewed | Interpretation and limits |
|---|---|---|
| `dsb_ct_joint_mlp` | Time-conditioned joint MLP, IPF mean matching | Continuous state, discretized SDE; output is a displacement. |
| `dsb_dt_joint_mlp` | Separate joint MLP for each edge and direction | Discrete time parameterization; continuous state. |
| `dsb_ct_joint_gbt` | Time-conditioned joint CatBoost regressor per direction | Same mean-matching targets, tree approximation. |
| `dsb_dt_joint_gbt` | Joint CatBoost model per edge and direction | Every edge is trained and sampled under its own index. |
| `dsb_ct_structural_gbt` | Time-conditioned scalar CatBoost models conditioned on DAG parents | Conditional factorization extension, including learned structure. |
| `dsb_dt_structural_gbt` | Scalar CatBoost models per feature, edge, and direction | Conditional factorization extension. |
| `dsbm_ct_joint_mlp` | Time-conditioned Brownian bridge drift regression | Canonical adapter uses squared loss and independent initial coupling. |
| `dsbm_ct_joint_gbt` | Time-conditioned joint CatBoost bridge drift regression | Same conditional targets as the neural implementation. |
| `dsbm_dt_joint_mlp` | Separate MLPs at forward/backward state times | Euler approximation of bridge dynamics. |
| `dsbm_dt_joint_gbt` | Separate joint CatBoost models at state times | Euler approximation with tree regressors. |
| `dsbm_dt_structural_gbt` | Conditional autoregressive scalar bridge models | Structural extension; not a literal implementation of the joint projection theorem. |
| `lightsb` | Gaussian-mixture adjusted potential, exact conditional-plan sampling | LightSB; optional SDE sampler. Full covariance requires `geotorch`. |
| `csbm` | Fixed categorical reference and alternating D-IMF updates | Semigroup reference, factorized endpoint head conditioned on the whole row. |
| `csbm_annealed` | CSBM with reference updates between outer iterations | Explicit continuation heuristic; no single fixed reference throughout fit. |
| `mixedsbm` | Shared MLP/optimizer, numerical drift plus categorical endpoint heads | Preserves the selected `feature/tuning` architecture; exact per-step categorical reference after this audit. |
| `ctgan` | Schema-aware wrapper around SDV CTGAN | Delegates adversarial training to the dependency. Real backend execution was unavailable. |
| `tabddpm` | Gaussian and multinomial diffusion | Joint unconditional generation of the row, including target; numerical coordinates use training z-scores. |
| `ve_score_sde_simplified` | VE denoising score matching and predictor–corrector sampling | Declared simplified score-SDE baseline; not full STaSy. |
| `tabpfgen` | `tabpfgen`/TabPFN wrapper with SGLD and label-prior handling | Third-party implementation and wrapper variant; real dependency execution was unavailable. |
| `tabbyflow` (standalone) | TabVFM/TabbyFlow MLP, conditional paths, endpoint loss, ODE sampling | Default OT path checked; optional VP/VE paths retain documented source approximations. |
| `forestdiffusion` | Forest-Flow and Forest-VP, XGBoost fields and schema-aware wrapper | Registry variant generates all modeled columns jointly and unconditionally. |

**ForestDiffusion import and integration.** Only the five Python implementation files under `sbtab/solvers/ForestDiffusion/` were taken from `origin/forest_diffusion` at `50635ca2064df3b3c1b65e41229d0380598172cb`. Unrelated branch experiments and artifacts were not imported. The new `sbtab/baselines/forest_diffusion/` wrapper and experiment adapter use a training-fitted representation: standardized numerical coordinates, full one-hot nominal blocks, nearest-support discrete decoding, fresh IDs, and inference checkpoints.

The family corresponds to the tree-based flow/diffusion approach in [Jolicoeur-Martineau et al.](https://proceedings.mlr.press/v238/jolicoeur-martineau24a/jolicoeur-martineau24a.pdf), also available on [arXiv](https://arxiv.org/abs/2309.09968), and its [upstream implementation](https://github.com/SamsungSAILMontreal/ForestDiffusion). The benchmark wrapper is an unconditional joint-row variant; it should not be described as a reproduction of the paper's label-conditioned experimental settings.

The imported implementation needed corrections before use:

- Chunking confused split lengths with split indices. `array_split` now preserves all rows, including uneven batches and small class subsets.
- XGBoost's data iterator now replays the same augmented observations when reset; quantile construction and tree training must see the same data. This follows the iterator contract described in the [XGBoost external-memory documentation](https://xgboost.readthedocs.io/en/stable/tutorials/external_memory.html).
- Fixed categorical vocabularies and full one-hot blocks prevent category shifts between training batches. Arbitrary class labels, binary support, single-output prediction shape, and local random streams are handled explicitly.
- VP denoising uses the full Tweedie expression `(x_t + sigma_t^2 * score_t) / alpha_t`; omitting division by `alpha_t` changes the denoised estimate.
- VP RePaint forward jumps use the transition's conditional mean and variance, rather than combining incompatible endpoint noise scales. Repeated score evaluation and class-prediction noise scaling/shape were corrected.
- VP training now excludes rows whose original supervised coordinate is missing: a finite noise target alone does not make that observation usable. Covariate widths/presence and imputation labels are validated. Generation with both class labels and covariates is explicitly rejected because it would require an unfitted `p(label|covariates)`. The wrapper does not expose imputation and removes retained training rows from its inference artifact.

Independent Forest tests check interpolation targets, full per-class training row counts, iterator replay, category/binary support, seed isolation, checkpoint restoration, reverse-SDE coefficients, exact forward RePaint transitions, and Tweedie recovery of a point mass from its analytic score. Real XGBoost fit/sample/checkpoint/evaluation runs cover all three schema regimes. Small CPU fit/sample checks also passed for random forests, LightGBM and CatBoost with one and two output coordinates. As in standard finite-horizon VP sampling, initialization from a Gaussian approximates the terminal noised-data marginal; finite time grids and learned trees add approximation error.

**IPF-DSB: six variants.** The inspected caches simulate complete trajectories of the opposite process. For a forward mean map `F`, the backward displacement target is `F(x) - F(y)` evaluated on a simulated edge `x -> y`; the resulting backward mean is `y + displacement`. The reverse stage uses the analogous backward-map expression. The sampler adds a learned displacement directly, without multiplying by the time interval a second time. The initial trajectory uses the analytic OU reference, and training caches remain stochastic even when final generation is requested without noise. These identities were checked against [De Bortoli et al., Proposition 3](https://arxiv.org/pdf/2106.01357) and independent trajectory/mean-map tests.

This audit found no additional error in the default mean-matching formulas. It corrected fit lifecycle defects in all four boosted variants: repeated fit now resets its seeded random stream; an unsuccessful refit invalidates the previous fitted flag; invalid training data is rejected. Learned DAG handling now preserves integer column labels. These changes are covered in `tests/solvers/test_boosted_ipf.py`; existing cache, sampling, horizon, and learning suites check formulas and behavior for the neural variants.

**IMF-DSBM: five variants.** For endpoint pair `(z0, z1)`, the numerical bridge is

```text
z_t = (1-t) z0 + t z1 + sigma * sqrt(t(1-t)) * epsilon
forward target  = (z1 - z_t) / (1-t)
backward target = (z0 - z_t) / t
```

The code's noise-corrected expressions are algebraically equivalent. The review checked alternating couplings, the real-data/prior endpoint anchors, forward/backward state-time indices, `dt` versus `sqrt(dt)` scaling, and independent random increments against [Shi et al., Algorithm 1](https://arxiv.org/html/2303.16852v3). Training couplings retain noise. The continuous-time endpoint clamp and finite Euler grid are numerical approximations.

New checks reject nonfinite training values in the joint MLP/GBT implementations and stop neural optimization on a nonfinite loss. Previously a fit could appear successful and subsequently produce NaNs. Tests are in `tests/solvers/test_dsbm_solvers.py`; the independent target and state-time oracles are in `test_dsbm_math.py`.

Direct low-level options must retain their declared interpretation: Huber regression is a robust-loss variant of the L2 projection; `first_coupling="ref"` is distinct from independent data/prior coupling; `noise=False` is a noiseless sampling heuristic, not a probability-flow ODE. Structural conditional factorization is an extension of the joint algorithm, not a consequence that can simply be attributed to the joint Markov-projection theorem.

**CSBM, annealed CSBM, and MixedSBM.** The canonical categorical reference uses `Q(t)=exp(tR)` and conditions via Bayes' rule. Forward/backward transition mixtures, masks, current-state clocks, per-column loss normalization, and endpoint anchoring were checked against [CSBM equations 7–10 and Algorithm 1](https://arxiv.org/html/2502.01416v2). Existing tests compare its transitions with independent matrix exponentials and enumerate conditional distributions. Annealing remains confined to the named heuristic, including restoration of the final reference from a checkpoint.

MixedSBM deliberately retains the shared network and optimizer imported from `feature/tuning`. Its `alpha` remains a **per-step** parameter, so changing the number of steps changes that reference law. Its ordered kernel remains the historical row-normalized Gaussian matrix, which is generally asymmetric; it is a valid alternative reference, not the symmetric kernel in the CSBM paper. This audit repaired its powers and conditioning rather than substituting the canonical CSBM reference.

| MixedSBM defect | Correction | Independent regression evidence |
|---|---|---|
| Ordered transitions changed from `Q^k` to an unrelated widened kernel at `k=30`. | Every cached transition is a power of the same matrix. | Matrix-power oracle at `k=1,29,30,40`; Chapman–Kolmogorov composition. |
| Epsilon denominators and epsilon sampling mass changed rare bridges into unrelated distributions. | Exact log-space conditioning; impossible bridges fail explicitly; zero-probability states remain impossible. | Ordered binary bridge with step probability `exp(-40000)`; periodic reference with unreachable endpoints. |
| Probability-space KL clipped very small predictions and lost meaningful gradients. | Model-induced log transition probabilities feed the KL directly; padded endpoint logits are masked. | Endpoint NLL and gradient oracle at logits `+1000,-1000`. |
| Small nominal alpha suffered cancellation. | Stable `log1p`/`expm1` calculations. | Rare bridge with `alpha=1e-20`. |
| `noise=False` also removed Brownian noise from training couplings. | Couplings always use the Brownian reference; the switch controls final generation only. | Zero-drift coupling has the expected diffusion variance; noiseless sampling remains deterministic given its start. |

The additional tests in `tests/solvers/test_msbm_reference_audit.py` also check asymmetric-kernel Bayes formulas, explicit endpoint mixtures, categorical-only losses, and invariance to duplicating a categorical column. MixedSBM loss normalization was already per-column; this audit did not introduce another division by feature count.

The corrected reference changes inference semantics. New checkpoints use **`sbtab.mixedsbm/4`**; older formats are rejected with an explicit retraining message. Loading old weights under a silently different categorical reference would not reproduce the saved model.

**LightSB.** The adjusted potential, log normalizer, training objective, conditional Gaussian-mixture plan, and associated drift agree with the [official implementation](https://raw.githubusercontent.com/ngushchin/LightSB/main/src/light_sb.py) and [LightSB paper](https://arxiv.org/abs/2310.01174). The objective remains `mean(log C(x0)) - mean(log v(x1))`. Existing independent NumPy/quadrature oracles verify normalization, mixture moments, drift, and convergence of the refined Euler sampler toward the conditional plan. No model-equation change was needed. This is LightSB, not LightSB-M. The full-covariance execution path was skipped because `geotorch` was unavailable; inspecting its algebra does not replace that missing runtime check.

**TabDDPM.** The review compared the Gaussian noise/clean-sample conversions, categorical Markov posterior, variational terms, and sampling paths with the [official implementation](https://raw.githubusercontent.com/yandex-research/tab-ddpm/main/tab_ddpm/gaussian_multinomial_diffsuion.py). It corrected unstable categorical block normalization: a cumulative log-sum subtraction could produce `-inf` for a later block with much smaller logits. Per-block `logsumexp` now preserves its normalization and gradients.

For `gaussian_loss_type="kl"`, the terminal decoder now uses a continuous Gaussian density appropriate for the wrapper's unbounded standardized numerical coordinates. The old image-specific bounded 8-bit likelihood saturated outside its image range. Default MSE training is unchanged by that likelihood correction. Sampling rejects infinities as well as NaNs. Low-level corrections make `gaussian_parametrization="x0"` regress clean samples, forward the DDIM `eta` argument, and return per-row zeros for absent ELBO blocks. Unsupported legacy image/likelihood methods and `multinomial_loss_type="vb_all"` are rejected explicitly. `mixed_elbo` is a Monte Carlo variational diagnostic, not an exact likelihood; its Gaussian term is in bits per dimension and categorical term in nats. See `tests/baselines/test_diffusion_math.py` and the wrapper tests. This remains a joint `(X,y)` wrapper variant with z-score preprocessing, not an exact reproduction of all original TabDDPM experiment settings.

**Simplified VE score-SDE.** The implementation's denoising objective is `E ||sigma(t) * score(x_t,t) + epsilon||^2`. Reverse predictor variance increments and Langevin corrector updates were checked against the [official score-SDE losses](https://raw.githubusercontent.com/yang-song/score_sde_pytorch/main/losses.py) and [sampling implementation](https://raw.githubusercontent.com/yang-song/score_sde_pytorch/main/sampling.py). A Gaussian score oracle tests reverse-process moments. The model is honestly named `ve_score_sde_simplified`: it does not implement STaSy's full architecture, per-sample self-paced learning, and fine-tuning procedure. Finite maximum noise still gives an approximate Gaussian initialization for general data.

**CTGAN and TabPFGen.** CTGAN delegates training to SDV rather than implementing a second GAN internally. The schema/roles, categorical metadata, output shape, label handling, and checkpoints were reviewed against the [official CTGAN implementation](https://raw.githubusercontent.com/sdv-dev/CTGAN/main/ctgan/synthesizers/ctgan.py). TabPFGen delegates to the [third-party `sebhaan/TabPFGen` implementation](https://raw.githubusercontent.com/sebhaan/TabPFGen/main/src/tabpfgen/tabpfgen.py) and adds declared label-prior resampling. The dependency's [references](https://github.com/sebhaan/TabPFGen#references) distinguish it from an author implementation of the paper. Fit stores conditioning data; it does not perform the gradient training counted by neural generators. Wrapper tests with controlled backends cannot validate either unavailable dependency's real training/sampling behavior.

**TabbyFlow standalone implementation.** Network, endpoint prediction loss, categorical softmax handling, conditional path derivatives, and velocity reconstruction were compared with the [TabVFM flow-matching source](https://raw.githubusercontent.com/rulnasution/tabular-flow-matching/main/baselines/tabvvfm/flow_matching.py) and [network source](https://raw.githubusercontent.com/rulnasution/tabular-flow-matching/main/baselines/tabvvfm/networks.py). The audit corrected stale best-loss state across refits, typed category collisions such as integer `1` versus string `"1"`, discrete support/dtype restoration, undeclared-column bootstrapping, and ID generation. The VE source uses standard deviation 2 rather than 1.

All four path derivatives have independent autograd checks, and the default OT endpoint is checked under Euler, midpoint, and RK4. OT has a standard Gaussian source and residual terminal noise `0.001`; cosine also has an exact Gaussian source. VP and VE retain source approximations. In particular, the VE endpoint source is data plus `N(0,4I)`, approximated by `N(0,4I)` during generation; correcting the noise scale does not make that approximation exact. The module states this limitation explicitly. This audit does not add a common experiment adapter for TabbyFlow.

**Validation boundaries and use of results.** See the task validation output for final combined test totals, failures or environment-specific reruns, and the exact committed revision. Family suites include analytic targets, finite-state enumeration, quadrature, sampler moments, fit/sample/checkpoint cycles, and regression tests that fail under the identified defects. CPU Forest-Flow and Forest-VP tree execution was exercised in bounded runs. Optional-dependency skips remain skips, especially real SDV/TabPFGen integration and full-covariance LightSB. CUDA behavior, all supported missing-data/imputation combinations, large-scale hyperparameter tuning, and real-data comparative quality are not established by this audit.

Unavailable names must not acquire results from a substitute implementation: faithful `stasy`, `lightsb_m`, and `tabsyn` remain unimplemented. TabbyFlow's standalone status and ForestDiffusion's newly integrated status supersede their former absence statements in older reports. For continuous-state models used on categorical tables, one-hot encoding and support decoding define an adapted representation; passing a validity test after decoding is not evidence of an exact discrete-state generative law.

**Final validation.** The complete suite passed on CPU with Python 3.11:

```text
/opt/anaconda3/envs/synth/bin/python -m pytest -o addopts='' -q -ra
1022 passed, 23 skipped, 22 warnings in 64.60s
```

The 23 skips comprise 11 CUDA cases, 11 SDV/TabPFGen dependency cases, and one full-covariance LightSB case. The final run used an unrestricted local process because macOS sandbox restrictions prevent the lazy-import subprocess from opening OpenMP shared memory. A legacy source-equivalence test was adjusted to ignore trailing whitespace only; executable historical metric code was unchanged. Forest modules also passed compilation; the patch passed `git diff --check`.
