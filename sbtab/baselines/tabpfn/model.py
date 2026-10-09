"""
TabPFGen wrapper (sebhaan/TabPFGen: SGLD in feature space + TabPFN labelling).
``tabpfgen`` / ``tabpfn`` are imported lazily, so ``import sbtab.baselines`` works without them.

Cost model - read this before comparing budgets
-----------------------------------------------
``fit()`` performs NO gradient updates: it stores a conditioning subset of the training rows.
Contexts above the pinned TabPFN row limit are sampled without replacement (10,000 on CUDA,
1,000 on CPU), using cfg.seed and target stratification for classification. Codecs and the
requested label prior use the full training input; validation/test rows are never used.
Sampling positions and settings are saved in checkpoints. ALL generation cost is paid at
``sample()`` time: ``n_sgld_steps`` SGLD steps, each
with a ``cdist`` of the synthetic batch against every context row, plus one TabPFN
fit / predict per generation call.  ``adaptation_cost_`` declares this:
``{"fit_updates": 0, "sgld_steps": ..., "context_rows": ...}`` and ``last_sample_cost_`` records
what a ``sample`` call actually spent.  ``save_checkpoint`` therefore PERSISTS THE TRAINING ROWS
- a TabPFGen checkpoint is as sensitive as the training data itself.

Label prior (classification)
----------------------------
Upstream never reproduces the training label prior: ``balance_classes=True`` generates equal
counts per class, ``False`` draws labels uniformly at random, and in both cases the final labels
are overwritten by a TabPFN argmax.  ``balance_classes`` therefore defaults to ``False`` and,
with ``preserve_label_prior=True`` (default), the wrapper over-generates and resamples the
GENERATED rows to the saved empirical TRAINING class frequencies (largest-remainder allocation,
counts sum to exactly ``n``).  Short classes are topped up by further generation (bounded
retries).  A class that cannot be produced is recorded in ``label_report_``; its quota is filled
with generated rows of the other classes - real rows are never injected.

Output size
-----------
Upstream returns ``K * floor(n / K)`` rows (classification) or ``10 * floor(n / 10)`` rows
(regression).  The wrapper requests the next multiple, selects exactly ``n`` rows and asserts it.

Labels
------
Labels are encoded to ``0..K-1`` before they reach upstream (which returns argmax indices) and
mapped back on output, so string labels and labels such as ``{-1, 1}`` round-trip.

Column roles
------------
``fit(data, *, continuous_cols=, discrete_cols=, categorical_cols=, target_col=, task=, id_col=)``.
TabPFGen moves points in a continuous feature space, so categorical FEATURES are one-hot encoded
(train-fitted vocabulary, argmax-decoded) and discrete features are decoded to the nearest
training-support value (``decoding_report_``).  Without explicit lists every feature is treated
as continuous (declared default) and ``task="auto"`` falls back to the deprecated dtype heuristic
with a warning.
"""

from __future__ import annotations

import hashlib
import importlib
import inspect
import math
import warnings
from dataclasses import asdict, dataclass, replace
from importlib import metadata as importlib_metadata
from typing import Any, Callable, Dict, List, Literal, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from sbtab.baselines.base import (
    ArrayLike,
    BaselineFitInfo,
    BaselineGenerativeModel,
    ColumnRoles,
    FreshIdFactory,
    explicit_roles_given,
    largest_remainder_allocation,
    next_multiple,
    resolve_column_roles,
    seed_everything,
    to_python_scalar,
    validate_n,
)
from sbtab.baselines.encoding import (
    MixedToContinuousCodec,
    fit_vocabulary,
    nearest_support_decode,
    restore_dtype,
)

TaskType = Literal["auto", "classification", "regression"]

_CHECKPOINT_FORMAT = "sbtab.baselines.tabpfgen"
_CHECKPOINT_VERSION = 1
REGRESSION_ROW_MULTIPLE = 10   # upstream stratifies regression generation over 10 quantile bins

GenerateFn = Callable[[int], Tuple[np.ndarray, np.ndarray]]


# ----------------------------------------------------------------------
# pure, library-free pieces
# ----------------------------------------------------------------------


class LabelCodec:
    """Train-fitted label <-> ``0..K-1`` index mapping (python scalars; dtype restored on decode)."""

    def __init__(self, classes: Optional[Sequence[Any]] = None, dtype: str = "object"):
        self.classes: List[Any] = list(classes) if classes is not None else []
        self.dtype = str(dtype)

    def fit(self, y: Sequence[Any]) -> "LabelCodec":
        s = pd.Series(y)
        self.dtype = str(s.dtype)
        self.classes = fit_vocabulary(s, "target")
        return self

    @property
    def n_classes(self) -> int:
        return len(self.classes)

    def encode(self, y: Sequence[Any]) -> np.ndarray:
        index = {v: i for i, v in enumerate(self.classes)}
        try:
            return np.asarray([index[to_python_scalar(v)] for v in pd.Series(y).tolist()], dtype=np.int64)
        except KeyError as e:
            raise ValueError(f"Label {e} was not seen in training.") from e

    def decode(self, idx: Sequence[int]) -> pd.Series:
        idx = np.asarray(idx)
        if idx.size and not np.all(np.isfinite(idx.astype(np.float64))):
            raise ValueError("Generated label indices are not finite.")
        idx_int = np.rint(idx.astype(np.float64)).astype(np.int64)
        if idx_int.size and (idx_int.min() < 0 or idx_int.max() >= self.n_classes):
            raise ValueError(f"Generated label index outside 0..{self.n_classes - 1}.")
        table = np.empty(self.n_classes, dtype=object)
        table[:] = self.classes
        return restore_dtype(table[idx_int], self.dtype)


def select_exact_rows(X: np.ndarray, y: np.ndarray, n: int, rng: Optional[np.random.Generator] = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Exactly ``n`` of the generated rows.  With ``rng`` a uniformly random subset (upstream output
    can be ordered by class / quantile bin, so taking the head would bias the result); without
    ``rng`` the head.  Raises if fewer than ``n`` rows are available - nothing is ever padded.
    """
    X, y = np.asarray(X), np.asarray(y)
    if len(X) != len(y):
        raise ValueError("X and y have different lengths.")
    if len(X) < n:
        raise ValueError(f"Only {len(X)} generated rows available, need {n}.")
    idx = np.arange(n) if rng is None else np.sort(rng.choice(len(X), size=n, replace=False))
    X_out, y_out = X[idx], y[idx]
    assert len(X_out) == n and len(y_out) == n
    return X_out, y_out


def _generate_checked(generate_fn: GenerateFn, m: int, n_features: Optional[int]) -> Tuple[np.ndarray, np.ndarray]:
    X, y = generate_fn(int(m))
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y)
    if X.ndim != 2 or len(X) != len(y):
        raise RuntimeError(f"Generator returned inconsistent shapes X={X.shape}, y={y.shape}.")
    if n_features is not None and X.shape[1] != n_features:
        raise RuntimeError(f"Generator returned {X.shape[1]} features, expected {n_features}.")
    keep = np.all(np.isfinite(X), axis=1)
    return X[keep], y[keep]


def generate_exact(
    generate_fn: GenerateFn,
    n: int,
    *,
    rng: np.random.Generator,
    row_multiple: int = 1,
    max_rounds: int = 5,
    n_features: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """Regression / no-prior path: request the next multiple of ``row_multiple``, top up, select ``n``."""
    parts_X: List[np.ndarray] = []
    parts_y: List[np.ndarray] = []
    have, rounds, requested = 0, 0, 0
    while have < n and rounds < max_rounds:
        m = next_multiple(n - have, row_multiple)
        X, y = _generate_checked(generate_fn, m, n_features)
        rounds += 1
        requested += m
        parts_X.append(X)
        parts_y.append(y)
        have += len(X)
    if have < n:
        raise RuntimeError(f"Generator produced {have} usable rows in {rounds} rounds, need {n}. Real rows are never used.")
    X_all, y_all = np.concatenate(parts_X), np.concatenate(parts_y)
    X_out, y_out = select_exact_rows(X_all, y_all, n, rng)
    return X_out, y_out, {"rounds": rounds, "requested_rows": requested, "generated_rows": int(have), "returned_rows": n}


def resample_to_label_prior(
    generate_fn: GenerateFn,
    n: int,
    class_freqs: Sequence[float],
    *,
    rng: np.random.Generator,
    row_multiple: Optional[int] = None,
    oversample: float = 2.0,
    max_rounds: int = 5,
    max_request_factor: float = 10.0,
    n_features: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Over-generate with ``generate_fn(m) -> (X, y_idx)`` and resample the GENERATED rows so that the
    returned label counts equal ``largest_remainder_allocation(n, class_freqs)``.

    * round 1 requests ``ceil(oversample * n)`` rows (rounded up to ``row_multiple``, default ``K``);
    * while a class is short (and rounds remain) more rows are generated; the request is sized
      from the observed per-class yield and capped at ``max_request_factor * n``;
    * a class still short after ``max_rounds`` is reported under ``report["unfilled"]`` and its
      quota is re-allocated to classes with surplus GENERATED rows, proportionally to the training
      frequencies.  Real rows are never injected.  If fewer than ``n`` rows exist in total, raises.
    """
    freqs = np.asarray(list(class_freqs), dtype=np.float64)
    K = int(freqs.size)
    mult = int(row_multiple) if row_multiple is not None else max(K, 1)
    target = largest_remainder_allocation(n, freqs)

    pool_X: List[np.ndarray] = []
    pool_y: List[np.ndarray] = []
    produced = np.zeros(K, dtype=np.int64)
    rounds, requested, out_of_range = 0, 0, 0

    def short() -> np.ndarray:
        return np.maximum(target - produced, 0)

    while rounds < max_rounds:
        if rounds == 0:
            m = int(math.ceil(max(float(oversample), 1.0) * n))
        else:
            total = max(int(produced.sum()), 1)
            yield_k = np.maximum(produced / total, 1.0 / (total + 1.0))   # unseen class: < 1 row per pool
            m = int(math.ceil(1.2 * float(np.max(short() / yield_k))))
        m = next_multiple(min(max(m, 1), int(math.ceil(max_request_factor * n))), mult)

        X, y = _generate_checked(generate_fn, m, n_features)
        rounds += 1
        requested += m
        y_int = np.rint(np.asarray(y, dtype=np.float64)).astype(np.int64)
        valid = (y_int >= 0) & (y_int < K)
        out_of_range += int((~valid).sum())
        X, y_int = X[valid], y_int[valid]
        pool_X.append(X)
        pool_y.append(y_int)
        produced += np.bincount(y_int, minlength=K)[:K]
        if not short().any():
            break

    X_pool = np.concatenate(pool_X) if pool_X else np.empty((0, n_features or 0))
    y_pool = np.concatenate(pool_y) if pool_y else np.empty(0, dtype=np.int64)
    if len(X_pool) < n:
        raise RuntimeError(
            f"Generator produced {len(X_pool)} usable rows in {rounds} rounds, need {n}. Real rows are never used."
        )

    # final per-class counts: the target where possible, deficits moved to classes with surplus
    final = np.minimum(target, produced)
    deficit = int(n - final.sum())
    while deficit > 0:
        surplus = produced - final
        has = surplus > 0
        weights = np.where(has, np.maximum(freqs, 1e-12), 0.0)
        extra = np.minimum(largest_remainder_allocation(deficit, weights), surplus)
        if extra.sum() == 0:   # pragma: no cover - impossible while len(pool) >= n
            raise RuntimeError("Could not re-allocate the label deficit.")
        final += extra
        deficit = int(n - final.sum())

    chosen: List[np.ndarray] = []
    for k in range(K):
        idx_k = np.flatnonzero(y_pool == k)
        if final[k] > 0:
            chosen.append(rng.choice(idx_k, size=int(final[k]), replace=False))
    idx = rng.permutation(np.concatenate(chosen)) if chosen else np.empty(0, dtype=np.int64)
    X_out, y_out = X_pool[idx], y_pool[idx]
    assert len(X_out) == n, (len(X_out), n)

    unfilled = {
        int(k): {"requested": int(target[k]), "produced": int(produced[k]), "returned": int(final[k])}
        for k in range(K) if produced[k] < target[k]
    }
    report = {
        "target_counts": [int(v) for v in target],
        "returned_counts": [int(v) for v in final],
        "produced_counts": [int(v) for v in produced],
        "rounds": rounds,
        "requested_rows": int(requested),
        "generated_rows": int(len(X_pool)),
        "out_of_range_labels": int(out_of_range),
        "unfilled": unfilled,
        "prior_preserved": not unfilled,
    }
    return X_out, y_out, report


def _package_version(name: str) -> Optional[str]:
    try:
        return importlib_metadata.version(name)
    except importlib_metadata.PackageNotFoundError:
        return None


def discover_pretrained_identity() -> Dict[str, Any]:
    """Whatever identifies the pretrained prior at runtime (best effort; ``None`` = not discoverable)."""
    ident: Dict[str, Any] = {
        "tabpfgen_version": _package_version("tabpfgen"),
        "tabpfn_version": _package_version("tabpfn"),
        "torch_version": _package_version("torch"),
        "tabpfn_default_model_path": None,
        "note": (
            "upstream TabPFGen instantiates TabPFNClassifier/TabPFNRegressor with their DEFAULT weights inside "
            "generate_*; the wrapper cannot pin a checkpoint, it can only record the package versions"
        ),
    }
    try:
        tabpfn = importlib.import_module("tabpfn")
        params = inspect.signature(tabpfn.TabPFNClassifier.__init__).parameters
        if "model_path" in params:
            ident["tabpfn_default_model_path"] = str(params["model_path"].default)
    except Exception:
        pass
    return ident


# ----------------------------------------------------------------------
# config / wrapper
# ----------------------------------------------------------------------


@dataclass
class TabPFGenConfig:
    """
    Wrapper config around sebhaan/TabPFGen.

    ``n_sgld_steps`` / ``sgld_step_size`` / ``sgld_noise_scale`` / ``device`` are the TabPFGen
    constructor parameters.  ``target_col`` may instead be given to ``fit``.
    """

    target_col: Optional[str] = None
    task: TaskType = "auto"

    # TabPFGen core sampling controls
    n_sgld_steps: int = 1000
    sgld_step_size: float = 0.01
    sgld_noise_scale: float = 0.01
    device: Literal["cpu", "cuda", "auto"] = "auto"

    # Upstream generation options.  balance_classes=False by default: True forces EQUAL class
    # counts.  Neither value preserves the training label prior - `preserve_label_prior` does.
    balance_classes: bool = False     # classification
    use_quantiles: bool = True        # regression

    seed: int = 0

    # label-prior preservation (classification)
    preserve_label_prior: bool = True
    prior_oversample: float = 2.0     # first request = ceil(prior_oversample * n) rows
    max_topup_rounds: int = 5         # bound on generation calls per sample()

    def __post_init__(self) -> None:
        if self.task not in ("auto", "classification", "regression"):
            raise ValueError(f"task must be 'auto', 'classification' or 'regression', got {self.task!r}.")
        if int(self.n_sgld_steps) < 1:
            raise ValueError("n_sgld_steps must be >= 1.")
        if float(self.prior_oversample) < 1.0:
            raise ValueError("prior_oversample must be >= 1.")
        if int(self.max_topup_rounds) < 1:
            raise ValueError("max_topup_rounds must be >= 1.")


def _generate_unbalanced_classification(generator, X_train, y_train, n_samples):
    """TabPFGen 0.1.4's unbalanced proposal, with label counting kept on CPU.

    Upstream calls np.unique on a CUDA tensor in this branch. Keep its scaler,
    initialization, SGLD and TabPFN refinement unchanged; only count labels before
    moving them to the device. Kept callable on CPU for an exact equivalence test.
    """
    import torch
    from tabpfn import TabPFNClassifier

    scaled = generator.scaler.fit_transform(X_train)
    labels = np.asarray(y_train)
    x_train = torch.tensor(scaled, device=generator.device, dtype=torch.float32)
    y_train = torch.tensor(labels, device=generator.device)
    x_synth = torch.randn(n_samples, X_train.shape[1], device=generator.device) * 0.01
    y_synth = torch.randint(0, len(np.unique(labels)), (n_samples,), device=generator.device)
    for _ in range(generator.n_sgld_steps):
        x_synth = generator._sgld_step(x_synth, y_synth, x_train, y_train)
    classifier = TabPFNClassifier(device=generator.device)
    classifier.fit(x_train.cpu().numpy(), y_train.cpu().numpy())
    probabilities = classifier.predict_proba(x_synth.detach().cpu().numpy())
    return generator.scaler.inverse_transform(x_synth.detach().cpu().numpy()), probabilities.argmax(axis=1)


def tabpfn_context_row_limit(device):
    """Context limit of the pinned TabPFN 2.0.9 default estimators."""
    import torch

    device = str(device)
    on_cpu = device.startswith("cpu") or (device == "auto" and not torch.cuda.is_available())
    return 1000 if on_cpu else 10_000


def validate_tabpfn_limits(n_rows, n_features, n_classes, device):
    """Validate an already selected context, including feature/class limits."""
    row_limit = tabpfn_context_row_limit(device)
    if n_rows > row_limit or n_features > 500 or n_classes > 10:
        raise ValueError(f"TabPFN default limits exceeded: {n_rows} rows (limit {row_limit}), "
                         f"{n_features} encoded features (limit 500), {n_classes} classes (limit 10). "
                         "The conditioning context must satisfy these limits.")


def plan_tabpfn_context(n_rows, n_features, n_classes, device):
    """Allow large training inputs by capping only the conditioning context."""
    limit = tabpfn_context_row_limit(device)
    n_context = min(int(n_rows), limit)
    validate_tabpfn_limits(n_context, n_features, n_classes, device)
    return {"n_train_rows": int(n_rows), "n_context_rows": n_context,
            "row_limit": limit, "subsampled": n_context < n_rows}


def sample_context_indices(y, n_context, task, seed):
    """Select training positions only, retaining every classification target class."""
    y = np.asarray(y)
    if not 0 < n_context <= len(y):
        raise ValueError("context size must be between 1 and the number of training rows")
    if n_context == len(y):
        return np.arange(len(y), dtype=np.int64)
    rng = np.random.default_rng(int(seed))
    if task == "classification":
        _, labels, counts = np.unique(y, return_inverse=True, return_counts=True)
        if n_context < len(counts):
            raise ValueError("context must contain at least one row of each target class")
        # Reserve one row per class, then distribute the remaining quota in
        # proportion to remaining support. This also retains singleton classes.
        quotas = 1 + largest_remainder_allocation(n_context - len(counts), counts - 1)
        selected = np.concatenate([rng.choice(np.flatnonzero(labels == k), size=int(quota), replace=False)
                                   for k, quota in enumerate(quotas)])
    else:
        selected = rng.choice(len(y), size=n_context, replace=False)
    return np.sort(selected).astype(np.int64)


def _default_generator_factory(cfg: TabPFGenConfig) -> Any:
    try:
        from tabpfgen import TabPFGen  # lazy
    except Exception as e:  # pragma: no cover - depends on the environment
        raise ImportError("TabPFGenGenerative requires `tabpfgen`. Install: pip install tabpfgen") from e
    class GPUCompatibleTabPFGen(TabPFGen):
        def validate_context(self, X, y, task):
            validate_tabpfn_limits(len(X), X.shape[1], len(np.unique(y)) if task == "classification" else 0,
                                   self.device.type)

        def generate_classification(self, X_train, y_train, n_samples, balance_classes=True):
            if self.device.type == "cuda" and not balance_classes:
                return _generate_unbalanced_classification(self, X_train, y_train, n_samples)
            return super().generate_classification(X_train, y_train, n_samples, balance_classes)

    return GPUCompatibleTabPFGen(
        n_sgld_steps=int(cfg.n_sgld_steps),
        sgld_step_size=float(cfg.sgld_step_size),
        sgld_noise_scale=float(cfg.sgld_noise_scale),
        device=str(cfg.device),
    )


class TabPFGenGenerative(BaselineGenerativeModel):
    """
    TabPFGen baseline; see the module docstring (cost model, label prior, exact ``n``, labels).

    ``generator_factory(cfg)`` can be injected (tests use a fake generator exposing
    ``generate_classification`` / ``generate_regression``).
    """

    variant_id = "tabpfgen_sgld_prior_resampled"

    def __init__(self, cfg: TabPFGenConfig, generator_factory: Optional[Callable[[TabPFGenConfig], Any]] = None):
        super().__init__(seed=cfg.seed)
        self.cfg = cfg
        # honest variant name: without prior resampling the labels are upstream's (balanced/uniform + argmax)
        self.variant_id = "tabpfgen_sgld_prior_resampled" if cfg.preserve_label_prior else "tabpfgen_sgld_upstream_labels"
        self._generator_factory = generator_factory or _default_generator_factory

        self._X_train: Optional[np.ndarray] = None
        self._y_train: Optional[np.ndarray] = None       # label INDICES (classification) or float target
        self._context_row_positions: Optional[np.ndarray] = None
        self.context_sampling_: Dict[str, Any] = {}
        self._feature_cols: Optional[List[Any]] = None
        self._target_col: Optional[Any] = None
        self._target_dtype: str = "float64"
        self._target_support: Optional[np.ndarray] = None
        self._task: Optional[str] = None
        self._codec: Optional[MixedToContinuousCodec] = None
        self._label_codec: Optional[LabelCodec] = None
        self._class_freqs: Optional[np.ndarray] = None
        self._id_col: Optional[Any] = None
        self._id_factory: Optional[FreshIdFactory] = None
        self._generator: Any = None

        self.role_source_: Optional[str] = None
        self.label_distribution_: Optional[List[List[Any]]] = None
        self.label_report_: Dict[str, Any] = {}
        self.decoding_report_: Dict[Any, Dict[str, Any]] = {}
        self.adaptation_cost_: Dict[str, Any] = {}
        self.last_sample_cost_: Dict[str, Any] = {}
        self.pretrained_identity_: Dict[str, Any] = {}
        self.conditioning_context_: Dict[str, Any] = {}
        self.sgld_settings_: Dict[str, Any] = {}


    # ------------------------------------------------------------------

    def _legacy_auto_task(self, s: pd.Series) -> str:
        warnings.warn(
            f"task='auto': the task for target {s.name!r} is INFERRED from dtype/cardinality. "
            "Pass task='classification'|'regression' (config or fit) to disable inference.",
            UserWarning,
            stacklevel=3,
        )
        if pd.api.types.is_bool_dtype(s) or pd.api.types.is_object_dtype(s) or isinstance(s.dtype, pd.CategoricalDtype):
            return "classification"
        s_num = pd.to_numeric(s, errors="coerce")
        nunique = int(pd.Series(s_num).nunique(dropna=True))
        return "classification" if pd.api.types.is_integer_dtype(s_num) and nunique <= 20 else "regression"

    def fit(
        self,
        data: ArrayLike,
        *,
        continuous_cols: Optional[Sequence[Any]] = None,
        discrete_cols: Optional[Sequence[Any]] = None,
        categorical_cols: Optional[Sequence[Any]] = None,
        target_col: Optional[Any] = None,
        task: Optional[str] = None,
        id_col: Optional[Any] = None,
        schema: Optional[Any] = None,
        transforms: Any = None,
        **kwargs: Any,
    ) -> "TabPFGenGenerative":
        """
        Store the conditioning context (NO training happens here).  ``schema`` contributes its role
        lists when no explicit lists are given; ``transforms`` is accepted and ignored.
        """
        self._reject_unknown_kwargs(kwargs, "TabPFGenGenerative.fit()")
        if not isinstance(data, pd.DataFrame):
            raise ValueError("TabPFGenGenerative requires a pandas DataFrame input (to locate target_col).")
        seed_everything(self.cfg.seed)

        df = data
        cols = list(df.columns)
        self.columns_ = cols

        if target_col is not None and self.cfg.target_col is not None and target_col != self.cfg.target_col:
            raise ValueError(f"fit(target_col={target_col!r}) disagrees with cfg.target_col={self.cfg.target_col!r}.")
        tcol = target_col if target_col is not None else self.cfg.target_col
        if tcol is None and schema is not None:
            tcol = getattr(schema, "target_col", None)
        if tcol is None:
            raise ValueError("TabPFGen needs a target: set cfg.target_col or pass fit(target_col=...).")
        if tcol not in cols:
            raise ValueError(f"target_col='{tcol}' not found in columns: {cols}")

        cfg_task = None if self.cfg.task == "auto" else self.cfg.task
        if task is not None and cfg_task is not None and task != cfg_task:
            raise ValueError(f"fit(task={task!r}) disagrees with cfg.task={cfg_task!r}.")
        declared_task = task if task is not None else cfg_task

        if explicit_roles_given(continuous_cols, discrete_cols, categorical_cols):
            # A. explicit lists: nothing is inferred; an unlisted target needs a declared task
            source = "explicit"
            roles = resolve_column_roles(
                cols, continuous_cols=continuous_cols, discrete_cols=discrete_cols,
                categorical_cols=categorical_cols, target_col=tcol, task=declared_task, id_col=id_col,
            )
        elif schema is not None:
            # B. legacy schema object: role lists from the schema; the target type is not in it
            source = "schema"
            lists = {
                name: [c for c in getattr(schema, name, []) if c in cols]
                for name in ("continuous_cols", "discrete_cols", "categorical_cols")
            }
            if id_col is None and getattr(schema, "id_col", None) in cols:
                id_col = schema.id_col
            listed = {c for group in lists.values() for c in group}
            if tcol not in listed and declared_task is None:
                declared_task = self._legacy_auto_task(df[tcol])
                source = "schema+inferred_target"
            roles = resolve_column_roles(cols, target_col=tcol, task=declared_task, id_col=id_col, **lists)
        else:
            # C. nothing declared: every FEATURE is continuous by declaration; only the task may be inferred
            source = "default_all_continuous"
            if declared_task is None:
                declared_task = self._legacy_auto_task(df[tcol])
                source = "default_all_continuous+inferred_task"
            features = [c for c in cols if c not in (tcol, id_col)]
            roles = resolve_column_roles(cols, continuous_cols=features, target_col=tcol, task=declared_task, id_col=id_col)
        roles = replace(roles, source=source)
        resolved_task = roles.task
        self.roles_ = roles
        self.role_source_ = source
        self._task = resolved_task
        self._target_col = tcol
        self._target_dtype = str(df[tcol].dtype)

        self._id_col = roles.id_col
        self._id_factory = FreshIdFactory.fit(df[self._id_col]) if self._id_col is not None else None

        feature_cols = [c for c in cols if c not in (tcol, self._id_col)]
        if len(feature_cols) < 1:
            raise ValueError("No feature columns after removing target_col.")
        self._feature_cols = feature_cols

        # features -> continuous matrix (one-hot categoricals; NOT z-scored: upstream scales internally)
        self._codec = MixedToContinuousCodec(standardize_numeric=False).fit(df, roles, columns=feature_cols)
        # Checkpoints store a contiguous context. Keep the fit-time context in
        # the same layout: scaler reductions over F/C arrays can differ slightly,
        # which TabPFN can amplify into different regression predictions.
        X = np.ascontiguousarray(self._codec.encode(df), dtype=np.float32)

        self._target_support = None
        if self._task == "classification":
            self._label_codec = LabelCodec().fit(df[tcol])
            y = self._label_codec.encode(df[tcol])
            counts = np.bincount(y, minlength=self._label_codec.n_classes).astype(np.float64)
            self._class_freqs = counts / counts.sum()
            self.label_distribution_ = [[c, int(k)] for c, k in zip(self._label_codec.classes, counts)]
        else:
            self._label_codec = None
            self._class_freqs = None
            self.label_distribution_ = None
            y = pd.to_numeric(df[tcol], errors="coerce").to_numpy(dtype=np.float32, copy=True)
            if np.isnan(y).any():
                raise ValueError("Regression target contains NaNs after numeric conversion.")
            if tcol in roles.discrete:
                self._target_support = np.unique(y.astype(np.float64))

        context = plan_tabpfn_context(len(X), X.shape[1],
                                     self._label_codec.n_classes if self._label_codec is not None else 0,
                                     self.cfg.device)
        positions = sample_context_indices(y, context["n_context_rows"], self._task, self.cfg.seed)
        self._context_row_positions = positions
        self.context_sampling_ = {**context, "seed": int(self.cfg.seed),
            "strategy": ("stratified_target_without_replacement" if self._task == "classification" else
                         "uniform_without_replacement") if context["subsampled"] else "all_rows",
            "positions_hash": hashlib.sha256(positions.astype("<i8").tobytes()).hexdigest()}
        self._X_train = np.ascontiguousarray(X[positions])
        self._y_train = np.ascontiguousarray(y[positions])
        self.fit_info_ = BaselineFitInfo(n_rows=int(df.shape[0]), n_cols=int(df.shape[1]), columns=cols)
        self._finalise_fit_records()
        self._generator = self._generator_factory(self.cfg)
        if hasattr(self._generator, "validate_context"):
            self._generator.validate_context(self._X_train, self._y_train, self._task)
        return self

    def _finalise_fit_records(self) -> None:
        self.n_updates_ = 0   # training-free: fit() stores the context, nothing else
        self.sgld_settings_ = {
            "n_sgld_steps": int(self.cfg.n_sgld_steps),
            "sgld_step_size": float(self.cfg.sgld_step_size),
            "sgld_noise_scale": float(self.cfg.sgld_noise_scale),
            "device": str(self.cfg.device),
            "balance_classes": bool(self.cfg.balance_classes),
            "use_quantiles": bool(self.cfg.use_quantiles),
        }
        self.conditioning_context_ = {
            "n_train_rows": int(self.fit_info_.n_rows) if self.fit_info_ is not None else len(self._X_train),
            "n_context_rows": int(self._X_train.shape[0]),
            "sampling": dict(self.context_sampling_),
            "n_encoded_features": int(self._X_train.shape[1]),
            "feature_columns": list(self._feature_cols),
            "encoded_feature_names": self._codec.encoded_names,
            "target_col": self._target_col,
            "task": self._task,
        }
        self.adaptation_cost_ = {
            "fit_updates": 0,
            "sgld_steps": int(self.cfg.n_sgld_steps),
            "context_rows": int(self._X_train.shape[0]),
            "cost_model": (
                "fit() performs no gradient updates (it stores the context). Every sample() generation call runs "
                "sgld_steps SGLD steps, each with a cdist of the synthetic batch against all context_rows, plus one "
                "TabPFN fit/predict on the context."
            ),
        }
        self.pretrained_identity_ = discover_pretrained_identity()

    # ------------------------------------------------------------------

    def _generate_fn(self) -> GenerateFn:
        gen, X, y = self._generator, self._X_train, self._y_train
        if self._task == "classification":
            return lambda m: gen.generate_classification(X, y, n_samples=int(m), balance_classes=bool(self.cfg.balance_classes))
        return lambda m: gen.generate_regression(X, y, n_samples=int(m), use_quantiles=bool(self.cfg.use_quantiles))

    def sample(self, n: int, seed: Optional[int] = None, **kwargs: Any) -> pd.DataFrame:
        """
        Exactly ``n`` rows, independently of the capped context size. ``seed=None`` falls back to
        ``cfg.seed`` (also used for context selection at fit time); torch + numpy are seeded, and
        the output row selection uses a private ``np.random.Generator`` derived from the same seed.
        """
        self._reject_unknown_kwargs(kwargs, "TabPFGenGenerative.sample()")
        if self._X_train is None or self._y_train is None or self._codec is None or self._generator is None:
            raise RuntimeError("Call fit() before sample().")
        n = validate_n(n)

        eff_seed = int(self.cfg.seed if seed is None else seed)
        seed_everything(eff_seed)
        rng = np.random.default_rng(eff_seed)
        n_feat = int(self._X_train.shape[1])

        if self._task == "classification":
            K = self._label_codec.n_classes
            if self.cfg.preserve_label_prior:
                Xs, ys, report = resample_to_label_prior(
                    self._generate_fn(), n, self._class_freqs, rng=rng, row_multiple=K,
                    oversample=float(self.cfg.prior_oversample), max_rounds=int(self.cfg.max_topup_rounds),
                    n_features=n_feat,
                )
                report["unfilled"] = {
                    repr(self._label_codec.classes[k]): v for k, v in report["unfilled"].items()
                }
            else:
                Xs, ys, report = generate_exact(
                    self._generate_fn(), n, rng=rng, row_multiple=K,
                    max_rounds=int(self.cfg.max_topup_rounds), n_features=n_feat,
                )
                report["prior_preserved"] = False
            report["classes"] = list(self._label_codec.classes)
            report["training_frequencies"] = [float(v) for v in self._class_freqs]
            y_out = self._label_codec.decode(ys)
        else:
            Xs, ys, report = generate_exact(
                self._generate_fn(), n, rng=rng, row_multiple=REGRESSION_ROW_MULTIPLE,
                max_rounds=int(self.cfg.max_topup_rounds), n_features=n_feat,
            )
            y_vals = np.asarray(ys, dtype=np.float64)
            if not np.all(np.isfinite(y_vals)):
                raise RuntimeError("Generator produced a non-finite regression target.")
            target_report = None
            if self._target_support is not None:   # declared-discrete regression target
                y_vals, target_report = nearest_support_decode(y_vals, self._target_support)
            y_out = restore_dtype(y_vals, self._target_dtype)

        out, decoding = self._codec.decode(Xs)
        if self._task != "classification" and target_report is not None:
            decoding[self._target_col] = target_report
        out[self._target_col] = pd.Series(y_out).reset_index(drop=True)
        if self._id_col is not None:
            out[self._id_col] = (self._id_factory or FreshIdFactory()).make(n)
        out = out[self.columns_].reset_index(drop=True)

        self.label_report_ = report
        self.decoding_report_ = decoding
        self.last_sample_cost_ = {
            "generation_calls": int(report["rounds"]),
            "generated_rows": int(report["generated_rows"]),
            "sgld_steps_total": int(report["rounds"]) * int(self.cfg.n_sgld_steps),
            "tabpfn_fit_predict_calls": int(report["rounds"]),
            "context_rows": int(self._X_train.shape[0]),
        }
        if len(out) != n:   # explicit raise: must survive `python -O`
            raise AssertionError(f"TabPFGen wrapper produced {len(out)} rows, expected exactly {n}.")
        return out

    # ------------------------------------------------------------------
    # checkpointing
    # ------------------------------------------------------------------

    def save_checkpoint(self, path: str) -> str:
        """
        Single ``torch.save`` file.  IT CONTAINS THE CONDITIONING (TRAINING) ROWS: TabPFGen has no
        trained parameters, the context is the model, so the checkpoint is as sensitive as the data.
        """
        if self._X_train is None or self._codec is None:
            raise RuntimeError("Call fit() before save_checkpoint().")
        import torch

        state = {
            "format": _CHECKPOINT_FORMAT,
            "format_version": _CHECKPOINT_VERSION,
            "variant_id": self.variant_id,
            "contains_training_rows": True,
            "cfg": asdict(self.cfg),
            "columns": list(self.columns_),
            "feature_cols": list(self._feature_cols),
            "target_col": self._target_col,
            "target_dtype": self._target_dtype,
            "target_support": None if self._target_support is None else [float(v) for v in self._target_support],
            "task": self._task,
            "roles": self.roles_.to_dict() if self.roles_ is not None else None,
            "role_source": self.role_source_,
            "codec": self._codec.state_dict(),
            "label_classes": None if self._label_codec is None else list(self._label_codec.classes),
            "label_dtype": None if self._label_codec is None else self._label_codec.dtype,
            "label_distribution": self.label_distribution_,
            "class_freqs": None if self._class_freqs is None else [float(v) for v in self._class_freqs],
            "id_col": self._id_col,
            "id_factory": self._id_factory.state_dict() if self._id_factory is not None else None,
            "X_train": torch.from_numpy(np.ascontiguousarray(self._X_train)),
            "y_train": torch.from_numpy(np.ascontiguousarray(self._y_train)),
            "context_row_positions": torch.from_numpy(self._context_row_positions),
            "context_sampling": dict(self.context_sampling_),
            "fit_info": asdict(self.fit_info_) if self.fit_info_ is not None else None,
            "sgld_settings": dict(self.sgld_settings_),
            "conditioning_context": dict(self.conditioning_context_),
            "adaptation_cost": dict(self.adaptation_cost_),
            "pretrained_identity": dict(self.pretrained_identity_),
        }
        torch.save(state, path)
        return str(path)

    @classmethod
    def load_checkpoint(
        cls,
        path: str,
        *,
        generator_factory: Optional[Callable[[TabPFGenConfig], Any]] = None,
        device: Optional[str] = None,
        **kwargs: Any,
    ) -> "TabPFGenGenerative":
        """Rebuild the wrapper (context rows included) without touching the original data."""
        cls._reject_unknown_kwargs(kwargs, "TabPFGenGenerative.load_checkpoint()")
        import torch

        state = torch.load(path, map_location="cpu", weights_only=True)
        if state.get("format") != _CHECKPOINT_FORMAT:
            raise ValueError(f"{path!r} is not a TabPFGen wrapper checkpoint.")
        if int(state.get("format_version", -1)) != _CHECKPOINT_VERSION:
            raise ValueError(f"Unsupported TabPFGen checkpoint version: {state.get('format_version')!r}.")

        cfg_dict = dict(state["cfg"])
        if device is not None:
            cfg_dict["device"] = device
        obj = cls(TabPFGenConfig(**cfg_dict), generator_factory=generator_factory)
        obj.columns_ = list(state["columns"])
        obj._feature_cols = list(state["feature_cols"])
        obj._target_col = state["target_col"]
        obj._target_dtype = state["target_dtype"]
        obj._target_support = None if state["target_support"] is None else np.asarray(state["target_support"], dtype=np.float64)
        obj._task = state["task"]
        obj.roles_ = ColumnRoles.from_dict(state["roles"]) if state["roles"] is not None else None
        obj.role_source_ = state["role_source"]
        obj._codec = MixedToContinuousCodec.from_state(state["codec"])
        if state["label_classes"] is not None:
            obj._label_codec = LabelCodec(state["label_classes"], state["label_dtype"])
            obj._class_freqs = np.asarray(state["class_freqs"], dtype=np.float64)
        obj.label_distribution_ = state["label_distribution"]
        obj._id_col = state["id_col"]
        obj._id_factory = FreshIdFactory.from_state(state["id_factory"])
        obj._X_train = state["X_train"].numpy()
        obj._y_train = state["y_train"].numpy()
        positions = state.get("context_row_positions")
        obj._context_row_positions = (positions.numpy() if positions is not None else
                                      np.arange(len(obj._X_train), dtype=np.int64))
        obj.context_sampling_ = dict(state.get("context_sampling", {
            "n_train_rows": len(obj._X_train), "n_context_rows": len(obj._X_train),
            "subsampled": False, "strategy": "legacy_full_context"}))
        if state.get("fit_info") is not None:
            obj.fit_info_ = BaselineFitInfo(**state["fit_info"])
        obj._finalise_fit_records()
        # identity of the prior the checkpoint was CREATED with is kept next to the current one
        obj.pretrained_identity_ = {"at_save": state["pretrained_identity"], "at_load": obj.pretrained_identity_}
        obj._generator = obj._generator_factory(obj.cfg)
        return obj
