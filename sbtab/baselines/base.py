"""
Common interface and column-role handling for the non-SB baselines.

Contract every baseline wrapper follows
---------------------------------------
* ``fit(data, *, continuous_cols=, discrete_cols=, categorical_cols=, target_col=, task=, id_col=)``
  trains on exactly the rows it is given.  No wrapper performs a hidden train/validation
  split, and no wrapper looks at anything but ``data``.
* Column roles come from the EXPLICIT lists.  When at least one explicit list is given, no
  dtype / cardinality inference runs anywhere in the wrapper: every column must be assigned
  a role (or be the ``id_col``), otherwise ``fit`` raises.  The legacy ``schema=`` /
  ``transforms=`` keyword arguments are still accepted for backward compatibility; explicit
  lists take precedence over them.
* The generator models the complete row ``(X, y)``.  ``target_col`` / ``task`` only say which
  role the target column has (``task="classification"`` -> categorical,
  ``task="regression"`` -> continuous unless it is listed in ``discrete_cols``).
* ``sample(n, seed)`` returns EXACTLY ``n`` rows, same columns, same order, same dtypes as
  the frame passed to ``fit``.  Continuous outputs are never clipped to the training range,
  output is never topped up with real rows, and real ids are never emitted (an ``id_col``
  is excluded from modelling and replaced by freshly generated ids that are disjoint from
  the training ids).
* ``save_checkpoint(path)`` / ``load_checkpoint(path)`` restore a fitted wrapper without
  refitting.
* ``variant_id`` names the algorithm variant that is actually implemented and ``n_updates``
  is the number of optimizer updates the last ``fit`` performed (0 for training-free models).
"""

from __future__ import annotations

import math
import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd


ArrayLike = Union[np.ndarray, pd.DataFrame]

VALID_TASKS: Tuple[str, ...] = ("classification", "regression")


@dataclass
class BaselineFitInfo:
    n_rows: int
    n_cols: int
    columns: list[str]


# ----------------------------------------------------------------------
# explicit column roles
# ----------------------------------------------------------------------


@dataclass(frozen=True)
class ColumnRoles:
    """
    Resolved column roles, each list in the column order of the fitted frame.

    ``source`` records where the roles came from:
      * ``"explicit"``                - explicit lists; no inference of any kind ran
      * ``"schema"``                  - legacy schema object, target role given by ``task``
      * ``"schema+inferred_target"``  - legacy schema object, target role inferred from
                                        dtype/cardinality (deprecated behaviour, warns)
      * ``"default_all_continuous"``  - nothing was declared; every column treated as continuous
    """

    continuous: Tuple[Any, ...]
    discrete: Tuple[Any, ...]
    categorical: Tuple[Any, ...]
    target_col: Optional[Any] = None
    task: Optional[str] = None
    id_col: Optional[Any] = None
    source: str = "explicit"

    @property
    def numeric(self) -> Tuple[Any, ...]:
        return (*self.continuous, *self.discrete)

    @property
    def modelled(self) -> Tuple[Any, ...]:
        return (*self.continuous, *self.discrete, *self.categorical)

    def role_of(self, col: Any) -> str:
        if col in self.continuous:
            return "continuous"
        if col in self.discrete:
            return "discrete"
        if col in self.categorical:
            return "categorical"
        if col == self.id_col and col is not None:
            return "id"
        raise KeyError(f"Column {col!r} has no role.")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "continuous": list(self.continuous),
            "discrete": list(self.discrete),
            "categorical": list(self.categorical),
            "target_col": self.target_col,
            "task": self.task,
            "id_col": self.id_col,
            "source": self.source,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ColumnRoles":
        return cls(
            continuous=tuple(d.get("continuous", ())),
            discrete=tuple(d.get("discrete", ())),
            categorical=tuple(d.get("categorical", ())),
            target_col=d.get("target_col"),
            task=d.get("task"),
            id_col=d.get("id_col"),
            source=str(d.get("source", "explicit")),
        )


def explicit_roles_given(
    continuous_cols: Optional[Sequence[Any]],
    discrete_cols: Optional[Sequence[Any]],
    categorical_cols: Optional[Sequence[Any]],
) -> bool:
    """True when the caller passed at least one explicit role list (even an empty one)."""
    return any(v is not None for v in (continuous_cols, discrete_cols, categorical_cols))


def _validate_task(task: Optional[str]) -> Optional[str]:
    if task is None:
        return None
    if task not in VALID_TASKS:
        raise ValueError(f"task must be one of {VALID_TASKS} or None, got {task!r}.")
    return task


def resolve_column_roles(
    columns: Sequence[Any],
    *,
    continuous_cols: Optional[Sequence[Any]] = None,
    discrete_cols: Optional[Sequence[Any]] = None,
    categorical_cols: Optional[Sequence[Any]] = None,
    target_col: Optional[Any] = None,
    task: Optional[str] = None,
    id_col: Optional[Any] = None,
) -> ColumnRoles:
    """
    Resolve explicit column roles.  PURE: it only sees column NAMES, never the data, so it
    cannot infer anything from dtypes or cardinalities.

    Rules
    -----
    * the three lists must be disjoint, duplicate-free and refer to existing columns;
    * ``target_col`` may be listed in one of the lists (that role wins, and must agree with
      ``task`` if ``task`` is given); if it is not listed, ``task`` is REQUIRED and decides:
      ``"classification"`` -> categorical, ``"regression"`` -> continuous;
    * ``id_col`` is never modelled and must not appear in any list;
    * every remaining column must have a role - unassigned columns raise ``ValueError``.
    """
    columns = list(columns)
    task = _validate_task(task)

    groups = {
        "continuous_cols": list(continuous_cols or []),
        "discrete_cols": list(discrete_cols or []),
        "categorical_cols": list(categorical_cols or []),
    }

    seen: Dict[Any, str] = {}
    for gname, cols in groups.items():
        for c in cols:
            if c not in columns:
                raise ValueError(f"{gname} refers to column {c!r} which is not in the data: {columns}")
            if c in seen:
                raise ValueError(
                    f"Column {c!r} is assigned twice ({seen[c]} and {gname}); roles must be disjoint."
                )
            seen[c] = gname

    if id_col is not None:
        if id_col not in columns:
            raise ValueError(f"id_col={id_col!r} is not in the data: {columns}")
        if id_col in seen:
            raise ValueError(f"id_col={id_col!r} must not also be listed in {seen[id_col]}.")
        if id_col == target_col:
            raise ValueError("id_col and target_col must be different columns.")

    if target_col is not None:
        if target_col not in columns:
            raise ValueError(f"target_col={target_col!r} is not in the data: {columns}")
        if target_col in seen:
            listed = seen[target_col]
            if task == "classification" and listed != "categorical_cols":
                raise ValueError(
                    f"task='classification' but target_col={target_col!r} is listed in {listed}; "
                    "a classification target must be categorical."
                )
            if task == "regression" and listed == "categorical_cols":
                raise ValueError(
                    f"task='regression' but target_col={target_col!r} is listed in categorical_cols."
                )
        else:
            if task is None:
                raise ValueError(
                    f"target_col={target_col!r} is not listed in continuous_cols/discrete_cols/"
                    "categorical_cols and task is None. Its role is never inferred from the "
                    "dtype: list it explicitly or pass task='classification'|'regression'."
                )
            gname = "categorical_cols" if task == "classification" else "continuous_cols"
            groups[gname].append(target_col)
            seen[target_col] = gname

    unassigned = [c for c in columns if c not in seen and c != id_col]
    if unassigned:
        raise ValueError(
            f"Columns without an explicit role: {unassigned}. Explicit column roles were given, "
            "so nothing is inferred from dtype/cardinality - assign every column to "
            "continuous_cols, discrete_cols or categorical_cols (or make it target_col/id_col)."
        )

    def ordered(gname: str) -> Tuple[Any, ...]:
        members = set(groups[gname])
        return tuple(c for c in columns if c in members)

    resolved_task = task
    if resolved_task is None and target_col is not None:
        resolved_task = "classification" if seen[target_col] == "categorical_cols" else "regression"

    return ColumnRoles(
        continuous=ordered("continuous_cols"),
        discrete=ordered("discrete_cols"),
        categorical=ordered("categorical_cols"),
        target_col=target_col,
        task=resolved_task,
        id_col=id_col,
        source="explicit",
    )


def legacy_target_role(df: pd.DataFrame, target_col: Any, task: Optional[str]) -> Tuple[str, bool]:
    """
    Role of the target column on the LEGACY ``schema=`` path (a ``TabularSchema`` does not
    carry the target type).  Returns ``(role, inferred)``.

    With ``task`` given nothing is inferred.  Without it the deprecated dtype/cardinality
    heuristic (``sbtab.data.schema.classify_feature_type``) is used and a warning is emitted -
    an integer class label with >= 20 classes, or a float-coded label, is silently treated as
    continuous by that heuristic, which is exactly why the explicit API exists.
    """
    task = _validate_task(task)
    if task == "classification":
        return "categorical", False
    if task == "regression":
        return "continuous", False

    from sbtab.data.schema import classify_feature_type  # lazy: only the legacy path needs it

    role = classify_feature_type(df[target_col])
    warnings.warn(
        f"The role of target column {target_col!r} was INFERRED from dtype/cardinality "
        f"({role!r}). Pass task='classification'|'regression' or explicit column lists to fit() "
        "to disable inference.",
        UserWarning,
        stacklevel=3,
    )
    return role, True


# ----------------------------------------------------------------------
# small pure helpers shared by the wrappers
# ----------------------------------------------------------------------


def seed_everything(seed: int) -> None:
    """Seed numpy and torch (CPU + all CUDA devices) global generators."""
    import torch  # local import keeps ``import sbtab.baselines.base`` torch-free

    seed = int(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)


def validate_n(n: Any) -> int:
    if isinstance(n, bool) or not isinstance(n, (int, np.integer)):
        raise TypeError(f"n must be an integer, got {type(n).__name__}.")
    if n <= 0:
        raise ValueError("n must be positive.")
    return int(n)


def to_python_scalar(v: Any) -> Any:
    """numpy scalar -> python scalar (so checkpoints contain only plain python + tensors)."""
    if isinstance(v, np.generic):
        return v.item()
    return v


def largest_remainder_allocation(n: int, weights: Iterable[float]) -> np.ndarray:
    """
    Split ``n`` items across classes proportionally to ``weights`` (Hamilton / largest
    remainder method).  The result sums to exactly ``n`` and every entry is within 1 of the
    exact quota ``n * w_k / sum(w)``.  Ties are broken by class index, so it is deterministic.
    """
    n = int(n)
    if n < 0:
        raise ValueError("n must be non-negative.")
    w = np.asarray(list(weights), dtype=np.float64)
    if w.ndim != 1 or w.size == 0:
        raise ValueError("weights must be a non-empty 1-D sequence.")
    if not np.all(np.isfinite(w)) or np.any(w < 0) or w.sum() <= 0:
        raise ValueError("weights must be finite, non-negative and not all zero.")
    quota = n * w / w.sum()
    counts = np.floor(quota + 1e-12).astype(np.int64)
    remainder = n - int(counts.sum())
    if remainder > 0:
        frac = quota - counts
        # stable argsort on -frac => largest remainder first, lowest index wins ties
        order = np.argsort(-frac, kind="stable")
        counts[order[:remainder]] += 1
    assert int(counts.sum()) == n
    return counts


def empirical_distribution(values: Sequence[Any]) -> List[List[Any]]:
    """
    Empirical distribution of a label column as ``[[value, count], ...]`` (python scalars,
    sorted by value; a list of pairs rather than a dict so that it survives JSON / torch
    ``weights_only`` checkpoints with non-string labels).
    """
    s = pd.Series(np.asarray(values, dtype=object))
    vc = s.value_counts(dropna=False)
    items = [(to_python_scalar(k), int(c)) for k, c in vc.items()]
    try:
        items.sort(key=lambda kv: kv[0])
    except TypeError:
        items.sort(key=lambda kv: str(kv[0]))
    return [[k, c] for k, c in items]


class FreshIdFactory:
    """
    Generates ids for synthetic rows that are guaranteed NOT to be training ids.

    Integer-like training ids -> ``max(train_id) + 1 + arange(n)``.
    Anything else             -> ``"<prefix><i>"`` with a prefix no training id starts with.
    Only the max / the prefix are stored, never the training ids themselves.
    """

    def __init__(self, kind: str = "int", start: int = 0, prefix: str = "synthetic_", dtype: str = "int64"):
        self.kind = kind
        self.start = int(start)
        self.prefix = str(prefix)
        self.dtype = str(dtype)

    @classmethod
    def fit(cls, ids: pd.Series) -> "FreshIdFactory":
        non_null = ids.dropna()
        if len(non_null) and pd.api.types.is_numeric_dtype(non_null.dtype) and not pd.api.types.is_bool_dtype(
            non_null.dtype
        ):
            vals = non_null.to_numpy(dtype=np.float64)
            if np.all(np.isfinite(vals)) and np.all(vals == np.round(vals)):
                return cls(kind="int", start=int(vals.max()) + 1, dtype=str(ids.dtype))
        prefix = "synthetic_"
        as_str = non_null.astype(str)
        while len(as_str) and as_str.str.startswith(prefix).any():
            prefix = "_" + prefix
        return cls(kind="str", prefix=prefix, dtype="object")

    def make(self, n: int) -> np.ndarray:
        if self.kind == "int":
            out = np.arange(self.start, self.start + int(n), dtype=np.int64)
            try:
                return out.astype(self.dtype)
            except (TypeError, ValueError):
                return out
        return np.asarray([f"{self.prefix}{i}" for i in range(int(n))], dtype=object)

    def state_dict(self) -> Dict[str, Any]:
        return {"kind": self.kind, "start": self.start, "prefix": self.prefix, "dtype": self.dtype}

    @classmethod
    def from_state(cls, state: Optional[Dict[str, Any]]) -> Optional["FreshIdFactory"]:
        if state is None:
            return None
        return cls(**state)


def next_multiple(n: int, k: int) -> int:
    """Smallest multiple of ``k`` that is >= ``n`` (``k`` >= 1)."""
    n, k = int(n), int(k)
    if k < 1:
        raise ValueError("k must be >= 1.")
    return int(math.ceil(n / k) * k) if n > 0 else 0


# ----------------------------------------------------------------------
# abstract interface
# ----------------------------------------------------------------------


class BaselineGenerativeModel(ABC):
    """
    Unified API for baseline generative models (see the module docstring for the contract).

    - ``fit(data, *, continuous_cols=..., discrete_cols=..., categorical_cols=..., target_col=..., task=...)``
    - ``sample(n, seed)`` -> exactly ``n`` rows, same columns / order / dtypes as the fit frame
    - ``save_checkpoint(path)`` / ``load_checkpoint(path)`` -> restore without refit
    - ``variant_id`` -> honest name of the implemented algorithm variant
    - ``n_updates`` -> optimizer updates performed by the last ``fit`` (0 if training-free)
    """

    #: honest identifier of the implemented algorithm variant; subclasses override it
    variant_id: str = "unspecified"

    def __init__(self, seed: int = 0):
        self.seed = int(seed)
        self.columns_: Optional[list[str]] = None
        self.fit_info_: Optional[BaselineFitInfo] = None
        self.roles_: Optional[ColumnRoles] = None
        self.n_updates_: Optional[int] = None

    @abstractmethod
    def fit(self, data: ArrayLike, **kwargs: Any) -> "BaselineGenerativeModel":
        raise NotImplementedError

    @abstractmethod
    def sample(self, n: int, seed: Optional[int] = None, **kwargs: Any) -> ArrayLike:
        raise NotImplementedError

    @abstractmethod
    def save_checkpoint(self, path: str) -> str:
        """Persist everything ``sample`` needs. Returns the path written."""
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def load_checkpoint(cls, path: str, **kwargs: Any) -> "BaselineGenerativeModel":
        """Rebuild a fitted wrapper from ``save_checkpoint`` output WITHOUT refitting."""
        raise NotImplementedError

    @property
    def n_updates(self) -> int:
        """Optimizer updates performed by the last ``fit`` (0 for training-free models)."""
        if self.n_updates_ is None:
            raise RuntimeError("Model is not fitted.")
        return int(self.n_updates_)

    def get_fit_info(self) -> BaselineFitInfo:
        if self.fit_info_ is None:
            raise RuntimeError("Model is not fitted.")
        return self.fit_info_

    # ----- helpers -----

    @staticmethod
    def _reject_unknown_kwargs(kwargs: Dict[str, Any], where: str) -> None:
        # A misspelled role list (``categorical_col=``) silently ignored would put the wrapper
        # back on an inference / all-continuous path, so unknown keywords are an error.
        if kwargs:
            raise TypeError(f"{where} got unexpected keyword argument(s): {sorted(kwargs)}")

    def _to_numpy_and_columns(self, data: ArrayLike) -> tuple[np.ndarray, Optional[list[str]]]:
        if isinstance(data, pd.DataFrame):
            return data.to_numpy(copy=True), list(data.columns)
        if isinstance(data, np.ndarray):
            return data, None
        raise TypeError(f"Unsupported data type: {type(data)}")

    def _format_output(self, x: np.ndarray) -> ArrayLike:
        if self.columns_ is not None:
            return pd.DataFrame(x, columns=self.columns_)
        return x
