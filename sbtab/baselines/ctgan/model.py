"""
CTGAN wrapper over SDV's ``CTGANSynthesizer`` (imported lazily - ``import sbtab.baselines``
works without ``sdv``).

Column roles
------------
``fit(data, *, continuous_cols=, discrete_cols=, categorical_cols=, target_col=, task=, id_col=)``.
With explicit lists the frame is modelled IN THE REPRESENTATION GIVEN (no inverse transform, no
dtype/cardinality inference) and the SDV metadata is built from the lists alone:

    categorical (incl. a classification target, also when stored as integers) -> "categorical"
    continuous and discrete (incl. a regression target)                        -> "numerical"

``build_sdv_metadata_dict`` is a pure function returning a plain dict, so this mapping is
testable without ``sdv``.  The legacy ``schema=`` / ``transforms=`` path is still accepted
(explicit lists win): the processed frame is inverse-transformed to a raw-like table for SDV and
the sample is transformed back - row-DROPPING steps of the pipeline are never re-applied to
generated rows.

Protocol guarantees
-------------------
* exactly ``n`` rows (asserted), same columns / order / dtypes as the fit frame;
* ``enforce_min_max_values`` defaults to ``False``: SDV's ``True`` CLIPS every numerical output to
  the observed training range, which is an unrequested modification of the model;
* discrete numeric columns are decoded to the nearest training-support value
  (``decoding_report_`` has the pre-decoding statistics);
* an ``id_col`` is never modelled and real training ids are never emitted;
* the label prior of a classification target is whatever CTGAN learned for that categorical
  column (CTGAN's training-by-sampling uses log-frequencies when ``log_frequency=True``); the
  empirical training label distribution is recorded in ``label_distribution_``.

Seeding
-------
``fit`` seeds torch + numpy globally (``ctgan`` trains from the global generators).  ``sample``
with a seed seeds torch + numpy, calls the synthesizer's ``reset_sampling()`` and, when the
synthesizer exposes it, ``_set_random_state(seed)`` - without the latter SDV re-seeds with its
own fixed constant and every seed would return the same rows.
"""

from __future__ import annotations

import os
import pickle
import warnings
from dataclasses import asdict, dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from sbtab.baselines.base import (
    ArrayLike,
    BaselineFitInfo,
    BaselineGenerativeModel,
    ColumnRoles,
    FreshIdFactory,
    empirical_distribution,
    explicit_roles_given,
    legacy_target_role,
    resolve_column_roles,
    seed_everything,
    validate_n,
)
from sbtab.baselines.encoding import fit_vocabulary, nearest_support_decode, numeric_matrix, restore_dtype

_CHECKPOINT_FORMAT = "sbtab.baselines.ctgan"
_CHECKPOINT_VERSION = 1
_SYNTH_FILE = "synthesizer.pkl"
_STATE_FILE = "wrapper_state.pt"
_TRANSFORMS_FILE = "transforms.pkl"
SDV_TABLE_NAME = "table"


# ----------------------------------------------------------------------
# pure, sdv-free pieces
# ----------------------------------------------------------------------


def validate_ctgan_batch(batch_size: int, pac: int) -> None:
    """``ctgan`` asserts ``batch_size % 2 == 0`` and the discriminator needs ``batch_size % pac == 0``."""
    if isinstance(batch_size, bool) or int(batch_size) != batch_size or int(batch_size) < 2:
        raise ValueError(f"batch_size must be an integer >= 2, got {batch_size!r}.")
    if isinstance(pac, bool) or int(pac) != pac or int(pac) < 1:
        raise ValueError(f"pac must be an integer >= 1, got {pac!r}.")
    if int(batch_size) % 2 != 0:
        raise ValueError(f"CTGAN requires an even batch_size, got {batch_size}.")
    if int(batch_size) % int(pac) != 0:
        raise ValueError(f"CTGAN requires batch_size % pac == 0, got batch_size={batch_size}, pac={pac}.")


def build_sdv_metadata_dict(
    columns: Sequence[Any],
    *,
    categorical_cols: Sequence[Any],
    numerical_cols: Sequence[Any],
    table_name: str = SDV_TABLE_NAME,
) -> Dict[str, Any]:
    """
    SDV single-table metadata (``Metadata.load_from_dict`` format) built ONLY from the explicit
    role lists - never from dtypes.  Every modelled column must be in exactly one list.
    """
    categorical, numerical = set(categorical_cols), set(numerical_cols)
    both = categorical & numerical
    if both:
        raise ValueError(f"Columns declared both categorical and numerical: {sorted(map(str, both))}")
    spec: Dict[str, Dict[str, str]] = {}
    for c in columns:
        if c in categorical:
            spec[str(c)] = {"sdtype": "categorical"}
        elif c in numerical:
            spec[str(c)] = {"sdtype": "numerical"}
        else:
            raise ValueError(f"Column {c!r} has no declared role; cannot build SDV metadata.")
    if len(spec) != len(list(columns)):
        raise ValueError("Column names must be unique after str() conversion.")
    return {
        "METADATA_SPEC_VERSION": "V1",
        "tables": {table_name: {"columns": spec}},
        "relationships": [],
    }


def ctgan_update_counts(n_rows: int, batch_size: int, epochs: int, discriminator_steps: int) -> Dict[str, int]:
    """Declared update counts of ``ctgan.CTGAN.fit``: ``steps_per_epoch = max(n_rows // batch_size, 1)``."""
    steps_per_epoch = max(int(n_rows) // int(batch_size), 1)
    gen = int(epochs) * steps_per_epoch
    return {
        "steps_per_epoch": steps_per_epoch,
        "generator_updates": gen,
        "discriminator_updates": gen * int(discriminator_steps),
    }


@dataclass
class CTGANConfig:
    """Configuration for the SDV CTGAN synthesizer."""

    embedding_dim: int = 128
    generator_dim: Tuple[int, ...] = (256, 256)
    discriminator_dim: Tuple[int, ...] = (256, 256)

    generator_lr: float = 2e-4
    generator_decay: float = 1e-6
    discriminator_lr: float = 2e-4
    discriminator_decay: float = 1e-6
    discriminator_steps: int = 1

    batch_size: int = 500
    epochs: int = 300
    pac: int = 10
    log_frequency: bool = True

    enforce_rounding: bool = True
    # False on purpose: True CLIPS numerical outputs to the observed training min/max.
    enforce_min_max_values: bool = False
    locales: Tuple[str, ...] = ("en_US",)

    enable_gpu: bool = True
    seed: int = 42
    verbose: bool = False

    def __post_init__(self) -> None:
        validate_ctgan_batch(self.batch_size, self.pac)
        if int(self.epochs) < 1:
            raise ValueError("epochs must be >= 1.")
        if int(self.discriminator_steps) < 1:
            raise ValueError("discriminator_steps must be >= 1.")
        self.generator_dim = tuple(int(v) for v in self.generator_dim)
        self.discriminator_dim = tuple(int(v) for v in self.discriminator_dim)
        self.locales = tuple(self.locales)


def _default_synthesizer_factory(metadata_dict: Dict[str, Any], cfg: CTGANConfig) -> Any:
    """Build the real SDV synthesizer (lazy import)."""
    try:
        from sdv.single_table import CTGANSynthesizer
    except Exception as e:  # pragma: no cover - depends on the environment
        raise ImportError(
            "CTGANWrapper requires the SDV library with CTGANSynthesizer. Install it with: pip install sdv"
        ) from e

    try:
        from sdv.metadata import Metadata

        metadata = Metadata.load_from_dict(metadata_dict)
    except ImportError:  # pragma: no cover - older SDV
        from sdv.metadata import SingleTableMetadata

        table = next(iter(metadata_dict["tables"].values()))
        metadata = SingleTableMetadata.load_from_dict(
            {"METADATA_SPEC_VERSION": "SINGLE_TABLE_V1", "columns": table["columns"]}
        )
    metadata.validate()

    return CTGANSynthesizer(
        metadata,
        enforce_rounding=bool(cfg.enforce_rounding),
        enforce_min_max_values=bool(cfg.enforce_min_max_values),
        epochs=int(cfg.epochs),
        verbose=bool(cfg.verbose),
        embedding_dim=int(cfg.embedding_dim),
        generator_dim=tuple(int(v) for v in cfg.generator_dim),
        discriminator_dim=tuple(int(v) for v in cfg.discriminator_dim),
        generator_lr=float(cfg.generator_lr),
        generator_decay=float(cfg.generator_decay),
        discriminator_lr=float(cfg.discriminator_lr),
        discriminator_decay=float(cfg.discriminator_decay),
        discriminator_steps=int(cfg.discriminator_steps),
        batch_size=int(cfg.batch_size),
        log_frequency=bool(cfg.log_frequency),
        pac=int(cfg.pac),
        enable_gpu=bool(cfg.enable_gpu),
        locales=list(cfg.locales),
    )


def _default_synthesizer_loader(filepath: str) -> Any:
    from sdv.single_table import CTGANSynthesizer  # lazy

    return CTGANSynthesizer.load(filepath)


class CTGANWrapper(BaselineGenerativeModel):
    """
    SDV CTGAN wrapper; see the module docstring for the contract.

    ``synthesizer_factory(metadata_dict, cfg)`` can be injected (tests use a tiny fake); by default
    the real ``sdv.single_table.CTGANSynthesizer`` is built lazily.
    """

    variant_id = "ctgan_sdv"
    max_topup_rounds = 5

    def __init__(self, cfg: CTGANConfig, synthesizer_factory: Optional[Callable[[Dict[str, Any], CTGANConfig], Any]] = None):
        super().__init__(seed=cfg.seed)
        self.cfg = cfg
        self._synthesizer_factory = synthesizer_factory or _default_synthesizer_factory

        self._fitted = False
        self.columns_: Optional[List[Any]] = None          # columns of the DataFrame passed to fit()
        self._dtypes: Dict[Any, str] = {}
        self._schema: Optional[Any] = None
        self._fitted_transforms: Optional[Any] = None

        self._model: Any = None
        self.metadata_dict_: Optional[Dict[str, Any]] = None

        self._train_repr: Optional[str] = None             # "given" | "raw" | "processed"
        self._raw_model_cols: List[Any] = []               # columns CTGAN actually models
        self._raw_return_cols: List[Any] = []              # raw-like columns incl. id if present
        self._raw_dtypes: Dict[Any, str] = {}

        self._id_col: Optional[Any] = None
        self._id_factory: Optional[FreshIdFactory] = None

        self._categorical_model_cols: List[Any] = []
        self._numerical_model_cols: List[Any] = []
        self._discrete_supports: Dict[Any, np.ndarray] = {}
        self._vocab: Dict[Any, List[Any]] = {}

        self.role_source_: Optional[str] = None
        self.label_distribution_: Optional[List[List[Any]]] = None
        self.decoding_report_: Dict[Any, Dict[str, Any]] = {}
        self.sample_report_: Dict[str, Any] = {}
        self.budget_: Dict[str, int] = {}

    # ------------------------------------------------------------------
    # role resolution
    # ------------------------------------------------------------------

    def _roles_from_schema(self, raw_like: pd.DataFrame, schema: Any, task: Optional[str]) -> ColumnRoles:
        columns = list(raw_like.columns)
        continuous = [c for c in getattr(schema, "continuous_cols", []) if c in columns]
        discrete = [c for c in getattr(schema, "discrete_cols", []) if c in columns]
        categorical = [c for c in getattr(schema, "categorical_cols", []) if c in columns]
        target = getattr(schema, "target_col", None)
        target = target if target in columns else None
        id_col = getattr(schema, "id_col", None)
        id_col = id_col if id_col in columns else None

        inferred = False
        if target is not None and target not in (*continuous, *discrete, *categorical):
            role, inferred = legacy_target_role(raw_like, target, task)
            {"categorical": categorical, "discrete": discrete}.get(role, continuous).append(target)

        # legacy behaviour: columns the schema does not mention are returned but not modelled -> error now
        roles = resolve_column_roles(
            columns, continuous_cols=continuous, discrete_cols=discrete, categorical_cols=categorical,
            target_col=target, id_col=id_col,
        )
        return ColumnRoles(
            continuous=roles.continuous, discrete=roles.discrete, categorical=roles.categorical,
            target_col=target, task=task or roles.task, id_col=id_col,
            source="schema+inferred_target" if inferred else "schema",
        )

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------

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
    ) -> "CTGANWrapper":
        """Train on exactly the rows of ``data`` (no internal split)."""
        self._reject_unknown_kwargs(kwargs, "CTGANWrapper.fit()")
        if not isinstance(data, pd.DataFrame):
            raise ValueError("CTGANWrapper expects a pandas DataFrame (column roles refer to column names).")
        if len(data) == 0:
            raise ValueError("Cannot fit on an empty frame.")

        # C2: ctgan trains from the GLOBAL torch / numpy generators.
        seed_everything(self.cfg.seed)

        self.columns_ = list(data.columns)
        self._dtypes = {c: str(data[c].dtype) for c in data.columns}

        if explicit_roles_given(continuous_cols, discrete_cols, categorical_cols):
            if transforms is not None:
                warnings.warn(
                    "Explicit column roles were given: `transforms` is ignored and the frame is modelled "
                    "in the representation passed to fit().",
                    UserWarning,
                    stacklevel=2,
                )
            roles = resolve_column_roles(
                self.columns_, continuous_cols=continuous_cols, discrete_cols=discrete_cols,
                categorical_cols=categorical_cols, target_col=target_col, task=task, id_col=id_col,
            )
            raw_like = data
            self._train_repr = "given"
            self._fitted_transforms = None
            self._schema = None
        elif schema is not None:
            self._schema = schema
            self._fitted_transforms = transforms
            if transforms is not None:
                raw_like = transforms.inverse_transform(data)
                self._train_repr = "processed"
            else:
                raw_like = data.copy()
                self._train_repr = "raw"
            if hasattr(schema, "validate"):
                schema.validate(raw_like)
            roles = self._roles_from_schema(raw_like, schema, task)
        else:
            raise ValueError(
                "Column roles are required: pass continuous_cols= / discrete_cols= / categorical_cols= "
                "(+ target_col=, task=) or, for backward compatibility, schema=."
            )

        self.roles_ = roles
        self.role_source_ = roles.source
        self._id_col = roles.id_col
        self._id_factory = FreshIdFactory.fit(raw_like[self._id_col]) if self._id_col is not None else None

        self._raw_return_cols = list(raw_like.columns)
        self._raw_model_cols = [c for c in raw_like.columns if c != self._id_col]
        self._numerical_model_cols = [c for c in self._raw_model_cols if c in set(roles.numeric)]
        self._categorical_model_cols = [c for c in self._raw_model_cols if c in set(roles.categorical)]
        self._raw_dtypes = {c: str(raw_like[c].dtype) for c in self._raw_model_cols}

        x_num = numeric_matrix(raw_like, self._numerical_model_cols)
        self._discrete_supports = {
            c: np.unique(x_num[:, self._numerical_model_cols.index(c)])
            for c in self._numerical_model_cols if c in set(roles.discrete)
        }
        self._vocab = {c: fit_vocabulary(raw_like[c], c) for c in self._categorical_model_cols}

        self.label_distribution_ = None
        if roles.target_col is not None and roles.target_col in roles.categorical:
            self.label_distribution_ = empirical_distribution(raw_like[roles.target_col].tolist())

        # C1: metadata from the explicit roles only
        self.metadata_dict_ = build_sdv_metadata_dict(
            self._raw_model_cols,
            categorical_cols=self._categorical_model_cols,
            numerical_cols=self._numerical_model_cols,
        )

        df_model = raw_like[self._raw_model_cols].copy()
        df_model.columns = [str(c) for c in df_model.columns]
        synthesizer = self._synthesizer_factory(self.metadata_dict_, self.cfg)
        synthesizer.fit(df_model)
        self._model = synthesizer

        self.budget_ = ctgan_update_counts(len(df_model), self.cfg.batch_size, self.cfg.epochs, self.cfg.discriminator_steps)
        self.n_updates_ = int(self.budget_["generator_updates"])
        self.fit_info_ = BaselineFitInfo(n_rows=int(data.shape[0]), n_cols=int(data.shape[1]), columns=list(self.columns_))
        self._fitted = True
        return self

    # ------------------------------------------------------------------

    def _seed_sampling(self, seed: int) -> None:
        seed_everything(seed)
        report = {"reset_sampling": False, "set_random_state": False}
        reset = getattr(self._model, "reset_sampling", None)
        if callable(reset):
            reset()
            report["reset_sampling"] = True
        setter = getattr(self._model, "_set_random_state", None)
        if callable(setter):
            setter(int(seed))
            report["set_random_state"] = True
        self.sample_report_["seeding"] = report

    def _sample_exact(self, n: int) -> pd.DataFrame:
        """Exactly ``n`` generated rows: top up (bounded) if the synthesizer returns fewer, truncate if more."""
        parts: List[pd.DataFrame] = []
        have = 0
        rounds = 0
        while have < n and rounds <= self.max_topup_rounds:
            part = self._model.sample(num_rows=int(n - have))
            rounds += 1
            if part is None or len(part) == 0:
                continue
            parts.append(part)
            have += len(part)
        self.sample_report_["synthesizer_calls"] = rounds
        if have < n:
            raise RuntimeError(
                f"CTGAN synthesizer produced {have} rows after {rounds} calls, expected {n}. "
                "Real rows are never used to fill the gap."
            )
        return pd.concat(parts, axis=0, ignore_index=True).iloc[:n].reset_index(drop=True)

    def _postprocess_raw_sample(self, raw_synth: pd.DataFrame) -> pd.DataFrame:
        out = pd.DataFrame(index=np.arange(len(raw_synth)))
        report: Dict[Any, Dict[str, Any]] = {}
        by_str = {str(c): c for c in self._raw_model_cols}
        missing = [s for s in by_str if s not in raw_synth.columns]
        if missing:
            raise RuntimeError(f"CTGAN synthetic table is missing expected columns: {missing}")

        for s_name, col in by_str.items():
            values = raw_synth[s_name]
            if col in self._discrete_supports:
                vals = pd.to_numeric(values, errors="coerce").to_numpy(dtype=np.float64)
                decoded, report[col] = nearest_support_decode(vals, self._discrete_supports[col])
                out[col] = restore_dtype(decoded, self._raw_dtypes[col])
            elif col in self._vocab:
                vocab = set(self._vocab[col])
                unknown = [v for v in pd.unique(values) if (v.item() if isinstance(v, np.generic) else v) not in vocab]
                if unknown:
                    raise RuntimeError(f"CTGAN produced categories outside the training vocabulary of {col!r}: {unknown[:5]}")
                out[col] = restore_dtype(values.to_numpy(), self._raw_dtypes[col])
            else:
                # continuous: as generated - no clipping, no snapping
                out[col] = restore_dtype(pd.to_numeric(values, errors="coerce").to_numpy(), self._raw_dtypes[col])

        if self._id_col is not None:
            # C4: never resample real training ids
            out[self._id_col] = (self._id_factory or FreshIdFactory()).make(len(out))
        self.decoding_report_ = report
        return out[self._raw_return_cols]

    def _apply_transforms_without_row_drops(self, raw_synth: pd.DataFrame) -> pd.DataFrame:
        """Legacy path: re-apply the fitted pipeline to generated rows, skipping any row-dropping step."""
        pipe = self._fitted_transforms
        steps = getattr(pipe, "transforms", None)
        skipped: List[str] = []
        if isinstance(steps, (list, tuple)):
            x = raw_synth.copy()
            for step in steps:
                y = step.transform(x)
                if len(y) != len(x):
                    skipped.append(getattr(step, "name", type(step).__name__))
                    continue
                x = y
        else:
            x = pipe.transform(raw_synth)
        self.sample_report_["skipped_row_dropping_transforms"] = skipped
        return x.reset_index(drop=True)

    def sample(self, n: int, seed: Optional[int] = None, **kwargs: Any) -> pd.DataFrame:
        """Exactly ``n`` rows in the representation / columns / order / dtypes of the fit frame."""
        self._reject_unknown_kwargs(kwargs, "CTGANWrapper.sample()")
        if not self._fitted or self._model is None:
            raise RuntimeError("Call fit() before sample().")
        n = validate_n(n)

        self.sample_report_ = {}
        if seed is not None:
            self._seed_sampling(int(seed))

        out = self._postprocess_raw_sample(self._sample_exact(n))

        if self._train_repr == "processed":
            if self._fitted_transforms is None:
                raise RuntimeError("Processed training representation requires fitted transforms.")
            out = self._apply_transforms_without_row_drops(out)
            if self._id_col is not None and self._id_col not in out.columns and self._id_col in (self.columns_ or []):
                out[self._id_col] = (self._id_factory or FreshIdFactory()).make(len(out))

        missing = [c for c in (self.columns_ or []) if c not in out.columns]
        if missing:
            raise RuntimeError(f"Synthetic data is missing expected columns: {missing}")
        out = out[self.columns_].reset_index(drop=True)

        # C5: exact row count is part of the contract (explicit raise: must survive `python -O`)
        if len(out) != n:
            raise AssertionError(f"CTGAN wrapper produced {len(out)} rows, expected exactly {n}.")
        return out

    # ------------------------------------------------------------------
    # checkpointing
    # ------------------------------------------------------------------

    def save_checkpoint(self, path: str) -> str:
        """
        ``path`` is a DIRECTORY: ``synthesizer.pkl`` (the synthesizer's own ``save``),
        ``wrapper_state.pt`` (plain python, ``weights_only`` loadable) and - legacy path only -
        ``transforms.pkl`` (pickle of the fitted pipeline).
        """
        if not self._fitted or self._model is None:
            raise RuntimeError("Call fit() before save_checkpoint().")
        import torch

        os.makedirs(path, exist_ok=True)
        self._model.save(os.path.join(path, _SYNTH_FILE))

        transforms_saved = False
        if self._fitted_transforms is not None:
            with open(os.path.join(path, _TRANSFORMS_FILE), "wb") as fh:
                pickle.dump(self._fitted_transforms, fh)
            transforms_saved = True

        cfg_dict = asdict(self.cfg)
        for k in ("generator_dim", "discriminator_dim", "locales"):
            cfg_dict[k] = list(cfg_dict[k])
        state = {
            "format": _CHECKPOINT_FORMAT,
            "format_version": _CHECKPOINT_VERSION,
            "variant_id": self.variant_id,
            "cfg": cfg_dict,
            "columns": list(self.columns_),
            "dtypes": [[c, self._dtypes[c]] for c in self.columns_],
            "roles": self.roles_.to_dict() if self.roles_ is not None else None,
            "train_repr": self._train_repr,
            "raw_model_cols": list(self._raw_model_cols),
            "raw_return_cols": list(self._raw_return_cols),
            "raw_dtypes": [[c, self._raw_dtypes[c]] for c in self._raw_model_cols],
            "numerical_model_cols": list(self._numerical_model_cols),
            "categorical_model_cols": list(self._categorical_model_cols),
            "discrete_supports": [[c, [float(v) for v in s]] for c, s in self._discrete_supports.items()],
            "vocab": [[c, list(v)] for c, v in self._vocab.items()],
            "metadata_dict": self.metadata_dict_,
            "label_distribution": self.label_distribution_,
            "id_col": self._id_col,
            "id_factory": self._id_factory.state_dict() if self._id_factory is not None else None,
            "budget": dict(self.budget_),
            "n_updates": int(self.n_updates_),
            "fit_info": asdict(self.fit_info_) if self.fit_info_ is not None else None,
            "transforms_saved": transforms_saved,
        }
        torch.save(state, os.path.join(path, _STATE_FILE))
        return str(path)

    @classmethod
    def load_checkpoint(
        cls,
        path: str,
        *,
        synthesizer_loader: Optional[Callable[[str], Any]] = None,
        synthesizer_factory: Optional[Callable[[Dict[str, Any], CTGANConfig], Any]] = None,
        transforms: Any = None,
        **kwargs: Any,
    ) -> "CTGANWrapper":
        """Rebuild a fitted wrapper without refitting. ``transforms`` overrides the pickled legacy pipeline."""
        cls._reject_unknown_kwargs(kwargs, "CTGANWrapper.load_checkpoint()")
        import torch

        state = torch.load(os.path.join(path, _STATE_FILE), map_location="cpu", weights_only=True)
        if state.get("format") != _CHECKPOINT_FORMAT:
            raise ValueError(f"{path!r} is not a CTGAN wrapper checkpoint.")
        if int(state.get("format_version", -1)) != _CHECKPOINT_VERSION:
            raise ValueError(f"Unsupported CTGAN checkpoint version: {state.get('format_version')!r}.")

        obj = cls(CTGANConfig(**state["cfg"]), synthesizer_factory=synthesizer_factory)
        obj.columns_ = list(state["columns"])
        obj._dtypes = {c: d for c, d in state["dtypes"]}
        obj.roles_ = ColumnRoles.from_dict(state["roles"]) if state["roles"] is not None else None
        obj.role_source_ = obj.roles_.source if obj.roles_ is not None else None
        obj._train_repr = state["train_repr"]
        obj._raw_model_cols = list(state["raw_model_cols"])
        obj._raw_return_cols = list(state["raw_return_cols"])
        obj._raw_dtypes = {c: d for c, d in state["raw_dtypes"]}
        obj._numerical_model_cols = list(state["numerical_model_cols"])
        obj._categorical_model_cols = list(state["categorical_model_cols"])
        obj._discrete_supports = {c: np.asarray(v, dtype=np.float64) for c, v in state["discrete_supports"]}
        obj._vocab = {c: list(v) for c, v in state["vocab"]}
        obj.metadata_dict_ = state["metadata_dict"]
        obj.label_distribution_ = state["label_distribution"]
        obj._id_col = state["id_col"]
        obj._id_factory = FreshIdFactory.from_state(state["id_factory"])
        obj.budget_ = dict(state["budget"])
        obj.n_updates_ = int(state["n_updates"])
        if state.get("fit_info") is not None:
            obj.fit_info_ = BaselineFitInfo(**state["fit_info"])

        if transforms is not None:
            obj._fitted_transforms = transforms
        elif state.get("transforms_saved"):
            with open(os.path.join(path, _TRANSFORMS_FILE), "rb") as fh:
                obj._fitted_transforms = pickle.load(fh)
        if obj._train_repr == "processed" and obj._fitted_transforms is None:
            raise RuntimeError("This checkpoint needs the fitted `transforms` pipeline: pass transforms=.")

        loader = synthesizer_loader or _default_synthesizer_loader
        obj._model = loader(os.path.join(path, _SYNTH_FILE))
        obj._fitted = True
        return obj
