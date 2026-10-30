"""
Loading of the tracked dataset bundles.

The bundles under ``sbtab/data/datasets/*.pkl`` are pickled ``dict[str, DataFrame]``.
They were written with numpy 2.x / pandas 3.x, while ``requirements.txt`` allows
numpy 1.26 / pandas 2.2, where a plain ``pickle.load`` fails
(``No module named 'numpy._core.numeric'``; ``StringDtype.__init__() takes ...``).

``load_bundle`` first tries a plain load and only then falls back to a
cross-version unpickler. The fallback touches pandas pickle internals and is a
compatibility shim, not a storage format: ``prepare_splits`` re-materialises every
dataset it uses as Parquet plus a JSON schema and a value-based fingerprint, and
all later stages read that copy.
"""
from __future__ import annotations

import hashlib
import importlib
import pickle
import sys
import warnings
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd

BUNDLE_DIR = Path(__file__).resolve().parent / "datasets"


def _install_numpy_core_aliases() -> None:
    """Let numpy-2 pickles (module path ``numpy._core``) resolve under numpy 1.x."""
    # numpy 1.26 ships a PARTIAL ``numpy._core`` stub (no ``numeric`` etc.), so the
    # presence of the package says nothing; alias each missing submodule.
    try:
        import numpy.core as _nc
    except Exception:  # pragma: no cover
        return
    sys.modules.setdefault("numpy._core", _nc)
    for sub in ("numeric", "multiarray", "umath", "_multiarray_umath", "fromnumeric", "_internal", "numerictypes"):
        name = f"numpy._core.{sub}"
        try:
            importlib.import_module(name)
        except Exception:
            try:
                sys.modules[name] = importlib.import_module(f"numpy.core.{sub}")
            except Exception:
                pass


def _compat_unpickler():
    from pandas.compat import pickle_compat as pc

    def _string_dtype(storage=None, na_value=None):
        return pd.StringDtype("python")

    class _StringArray(pd.arrays.StringArray):
        def __setstate__(self, state):
            if isinstance(state, tuple) and len(state) == 2:
                arr = np.array(state[1], dtype=object, copy=True)
                arr[pd.isna(arr)] = pd.NA
                return super().__setstate__({"_ndarray": arr, "_dtype": pd.StringDtype("python")})
            return super().__setstate__(state)

    class _Categorical(pd.Categorical):
        def __setstate__(self, state):
            if isinstance(state, tuple) and len(state) == 2:
                return super().__setstate__({"_ndarray": np.asarray(state[1]), "_dtype": state[0]})
            return super().__setstate__(state)

    class _Unpickler(pc.Unpickler):
        def find_class(self, module, name):
            if module.startswith("pandas"):
                if name == "StringDtype":
                    return _string_dtype
                if name == "StringArray":
                    return _StringArray
                if name == "Categorical":
                    return _Categorical
            return super().find_class(module, name)

    return _Unpickler


def load_bundle(path) -> Dict[str, pd.DataFrame]:
    path = Path(path)
    try:
        with open(path, "rb") as fh:
            bundle = pickle.load(fh)
    except Exception as first_error:
        _install_numpy_core_aliases()
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                with open(path, "rb") as fh:
                    bundle = _compat_unpickler()(fh).load()
        except Exception as second_error:
            raise RuntimeError(
                f"cannot load dataset bundle {path}: plain pickle failed with "
                f"{type(first_error).__name__}: {first_error}; the cross-version fallback failed with "
                f"{type(second_error).__name__}: {second_error}. The bundle was written with numpy 2 / "
                f"pandas 3; load it in such an environment or regenerate it."
            ) from second_error
    if not isinstance(bundle, dict):
        raise TypeError(f"{path} does not contain a dict of DataFrames")
    return bundle


def normalise_frame(df: pd.DataFrame) -> pd.DataFrame:
    """
    Version-independent dtypes: pandas string/category/bool columns become plain
    Python ``object`` strings (missing -> None) so that the same bundle yields the
    same values, the same Parquet file and the same fingerprint in every environment.
    """
    out = pd.DataFrame(index=pd.RangeIndex(len(df)))
    for col in df.columns:
        s = df[col].reset_index(drop=True)
        if pd.api.types.is_bool_dtype(s.dtype):
            out[col] = s.map({True: "True", False: "False"}).astype(object)
        elif pd.api.types.is_numeric_dtype(s.dtype):
            out[col] = pd.to_numeric(s, errors="raise")
        else:
            vals = s.astype(object)
            out[col] = vals.where(~pd.isna(vals), None).map(lambda v: None if v is None else str(v)).astype(object)
    out.columns = [str(c) for c in df.columns]
    return out


def frame_fingerprint(df: pd.DataFrame) -> str:
    """SHA-256 over column names and canonicalised values (not over a pickle)."""
    h = hashlib.sha256()
    h.update(repr(list(df.columns)).encode())
    for col in df.columns:
        s = df[col]
        if pd.api.types.is_numeric_dtype(s.dtype):
            h.update(np.ascontiguousarray(s.to_numpy(dtype=np.float64)).tobytes())
        else:
            h.update("\x1f".join("\x00" if v is None or (isinstance(v, float) and np.isnan(v)) else str(v)
                                 for v in s.tolist()).encode())
    return h.hexdigest()
