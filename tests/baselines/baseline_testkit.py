"""Shared helpers for the baseline-wrapper tests (tiny, CPU-only, deterministic)."""

from __future__ import annotations

import numpy as np
import pandas as pd

# Reason strings for the library-dependent tests.  A SKIP IS NOT A PASS.
SDV_SKIP_REASON = (
    "sdv is not installed: the CTGAN tests that drive the real SDV CTGANSynthesizer are SKIPPED. "
    "A skip is NOT a validation of the CTGAN adapter - only the library-free logic (metadata dict, "
    "batch/pac validation, exact-n, seeding hooks, id handling, checkpoint state) was tested."
)
TABPFGEN_SKIP_REASON = (
    "tabpfgen is not installed: the tests that drive the real TabPFGen/TabPFN generator are SKIPPED. "
    "A skip is NOT a validation of the TabPFGen adapter - only the library-free logic (label-prior "
    "allocation/resampling, exact-n selection, label encoding, checkpoint state) was tested."
)

MIXED_ROLES = dict(
    continuous_cols=["x_cont", "x_f32"],
    discrete_cols=["n_visits"],
    categorical_cols=["cat_code"],
    target_col="label",
    task="classification",
)


def make_mixed_frame(n: int = 240, seed: int = 0) -> pd.DataFrame:
    """
    Benchmark-style "common representation" frame with deliberately interleaved column order:

      x_cont    float64  standardised continuous
      cat_code  int64    integer-coded NOMINAL feature, vocabulary {0, 1, 3, 4} (code 2 absent)
      n_visits  int64    discrete count on its ORIGINAL scale, gappy support {0, 5, 10, 50}
      x_f32     float32  continuous (dtype must survive)
      label     int64    integer class target, imbalanced prior
    """
    rng = np.random.default_rng(seed)
    label = rng.choice([0, 1, 2], size=n, p=[0.6, 0.3, 0.1])
    return pd.DataFrame(
        {
            "x_cont": rng.normal(size=n) + 0.5 * label,
            "cat_code": rng.choice([0, 1, 3, 4], size=n).astype(np.int64),
            "n_visits": rng.choice([0, 5, 10, 50], size=n, p=[0.4, 0.3, 0.2, 0.1]).astype(np.int64),
            "x_f32": rng.normal(size=n).astype(np.float32),
            "label": label.astype(np.int64),
        }
    )


def assert_same_schema(out: pd.DataFrame, ref: pd.DataFrame) -> None:
    assert list(out.columns) == list(ref.columns), "column order changed"
    assert out.dtypes.astype(str).to_dict() == ref.dtypes.astype(str).to_dict(), "dtypes changed"


def assert_mixed_output_valid(out: pd.DataFrame, ref: pd.DataFrame, n: int) -> None:
    assert len(out) == n
    assert_same_schema(out, ref)
    assert not out.isna().any().any()
    assert set(out["cat_code"].unique()) <= set(ref["cat_code"].unique())
    assert set(out["label"].unique()) <= set(ref["label"].unique())
    assert set(out["n_visits"].unique()) <= set(ref["n_visits"].unique())
