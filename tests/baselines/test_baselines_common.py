"""Library-free tests for sbtab.baselines.base / sbtab.baselines.encoding."""

from __future__ import annotations

import os
import pathlib
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

from sbtab.baselines.base import (
    BaselineGenerativeModel,
    ColumnRoles,
    FreshIdFactory,
    empirical_distribution,
    largest_remainder_allocation,
    next_multiple,
    resolve_column_roles,
)
from sbtab.baselines.encoding import (
    MixedToContinuousCodec,
    argmax_decode,
    fit_standardizer,
    nearest_support_decode,
)

from baseline_testkit import MIXED_ROLES, assert_same_schema, make_mixed_frame


# ----------------------------------------------------------------------
# packaging / interface
# ----------------------------------------------------------------------


def test_package_imports_without_third_party_generators():
    import sbtab.baselines  # noqa: F401
    import sbtab.baselines.ctgan as ctgan_pkg
    import sbtab.baselines.stasy as ve_pkg
    import sbtab.baselines.tabddpm as ddpm_pkg
    import sbtab.baselines.tabpfn as pfn_pkg

    assert {"CTGANWrapper", "CTGANConfig"} <= set(ctgan_pkg.__all__)
    assert {"TabDDPMWrapper", "TabDDPMConfig"} <= set(ddpm_pkg.__all__)
    assert {"TabPFGenGenerative", "TabPFGenConfig"} <= set(pfn_pkg.__all__)
    assert {"VEScoreSDEBaseline", "VEScoreSDEConfig", "FAITHFULNESS"} <= set(ve_pkg.__all__)


def test_third_party_generator_imports_stay_lazy():
    """Importing the packages must not import sdv / ctgan / tabpfgen / tabpfn (fresh interpreter)."""
    repo_root = str(pathlib.Path(__file__).resolve().parents[2])
    code = (
        "import sys\n"
        "import sbtab.baselines, sbtab.baselines.ctgan, sbtab.baselines.tabpfn\n"
        "import sbtab.baselines.stasy, sbtab.baselines.tabddpm\n"
        "bad = [m for m in ('sdv', 'ctgan', 'tabpfgen', 'tabpfn') if m in sys.modules]\n"
        "assert not bad, bad\n"
    )
    env = {**os.environ, "PYTHONPATH": repo_root}
    proc = subprocess.run([sys.executable, "-c", code], cwd=repo_root, env=env, capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr


def test_abstract_interface_requires_checkpointing():
    class Incomplete(BaselineGenerativeModel):
        def fit(self, data, **kwargs):
            return self

        def sample(self, n, seed=None, **kwargs):
            return np.zeros((n, 1))

    with pytest.raises(TypeError):
        Incomplete()  # save_checkpoint / load_checkpoint are abstract


def test_all_wrappers_share_the_interface():
    from sbtab.baselines.ctgan import CTGANConfig, CTGANWrapper
    from sbtab.baselines.stasy import VEScoreSDEBaseline, VEScoreSDEConfig
    from sbtab.baselines.tabddpm import TabDDPMConfig, TabDDPMWrapper
    from sbtab.baselines.tabpfn import TabPFGenConfig, TabPFGenGenerative

    models = [
        CTGANWrapper(CTGANConfig()),
        VEScoreSDEBaseline(VEScoreSDEConfig(steps=1)),
        TabDDPMWrapper(TabDDPMConfig(steps=1)),
        TabPFGenGenerative(TabPFGenConfig(target_col="y")),
    ]
    ids = []
    for m in models:
        assert isinstance(m, BaselineGenerativeModel)
        assert isinstance(m.variant_id, str) and m.variant_id not in ("", "unspecified")
        ids.append(m.variant_id)
        for name in ("fit", "sample", "save_checkpoint", "load_checkpoint"):
            assert callable(getattr(m, name))
        with pytest.raises(RuntimeError):
            _ = m.n_updates  # not fitted yet
    assert len(set(ids)) == len(ids)


# ----------------------------------------------------------------------
# explicit column roles
# ----------------------------------------------------------------------


def test_roles_follow_frame_order_and_place_target_by_task():
    cols = ["x_cont", "cat_code", "n_visits", "x_f32", "label"]
    roles = resolve_column_roles(cols, **MIXED_ROLES)
    assert roles.continuous == ("x_cont", "x_f32")
    assert roles.discrete == ("n_visits",)
    assert roles.categorical == ("cat_code", "label")  # classification target -> categorical
    assert roles.task == "classification" and roles.source == "explicit"

    reg = resolve_column_roles(["a", "y"], continuous_cols=["a"], target_col="y", task="regression")
    assert reg.continuous == ("a", "y") and reg.categorical == ()

    count_target = resolve_column_roles(["a", "y"], continuous_cols=["a"], discrete_cols=["y"], target_col="y")
    assert count_target.discrete == ("y",) and count_target.task == "regression"


def test_roles_are_never_guessed():
    cols = ["a", "b", "y"]
    with pytest.raises(ValueError, match="without an explicit role"):
        resolve_column_roles(cols, continuous_cols=["a"], target_col="y", task="regression")
    with pytest.raises(ValueError, match="never inferred"):
        resolve_column_roles(cols, continuous_cols=["a", "b"], target_col="y")  # no task, not listed
    with pytest.raises(ValueError, match="assigned twice"):
        resolve_column_roles(cols, continuous_cols=["a", "b"], categorical_cols=["b", "y"])
    with pytest.raises(ValueError, match="not in the data"):
        resolve_column_roles(cols, continuous_cols=["a", "b", "zzz"], categorical_cols=["y"])
    with pytest.raises(ValueError, match="must be categorical"):
        resolve_column_roles(cols, continuous_cols=["a", "b", "y"], target_col="y", task="classification")
    with pytest.raises(ValueError, match="task must be"):
        resolve_column_roles(cols, continuous_cols=["a", "b"], target_col="y", task="multiclass")
    with pytest.raises(ValueError, match="id_col"):
        resolve_column_roles(cols, continuous_cols=["a", "b"], categorical_cols=["y"], id_col="a")


def test_roles_round_trip_through_dict():
    roles = resolve_column_roles(["i", "a", "y"], continuous_cols=["a"], target_col="y", task="classification", id_col="i")
    assert ColumnRoles.from_dict(roles.to_dict()) == roles
    assert roles.role_of("i") == "id" and roles.role_of("y") == "categorical"


# ----------------------------------------------------------------------
# largest-remainder allocation
# ----------------------------------------------------------------------


def test_largest_remainder_sums_to_n_and_is_within_one_count():
    rng = np.random.default_rng(0)
    for _ in range(300):
        k = int(rng.integers(1, 9))
        w = rng.random(k) + 1e-3
        n = int(rng.integers(0, 500))
        counts = largest_remainder_allocation(n, w)
        assert counts.sum() == n
        assert np.all(counts >= 0)
        assert np.all(np.abs(counts - n * w / w.sum()) < 1.0)


def test_largest_remainder_known_cases():
    assert largest_remainder_allocation(10, [0.6, 0.3, 0.1]).tolist() == [6, 3, 1]
    assert largest_remainder_allocation(1, [0.6, 0.3, 0.1]).tolist() == [1, 0, 0]
    assert largest_remainder_allocation(7, [1, 1, 1]).tolist() == [3, 2, 2]  # ties -> lowest index first
    assert largest_remainder_allocation(5, [0.0, 1.0]).tolist() == [0, 5]    # zero-weight class gets nothing
    with pytest.raises(ValueError):
        largest_remainder_allocation(5, [0.0, 0.0])
    with pytest.raises(ValueError):
        largest_remainder_allocation(5, [0.5, -0.1])


def test_next_multiple():
    assert [next_multiple(n, 10) for n in (1, 9, 10, 11, 25)] == [10, 10, 10, 20, 30]
    assert next_multiple(7, 1) == 7 and next_multiple(7, 3) == 9
    with pytest.raises(ValueError):
        next_multiple(5, 0)


def test_empirical_distribution_is_plain_python():
    dist = empirical_distribution(np.array([1, -1, 1, 1], dtype=np.int64))
    assert dist == [[-1, 1], [1, 3]]
    assert all(type(v) is int for pair in dist for v in pair)


# ----------------------------------------------------------------------
# ids
# ----------------------------------------------------------------------


def test_fresh_ids_are_disjoint_from_training_ids():
    train_int = pd.Series(np.arange(1000, 1240))
    fac = FreshIdFactory.fit(train_int)
    fresh = fac.make(500)
    assert len(set(fresh)) == 500 and not set(fresh) & set(train_int)
    assert FreshIdFactory.from_state(fac.state_dict()).make(3).tolist() == fresh[:3].tolist()

    train_str = pd.Series(["synthetic_0", "u17", "u18"])
    fresh_str = FreshIdFactory.fit(train_str).make(10)
    assert not set(fresh_str) & set(train_str)


# ----------------------------------------------------------------------
# decoding helpers
# ----------------------------------------------------------------------


def test_zero_variance_column_gets_scale_one():
    x = np.column_stack([np.full(50, 5000.0), np.arange(50, dtype=float)])
    mean, scale = fit_standardizer(x)
    assert scale[0] == 1.0 and mean[0] == 5000.0
    assert scale[1] == pytest.approx(np.arange(50).std())


def test_nearest_support_decode_and_report():
    support = np.array([0.0, 5.0, 10.0, 50.0])
    raw = np.array([-3.0, 2.4, 2.6, 7.5, 31.0, 80.0, 10.0])
    decoded, rep = nearest_support_decode(raw, support)
    assert decoded.tolist() == [0.0, 0.0, 5.0, 5.0, 50.0, 50.0, 10.0]  # tie 7.5 -> lower value
    assert rep["n"] == 7 and rep["support_size"] == 4
    assert rep["out_of_support_rate"] == pytest.approx(6 / 7)   # only the exact 10.0 was on support
    assert rep["out_of_range_rate"] == pytest.approx(2 / 7)     # -3 and 80
    assert rep["max_abs_shift"] == pytest.approx(30.0)
    with pytest.raises(ValueError):
        nearest_support_decode(np.array([np.nan]), support)


def test_argmax_decode_report():
    codes, rep = argmax_decode(np.array([[0.9, 0.1, 0.0], [0.4, 0.45, 0.1]]))
    assert codes.tolist() == [0, 1]
    assert rep["ambiguous_rate"] == pytest.approx(0.5)


def test_codec_round_trip_is_exact():
    """encode -> decode of the TRAINING rows must reproduce them (one-hot/argmax + support decoding)."""
    df = make_mixed_frame()
    roles = resolve_column_roles(list(df.columns), **MIXED_ROLES)
    codec = MixedToContinuousCodec(standardize_numeric=True).fit(df, roles)

    z = codec.encode(df)
    # 3 numeric + 4 (cat_code vocabulary) + 3 (label vocabulary)
    assert z.shape == (len(df), 3 + 4 + 3) and z.dtype == np.float32
    assert codec.vocab["cat_code"] == [0, 1, 3, 4] and codec.vocab["label"] == [0, 1, 2]
    # numeric block is z-scored on the fit rows
    assert np.allclose(z[:, :3].mean(axis=0), 0.0, atol=1e-5)
    assert np.allclose(z[:, :3].std(axis=0), 1.0, atol=1e-4)

    back, report = codec.decode(z)
    assert_same_schema(back, df)
    for c in ("cat_code", "n_visits", "label"):
        assert back[c].tolist() == df[c].tolist()
    # float32 z-score round trip: relative error ~1e-7 of the column scale
    assert np.allclose(back["x_cont"], df["x_cont"], atol=1e-5)
    assert set(report) == {"n_visits", "cat_code", "label"}
    assert report["n_visits"]["out_of_range_rate"] == 0.0

    restored = MixedToContinuousCodec.from_state(codec.state_dict())
    back2, _ = restored.decode(z)
    pd.testing.assert_frame_equal(back, back2)


def test_codec_rejects_missing_values_and_unseen_categories():
    df = make_mixed_frame()
    roles = resolve_column_roles(list(df.columns), **MIXED_ROLES)
    codec = MixedToContinuousCodec().fit(df, roles)

    unseen = df.copy()
    unseen.loc[0, "cat_code"] = 2  # code 2 is not in the training vocabulary
    with pytest.raises(ValueError, match="outside its vocabulary"):
        codec.encode(unseen)

    with_nan = df.copy()
    with_nan.loc[0, "x_cont"] = np.nan
    with pytest.raises(ValueError, match="NaN"):
        MixedToContinuousCodec().fit(with_nan, roles)
