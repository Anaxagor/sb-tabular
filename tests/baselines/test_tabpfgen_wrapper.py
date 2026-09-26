"""
TabPFGen wrapper tests.

`tabpfgen` / `tabpfn` are optional.  Label-prior allocation + resampling, exact-n selection, label
encoding, seeding and checkpoint state are pure python and are tested ALWAYS, with a fake generator
that reproduces the upstream quirks the wrapper has to compensate for:

  * classification returns K * floor(n / K) rows, regression 10 * floor(n / 10) rows,
  * labels are balanced (balance_classes=True) or uniform (False) - never the training prior,
  * labels come back as argmax INDICES 0..K-1.

The tests that drive the real library are at the bottom; they are skipped when it is missing, and a
skip is NOT a validation of the adapter.
"""

from __future__ import annotations

import inspect
import warnings

import numpy as np
import pandas as pd
import pytest
import torch

from sbtab.baselines.base import largest_remainder_allocation
from sbtab.baselines.tabpfn import (
    LabelCodec,
    TabPFGenConfig,
    TabPFGenGenerative,
    generate_exact,
    resample_to_label_prior,
    select_exact_rows,
)
from sbtab.baselines.tabpfn import model as pfn_model

from baseline_testkit import TABPFGEN_SKIP_REASON, make_mixed_frame

GENERATED_MARK = 1000.0   # fake generated features live around 1000; training features around 0


class FakeTabPFGen:
    """Stand-in for tabpfgen.TabPFGen with the upstream row-count / label behaviour."""

    def __init__(self, never_class=None, empty=False):
        self.never_class = never_class
        self.empty = empty
        self.calls = []

    def _features(self, m, d):
        return GENERATED_MARK + np.random.normal(size=(m, d))

    def generate_classification(self, X_train, y_train, n_samples, balance_classes=True):
        y_train = np.asarray(y_train)
        assert np.issubdtype(y_train.dtype, np.integer), "upstream must receive ENCODED labels"
        K = int(y_train.max()) + 1
        assert set(np.unique(y_train)) <= set(range(K))
        self.calls.append({"n_samples": int(n_samples), "balance_classes": bool(balance_classes),
                           "torch_seed": torch.initial_seed(), "context_rows": len(X_train)})
        m = 0 if self.empty else K * (int(n_samples) // K)          # upstream floors to a multiple of K
        y = np.repeat(np.arange(K), m // K) if balance_classes else np.random.randint(0, K, size=m)
        if self.never_class is not None:
            y = np.where(y == self.never_class, (self.never_class + 1) % K, y)
        return self._features(m, X_train.shape[1]), y               # argmax indices 0..K-1

    def generate_regression(self, X_train, y_train, n_samples, use_quantiles=True):
        self.calls.append({"n_samples": int(n_samples), "torch_seed": torch.initial_seed()})
        m = 10 * (int(n_samples) // 10)                              # upstream floors to a multiple of 10
        return self._features(m, X_train.shape[1]), np.random.normal(size=m)


def fake_factory(**kw):
    return lambda cfg: FakeTabPFGen(**kw)


def uniform_label_generator(K, d=2, floor=True, never=None, rng=None):
    rng = rng or np.random.default_rng(0)

    def generate(m):
        m = K * (m // K) if floor else m
        y = rng.integers(0, K, size=m)
        if never is not None:
            y = np.where(y == never, (never + 1) % K, y)
        return GENERATED_MARK + rng.normal(size=(m, d)), y

    return generate


# ----------------------------------------------------------------------
# P1: label prior (pure)
# ----------------------------------------------------------------------


@pytest.mark.parametrize("n", [1, 7, 10, 23, 101])
def test_resampling_matches_the_training_prior_exactly(n):
    freqs = [0.7, 0.2, 0.1]
    X, y, rep = resample_to_label_prior(uniform_label_generator(3), n, freqs, rng=np.random.default_rng(1))
    counts = np.bincount(y, minlength=3)
    assert len(X) == len(y) == n and counts.sum() == n
    assert counts.tolist() == largest_remainder_allocation(n, freqs).tolist()
    assert np.all(np.abs(counts - n * np.asarray(freqs)) < 1.0), "within one count of the training frequencies"
    assert rep["prior_preserved"] and rep["unfilled"] == {}
    assert np.all(X > GENERATED_MARK - 50)


def test_resampling_tops_up_a_rare_class_by_further_generation():
    # prior is 95% class 0 but the generator is uniform over 5 classes: one 2n request cannot hold it
    freqs = [0.95, 0.0125, 0.0125, 0.0125, 0.0125]
    X, y, rep = resample_to_label_prior(uniform_label_generator(5), 200, freqs, rng=np.random.default_rng(2))
    assert np.bincount(y, minlength=5).tolist() == largest_remainder_allocation(200, freqs).tolist()
    assert rep["rounds"] > 1 and rep["generated_rows"] > 400 and rep["prior_preserved"]


def test_unproducible_class_is_reported_and_never_filled_with_real_rows():
    freqs = [0.5, 0.3, 0.2]
    gen = uniform_label_generator(3, never=2)
    X, y, rep = resample_to_label_prior(gen, 50, freqs, rng=np.random.default_rng(3), max_rounds=4)
    assert len(X) == 50 and 2 not in set(y)
    assert rep["rounds"] == 4, "retries must be bounded"
    assert rep["prior_preserved"] is False
    assert rep["unfilled"] == {2: {"requested": 10, "produced": 0, "returned": 0}}
    # the 10 missing rows were re-allocated to the producible classes proportionally (0.5 : 0.3)
    assert np.bincount(y, minlength=3).tolist() == [25 + 6, 15 + 4, 0]
    assert np.all(X > GENERATED_MARK - 50), "every returned row must be a GENERATED row"


def test_generator_that_returns_nothing_raises():
    empty = lambda m: (np.empty((0, 2)), np.empty(0, dtype=int))  # noqa: E731
    with pytest.raises(RuntimeError, match="Real rows are never used"):
        resample_to_label_prior(empty, 10, [0.5, 0.5], rng=np.random.default_rng(0), max_rounds=3)
    with pytest.raises(RuntimeError, match="Real rows are never used"):
        generate_exact(empty, 10, rng=np.random.default_rng(0), max_rounds=3)


# ----------------------------------------------------------------------
# P2: exact n (pure)
# ----------------------------------------------------------------------


@pytest.mark.parametrize("n", [1, 9, 10, 11, 25])
def test_generate_exact_compensates_the_floor_to_ten(n):
    requested = []

    def floor_to_ten(m):
        requested.append(m)
        k = 10 * (m // 10)
        return np.full((k, 2), GENERATED_MARK), np.zeros(k)

    X, y, rep = generate_exact(floor_to_ten, n, rng=np.random.default_rng(0), row_multiple=10)
    assert len(X) == len(y) == n
    assert all(m % 10 == 0 for m in requested) and rep["rounds"] == 1


def test_select_exact_rows():
    X, y = np.arange(40).reshape(20, 2), np.arange(20)
    Xh, yh = select_exact_rows(X, y, 7)
    assert yh.tolist() == list(range(7))
    Xr, yr = select_exact_rows(X, y, 7, np.random.default_rng(0))
    assert len(yr) == 7 and len(set(yr)) == 7 and np.array_equal(Xr[:, 0] // 2, yr)
    with pytest.raises(ValueError, match="need 30"):
        select_exact_rows(X, y, 30)


# ----------------------------------------------------------------------
# P3: label encoding (pure)
# ----------------------------------------------------------------------


@pytest.mark.parametrize(
    "labels",
    [["pos", "neg", "neg", "unk"], [-1, 1, 1, -1], [3, 7, 7, 11], [True, False, True, True]],
)
def test_label_codec_round_trip(labels):
    s = pd.Series(labels)
    codec = LabelCodec().fit(s)
    idx = codec.encode(s)
    assert idx.dtype == np.int64 and sorted(set(idx)) == list(range(codec.n_classes))
    back = codec.decode(idx)
    assert back.tolist() == s.tolist() and str(back.dtype) == str(s.dtype)
    # upstream returns float/argmax indices: they must map back too
    assert codec.decode(idx.astype(np.float32)).tolist() == s.tolist()


def test_label_codec_rejects_unknown_labels_and_indices():
    codec = LabelCodec().fit(pd.Series([-1, 1]))
    assert codec.classes == [-1, 1]
    with pytest.raises(ValueError, match="not seen in training"):
        codec.encode(pd.Series([0]))
    with pytest.raises(ValueError, match="outside 0..1"):
        codec.decode(np.array([2]))


# ----------------------------------------------------------------------
# wrapper with a fake generator
# ----------------------------------------------------------------------


def frame(labels, n=200, p=(0.7, 0.2, 0.1), seed=0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    y = rng.choice(np.asarray(labels, dtype=object), size=n, p=p[: len(labels)])
    return pd.DataFrame({"f0": rng.normal(size=n), "y": pd.Series(y).infer_objects(), "f1": rng.normal(size=n)})


def test_config_defaults():
    cfg = TabPFGenConfig(target_col="y")
    assert cfg.balance_classes is False and cfg.preserve_label_prior is True
    with pytest.raises(ValueError, match="task must be"):
        TabPFGenConfig(target_col="y", task="clustering")


@pytest.mark.parametrize("labels,p", [(["neg", "pos", "unk"], (0.7, 0.2, 0.1)), ([-1, 1], (0.85, 0.15))])
def test_wrapper_preserves_training_label_prior_and_label_values(labels, p):
    df = frame(labels, p=p)
    m = TabPFGenGenerative(TabPFGenConfig(target_col="y", task="classification", seed=4),
                           generator_factory=fake_factory()).fit(df, continuous_cols=["f0", "f1"])
    n = 57
    out = m.sample(n, seed=1)

    assert len(out) == n and list(out.columns) == ["f0", "y", "f1"]
    assert str(out["y"].dtype) == str(df["y"].dtype) and set(out["y"]) <= set(labels)
    train_freq = df["y"].value_counts(normalize=True)
    got = out["y"].value_counts()
    for lab in labels:
        assert abs(got.get(lab, 0) - n * train_freq[lab]) < 1.0, "label counts must follow the TRAINING prior"
    assert m.label_report_["prior_preserved"] and m._generator.calls[0]["balance_classes"] is False
    assert (out[["f0", "f1"]].to_numpy() > GENERATED_MARK - 50).all(), "output must consist of generated rows only"
    assert [c for c, _ in m.label_distribution_] == sorted(labels)


def test_wrapper_reports_a_class_the_generator_cannot_produce():
    df = frame(["a", "b", "c"])
    m = TabPFGenGenerative(TabPFGenConfig(target_col="y", task="classification", max_topup_rounds=3),
                           generator_factory=fake_factory(never_class=2)).fit(df, continuous_cols=["f0", "f1"])
    out = m.sample(40, seed=0)
    assert len(out) == 40 and "c" not in set(out["y"])
    assert list(m.label_report_["unfilled"]) == ["'c'"] and m.label_report_["prior_preserved"] is False
    assert m.last_sample_cost_["generation_calls"] == 3
    assert (out[["f0", "f1"]].to_numpy() > GENERATED_MARK - 50).all()


@pytest.mark.parametrize("n", [1, 7, 23])
def test_classification_returns_exactly_n(n):
    m = TabPFGenGenerative(TabPFGenConfig(target_col="y", task="classification"),
                           generator_factory=fake_factory()).fit(frame(["a", "b", "c"]), continuous_cols=["f0", "f1"])
    assert len(m.sample(n, seed=0)) == n


@pytest.mark.parametrize("n", [1, 9, 10, 11, 25])
def test_regression_returns_exactly_n(n):
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"f0": rng.normal(size=90), "f1": rng.normal(size=90), "y": rng.normal(size=90)})
    m = TabPFGenGenerative(TabPFGenConfig(target_col="y", task="regression"), generator_factory=fake_factory())
    m.fit(df, continuous_cols=["f0", "f1"], target_col="y", task="regression")
    out = m.sample(n, seed=0)
    assert len(out) == n and out["y"].dtype == np.float64
    assert all(c["n_samples"] % 10 == 0 for c in m._generator.calls), "requests must be multiples of 10"


def test_without_prior_preservation_the_variant_is_named_differently():
    df = frame(["a", "b", "c"])
    m = TabPFGenGenerative(TabPFGenConfig(target_col="y", task="classification", preserve_label_prior=False),
                           generator_factory=fake_factory()).fit(df, continuous_cols=["f0", "f1"])
    assert m.variant_id == "tabpfgen_sgld_upstream_labels"
    assert len(m.sample(23, seed=0)) == 23 and m.label_report_["prior_preserved"] is False
    assert TabPFGenGenerative(TabPFGenConfig(target_col="y")).variant_id == "tabpfgen_sgld_prior_resampled"


def test_mixed_feature_types_are_encoded_for_the_continuous_generator():
    df = make_mixed_frame()
    m = TabPFGenGenerative(TabPFGenConfig(), generator_factory=fake_factory())
    m.fit(df, continuous_cols=["x_cont", "x_f32"], discrete_cols=["n_visits"], categorical_cols=["cat_code"],
          target_col="label", task="classification")
    assert m._X_train.shape == (len(df), 3 + 4), "3 numeric features + one-hot(cat_code: 4)"
    assert m.conditioning_context_["n_train_rows"] == len(df)
    out = m.sample(31, seed=0)
    assert list(out.columns) == list(df.columns)
    assert out.dtypes.astype(str).to_dict() == df.dtypes.astype(str).to_dict()
    assert set(out["cat_code"]) <= {0, 1, 3, 4} and set(out["n_visits"]) <= {0, 5, 10, 50}
    assert set(m.decoding_report_) == {"n_visits", "cat_code"}


def test_roles_and_task_are_not_inferred_when_declared(monkeypatch):
    df = frame([0, 1, 2])
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        m = TabPFGenGenerative(TabPFGenConfig(target_col="y"), generator_factory=fake_factory())
        m.fit(df, continuous_cols=["f0", "f1"], task="classification")
    assert m.role_source_ == "explicit" and m._task == "classification"

    with pytest.raises(ValueError, match="never inferred"):
        TabPFGenGenerative(TabPFGenConfig(target_col="y"), generator_factory=fake_factory()).fit(df, continuous_cols=["f0", "f1"])
    with pytest.raises(ValueError, match="disagrees"):
        TabPFGenGenerative(TabPFGenConfig(target_col="y"), generator_factory=fake_factory()).fit(
            df, continuous_cols=["f0", "y"], target_col="f1", task="regression")

    # legacy call (no roles, task="auto") still works but says that it guessed
    with pytest.warns(UserWarning, match="INFERRED from dtype/cardinality"):
        legacy = TabPFGenGenerative(TabPFGenConfig(target_col="y"), generator_factory=fake_factory()).fit(df)
    assert legacy.role_source_ == "default_all_continuous+inferred_task"


def test_seeding_uses_torch_and_cfg_seed():
    df = frame(["a", "b", "c"])
    m = TabPFGenGenerative(TabPFGenConfig(target_col="y", task="classification", seed=21),
                           generator_factory=fake_factory()).fit(df, continuous_cols=["f0", "f1"])
    a = m.sample(30, seed=5)
    assert m._generator.calls[-1]["torch_seed"] == 5, "torch must be seeded, not only numpy"
    pd.testing.assert_frame_equal(a, m.sample(30, seed=5))
    assert not a.equals(m.sample(30, seed=6))

    default = m.sample(30)                       # no seed -> cfg.seed
    assert m._generator.calls[-1]["torch_seed"] == 21
    pd.testing.assert_frame_equal(default, m.sample(30, seed=21))


def test_cost_model_and_identity_are_declared():
    df = frame(["a", "b", "c"], n=120)
    m = TabPFGenGenerative(TabPFGenConfig(target_col="y", task="classification", n_sgld_steps=77),
                           generator_factory=fake_factory()).fit(df, continuous_cols=["f0", "f1"])
    assert m.n_updates == 0
    assert {k: m.adaptation_cost_[k] for k in ("fit_updates", "sgld_steps", "context_rows")} == {
        "fit_updates": 0, "sgld_steps": 77, "context_rows": 120}
    assert m.sgld_settings_["n_sgld_steps"] == 77 and m.conditioning_context_["feature_columns"] == ["f0", "f1"]
    assert {"tabpfgen_version", "tabpfn_version", "tabpfn_default_model_path"} <= set(m.pretrained_identity_)
    m.sample(10, seed=0)
    cost = m.last_sample_cost_
    assert cost["sgld_steps_total"] == 77 * cost["generation_calls"] and cost["context_rows"] == 120


def test_checkpoint_persists_the_conditioning_rows(tmp_path):
    df = frame(["neg", "pos", "unk"])
    m = TabPFGenGenerative(TabPFGenConfig(target_col="y", task="classification", seed=2),
                           generator_factory=fake_factory()).fit(df, continuous_cols=["f0", "f1"])
    expected = m.sample(33, seed=8)
    path = str(tmp_path / "tabpfgen.pt")
    m.save_checkpoint(path)

    state = torch.load(path, map_location="cpu", weights_only=True)
    assert state["contains_training_rows"] is True
    assert np.allclose(state["X_train"].numpy(), df[["f0", "f1"]].to_numpy(dtype=np.float32))
    assert state["label_classes"] == ["neg", "pos", "unk"] and state["adaptation_cost"]["fit_updates"] == 0

    loaded = TabPFGenGenerative.load_checkpoint(path, generator_factory=fake_factory())
    assert loaded.label_distribution_ == m.label_distribution_ and loaded.n_updates == 0
    assert set(loaded.pretrained_identity_) == {"at_save", "at_load"}
    pd.testing.assert_frame_equal(loaded.sample(33, seed=8), expected)


def test_no_leftover_citation_markers():
    src = inspect.getsource(pfn_model)
    assert "oaicite" not in src and "contentReference" not in src


# ----------------------------------------------------------------------
# real library (skipped when missing - a skip is NOT a validation of the adapter)
# ----------------------------------------------------------------------


def test_real_tabpfgen_classification_end_to_end():
    pytest.importorskip("tabpfgen", reason=TABPFGEN_SKIP_REASON)
    df = frame(["neg", "pos", "unk"], n=90)
    cfg = TabPFGenConfig(target_col="y", task="classification", n_sgld_steps=5, device="cpu", seed=1)
    m = TabPFGenGenerative(cfg).fit(df, continuous_cols=["f0", "f1"])
    n = 23
    out = m.sample(n, seed=2)
    assert len(out) == n and list(out.columns) == list(df.columns) and set(out["y"]) <= {"neg", "pos", "unk"}
    if m.label_report_["prior_preserved"]:
        freq = df["y"].value_counts(normalize=True)
        assert all(abs(out["y"].value_counts().get(k, 0) - n * freq[k]) < 1.0 for k in freq.index)
    assert m.pretrained_identity_["tabpfgen_version"] is not None


def test_real_tabpfgen_regression_returns_exactly_n():
    pytest.importorskip("tabpfgen", reason=TABPFGEN_SKIP_REASON)
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"f0": rng.normal(size=80), "f1": rng.normal(size=80), "y": rng.normal(size=80)})
    cfg = TabPFGenConfig(target_col="y", task="regression", n_sgld_steps=5, device="cpu", seed=1)
    m = TabPFGenGenerative(cfg).fit(df, continuous_cols=["f0", "f1"], target_col="y", task="regression")
    assert len(m.sample(17, seed=0)) == 17
