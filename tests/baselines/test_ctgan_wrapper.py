"""
CTGAN wrapper tests.

`sdv` is an optional dependency.  Everything that can be checked WITHOUT it is checked here with
pure functions and a tiny fake synthesizer (it stands in for the library object; it is not a model
of CTGAN).  The tests that drive the real SDV CTGANSynthesizer are at the bottom and are skipped
when `sdv` is missing - and a skip is NOT a validation of the adapter.
"""

from __future__ import annotations

import pickle
import sys
import types
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch

from sbtab.baselines.ctgan import (
    CTGANConfig,
    CTGANWrapper,
    build_sdv_metadata_dict,
    ctgan_update_counts,
    validate_ctgan_batch,
)
from sbtab.baselines.ctgan import model as ctgan_model

from baseline_testkit import MIXED_ROLES, SDV_SKIP_REASON, assert_mixed_output_valid, make_mixed_frame


class FakeSynthesizer:
    """Independent-marginals stand-in for sdv.single_table.CTGANSynthesizer (same method names)."""

    def __init__(self, metadata_dict, cfg, rows_per_call=None, surplus=0):
        self.metadata_dict = metadata_dict
        self.cfg = cfg
        self.rows_per_call = rows_per_call
        self.surplus = surplus
        self.sample_calls = []
        self.reset_calls = 0
        self.random_state_seeds = []
        self.torch_seed_at_fit = None
        self.columns = {}

    def fit(self, df):
        self.torch_seed_at_fit = torch.initial_seed()
        table = next(iter(self.metadata_dict["tables"].values()))["columns"]
        for col, spec in table.items():
            if spec["sdtype"] == "categorical":
                self.columns[col] = ("cat", df[col].unique().tolist())
            else:
                self.columns[col] = ("num", float(df[col].mean()), float(df[col].std()))

    def sample(self, num_rows):
        self.sample_calls.append(int(num_rows))
        m = int(num_rows) + self.surplus if self.rows_per_call is None else min(int(num_rows), self.rows_per_call)
        data = {}
        for col, spec in self.columns.items():
            if spec[0] == "cat":
                data[col] = np.random.choice(np.asarray(spec[1], dtype=object), size=m)
            else:
                data[col] = np.random.normal(spec[1], spec[2], size=m)
        return pd.DataFrame(data)

    def reset_sampling(self):
        self.reset_calls += 1

    def _set_random_state(self, seed):
        self.random_state_seeds.append(seed)

    def save(self, filepath):
        with open(filepath, "wb") as fh:
            pickle.dump(self, fh)

    @classmethod
    def load(cls, filepath):
        with open(filepath, "rb") as fh:
            return pickle.load(fh)


def fake_factory(**fake_kwargs):
    return lambda metadata_dict, cfg: FakeSynthesizer(metadata_dict, cfg, **fake_kwargs)


def fitted(df, **fake_kwargs) -> CTGANWrapper:
    return CTGANWrapper(CTGANConfig(epochs=3, batch_size=60, pac=10, seed=11), synthesizer_factory=fake_factory(**fake_kwargs)).fit(
        df, **MIXED_ROLES
    )


# ----------------------------------------------------------------------
# C1: metadata from explicit roles (pure)
# ----------------------------------------------------------------------


def test_metadata_dict_comes_from_explicit_roles_only():
    md = build_sdv_metadata_dict(
        ["x_cont", "cat_code", "n_visits", "label"],
        categorical_cols=["cat_code", "label"],
        numerical_cols=["x_cont", "n_visits"],
    )
    assert md == {
        "METADATA_SPEC_VERSION": "V1",
        "tables": {
            "table": {
                "columns": {
                    "x_cont": {"sdtype": "numerical"},
                    "cat_code": {"sdtype": "categorical"},   # integer-coded nominal feature
                    "n_visits": {"sdtype": "numerical"},
                    "label": {"sdtype": "categorical"},      # integer class target
                }
            }
        },
        "relationships": [],
    }
    with pytest.raises(ValueError, match="no declared role"):
        build_sdv_metadata_dict(["a", "b"], categorical_cols=["a"], numerical_cols=[])
    with pytest.raises(ValueError, match="both categorical and numerical"):
        build_sdv_metadata_dict(["a"], categorical_cols=["a"], numerical_cols=["a"])


def test_wrapper_declares_integer_class_target_and_codes_categorical(mixed_frame):
    m = fitted(mixed_frame)
    cols = m.metadata_dict_["tables"]["table"]["columns"]
    assert cols["label"] == {"sdtype": "categorical"} and cols["cat_code"] == {"sdtype": "categorical"}
    assert cols["n_visits"] == {"sdtype": "numerical"} and cols["x_cont"] == {"sdtype": "numerical"}
    assert list(cols) == list(mixed_frame.columns)
    assert m._model.metadata_dict is m.metadata_dict_, "the synthesizer must be built from exactly this metadata"
    assert m.role_source_ == "explicit"


def test_explicit_roles_never_trigger_dtype_inference(mixed_frame, monkeypatch):
    import sbtab.data.schema as schema_mod

    monkeypatch.setattr(schema_mod, "classify_feature_type",
                        lambda *a, **k: (_ for _ in ()).throw(AssertionError("inference ran")))
    assert len(fitted(mixed_frame).sample(5, seed=0)) == 5


# ----------------------------------------------------------------------
# C3: config validation (pure)
# ----------------------------------------------------------------------


def test_batch_size_must_be_even_and_divisible_by_pac():
    validate_ctgan_batch(500, 10)
    CTGANConfig()                                     # the defaults are valid
    CTGANConfig(batch_size=256, pac=8)
    with pytest.raises(ValueError, match="even batch_size"):
        CTGANConfig(batch_size=25, pac=5)
    with pytest.raises(ValueError, match="batch_size % pac == 0"):
        CTGANConfig(batch_size=256, pac=10)
    with pytest.raises(ValueError, match="pac must be"):
        CTGANConfig(batch_size=256, pac=0)


def test_min_max_clipping_is_off_by_default():
    assert CTGANConfig().enforce_min_max_values is False


def test_declared_update_counts():
    assert ctgan_update_counts(240, 60, 3, 1) == {"steps_per_epoch": 4, "generator_updates": 12, "discriminator_updates": 12}
    assert ctgan_update_counts(10, 60, 3, 5)["steps_per_epoch"] == 1  # ctgan: max(n // batch, 1)
    m = fitted(make_mixed_frame())
    assert m.n_updates == 12 and m.budget_["discriminator_updates"] == 12


def test_default_factory_wiring_against_a_stub_sdv(monkeypatch):
    """Wiring only (a stub `sdv` module, NOT the library): metadata dict + clipping flag reach the constructor."""
    captured = {}

    class StubMetadata:
        @classmethod
        def load_from_dict(cls, d):
            obj = cls()
            obj.d = d
            obj.validated = False
            return obj

        def validate(self):
            self.validated = True

    class StubSynth:
        def __init__(self, metadata, **kwargs):
            captured["metadata"] = metadata
            captured["kwargs"] = kwargs

    sdv = types.ModuleType("sdv")
    sdv.single_table = types.ModuleType("sdv.single_table")
    sdv.single_table.CTGANSynthesizer = StubSynth
    sdv.metadata = types.ModuleType("sdv.metadata")
    sdv.metadata.Metadata = StubMetadata
    for name, mod in (("sdv", sdv), ("sdv.single_table", sdv.single_table), ("sdv.metadata", sdv.metadata)):
        monkeypatch.setitem(sys.modules, name, mod)

    md = build_sdv_metadata_dict(["a", "y"], categorical_cols=["y"], numerical_cols=["a"])
    ctgan_model._default_synthesizer_factory(md, CTGANConfig(epochs=7, batch_size=20, pac=2))
    assert captured["metadata"].d is md and captured["metadata"].validated
    assert captured["kwargs"]["enforce_min_max_values"] is False
    assert captured["kwargs"]["epochs"] == 7 and captured["kwargs"]["pac"] == 2 and captured["kwargs"]["batch_size"] == 20


# ----------------------------------------------------------------------
# C2: seeding
# ----------------------------------------------------------------------


def test_fit_and_sample_are_seeded(mixed_frame):
    torch.manual_seed(987654)
    m = fitted(mixed_frame)
    assert m._model.torch_seed_at_fit == 11, "fit() must seed torch with cfg.seed before training"

    a = m.sample(30, seed=5)
    assert torch.initial_seed() == 5
    assert m._model.reset_calls == 1 and m._model.random_state_seeds == [5]
    assert m.sample_report_["seeding"] == {"reset_sampling": True, "set_random_state": True}
    pd.testing.assert_frame_equal(a, m.sample(30, seed=5))
    assert not a.equals(m.sample(30, seed=6))

    m.sample(3)  # no seed -> the synthesizer's sampling state is left alone
    assert m._model.reset_calls == 3 and m._model.random_state_seeds == [5, 5, 6]


# ----------------------------------------------------------------------
# C5: exactly n rows / C4 ids / decoding
# ----------------------------------------------------------------------


@pytest.mark.parametrize("n", [1, 59, 61])
def test_returns_exactly_n_rows_in_the_fit_schema(mixed_frame, n):
    m = fitted(mixed_frame)
    assert_mixed_output_valid(m.sample(n, seed=0), mixed_frame, n)
    rep = m.decoding_report_
    assert set(rep) == {"n_visits"} and rep["n_visits"]["out_of_support_rate"] > 0.9


def test_short_synthesizer_output_is_topped_up_by_generation_only(mixed_frame):
    m = fitted(mixed_frame, rows_per_call=20)
    out = m.sample(50, seed=0)
    assert len(out) == 50 and m._model.sample_calls == [50, 30, 10]


def test_surplus_synthesizer_output_is_truncated(mixed_frame):
    m = fitted(mixed_frame, surplus=7)
    assert len(m.sample(50, seed=0)) == 50


def test_a_synthesizer_that_returns_nothing_raises_instead_of_using_real_rows(mixed_frame):
    m = fitted(mixed_frame, rows_per_call=0)
    with pytest.raises(RuntimeError, match="Real rows are never used"):
        m.sample(10, seed=0)


def test_real_ids_are_never_emitted(mixed_frame):
    df = mixed_frame.copy()
    df.insert(0, "row_id", np.arange(500, 500 + len(df)))
    m = CTGANWrapper(CTGANConfig(epochs=1), synthesizer_factory=fake_factory()).fit(df, id_col="row_id", **MIXED_ROLES)
    assert "row_id" not in m.metadata_dict_["tables"]["table"]["columns"], "ids must not be modelled"
    out = m.sample(300, seed=0)
    assert list(out.columns) == list(df.columns)
    assert not set(out["row_id"]) & set(df["row_id"]) and out["row_id"].is_unique


def test_row_dropping_transforms_are_not_reapplied_to_generated_rows(mixed_frame):
    class DropRows:
        name = "drop_rows"

        def transform(self, df):
            return df[df["x_cont"] <= 0.0].copy()      # would delete ~half of any generated sample

        def inverse_transform(self, df):
            return df

    class Scale:
        name = "scale"

        def transform(self, df):
            out = df.copy()
            out["x_cont"] = out["x_cont"] * 2.0
            return out

        def inverse_transform(self, df):
            out = df.copy()
            out["x_cont"] = out["x_cont"] / 2.0
            return out

    class Pipe:
        transforms = [DropRows(), Scale()]

        def transform(self, df):
            for t in self.transforms:
                df = t.transform(df)
            return df

        def inverse_transform(self, df):
            for t in reversed(self.transforms):
                df = t.inverse_transform(df)
            return df

    schema = SimpleNamespace(continuous_cols=["x_cont", "x_f32"], discrete_cols=["n_visits"],
                             categorical_cols=["cat_code"], target_col="label", id_col=None)
    m = CTGANWrapper(CTGANConfig(epochs=1), synthesizer_factory=fake_factory())
    m.fit(mixed_frame, schema=schema, transforms=Pipe(), task="classification")
    out = m.sample(80, seed=0)
    assert len(out) == 80, "a row-dropping pipeline step was re-applied to generated rows"
    assert m.sample_report_["skipped_row_dropping_transforms"] == ["drop_rows"]
    assert (out["x_cont"] > 0).any(), "rows the dropper would have removed must survive"
    assert list(out.columns) == list(mixed_frame.columns)


# ----------------------------------------------------------------------
# checkpointing
# ----------------------------------------------------------------------


def test_checkpoint_round_trip_without_refit(mixed_frame, tmp_path):
    m = fitted(mixed_frame)
    expected = m.sample(40, seed=3)
    path = str(tmp_path / "ctgan_ckpt")
    m.save_checkpoint(path)

    loaded = CTGANWrapper.load_checkpoint(path, synthesizer_loader=FakeSynthesizer.load)
    assert loaded.metadata_dict_ == m.metadata_dict_
    assert loaded.label_distribution_ == m.label_distribution_
    assert loaded.n_updates == m.n_updates and loaded.variant_id == "ctgan_sdv"
    assert loaded.cfg == m.cfg
    pd.testing.assert_frame_equal(loaded.sample(40, seed=3), expected)


# ----------------------------------------------------------------------
# real library (skipped when sdv is missing - a skip is NOT a validation of the adapter)
# ----------------------------------------------------------------------


def test_real_sdv_ctgan_end_to_end(mixed_frame, tmp_path):
    pytest.importorskip("sdv", reason=SDV_SKIP_REASON)
    cfg = CTGANConfig(epochs=2, batch_size=60, pac=10, embedding_dim=16, generator_dim=(32, 32),
                      discriminator_dim=(32, 32), enable_gpu=False, seed=1)
    m = CTGANWrapper(cfg).fit(mixed_frame, **MIXED_ROLES)
    out = m.sample(61, seed=4)
    assert_mixed_output_valid(out, mixed_frame, 61)
    pd.testing.assert_frame_equal(out, m.sample(61, seed=4))
    assert not out.equals(m.sample(61, seed=5)), "different sample seeds must give different rows"

    path = str(tmp_path / "ctgan_real")
    m.save_checkpoint(path)
    pd.testing.assert_frame_equal(CTGANWrapper.load_checkpoint(path).sample(61, seed=4), out)


def test_real_sdv_does_not_clip_to_training_range():
    pytest.importorskip("sdv", reason=SDV_SKIP_REASON)
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"x": rng.normal(size=400), "y": rng.integers(0, 2, size=400)})
    cfg = CTGANConfig(epochs=2, batch_size=40, pac=10, enable_gpu=False, seed=1)
    m = CTGANWrapper(cfg).fit(df, continuous_cols=["x"], target_col="y", task="classification")
    assert m._model.enforce_min_max_values is False
