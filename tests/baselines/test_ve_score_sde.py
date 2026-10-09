"""
Tests for the simplified VE score-SDE baseline (formerly mis-advertised as STaSy).
Tiny budgets: contracts only, not sample quality.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest
import torch

import sbtab.baselines.stasy as ve_pkg
from sbtab.baselines.stasy import (
    FAITHFULNESS,
    STaSyConfig,
    STaSyGenerative,
    VEScoreSDEBaseline,
    VEScoreSDEConfig,
)

from baseline_testkit import MIXED_ROLES, assert_mixed_output_valid, make_mixed_frame

TINY = dict(hidden_dim=32, n_layers=2, time_emb_dim=16, batch_size=64, lr=3e-3, sigma_max=10.0,
            n_sampling_steps=25, device="cpu")


def tiny_cfg(**over) -> VEScoreSDEConfig:
    kw = {**TINY, "steps": 40, "seed": 3}
    kw.update(over)
    if "n_epochs" in over:
        kw.pop("steps", None)
    return VEScoreSDEConfig(**kw)


# ----------------------------------------------------------------------
# S1: honest naming
# ----------------------------------------------------------------------


def test_faithfulness_declaration():
    assert FAITHFULNESS["is_faithful_stasy"] is False
    assert FAITHFULNESS["variant_id"] == "ve_score_sde_simplified" == VEScoreSDEBaseline.variant_id
    missing = " ".join(FAITHFULNESS["missing"]).lower()
    for component in ("self-paced", "fine-tuning", "sub-vp", "probability-flow", "ncsnpp"):
        assert component in missing, f"{component!r} must be declared as missing"
    assert FAITHFULNESS["present"], "present components must be listed too"
    assert "NOT STaSy" in ve_pkg.model.__doc__


def test_deprecated_aliases_warn_that_this_is_not_stasy():
    with pytest.warns(DeprecationWarning, match="NOT a faithful implementation of STaSy"):
        cfg = STaSyConfig(n_epochs=1, **TINY)
    with pytest.warns(DeprecationWarning, match="NOT a faithful implementation of STaSy"):
        model = STaSyGenerative(cfg)
    assert isinstance(cfg, VEScoreSDEConfig) and isinstance(model, VEScoreSDEBaseline)
    assert model.variant_id == "ve_score_sde_simplified"


def test_deprecated_use_self_paced_maps_to_time_curriculum():
    with pytest.warns(DeprecationWarning, match="NOT STaSy's per-sample self-paced"):
        cfg = STaSyConfig(n_epochs=2, use_self_paced=True)
    assert cfg.time_curriculum is True and cfg.use_self_paced is True
    with pytest.warns(DeprecationWarning):
        legacy_default = STaSyConfig()           # old call sites passed no budget
    assert legacy_default.n_epochs == 100 and legacy_default.time_curriculum is False
    with pytest.warns(DeprecationWarning), pytest.raises(TypeError):
        STaSyConfig(n_epochs=1, use_self_paced=True, time_curriculum=False)


# ----------------------------------------------------------------------
# S2: curriculum / budget
# ----------------------------------------------------------------------


def test_time_curriculum_is_off_by_default():
    cfg = VEScoreSDEConfig(steps=100)
    assert cfg.time_curriculum is False
    assert all(cfg.curriculum_t_max(s, 100) == 1.0 for s in (0, 1, 50, 99))


def test_enabled_curriculum_finishes_by_half_of_training():
    cfg = VEScoreSDEConfig(steps=100, time_curriculum=True, sp_start_ratio=0.25)
    t_max = [cfg.curriculum_t_max(s, 100) for s in range(100)]
    assert t_max[0] == pytest.approx(0.25)
    assert all(b >= a for a, b in zip(t_max, t_max[1:])), "ramp must be monotone"
    assert t_max[49] < 1.0 and t_max[50] == pytest.approx(1.0)
    assert all(v == 1.0 for v in t_max[50:]), "the second half of training must see the full noise range"


def test_budget_is_in_optimizer_steps_and_keeps_partial_batches(mixed_frame):
    with pytest.raises(ValueError, match="Ambiguous training budget"):
        VEScoreSDEConfig(steps=5, n_epochs=1)
    with pytest.raises(ValueError, match="No training budget"):
        VEScoreSDEConfig()

    m = VEScoreSDEBaseline(tiny_cfg(steps=7)).fit(mixed_frame, **MIXED_ROLES)
    assert m.total_steps_ == 7 and m.n_updates == 7
    # 240 rows / batch 64 -> 4 batches per epoch, the partial one included
    m = VEScoreSDEBaseline(tiny_cfg(n_epochs=3)).fit(mixed_frame, **MIXED_ROLES)
    assert m.n_batches_per_epoch_ == 4 and m.total_steps_ == 12 and m.n_updates == 12


# ----------------------------------------------------------------------
# S4: schema awareness
# ----------------------------------------------------------------------


def test_mixed_types_are_encoded_and_decoded(mixed_frame):
    m = VEScoreSDEBaseline(tiny_cfg()).fit(mixed_frame, **MIXED_ROLES)
    assert m._dim == 3 + 4 + 3, "3 numeric + one-hot(cat_code: 4) + one-hot(label: 3)"
    out = m.sample(150, seed=0)
    assert_mixed_output_valid(out, mixed_frame, 150)
    assert 2 not in set(out["cat_code"])

    rep = m.decoding_report_
    assert set(rep) == {"n_visits", "cat_code", "label"}, "continuous columns are never decoded/snapped"
    assert rep["n_visits"]["kind"] == "nearest_support" and rep["n_visits"]["out_of_support_rate"] > 0.9
    assert rep["cat_code"]["kind"] == "argmax" and rep["cat_code"]["n_classes"] == 4
    assert m.label_distribution_[0] == [0, int((mixed_frame.label == 0).sum())]


def test_string_labels_round_trip():
    df = make_mixed_frame()
    df["label"] = df["label"].map({0: "neg", 1: "pos", 2: "unk"})
    m = VEScoreSDEBaseline(tiny_cfg(steps=5)).fit(df, **MIXED_ROLES)
    out = m.sample(40, seed=0)
    assert out["label"].dtype == object and set(out["label"]) <= {"neg", "pos", "unk"}


def test_explicit_roles_never_trigger_dtype_inference(mixed_frame, monkeypatch):
    import sbtab.data.schema as schema_mod

    monkeypatch.setattr(schema_mod, "classify_feature_type",
                        lambda *a, **k: (_ for _ in ()).throw(AssertionError("inference ran")))
    assert len(VEScoreSDEBaseline(tiny_cfg(steps=3)).fit(mixed_frame, **MIXED_ROLES).sample(4, seed=0)) == 4


def test_undeclared_frame_defaults_to_all_continuous_with_a_warning():
    rng = np.random.default_rng(0)
    df = pd.DataFrame(rng.normal(size=(80, 3)), columns=["a", "b", "c"])
    with pytest.warns(UserWarning, match="treated as CONTINUOUS"):
        m = VEScoreSDEBaseline(tiny_cfg(steps=3)).fit(df)
    assert m.role_source_ == "default_all_continuous" and m.decoding_report_ == {}
    assert list(m.sample(5, seed=0).columns) == ["a", "b", "c"]

    arr_model = VEScoreSDEBaseline(tiny_cfg(steps=3)).fit(df.to_numpy())
    out = arr_model.sample(6, seed=0)
    assert isinstance(out, np.ndarray) and out.shape == (6, 3)

    with pytest.raises(TypeError, match="unexpected keyword"):
        VEScoreSDEBaseline(tiny_cfg(steps=3)).fit(df, continuous_col=["a"])


# ----------------------------------------------------------------------
# exact n, seeding, ids, checkpoint
# ----------------------------------------------------------------------


@pytest.mark.parametrize("n", [1, 63, 65])
def test_returns_exactly_n_rows(mixed_frame, n):
    m = VEScoreSDEBaseline(tiny_cfg(steps=5)).fit(mixed_frame, **MIXED_ROLES)
    assert_mixed_output_valid(m.sample(n, seed=0), mixed_frame, n)


def test_seeded_fit_and_sample_are_reproducible(mixed_frame):
    def run(seed: int, global_state: int) -> VEScoreSDEBaseline:
        torch.manual_seed(global_state)   # different global states: only fit() reseeding can align them
        np.random.seed(global_state)
        return VEScoreSDEBaseline(tiny_cfg(seed=seed)).fit(mixed_frame, **MIXED_ROLES)

    a, b, c = run(3, 111), run(3, 222), run(4, 111)
    pd.testing.assert_frame_equal(a.sample(32, seed=5), b.sample(32, seed=5))
    assert not a.sample(32, seed=5).equals(c.sample(32, seed=5)), "cfg.seed must matter"
    assert not a.sample(32, seed=5).equals(a.sample(32, seed=6)), "sample seed must matter"

    # sampling noise comes from a private generator: the global RNG state is irrelevant
    torch.manual_seed(1)
    first = a.sample(16, seed=7)
    torch.manual_seed(2)
    pd.testing.assert_frame_equal(first, a.sample(16, seed=7))


def test_real_ids_are_never_emitted(mixed_frame):
    df = mixed_frame.copy()
    df.insert(2, "row_id", [f"patient_{i}" for i in range(len(df))])
    m = VEScoreSDEBaseline(tiny_cfg(steps=5)).fit(df, id_col="row_id", **MIXED_ROLES)
    out = m.sample(300, seed=0)
    assert list(out.columns) == list(df.columns)
    assert not set(out["row_id"]) & set(df["row_id"]) and out["row_id"].is_unique


def test_checkpoint_reload_reproduces_samples_exactly(mixed_frame, tmp_path):
    m = VEScoreSDEBaseline(tiny_cfg(n_epochs=4)).fit(mixed_frame, **MIXED_ROLES)
    expected = m.sample(65, seed=9)
    path = str(tmp_path / "ve.pt")
    m.save_checkpoint(path)

    state = torch.load(path, map_location="cpu", weights_only=True)
    assert state["variant_id"] == "ve_score_sde_simplified" and state["is_faithful_stasy"] is False
    assert state["total_steps"] == 16 and state["n_updates"] == 16

    loaded = VEScoreSDEBaseline.load_checkpoint(path)
    assert loaded.total_steps_ == 16 and loaded.n_updates == 16
    assert loaded.label_distribution_ == m.label_distribution_
    pd.testing.assert_frame_equal(loaded.sample(65, seed=9), expected)


def test_alias_checkpoint_loads_as_the_honest_class(mixed_frame, tmp_path):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        legacy = STaSyGenerative(STaSyConfig(n_epochs=1, **TINY)).fit(mixed_frame, **MIXED_ROLES)
    path = str(tmp_path / "legacy.pt")
    legacy.save_checkpoint(path)
    loaded = STaSyGenerative.load_checkpoint(path)
    assert type(loaded) is VEScoreSDEBaseline
    pd.testing.assert_frame_equal(loaded.sample(8, seed=1), legacy.sample(8, seed=1))
