"""
TabDDPM wrapper tests.  Budgets are deliberately tiny (<= 300 optimizer steps, <= 50 diffusion
timesteps, 2-layer MLP): these tests check CONTRACTS (budget, seeding, types, exact n, scale
handling, checkpointing), not sample quality.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest
import torch

from sbtab.baselines.tabddpm import (
    FoundNANsError,
    TabDDPMConfig,
    TabDDPMWrapper,
    ema_decay_at,
)
from sbtab.baselines.tabddpm.modules import MLP

from baseline_testkit import MIXED_ROLES, assert_mixed_output_valid, assert_same_schema

TINY = dict(num_timesteps=20, batch_size=64, d_layers=[32, 32], lr=2e-3, device="cpu")


def tiny_cfg(**over) -> TabDDPMConfig:
    kw = {**TINY, "steps": 30, "seed": 3}
    kw.update(over)
    if "n_epochs" in over:
        kw.pop("steps", None)
        kw["n_epochs"] = over["n_epochs"]
    return TabDDPMConfig(**kw)


# ----------------------------------------------------------------------
# D1: one explicit training budget
# ----------------------------------------------------------------------


def test_budget_both_or_neither_raise():
    with pytest.raises(ValueError, match="Ambiguous training budget"):
        TabDDPMConfig(steps=10, n_epochs=2)
    with pytest.raises(ValueError, match="No training budget"):
        TabDDPMConfig()
    with pytest.raises(ValueError, match="positive integer"):
        TabDDPMConfig(steps=0)

    # a config mutated after construction is re-validated when the budget is resolved
    cfg = TabDDPMConfig(steps=10)
    cfg.n_epochs = 3
    with pytest.raises(ValueError, match="Ambiguous training budget"):
        cfg.effective_budget(4)


def test_effective_budget_formula():
    assert TabDDPMConfig(steps=7).effective_budget(4) == 7
    assert TabDDPMConfig(n_epochs=3).effective_budget(4) == 12
    assert TabDDPMConfig(n_epochs=3).effective_budget(0) == 3  # an epoch is never less than one step


def test_n_epochs_really_sets_the_training_budget(mixed_frame):
    # 240 rows / batch 64 -> 4 batches per epoch (the partial 48-row batch is kept)
    for n_epochs, expected in ((2, 8), (3, 12)):
        m = TabDDPMWrapper(tiny_cfg(n_epochs=n_epochs)).fit(mixed_frame, **MIXED_ROLES)
        assert m.n_batches_per_epoch_ == 4
        assert m.total_steps_ == expected
        assert m.n_updates == expected


def test_steps_really_sets_the_training_budget(mixed_frame):
    for steps in (5, 11):
        m = TabDDPMWrapper(tiny_cfg(steps=steps)).fit(mixed_frame, **MIXED_ROLES)
        assert m.total_steps_ == steps and m.n_updates == steps


# ----------------------------------------------------------------------
# D5: seeding
# ----------------------------------------------------------------------


def test_seeded_fit_is_reproducible(mixed_frame):
    def run(seed: int, global_state: int) -> pd.DataFrame:
        # the two "same seed" runs start from DIFFERENT global RNG states: only fit() reseeding
        # with cfg.seed (torch + numpy + batch order) can make them agree
        torch.manual_seed(global_state)
        np.random.seed(global_state)
        m = TabDDPMWrapper(tiny_cfg(seed=seed)).fit(mixed_frame, **MIXED_ROLES)
        return m.sample(32, seed=5)

    a, b, c = run(3, 111), run(3, 222), run(4, 111)
    pd.testing.assert_frame_equal(a, b)
    assert not a.equals(c), "a different cfg.seed must give a different model"


def test_sample_seed_is_effective(mixed_frame):
    m = TabDDPMWrapper(tiny_cfg()).fit(mixed_frame, **MIXED_ROLES)
    pd.testing.assert_frame_equal(m.sample(16, seed=1), m.sample(16, seed=1))
    assert not m.sample(16, seed=1).equals(m.sample(16, seed=2))


# ----------------------------------------------------------------------
# D4: explicit roles, types, order
# ----------------------------------------------------------------------


def test_integer_coded_categoricals_and_class_target_stay_integers_in_vocabulary(mixed_frame):
    m = TabDDPMWrapper(tiny_cfg()).fit(mixed_frame, **MIXED_ROLES)
    out = m.sample(200, seed=0)

    assert_mixed_output_valid(out, mixed_frame, 200)
    assert out["cat_code"].dtype == np.int64 and out["label"].dtype == np.int64
    assert 2 not in set(out["cat_code"]), "code 2 never occurs in training and must never be generated"
    assert out["x_f32"].dtype == np.float32

    # both went through the MULTINOMIAL block, only the 3 numeric columns through the Gaussian one
    assert m.num_numerical_features == 3
    assert m.num_classes.tolist() == [4, 3]
    assert m.role_source_ == "explicit"
    assert m.label_distribution_ == [[0, int((mixed_frame.label == 0).sum())],
                                     [1, int((mixed_frame.label == 1).sum())],
                                     [2, int((mixed_frame.label == 2).sum())]]


def test_discrete_column_is_decoded_to_training_support_with_report(mixed_frame):
    m = TabDDPMWrapper(tiny_cfg()).fit(mixed_frame, **MIXED_ROLES)
    out = m.sample(300, seed=0)

    assert set(out["n_visits"].unique()) <= {0, 5, 10, 50}
    rep = m.decoding_report_["n_visits"]
    assert rep["n"] == 300 and rep["support_size"] == 4
    # a Gaussian diffusion output is a real number: before decoding essentially nothing sits on the support
    assert rep["out_of_support_rate"] > 0.9
    assert 0.0 <= rep["out_of_range_rate"] <= 1.0
    assert rep["max_abs_shift"] >= rep["mean_abs_shift"] > 0.0
    assert set(m.decoding_report_) == {"n_visits"}, "continuous columns must not be decoded/snapped"


def test_explicit_roles_never_trigger_dtype_inference(mixed_frame, monkeypatch):
    import sbtab.data.schema as schema_mod

    def boom(*a, **k):
        raise AssertionError("dtype/cardinality inference ran although explicit roles were given")

    monkeypatch.setattr(schema_mod, "classify_feature_type", boom)
    m = TabDDPMWrapper(tiny_cfg(steps=3)).fit(mixed_frame, **MIXED_ROLES)
    assert len(m.sample(5, seed=0)) == 5


def test_missing_or_misspelled_roles_are_errors(mixed_frame):
    with pytest.raises(ValueError, match="Column roles are required"):
        TabDDPMWrapper(tiny_cfg()).fit(mixed_frame)
    with pytest.raises(ValueError, match="without an explicit role"):
        TabDDPMWrapper(tiny_cfg()).fit(mixed_frame, continuous_cols=["x_cont"], target_col="label", task="classification")
    with pytest.raises(TypeError, match="unexpected keyword"):
        TabDDPMWrapper(tiny_cfg()).fit(mixed_frame, categorical_col=["cat_code"], **MIXED_ROLES)


def test_legacy_schema_path_still_works_and_flags_inference(mixed_frame):
    from sbtab.data.schema import TabularSchema

    schema = TabularSchema(
        continuous_cols=["x_cont", "x_f32"], discrete_cols=["n_visits"], categorical_cols=["cat_code"],
        target_col="label",
    )
    with pytest.warns(UserWarning, match="INFERRED from dtype/cardinality"):
        m = TabDDPMWrapper(tiny_cfg(steps=3)).fit(mixed_frame, schema=schema)
    assert m.role_source_ == "schema+inferred_target"
    assert_same_schema(m.sample(9, seed=0), mixed_frame)

    # task= removes the inference (and the warning)
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        m2 = TabDDPMWrapper(tiny_cfg(steps=3)).fit(mixed_frame, schema=schema, task="classification")
    assert m2.role_source_ == "schema"
    assert m2.num_classes.tolist() == [4, 3]


# ----------------------------------------------------------------------
# D3: numeric block scale handling
# ----------------------------------------------------------------------


def test_numeric_block_is_scale_invariant():
    """
    Multiplying ONE column by 1e5 must not wreck the OTHER columns.

    Setup (all seeds fixed): 512 rows, 3 independent Gaussian columns, cfg.seed=0, 300 steps,
    50 timesteps, MLP [64, 64], lr=1e-2, 512 samples drawn with sample seed 11.

    Threshold derivation: the standard error of a mean of 512 draws is sigma / sqrt(512) =
    0.044 sigma.  We allow 3 SE (0.13 sigma) of sampling noise plus the same amount again for the
    bias of a 300-step model: TAU = 0.25 sigma.  (Measured while writing the test: <= 0.09 sigma over
    cfg.seed in 0..5.  With the internal standardiser disabled the same run is off by 11 .. 3900
    sigma on the untouched columns, so the threshold separates the two regimes by > 40x.)
    """
    TAU = 0.25
    rng = np.random.default_rng(7)
    n = 512
    base = pd.DataFrame({"a": rng.normal(size=n), "b": rng.normal(size=n) * 2 + 3, "c": rng.normal(size=n) * 0.5 - 2})
    big = base.copy()
    big["a"] = big["a"] * 1e5

    def run(df: pd.DataFrame) -> pd.DataFrame:
        cfg = TabDDPMConfig(steps=300, num_timesteps=50, batch_size=128, d_layers=[64, 64], lr=1e-2, device="cpu", seed=0)
        return TabDDPMWrapper(cfg).fit(df, continuous_cols=list(df.columns)).sample(512, seed=11)

    s_base, s_big = run(base), run(big)

    z_big = ((s_big.mean() - big.mean()) / big.std()).abs()
    assert (z_big < TAU).all(), f"sample means off by {z_big.to_dict()} sigma"
    ratio = s_big.std() / big.std()
    assert ((ratio > 0.5) & (ratio < 2.0)).all(), f"sample std ratio {ratio.to_dict()}"

    # The standardised training matrices of `base` and `big` are mathematically identical, so the
    # untouched columns must agree between the two fits; 1 SE (0.044 sigma) absorbs float rounding.
    drift = ((s_base[["b", "c"]].mean() - s_big[["b", "c"]].mean()) / base[["b", "c"]].std()).abs()
    assert (drift < 0.044).all(), f"rescaling column a moved the other columns by {drift.to_dict()} sigma"


def test_standardiser_is_fitted_on_fit_rows_and_handles_zero_variance():
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"x": rng.normal(1000.0, 10.0, size=100), "const": np.full(100, 7.0), "k": np.full(100, 3, dtype=np.int64)})
    m = TabDDPMWrapper(tiny_cfg(steps=5)).fit(df, continuous_cols=["x", "const"], discrete_cols=["k"])
    assert m._num_mean[0] == pytest.approx(df["x"].mean())
    assert m._num_scale[0] == pytest.approx(df["x"].std(ddof=0))
    assert m._num_scale[1] == 1.0 and m._num_scale[2] == 1.0  # zero variance -> scale 1 (deterministic)
    out = m.sample(20, seed=0)
    assert np.isfinite(out.to_numpy(dtype=float)).all()
    assert (out["k"] == 3).all() and out["k"].dtype == np.int64


def test_nan_input_is_rejected(mixed_frame):
    bad = mixed_frame.copy()
    bad.loc[3, "x_cont"] = np.nan
    with pytest.raises(ValueError, match="NaN"):
        TabDDPMWrapper(tiny_cfg()).fit(bad, **MIXED_ROLES)


# ----------------------------------------------------------------------
# exact n, D11
# ----------------------------------------------------------------------


@pytest.mark.parametrize("n", [1, 63, 65])  # 1, batch-1, batch+1 for batch_size=64
def test_returns_exactly_n_rows(mixed_frame, n):
    m = TabDDPMWrapper(tiny_cfg(steps=5)).fit(mixed_frame, **MIXED_ROLES)
    assert_mixed_output_valid(m.sample(n, seed=0), mixed_frame, n)


def test_never_denoises_more_rows_than_needed(mixed_frame, monkeypatch):
    m = TabDDPMWrapper(tiny_cfg(steps=5)).fit(mixed_frame, **MIXED_ROLES)
    chunks = []
    original = m.diffusion.sample

    def spy(num_samples, y_dist):
        chunks.append(int(num_samples))
        return original(num_samples, y_dist)

    monkeypatch.setattr(m.diffusion, "sample", spy)
    m.sample(1, seed=0)
    m.sample(65, seed=0)
    assert chunks == [1, 64, 1]


def test_sampling_prints_nothing(mixed_frame, capsys):
    m = TabDDPMWrapper(tiny_cfg(steps=5)).fit(mixed_frame, **MIXED_ROLES)
    capsys.readouterr()
    m.sample(10, seed=0)
    captured = capsys.readouterr()
    assert captured.out == "" and captured.err == ""


@pytest.mark.parametrize("bad_n", [0, -3])
def test_invalid_n(mixed_frame, bad_n):
    m = TabDDPMWrapper(tiny_cfg(steps=2)).fit(mixed_frame, **MIXED_ROLES)
    with pytest.raises(ValueError):
        m.sample(bad_n)


# ----------------------------------------------------------------------
# D10: ids
# ----------------------------------------------------------------------


def test_real_ids_are_never_emitted(mixed_frame):
    df = mixed_frame.copy()
    df.insert(1, "row_id", np.arange(1000, 1000 + len(df)))
    m = TabDDPMWrapper(tiny_cfg(steps=5)).fit(df, id_col="row_id", **MIXED_ROLES)
    out = m.sample(400, seed=0)
    assert list(out.columns) == list(df.columns)
    assert out["row_id"].is_unique
    assert not set(out["row_id"]) & set(df["row_id"]), "synthetic rows carry real training ids"


# ----------------------------------------------------------------------
# D6 / D7 / D8 and config validation
# ----------------------------------------------------------------------


def test_gaussian_loss_type_is_passed_through(mixed_frame):
    m = TabDDPMWrapper(tiny_cfg(steps=3, gaussian_loss_type="kl")).fit(mixed_frame, **MIXED_ROLES)
    assert m.diffusion.gaussian_loss_type == "kl"
    assert np.isfinite(m.final_loss_)
    assert TabDDPMWrapper(tiny_cfg(steps=3)).fit(mixed_frame, **MIXED_ROLES).diffusion.gaussian_loss_type == "mse"
    with pytest.raises(ValueError, match="gaussian_loss_type"):
        tiny_cfg(gaussian_loss_type="huber")


def test_found_nans_error_is_a_regular_exception():
    assert issubclass(FoundNANsError, Exception)
    try:
        raise FoundNANsError()
    except Exception as e:  # an Optuna objective's `except Exception` must be able to catch it
        assert "NAN" in str(e).upper()


def test_integer_dropout_from_json_is_accepted(mixed_frame):
    cfg = tiny_cfg(steps=2, dropout=0)  # e.g. json.loads('{"dropout": 0}')
    assert isinstance(cfg.dropout, float) and cfg.dropout == 0.0
    assert TabDDPMWrapper(cfg).fit(mixed_frame, **MIXED_ROLES).n_updates == 2
    assert MLP.make_baseline(4, [8], 0, 2)(torch.zeros(3, 4)).shape == (3, 2)
    with pytest.raises(ValueError, match="dropout"):
        tiny_cfg(dropout=1.5)


def test_d_layers_are_validated_early_with_a_clear_error():
    with pytest.raises(ValueError, match="except the first and the last must be equal"):
        tiny_cfg(d_layers=[32, 64, 128, 32])
    with pytest.raises(ValueError, match="at least one"):
        tiny_cfg(d_layers=[])
    assert tiny_cfg(d_layers=[16, 64, 64, 32]).d_layers == [16, 64, 64, 32]
    with pytest.raises(ValueError, match="num_timesteps > 20"):
        tiny_cfg(scheduler="linear", num_timesteps=20)


# ----------------------------------------------------------------------
# EMA
# ----------------------------------------------------------------------


def test_ema_warmup_schedule():
    assert ema_decay_at(0, 0.999, True) == pytest.approx(0.1)
    assert ema_decay_at(90, 0.999, True) == pytest.approx(0.91)
    assert ema_decay_at(10**6, 0.999, True) == pytest.approx(0.999)
    assert ema_decay_at(0, 0.999, False) == pytest.approx(0.999)


def test_short_run_ema_is_not_dominated_by_the_random_init(mixed_frame):
    """
    Same seed => identical raw weights in both runs.  Without warm-up the EMA after 40 updates is
    0.999**40 = 96% random init, i.e. ||ema - raw|| ~ ||init - raw||.  With warm-up the weight left
    on the init is 40! * 9! / 49! < 1e-9, so the EMA must sit much closer to the raw weights.
    """
    def dist_ema_raw(warmup: bool) -> float:
        m = TabDDPMWrapper(tiny_cfg(steps=40, ema_warmup=warmup)).fit(mixed_frame, **MIXED_ROLES)
        raw = torch.cat([p.detach().flatten() for p in m.diffusion._denoise_fn.parameters()])
        ema = torch.cat([p.detach().flatten() for p in m.ema_model.parameters()])
        return float((raw - ema).norm())

    with_warmup, without = dist_ema_raw(True), dist_ema_raw(False)
    assert with_warmup > 0.0, "EMA must not simply equal the raw weights"
    assert with_warmup < 0.5 * without


def test_sampling_uses_ema_weights_in_eval_mode(mixed_frame):
    m = TabDDPMWrapper(tiny_cfg(steps=40, dropout=0.2)).fit(mixed_frame, **MIXED_ROLES)
    ema_out = m.sample(64, seed=1)
    raw_out = m.sample(64, seed=1, use_ema=False)
    assert not ema_out.equals(raw_out)
    assert not m.diffusion.training and not m.ema_model.training
    # eval mode => dropout inactive => a repeated seeded call is identical
    pd.testing.assert_frame_equal(ema_out, m.sample(64, seed=1))


# ----------------------------------------------------------------------
# checkpointing
# ----------------------------------------------------------------------


def test_checkpoint_reload_reproduces_samples_exactly(mixed_frame, tmp_path):
    df = mixed_frame.copy()
    df.insert(0, "row_id", np.arange(len(df)))
    m = TabDDPMWrapper(tiny_cfg(n_epochs=5)).fit(df, id_col="row_id", **MIXED_ROLES)
    expected = m.sample(65, seed=9)

    path = tmp_path / "tabddpm.pt"
    m.save_checkpoint(str(path))
    state = torch.load(path, map_location="cpu", weights_only=True)
    assert state["total_steps"] == 20 and state["n_updates"] == 20      # resolved budget persisted
    assert state["cfg"]["n_epochs"] == 5 and state["cfg"]["steps"] is None
    assert state["label_distribution"] == m.label_distribution_
    raw, ema = state["denoiser_raw"], state["denoiser_ema"]
    assert raw.keys() == ema.keys()
    assert any(not torch.equal(raw[k], ema[k]) for k in raw), "raw and EMA weights must both be stored"

    loaded = TabDDPMWrapper.load_checkpoint(str(path), device="cpu")
    assert loaded.total_steps_ == 20 and loaded.n_updates == 20
    assert loaded.variant_id == m.variant_id == "tabddpm_mlp_joint_xy"
    got = loaded.sample(65, seed=9)
    pd.testing.assert_frame_equal(got, expected)  # bit-exact on CPU, exactly n rows, same dtypes
    assert len(got) == 65
    assert loaded.decoding_report_.keys() == m.decoding_report_.keys()
