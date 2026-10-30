"""Numerical failures stop before optimization and retain their actual stage."""
import numpy as np
import pandas as pd
import pytest
import torch

from sbtab.baselines.stasy.model import VEScoreSDEBaseline, VEScoreSDEConfig, _langevin_step_size
from sbtab.baselines.tabbyflow.model import (
    TabbyFlowConfig, TabbyFlowMatchingLoss, TabbyFlowSynthesizer,
)
from sbtab.baselines.tabddpm.model import TabDDPMConfig, TabDDPMWrapper
from sbtab.baselines.tabddpm.gaussian_multinomial_diffsuion import GaussianMultinomialDiffusion
from sbtab.data.schema import TabularSchema
from sbtab.numerics import NumericalError


class _BadGradient(torch.autograd.Function):
    @staticmethod
    def forward(ctx, value):
        return value.clone()

    @staticmethod
    def backward(ctx, grad):
        return torch.full_like(grad, float("inf"))


def _bad_scalar(value, kind):
    return value * float("inf") if kind == "loss" else _BadGradient.apply(value)


def _frame():
    return pd.DataFrame({"x": np.linspace(-1, 1, 12), "y": np.linspace(-2, 2, 12)})


def _ve(**kw):
    return VEScoreSDEBaseline(VEScoreSDEConfig(
        steps=3, hidden_dim=8, n_layers=1, time_emb_dim=4, batch_size=12,
        sigma_max=1.0, n_sampling_steps=3, device="cpu", **kw))


@pytest.mark.parametrize("kind", ["loss", "gradient"])
def test_ve_bad_training_never_steps_optimizer_or_keeps_a_stale_network(monkeypatch, kind):
    model = _ve().fit(_frame(), continuous_cols=["x", "y"])

    class BrokenScore(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor(0.25))

        def forward(self, x, sigma):
            return _bad_scalar(self.weight, kind).expand_as(x)

    net = BrokenScore()
    before = net.weight.detach().clone()
    monkeypatch.setattr(model, "_build_net", lambda: net)
    monkeypatch.setattr(torch.optim.Adam, "step", lambda *a, **k: pytest.fail("optimizer stepped"))
    with pytest.raises(NumericalError) as caught:
        model.fit(_frame(), continuous_cols=["x", "y"])
    assert caught.value.details["stage"] == "training"
    assert caught.value.details["step"] == 0
    assert caught.value.details["tensor"] == kind
    torch.testing.assert_close(net.weight, before, rtol=0, atol=0)
    assert model._score_net is None and model.n_updates == 0
    with pytest.raises(RuntimeError, match="Call fit"):
        model.sample(2)


@pytest.mark.parametrize("kind", ["loss", "gradient"])
def test_tabddpm_bad_training_preserves_weights_and_ema(monkeypatch, kind):
    model = TabDDPMWrapper(TabDDPMConfig(steps=2, num_timesteps=4, d_layers=[8], device="cpu"))
    before = {}

    def broken_loss(diffusion, x, out_dict):
        before["weights"] = {k: v.detach().clone() for k, v in diffusion._denoise_fn.state_dict().items()}
        before["ema"] = {k: v.detach().clone() for k, v in model.ema_model.state_dict().items()}
        value = next(diffusion._denoise_fn.parameters()).flatten()[0]
        return _bad_scalar(value, kind), value.new_zeros(())

    monkeypatch.setattr(GaussianMultinomialDiffusion, "mixed_loss", broken_loss)
    monkeypatch.setattr(torch.optim.AdamW, "step", lambda *a, **k: pytest.fail("optimizer stepped"))
    with pytest.raises(NumericalError) as caught:
        model.fit(_frame(), continuous_cols=["x", "y"])
    assert caught.value.details["tensor"] == kind
    assert not model._fitted and model.n_updates == 0
    for name, module in [("weights", model.diffusion._denoise_fn), ("ema", model.ema_model)]:
        for key, value in module.state_dict().items():
            torch.testing.assert_close(value, before[name][key], rtol=0, atol=0)


@pytest.mark.parametrize("kind", ["loss", "gradient"])
def test_tabbyflow_bad_training_preserves_weights(monkeypatch, kind):
    model = TabbyFlowSynthesizer(TabbyFlowConfig(max_train_steps=2, n_frequencies=4, device="cpu"))
    before = {}

    def broken_loss(loss_fn, net, x):
        before.update({k: v.detach().clone() for k, v in net.state_dict().items()})
        return _bad_scalar(next(net.parameters()).flatten()[0], kind)

    monkeypatch.setattr(TabbyFlowMatchingLoss, "forward", broken_loss)
    monkeypatch.setattr(torch.optim.Adam, "step", lambda *a, **k: pytest.fail("optimizer stepped"))
    with pytest.raises(NumericalError) as caught:
        model.fit(_frame(), schema=TabularSchema(["x", "y"], [], []), task_type="regression")
    assert caught.value.details["tensor"] == kind
    assert not model._fitted and model.n_updates == 0
    for key, value in model.net.state_dict().items():
        torch.testing.assert_close(value, before[key], rtol=0, atol=0)


@pytest.mark.parametrize("bad_call, expected", [(1, "corrector_score"), (2, "predictor_score"), (7, "denoise_score")])
def test_ve_nonfinite_score_reports_exact_stage_and_noise_level(bad_call, expected):
    model = _ve()
    model._dim = 2

    class ControlledScore(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.calls = 0

        def forward(self, x, sigma):
            self.calls += 1
            return torch.full_like(x, float("nan")) if self.calls == bad_call else -x

    model._score_net = ControlledScore()
    with pytest.raises(NumericalError) as caught:
        model._sample_encoded(2, seed=0)
    details = caught.value.details
    assert details["stage"] == "sampling" and details["tensor"] == expected
    assert details["step"] == (3 if expected == "denoise_score" else 0)
    assert np.isfinite(details["sigma"]) and 0 < details["sigma"] <= 1
    assert model._score_net.calls == bad_call


def test_ve_corrector_norm_does_not_overflow_for_finite_float32_scores():
    score = torch.full((3, 4), 1e20, dtype=torch.float32)
    noise = torch.ones_like(score)
    assert torch.isinf(score.norm(dim=-1)).all()  # the old reduction silently yielded alpha=0
    alpha = _langevin_step_size(score, noise, 0.16)
    expected = 2.0 * (0.16 * 2 / (2 * float(score[0, 0]))) ** 2
    assert alpha.dtype == torch.float64 and torch.isfinite(alpha) and alpha > 0
    assert float(alpha) == pytest.approx(expected, rel=1e-12, abs=0)
    update = alpha * score + (2 * alpha).sqrt() * noise
    assert torch.isfinite(update).all() and (update != 0).all()


@pytest.mark.parametrize("sampler", ["sample", "sample_ddim"])
@pytest.mark.parametrize("kind", ["denoiser_output", "gaussian_state"])
def test_tabddpm_sampling_reports_first_bad_timestep(monkeypatch, sampler, kind):
    class Score(torch.nn.Module):
        def forward(self, x, t, **kwargs):
            return torch.full_like(x, float("nan")) if kind == "denoiser_output" else torch.zeros_like(x)

    diffusion = GaussianMultinomialDiffusion(np.array([0]), 2, Score(), num_timesteps=4)
    if kind == "gaussian_state":
        if sampler == "sample":
            monkeypatch.setattr(diffusion, "gaussian_p_sample", lambda score, x, *a, **k: {"sample": x * float("inf")})
        else:
            monkeypatch.setattr(diffusion, "gaussian_ddim_step", lambda score, x, *a, **k: x * float("inf"))
    with pytest.raises(NumericalError) as caught:
        getattr(diffusion, sampler)(2, torch.ones(1))
    assert caught.value.details["tensor"] == kind
    assert caught.value.details["timestep"] == 3
    assert caught.value.details["sampler"] == ("ddpm" if sampler == "sample" else "ddim")


def test_ve_finite_smoke_and_checkpoint_multiple_sample_sizes(tmp_path):
    model = _ve().fit(_frame(), continuous_cols=["x", "y"])
    model.save_checkpoint(tmp_path / "ve.pt")
    restored = VEScoreSDEBaseline.load_checkpoint(tmp_path / "ve.pt", device="cpu")
    for n in (1, 2, 21, 64):
        actual = model.sample(n, seed=7)
        assert np.isfinite(actual.to_numpy()).all()
        pd.testing.assert_frame_equal(actual, restored.sample(n, seed=7))
