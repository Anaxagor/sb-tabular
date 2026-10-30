"""Analytic oracles for the active Gaussian/multinomial and VE samplers."""
import math

import numpy as np
import pytest
import torch

from sbtab.baselines.tabddpm.gaussian_multinomial_diffsuion import GaussianMultinomialDiffusion
from sbtab.baselines.tabddpm.utils import index_to_log_onehot, sliced_logsumexp
from sbtab.baselines.stasy.model import VEScoreSDEBaseline, VEScoreSDEConfig


def _diffusion():
    return GaussianMultinomialDiffusion(
        num_classes=np.array([2, 3]), num_numerical_features=1,
        denoise_fn=torch.nn.Identity(), num_timesteps=10,
    )


def test_multinomial_posterior_matches_enumerated_markov_chain():
    diffusion = _diffusion()
    start = index_to_log_onehot(torch.tensor([[1, 0]]), diffusion.num_classes)
    observed = index_to_log_onehot(torch.tensor([[0, 2]]), diffusion.num_classes)
    for t in (0, 1, 5, 9):
        got = diffusion.q_posterior(start, observed, torch.tensor([t])).exp()[0]
        offset = 0
        for width, x0, xt in ((2, 1, 0), (3, 0, 2)):
            previous = torch.eye(width)[x0]
            for step in range(t):
                a = diffusion.alphas[step]
                transition = a * torch.eye(width) + (1 - a) / width
                previous = previous @ transition
            a = diffusion.alphas[t]
            likelihood = a * torch.eye(width)[:, xt] + (1 - a) / width
            expected = previous * likelihood
            expected /= expected.sum()
            torch.testing.assert_close(got[offset:offset + width], expected, atol=1e-6, rtol=1e-5)
            offset += width


def test_categorical_normalization_is_independent_of_other_columns_scale():
    logits = torch.tensor([[0.0, -1.0, -100.0, -101.0]], requires_grad=True)
    normalizer = sliced_logsumexp(logits, torch.tensor([0, 2, 4]))
    log_prob = logits - normalizer
    expected = torch.log_softmax(torch.tensor([[0.0, -1.0]]), dim=1)
    torch.testing.assert_close(log_prob[:, :2], expected)
    torch.testing.assert_close(log_prob[:, 2:], expected)
    loss = -log_prob[:, [1, 3]].sum()
    loss.backward()
    assert torch.isfinite(logits.grad).all()
    torch.testing.assert_close(logits.grad[:, :2], logits.grad[:, 2:])


def test_gaussian_epsilon_oracle_recovers_clean_final_sample():
    diffusion = _diffusion()
    clean = torch.tensor([[-2.0], [0.1], [3.0]])
    noise = torch.tensor([[0.3], [-0.7], [1.2]])
    timestep = torch.zeros(3, dtype=torch.long)
    perturbed = diffusion.gaussian_q_sample(clean, timestep, noise=noise)
    decoded = diffusion.gaussian_p_sample(noise, perturbed, timestep)["sample"]
    torch.testing.assert_close(decoded, clean, check_dtype=False)


def test_gaussian_decoder_nll_is_continuous_and_translation_invariant():
    diffusion = _diffusion()
    t = torch.zeros(3, dtype=torch.long)
    clean = torch.tensor([[-3.0], [0.0], [3.0]])
    # All three rows have the same reconstruction residual; image tail bins at
    # +/-1 must not give the outlying rows a different likelihood.
    model_mean = clean + 0.4
    sqrt_alpha = diffusion.sqrt_alphas_cumprod[0]
    perturbed = sqrt_alpha * model_mean
    got = diffusion._vb_terms_bpd(torch.zeros_like(clean), clean, perturbed, t)["output"]
    variance = float(diffusion.posterior_variance[1])
    expected = 0.5 * (math.log(2 * math.pi * variance) + 0.4 ** 2 / variance) / math.log(2)
    torch.testing.assert_close(got, torch.full_like(got, expected), atol=1e-5, rtol=1e-5)


def test_diffusion_rejects_nonfinite_generated_values(monkeypatch):
    diffusion = _diffusion()
    monkeypatch.setattr(diffusion, "sample", lambda n, y: (
        torch.full((n, 3), float("inf")), {"y": torch.zeros(n, dtype=torch.long)},
    ))
    from sbtab.baselines.tabddpm.utils import FoundNANsError
    with pytest.raises(FoundNANsError):
        diffusion.sample_all(2, 2, torch.ones(1))


def test_ve_reverse_sampler_with_exact_gaussian_score_recovers_data_moments():
    model = VEScoreSDEBaseline(VEScoreSDEConfig(
        steps=1, sigma_min=0.01, sigma_max=50.0,
        n_sampling_steps=400, n_corrector_steps=0, device="cpu",
    ))

    class GaussianScore(torch.nn.Module):
        def forward(self, x, sigma):
            # Convolution of N(2, 1) with N(0, sigma^2).
            return -(x - 2.0) / (1.0 + sigma.square())

    model._score_net = GaussianScore()
    model._dim = 1
    sample = model._sample_encoded(16000, seed=123)
    assert float(sample.mean()) == pytest.approx(2.0, abs=0.04)
    assert float(sample.var()) == pytest.approx(1.0, abs=0.06)


@pytest.mark.parametrize("method,args", [
    ("p_sample_loop", (None, {})),
    ("_sample", (None, {})),
    ("interpolate", (None, None)),
    ("nll", (None, {})),
    ("log_prob", (None, {})),
])
def test_unsupported_image_era_apis_fail_before_sampling(method, args):
    diffusion = _diffusion()
    rng_before = torch.random.get_rng_state().clone()
    with pytest.raises(NotImplementedError, match="sample|mixed_elbo"):
        getattr(diffusion, method)(*args)
    assert torch.equal(torch.random.get_rng_state(), rng_before)


def test_unsupported_vb_all_rejected_at_construction():
    with pytest.raises(NotImplementedError, match="vb_all.*unsupported"):
        GaussianMultinomialDiffusion(
            num_classes=np.array([2]), num_numerical_features=0,
            denoise_fn=torch.nn.Identity(), num_timesteps=10,
            multinomial_loss_type="vb_all",
        )


def test_gaussian_x0_mse_uses_clean_target_instead_of_noise():
    diffusion = _diffusion()
    diffusion.gaussian_parametrization = "x0"
    clean = torch.tensor([[-2.0], [3.0]])
    noise = torch.tensor([[0.3], [-0.7]])
    t = torch.tensor([2, 6])
    perturbed = diffusion.gaussian_q_sample(clean, t, noise=noise)
    loss = diffusion._gaussian_loss(clean, clean, perturbed, t, noise)
    torch.testing.assert_close(loss, torch.zeros(2))


class _ZeroDenoiser(torch.nn.Module):
    def forward(self, x, t, **kwargs):
        return torch.zeros_like(x)


@pytest.mark.parametrize("n_numeric,categories", [(1, [0]), (0, [2]), (1, [2])])
def test_mixed_elbo_handles_absent_blocks_per_row(n_numeric, categories):
    diffusion = GaussianMultinomialDiffusion(
        num_classes=np.array(categories), num_numerical_features=n_numeric,
        denoise_fn=_ZeroDenoiser(), num_timesteps=4,
    )
    data = torch.tensor([[0.0], [1.0]])
    if n_numeric and categories[0]:
        data = data.repeat(1, 2)
    result = diffusion.mixed_elbo(data, {})
    for name, value in result.items():
        assert value.shape[0] == len(data), name
        assert torch.isfinite(value).all(), name
    if not n_numeric:
        torch.testing.assert_close(result["total_gaussian"], torch.zeros(2))
    if not categories[0]:
        torch.testing.assert_close(result["total_multinomial"], torch.zeros(2))


def test_gaussian_ddim_eta_controls_stochastic_sampling():
    diffusion = GaussianMultinomialDiffusion(
        num_classes=np.array([0]), num_numerical_features=1,
        denoise_fn=_ZeroDenoiser(), num_timesteps=10,
    )
    start = torch.zeros(8, 1)
    torch.manual_seed(123)
    deterministic = diffusion.gaussian_ddim_sample(start, 10, {}, eta=0.0)
    torch.manual_seed(123)
    stochastic = diffusion.gaussian_ddim_sample(start, 10, {}, eta=0.7)
    assert torch.isfinite(stochastic).all()
    assert not torch.equal(stochastic, deterministic)
