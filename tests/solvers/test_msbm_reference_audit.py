"""Independent finite-state oracles for MSBM reference and IMF dynamics."""
import pytest
import torch

from sbtab.bridge.reference import IncompatibleBridgeError
from sbtab.solvers.msbm import MixedSBMConfig, MixedSBMSolver
from sbtab.solvers.msbm.losses import CSBMLoss, MixedSBMLoss
from sbtab.solvers.msbm.reference import CategoricalReference


def ordered_ref(S=5, K=40, alpha=0.45):
    return CategoricalReference([S], torch.tensor([True]), K, alpha=alpha)


def test_ordered_powers_obey_chapman_kolmogorov_past_old_approximation_cutoff():
    ref = ordered_ref()
    idx = torch.arange(5, dtype=torch.float64)
    q = torch.softmax(-4 * (idx[:, None] - idx[None, :]).square() / (0.45 * 4) ** 2, -1)
    # Row normalisation makes this kernel asymmetric at the boundaries; preserve it.
    assert not torch.allclose(q, q.T)
    for k in (1, 29, 30, 40):
        torch.testing.assert_close(ref._powers[0, k], torch.linalg.matrix_power(q, k), atol=1e-13, rtol=1e-13)
    torch.testing.assert_close(ref._powers[0, 11] @ ref._powers[0, 29], ref._powers[0, 40])


@pytest.mark.parametrize("n", [0, 1, 11, 29, 30, 39, 40])
def test_ordered_bridge_equals_bayes_rule_and_preserves_endpoints(n):
    ref = ordered_ref()
    start, end = torch.tensor([[0], [4]]), torch.tensor([[4], [1]])
    q = ref._powers[0, 1]
    left, right = torch.linalg.matrix_power(q, n), torch.linalg.matrix_power(q, 40 - n)
    oracle = torch.stack([left[a, :] * right[:, b] for a, b in zip(start[:, 0], end[:, 0])])
    oracle /= oracle.sum(-1, keepdim=True)
    actual = ref.bridge_at_time(start, end, n, 40)[:, 0]
    torch.testing.assert_close(actual, oracle, atol=1e-13, rtol=1e-13)


@pytest.mark.parametrize("n", [0, 1, 10, 20, 39, 40])
def test_tiny_alpha_bridge_keeps_rare_paths_without_uniform_fallback(n):
    ref = ordered_ref(S=2, alpha=0.01)
    # A switch has probability exp(-40000): it underflows even in float64.
    # Conditioned on opposite endpoints, one jump occurs at a uniform step.
    p = ref.bridge_at_time(torch.tensor([[0]]), torch.tensor([[1]]), n, 40)[0, 0]
    torch.testing.assert_close(p, torch.tensor([1 - n / 40, n / 40], dtype=torch.float64), atol=2e-10, rtol=2e-10)


@pytest.mark.parametrize("direction", ["f", "b"])
def test_model_induced_step_matches_explicit_endpoint_mixture(direction):
    ref = ordered_ref(K=40)
    logits = torch.tensor([[[0.4, -0.2, 1.2, 0.1, -0.5]]], requires_grad=True)
    x = torch.tensor([[2]])
    endpoint_weights = logits.double().softmax(-1)[0, 0]
    if direction == "f":
        oracle = sum(endpoint_weights[j] * ref.bridge_next_given_prev(x, torch.tensor([[j]]), 9, 40)
                     for j in range(5))
        result = ref.model_induced_next_step(logits, x, 9, 40)
    else:
        oracle = sum(endpoint_weights[j] * ref.bridge_prev_given_next(torch.tensor([[j]]), x, 31)
                     for j in range(5))
        result = ref.model_induced_prev_step(logits, x, 31)
    torch.testing.assert_close(result, oracle, atol=1e-13, rtol=1e-13)
    (-result[..., 0].log().mean()).backward()
    assert torch.isfinite(logits.grad).all()


@pytest.mark.parametrize("direction", ["f", "b"])
def test_extreme_logits_have_finite_loss_and_useful_gradients(direction):
    ref = ordered_ref(S=2, alpha=0.01)
    logits = torch.tensor([[[1000., -1000.]]], requires_grad=True)
    loss_fn = CSBMLoss(ref, lmbda=0)
    # At the endpoint step, the categorical loss is exactly endpoint NLL,
    # even though its probability exp(-2000) underflows in probability space.
    x, endpoint = torch.tensor([[0]]), torch.tensor([[1]])
    loss = (loss_fn.forward_loss(logits, endpoint, x, torch.tensor([39]), 40) if direction == "f"
            else loss_fn.backward_loss(logits, endpoint, x, torch.tensor([1])))
    assert loss.item() == pytest.approx(2000)
    loss.backward()
    torch.testing.assert_close(logits.grad, torch.tensor([[[1., -1.]]]))


def test_periodic_reference_rejects_impossible_bridge_and_preserves_zeros():
    ref = CategoricalReference([2], torch.tensor([False]), 4, alpha=1)
    with pytest.raises(IncompatibleBridgeError):
        ref.bridge_at_time(torch.tensor([[0]]), torch.tensor([[1]]), 2, 4)
    p = ref.bridge_at_time(torch.tensor([[0]]), torch.tensor([[0]]), 1, 4)
    assert torch.equal(p, torch.tensor([[[0., 1.]]], dtype=torch.float64))
    logits = torch.tensor([[[0.2, 0.7]]], requires_grad=True)
    result = ref.model_induced_next_step(logits, torch.tensor([[0]]), 0, 4)
    assert torch.equal(result, p)
    result.sum().backward()
    assert torch.isfinite(logits.grad).all()
    with pytest.raises(ValueError, match="sum to one"):
        ref.sample_from_probs(torch.zeros_like(p))


def test_mixed_categorical_loss_is_normalised_once_and_padded_logits_are_masked():
    x, target = torch.tensor([[0], [1]]), torch.tensor([[1], [0]])
    logits = torch.tensor([[[0.2, -0.1]], [[0.7, -0.2]]], requires_grad=True)
    losses = []
    for D in (1, 3):
        ref = CategoricalReference([2] * D, torch.tensor([False] * D), 4, alpha=.2)
        loss_fn = MixedSBMLoss(ref, lambda_num=0, lambda_cat=.7)
        losses.append(loss_fn(None, None, logits.repeat(1, D, 1), target.repeat(1, D),
                              x.repeat(1, D), torch.tensor([1, 3]), 4, "b"))
    torch.testing.assert_close(*losses)
    ref = CategoricalReference([2, 3], torch.tensor([False, False]), 4, alpha=.2)
    padded_logits = torch.tensor([[[0.2, -0.1, 10000.], [.2, .3, -.1]]], requires_grad=True)
    loss_fn = CSBMLoss(ref)
    loss = loss_fn.backward_loss(padded_logits, torch.tensor([[1, 2]]), torch.tensor([[0, 0]]), torch.tensor([1]))
    loss.backward()
    assert padded_logits.grad[0, 0, 2] == 0
    assert torch.isfinite(loss)


def test_deterministic_generation_does_not_remove_noise_from_training_couplings(tmp_path):
    cfg = MixedSBMConfig(num_steps=4, sigma=.7, noise=False, hidden_dim=8, n_layers=1,
                         time_dim=4, fb_sequence=("b",), steps_per_direction=1, min_steps_per_direction=0)
    solver = MixedSBMSolver(1, [], [], cfg)
    for parameter in solver.model.parameters():
        parameter.data.zero_()
    x, cat = torch.zeros(4096, 1), torch.empty(4096, 0, dtype=torch.long)
    z0, _, _, _ = solver._generate_coupling(x, cat, x, cat, solver.model.state_dict(), "b", seed=19)
    assert z0.var().item() == pytest.approx(.7 ** 2, abs=.04)
    generated, _, _ = solver.sampler.simulate(x, cat, solver.model, "b", seed=19)
    assert torch.equal(generated, x)
    old = solver.state_dict()
    old["format"] = "sbtab.mixedsbm/3"
    torch.save(old, tmp_path / "old.pt")
    with pytest.raises(ValueError, match="older references require retraining"):
        MixedSBMSolver.load_checkpoint(tmp_path / "old.pt")


def test_nominal_tiny_alpha_retains_a_reachable_rare_endpoint():
    ref = CategoricalReference([2], torch.tensor([False]), 4, alpha=1e-20)
    p = ref.bridge_at_time(torch.tensor([[0]]), torch.tensor([[1]]), 1, 4)
    torch.testing.assert_close(p, torch.tensor([[[.75, .25]]], dtype=torch.float64), atol=1e-14, rtol=1e-14)
