"""Fault injection: reject the first numerical failure without applying an update."""
import json

import pytest
import torch

from sbtab.bridge.reference import CategoricalReference as SharedReference, IncompatibleBridgeError
from sbtab.bridge.sde import EulerMaruyama
from sbtab.bridge.timegrid import TimeGrid
from sbtab.numerics import NumericalError, check_gradients, require_finite
from sbtab.solvers.continuous_time.joint_distribution.mlp.ipf_dsb.solver import IPFDSBConfig, IPFDSBSolver
from sbtab.solvers.discrete_time.joint_distribution.mlp.ipf_dsb.solver import (
    IPFDSBConfig as DTConfig, IPFDSBSolver as DTSolver,
)
from sbtab.solvers.continuous_time.joint_distribution.mlp.imf_dsbm.solver import IMFDSBMConfig, IMFDSBMSolver
from sbtab.solvers.msbm import MixedSBMConfig, MixedSBMSolver
from sbtab.solvers.msbm.pathsampler import MixedPathSampler
from sbtab.solvers.msbm.reference import CategoricalReference as MixedReference


def _training_case(kind, grad_clip):
    if kind in ("dsb", "dsb_dt"):
        cls, config = (IPFDSBSolver, IPFDSBConfig) if kind == "dsb" else (DTSolver, DTConfig)
        solver = cls(1, config(ipf_iters=1, num_steps=2, horizon=.1, hidden_units=4,
            time_features=4, n_layers=1, steps_per_phase=3, cache_batches=1,
            batch_size=2, grad_clip=grad_clip))
        net = solver.net_b
        call = lambda: solver._train_phase(0, "backward", "reference", torch.zeros(2, 1), solver._generator(1))
    elif kind == "dsbm":
        solver = IMFDSBMSolver(1, IMFDSBMConfig(hidden_dim=4, num_steps=2, inner_iters=1,
            fb_sequence=("b",), grad_clip=grad_clip))
        net = solver.model.net("b")
        call = lambda: solver._train_direction("b", torch.zeros(2, 1), torch.ones(2, 1), 1)
    else:
        solver = MixedSBMSolver(1, [], None, MixedSBMConfig(hidden_dim=4, time_dim=4, n_layers=1,
            num_steps=2, steps_per_direction=1, min_steps_per_direction=0,
            fb_sequence=("b",), grad_clip=grad_clip))
        net = solver.model
        call = lambda: solver.updater.train_step(torch.zeros(2, 1), None, torch.ones(2, 1), None, "b")
    return solver, net, call


@pytest.mark.parametrize("kind", ["dsb", "dsb_dt", "dsbm", "mixed"])
@pytest.mark.parametrize("bad", [float("nan"), float("inf")])
def test_nonfinite_loss_is_rejected_before_backward_or_optimizer(monkeypatch, kind, bad):
    solver, net, call = _training_case(kind, 1.0)
    before = {name: p.detach().clone() for name, p in net.named_parameters()}
    step_calls = []
    monkeypatch.setattr(torch.optim.Adam, "step", lambda *a, **kw: step_calls.append("Adam"))
    monkeypatch.setattr(torch.optim.AdamW, "step", lambda *a, **kw: step_calls.append("AdamW"))
    bad_loss = lambda *a, **kw: next(p for p in net.parameters() if p.requires_grad).sum() * 0 + bad
    if kind.startswith("dsb") and kind != "dsbm":
        monkeypatch.setattr(solver, "_batch_loss", bad_loss)
    elif kind == "dsbm":
        monkeypatch.setattr(solver, "loss_fn", bad_loss)
    else:
        monkeypatch.setattr(solver.updater.loss_fn, "forward", bad_loss)
    with pytest.raises(NumericalError, match="non-finite loss") as error:
        call()
    assert not step_calls
    assert error.value.details["stage"] == "training" and error.value.details["step"] == 0
    json.dumps(error.value.details)
    for name, p in net.named_parameters():
        torch.testing.assert_close(p, before[name], rtol=0, atol=0)
        assert p.grad is None


@pytest.mark.parametrize("kind", ["dsb", "dsb_dt", "dsbm", "mixed"])
@pytest.mark.parametrize("grad_clip", [None, 1.0])
def test_nonfinite_gradient_is_rejected_before_any_parameter_update(monkeypatch, kind, grad_clip):
    _, net, call = _training_case(kind, grad_clip)
    before = {name: p.detach().clone() for name, p in net.named_parameters()}
    step_calls = []
    monkeypatch.setattr(torch.optim.Adam, "step", lambda *a, **kw: step_calls.append("Adam"))
    monkeypatch.setattr(torch.optim.AdamW, "step", lambda *a, **kw: step_calls.append("AdamW"))
    hooks = [p.register_hook(lambda grad: torch.full_like(grad, float("inf")))
             for p in net.parameters() if p.requires_grad]
    try:
        with pytest.raises(NumericalError, match="non-finite gradient"):
            call()
    finally:
        for hook in hooks:
            hook.remove()
    assert not step_calls
    for name, p in net.named_parameters():
        torch.testing.assert_close(p, before[name], rtol=0, atol=0)


def test_large_finite_gradients_have_stable_norm_and_are_not_zeroed():
    p = torch.nn.Parameter(torch.zeros(2))
    p.grad = torch.tensor([3e30, 4e30])
    norm = check_gradients([p], max_norm=1.0)
    assert norm.dtype == torch.float64 and norm.item() == pytest.approx(5e30)
    torch.testing.assert_close(p.grad, torch.tensor([.6, .8]))
    require_finite(torch.tensor([1e35]), "finite_sample")  # magnitude alone is not invalidity


def test_bad_gradient_does_not_modify_other_gradients():
    a, b = torch.nn.Parameter(torch.zeros(2)), torch.nn.Parameter(torch.zeros(2))
    a.grad, b.grad = torch.tensor([3., 4.]), torch.tensor([float("nan"), 1.])
    with pytest.raises(NumericalError):
        check_gradients([a, b], max_norm=1.)
    torch.testing.assert_close(a.grad, torch.tensor([3., 4.]), rtol=0, atol=0)


def test_gradient_diagnostics_keep_parameter_index_when_other_parameters_are_unused():
    unused, bad = torch.nn.Parameter(torch.zeros(1)), torch.nn.Parameter(torch.zeros(1))
    bad.grad = torch.full_like(bad, float("inf"))
    with pytest.raises(NumericalError) as error:
        check_gradients([unused, bad])
    assert error.value.details["parameter_index"] == 1


class _BadDrift(torch.nn.Module):
    def __init__(self, mixed=False):
        super().__init__()
        self.calls, self.mixed = 0, mixed

    def forward(self, x, *args):
        self.calls += 1
        v = torch.full_like(x, float("inf"))
        return (v, torch.empty(len(x), 0, 0)) if self.mixed else v


@pytest.mark.parametrize("stage", ["sampling", "coupling"])
@pytest.mark.parametrize("kind", ["dsb", "dsb_dt", "dsbm", "mixed"])
def test_bad_trajectory_stops_at_first_step_with_context(monkeypatch, kind, stage):
    solver, _, _ = _training_case(kind, 1.)
    field = _BadDrift(mixed=kind == "mixed")
    if kind in ("dsb", "dsb_dt"):
        monkeypatch.setattr(solver, "_mean_map", lambda which, x, k: field(x))
        solver._fitted = True
        call = (lambda: solver.sample(2, seed=1, batch_size=1)) if stage == "sampling" else (
            lambda: solver._make_cache("backward", "reference", torch.zeros(2, 1), solver._generator(1)))
    elif kind == "dsbm":
        call = lambda: solver._sample_sde(field, "b", torch.zeros(2, 1), noise=False, stage=stage)
    else:
        sampler = MixedPathSampler(TimeGrid.uniform(2), None, EulerMaruyama(False), has_cont=True, has_cat=False)
        call = lambda: sampler.simulate(torch.zeros(2, 1), torch.empty(2, 0, dtype=torch.long), field,
                                       "b", batch_size=1, stage=stage)
    with pytest.raises(NumericalError) as error:
        call()
    assert field.calls == 1
    assert error.value.details["stage"] == stage
    assert "direction" in error.value.details and "step" in error.value.details
    if kind == "mixed" or (kind in ("dsb", "dsb_dt") and stage == "sampling"):
        assert error.value.details["chunk"] == 0


@pytest.mark.parametrize("kind", ["shared", "mixed"])
@pytest.mark.parametrize("direction", ["f", "b"])
@pytest.mark.parametrize("bad", [float("nan"), float("inf")])
def test_bad_logits_are_numerical_errors_not_impossible_bridges(kind, direction, bad):
    ref = (SharedReference([2], torch.tensor([False]), TimeGrid.uniform(2)) if kind == "shared"
           else MixedReference([2], torch.tensor([False]), 2, alpha=.2))
    logits, state = torch.tensor([[[bad, 0.]]]), torch.zeros((1, 1), dtype=torch.long)
    with pytest.raises(NumericalError, match="categorical_logits"):
        if direction == "f":
            ref.model_induced_next_step(logits, state, 0, **({"K": 2} if kind == "mixed" else {}))
        else:
            ref.model_induced_prev_step(logits, state, 2)


def test_true_unreachable_bridge_keeps_distinct_error_and_structural_zeros():
    ref = MixedReference([2], torch.tensor([False]), 4, alpha=1.)  # deterministic alternating chain
    state, other = torch.tensor([[0]]), torch.tensor([[1]])
    with pytest.raises(IncompatibleBridgeError):
        ref.bridge_at_time(state, other, 2, 4)
    result = ref.model_induced_next_step(torch.tensor([[[0., float("-inf")]]]), state, 0, 4)
    torch.testing.assert_close(result, torch.tensor([[[0., 1.]]], dtype=torch.float64))


def test_shared_normalizer_distinguishes_zero_mass_from_numerical_failure():
    ref = SharedReference([2], torch.tensor([False]), TimeGrid.uniform(2))
    with pytest.raises(IncompatibleBridgeError):
        ref._normalise(torch.zeros(1, 2), 0, "test_bridge")
    with pytest.raises(NumericalError, match="categorical_weights"):
        ref._normalise(torch.tensor([[float("nan"), 1.]]), 0, "test_bridge")
