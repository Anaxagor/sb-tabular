"""CSBM architecture/optimizer configuration and inference compatibility."""
from dataclasses import asdict
from pathlib import Path

import pandas as pd
import pytest
import torch
from torch import nn

from sbtab.adapters.native import AnnealedCSBMAdapter, CSBMAdapter
from sbtab.data.dataset_schema import ColumnSpec, DatasetSchema
from sbtab.experiments.tune import load_search_space
from sbtab.solvers.csbm import AnnealedCSBMConfig, CSBMConfig, CSBMSolver


NEW_FIELDS = ("n_layers", "dropout", "forward_lr", "backward_lr",
              "forward_weight_decay", "backward_weight_decay")


def tiny_config(config_cls=CSBMConfig, **overrides):
    settings = dict(num_outer_iterations=1, epochs=1, num_steps=3,
                    batch_size=4, emb_dim=2, hidden_dim=8, time_dim=4, seed=17)
    return config_cls(**{**settings, **overrides})


def table():
    return torch.tensor([[0, 0], [1, 1], [2, 0], [0, 1], [1, 0], [2, 1]])


def assert_architecture(solver, n_layers, dropout):
    for model in (solver.updater.forward_model, solver.updater.backward_model):
        assert len([layer for layer in model.net if isinstance(layer, nn.Linear)]) == n_layers + 1
        dropouts = [layer for layer in model.net if isinstance(layer, nn.Dropout)]
        assert len(dropouts) == (n_layers if dropout > 0 else 0)
        assert all(layer.p == dropout for layer in dropouts)


@pytest.mark.parametrize("config_cls", [CSBMConfig, AnnealedCSBMConfig])
def test_defaults_preserve_architecture_optimizer_and_positional_config(config_cls):
    # Exact argument order of the configuration before the six new keyword options.
    legacy_args = (3, 15, 264, .006, 50, 1., .2, .001, 16, 256, 64, "cpu", 41)
    if config_cls is AnnealedCSBMConfig:
        legacy_args += (3, .8, .02)
    cfg = config_cls(*legacy_args)
    assert cfg.lr == .006 and cfg.seed == 41 and cfg.num_steps == 50
    if config_cls is AnnealedCSBMConfig:
        assert (cfg.anneal_every, cfg.anneal_multiplier, cfg.min_mixing_rate) == (3, .8, .02)
    solver = CSBMSolver([3, 2], [False, False], cfg)
    assert_architecture(solver, 2, 0)
    assert "net.4.weight" in solver.updater.forward_model.state_dict()
    for opt in (solver.updater.forward_opt, solver.updater.backward_opt):
        assert opt.param_groups[0]["lr"] == .006
        assert opt.param_groups[0]["weight_decay"] == .01


@pytest.mark.parametrize("forward,backward,expected", [
    (None, None, (.004, .004)),
    (.002, None, (.002, .004)),
    (None, .007, (.004, .007)),
    (.002, .007, (.002, .007)),
])
def test_direction_rates_override_only_their_own_legacy_fallback(forward, backward, expected):
    cfg = tiny_config(lr=.004, forward_lr=forward, backward_lr=backward,
                      forward_weight_decay=0., backward_weight_decay=.002,
                      n_layers=4, dropout=.2)
    solver = CSBMSolver([3, 2], [False, False], cfg)
    assert_architecture(solver, 4, .2)
    assert (solver.updater.forward_opt.param_groups[0]["lr"],
            solver.updater.backward_opt.param_groups[0]["lr"]) == expected
    assert solver.updater.forward_opt.param_groups[0]["weight_decay"] == 0
    assert solver.updater.backward_opt.param_groups[0]["weight_decay"] == .002
    # The two optimizers must own the corresponding distinct networks.
    assert {id(p) for p in solver.updater.forward_opt.param_groups[0]["params"]} == {
        id(p) for p in solver.updater.forward_model.parameters()}
    assert {id(p) for p in solver.updater.backward_opt.param_groups[0]["params"]} == {
        id(p) for p in solver.updater.backward_model.parameters()}


@pytest.mark.parametrize("name", ["lr", "forward_lr", "backward_lr"])
@pytest.mark.parametrize("value", [0, -.01, float("nan"), float("inf"), True])
def test_invalid_learning_rate_is_rejected_instead_of_falling_back(name, value):
    with pytest.raises(ValueError, match=name):
        tiny_config(**{name: value})


@pytest.mark.parametrize("name", ["forward_weight_decay", "backward_weight_decay"])
@pytest.mark.parametrize("value", [-.01, float("nan"), float("inf"), True])
def test_invalid_weight_decay_is_rejected(name, value):
    with pytest.raises(ValueError, match=name):
        tiny_config(**{name: value})


@pytest.mark.parametrize("value", [0, -1, 2.5, 3., True, float("nan"), float("inf")])
def test_depth_must_be_a_positive_integer(value):
    with pytest.raises(ValueError, match="n_layers"):
        tiny_config(n_layers=value)


@pytest.mark.parametrize("value", [-.1, 1., float("nan"), float("inf"), True])
def test_dropout_must_be_a_valid_probability(value):
    with pytest.raises(ValueError, match="dropout"):
        tiny_config(dropout=value)


@pytest.mark.parametrize("config_cls", [CSBMConfig, AnnealedCSBMConfig])
def test_nondefault_architecture_and_optimizers_survive_checkpoint(config_cls, tmp_path):
    cfg = tiny_config(config_cls, n_layers=3, dropout=.25, forward_lr=.003, backward_lr=.007,
                      forward_weight_decay=.0003, backward_weight_decay=.002)
    solver = CSBMSolver([3, 2], [False, False], cfg).fit(table())
    expected = solver.sample(19, seed=33, batch_size=7)
    solver.save_checkpoint(tmp_path / "model.pt")
    restored = CSBMSolver.load_checkpoint(tmp_path / "model.pt")
    assert asdict(restored.cfg) == asdict(cfg)
    assert_architecture(restored, 3, .25)
    assert restored.updater.forward_opt.param_groups[0]["lr"] == .003
    assert restored.updater.backward_opt.param_groups[0]["weight_decay"] == .002
    assert torch.equal(restored.sample(19, seed=33, batch_size=7), expected)


@pytest.mark.parametrize("config_cls", [CSBMConfig, AnnealedCSBMConfig])
def test_old_csbm_2_checkpoint_missing_new_fields_still_loads(config_cls, tmp_path):
    solver = CSBMSolver([3, 2], [False, False], tiny_config(config_cls, lr=.006)).fit(table())
    state = solver.state_dict()
    assert state["format"] == "sbtab.csbm/2"
    for name in NEW_FIELDS:
        del state["config"][name]
    assert "net.4.weight" in state["forward_model"]
    torch.save(state, tmp_path / "legacy.pt")
    restored = CSBMSolver.load_checkpoint(tmp_path / "legacy.pt")
    assert_architecture(restored, 2, 0.)
    assert restored.updater.forward_opt.param_groups[0]["lr"] == .006
    assert restored.updater.backward_opt.param_groups[0]["lr"] == .006
    assert restored.updater.forward_opt.param_groups[0]["weight_decay"] == .01
    assert restored.updater.backward_opt.param_groups[0]["weight_decay"] == .01
    assert torch.equal(restored.sample(19, seed=33), solver.sample(19, seed=33))


def test_dropout_fit_is_seeded_independently_of_callers_and_restores_rng():
    cfg = tiny_config(n_layers=3, dropout=.4)
    first = CSBMSolver([3, 2], [False, False], cfg)
    second = CSBMSolver([3, 2], [False, False], cfg)
    torch.manual_seed(12)
    caller_state = torch.get_rng_state().clone()
    first.fit(table())
    assert torch.equal(torch.get_rng_state(), caller_state)
    torch.manual_seed(500)
    torch.rand(1000)
    caller_state = torch.get_rng_state().clone()
    second.fit(table())
    assert torch.equal(torch.get_rng_state(), caller_state)
    for name, weight in first.updater.backward_model.state_dict().items():
        assert torch.equal(weight, second.updater.backward_model.state_dict()[name])
    assert torch.equal(first.sample(32, seed=11), second.sample(32, seed=11))


@pytest.mark.parametrize("model,adapter_cls", [("csbm", CSBMAdapter), ("csbm_annealed", AnnealedCSBMAdapter)])
@pytest.mark.parametrize("kind", ["production", "smoke"])
def test_search_space_reaches_the_configuration_and_smoke_runs_nondefaults(model, adapter_cls, kind, tmp_path):
    folder = Path("configs/search_spaces") / ("smoke" if kind == "smoke" else "")
    space = load_search_space(folder / f"{model}.yaml", model, kind)
    assert space["version"] == 2
    assert "lr" not in space["params"]
    config = dict(space["fixed"])
    for name, definition in space["params"].items():
        config[name] = definition["choices"][-1] if definition["type"] == "categorical" else definition["low"]
    adapter = adapter_cls()
    adapter.config = adapter.resolve_config(config)
    adapter.seed = 7
    cfg = adapter._solver_config()
    for name in NEW_FIELDS:
        assert getattr(cfg, name) == config[name]
    if kind == "production":
        assert set(NEW_FIELDS) <= set(space["params"])
        return
    assert cfg.n_layers != 2 and cfg.dropout > 0
    assert cfg.forward_lr != cfg.backward_lr
    assert cfg.forward_weight_decay != cfg.backward_weight_decay
    frame = pd.DataFrame(table().numpy(), columns=["a", "b"], dtype=float)
    schema = DatasetSchema("toy", (ColumnSpec("a", "categorical"), ColumnSpec("b", "categorical")))
    adapter.fit(frame, schema, config, seed=7)
    assert_architecture(adapter.solver, cfg.n_layers, cfg.dropout)
    description = adapter.describe()
    assert description["architecture"]["n_layers"] == cfg.n_layers
    assert description["optimizers"]["forward"] == {"lr": cfg.forward_lr, "weight_decay": cfg.forward_weight_decay}
    assert description["optimizers"]["backward"] == {"lr": cfg.backward_lr, "weight_decay": cfg.backward_weight_decay}
    expected = adapter.sample(19, seed=33)
    adapter.save_checkpoint(tmp_path / "adapter")
    restored = adapter_cls.load_checkpoint(tmp_path / "adapter")
    pd.testing.assert_frame_equal(restored.sample(19, seed=33), expected)
    assert restored.describe()["architecture"] == description["architecture"]
    assert restored.describe()["optimizers"] == description["optimizers"]
