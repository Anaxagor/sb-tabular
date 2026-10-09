from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("sklearn")

from sbtab.baselines.tabbyflow import TabbyFlowConfig, TabbyFlowSynthesizer  # noqa: E402
from sbtab.data.schema import TabularSchema  # noqa: E402


def test_tabbyflow_mixed_data_fit_sample_smoke() -> None:
    train = pd.DataFrame(
        {
            "continuous": np.linspace(-1.0, 1.0, 12),
            "discrete": [0, 1, 2] * 4,
            "category": ["a", "b"] * 6,
            "target": [0, 1] * 6,
        }
    )
    schema = TabularSchema(
        continuous_cols=["continuous"],
        discrete_cols=["discrete"],
        categorical_cols=["category"],
        target_col="target",
    )
    config = TabbyFlowConfig(
        max_train_steps=2,
        batch_size=8,
        n_frequencies=2,
        cond_vel="ot",
        ode_solver="euler",
        ode_steps=2,
        sample_batch_size=4,
        device="cpu",
        seed=42,
    )

    model = TabbyFlowSynthesizer(config).fit(
        train,
        schema=schema,
        task_type="classification",
    )

    for method in ("euler", "midpoint", "rk4"):
        model.cfg.ode_solver = method
        synthetic = model.sample(5, seed=43)
        assert synthetic.shape == (5, 4)
        assert list(synthetic.columns) == list(train.columns)
        assert np.isfinite(synthetic[["continuous", "discrete"]].to_numpy()).all()
        assert set(synthetic["category"]).issubset(set(train["category"]))
        assert set(synthetic["target"]).issubset(set(train["target"]))
        assert set(synthetic["discrete"]).issubset(set(train["discrete"]))
        assert synthetic.dtypes.equals(train.dtypes)


def _tiny_model(**kwargs):
    return TabbyFlowSynthesizer(TabbyFlowConfig(
        max_train_steps=2, batch_size=8, n_frequencies=2, ode_steps=2,
        sample_batch_size=4, device="cpu", **kwargs,
    ))


def test_refit_resets_best_loss_and_preprocessor():
    first = pd.DataFrame({"x": np.linspace(-1, 1, 8)})
    schema = TabularSchema(["x"], [], [])
    model = _tiny_model().fit(first, schema=schema, task_type="regression")
    # Any new training loss is above this old result. It must not participate in
    # checkpoint selection after a fresh fit (even on a new schema).
    model.best_train_loss_ = -float("inf")
    second = pd.DataFrame({"category": ["a", "b"] * 4})
    model.fit(second, schema=TabularSchema([], [], ["category"]), task_type="classification")
    assert model.quantile is None
    assert model.actual_train_steps_ == 2
    assert list(model.sample(3, seed=1).columns) == ["category"]


def test_ids_are_fresh_and_undeclared_columns_are_rejected():
    train = pd.DataFrame({"x": np.linspace(-1, 1, 8), "id": np.arange(8)})
    model = _tiny_model().fit(
        train, schema=TabularSchema(["x"], [], [], id_col="id"), task_type="regression",
    )
    assert not set(model.sample(10, seed=1)["id"]) & set(train["id"])
    with pytest.raises(ValueError, match="without an explicit role"):
        _tiny_model().fit(train, schema=TabularSchema(["x"], [], []), task_type="regression")


def test_categorical_values_with_same_string_representation_remain_distinct():
    train = pd.DataFrame({"category": [1, "1"] * 4})
    model = _tiny_model()
    encoded = model._fit_preprocessor(train, TabularSchema([], [], ["category"]), "classification")
    assert model.cat_sizes_ == [2]
    pd.testing.assert_frame_equal(model._decode(encoded, seed=1), train)


def test_ve_source_noise_matches_path_scale(monkeypatch):
    from sbtab.baselines.tabbyflow import model as module
    train = pd.DataFrame({"x": np.linspace(-1, 1, 8)})
    with pytest.warns(UserWarning, match="approximation"):
        model = _tiny_model(cond_vel="ve").fit(train, schema=TabularSchema(["x"], [], []), task_type="regression")
    sources = []

    def capture(field, x, **kwargs):
        sources.append(x.clone())
        return x

    monkeypatch.setattr(module, "integrate_tabbyflow_fixed_step", capture)
    model.sample(4, seed=123)
    expected = 2 * torch.randn(4, 1, generator=torch.Generator().manual_seed(123))
    torch.testing.assert_close(sources[0], expected)


@pytest.mark.parametrize("path_name", ["ot", "vp", "ve", "cos"])
def test_ode_velocity_is_derivative_of_conditional_path(path_name):
    from sbtab.baselines.tabbyflow.model import TabbyFlowConditionalPath, TabbyFlowODEField

    class EndpointOracle(torch.nn.Module):
        def forward(self, t, x):
            return torch.full_like(x, 2.0)

    path = TabbyFlowConditionalPath(path_name)
    t = torch.tensor([0.13, 0.51, 0.89], dtype=torch.float64, requires_grad=True)
    alpha, beta, _, _ = path.coefficients(t)
    x = 2 * alpha - 0.7 * beta
    derivative = torch.autograd.grad(x.sum(), t)[0]
    field = TabbyFlowODEField(EndpointOracle(), path, 1, [])
    got = torch.cat([field(ti, xi.reshape(1, 1)).reshape(1) for ti, xi in zip(t, x)])
    torch.testing.assert_close(got, derivative)


@pytest.mark.parametrize("method", ["euler", "midpoint", "rk4"])
def test_ot_oracle_transports_to_endpoint_with_declared_residual_noise(method):
    from sbtab.baselines.tabbyflow.model import (
        TabbyFlowConditionalPath, TabbyFlowODEField, integrate_tabbyflow_fixed_step,
    )

    class EndpointOracle(torch.nn.Module):
        def forward(self, t, x):
            return torch.full_like(x, 2.0)

    source = torch.tensor([[-1.0], [0.0], [1.0]], dtype=torch.float64)
    field = TabbyFlowODEField(EndpointOracle(), TabbyFlowConditionalPath("ot"), 1, [])
    result = integrate_tabbyflow_fixed_step(field, source, n_steps=20, method=method)
    torch.testing.assert_close(result, 2 + 0.001 * source)
