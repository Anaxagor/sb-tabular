from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("torch")
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
