import json

import numpy as np
import pandas as pd
import pytest

from sbtab.data.dataset_schema import ColumnSpec, DatasetSchema
from sbtab.evaluation.validity import numerical_diagnostics


def test_diagnostics_handle_float64_extremes_without_overflow_or_nonstandard_json():
    schema = DatasetSchema("extremes", (ColumnSpec("x", "continuous"),))
    synth = pd.DataFrame({"x": [-1.7e308, 1.7e308, np.nan, np.inf]})
    train = pd.DataFrame({"x": [-1., 1.]})
    with np.errstate(over="raise", invalid="raise"):
        out = numerical_diagnostics(synth, schema, train)
    json.dumps(out, allow_nan=False)
    assert out["columns"]["x"]["max_abs"] == 1.7e308
    assert out["columns"]["x"]["n_nonfinite"] == 2
    assert out["columns"]["x"]["extreme_finite_fraction"] == 1.0
    assert out["columns"]["x"]["train_min"] == -1.0


def test_no_finite_output_has_no_fake_finite_tail_statistic():
    schema = DatasetSchema("empty", (ColumnSpec("x", "continuous"),))
    out = numerical_diagnostics(pd.DataFrame({"x": [np.nan]}), schema, pd.DataFrame({"x": [0.]}))
    assert out["columns"]["x"]["max_abs"] is None and out["warnings"] == []
    json.dumps(out, allow_nan=False)


@pytest.mark.parametrize("ratio", [0, 1, np.inf, np.nan])
def test_invalid_diagnostic_threshold_rejected(ratio):
    schema = DatasetSchema("empty", (ColumnSpec("x", "continuous"),))
    with pytest.raises(ValueError, match="extreme_ratio"):
        numerical_diagnostics(pd.DataFrame({"x": [0.]}), schema, pd.DataFrame({"x": [0.]}), extreme_ratio=ratio)
