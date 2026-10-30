"""Fixtures for the baseline-wrapper tests."""

from __future__ import annotations

import pandas as pd
import pytest
import torch

from baseline_testkit import make_mixed_frame


@pytest.fixture(autouse=True)
def _two_threads():
    torch.set_num_threads(2)
    yield


@pytest.fixture
def mixed_frame() -> pd.DataFrame:
    return make_mixed_frame()
