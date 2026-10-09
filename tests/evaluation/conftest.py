"""Shared builders for the metric tests (fixtures, so no package/__init__ is required)."""
import pytest

from sbtab.data.dataset_schema import ColumnSpec, DatasetSchema


def _schema(cols, task=None, name="toy"):
    """cols: iterable of (name, type) or (name, type, "target")."""
    specs = []
    for c in cols:
        role = c[2] if len(c) > 2 else "feature"
        specs.append(ColumnSpec(c[0], c[1], role=role))
    return DatasetSchema(name=name, columns=tuple(specs), task=task)


@pytest.fixture
def make_schema():
    return _schema


def _statuses(obj):
    """Every value stored under a 'status' key, recursively."""
    out = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k == "status":
                out.append(v)
            out.extend(_statuses(v))
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            out.extend(_statuses(v))
    return out


@pytest.fixture
def collect_statuses():
    return _statuses
