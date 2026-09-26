"""
The example scripts must stay THIN callers of the canonical building blocks.

  * static: no metric implementation, no pickle / .pkl access and no dead API under
    ``examples/``; every script documents its registry id, the meaning of ``--quick``
    and the staged pipeline, and exposes ``main(quick: bool = False)``;
  * integration: every script really runs end to end with ``main(quick=True)``;
  * the notebook is only checked for valid nbformat-4 structure and cleared outputs.
"""
from __future__ import annotations

import ast
import importlib.util
import inspect
import json
import os
import re
import sys
import time
from pathlib import Path

os.environ["TQDM_DISABLE"] = "1"        # before anything imports tqdm

import pytest

from sbtab.evaluation import STATUSES
from sbtab.solvers.registry import get_entry

REPO = Path(__file__).resolve().parents[2]
EXAMPLES = REPO / "examples"
NOTEBOOK = EXAMPLES / "tabddpm_example.ipynb"

# a function with one of these fragments in its name would be a duplicate metric implementation
FORBIDDEN_FUNCTION_FRAGMENTS = ("avg_wd", "wasserstein", "kl_hist", "kl_div", "frobenius", "utility_delta",
                                "make_regressor", "js_div")
PICKLE_MODULES = {"pickle", "cPickle", "_pickle", "dill", "joblib", "cloudpickle"}
# the raw material of a hand-rolled metric: only sbtab.evaluation may use it
METRIC_PRIMITIVE_MODULES = ("scipy", "sklearn.metrics")
METRIC_PRIMITIVE_NAMES = {"wasserstein_distance", "jensenshannon", "rel_entr", "kl_div", "entropy", "corrcoef",
                          "histogram", "histogramdd", "r2_score", "rbf_kernel", "cdist", "pdist"}
CANONICAL_METRICS = {"MetricContext", "check_validity", "tuning_objective", "marginal_metrics", "association_metrics",
                     "conditional_metrics", "mmd_metrics"}
DEAD_APIS = {
    "TabularSchema(feature_cols=...)": re.compile(r"feature_cols\s*="),
    "TransformPipeline.default_continuous_dropna": re.compile(r"default_continuous_dropna"),
    "SplitConfigHoldout(random_seed=...)": re.compile(r"random_seed\s*="),
    "sbtab.solvers.ipf_dsb": re.compile(r"sbtab\.solvers\.ipf_dsb"),
    'first_coupling="ref"': re.compile(r"""first_coupling\s*=\s*["']ref["']"""),
    "sbtab.evaluation.metrics": re.compile(r"sbtab\.evaluation\.metrics"),
}
PIPELINE_STAGES = ("sbtab.experiments.prepare_splits", "sbtab.experiments.tune", "sbtab.experiments.cross_validate",
                   "sbtab.experiments.calculate_metrics")
EXPECTED = {        # script -> (registry id, dataset)
    "California_Housing_example.py": ("dsb_ct_joint_mlp", "california_housing"),
    "boosted_dsbm_example.py": ("dsbm_dt_joint_gbt", "california_housing"),
    "feature_wise_discrete_time_boosting-example.py": ("dsbm_dt_structural_gbt", "california_housing"),   # IMF, as at 6e2d887
    "structural_discrete_time_boost_ipf_example.py": ("dsb_dt_structural_gbt", "california_housing"),
    "joint_continuous_time_boost_example.py": ("dsbm_ct_joint_gbt", "california_housing"),             # IMF, as at 6e2d887
    "joint_continuous_time_boost_ipf_example.py": ("dsb_ct_joint_gbt", "california_housing"),
    "joint_discrete_time_mlp_example.py": ("dsbm_dt_joint_mlp", "california_housing"),
    "light_sb_example.py": ("lightsb", "california_housing"),
    "mixedsbm_example.py": ("mixedsbm", "insurance"),
    "csbm_example.py": ("csbm", "car_evaluation"),
}
QUICK_SECONDS = 60.0


def _python_files():
    """Every .py under examples/; the local, git-ignored symlink examples/sbtab is not part of the examples."""
    found = []
    for root, dirs, files in os.walk(EXAMPLES, followlinks=False):
        dirs[:] = [d for d in dirs if d != "__pycache__" and not (Path(root) / d).is_symlink()]
        found += [Path(root) / f for f in files if f.endswith(".py")]
    return sorted(found)


PY_FILES = _python_files()
SCRIPTS = [p for p in PY_FILES if p.name != "_common.py"]


def _notebook_cells(cell_type: str):
    nb = json.loads(NOTEBOOK.read_text())
    return ["".join(c["source"]) if isinstance(c["source"], list) else c["source"]
            for c in nb["cells"] if c["cell_type"] == cell_type]


def _sources():
    """(label, python source) of every script and of every notebook code cell."""
    out = [(str(p.relative_to(REPO)), p.read_text()) for p in PY_FILES]
    out += [(f"{NOTEBOOK.name}[code cell {i}]", src) for i, src in enumerate(_notebook_cells("code"))]
    return out


def _load(path: Path, monkeypatch):
    monkeypatch.syspath_prepend(str(EXAMPLES))      # the scripts do `from _common import ...`
    name = "sbtab_example_" + re.sub(r"\W", "_", path.stem)
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# --------------------------------------------------------------------------- static checks
def test_the_expected_examples_exist():
    assert (EXAMPLES / "_common.py") in PY_FILES
    assert {p.name for p in SCRIPTS} == set(EXPECTED)


@pytest.mark.parametrize("label, source", _sources(), ids=[s[0] for s in _sources()])
def test_no_duplicate_metric_implementation_and_no_pickle(label, source):
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            hits = [f for f in FORBIDDEN_FUNCTION_FRAGMENTS if f in node.name.lower()]
            assert not hits, f"{label}: function {node.name!r} looks like a metric implementation; use sbtab.evaluation"
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            modules = [a.name for a in node.names] if isinstance(node, ast.Import) else [node.module or ""]
            for module in modules:
                assert module.split(".")[0] not in PICKLE_MODULES, (
                    f"{label}: imports {module}; datasets are loaded through sbtab.data.registry.load_dataset")
                assert not module.startswith(METRIC_PRIMITIVE_MODULES), (
                    f"{label}: imports {module}; metrics come from sbtab.evaluation")
                assert not module.startswith("sbtab.evaluation."), (
                    f"{label}: imports the private module {module}; use the public sbtab.evaluation API")
        elif isinstance(node, (ast.Attribute, ast.Name)):
            name = node.attr if isinstance(node, ast.Attribute) else node.id
            assert name not in ("read_pickle", "to_pickle"), f"{label}: uses pandas pickle I/O"
            assert name not in METRIC_PRIMITIVE_NAMES, f"{label}: uses the metric primitive {name!r}"
        elif isinstance(node, ast.Constant) and isinstance(node.value, str):
            assert ".pkl" not in node.value and ".pickle" not in node.value, f"{label}: refers to a pickle file"


@pytest.mark.parametrize("label, source", _sources(), ids=[s[0] for s in _sources()])
def test_no_dead_api(label, source):
    hits = [name for name, pattern in DEAD_APIS.items() if pattern.search(source)]
    assert not hits, f"{label} references dead APIs: {hits}"


def test_no_dead_api_in_notebook_text():
    text = "\n".join(_notebook_cells("markdown"))
    hits = [name for name, pattern in DEAD_APIS.items() if pattern.search(text)]
    assert not hits, f"notebook markdown references dead APIs: {hits}"


@pytest.mark.parametrize("path", SCRIPTS, ids=[p.name for p in SCRIPTS])
def test_script_is_thin_and_documented(path):
    source = path.read_text()
    tree = ast.parse(source)
    assert len(source.splitlines()) <= 70, "an example is a thin caller of examples/_common.py"
    model_id, dataset = EXPECTED[path.name]
    doc = ast.get_docstring(tree) or ""
    assert model_id in doc and dataset in doc
    assert "--quick" in doc and "NOTHING about model quality" in doc
    for stage in PIPELINE_STAGES:
        assert stage in doc, f"the docstring must show the staged pipeline ({stage})"
    imported = {n.module for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)}
    assert "_common" in imported
    mains = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "main"]
    assert len(mains) == 1
    assert [f.name for f in tree.body if isinstance(f, ast.FunctionDef)] == ["main"], "helpers belong in _common.py"


@pytest.mark.parametrize("path", SCRIPTS, ids=[p.name for p in SCRIPTS])
def test_script_declares_a_supported_model_and_a_valid_config(path, monkeypatch):
    module = _load(path, monkeypatch)
    common = sys.modules["_common"]
    model_id, dataset = EXPECTED[path.name]
    assert (module.MODEL_ID, module.DATASET) == (model_id, dataset)
    entry = get_entry(model_id)
    assert entry.status == "supported"

    quick = inspect.signature(module.main).parameters["quick"]
    assert quick.default is False

    from sbtab.data.registry import load_dataset_config, schema_from_config
    from sbtab.solvers.registry import get_adapter_class
    schema = schema_from_config(load_dataset_config(dataset, common.DATASET_CONFIGS))
    assert schema.regime in entry.regimes
    adapter_cls = get_adapter_class(model_id)
    for config in (module.CONFIG, common.smoke_config(model_id)):      # strict: an unknown key raises here
        effective = adapter_cls.resolve_config(config)
        assert "noise" not in config
        assert effective.get("first_coupling", "ind") == "ind"
        if "gamma_max" in config:                                      # boosted IPF: Euler step of the OU reference
            assert float(effective["alpha_ou"]) * float(effective["gamma_max"]) < 1.0


def test_common_scores_with_the_canonical_metric_package():
    tree = ast.parse((EXAMPLES / "_common.py").read_text())
    imported = {a.name for n in ast.walk(tree) if isinstance(n, ast.ImportFrom) and n.module == "sbtab.evaluation"
                for a in n.names}
    assert CANONICAL_METRICS <= imported
    called = {n.func.id for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)}
    assert (CANONICAL_METRICS - {"MetricContext"}) <= called


def test_cli_exposes_the_quick_flag(monkeypatch):
    monkeypatch.syspath_prepend(str(EXAMPLES))
    import _common
    calls = []
    for argv, want in ((["example.py"], False), (["example.py", "--quick"], True)):
        monkeypatch.setattr(sys, "argv", argv)
        _common.cli(lambda quick=False: calls.append(quick), "doc")
        assert calls[-1] is want


# --------------------------------------------------------------------------- notebook
def test_notebook_is_valid_nbformat4_without_outputs():
    nb = json.loads(NOTEBOOK.read_text())
    assert nb["nbformat"] == 4 and isinstance(nb["nbformat_minor"], int)
    assert isinstance(nb["metadata"], dict) and nb["cells"]
    for cell in nb["cells"]:
        assert cell["cell_type"] in ("markdown", "code", "raw")
        assert isinstance(cell["source"], (str, list)) and isinstance(cell["metadata"], dict)
        if cell["cell_type"] == "code":
            assert cell["outputs"] == [], "notebook outputs must be cleared"
            assert cell["execution_count"] is None
    code = "\n".join(_notebook_cells("code"))
    assert "tabddpm" in code and "_common" in code
    assert "steps" in "\n".join(_notebook_cells("markdown"))
    nbformat = pytest.importorskip("nbformat")
    nbformat.validate(nbformat.reads(NOTEBOOK.read_text(), as_version=4))


# --------------------------------------------------------------------------- integration
@pytest.mark.integration
@pytest.mark.parametrize("path", SCRIPTS, ids=[p.name for p in SCRIPTS])
def test_example_runs_quick(path, monkeypatch, tmp_path, capsys):
    monkeypatch.setenv("TQDM_DISABLE", "1")
    monkeypatch.chdir(tmp_path)                     # the examples must not depend on the working directory
    module = _load(path, monkeypatch)
    common = sys.modules["_common"]

    start = time.perf_counter()
    result = module.main(quick=True)
    elapsed = time.perf_counter() - start
    assert elapsed < QUICK_SECONDS, f"--quick took {elapsed:.1f}s"

    model_id, dataset = EXPECTED[path.name]
    assert (result["model"], result["dataset"], result["quick"]) == (model_id, dataset, True)
    assert 0 < result["n_train"] <= common.QUICK_MAX_ROWS
    synth = result["synthetic"]
    assert len(synth) == result["n_train"]
    assert list(synth.columns) == list(result["preprocessor"].schema.column_order)
    assert not set(synth.index.names) & {"row_id"}                 # generated ids, never real row ids

    # every reported number comes from the canonical metric package, with its status
    for key in ("validity", "objective", "marginal", "association", "conditional", "mmd"):
        assert result[key]["status"] in STATUSES, key
        assert result[key]["metric_version"] == "sbtab.metrics/1"
    assert result["validity"]["status"] == "ok", result["validity"]["reasons"]
    assert result["objective"]["objective"] is not None
    assert result["mmd"]["kernels"]["full"]["mmd2_unbiased_mean"] is not None
    assert result["synthetic_raw"].shape == synth.shape

    out = capsys.readouterr().out
    assert "NOTHING about model quality" in out
    assert "sbtab.experiments.prepare_splits" in out
    assert not list(tmp_path.iterdir()), "an example must not write into the working directory"
