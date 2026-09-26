"""
Guards for the ``sbtab/experiments/legacy`` namespace.

The legacy scripts are frozen historical code (metric definitions ``legacy/0``).
These tests make sure that
  (i)   no duplicate metric implementation survives outside ``legacy_metrics.py``;
  (ii)  nothing imports the non-existent ``sbtab.evaluation.metrics`` package any more;
  (iii) ``legacy_metrics`` imports and still computes the historical definitions;
  (iv)  no developer-machine Windows path remains, except inside the tracked historical
        result files, which must stay byte-identical to what was committed;
  (v)   every consolidated helper is value-identical to the historical copy it replaced.

They deliberately do NOT run any legacy script end-to-end (none has been run on this
branch; see ``sbtab/experiments/legacy/README.md``).
"""
from __future__ import annotations

import ast
import contextlib
import io
import json
import math
import re
import subprocess
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

import numpy as np
import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[2]
LEGACY = REPO / "sbtab" / "experiments" / "legacy"
LEGACY_METRICS = LEGACY / "legacy_metrics.py"

# Commit that still holds every legacy file at its PRE-MOVE path
# (``sbtab/experiments/<same relative path without legacy/>``). It is ``HEAD`` at the time
# the legacy namespace was created; pinning the hash keeps these tests meaningful after
# the move itself has been committed and ``HEAD`` no longer contains the old paths.
PRE_MOVE_COMMIT = "6e2d887"

FORBIDDEN_EXACT = {
    "avg_wd", "avg_kl_hist", "utility_delta_r2_percent", "make_regressor", "sliced_wasserstein",
    "load_best_params", "_common_numeric_cols", "common_numeric_cols",
}
FORBIDDEN_PREFIXES = (
    "corr_frobenius", "average_wd", "avg_wd_", "avg_kl_hist_", "utility_delta_r2_percent_",
    "make_regressor_", "export_trials_csv", "resolve_target_col", "build_transforms",
)
FORBIDDEN_ASSIGNMENTS = {"TARGET_COL_BY_DATASET"}

RESULT_DIRS = (
    "calculating_metrics/dsbm_kfold_eval",
    "calculating_metrics/tabpfgen_kfold_eval",
    "tuning_script/dsbm_optuna_results",
    "tuning_results/best_params",
)


def _legacy_scripts() -> List[Path]:
    return sorted(
        p for p in LEGACY.rglob("*.py")
        if p != LEGACY_METRICS and "__pycache__" not in p.parts
    )


def _pre_move_path(p: Path) -> str:
    return "sbtab/experiments/" + p.relative_to(LEGACY).as_posix()


def _git_show(spec: str) -> Optional[bytes]:
    try:
        r = subprocess.run(["git", "show", spec], cwd=REPO, capture_output=True, timeout=60)
    except (OSError, subprocess.SubprocessError):
        return None
    return r.stdout if r.returncode == 0 else None


# --------------------------------------------------------------------------- (i)

def test_there_are_legacy_scripts_to_scan() -> None:
    scripts = [p for p in _legacy_scripts() if p.name != "__init__.py"]
    assert len(scripts) >= 17, [p.name for p in scripts]


def test_no_duplicate_metric_implementation_survives() -> None:
    offenders: List[str] = []
    for path in _legacy_scripts():
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                if node.name in FORBIDDEN_EXACT or node.name.startswith(FORBIDDEN_PREFIXES):
                    offenders.append(f"{path.relative_to(REPO)}:{node.lineno} defines {node.name}")
        for node in tree.body:
            targets = []
            if isinstance(node, ast.Assign):
                targets = [t for t in node.targets if isinstance(t, ast.Name)]
            elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
                targets = [node.target]
            for t in targets:
                if t.id in FORBIDDEN_ASSIGNMENTS:
                    offenders.append(f"{path.relative_to(REPO)}:{node.lineno} re-defines {t.id}")
    assert not offenders, "duplicate legacy helpers found:\n" + "\n".join(offenders)


def test_every_script_is_bannered_and_warns_from_main() -> None:
    for path in _legacy_scripts():
        if path.name == "__init__.py":
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"))
        doc = ast.get_docstring(tree) or ""
        assert doc.startswith("LEGACY"), f"{path.name}: module docstring must start with the LEGACY banner"
        for needle in ("legacy/0", "NOT the canonical protocol", "sbtab.experiments.<stage>", "sbtab.metrics/1"):
            assert needle in doc, f"{path.name}: banner lacks {needle!r}"
        mains = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "main"]
        assert len(mains) == 1, path.name
        first = mains[0].body[0]
        assert (
            isinstance(first, ast.Expr)
            and isinstance(first.value, ast.Call)
            and getattr(first.value.func, "id", None) == "print_legacy_warning"
        ), f"{path.name}: main() must start with print_legacy_warning(...)"
        expected = "sbtab.experiments.legacy." + ".".join(path.relative_to(LEGACY).with_suffix("").parts)
        assert first.value.args[0].value == expected, path.name


def test_legacy_warning_goes_to_stderr_once(capsys: pytest.CaptureFixture) -> None:
    from sbtab.experiments.legacy import legacy_metrics as lm

    name = "tests.fake_legacy_script"
    lm._WARNED_SCRIPTS.discard(name)
    lm.print_legacy_warning(name)
    lm.print_legacy_warning(name)
    out = capsys.readouterr()
    assert out.out == ""
    assert out.err.count("LEGACY SCRIPT") == 1
    assert "legacy/0" in out.err and "sbtab.metrics/1" in out.err and name in out.err


# --------------------------------------------------------------------------- (ii)

_STALE_IMPORT_RE = re.compile(
    r"^\s*(?:from\s+sbtab\.evaluation\.metrics\b|import\s+sbtab\.evaluation\.metrics\b"
    r"|from\s+sbtab\.evaluation\s+import\s+(?:.*\b)?metrics\b)",
    re.MULTILINE,
)


def _imports_stale_package(path: Path) -> bool:
    text = path.read_text(encoding="utf-8", errors="replace")
    if path.suffix == ".ipynb":
        return "sbtab.evaluation.metrics" in text
    try:
        tree = ast.parse(text)
    except SyntaxError:  # somebody's work in progress -- fall back to a textual check
        return bool(_STALE_IMPORT_RE.search(text))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            if any(a.name == "sbtab.evaluation.metrics" or a.name.startswith("sbtab.evaluation.metrics.")
                   for a in node.names):
                return True
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            if node.module == "sbtab.evaluation.metrics" or node.module.startswith("sbtab.evaluation.metrics."):
                return True
            if node.module == "sbtab.evaluation" and any(a.name == "metrics" for a in node.names):
                return True
    return False


def _source_files(*roots: str) -> Iterator[Path]:
    for r in roots:
        base = REPO / r
        if not base.is_dir():
            continue
        for p in base.rglob("*"):
            if p.suffix in (".py", ".ipynb") and "__pycache__" not in p.parts and p.is_file():
                yield p


def test_nothing_imports_the_nonexistent_evaluation_metrics_package() -> None:
    scanned = list(_source_files("sbtab", "examples"))
    assert len(scanned) > 50, "scan found suspiciously few files"  # guard against a vacuous pass
    stale =[str(p.relative_to(REPO)) for p in _source_files("sbtab", "examples") if _imports_stale_package(p)]
    assert not stale, f"stale imports of sbtab.evaluation.metrics: {stale}"


def test_legacy_code_does_not_import_the_canonical_packages() -> None:
    """Legacy code must stay independent of the canonical metric package and stages."""
    banned = ("sbtab.evaluation", "sbtab.experiments.tune", "sbtab.experiments.cross_validate",
              "sbtab.experiments.prepare_splits", "sbtab.experiments.calculate_metrics",
              "sbtab.experiments.aggregate_results", "sbtab.experiments.runner",
              "sbtab.experiments.experiment_common")
    bad: List[str] = []
    for path in list(_legacy_scripts()) + [LEGACY_METRICS]:
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            mods: List[str] = []
            if isinstance(node, ast.Import):
                mods = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                mods = [node.module]
            for m in mods:
                if any(m == b or m.startswith(b + ".") for b in banned):
                    bad.append(f"{path.relative_to(REPO)}:{node.lineno} imports {m}")
    assert not bad, "\n".join(bad)


# --------------------------------------------------------------------------- (iii)

@pytest.fixture(scope="module")
def lm():
    from sbtab.experiments.legacy import legacy_metrics

    return legacy_metrics


def test_legacy_metrics_version_and_surface(lm) -> None:
    assert lm.LEGACY_METRIC_VERSION == "legacy/0"
    for name in lm.__all__:
        assert hasattr(lm, name), name
    doc = lm.__doc__ or ""
    for needle in ("sbtab.metrics/1", "UNION", "1e-12", "n_bins=50", "20", "RAW UNITS", "R^2",
                   "cat_features", "seed + 1", "len(test fold)", "SILENT FALLBACK",
                   "HistGradientBoostingRegressor", "Pearson", "fillna"):
        assert needle in doc, f"legacy_metrics docstring must document {needle!r}"
    assert sorted(lm.TAPFN_DEFAULT_DATASET_ORDER) == sorted(lm.TARGET_COL_BY_DATASET)
    assert lm.TARGET_COL_BY_DATASET["online_news_popularity"] == " shares"  # leading space is real
    assert lm.average_wd is lm.avg_wd


def test_avg_wd_of_a_pure_shift_is_the_shift(lm) -> None:
    rng = np.random.default_rng(0)
    real = pd.DataFrame({"a": rng.normal(size=500), "b": rng.exponential(size=500)})
    for c in (0.25, 3.0):
        assert lm.avg_wd(real, real + c, ["a", "b"]) == pytest.approx(c, abs=1e-12)
    # two columns shifted by 1 and 3 -> mean 2 ; the target is NOT rescaled by the helper
    synth = pd.DataFrame({"a": real["a"] + 1.0, "b": real["b"] + 3.0})
    assert lm.avg_wd(real, synth, ["a", "b"]) == pytest.approx(2.0, abs=1e-12)
    assert lm.avg_wd_autocols(real, synth) == pytest.approx(2.0, abs=1e-12)
    assert lm.average_wd_processed(real, synth, exclude_cols=["b"]) == pytest.approx(1.0, abs=1e-12)


def test_avg_kl_hist_identical_samples_is_zero_and_hand_value(lm) -> None:
    rng = np.random.default_rng(1)
    real = pd.DataFrame({"a": rng.normal(size=300), "b": rng.uniform(size=300)})
    assert lm.avg_kl_hist(real, real.copy(), ["a", "b"], n_bins=20) == pytest.approx(0.0, abs=1e-15)
    # hand computed: 2 bins on [0, 1]; real counts (2, 2) -> p=(.5,.5); synth counts (1, 3) -> q=(.25,.75)
    # KL(p||q) = .5 ln 2 + .5 ln(2/3) = .5 ln(4/3)
    r = pd.DataFrame({"x": [0.0, 0.0, 1.0, 1.0]})
    s = pd.DataFrame({"x": [0.0, 1.0, 1.0, 1.0]})
    assert lm.avg_kl_hist(r, s, ["x"], n_bins=2) == pytest.approx(0.5 * math.log(4.0 / 3.0), abs=1e-9)
    # legacy quirk: the bin grid is the UNION range, so one synthetic outlier changes the value
    s_out = pd.DataFrame({"x": [0.0, 1.0, 1.0, 101.0]})
    assert lm.avg_kl_hist(r, s_out, ["x"], n_bins=2) != pytest.approx(0.5 * math.log(4.0 / 3.0), abs=1e-3)
    # legacy quirk: a degenerate pooled range contributes exactly 0
    const = pd.DataFrame({"x": [2.0] * 5})
    assert lm.avg_kl_hist(const, const, ["x"]) == 0.0


def test_behavioural_variants_are_really_different(lm) -> None:
    rng = np.random.default_rng(2)
    real = pd.DataFrame({"a": rng.normal(size=50), "b": rng.normal(size=50), "k": np.ones(50)})
    synth = pd.DataFrame({"a": rng.normal(size=50), "b": rng.normal(size=50), "k": np.ones(50)})
    assert math.isnan(lm.corr_frobenius_raw(real, synth, ["a", "b", "k"]))
    v = lm.corr_frobenius_fillna0(real, synth, ["a", "b", "k"])
    assert math.isfinite(v)
    assert lm.corr_frobenius_fillna0_autocols(real, synth) == v
    assert lm.corr_frobenius_raw(real, synth, ["a", "b"]) == lm.corr_frobenius_fillna0(real, synth, ["a", "b"])

    with pytest.warns(RuntimeWarning):
        assert math.isnan(lm.avg_wd(real, synth, []))
    for fn in (lm.avg_wd_autocols, lm.avg_kl_hist_autocols, lm.corr_frobenius_fillna0_autocols):
        with pytest.raises(ValueError, match="No common numeric columns"):
            fn(real, synth, [])

    spaced = pd.DataFrame(columns=["x", "shares"])
    assert lm.resolve_target_col_whitespace_tolerant(spaced, "online_news_popularity") == "shares"
    with pytest.raises(ValueError):
        lm.resolve_target_col_last_column_fallback("online_news_popularity", spaced, strict=False)
    with pytest.raises(KeyError):
        lm.resolve_target_col_whitespace_tolerant(spaced, "unmapped")
    with contextlib.redirect_stdout(io.StringIO()) as buf:
        assert lm.resolve_target_col_last_column_fallback("unmapped", spaced, strict=False) == "shares"
    assert "[WARN]" in buf.getvalue()

    class _Boom(Exception):
        pass

    def _raise(exc):
        def factory(*a, **k):
            raise exc
        return factory

    import catboost

    original = catboost.CatBoostRegressor
    try:
        catboost.CatBoostRegressor = _raise(_Boom("construction failed"))
        assert type(lm.make_regressor_broad_fallback(0)).__name__ == "HistGradientBoostingRegressor"
        with pytest.raises(_Boom):
            lm.make_regressor_importerror_fallback(0)
    finally:
        catboost.CatBoostRegressor = original
    assert type(lm.make_regressor_importerror_fallback(0)).__name__ == "CatBoostRegressor"


# --------------------------------------------------------------------------- (iv)

_WINDOWS_USER_PATHS = (b"C:/Users", b"C:\\Users", b"C:\\\\Users")


def _tracked_result_files() -> List[Path]:
    out: List[Path] = []
    for d in RESULT_DIRS:
        out += sorted(p for p in (LEGACY / d).iterdir() if p.is_file() and p.suffix in (".json", ".csv"))
    return out


def test_no_windows_user_path_outside_tracked_result_json() -> None:
    allowed = {p for p in _tracked_result_files() if p.suffix == ".json"}
    offenders = []
    for p in sorted(LEGACY.rglob("*")):
        if not p.is_file() or "__pycache__" in p.parts or p in allowed:
            continue
        data = p.read_bytes()
        if any(n in data for n in _WINDOWS_USER_PATHS):
            offenders.append(str(p.relative_to(REPO)))
    assert not offenders, f"developer-machine Windows paths found in: {offenders}"
    # the historical record itself still carries them (that is a provenance fact, not a bug to fix)
    carrying = [p for p in allowed if any(n in p.read_bytes() for n in _WINDOWS_USER_PATHS)]
    assert len(carrying) == 9 and all(p.parent.name == "dsbm_kfold_eval" for p in carrying)


def test_tracked_result_files_are_byte_identical_to_the_pre_move_commit() -> None:
    files = _tracked_result_files()
    assert len(files) >= 3
    if _git_show(f"{PRE_MOVE_COMMIT}:{_pre_move_path(files[0])}") is None:
        pytest.skip(f"git object {PRE_MOVE_COMMIT} unavailable (shallow clone or no git)")
    compared = 0
    for p in files:
        blob = _git_show(f"{PRE_MOVE_COMMIT}:{_pre_move_path(p)}")
        assert blob is not None, f"{_pre_move_path(p)} missing at {PRE_MOVE_COMMIT}"
        assert blob == p.read_bytes(), f"{p.relative_to(REPO)} differs from {PRE_MOVE_COMMIT}:{_pre_move_path(p)}"
        compared += 1
    assert compared == len(files) >= 3
    must_have = {
        "calculating_metrics/dsbm_kfold_eval/diabetes_kfold_summary.json",
        "calculating_metrics/tabpfgen_kfold_eval/diabetes_kfold_summary.json",
        "tuning_script/dsbm_optuna_results/diabetes_best.json",
        "tuning_results/best_params/dsbm_best_params.json",
    }
    assert must_have <= {p.relative_to(LEGACY).as_posix() for p in files}


def test_provenance_counts_quoted_in_the_readme_are_still_true() -> None:
    optuna = sorted((LEGACY / "tuning_script" / "dsbm_optuna_results").glob("*_best.json"))
    noise_false = [p for p in optuna if json.loads(p.read_text())["best_params"]["noise"] is False]
    assert (len(noise_false), len(optuna)) == (3, 9)
    best = json.loads((LEGACY / "tuning_results" / "best_params" / "dsbm_best_params.json").read_text())
    assert (sum(v["noise"] is False for v in best.values()), len(best)) == (5, 14)
    assert sum(v["imf_len"] == 9 for v in best.values()) == 5          # outside the tracked {3, 5, 7}
    assert all("grad_clip" in v for v in best.values())                 # never tuned by tracked code
    readme = (LEGACY / "README.md").read_text(encoding="utf-8")
    for needle in ("3 of 9", "5 of 14", "UNRESOLVED", "PROVENANCE", "has been run end-to-end"):
        assert needle in readme, needle


# --------------------------------------------------------------------------- (v)

_CONSOLIDATED = ("avg_wd", "avg_kl_hist", "corr_frobenius", "average_wd", "average_wd_processed",
                 "_common_numeric_cols")


def _historical_namespace(script: Path) -> Optional[Dict]:
    blob = _git_show(f"{PRE_MOVE_COMMIT}:{_pre_move_path(script)}")
    if blob is None:
        return None
    from scipy.stats import wasserstein_distance

    keep = [n for n in ast.parse(blob.decode("utf-8")).body
            if isinstance(n, ast.FunctionDef) and n.name in _CONSOLIDATED]
    future = ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)
    mod = ast.Module(body=[future] + keep, type_ignores=[])
    ast.fix_missing_locations(mod)
    ns: Dict = {"np": np, "pd": pd, "wasserstein_distance": wasserstein_distance}
    exec(compile(mod, str(script), "exec"), ns)  # historical source of this very repository
    return ns


def _bindings(script: Path, lm) -> Dict:
    out = {}
    for node in ast.parse(script.read_text(encoding="utf-8")).body:
        if isinstance(node, ast.ImportFrom) and node.module == "sbtab.experiments.legacy.legacy_metrics":
            for a in node.names:
                out[a.asname or a.name] = getattr(lm, a.name)
    return out


def _outcome(fn, *args, **kwargs) -> Tuple:
    buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf), np.errstate(all="ignore"):
            import warnings

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                value = fn(*args, **kwargs)
        return ("ok", value, buf.getvalue())
    except Exception as e:  # noqa: BLE001 - the exception itself is the compared behaviour
        return ("exc", type(e).__name__, str(e), buf.getvalue())


def _same(a, b) -> bool:
    if isinstance(a, float) and isinstance(b, float):
        return (math.isnan(a) and math.isnan(b)) or a == b
    if isinstance(a, tuple) and isinstance(b, tuple):
        return len(a) == len(b) and all(_same(x, y) for x, y in zip(a, b))
    return a == b


def test_each_script_imports_the_variant_that_matches_its_historical_copy(lm) -> None:
    scripts = [p for p in _legacy_scripts() if p.name != "__init__.py"]
    # Skip ONLY when git history is genuinely unavailable -- probe a file known to exist there,
    # so that an unrelated new file can never switch this test off.
    if _git_show(f"{PRE_MOVE_COMMIT}:sbtab/experiments/joint_continuous_metrics.py") is None:
        pytest.skip(f"git object {PRE_MOVE_COMMIT} unavailable (shallow clone or no git)")

    rng = np.random.default_rng(0)
    n = 80
    real = pd.DataFrame({"a": rng.normal(size=n), "b": rng.exponential(size=n), "k": np.ones(n),
                         "y": rng.normal(5, 40, size=n), "s": ["x"] * n})
    synth = pd.DataFrame({"a": rng.normal(.3, 1.2, size=n), "b": rng.exponential(1.5, size=n), "k": np.ones(n),
                          "y": rng.normal(9, 40, size=n), "s": ["z"] * n})
    num = ["a", "b", "k", "y"]
    cases = {
        "avg_wd": [((real, synth, num), {}), ((real, synth, []), {}), ((real, synth), {"cols": None})],
        "average_wd": [((real, synth, num), {}), ((real, synth, []), {})],
        "avg_kl_hist": [((real, synth, num), {"n_bins": 20}), ((real, synth, num), {}),
                        ((real, synth, []), {}), ((real, synth), {"cols": None, "n_bins": 20})],
        "corr_frobenius": [((real, synth, num), {}), ((real, synth, ["a", "b", "y"]), {}),
                           ((real, synth, []), {}), ((real, synth), {"cols": None})],
        "average_wd_processed": [((real, synth), {}), ((real[["s"]], synth[["s"]]), {})],
        "_common_numeric_cols": [((real, synth), {}), ((real, synth[["b", "a"]]), {})],
    }
    compared = 0
    scripts_with_history = 0
    for script in scripts:
        hist = _historical_namespace(script)
        if hist is None:  # a file added after the move has no historical copy to compare with
            continue
        scripts_with_history += 1
        new = _bindings(script, lm)
        for name in _CONSOLIDATED:
            if name not in hist or name not in new:
                continue
            for args, kwargs in cases[name]:
                h, c = _outcome(hist[name], *args, **kwargs), _outcome(new[name], *args, **kwargs)
                assert h[0] == c[0] and all(_same(x, y) for x, y in zip(h[1:], c[1:])), (
                    f"{script.name}::{name}{kwargs}: historical={h} consolidated={c}"
                )
                compared += 1
    assert scripts_with_history == 17, scripts_with_history
    assert compared >= 100, compared


def test_sliced_wasserstein_is_the_de5acc9_function(lm) -> None:
    blob = _git_show("de5acc9:sbtab/evaluation/metrics/statistical.py")
    if blob is None:
        pytest.skip("git object de5acc9 unavailable (it lives on origin/feat/evaluation-metrics only)")
    src = blob.decode("utf-8")
    fn = next(n for n in ast.parse(src).body if isinstance(n, ast.FunctionDef) and n.name == "sliced_wasserstein")
    historical = "".join(src.splitlines(keepends=True)[fn.lineno - 1:fn.end_lineno])
    new_src = LEGACY_METRICS.read_text(encoding="utf-8")
    fn2 = next(n for n in ast.parse(new_src).body
               if isinstance(n, ast.FunctionDef) and n.name == "sliced_wasserstein")
    assert ast.dump(fn) == ast.dump(fn2)
    assert "".join(new_src.splitlines(keepends=True)[fn2.lineno - 1:fn2.end_lineno]) == historical

    import torch

    x = np.random.default_rng(0).normal(size=(64, 3))
    assert lm.sliced_wasserstein(x, x + 5.0) == pytest.approx(0.0, abs=1e-8)  # legacy quirk: centred first
    torch.manual_seed(0)
    a = lm.sliced_wasserstein(x, 2.0 * x)
    torch.manual_seed(0)
    assert a == lm.sliced_wasserstein(x, 2.0 * x) and a > 0.0
