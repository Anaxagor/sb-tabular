"""Experiment-stage acceptance tests: Optuna budget/resume, sample sizes, persistence, timing, registry coverage."""
import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from sbtab.adapters.base import AdapterConfigError, ModelAdapter
from sbtab.data.dataset_schema import ColumnSpec, DatasetSchema
from sbtab.data.loading import frame_fingerprint
from sbtab.experiments import aggregate_results, calculate_metrics, cross_validate, prepare_splits as ps, tune
from sbtab.experiments.experiment_common import SeedLedger, StageError, json_safe, load_protocol
from sbtab.solvers.registry import get_adapter_class, missing_requirements, solver_registry

SMOKE_SPACE = "configs/search_spaces/smoke/{}.yaml"


def make_dataset(tmp_path, monkeypatch, task="classification", n=260):
    rng = np.random.default_rng(0)
    c = rng.integers(0, 3, n)
    y = np.where(rng.random(n) < 0.25 + 0.25 * (c == 2), "yes", "no") if task == "classification" else rng.normal(size=n) * 3 + c + 10
    f = pd.DataFrame({"x": rng.normal(size=n) + c, "k": rng.integers(0, 4, n).astype(float),
                      "c": np.array(["a", "b", "c"])[c], "y": y}, index=pd.RangeIndex(n, name="row_id"))
    schema = DatasetSchema("toy", (ColumnSpec("x", "continuous"), ColumnSpec("k", "discrete"), ColumnSpec("c", "categorical"),
                                   ColumnSpec("y", "categorical" if task == "classification" else "continuous", role="target")), task=task)
    manifest = {"fingerprint": frame_fingerprint(f), "missing_counts": {}, "dataset": "toy", "source": {}, "n_rows": n,
                "n_columns": 4, "schema_hash": schema.hash(), "regime": schema.regime, "task": task, "target": "y",
                "dropped_columns": [], "value_maps": {}, "missing_policy": "reject", "row_filtering": "none"}
    monkeypatch.setattr(ps, "load_dataset", lambda name, config_dir=None: (f, schema, manifest))
    root = tmp_path / "sbtab_smoke_v2"
    assert ps.run("toy", load_protocol(smoke=True), root)["split_status"] == "ok"
    return root, root / "toy" / "splits.json", f, schema


@pytest.fixture(autouse=True)
def quiet(monkeypatch):
    monkeypatch.setenv("TQDM_DISABLE", "1")


# --------------------------------------------------------------------------- Optuna / resume
def test_tabpfgen_large_training_pool_is_subsampled_in_tuning_and_cv(tmp_path, monkeypatch):
    from sbtab.baselines.tabpfn import model as pfn

    root, splits_path, _, _ = make_dataset(tmp_path, monkeypatch, n=1500)
    splits = json.loads(splits_path.read_text())
    seen = []

    class Generator:
        def validate_context(self, X, y, task):
            seen.append(len(X))
            assert len(X) == len(y) == 1000
            assert set(y) == {0, 1}

        def generate_classification(self, X, y, n_samples, balance_classes):
            return np.random.normal(size=(n_samples, X.shape[1])), np.random.randint(0, 2, n_samples)

    monkeypatch.setattr(tune, "missing_requirements", lambda model: ())
    monkeypatch.setattr(cross_validate, "missing_requirements", lambda model: ())
    monkeypatch.setattr(pfn, "_default_generator_factory", lambda cfg: Generator())
    space = SMOKE_SPACE.format("tabpfgen")
    preview = tune.run("toy", "tabpfgen", splits_path, space, resume=False, smoke=True, dry_run=True)
    assert preview["tabpfgen_context"] == {
        "n_train_rows": 1275, "n_context_rows": 1000, "row_limit": 1000, "subsampled": True}
    assert not list(root.rglob("study.sqlite3"))
    tuned = tune.run("toy", "tabpfgen", splits_path, space, resume=False, smoke=True)
    assert tuned["counts"]["COMPLETE"] == 3
    run = Path(tuned["run_dir"])
    for status_path in (run / "tuning").glob("trial-*/status.json"):
        status = json.loads(status_path.read_text())
        assert status["describe"]["conditioning_context"]["n_context_rows"] == 1000
        assert len(pd.read_parquet(status_path.parent / "synthetic.parquet")) == splits["n_V"]
    cv = cross_validate.run("toy", "tabpfgen", run / "tuning/selected_config.json", splits_path,
                            root, smoke=True)
    assert cv["n_ok"] == 5 and seen == [1000] * 8
    for fold in splits["folds"]:
        assert len(pd.read_parquet(run / "cv" / f"fold-{fold['fold']}" / "synthetic.parquet")) == len(fold["train_row_ids"])


@pytest.mark.integration
def test_tuning_budget_resume_failures_best_selection_and_downstream_stages(tmp_path, monkeypatch):
    root, splits_path, frame, schema = make_dataset(tmp_path, monkeypatch)
    space = SMOKE_SPACE.format("mixedsbm")
    splits = json.loads(splits_path.read_text())

    # --- interrupted after one trial -> provisional; a second start without --resume is refused
    out = tune.run("toy", "mixedsbm", splits_path, space, resume=False, smoke=True, max_new_trials=1)
    assert out["counts"]["allocated"] == 1 and out["selection"] == "provisional"
    with pytest.raises(StageError):
        tune.run("toy", "mixedsbm", splits_path, space, resume=False, smoke=True)

    # --- make the NEXT trial fail at runtime: the failed record is kept and COUNTED, never replaced
    real_fit, calls = ModelAdapter.fit, {"n": 0}

    def flaky(self, *a, **k):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("injected training failure")
        return real_fit(self, *a, **k)

    monkeypatch.setattr(ModelAdapter, "fit", flaky)
    out = tune.run("toy", "mixedsbm", splits_path, space, resume=True, smoke=True)
    monkeypatch.setattr(ModelAdapter, "fit", real_fit)
    budget = load_protocol(smoke=True)["tuning"]["n_trials"]
    assert out["budget"] == budget == 3 and out["new_trials"] == 2           # only the REMAINING budget
    assert out["counts"] == {"allocated": 3, "COMPLETE": 2, "FAIL": 1, "PRUNED": 0, "RUNNING": 0, "WAITING": 0}
    again = tune.run("toy", "mixedsbm", splits_path, space, resume=True, smoke=True)
    assert again["new_trials"] == 0 and again["counts"]["allocated"] == 3    # never a second full budget

    run_dir = next((root / "toy" / "mixedsbm").iterdir())
    tdir = run_dir / "tuning"
    trials = pd.read_csv(tdir / "trials.csv")
    assert list(trials["state"]) == ["COMPLETE", "FAIL", "COMPLETE"]
    failed = json.loads((tdir / "trial-001" / "status.json").read_text())
    assert failed["status"] == "training_failed" and "injected training failure" in failed["failure"]["message"]
    assert (tdir / "trial-001" / "timing.json").exists()                     # elapsed time of a failed trial is kept

    # --- best = exact minimum finite COMPLETE objective; final only after budget + checkpoint reload
    best = json.loads((tdir / "best.json").read_text())
    done = trials[trials["state"] == "COMPLETE"]
    assert best["selection"] == "final" and best["reload_verified"] is True
    assert best["trial"] == int(done.loc[done["objective"].idxmin(), "trial"]) and best["objective"] == pytest.approx(done["objective"].min())
    assert (tdir / best["checkpoint"] / "adapter.json").exists()             # pointer to the exact selected checkpoint
    for t in done["trial"]:
        d = tdir / f"trial-{int(t):03d}"
        assert len(pd.read_parquet(d / "synthetic.parquet")) == splits["n_V"]      # exactly len(V), no sample cap
        cfgj = json.loads((d / "config.json").read_text())
        assert cfgj["effective_config"]["lr"] == pytest.approx(cfgj["sampled_params"]["lr"])   # sampled value is the instantiated one
        assert json.loads((d / "status.json").read_text())["n_updates"] > 0
    assert (tdir / "sampler.pkl").exists()                                   # sampler RNG state, not just the study DB

    # --- resume is refused when the compatibility fingerprint changes (here: a different search space)
    other = tmp_path / "other.yaml"
    sp = yaml.safe_load(open(space))
    sp["fixed"]["num_steps"] = 6
    other.write_text(yaml.safe_dump(sp))
    with pytest.raises(StageError) as e:
        tune.run("toy", "mixedsbm", splits_path, other, resume=True, smoke=True, run_id=run_dir.name)
    assert "search_space_hash" in str(e.value)

    # ------------------------------------------------------------------ CV: fresh fits, len(T_k) rows
    loads, real_load = [], ModelAdapter.load_checkpoint.__func__

    def spy_load(cls, path):
        loads.append(str(path))
        return real_load(cls, path)

    monkeypatch.setattr(ModelAdapter, "load_checkpoint", classmethod(spy_load))
    cv = cross_validate.run("toy", "mixedsbm", tdir / "selected_config.json", splits_path, root, smoke=True)
    assert cv["n_ok"] == 5 and cv["folds_status"] == {str(k): "ok" for k in range(5)}
    assert len(loads) == 5 and all("/cv/fold-" in p for p in loads)           # only each fold's own checkpoint (reload check)
    assert not any("/tuning/" in p for p in loads)                           # tuned weights are never loaded
    selected = json.loads((tdir / "selected_config.json").read_text())["config"]
    for k, fold in enumerate(splits["folds"]):
        m = json.loads((run_dir / "cv" / f"fold-{k}" / "manifest.json").read_text())
        assert m["n_generated"] == len(fold["train_row_ids"]) == len(pd.read_parquet(run_dir / "cv" / f"fold-{k}" / "synthetic.parquet"))
        assert m["train_row_hash"] == fold["train_hash"] == m["preprocessor"]["fit_row_hash"]
        assert m["effective_config"] == selected and m["reload_verified"]["ok"]
        assert {"preprocessing_seconds", "model_init_seconds", "generator_fit_seconds", "checkpoint_io_seconds", "generation_seconds",
                "inverse_transform_seconds", "training_total_seconds", "stage_wall_seconds"} <= set(m["timing"])
    assert len({json.loads((run_dir / "cv" / f"fold-{k}" / "manifest.json").read_text())["seeds"]["fit"] for k in range(5)}) == 5

    # ------------------------------------------------------------------ metrics: saved artifacts only
    before = {p: open(p, "rb").read() for p in glob.glob(str(run_dir / "cv" / "**" / "*"), recursive=True) if not p.endswith(("/", "checkpoint", "preprocessor")) and not __import__("os").path.isdir(p)}
    boom = lambda *a, **k: (_ for _ in ()).throw(AssertionError("generator fit()/load_checkpoint() called by calculate_metrics"))
    monkeypatch.setattr(ModelAdapter, "fit", boom)
    monkeypatch.setattr(ModelAdapter, "load_checkpoint", classmethod(boom))
    res = calculate_metrics.run(run_dir / "cv" / "cv_run_manifest.json", "configs/metrics/metrics_v1.yaml")
    ev_dir = __import__("pathlib").Path(res["evaluation_dir"])
    assert {p: open(p, "rb").read() for p in before} == before              # generator artifacts untouched, fit time preserved
    for name in ("per_fold.csv", "summary.json", "summary.csv"):
        assert (ev_dir / name).exists()
    for name in ("metrics.json", "per_feature.parquet", "per_condition.parquet", "associations.npz", "utility_predictions.parquet", "timing.json"):
        assert (ev_dir / "fold-0" / name).exists()
    json.loads((ev_dir / "summary.json").read_text(), parse_constant=lambda c: (_ for _ in ()).throw(ValueError(f"non-standard JSON token {c}")))

    summary = json.loads((ev_dir / "summary.json").read_text())
    by = {m["metric"]: m for m in summary["metrics"]}
    fit_t = by["generator.generator_fit_seconds"]
    assert fit_t["n_expected"] == 5 and fit_t["n_valid"] == 5 and fit_t["complete"] and fit_t["std_ddof"] == 1
    vals = list(fit_t["fold_values"].values())
    assert fit_t["mean"] == pytest.approx(np.mean(vals)) and fit_t["std"] == pytest.approx(np.std(vals, ddof=1))
    assert vals == [json.loads((run_dir / "cv" / f"fold-{k}" / "timing.json").read_text())["seconds"]["generator_fit_seconds"] for k in range(5)]

    # the utility learner saw the FULL synthetic table, and the real reference is cached per dataset, not per model
    m0 = json.loads((ev_dir / "fold-0" / "metrics.json").read_text())
    assert m0["utility"]["n_synth_train_rows"] == len(splits["folds"][0]["train_row_ids"]) and m0["utility"]["task"] == "classification"
    assert (root / "toy" / "utility_config.json").exists() and len(list((root / "toy" / "utility_reference").glob("*/fold-*.json"))) == 5
    first = pd.read_csv(ev_dir / "per_fold.csv")
    calculate_metrics.run(run_dir / "cv" / "cv_run_manifest.json", "configs/metrics/metrics_v1.yaml")     # second pass: cache hits
    second = pd.read_csv(ev_dir / "per_fold.csv")
    real = lambda d: d[d["metric"].str.startswith("utility.real") & ~d["metric"].str.contains("seconds")].set_index(["fold", "metric"])["value"]
    pd.testing.assert_series_equal(real(first), real(second))                # bit-identical real reference

    # a changed metric configuration writes to a NEW namespace instead of overwriting
    cfg2 = tmp_path / "metrics_v1b.yaml"
    d2 = yaml.safe_load(open("configs/metrics/metrics_v1.yaml"))
    d2["conditional_min_rows"] = 5
    cfg2.write_text(yaml.safe_dump(d2))
    res2 = calculate_metrics.run(run_dir / "cv" / "cv_run_manifest.json", cfg2, folds=[0])
    assert res2["evaluation_dir"] != res["evaluation_dir"] and (ev_dir / "summary.json").exists()

    # a model with fewer valid folds is never presented as a complete five-fold comparison
    part = aggregate_results.aggregate_run(res2["evaluation_dir"], n_expected=5)
    assert part["metrics"][0]["n_valid"] <= 1 and not part["metrics"][0]["complete"]
    assert any(e["status"] == "not_evaluated" for e in part["metrics"][0]["excluded"])



# --------------------------------------------------------------------------- effective parameters / registry
def test_every_search_space_key_is_consumed_by_its_adapter_and_noise_is_never_searched():
    paths = glob.glob("configs/search_spaces/*.yaml") + glob.glob("configs/search_spaces/smoke/*.yaml")
    assert len(paths) >= 36
    for p in paths:
        d = yaml.safe_load(open(p))
        s = tune.load_search_space(p, d["model"], d["kind"])
        assert "noise" not in s["params"] and "noise" not in s["fixed"]
        adapter = get_adapter_class(d["model"])
        assert set(s["params"]) | set(s["fixed"]) <= set(adapter.DEFAULTS)
    supported = {e.id for e in solver_registry.values() if e.status == "supported"}
    assert supported <= {yaml.safe_load(open(p))["model"] for p in glob.glob("configs/search_spaces/*.yaml")}


def test_unknown_or_ignored_hyperparameters_are_rejected(tmp_path):
    for model, key in (("mixedsbm", "n_epochs"), ("tabddpm", "n_epochs"), ("dsbm_dt_joint_gbt", "noise"), ("lightsb", "is_diagonal")):
        with pytest.raises(AdapterConfigError):
            get_adapter_class(model).resolve_config({key: 1})
    bad = tmp_path / "bad.yaml"
    bad.write_text(yaml.safe_dump({"model": "mixedsbm", "kind": "smoke", "fixed": {}, "params": {"noise": {"type": "categorical", "choices": [True, False]}}}))
    with pytest.raises((StageError, AdapterConfigError)):
        tune.load_search_space(bad, "mixedsbm", "smoke")
    with pytest.raises(StageError):
        tune.load_search_space(SMOKE_SPACE.format("mixedsbm"), "mixedsbm", "production")   # smoke space, production protocol


def test_registry_is_complete_and_honest():
    import importlib, os
    ids = set(solver_registry)
    assert {"dsb_ct_joint_mlp", "dsbm_ct_joint_mlp", "dsb_ct_joint_gbt", "dsbm_ct_joint_gbt", "dsb_ct_structural_gbt", "dsb_dt_joint_mlp",
            "dsbm_dt_joint_mlp", "dsb_dt_joint_gbt", "dsbm_dt_joint_gbt", "dsb_dt_structural_gbt", "dsbm_dt_structural_gbt",
            "lightsb", "csbm", "mixedsbm", "ctgan", "tabddpm", "tabpfgen"} <= ids
    for e in solver_registry.values():
        if e.status == "unavailable":
            assert e.adapter is None and e.implementation is None and len(e.notes) > 20
            with pytest.raises(LookupError):
                get_adapter_class(e.id)
        else:
            assert os.path.isdir(e.implementation), e.implementation
            adapter = get_adapter_class(e.id)
            assert adapter.registry_id == e.id and set(adapter.supported_regimes) == set(e.regimes), e.id
    assert {"tabsyn", "tabbyflow", "lightsb_m", "stasy"} <= {e.id for e in solver_registry.values() if e.status == "unavailable"}
    assert solver_registry["forestdiffusion"].status == "supported"
    assert "NOT" in solver_registry["ve_score_sde_simplified"].notes and "STaSy" in solver_registry["ve_score_sde_simplified"].notes


def test_seed_ledger_gives_distinct_recorded_seeds():
    s = SeedLedger(5)
    seeds = [s.seed("fit", k) for k in range(5)] + [s.seed("sample", k) for k in range(5)]
    assert len(set(seeds)) == 10 and s.seed("fit", 0) == SeedLedger(5).seed("fit", 0) and SeedLedger(6).seed("fit", 0) != seeds[0]
    assert set(s.to_dict()["derived"]) == {f"{p}:{k}" for p in ("fit", "sample") for k in range(5)}
    assert json.dumps(json_safe({"a": float("nan"), "b": np.float64("inf"), "c": np.int64(3)}), allow_nan=False) == '{"a": null, "b": null, "c": 3}'


# --------------------------------------------------------------------------- every supported entry: fit -> sample -> serialise -> metrics
def _regime_tables():
    rng = np.random.default_rng(1)
    n = 90
    c = rng.integers(0, 3, n)
    mixed = pd.DataFrame({"x": rng.normal(size=n) + c, "k": rng.integers(0, 3, n).astype(float), "c": c, "y": (rng.random(n) < 0.4).astype(int)})
    ms = DatasetSchema("mixed", (ColumnSpec("x", "continuous"), ColumnSpec("k", "discrete"), ColumnSpec("c", "categorical"),
                                 ColumnSpec("y", "categorical", role="target")), task="classification")
    cont = pd.DataFrame({"a": rng.normal(size=n), "b": rng.normal(size=n), "y": rng.normal(size=n)})
    cs = DatasetSchema("cont", (ColumnSpec("a", "continuous"), ColumnSpec("b", "continuous"), ColumnSpec("y", "continuous", role="target")), task="regression")
    disc = pd.DataFrame({"u": rng.integers(0, 3, n), "v": rng.integers(0, 4, n).astype(float), "y": (rng.random(n) < 0.5).astype(int)})
    ds = DatasetSchema("disc", (ColumnSpec("u", "categorical"), ColumnSpec("v", "discrete"), ColumnSpec("y", "categorical", role="target")), task="classification")
    for t in (mixed, cont, disc):
        t.index.name = "row_id"
    return {"mixed": (mixed, ms), "continuous": (cont, cs), "discrete": (disc, ds)}


SUPPORTED = sorted(e.id for e in solver_registry.values() if e.status in ("supported", "heuristic"))


@pytest.mark.integration
@pytest.mark.parametrize("model_id", SUPPORTED)
def test_every_supported_entry_fits_samples_serialises_and_is_evaluated_in_each_compatible_regime(tmp_path, model_id):
    missing = missing_requirements(model_id)
    if missing:
        pytest.skip(f"optional packages {list(missing)} are not installed. This skip is NOT a validation of the "
                    f"{model_id} adapter: its fit/sample/serialise path was not exercised here.")
    from sbtab import evaluation as ev
    entry, adapter_cls = solver_registry[model_id], get_adapter_class(model_id)
    space = yaml.safe_load(open(SMOKE_SPACE.format(model_id)))
    # a complete trial configuration: the fixed block plus a deterministic value for every SEARCHED key
    config = {**space["fixed"], **{k: (v["choices"][0] if v["type"] == "categorical" else v["low"]) for k, v in space["params"].items()}}
    tables = _regime_tables()
    for regime in entry.regimes:
        train, schema = tables[regime]
        if model_id == "tabpfgen" and schema.target is None:
            continue
        a = adapter_cls().fit(train, schema, config, seed=3)
        assert a.n_updates is None or a.n_updates >= 0
        for n in (1, 64, 65):
            g = a.sample(n, seed=11)
            assert len(g) == n and list(g.columns) == schema.column_order and g.index.name == "synthetic_id"
        a.save_checkpoint(tmp_path / f"{regime}")
        b = adapter_cls.load_checkpoint(tmp_path / f"{regime}")
        g1, g2 = a.sample(40, seed=5), b.sample(40, seed=5)
        np.testing.assert_allclose(g1.to_numpy(dtype=float), g2.to_numpy(dtype=float), atol=1e-6)     # reload without refit
        ctx = ev.MetricContext.fit(train, schema, ev.MetricConfig(), train_row_ids=list(train.index))
        validity = ev.check_validity(g1, schema, ctx)
        assert validity["status"] == "ok", (model_id, regime, validity)
        obj = ev.tuning_objective(train, g1, schema)
        assert obj["status"] == "ok" and np.isfinite(obj["objective"]) and obj["regime"] == regime
    unsupported = set(tables) - set(entry.regimes)
    for regime in unsupported:                                          # an incompatible regime is refused, not mangled
        with pytest.raises(ValueError):
            adapter_cls().fit(tables[regime][0], tables[regime][1], config, seed=0)


# --------------------------------------------------------------------------- ranks
def _records(rows):
    return pd.DataFrame([{"dataset": ds, "fold": f, "model": m, "metric": "wd", "value": v, "status": st,
                          "direction": "lower_is_better"} for ds, f, m, v, st in rows])


def test_ranks_use_identical_fold_sets_and_never_compare_across_datasets():
    rows = []
    for f in range(5):                                         # A and B on d1 (A always better); C only on d2
        rows += [("d1", f, "A", 0.1 + 0.01 * f, "ok"), ("d1", f, "B", 0.5, "ok"), ("d2", f, "C", 0.2, "ok")]
    r = aggregate_results.average_ranks(_records(rows), "wd")
    assert r["per_dataset"]["d1"]["ranks"] == {"A": 1.0, "B": 2.0} and r["per_dataset"]["d1"]["n_models"] == 2
    assert r["per_dataset"]["d2"]["status"] == "not_applicable"            # a single model is not a comparison
    assert r["overall"]["status"] == "not_applicable" and r["overall"]["n_models"] == 3   # C shares no cell with A, B
    # an explicit model set that WAS run on common cells gets an overall rank, with the number of compared models
    o = aggregate_results.average_ranks(_records(rows), "wd", models=["A", "B"])["overall"]
    assert o["status"] == "ok" and o["ranks"] == {"A": 1.0, "B": 2.0} and o["n_models"] == 2 and o["n_cells"] == 5

    # a model with a failed fold is NOT ranked on a different fold set: it is excluded and listed
    broken = [x for x in rows if not (x[2] == "B" and x[1] == 3)] + [("d1", 3, "B", None, "sampling_failed")]
    rb = aggregate_results.average_ranks(_records(broken + [("d1", f, "D", 0.3, "ok") for f in range(5)]), "wd")
    d1 = rb["per_dataset"]["d1"]
    assert d1["models"] == ["A", "D"] and d1["excluded_models"] == ["B"] and d1["n_models"] == 2 and d1["n_cells"] == 5

    hi = _records([("d", f, m, v, "ok") for f in range(3) for m, v in (("A", 0.9), ("B", 0.2))]).assign(direction="higher_is_better")
    assert aggregate_results.average_ranks(hi, "wd")["overall"]["ranks"] == {"A": 1.0, "B": 2.0}     # higher is better
    signed = _records([("d", f, m, v, "ok") for f in range(3) for m, v in (("A", -0.001), ("B", 0.2))]).assign(direction="closer_to_zero_is_better")
    assert aggregate_results.average_ranks(signed, "wd")["overall"]["ranks"] == {"A": 1.0, "B": 2.0}  # signed MMD: |value|


# --------------------------------------------------------------------------- stale RUNNING trials
@pytest.mark.integration
def test_stale_running_trial_of_a_dead_process_is_reconciled_kept_and_counted(tmp_path, monkeypatch):
    import optuna
    root, splits_path, _, _ = make_dataset(tmp_path, monkeypatch)
    space = SMOKE_SPACE.format("mixedsbm")
    tune.run("toy", "mixedsbm", splits_path, space, resume=False, smoke=True, max_new_trials=1)
    tdir = next((root / "toy" / "mixedsbm").iterdir()) / "tuning"
    # a process that died mid-trial leaves an allocated trial in state RUNNING
    study = optuna.load_study(study_name="study", storage=f"sqlite:///{tdir / 'study.sqlite3'}")
    study.ask()
    assert tune.counts(study)["RUNNING"] == 1

    out = tune.run("toy", "mixedsbm", splits_path, space, resume=True, smoke=True)
    assert out["stale_running_reconciled"] == [1]
    assert out["counts"] == {"allocated": 3, "COMPLETE": 2, "FAIL": 1, "PRUNED": 0, "RUNNING": 0, "WAITING": 0}
    assert out["new_trials"] == 1                                  # the stale record COUNTS toward the budget of 3
    trials = pd.read_csv(tdir / "trials.csv")
    assert list(trials["state"]) == ["COMPLETE", "FAIL", "COMPLETE"] and "stale_running" in trials.loc[1, "failure"]
    assert json.loads((tdir / "trial-001" / "status.json").read_text())["failure"]["type"] == "Interrupted"
    assert json.loads((tdir / "best.json").read_text())["selection"] == "final"


# --------------------------------------------------------------------------- effective parameters
def _instantiated_config(adapter) -> str:
    """A canonical dump of what was actually instantiated (solver / wrapper config), not of what was requested."""
    import dataclasses
    obj = getattr(adapter, "solver", None) or getattr(adapter, "model", None)
    cfg = getattr(obj, "cfg", None) or getattr(obj, "config", None)
    return json.dumps(dataclasses.asdict(cfg) if dataclasses.is_dataclass(cfg) else vars(cfg), sort_keys=True, default=str)


def _two_values(spec):
    if spec["type"] == "categorical":
        return spec["choices"][0], spec["choices"][-1]
    lo, hi = spec["low"], spec["high"]
    return (int(lo), int(hi)) if spec["type"] == "int" else (float(lo), float(hi))


@pytest.mark.parametrize("model_id", SUPPORTED)
def test_every_searched_parameter_changes_the_instantiated_model(model_id):
    """Detects accepted-but-ignored hyperparameters (the old `n_epochs`, hard-coded architectures, ...)."""
    if missing_requirements(model_id):
        pytest.skip(f"optional packages {list(missing_requirements(model_id))} are not installed. This skip is NOT a "
                    f"validation of the {model_id} adapter.")
    entry, adapter_cls = solver_registry[model_id], get_adapter_class(model_id)
    train, schema = _regime_tables()[entry.regimes[0]]
    def build(config):
        a = adapter_cls()
        a.schema, a.seed, a.config = schema, 0, adapter_cls.resolve_config(config)
        a._prepare(train)
        a._build_model()                                       # construction only: nothing is trained
        return _instantiated_config(a)

    # each space is varied against ITS OWN base: smoke and production ranges are not interchangeable
    for path in (SMOKE_SPACE.format(model_id), f"configs/search_spaces/{model_id}.yaml"):
        space = yaml.safe_load(open(path))
        assert space["params"], path
        low = {**space["fixed"], **{k: _two_values(v)[0] for k, v in space["params"].items()}}
        high = {**space["fixed"], **{k: _two_values(v)[1] for k, v in space["params"].items()}}
        build(low), build(high)                                # cross-parameter constraints hold at both corners
        for key, spec in space["params"].items():
            lo, hi = _two_values(spec)
            assert lo != hi, (path, key)
            assert build({**low, key: lo}) != build({**low, key: hi}), \
                f"{path}: sampled hyperparameter {key!r} does not change the instantiated model"


def test_production_ipf_search_spaces_respect_the_ou_stability_bound_everywhere():
    """alpha_ou * max(gamma) < 1 over the WHOLE space, so no allocated trial is wasted on an invalid grid."""
    import itertools
    from sbtab.bridge.timegrid import TimeGrid
    for model in ("dsb_ct_joint_mlp", "dsb_dt_joint_mlp"):
        sp = yaml.safe_load(open(f"configs/search_spaces/{model}.yaml"))
        P, F = sp["params"], sp["fixed"]
        for N, h, a in itertools.product(P["num_steps"]["choices"], (P["horizon"]["low"], P["horizon"]["high"]), P["alpha_ou"]["choices"]):
            g = TimeGrid(num_steps=N, gamma_min=1e-4, gamma_max=1e-2, schedule=F["schedule"], horizon=h).dt().max()
            assert a * float(g) < 1.0, (model, N, h, a)
    for model in ("dsb_dt_joint_gbt", "dsb_ct_joint_gbt", "dsb_dt_structural_gbt", "dsb_ct_structural_gbt"):
        P = yaml.safe_load(open(f"configs/search_spaces/{model}.yaml"))["params"]
        assert max(P["alpha_ou"]["choices"]) * P["gamma_max"]["high"] < 1.0
