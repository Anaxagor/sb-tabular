"""Canonical stage boundaries cannot bypass tuning or spend a study budget twice."""
from pathlib import Path

import optuna
import pytest
import yaml

from sbtab.experiments import cross_validate, tune
from sbtab.experiments.experiment_common import StageError, canonical_hash, read_json, write_json
from test_stages import make_dataset


def test_standalone_tuning_lock_preserves_an_active_trial(tmp_path, monkeypatch):
    _, splits, _, _ = make_dataset(tmp_path, monkeypatch)
    space = "configs/search_spaces/smoke/mixedsbm.yaml"
    real_fit = tune.fit_generate
    attempted = []

    def fit_with_competing_resume(*args, **kwargs):
        if not attempted:
            attempted.append(True)
            with pytest.raises(StageError, match="another process owns"):
                tune.run("toy", "mixedsbm", splits, space, resume=True, smoke=True, run_id="race")
        return real_fit(*args, **kwargs)

    monkeypatch.setattr(tune, "fit_generate", fit_with_competing_resume)
    result = tune.run("toy", "mixedsbm", splits, space, resume=False, smoke=True, run_id="race")
    assert result["selection"] == "final"
    assert result["counts"] == {"allocated": 3, "COMPLETE": 3, "FAIL": 0, "PRUNED": 0, "RUNNING": 0, "WAITING": 0}
    assert result["stale_running_reconciled"] == []


def test_an_overspent_study_is_not_a_canonical_final_selection(tmp_path, monkeypatch):
    _, splits, _, _ = make_dataset(tmp_path, monkeypatch)
    space = "configs/search_spaces/smoke/mixedsbm.yaml"
    result = tune.run("toy", "mixedsbm", splits, space, resume=False, smoke=True, max_new_trials=1)
    tuning_dir = Path(result["run_dir"]) / "tuning"
    study = optuna.load_study(study_name="study", storage=f"sqlite:///{tuning_dir / 'study.sqlite3'}")
    for _ in range(3):
        study.ask()
    with pytest.raises(StageError, match="exceeds the allocated trial budget"):
        tune.run("toy", "mixedsbm", splits, space, resume=True, smoke=True)
    assert tune.counts(study)["allocated"] == 4
    assert tune.counts(study)["RUNNING"] == 3
    assert not (tuning_dir / "selected_config.json").exists()


def test_standalone_cv_lock_prevents_overlapping_fold_writes_and_manifest_merge(tmp_path, monkeypatch):
    root, splits, _, _ = make_dataset(tmp_path, monkeypatch)
    result = tune.run("toy", "mixedsbm", splits, "configs/search_spaces/smoke/mixedsbm.yaml",
                      resume=False, smoke=True)
    run_dir = Path(result["run_dir"])
    selected = run_dir / "tuning/selected_config.json"
    real_fit = cross_validate.fit_generate
    attempted = []

    def fit_with_competing_fold(*args, **kwargs):
        if not attempted:
            attempted.append(True)
            with pytest.raises(StageError, match="another process owns"):
                cross_validate.run("toy", "mixedsbm", selected, splits, root, smoke=True, folds=[1])
        return real_fit(*args, **kwargs)

    monkeypatch.setattr(cross_validate, "fit_generate", fit_with_competing_fold)
    first = cross_validate.run("toy", "mixedsbm", selected, splits, root, smoke=True, folds=[0])
    assert first["folds_status"] == {"0": "ok"}
    assert not (run_dir / "cv/fold-1").exists()
    saved = (run_dir / "cv/fold-0/manifest.json").read_bytes()
    second = cross_validate.run("toy", "mixedsbm", selected, splits, root, smoke=True, folds=[1])
    assert second["folds_status"] == {"0": "ok", "1": "ok"}
    assert (run_dir / "cv/fold-0/manifest.json").read_bytes() == saved


def _selection_artifacts(tmp_path, with_study=False):
    selected = {"version": tune.TUNING_VERSION, "model": "mixedsbm", "source": "tuning", "trial": 2,
                "objective": 0.2, "config": {"lr": 0.001}, "compatibility": {"membership_hash": "same-data"}}
    best = {key: value for key, value in selected.items() if key not in ("source", "config")}
    best.update(selection="final", reload_verified=True, budget=100,
                counts={"allocated": 100, "COMPLETE": 97, "FAIL": 2, "PRUNED": 1, "RUNNING": 0, "WAITING": 0})
    path = tmp_path / "tuning/selected_config.json"
    write_json(path, selected)
    write_json(path.parent / "best.json", best)
    write_json(path.parent / "trial-002/config.json", {"effective_config": selected["config"]})
    if with_study:
        study = optuna.create_study(study_name="study", storage=f"sqlite:///{path.parent / 'study.sqlite3'}")
        study.set_user_attr("compatibility", selected["compatibility"])
        study.set_user_attr("compatibility_hash", canonical_hash(selected["compatibility"]))
        records = []
        for index in range(100):
            state = optuna.trial.TrialState.FAIL if index < 2 else (
                optuna.trial.TrialState.PRUNED if index == 99 else optuna.trial.TrialState.COMPLETE)
            value = (0.2 if index == 2 else 1.0 + index) if state == optuna.trial.TrialState.COMPLETE else None
            records.append(optuna.trial.create_trial(state=state, value=value,
                           user_attrs={"effective_config": selected["config"]}))
        study.add_trials(records)
    return path


def test_selected_hyperparameters_are_checked_without_loading_tuned_weights(tmp_path, monkeypatch):
    from sbtab.adapters.base import ModelAdapter
    path = _selection_artifacts(tmp_path, with_study=True)
    def forbidden(*args, **kwargs):
        pytest.fail("CV must not load tuned weights")
    monkeypatch.setattr(ModelAdapter, "load_checkpoint", forbidden)
    assert cross_validate.load_tuned_selection(path, "mixedsbm", 100) == read_json(path)


@pytest.mark.parametrize("change", ["nonminimum", "effective_config", "compatibility", "study_missing"])
def test_consistent_json_pointers_cannot_override_the_completed_optuna_study(tmp_path, change):
    path = _selection_artifacts(tmp_path, with_study=change != "study_missing")
    selected, best = read_json(path), read_json(path.parent / "best.json")
    if change == "nonminimum":
        selected.update(trial=3, objective=4.0)
        best.update(trial=3, objective=4.0)
    elif change == "effective_config":
        selected["config"]["lr"] = 0.3
    elif change == "compatibility":
        selected["compatibility"] = best["compatibility"] = {"membership_hash": "other-data"}
    write_json(path, selected)
    write_json(path.parent / "best.json", best)
    write_json(path.parent / f"trial-{selected['trial']:03d}/config.json", {"effective_config": selected["config"]})
    with pytest.raises(StageError, match="Optuna"):
        cross_validate.load_tuned_selection(path, "mixedsbm", 100)


@pytest.mark.parametrize("change", ["manual", "missing_provenance", "no_best", "provisional", "wrong_budget",
                                     "unspent", "active_trial", "unchecked_reload", "wrong_trial", "changed_config"])
def test_cv_rejects_bypassed_or_modified_tuning_selection(tmp_path, change):
    path = _selection_artifacts(tmp_path)
    selected, best = read_json(path), read_json(path.parent / "best.json")
    if change == "manual":
        selected["source"] = "manual"
    elif change == "missing_provenance":
        selected.pop("compatibility")
    elif change == "no_best":
        (path.parent / "best.json").unlink()
    elif change == "provisional":
        best["selection"] = "provisional"
    elif change == "wrong_budget":
        best["budget"] = 3
    elif change == "unspent":
        best["counts"]["allocated"] = 99
    elif change == "active_trial":
        best["counts"]["RUNNING"] = 1
    elif change == "unchecked_reload":
        best["reload_verified"] = False
    elif change == "wrong_trial":
        selected["trial"] = 1
    elif change == "changed_config":
        selected["config"]["lr"] = 0.3
    write_json(path, selected)
    if change != "no_best":
        write_json(path.parent / "best.json", best)
    with pytest.raises(StageError):
        cross_validate.load_tuned_selection(path, "mixedsbm", 100)


def test_float_search_step_is_used_in_optuna_distribution(tmp_path):
    path = tmp_path / "space.yaml"
    path.write_text(yaml.safe_dump({"model": "mixedsbm", "kind": "smoke", "fixed": {},
                                   "params": {"lr": {"type": "float", "low": 0.1, "high": 0.5, "step": 0.1}}}))
    space = tune.load_search_space(path, "mixedsbm", "smoke")
    study = optuna.create_study(sampler=optuna.samplers.RandomSampler(seed=5))
    for _ in range(5):
        trial = study.ask()
        cfg = tune.suggest(trial, space)
        assert trial.distributions["lr"].step == 0.1
        assert cfg["lr"] * 10 == pytest.approx(round(cfg["lr"] * 10))
        study.tell(trial, cfg["lr"])


@pytest.mark.parametrize("spec", [
    {"type": "float", "low": 0.1, "high": 0.5, "step": 0},
    {"type": "float", "low": 0.1, "high": 0.5, "step": 0.1, "log": True},
    {"type": "int", "low": 1, "high": 9, "step": 1.5},
    {"type": "int", "low": 1, "high": 9, "step": 2, "log": True},
])
def test_invalid_search_steps_are_rejected_before_any_trial_is_allocated(tmp_path, spec):
    path = tmp_path / "space.yaml"
    path.write_text(yaml.safe_dump({"model": "mixedsbm", "kind": "smoke", "params": {"lr": spec}}))
    with pytest.raises(StageError, match="step"):
        tune.load_search_space(path, "mixedsbm", "smoke")
