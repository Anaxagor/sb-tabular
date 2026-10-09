"""TabbyFlow inference checkpoints and the real canonical experiment stages."""
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from sbtab.adapters.base import AdapterConfigError
from sbtab.adapters.tabbyflow import TabbyFlowAdapter
from sbtab.baselines.tabbyflow import TabbyFlowConfig, TabbyFlowSynthesizer
from sbtab.data.dataset_schema import ColumnSpec, DatasetSchema
from sbtab.data.loading import frame_fingerprint
from sbtab.data.preprocessing import CommonPreprocessor
from sbtab.data.schema import TabularSchema
from sbtab.experiments import calculate_metrics, cross_validate, prepare_splits, tune
from sbtab.experiments.experiment_common import load_protocol, read_json


TINY = dict(max_train_steps=4, batch_size=16, n_frequencies=4, ode_steps=3, sample_batch_size=11, device="cpu")


def _dataset(regime, n=120):
    rng = np.random.default_rng(251)
    data, specs = {}, []
    if regime != "discrete":
        data["x"] = np.linspace(-2, 2, n)
        specs.append(ColumnSpec("x", "continuous"))
    if regime != "continuous":
        data.update(d=np.tile([0., 2., 7.], n // 3), c=np.tile(["a", "b", "c"], n // 3))
        specs += [ColumnSpec("d", "discrete"), ColumnSpec("c", "categorical")]
    regression = regime == "continuous"
    data["y"] = 10 + rng.normal(size=n) if regression else np.tile(["yes", "no"], n // 2)
    specs.append(ColumnSpec("y", "continuous" if regression else "categorical", role="target"))
    schema = DatasetSchema("tabby_toy", tuple(specs), task="regression" if regression else "classification")
    return pd.DataFrame(data, index=pd.RangeIndex(n, name="row_id")), schema


@pytest.mark.parametrize("regime", ["continuous", "discrete", "mixed"])
def test_adapter_checkpoint_restores_train_only_representation_without_refitting(tmp_path, monkeypatch, regime):
    raw, schema = _dataset(regime)
    raw = raw.iloc[:90]
    common = CommonPreprocessor(schema).fit(raw).transform(raw)
    torch.manual_seed(781)
    rng_before = torch.get_rng_state().clone()
    adapter = TabbyFlowAdapter().fit(common, schema, TINY, seed=25)
    assert torch.equal(torch.get_rng_state(), rng_before)
    assert adapter.n_updates == 4 and adapter.n_fit_rows == 90
    assert adapter._encoded is None
    if schema.continuous or schema.discrete:
        cols = adapter.model.numeric_cols_
        np.testing.assert_array_equal(adapter.model.quantile.quantiles_[-1], common[cols].max().to_numpy())
    generated = adapter.sample(27, seed=17)
    assert torch.equal(torch.get_rng_state(), rng_before)
    pd.testing.assert_frame_equal(generated, adapter.sample(27, seed=17))
    assert np.isfinite(generated.to_numpy()).all()
    for c in schema.finite_support:
        assert set(generated[c]) <= set(common[c])
    assert list(generated) == schema.column_order and len(generated) == 27
    assert generated.index.name == "synthetic_id"

    checkpoint = adapter.save_checkpoint(tmp_path / "checkpoint")
    state = torch.load(checkpoint / "tabbyflow.pt", weights_only=True)
    assert "training_rows" not in state and "optimizer" not in state
    assert state["config"] == asdict(adapter.model.cfg)

    def no_fit(*a, **kw):
        raise AssertionError("checkpoint loading cannot refit preprocessing or the network")
    monkeypatch.setattr(TabbyFlowSynthesizer, "fit", no_fit)
    monkeypatch.setattr(TabbyFlowSynthesizer, "_fit_preprocessor", no_fit)
    from sklearn.preprocessing import QuantileTransformer
    monkeypatch.setattr(QuantileTransformer, "fit", no_fit)
    loaded = TabbyFlowAdapter.load_checkpoint(checkpoint)
    assert loaded.fit_row_hash == adapter.fit_row_hash and loaded.n_updates == adapter.n_updates
    pd.testing.assert_frame_equal(loaded.sample(27, seed=17), generated)
    assert not loaded.model.net.training


def test_standalone_checkpoint_keeps_typed_categories_and_identifiers(tmp_path):
    frame = pd.DataFrame({"c": pd.Categorical([1, "1"] * 6, categories=["1", 1], ordered=True),
                          "id": np.arange(12)})
    model = TabbyFlowSynthesizer(TabbyFlowConfig(**TINY)).fit(
        frame, schema=TabularSchema([], [], ["c"], id_col="id"), task_type="classification")
    expected = model.sample(13, seed=59)
    model.save_checkpoint(tmp_path / "model.pt")
    loaded = TabbyFlowSynthesizer.load_checkpoint(tmp_path / "model.pt", device="cpu")
    assert loaded.quantile is None and loaded.d_cont_ == 0
    pd.testing.assert_frame_equal(loaded.sample(13, seed=59), expected)
    assert loaded.raw_dtypes_["c"] == frame["c"].dtype
    assert set(loaded.cat_decode_maps_["c"].values()) == {1, "1"}
    assert not set(expected["id"]) & set(frame["id"])


@pytest.mark.parametrize("key,value", [("lr", 0), ("lr", float("nan")), ("weight_decay", -1),
                                       ("scheduler_factor", 1), ("ode_steps", 0), ("batch_size", 1.5),
                                       ("cond_vel", "unknown"), ("ode_solver", "unknown")])
def test_invalid_configs_are_rejected_before_training(key, value):
    with pytest.raises(ValueError):
        TabbyFlowAdapter.resolve_config({key: value})


def test_config_keys_match_standalone_api_and_cuda_never_silently_falls_back(monkeypatch):
    assert set(TabbyFlowAdapter.DEFAULTS) == set(asdict(TabbyFlowConfig())) - {"seed"}
    with pytest.raises(AdapterConfigError, match="not consumed"):
        TabbyFlowAdapter.resolve_config({"n_epochs": 10})
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="CUDA.*unavailable"):
        TabbyFlowSynthesizer(TabbyFlowConfig(device="cuda"))


def test_numpy_config_scalars_produce_weights_only_loadable_checkpoint(tmp_path):
    values = asdict(TabbyFlowConfig(**TINY))
    config = TabbyFlowConfig(**{k: np.int64(v) if type(v) is int else np.float64(v) if type(v) is float
                               else np.str_(v) for k, v in values.items()})
    assert all(type(v) in {int, float, str} for v in asdict(config).values())
    frame = pd.DataFrame({"c": pd.Series([np.int64(1), np.str_("1")] * 6, dtype=object)})
    model = TabbyFlowSynthesizer(config).fit(frame, schema=TabularSchema([], [], ["c"]), task_type="classification")
    expected = model.sample(15, seed=29)
    # Configs are mutable for standalone inference; save normalizes that case too.
    model.cfg.sample_batch_size = np.int64(model.cfg.sample_batch_size)
    path = model.save_checkpoint(tmp_path / "numpy.pt")
    state = torch.load(path, weights_only=True)
    assert all(type(v) in {int, float, str} for v in state["config"].values())
    pd.testing.assert_frame_equal(TabbyFlowSynthesizer.load_checkpoint(path).sample(15, seed=29), expected)


@pytest.mark.parametrize("unused", [False, True])
def test_timestamp_category_metadata_is_rejected_before_transform_fit(monkeypatch, unused):
    from sklearn.preprocessing import QuantileTransformer
    timestamp = pd.Timestamp("2026-01-01")
    labels = pd.Categorical(["valid"] * 12, categories=["valid", timestamp]) if unused else \
        pd.Categorical([timestamp, pd.Timestamp("2026-01-02")] * 6)
    frame = pd.DataFrame({"x": np.linspace(-1, 1, 12), "category": labels})
    def no_fit(*a, **kw):
        raise AssertionError("unsupported category metadata must be rejected before transform fitting")
    monkeypatch.setattr(QuantileTransformer, "fit_transform", no_fit)
    model = TabbyFlowSynthesizer(TabbyFlowConfig(**TINY))
    with pytest.raises(TypeError, match="categorical column 'category'.*Timestamp"):
        model.fit(frame, schema=TabularSchema(["x"], [], ["category"]), task_type="classification")
    assert model.net is None and not model._fitted


def test_failed_refit_and_nonfinite_generation_are_not_hidden(tmp_path, monkeypatch):
    from sbtab.baselines.tabbyflow import model as module
    frame = pd.DataFrame({"x": np.linspace(-1, 1, 12)})
    model = TabbyFlowSynthesizer(TabbyFlowConfig(**TINY)).fit(
        frame, schema=TabularSchema(["x"], [], []), task_type="regression")
    monkeypatch.setattr(module, "integrate_tabbyflow_fixed_step", lambda field, x, **kw: torch.full_like(x, float("nan")))
    with pytest.raises(FloatingPointError, match="latent"):
        model.sample(5, seed=1)
    with pytest.raises(ValueError, match="two rows"):
        model.fit(frame.iloc[:1], schema=TabularSchema(["x"], [], []), task_type="regression")
    with pytest.raises(RuntimeError, match="fit"):
        model.sample(5, seed=1)
    with pytest.raises(RuntimeError, match="fit"):
        model.save_checkpoint(tmp_path / "invalid.pt")


@pytest.mark.parametrize("regime", ["continuous", "discrete", "mixed"])
def test_real_tuning_cv_metrics_smoke_for_each_regime(tmp_path, monkeypatch, regime):
    frame, schema = _dataset(regime)
    source = {"fingerprint": frame_fingerprint(frame), "missing_counts": {}, "dataset": schema.name,
              "source": {}, "n_rows": len(frame), "n_columns": len(frame.columns), "schema_hash": schema.hash(),
              "regime": schema.regime, "task": schema.task, "target": schema.target,
              "dropped_columns": [], "value_maps": {}, "missing_policy": "reject", "row_filtering": "none"}
    monkeypatch.setattr(prepare_splits, "load_dataset", lambda name, config_dir=None: (frame, schema, source))
    protocol = load_protocol(smoke=True)
    root = tmp_path / protocol.id
    prepared = prepare_splits.run(schema.name, protocol, root)
    assert prepared["split_status"] == "ok"
    splits_path = root / schema.name / "splits.json"
    splits = read_json(splits_path)
    result = tune.run(schema.name, "tabbyflow", splits_path, "configs/search_spaces/smoke/tabbyflow.yaml",
                      resume=False, smoke=True)
    assert result["selection"] == "final" and result["counts"]["COMPLETE"] == 3
    run = Path(result["run_dir"])
    for path in (run / "tuning").glob("trial-*/synthetic.parquet"):
        assert len(pd.read_parquet(path)) == splits["n_V"]
    cv = cross_validate.run(schema.name, "tabbyflow", run / "tuning/selected_config.json", splits_path,
                            root, smoke=True, folds=[0])
    assert cv["n_ok"] == 1
    fold = read_json(run / "cv/fold-0/manifest.json")
    assert fold["reload_verified"]["ok"] and fold["n_updates"] == 200
    assert fold["n_generated"] == len(splits["folds"][0]["train_row_ids"])
    assert fold["preprocessor"]["fit_row_hash"] == splits["folds"][0]["train_hash"]
    metrics = calculate_metrics.run(cv["cv_run_manifest"], protocol["metrics_config"], folds=[0])
    report = read_json(Path(metrics["evaluation_dir"]) / "fold-0/metrics.json")
    assert report["status"] == "ok" and report["marginal"]["metric_version"] == "sbtab.metrics/2"
    assert report["utility"]["status"] == "ok"
    resumed = tune.run(schema.name, "tabbyflow", splits_path, "configs/search_spaces/smoke/tabbyflow.yaml",
                       resume=True, smoke=True)
    assert resumed["new_trials"] == 0
