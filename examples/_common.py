"""
Shared plumbing of the example scripts. It is NOT part of the benchmark protocol.

Every example does the same five things, each through the canonical building block:

    1. load a dataset                       sbtab.data.registry.load_dataset
    2. train-only common preprocessing      sbtab.data.preprocessing.CommonPreprocessor
    3. fit a registered adapter and sample  sbtab.solvers.registry + the adapter contract
    4. score against held-out rows          sbtab.evaluation (the single metric package)
    5. show samples in raw units / labels   CommonPreprocessor.inverse_transform

No metric formula is defined here or anywhere else under ``examples/``: ``report``
only prints fields of the dictionaries returned by ``sbtab.evaluation``.

The examples use ONE illustrative train / held-out split and ONE hand-written
config. A number that is meant to be compared or reported comes from the staged
pipeline instead (see ``pipeline_commands``): validated immutable splits, a
persistent tuning study, five fresh fold fits and metrics from saved artifacts.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, Optional

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:          # `python examples/<script>.py` works without installing sbtab
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import pandas as pd
import yaml
from sklearn.model_selection import train_test_split

from sbtab.data.preprocessing import CommonPreprocessor, UnseenCategoryError
from sbtab.data.registry import load_dataset
from sbtab.evaluation import (MetricConfig, MetricContext, association_metrics, check_validity, conditional_metrics,
                              marginal_metrics, mmd_metrics, tuning_objective)
from sbtab.experiments.experiment_common import seed_everything
from sbtab.solvers.registry import get_adapter_class, get_entry, missing_requirements

DATASET_CONFIGS = REPO_ROOT / "configs" / "datasets"
SMOKE_SPACES = REPO_ROOT / "configs" / "search_spaces" / "smoke"
QUICK_MAX_ROWS = 600        # --quick: cap on the training rows AND on the held-out rows

QUICK_BANNER = ("--quick: tiny bounded config from configs/search_spaces/smoke/ on <= %d training rows.\n"
                "         It only shows that the code path runs; the numbers below say NOTHING about model quality."
                % QUICK_MAX_ROWS)


# --------------------------------------------------------------------------- configs
def smoke_config(model_id: str, **overrides: Any) -> Dict[str, Any]:
    """
    The bounded smoke search space of ``model_id`` collapsed to one config: its
    ``fixed`` block plus ONE value per ``params`` entry (range midpoint, geometric
    for log-scaled ranges; first choice of a categorical). Adapter configs are
    strict, so every key here is one the adapter really consumes.
    """
    space = yaml.safe_load((SMOKE_SPACES / f"{model_id}.yaml").read_text())
    if space.get("model") != model_id or space.get("kind") != "smoke":
        raise ValueError(f"{model_id}.yaml is not the smoke search space of {model_id!r}")
    config = dict(space.get("fixed") or {})
    for name, p in (space.get("params") or {}).items():
        if p["type"] == "categorical":
            config[name] = p["choices"][0]
        elif p.get("log"):
            config[name] = float(np.sqrt(float(p["low"]) * float(p["high"])))
        else:
            mid = (p["low"] + p["high"]) / 2
            config[name] = int(mid) if p["type"] == "int" else float(mid)
    config.update(overrides)
    return config


def pipeline_commands(model_id: str, dataset: str, root: str = "artifacts/sbtab_8515_hpo100_cv5_v2") -> str:
    """The canonical staged pipeline for a real run of ``model_id`` on ``dataset`` (copy-pasteable)."""
    data, run = f"{root}/{dataset}", f"{root}/{dataset}/{model_id}/<run-id>"
    pad = " \\\n      "
    return "\n".join([
        f"  python -m sbtab.experiments.prepare_splits --dataset {dataset} --output-root {root}",
        f"  python -m sbtab.experiments.tune --dataset {dataset} --model {model_id}{pad}"
        f"--splits {data}/splits.json --search-space configs/search_spaces/{model_id}.yaml --resume",
        f"  python -m sbtab.experiments.cross_validate --dataset {dataset} --model {model_id}{pad}"
        f"--selected-config {run}/tuning/selected_config.json --splits {data}/splits.json",
        f"  python -m sbtab.experiments.calculate_metrics --cv-run {run}/cv/cv_run_manifest.json",
        f"then, over all finished runs:  python -m sbtab.experiments.aggregate_results --output-root {root}",
    ])


# --------------------------------------------------------------------------- data
def _subsample(index: pd.Index, max_rows: Optional[int], seed: int) -> pd.Index:
    if max_rows is None or len(index) <= max_rows:
        return index
    keep = np.random.default_rng(seed).choice(len(index), size=int(max_rows), replace=False)
    return index[np.sort(keep)]


def illustrative_split(frame: pd.DataFrame, schema, test_size: float = 0.2, seed: int = 0,
                       max_rows: Optional[int] = None):
    """
    ILLUSTRATIVE split, NOT the benchmark protocol: one scikit-learn hold-out
    (stratified on a classification target), optionally subsampled for --quick.
    The protocol's splits (85/15 tuning split, 5 CV folds inside the 85 % pool,
    explicit support validation, immutable row-id manifests) are produced by
    ``python -m sbtab.experiments.prepare_splits``.
    """
    stratify = frame[schema.target] if schema.task == "classification" else None
    train_ids, held_ids = train_test_split(frame.index.to_numpy(), test_size=test_size, random_state=seed,
                                           shuffle=True, stratify=stratify)
    train_ids = _subsample(pd.Index(np.sort(train_ids)), max_rows, seed)
    held_ids = _subsample(pd.Index(np.sort(held_ids)), max_rows, seed + 1)
    return frame.loc[train_ids], frame.loc[held_ids]


# --------------------------------------------------------------------------- evaluation (canonical functions only)
def evaluate(train: pd.DataFrame, held_out: pd.DataFrame, synth: pd.DataFrame, schema) -> Dict[str, Any]:
    """Everything a metric learns is learned from the TRAINING rows; synth is compared with the held-out rows."""
    ctx = MetricContext.fit(train, schema, MetricConfig(), train_row_ids=train.index.to_numpy())
    marginal, per_feature = marginal_metrics(ctx, held_out, synth)
    association, _ = association_metrics(ctx, held_out, synth)
    conditional, _ = conditional_metrics(ctx, held_out, synth)
    return {
        "validity": check_validity(synth, schema, ctx),
        "objective": tuning_objective(held_out, synth, schema),
        "marginal": marginal, "per_feature": per_feature,
        "association": association, "conditional": conditional,
        "mmd": mmd_metrics(ctx, held_out, synth, real_row_ids=held_out.index.to_numpy()),
    }


# --------------------------------------------------------------------------- the example
def run_example(model_id: str, dataset: str, config: Dict[str, Any], quick: bool = False, seed: int = 0,
                test_size: float = 0.2, verbose: bool = True) -> Dict[str, Any]:
    """Load -> split -> preprocess (train only) -> fit -> sample len(train) -> evaluate -> report."""
    entry = get_entry(model_id)
    missing = missing_requirements(model_id)
    if missing:
        raise SystemExit(f"{model_id} needs the optional packages {list(missing)}, which are not installed")

    frame, schema, manifest = load_dataset(dataset, config_dir=DATASET_CONFIGS)
    if schema.regime not in entry.regimes:
        raise SystemExit(f"{model_id} does not support the {schema.regime!r} regime of {dataset}")
    train_raw, held_raw = illustrative_split(frame, schema, test_size, seed, QUICK_MAX_ROWS if quick else None)

    pre = CommonPreprocessor(schema).fit(train_raw)         # TRAIN rows only; held-out rows are only transformed
    try:
        train, held_out = pre.transform(train_raw), pre.transform(held_raw)
    except UnseenCategoryError as e:                        # the vocabulary is never expanded by held-out rows
        raise SystemExit(f"the illustrative split left a category out of the training rows ({e}). The staged "
                         "pipeline validates category support explicitly: see sbtab.experiments.prepare_splits") from e

    seed_everything(seed)
    adapter = get_adapter_class(model_id)()
    t0 = time.perf_counter()
    adapter.fit(train, schema, config, seed=seed)           # strict config: an unknown key raises
    t1 = time.perf_counter()
    synth = adapter.sample(len(train), seed=seed + 1)       # common representation, schema column order
    t2 = time.perf_counter()

    result: Dict[str, Any] = {
        "model": model_id, "dataset": dataset, "regime": schema.regime, "quick": bool(quick),
        "n_rows_dataset": int(manifest["n_rows"]), "n_train": int(len(train)), "n_held_out": int(len(held_out)),
        "config": dict(config), "effective_config": dict(adapter.config), "n_updates": adapter.n_updates,
        "fit_seconds": t1 - t0, "sample_seconds": t2 - t1, "describe": adapter.describe(),
        "schema": schema, "adapter": adapter, "preprocessor": pre, "train_raw": train_raw, "held_out_raw": held_raw,
        "synthetic": synth, "synthetic_raw": None,
    }
    result.update(evaluate(train, held_out, synth, schema))
    if result["validity"]["status"] == "ok":                # raw units / labels exist only for a valid table
        result["synthetic_raw"] = pre.inverse_transform(synth)
    if verbose:
        report(result)
    return result


# --------------------------------------------------------------------------- printing
def _num(v: Any) -> str:
    return "n/a" if v is None else f"{v:.5g}"


def _setting(v: Any) -> str:
    return f"{v:.4g}" if isinstance(v, float) else str(v)


def report(r: Dict[str, Any]) -> None:
    line = "-" * 78
    print(line)
    print(f"{r['model']} on {r['dataset']} ({r['regime']} regime): fitted on {r['n_train']} rows, "
          f"scored against {r['n_held_out']} held-out rows (illustrative split)")
    if r["quick"]:
        print(QUICK_BANNER)
    print(line)
    print("config            " + ", ".join(f"{k}={_setting(v)}" for k, v in r["config"].items()))
    updates = r["n_updates"] if r["n_updates"] is not None else "n/a (the solver reports no optimiser-update count)"
    print(f"generator         fit={r['fit_seconds']:.1f}s  sample={r['sample_seconds']:.1f}s  n_updates={updates}")
    v = r["validity"]
    print(f"validity          status={v['status']}  invalid_rows={v['n_invalid_rows']}/{v['n_rows']}  reasons={v['reasons']}")
    o = r["objective"]
    print(f"tuning objective  status={o['status']}  objective={_num(o['objective'])}  "
          f"(mean_wd={_num(o['mean_wd'])}, mean_js={_num(o['mean_js'])})")
    for name, g in r["marginal"]["groups"].items():
        if name == "finite_combined" or g["status"] == "not_applicable":
            continue
        scores = "  ".join(f"mean_{m}={_num(g[f'mean_{m}'])}" for m in ("wd", "kl", "js") if f"mean_{m}" in g)
        print(f"marginal          {name} ({g['n_columns']} columns)  status={g['status']}  {scores}")
    for name, b in r["association"]["blocks"].items():
        if b["status"] != "not_applicable":
            print(f"association       {name}  status={b['status']}  rmse={_num(b.get('offdiag_rmse', b.get('rmse')))}")
    c = r["conditional"]
    sec = c["secondary_across_conditioners"] or {}
    print(f"conditional       status={c['status']}  conditioners={c['n_conditioners']}  "
          f"wd_weighted_mean={_num(sec.get('wd_weighted_mean'))}  js_weighted_mean={_num(sec.get('js_weighted_mean'))}")
    for name, k in r["mmd"]["kernels"].items():
        if k["status"] == "not_applicable":
            continue
        floor = (k.get("floor") or {}).get("real_real_mean")
        print(f"MMD^2 ({name})".ljust(18) + f"status={k['status']}  signed unbiased={_num(k.get('mmd2_unbiased_mean'))}  "
              f"real-real floor={_num(floor)}")
    print(line)
    if r["synthetic_raw"] is not None:
        print("synthetic rows, inverse-transformed to raw units / labels:")
        print(r["synthetic_raw"].head().to_string())
    else:
        print("the generated table is INVALID; head of the common representation:")
        print(r["synthetic"].head().to_string())
    print(line)
    print("For a result you can report, run the staged pipeline instead of this script:")
    print(pipeline_commands(r["model"], r["dataset"]))


def cli(main: Callable[..., Any], doc: Optional[str]) -> None:
    ap = argparse.ArgumentParser(description=(doc or "").replace("%", "%%"),
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--quick", action="store_true",
                    help="tiny bounded smoke config on <= %d training rows; says NOTHING about model quality" % QUICK_MAX_ROWS)
    main(quick=ap.parse_args().quick)
