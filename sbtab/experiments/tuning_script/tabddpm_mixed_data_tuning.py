
from __future__ import annotations


import argparse
import json
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import optuna
import pandas as pd
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
from scipy.stats import wasserstein_distance
from ucimlrepo import fetch_ucirepo

from sbtab.baselines.tabddpm.model import TabDDPMConfig, TabDDPMWrapper
from sbtab.data.datamodule import TabularDataModule
from sbtab.data.schema import TabularSchema
from sbtab.data.splits import SplitConfigHoldout
from sbtab.transforms.pipeline import TransformPipeline


# ---------------------------------------------------------------------
# UCI dataset loading
# ---------------------------------------------------------------------

def _slugify(s: str) -> str:
    s = s.strip().lower()
    s = re.sub(r"[^a-z0-9]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s or "dataset"


def load_uci_dataset(dataset_id: int) -> tuple[pd.DataFrame, str]:
    ds = fetch_ucirepo(id=int(dataset_id))

    name = getattr(getattr(ds, "metadata", None), "name", None)
    if name is None:
        try:
            name = ds.metadata.get("name", f"uci_{dataset_id}")
        except Exception:
            name = f"uci_{dataset_id}"

    X = ds.data.features.copy()
    y = ds.data.targets

    if y is None:
        df = X.copy()
    else:
        if isinstance(y, pd.Series):
            y_df = y.to_frame()
        elif isinstance(y, pd.DataFrame):
            y_df = y.copy()
        else:
            y_df = pd.DataFrame(y)

        new_cols = []
        for col in y_df.columns:
            if col in X.columns:
                new_cols.append(f"target__{col}")
            else:
                new_cols.append(str(col))
        y_df.columns = new_cols

        df = pd.concat([X, y_df], axis=1)

    return df, _slugify(str(name))


# ---------------------------------------------------------------------
# Pipeline selection
# ---------------------------------------------------------------------

def build_transforms(schema: TabularSchema, *, missing_strategy: str, cat_encoding: str) -> TransformPipeline:
    if schema.has_categorical:
        if missing_strategy == "drop":
            # if your repo exposes a mixed drop+integer pipeline, use it; otherwise fail loudly
            if hasattr(TransformPipeline, "default_drop_scale_integer_encode"):
                return TransformPipeline.default_drop_scale_integer_encode()
            raise ValueError(
                "missing_strategy='drop' is not supported for mixed data with the current "
                "TransformPipeline API unless default_drop_scale_integer_encode() exists."
            )

        if cat_encoding == "onehot":
            return TransformPipeline.default_impute_scale_encode()
        if cat_encoding == "integer":
            return TransformPipeline.default_impute_scale_integer_encode()
        raise ValueError(f"Unknown cat_encoding={cat_encoding!r}")

    if missing_strategy == "drop":
        return TransformPipeline.default_dropna_and_scale()
    return TransformPipeline.default_impute_and_scale()


# ---------------------------------------------------------------------
# Composite metric helpers
# ---------------------------------------------------------------------

def _js_divergence_from_counts(p: np.ndarray, q: np.ndarray, eps: float = 1e-12) -> float:
    """
    Jensen–Shannon divergence between two discrete distributions.
    """
    p = p.astype(np.float64)
    q = q.astype(np.float64)

    p = p + eps
    q = q + eps
    p = p / p.sum()
    q = q / q.sum()

    m = 0.5 * (p + q)

    kl_pm = np.sum(p * (np.log(p) - np.log(m)))
    kl_qm = np.sum(q * (np.log(q) - np.log(m)))

    return float(0.5 * (kl_pm + kl_qm))


def _empirical_probs_from_values(values: pd.Series, universe: List[Any]) -> np.ndarray:
    counts = values.value_counts(normalize=True, dropna=False)
    return np.array([counts.get(v, 0.0) for v in universe], dtype=np.float64)


def _js_for_categorical_like(real: pd.Series, synth: pd.Series) -> float:
    universe = list(pd.Index(real.astype("object")).append(pd.Index(synth.astype("object"))).drop_duplicates())
    if len(universe) == 0:
        return 0.0
    p = _empirical_probs_from_values(real.astype("object"), universe)
    q = _empirical_probs_from_values(synth.astype("object"), universe)
    return _js_divergence_from_counts(p, q)


def _js_for_discrete_numeric(real: pd.Series, synth: pd.Series) -> float:
    """
    JS divergence for discrete numeric variables.
    We round before counting to avoid tiny inverse-transform floating noise.
    """
    r = pd.to_numeric(real, errors="coerce")
    s = pd.to_numeric(synth, errors="coerce")

    r = pd.Series(np.rint(r.to_numpy(dtype=np.float64)).astype(np.int64), index=real.index)
    s = pd.Series(np.rint(s.to_numpy(dtype=np.float64)).astype(np.int64), index=synth.index)

    universe = sorted(set(r.tolist()) | set(s.tolist()))
    if len(universe) == 0:
        return 0.0

    p = _empirical_probs_from_values(r, universe)
    q = _empirical_probs_from_values(s, universe)
    return _js_divergence_from_counts(p, q)


def compute_composite_metric(
    real_raw: pd.DataFrame,
    synth_raw: pd.DataFrame,
    *,
    schema: TabularSchema,
) -> Tuple[float, float, float, Dict[str, Dict[str, float]]]:
    """
    Composite metric:
        score = mean_WD(continuous_cols) + mean_JS(categorical_cols + discrete_cols)

    Returns:
        total_score, wd_num_mean, js_disc_cat_mean, per_feature_scores
    """
    per_feature_scores: Dict[str, Dict[str, float]] = {}

    wd_scores: List[float] = []
    for col in schema.continuous_cols:
        if col in real_raw.columns and col in synth_raw.columns:
            r = pd.to_numeric(real_raw[col], errors="coerce").to_numpy(dtype=np.float32)
            s = pd.to_numeric(synth_raw[col], errors="coerce").to_numpy(dtype=np.float32)
            score = float(wasserstein_distance(r, s))
            wd_scores.append(score)
            per_feature_scores[col] = {"metric": "wd", "value": score}

    js_scores: List[float] = []

    for col in schema.discrete_cols:
        if col in real_raw.columns and col in synth_raw.columns:
            score = _js_for_discrete_numeric(real_raw[col], synth_raw[col])
            js_scores.append(score)
            per_feature_scores[col] = {"metric": "js_discrete", "value": score}

    for col in schema.categorical_cols:
        if col in real_raw.columns and col in synth_raw.columns:
            score = _js_for_categorical_like(real_raw[col], synth_raw[col])
            js_scores.append(score)
            per_feature_scores[col] = {"metric": "js_categorical", "value": score}

    wd_num_mean = float(np.mean(wd_scores)) if wd_scores else 0.0
    js_disc_cat_mean = float(np.mean(js_scores)) if js_scores else 0.0
    total_score = wd_num_mean + js_disc_cat_mean

    return total_score, wd_num_mean, js_disc_cat_mean, per_feature_scores


# ---------------------------------------------------------------------
# Objective
# ---------------------------------------------------------------------

def _suggest_mlp_layers(trial: optuna.Trial) -> List[int]:
    def suggest_dim(name: str) -> int:
        t = trial.suggest_int(name, d_min, d_max)
        return 2 ** t

    min_n_layers, max_n_layers, d_min, d_max = 1, 4, 7, 10
    n_layers = 2 * trial.suggest_int("n_layers", min_n_layers, max_n_layers)

    d_first = [suggest_dim("d_first")] if n_layers else []
    d_middle = [suggest_dim("d_middle")] * (n_layers - 2) if n_layers > 2 else []
    d_last = [suggest_dim("d_last")] if n_layers > 1 else []

    return d_first + d_middle + d_last


def make_objective_for_dataset(
    train_proc: pd.DataFrame,
    val_proc: pd.DataFrame,
    *,
    seed: int,
    device: str,
    schema: TabularSchema,
    fitted_transforms: Any,
):
    def objective(trial: optuna.Trial) -> float:
        batch_size = trial.suggest_categorical("batch_size", [256, 500, 1000])
        steps = trial.suggest_categorical("steps", [5000, 20000, 30000])
        num_timesteps = trial.suggest_categorical("num_timesteps", [100, 1000])
        lr = trial.suggest_float("lr", 1e-5, 3e-3, log=True)

        cfg = TabDDPMConfig(
            steps=int(steps),
            num_timesteps=int(num_timesteps),
            batch_size=int(batch_size),
            lr=float(lr),
            weight_decay=0.0,
            d_layers=_suggest_mlp_layers(trial),
            dropout=0.0,
            scheduler="cosine",
            device=device,
            seed=seed,
        )

        try:
            model = TabDDPMWrapper(cfg=cfg)
            model.fit(train_proc, schema=schema, transforms=fitted_transforms)

            synth_proc = model.sample(n=len(val_proc), seed=seed + 123)

            # evaluate in the raw/inverse-transformed space for correct per-type metrics
            real_raw = fitted_transforms.inverse_transform(val_proc)
            synth_raw = fitted_transforms.inverse_transform(synth_proc)

            total_score, wd_num_mean, js_disc_cat_mean, _ = compute_composite_metric(
                real_raw,
                synth_raw,
                schema=schema,
            )

            trial.set_user_attr("wd_num_mean", wd_num_mean)
            trial.set_user_attr("js_disc_cat_mean", js_disc_cat_mean)

            trial.report(total_score, step=0)
            if trial.should_prune():
                raise optuna.TrialPruned()

            return total_score

        except optuna.TrialPruned:
            raise
        except Exception as e:
            trial.set_user_attr("exception", repr(e))
            return float("inf")

    return objective


# ---------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------

def export_trials_csv(study: optuna.Study, out_csv: Path) -> None:
    rows = []
    for tr in study.trials:
        row = {
            "trial_number": tr.number,
            "state": str(tr.state),
            "value": tr.value,
            "wd_num_mean": tr.user_attrs.get("wd_num_mean"),
            "js_disc_cat_mean": tr.user_attrs.get("js_disc_cat_mean"),
            **tr.params,
        }
        if "exception" in tr.user_attrs:
            row["exception"] = tr.user_attrs["exception"]
        rows.append(row)
    pd.DataFrame(rows).to_csv(out_csv, index=False)


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--dataset-ids",
        type=str,
        default="1,2,27,560",
        help="Comma-separated UCI dataset ids, e.g. '2,15,109'",
    )
    ap.add_argument("--test-size", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", type=str, default="cuda")

    ap.add_argument("--n-trials", type=int, default=50)
    ap.add_argument("--timeout", type=int, default=0, help="Seconds per dataset (0 => no timeout)")
    ap.add_argument("--storage", type=str, default="sqlite:///tabddpm_uci_optuna.db")
    ap.add_argument("--study-prefix", type=str, default="tabddpm_uci")

    ap.add_argument("--outdir", type=str, default="tabddpm_uci_optuna_results")
    ap.add_argument("--export-trials", action="store_true")
    ap.add_argument(
        "--missing-strategy",
        type=str,
        default="drop",
        choices=["impute", "drop"],
        help="Which TransformPipeline variant to use.",
    )
    ap.add_argument(
        "--cat-encoding",
        type=str,
        default="integer",
        choices=["onehot", "integer"],
        help="Categorical representation used by the repository transform pipeline.",
    )

    args = ap.parse_args()

    dataset_ids = [int(x.strip()) for x in args.dataset_ids.split(",") if x.strip()]

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    sampler = TPESampler(seed=args.seed)
    pruner = MedianPruner(n_startup_trials=10, n_warmup_steps=0)

    summary_rows = []

    for dataset_id in dataset_ids:
        df, ds_slug = load_uci_dataset(dataset_id)

        print("\n" + "=" * 90)
        print(f"UCI DATASET ID={dataset_id}  NAME={ds_slug}")
        print("=" * 90)

        if df.shape[1] < 2:
            raise ValueError(f"Dataset id={dataset_id} has <2 columns; cannot tune TabDDPM.")

        schema = TabularSchema.infer_from_dataframe(df)
        transforms = build_transforms(
            schema,
            missing_strategy=args.missing_strategy,
            cat_encoding=args.cat_encoding,
        )

        dm = TabularDataModule(
            df=df,
            schema=schema,
            transforms=transforms,
            reset_index=True,
        )
        dm.prepare_holdout(
            SplitConfigHoldout(
                val_size=args.test_size,
                shuffle=True,
                random_state=args.seed,
            )
        )
        holdout = dm.get_holdout()

        train_proc = holdout.train
        val_proc = holdout.val
        fitted_transforms = holdout.transforms

        print(
            "Schema: "
            f"continuous={len(schema.continuous_cols)}, "
            f"discrete={len(schema.discrete_cols)}, "
            f"categorical={len(schema.categorical_cols)}"
        )
        print(
            "Schema with col names: "
            f"continuous={schema.continuous_cols}, "
            f"discrete={schema.discrete_cols}, "
            f"categorical={schema.categorical_cols}"
        )
        print(f"Pipeline: {transforms.__class__.__name__}")
        print(f"Train size (processed): {len(train_proc)}")
        print(f"Val size   (processed): {len(val_proc)}")
        print(f"Processed columns: {len(train_proc.columns)}")
        print(f"Categorical encoding: {args.cat_encoding}")

        study_name = f"{args.study_prefix}__uci_{dataset_id}__{ds_slug}"
        study = optuna.create_study(
            study_name=study_name,
            storage=args.storage if args.storage != ":memory:" else None,
            load_if_exists=True,
            direction="minimize",
            sampler=sampler,
            pruner=pruner,
        )

        objective = make_objective_for_dataset(
            train_proc=train_proc,
            val_proc=val_proc,
            seed=args.seed,
            device=args.device,
            schema=schema,
            fitted_transforms=fitted_transforms,
        )

        t0 = time.time()
        study.optimize(
            objective,
            n_trials=int(args.n_trials),
            timeout=(args.timeout if args.timeout > 0 else None),
            gc_after_trial=True,
            show_progress_bar=True,
        )
        elapsed = time.time() - t0

        best = study.best_trial
        print("\n--- BEST RESULT ---")
        print(f"Dataset id={dataset_id} ({ds_slug})")
        print(f"Best composite score: {best.value}")
        print(f"  wd_num_mean     : {best.user_attrs.get('wd_num_mean')}")
        print(f"  js_disc_cat_mean: {best.user_attrs.get('js_disc_cat_mean')}")
        print("Best params:")
        for k, v in best.params.items():
            print(f"  {k}: {v}")
        print(f"Trials: {len(study.trials)}  Elapsed: {elapsed:.1f}s")

        best_json = {
            "dataset_id": dataset_id,
            "dataset_name": ds_slug,
            "best_composite_score": float(best.value),
            "best_wd_num_mean": best.user_attrs.get("wd_num_mean"),
            "best_js_disc_cat_mean": best.user_attrs.get("js_disc_cat_mean"),
            "best_trial": int(best.number),
            "n_trials": int(len(study.trials)),
            "elapsed_sec": float(elapsed),
            "best_params": dict(best.params),
            "missing_strategy": args.missing_strategy,
            "cat_encoding": args.cat_encoding,
        }
        (outdir / f"uci_{dataset_id}_{ds_slug}_best.json").write_text(
            json.dumps(best_json, indent=2), encoding="utf-8"
        )

        if args.export_trials:
            export_trials_csv(study, outdir / f"uci_{dataset_id}_{ds_slug}_trials.csv")

        summary_rows.append(
            {
                "dataset_id": dataset_id,
                "dataset_name": ds_slug,
                "best_composite_score": float(best.value),
                "best_wd_num_mean": best.user_attrs.get("wd_num_mean"),
                "best_js_disc_cat_mean": best.user_attrs.get("js_disc_cat_mean"),
                "best_trial": int(best.number),
                "n_trials": int(len(study.trials)),
                "elapsed_sec": float(elapsed),
                "missing_strategy": args.missing_strategy,
                "cat_encoding": args.cat_encoding,
                **best.params,
            }
        )

    summary_df = pd.DataFrame(summary_rows).sort_values("best_composite_score", ascending=True)
    out_csv = outdir / "tabddpm_uci_optuna_summary.csv"
    summary_df.to_csv(out_csv, index=False)

    print("\n" + "=" * 90)
    print("FINAL SUMMARY (sorted by best_composite_score)")
    print("=" * 90)
    with pd.option_context("display.max_columns", 200, "display.width", 200):
        print(summary_df)
    print(f"\nSaved summary CSV to: {out_csv}")
    print(f"Saved per-dataset best JSON files to: {outdir}")


if __name__ == "__main__":
    main()
