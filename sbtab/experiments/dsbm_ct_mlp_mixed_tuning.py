"""
Optuna tuning: DSBM + CT + MLP + Joint on mixed/categorical tabular data.

Follows the same pattern as light_sb_mixed_tuning.py:
  - Uses datasets_mixed.pkl with df.attrs metadata
    (feature_types, target_variable, task_type)
  - Pipeline: TransformPipeline.default_impute_scale_encode() for mixed,
    default_dropna_and_scale() for continuous-only
  - DSBM is fitted on [encoded_features | numeric_target]
  - Tuning objective (computed in original space after inverse_transform):
      Mixed data:         Mean_WD(cont) + Mean_JS(disc+cat)
      Pure discrete/cat:  Mean_JS(all features)

Usage:
  python -m sbtab.experiments.dsbm_ct_mlp_mixed_tuning \\
      --pickle sbtab/data/datasets/datasets_mixed.pkl
"""
from __future__ import annotations

import argparse
import json
import pickle
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import optuna
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner
from scipy.spatial.distance import jensenshannon
from scipy.stats import wasserstein_distance
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from sbtab.data.schema import TabularSchema
from sbtab.transforms.pipeline import TransformPipeline
from sbtab.solvers.continuous_time.joint_distribution.mlp.imf_dsbm.solver import (
    IMFDSBMConfig,
    IMFDSBMSolver,
)


# ---------------------------------------------------------------------------
# Objective-metric helpers
# ---------------------------------------------------------------------------

def _avg_wd(real: pd.DataFrame, synth: pd.DataFrame, cols: List[str]) -> float:
    if not cols:
        return 0.0
    return float(np.mean([
        wasserstein_distance(real[c].to_numpy(), synth[c].to_numpy()) for c in cols
    ]))


def _js_col(real_s: pd.Series, synth_s: pd.Series, eps: float = 1e-12) -> float:
    all_cats = sorted(set(real_s.astype(str).unique()) | set(synth_s.astype(str).unique()))
    rc = real_s.astype(str).value_counts()
    sc = synth_s.astype(str).value_counts()
    p = np.array([rc.get(c, 0) for c in all_cats], dtype=float) + eps
    q = np.array([sc.get(c, 0) for c in all_cats], dtype=float) + eps
    p /= p.sum()
    q /= q.sum()
    return float(jensenshannon(p, q) ** 2)


def _avg_js(real: pd.DataFrame, synth: pd.DataFrame, cols: List[str]) -> float:
    if not cols:
        return 0.0
    return float(np.mean([_js_col(real[c], synth[c]) for c in cols]))


def tuning_objective_value(
    test_feat_orig: pd.DataFrame,
    synth_feat_orig: pd.DataFrame,
    continuous_cols: List[str],
    disc_cat_cols: List[str],
    feature_cols: List[str],
    is_mixed: bool,
) -> float:
    """Scalar objective to minimize.

    Mixed:         Mean_WD(cont) + Mean_JS(disc+cat)
    Pure discrete: Mean_JS(all features)
    """
    if is_mixed:
        wd = _avg_wd(test_feat_orig, synth_feat_orig, continuous_cols)
        js = _avg_js(test_feat_orig, synth_feat_orig, disc_cat_cols)
        return float(wd + js)
    else:
        return _avg_js(test_feat_orig, synth_feat_orig, feature_cols)


def export_trials_csv(study: optuna.Study, out_csv: Path) -> None:
    rows = []
    for tr in study.trials:
        row = {"trial_number": tr.number, "state": str(tr.state), "value": tr.value, **tr.params}
        if "exception" in tr.user_attrs:
            row["exception"] = tr.user_attrs["exception"]
        rows.append(row)
    pd.DataFrame(rows).to_csv(out_csv, index=False)


# ---------------------------------------------------------------------------
# Objective factory
# ---------------------------------------------------------------------------

def make_objective(
    train_feat_enc: pd.DataFrame,
    test_feat_orig: pd.DataFrame,
    pipe: TransformPipeline,
    train_target: np.ndarray,
    target_col: str,
    continuous_cols: List[str],
    disc_cat_cols: List[str],
    feature_cols: List[str],
    is_mixed: bool,
    seed: int,
    max_train_size: int,
    device: str,
):
    encoded_feat_cols = list(train_feat_enc.columns)

    def objective(trial: optuna.Trial) -> float:
        # --- SB hyperparameters ---
        sigma          = trial.suggest_float("sigma", 0.03, 0.50, log=True)
        num_steps      = trial.suggest_int("num_steps", 50, 1000, step=50)
        eps            = trial.suggest_float("eps", 1e-4, 5e-3, log=True)
        imf_len        = trial.suggest_int("imf_len", 1, 4) * 2 + 1  # 3, 5, 7, 9
        first_coupling = trial.suggest_categorical("first_coupling", ["ref", "ind"])
        noise          = trial.suggest_categorical("noise", [True, False])

        # --- MLP training hyperparameters ---
        inner_iters  = trial.suggest_int("inner_iters", 500, 5000, step=500)
        batch_size   = trial.suggest_categorical("batch_size", [128, 256, 512])
        lr           = trial.suggest_float("lr", 1e-5, 1e-2, log=True)
        weight_decay = trial.suggest_float("weight_decay", 0.0, 1e-3)
        grad_clip    = trial.suggest_float("grad_clip", 0.5, 5.0)

        fb_sequence = tuple("b" if i % 2 == 0 else "f" for i in range(imf_len))

        cfg = IMFDSBMConfig(
            fb_sequence=fb_sequence,
            num_steps=num_steps,
            sigma=sigma,
            eps=eps,
            first_coupling=first_coupling,
            inner_iters=inner_iters,
            batch_size=batch_size,
            lr=lr,
            weight_decay=weight_decay,
            grad_clip=grad_clip,
            noise=noise,
            device=device,
            seed=seed + trial.number,
        )

        try:
            tr_feat = train_feat_enc
            tr_tgt = train_target
            if max_train_size > 0 and len(tr_feat) > max_train_size:
                rng = np.random.default_rng(seed + trial.number)
                idx = rng.choice(len(tr_feat), size=max_train_size, replace=False)
                tr_feat = tr_feat.iloc[idx].reset_index(drop=True)
                tr_tgt = tr_tgt[idx]

            train_enc = pd.concat(
                [tr_feat.reset_index(drop=True), pd.DataFrame({target_col: tr_tgt})],
                axis=1,
            )
            encoded_all_cols = encoded_feat_cols + [target_col]

            dim = train_enc.shape[1]
            solver = IMFDSBMSolver(dim=dim, cfg=cfg)
            solver.fit(train_enc)

            x_synth = solver.sample(
                n=len(test_feat_orig),
                seed=seed + trial.number + 9999,
            )
            synth_enc = pd.DataFrame(x_synth, columns=encoded_all_cols)
            synth_feat_enc = synth_enc[encoded_feat_cols].copy()
            synth_feat_orig = pipe.inverse_transform(synth_feat_enc)

            score = tuning_objective_value(
                test_feat_orig, synth_feat_orig,
                continuous_cols, disc_cat_cols, feature_cols, is_mixed,
            )

            trial.report(score, step=0)
            if trial.should_prune():
                raise optuna.TrialPruned()

            return score

        except optuna.TrialPruned:
            raise
        except Exception as e:
            trial.set_user_attr("exception", repr(e))
            return float("inf")

    return objective


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Optuna tuning: DSBM+CT+MLP+joint on mixed/categorical tabular data"
    )
    ap.add_argument("--pickle", type=str,
                    default="sbtab/data/datasets/datasets_mixed.pkl")
    ap.add_argument("--datasets", type=str, default="ALL")
    ap.add_argument("--test-size", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-train-size", type=int, default=5000, dest="max_train_size")

    ap.add_argument("--n-trials", type=int, default=30)
    ap.add_argument("--timeout", type=int, default=0,
                    help="Per-dataset timeout in seconds (0 = no limit).")
    ap.add_argument("--storage", type=str, default="sqlite:///dsbm_ct_mlp_mixed_optuna.db")
    ap.add_argument("--study-prefix", type=str, default="dsbm_ct_mlp_mixed")
    ap.add_argument("--outdir", type=str, default="dsbm_ct_mlp_mixed_optuna_results")
    ap.add_argument("--export-trials", action="store_true")
    ap.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda", "auto"])

    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    with open(args.pickle, "rb") as f:
        all_data: Dict[str, pd.DataFrame] = pickle.load(f)

    if args.datasets.strip().upper() == "ALL":
        ds_list = list(all_data.keys())
    else:
        ds_list = [d.strip() for d in args.datasets.split(",") if d.strip()]
        missing = [d for d in ds_list if d not in all_data]
        if missing:
            raise KeyError(f"Datasets not found in pickle: {missing}")

    sampler = TPESampler(seed=args.seed)
    pruner = MedianPruner(n_startup_trials=5, n_warmup_steps=0)
    summary_rows: List[Dict] = []

    for ds_name in ds_list:
        print("\n" + "=" * 90)
        print(f"DATASET: {ds_name}  [DSBM+CT+MLP+joint mixed tuning]")
        print("=" * 90)

        df_raw = all_data[ds_name].copy()

        feature_types = df_raw.attrs.get("feature_types", {})
        target_col: str = df_raw.attrs.get("target_variable", df_raw.columns[-1])
        task_type: str = df_raw.attrs.get("task_type", "regression")

        continuous_cols: List[str] = list(feature_types.get("continuous", []))
        discrete_cols: List[str] = list(feature_types.get("discrete", []))
        categorical_cols: List[str] = list(feature_types.get("categorical", []))
        feature_cols: List[str] = continuous_cols + discrete_cols + categorical_cols
        disc_cat_cols = discrete_cols + categorical_cols

        is_mixed = len(continuous_cols) > 0

        print(
            f"  target={target_col}  task={task_type}  "
            f"cont={len(continuous_cols)}  disc={len(discrete_cols)}  "
            f"cat={len(categorical_cols)}  is_mixed={is_mixed}"
        )

        if len(feature_cols) < 1:
            print("[SKIP] No feature columns.")
            continue

        # --- Encode target to numeric ---
        target_series = df_raw[target_col]
        if (
            pd.api.types.is_object_dtype(target_series)
            or pd.api.types.is_bool_dtype(target_series)
            or isinstance(target_series.dtype, pd.CategoricalDtype)
        ):
            le = LabelEncoder()
            df_raw = df_raw.copy()
            df_raw[target_col] = le.fit_transform(target_series.astype(str)).astype(float)
        else:
            df_raw[target_col] = pd.to_numeric(df_raw[target_col], errors="coerce")

        # --- Holdout split ---
        idx_all = np.arange(len(df_raw))
        train_idx, test_idx = train_test_split(
            idx_all, test_size=args.test_size, random_state=args.seed, shuffle=True,
        )
        df_train = df_raw.iloc[train_idx].copy().reset_index(drop=True)
        df_test = df_raw.iloc[test_idx].copy().reset_index(drop=True)

        # --- Transform features (fit on train only) ---
        schema = TabularSchema(
            continuous_cols=continuous_cols,
            discrete_cols=discrete_cols,
            categorical_cols=categorical_cols,
        )
        if is_mixed or categorical_cols:
            pipe = TransformPipeline.default_impute_scale_encode()
        else:
            pipe = TransformPipeline.default_dropna_and_scale()

        pipe.fit(df_train[feature_cols], schema)

        train_feat_enc = pipe.transform(df_train[feature_cols]).reset_index(drop=True)
        test_feat_enc = pipe.transform(df_test[feature_cols]).reset_index(drop=True)

        train_target = df_train[target_col].to_numpy()

        # Inverse-transform test features for JS/WD metric in original space
        test_feat_orig = pipe.inverse_transform(test_feat_enc)

        print(f"  train={len(train_feat_enc)}  test={len(test_feat_enc)}"
              f"  encoded_dim={train_feat_enc.shape[1]}")

        obj_label = "WD+JS" if is_mixed else "JS"

        objective_fn = make_objective(
            train_feat_enc=train_feat_enc,
            test_feat_orig=test_feat_orig,
            pipe=pipe,
            train_target=train_target,
            target_col=target_col,
            continuous_cols=continuous_cols,
            disc_cat_cols=disc_cat_cols,
            feature_cols=feature_cols,
            is_mixed=is_mixed,
            seed=args.seed,
            max_train_size=args.max_train_size,
            device=args.device,
        )

        study_name = f"{args.study_prefix}__{ds_name}"
        storage = args.storage if args.storage and args.storage != ":memory:" else None
        study = optuna.create_study(
            study_name=study_name,
            storage=storage,
            load_if_exists=True,
            direction="minimize",
            sampler=sampler,
            pruner=pruner,
        )

        t0 = time.time()
        study.optimize(
            objective_fn,
            n_trials=args.n_trials,
            timeout=args.timeout if args.timeout > 0 else None,
            gc_after_trial=True,
            show_progress_bar=True,
        )
        elapsed = time.time() - t0

        best = study.best_trial
        print(f"\n  Best {obj_label}: {best.value:.6f}")
        print(f"  Best params: {best.params}")
        print(f"  Trials: {len(study.trials)}  Elapsed: {elapsed:.1f}s")

        best_json = {
            "dataset": ds_name,
            "solver": "DSBM+CT+MLP+joint (mixed)",
            "tuning_objective": obj_label,
            "best_value": float(best.value),
            "best_trial": int(best.number),
            "n_trials": int(len(study.trials)),
            "elapsed_sec": float(elapsed),
            "best_params": dict(best.params),
        }
        (outdir / f"{ds_name}_best.json").write_text(
            json.dumps(best_json, indent=2), encoding="utf-8"
        )

        if args.export_trials:
            export_trials_csv(study, outdir / f"{ds_name}_trials.csv")

        summary_rows.append({
            "dataset": ds_name,
            "is_mixed": is_mixed,
            "tuning_objective": obj_label,
            "best_value": float(best.value),
            "best_trial": int(best.number),
            "n_trials": int(len(study.trials)),
            "elapsed_sec": float(elapsed),
            **best.params,
        })

    summary_df = pd.DataFrame(summary_rows)
    if not summary_df.empty and "best_value" in summary_df.columns:
        summary_df = summary_df.sort_values("best_value", ascending=True)

    out_csv = outdir / "ct_mlp_mixed_optuna_summary.csv"
    summary_df.to_csv(out_csv, index=False)

    print("\n" + "=" * 90)
    print("TUNING DONE. Summary:")
    with pd.option_context("display.max_columns", 200, "display.width", 200):
        print(summary_df.to_string(index=False))
    print(f"\nSaved to: {out_csv}")
    print("=" * 90)


if __name__ == "__main__":
    main()
