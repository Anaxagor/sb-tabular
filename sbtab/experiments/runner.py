"""
One fit-and-generate unit of work, shared by tuning (fit T, generate len(V)) and
cross-validation (fit T_k, generate len(T_k)). Nothing here knows about Optuna or
folds, so the two stages cannot drift apart.

Everything is persisted even when a later step fails: a training failure still
leaves a status record and its elapsed time; a sampling failure still leaves the
checkpoint; nothing here computes a metric.
"""
from __future__ import annotations

import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

from sbtab.data.dataset_schema import DatasetSchema
from sbtab.data.preprocessing import CommonPreprocessor, row_id_hash
from sbtab.evaluation.validity import check_validity, numerical_diagnostics
from sbtab.experiments.experiment_common import (
    SeedLedger, TIMING_DEFINITIONS, Timer, append_jsonl, hardware_info, peak_memory_mb, seed_everything, write_json,
)
from sbtab.solvers.registry import get_adapter_class

SYNTHETIC_FORMAT = {
    "file": "synthetic.parquet",
    "representation": "common (continuous standardised with the fold scaler, discrete original values, "
                      "categorical codes of the fold vocabulary); raw units via preprocessor/preprocessor.json",
    "index": "synthetic_id — generated ids, never joinable to real row ids",
}


def write_synthetic(path: Path, synth: pd.DataFrame, schema: DatasetSchema) -> None:
    out = synth.copy()
    for c in schema.categorical:
        # Preserve malformed outputs for validity diagnostics; never repair them by truncation.
        if c not in out or not pd.api.types.is_numeric_dtype(out[c].dtype):
            continue
        values = out[c].to_numpy(dtype=np.float64)
        if (np.isfinite(values).all() and np.equal(values, np.floor(values)).all()
                and ((values >= -(2.0 ** 63)) & (values < 2.0 ** 63)).all()):
            out[c] = out[c].astype(np.int64)
    tmp = path.with_name(path.name + ".tmp")
    out.reset_index().to_parquet(tmp, index=False)
    tmp.replace(path)


def read_synthetic(path: Path, schema: DatasetSchema) -> pd.DataFrame:
    df = pd.read_parquet(path).set_index("synthetic_id")
    return df[schema.column_order]


def generated_validity(synth: pd.DataFrame, pre: CommonPreprocessor, n_expected: int) -> dict:
    """Validate common-space output using only the fitted training vocabulary."""
    support = {**pre.support, **{c: list(range(len(v))) for c, v in pre.vocab.items()}}
    validity = check_validity(synth, pre.schema, support=support)
    if len(synth) != n_expected:
        validity["status"] = "invalid_generated_data"
        validity["reasons"].append("wrong_row_count")
    validity["n_expected"] = int(n_expected)
    return validity


class InvalidGeneratedData(ValueError):
    def __init__(self, validity: dict):
        self.details = {"validity": validity}
        super().__init__("invalid generated data: " + ", ".join(validity["reasons"]))


def failure_record(error: Exception, stage: str) -> dict:
    record = {"type": type(error).__name__, "message": str(error), "stage": stage,
              "trace": traceback.format_exc()}
    if isinstance(getattr(error, "details", None), dict):
        record["details"] = error.details
    return record


def fit_generate(model_id: str, train_raw: pd.DataFrame, schema: DatasetSchema, config: dict, seeds: SeedLedger,
                 scope: tuple, n_samples: int, out_dir: Path, verify_reload: bool = True) -> dict:
    """
    Fresh preprocessing + fresh model on ``train_raw`` only; save checkpoint and the
    complete generated table. ``scope`` indexes the seed ledger, e.g. ("trial", 3).
    Returns a JSON-safe record; the generated table is at out_dir/synthetic.parquet.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    log = out_dir / "training_log.jsonl"
    timer = Timer()
    t_wall = time.perf_counter()
    rec = {
        "model": model_id, "status": "ok", "failure": None, "n_train_rows": int(len(train_raw)),
        "n_requested": int(n_samples), "n_generated": None, "train_row_hash": row_id_hash(train_raw.index),
        "requested_config": dict(config), "effective_config": None, "n_updates": None, "describe": None,
        "decoding_report": None, "checkpoint": None, "reload_verified": None,
        "checkpoint_loaded": None, "sampling_probe": None, "failure_kind": None,
        "validity": None, "numerical_diagnostics": None,
        "synthetic": None, "synthetic_format": SYNTHETIC_FORMAT,
    }
    fit_seed = seeds.seed("fit", *scope[1:]) if scope else seeds.seed("fit")
    sample_seed = seeds.seed("sample", *scope[1:]) if scope else seeds.seed("sample")
    rec["seeds"] = {"fit": fit_seed, "sample": sample_seed, "scope": list(scope)}
    adapter = None
    stage = "preprocessing"
    try:
        with timer.measure("preprocessing_seconds"):
            pre = CommonPreprocessor(schema).fit(train_raw)
            train = pre.transform(train_raw)
            pre.save(out_dir / "preprocessor")
        rec["preprocessor"] = {"fit_row_hash": pre.fit_row_hash, "n_fit_rows": pre.n_fit_rows,
                               "imputed_counts": pre.fit_imputed_counts,
                               "constant_columns": [c for c, v in pre.constant.items() if v]}
        append_jsonl(log, {"event": "preprocessed", "fit_row_hash": pre.fit_row_hash})

        seed_everything(fit_seed)
        stage = "training"
        adapter = get_adapter_class(model_id)()
        try:
            adapter.fit(train, schema, config, seed=fit_seed)
        except Exception as e:
            rec.update(status="training_failed", failure_kind="training_failed", failure=failure_record(e, stage))
            raise _Handled()
        finally:
            for key, seconds in adapter.last_fit_timing.items():
                timer.add(key, seconds)
        if adapter.fit_row_hash != pre.fit_row_hash:
            raise RuntimeError("adapter and preprocessor were fitted on different rows")
        rec.update(effective_config=adapter.config, n_updates=adapter.n_updates, describe=adapter.describe())
        append_jsonl(log, {"event": "fitted", "n_updates": adapter.n_updates,
                           "generator_fit_seconds": timer.seconds["generator_fit_seconds"]})

        stage = "checkpoint_save"
        with timer.measure("checkpoint_io_seconds"):
            ckpt = adapter.save_checkpoint(out_dir / "checkpoint")
        rec["checkpoint"] = str(ckpt.name)

        stage = "generation"
        try:
            synth = adapter.sample(n_samples, seed=sample_seed)
        except Exception as e:
            rec.update(status="sampling_failed", failure_kind="sampling_failed", failure=failure_record(e, stage))
            raise _Handled()
        finally:
            for key, seconds in adapter.last_sample_timing.items():
                timer.add(key, seconds)
        rec.update(n_generated=int(len(synth)), decoding_report=adapter.decoding_report_)
        rec["validity"] = generated_validity(synth, pre, n_samples)
        invalid = rec["validity"]["status"] != "ok"
        if invalid:
            rec.update(status="invalid_generated_data", failure_kind="invalid_generated_data")
        rec["numerical_diagnostics"] = numerical_diagnostics(synth, schema, train)
        try:
            write_synthetic(out_dir / "synthetic.parquet", synth, schema)
            rec["synthetic"] = "synthetic.parquet"
        except Exception as e:
            if not invalid:
                raise
            # A malformed mixed-type column may not be representable in Parquet.
            # Retain its validity evidence and keep serialization as a secondary cause.
            rec["serialization_failure"] = failure_record(e, "synthetic_save")
        append_jsonl(log, {"event": "generated", "n": int(len(synth))})
        if invalid:
            raise InvalidGeneratedData(rec["validity"])

        if verify_reload:
            stage = "checkpoint_load"
            rec["checkpoint_loaded"] = False
            rec["reload_verified"] = {"ok": False}
            with timer.measure("checkpoint_io_seconds"):
                again = type(adapter).load_checkpoint(out_dir / "checkpoint")
                rec["checkpoint_loaded"] = True
                stage = "sampling_probe"
                k = min(64, n_samples)
                rec["sampling_probe"] = {"ok": False, "rows": int(k), "seed": sample_seed + 1}
                probes = []
                for label, model in (("original", adapter), ("reloaded", again)):
                    rec["sampling_probe"]["active_model"] = label
                    probe = model.sample(k, seed=sample_seed + 1)
                    validity = generated_validity(probe, pre, k)
                    rec["sampling_probe"][label] = validity
                    if validity["status"] != "ok":
                        raise InvalidGeneratedData(validity)
                    probes.append(probe.to_numpy(dtype=np.float64))
                rec["sampling_probe"].pop("active_model")
                rec["sampling_probe"]["ok"] = True
                a, b = probes
            # Both probes have already passed finite-value and category checks.
            with np.errstate(over="ignore"):
                differences = np.abs(a - b)
                matches = bool(np.allclose(a, b, atol=1e-6, rtol=0, equal_nan=False))
            finite_diff = bool(np.isfinite(differences).all())
            diff = float(differences.max()) if differences.size and finite_diff else (0.0 if not differences.size else None)
            rec["reload_verified"] = {"ok": matches and finite_diff, "rows": int(k),
                                      "max_abs_diff": diff, "tolerance": 1e-6, "rtol": 0,
                                      "device": str(adapter.config.get("device", "cpu"))}
            if not rec["reload_verified"]["ok"]:
                stage = "checkpoint_comparison"
                raise RuntimeError(f"checkpoint reload does not reproduce samples (max diff {diff})")
    except _Handled:
        pass
    except Exception as e:                      # unexpected: keep the record, never lose the elapsed time
        if rec["status"] == "ok":
            rec.update(status="training_failed" if rec["checkpoint"] is None else "sampling_failed",
                       failure_kind={"checkpoint_load": "checkpoint_load_failed", "sampling_probe": "sampling_probe_failed",
                                     "checkpoint_comparison": "checkpoint_mismatch"}.get(stage, stage + "_failed"))
        rec["failure"] = failure_record(e, stage)
    finally:
        s = timer.seconds
        s["training_total_seconds"] = sum(s.get(k, 0.0) for k in
                                          ("preprocessing_seconds", "model_init_seconds", "generator_fit_seconds"))
        s["stage_wall_seconds"] = time.perf_counter() - t_wall
        timing = {"seconds": timer.to_dict(), "definitions": TIMING_DEFINITIONS, "hardware": hardware_info(),
                  "peak_memory_mb": peak_memory_mb(), "n_updates": rec["n_updates"],
                  "sampled_rows": rec["n_generated"], "fit_segments": 1,
                  "note": "generator_fit_seconds is cumulative over fit segments; this run had one segment"}
        write_json(out_dir / "timing.json", timing)
        rec["timing"] = timer.to_dict()
        append_jsonl(log, {"event": "finished", "status": rec["status"]})
    return rec


class _Handled(Exception):
    pass
