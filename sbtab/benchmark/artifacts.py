"""Reviewable local artifacts for completed cross-validation generation runs.

The writer stores the post-policy raw dataset once, one decoded synthetic table
per fold, and a manifest containing every positional split and runtime control.
It does not duplicate real train/test tables because they are reconstructed
exactly from the stored raw frame and split positions.

Artifact writing is intentionally create-only. A caller must choose a new
directory for every run; benchmark code never overwrites prior evidence.
"""

from __future__ import annotations

import json
import math
from dataclasses import fields, is_dataclass
from enum import Enum
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd

from sbtab.benchmark.runner import CrossValidationResult
from sbtab.benchmark.validation import ContractViolation


CROSS_VALIDATION_ARTIFACT_VERSION = 1


def _json_value(value: object) -> object:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ContractViolation(
                "Artifact metadata must not contain non-finite floats."
            )
        return value
    if isinstance(value, np.generic):
        return _json_value(value.item())
    if isinstance(value, Enum):
        return _json_value(value.value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, pd.Timedelta):
        return value.isoformat()
    if is_dataclass(value):
        return {
            field.name: _json_value(getattr(value, field.name))
            for field in fields(value)
        }
    if isinstance(value, Mapping):
        result: dict[str, object] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise ContractViolation(
                    "Artifact metadata mapping keys must be strings."
                )
            result[key] = _json_value(item)
        return result
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    raise ContractViolation(
        "Artifact metadata contains unsupported value "
        f"{value!r} of type {type(value).__name__}."
    )


def _dataset_manifest(result: CrossValidationResult) -> dict[str, object]:
    dataset = result.dataset
    return {
        "name": dataset.name,
        "target": dataset.target,
        "task": _json_value(dataset.task),
        "identifier": dataset.identifier,
        "raw_columns": list(dataset.frame.columns),
        "modeled_columns": [
            {
                "name": column.name,
                "kind": column.kind.value,
                "ordered_values": _json_value(column.ordered_values),
            }
            for column in dataset.columns
        ],
        "rows": len(dataset.frame),
    }


def _config_manifest(result: CrossValidationResult) -> dict[str, object]:
    config = result.config
    return {
        "split": {
            "type": type(config.split).__name__,
            **{
                field.name: _json_value(getattr(config.split, field.name))
                for field in fields(config.split)
            },
        },
        "missing_policy": config.missing_policy.value,
        "run_id": config.run_id,
        "training_seed": config.training_seed,
        "sample_seed": config.sample_seed,
        "device": config.device,
        "artifact_dir": str(config.artifact_dir),
    }


def write_cross_validation_artifacts(
    result: CrossValidationResult,
    output_dir: Path,
) -> Path:
    """Create a complete local handoff directory for one generation run.

    Parameters
    ----------
    result:
        Successful pre-evaluation result returned by
        :func:`run_cross_validation`.
    output_dir:
        New destination directory. It and any manifest or table within it must
        not already exist. Parent directories may be created.

    Returns
    -------
    Path
        Path to the manifest written last after every table succeeds.
    """

    if not isinstance(result, CrossValidationResult):
        raise ContractViolation("result must be CrossValidationResult.")
    if not isinstance(output_dir, Path):
        raise ContractViolation("output_dir must be pathlib.Path.")
    try:
        output_dir.mkdir(parents=True, exist_ok=False)
    except FileExistsError as error:
        raise ContractViolation(
            f"Artifact directory already exists: {output_dir}."
        ) from error

    real_path = output_dir / "real-post-policy.csv"
    result.dataset.frame.to_csv(real_path, index=False)

    fold_entries: list[dict[str, object]] = []
    for fold in result.folds:
        fold_dir = output_dir / f"fold-{fold.split.fold_id}"
        fold_dir.mkdir()
        synthetic_path = fold_dir / "synthetic.csv"
        fold.synthetic_raw.to_csv(synthetic_path, index=False)
        fold_entries.append(
            {
                "fold_id": fold.split.fold_id,
                "train_positions": list(fold.split.train_positions),
                "test_positions": list(fold.split.test_positions),
                "train_rows": len(fold.train_raw),
                "test_rows": len(fold.test_raw),
                "synthetic_rows": len(fold.synthetic_raw),
                "fit_seconds": fold.fit_seconds,
                "sample_seconds": fold.sample_seconds,
                "synthetic_path": str(synthetic_path.relative_to(output_dir)),
            }
        )

    manifest = {
        "artifact_type": "cross_validation_generation",
        "artifact_version": CROSS_VALIDATION_ARTIFACT_VERSION,
        "adapter_name": result.adapter_name,
        "dataset": _dataset_manifest(result),
        "config": _config_manifest(result),
        "missing_report": _json_value(result.missing_report),
        "real_path": str(real_path.relative_to(output_dir)),
        "folds": fold_entries,
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return manifest_path
