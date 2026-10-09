"""
Shared infrastructure for the experiment stages: protocol/config loading, hashes
and provenance, the seed ledger, statuses, atomic local persistence, artifact
paths and timing. Orchestration lives in the stage modules; reusable solver and
metric code lives outside ``sbtab.experiments``.
"""
from __future__ import annotations

import hashlib
from importlib import metadata
import json
import math
import os
import platform
import subprocess
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterator, Optional

import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
# v2 = v1 + the dataset-eligibility rule (rows with a finite-support value occurring < 3 times are removed
# before splitting). v1 stays available, frozen, via --protocol configs/protocols/sbtab_8515_hpo100_cv5_v1.yaml.
PRODUCTION_PROTOCOL = Path("configs/protocols/sbtab_8515_hpo100_cv5_v2.yaml")
SMOKE_PROTOCOL = Path("configs/protocols/sbtab_smoke_v2.yaml")

STATUSES = (
    "ok", "not_applicable", "insufficient_data", "incomplete_conditional_coverage", "undefined",
    "invalid_generated_data", "training_failed", "sampling_failed", "utility_fit_failed", "blocked_support",
)

# Values the canonical production protocol must resolve to (checked by --dry-run and tests).
PRODUCTION_CONSTANTS = {
    ("split", "test_size"): 0.15, ("split", "random_state"): 5,
    ("tuning", "n_trials"): 100, ("tuning", "sampler_seed"): 5, ("tuning", "n_jobs"): 1,
    ("tuning", "pruner"): "none", ("tuning", "direction"): "minimize",
    ("cv", "n_splits"): 5, ("cv", "shuffle"): True, ("cv", "random_state"): 42, ("cv", "population"): "T",
}


# --------------------------------------------------------------------------- JSON / hashing
def json_safe(obj: Any) -> Any:
    """Standards-compliant JSON: numpy scalars unwrapped, NaN / +-inf -> null."""
    if isinstance(obj, dict):
        return {str(k): json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [json_safe(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return [json_safe(v) for v in obj.tolist()]
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        f = float(obj)
        return f if math.isfinite(f) else None
    if isinstance(obj, Path):
        return str(obj)
    return obj


def canonical_hash(obj: Any) -> str:
    return hashlib.sha256(json.dumps(json_safe(obj), sort_keys=True, allow_nan=False).encode()).hexdigest()


def file_hash(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def atomic_write_text(path, text: str) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp{os.getpid()}")
    with open(tmp, "w", encoding="utf-8") as fh:
        fh.write(text)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)
    return path


def write_json(path, obj: Any) -> Path:
    return atomic_write_text(path, json.dumps(json_safe(obj), indent=1, allow_nan=False, sort_keys=False))


def read_json(path) -> Any:
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


@contextmanager
def file_lock(path, blocking: bool = True):
    """Process lock for shared cluster artifacts; the filesystem must support flock.

    Keep the lock file in place: unlinking it would let another process lock a
    different inode. The OS releases the lock when a killed worker exits.
    """
    import fcntl
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+", encoding="utf-8") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | (0 if blocking else fcntl.LOCK_NB))
        except BlockingIOError as e:
            raise StageError("undefined", f"another process owns {path}") from e
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def append_jsonl(path, record: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(json_safe(record), allow_nan=False) + "\n")


# --------------------------------------------------------------------------- protocol
@dataclass(frozen=True)
class Protocol:
    path: str
    data: dict

    @property
    def id(self) -> str:
        return self.data["protocol_id"]

    @property
    def kind(self) -> str:
        return self.data["kind"]

    def hash(self) -> str:
        return canonical_hash(self.data)

    @property
    def eligibility(self) -> dict:
        """Dataset-eligibility rule; empty for protocols (v1) that never remove a row."""
        return dict(self.data.get("eligibility") or {})

    def __getitem__(self, key):
        return self.data[key]


def load_yaml(path) -> dict:
    with open(path, encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def load_protocol(path=None, smoke: bool = False) -> Protocol:
    """
    Production constants come from the versioned protocol file, never from CLI
    flags. ``--smoke`` selects the separate smoke protocol (its own id and hash).
    """
    if path is None:
        path = SMOKE_PROTOCOL if smoke else PRODUCTION_PROTOCOL
    data = load_yaml(path)
    for key in ("protocol_id", "kind", "split", "tuning", "cv", "seeds", "metrics_config"):
        if key not in data:
            raise ValueError(f"protocol {path} lacks '{key}'")
    if smoke and data["kind"] != "smoke":
        raise ValueError("--smoke requires a protocol with kind: smoke")
    if not smoke and data["kind"] == "smoke":
        raise ValueError(f"{path} is a smoke protocol; pass --smoke to run it explicitly")
    if data["kind"] == "production":
        assert_production_constants(data)
    if "eligibility" in data:
        from sbtab.data.eligibility import validate_rule
        validate_rule(data["eligibility"])          # unknown keys / bad thresholds are rejected up front
    if "support_repair" in data["split"]:
        from sbtab.data.support_split import validate_repair_rule
        validate_repair_rule(data["split"]["support_repair"])
    return Protocol(path=str(path), data=data)


def assert_production_constants(data: dict) -> None:
    wrong = {f"{a}.{b}": (data[a].get(b), want) for (a, b), want in PRODUCTION_CONSTANTS.items()
             if data[a].get(b) != want}
    if wrong:
        raise ValueError("a protocol with kind: production must keep the canonical constants; "
                         f"found (actual, required): {wrong}. Use a new protocol id and kind for a variant.")


def load_metric_config(protocol: Protocol) -> dict:
    return load_yaml(REPO_ROOT / protocol["metrics_config"] if not Path(protocol["metrics_config"]).is_absolute()
                     else protocol["metrics_config"])


# --------------------------------------------------------------------------- provenance
SOURCE_HASH_VERSION = "sbtab.source-files/1"


def source_content_hash() -> str:
    """Hash runtime Python/config files, independently of Git and checkout location.

    Dataset bundles have their own value fingerprints in the split artifacts.
    Bytecode, notebooks, images, logs and documentation are not runtime sources.
    Re-read files each time so edits during a queued/running experiment are detected.
    """
    sources = {}
    for directory in (REPO_ROOT / "sbtab", REPO_ROOT / "configs"):
        if not directory.is_dir():
            raise RuntimeError(f"source directory is missing: {directory}")
        for path in sorted(directory.rglob("*")):
            if path.is_file() and path.suffix in {".py", ".yaml", ".yml", ".json", ".toml"}:
                sources[path.relative_to(REPO_ROOT).as_posix()] = file_hash(path)
    return canonical_hash({"version": SOURCE_HASH_VERSION, "files": sources})


def _git(*args: str) -> Optional[str]:
    try:
        out = subprocess.run(["git", *args], cwd=REPO_ROOT, capture_output=True,
                             text=True, encoding="utf-8", timeout=30)
        return out.stdout if out.returncode == 0 else None
    except Exception:
        return None


def source_provenance() -> dict:
    """Runtime content identity, plus best-effort Git metadata for reporting only."""
    commit = (_git("rev-parse", "HEAD") or "").strip() or None
    diff = _git("diff", "HEAD", "--", "sbtab", "configs") or ""
    untracked = _git("ls-files", "--others", "--exclude-standard", "--", "sbtab", "configs") or ""
    h = hashlib.sha256(diff.encode())
    for rel in sorted(p for p in untracked.splitlines() if p):
        fp = REPO_ROOT / rel
        if fp.is_file():
            h.update(rel.encode())
            h.update(fp.read_bytes())
    dirty = bool(diff.strip() or untracked.strip())
    return {"commit": commit, "dirty": dirty, "dirty_diff_hash": h.hexdigest() if dirty else None,
            "source_hash_version": SOURCE_HASH_VERSION, "source_hash": source_content_hash()}


def implementation_hash(provenance: Optional[dict] = None) -> str:
    # Git may be absent on a compute node or point to a different metadata-only
    # commit. Neither changes the implementation that the worker will execute.
    if provenance is None:
        return source_content_hash()
    if provenance.get("source_hash_version") != SOURCE_HASH_VERSION:
        raise ValueError("legacy or unsupported source provenance; create a new run")
    return provenance["source_hash"]


def library_versions() -> dict:
    out = {"python": sys.version.split()[0]}
    for mod in ("numpy", "pandas", "scipy", "sklearn", "torch", "catboost", "xgboost", "optuna", "pyarrow",
                "pgmpy", "networkx", "sdv", "ctgan", "tabpfgen", "tabpfn", "yaml", "tqdm", "geotorch"):
        try:
            # Some packages (including TabPFGen) expose no __version__. Read the
            # installed distribution so environment drift is still detected.
            out[mod] = metadata.version({"sklearn": "scikit-learn", "yaml": "PyYAML"}.get(mod, mod))
        except metadata.PackageNotFoundError:
            out[mod] = None
    return out


def hardware_info() -> dict:
    info = {"platform": platform.platform(), "machine": platform.machine(), "processor": platform.processor(),
            "cpu_count": os.cpu_count()}
    try:
        import torch
        info["torch_threads"] = torch.get_num_threads()
        info["cuda_available"] = bool(torch.cuda.is_available())
        if info["cuda_available"]:
            info["cuda_device"] = torch.cuda.get_device_name(0)
    except Exception:
        pass
    return info


def peak_memory_mb() -> Optional[float]:
    try:
        import resource
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return rss / (1024.0 * 1024.0) if sys.platform == "darwin" else rss / 1024.0
    except Exception:
        return None


# --------------------------------------------------------------------------- seeds
class SeedLedger:
    """
    Every random stream gets its own recorded seed, derived from (base, purpose,
    indices). Distinct purposes never share a seed, so e.g. sampling batches or the
    reference draw and the dynamics noise are not reseeded to identical outputs.
    """

    def __init__(self, base: int):
        self.base = int(base)
        self.records: Dict[str, int] = {}

    def seed(self, purpose: str, *index: int) -> int:
        key = ":".join([purpose, *[str(int(i)) for i in index]])
        if key not in self.records:
            digest = hashlib.sha256(f"{self.base}:{key}".encode()).digest()
            self.records[key] = int.from_bytes(digest[:4], "little") % (2 ** 31 - 1)
        return self.records[key]

    def to_dict(self) -> dict:
        return {"base": self.base, "derived": dict(self.records)}


def seed_everything(seed: int) -> None:
    import random
    random.seed(seed)
    np.random.seed(seed % (2 ** 32))
    try:
        import torch
        torch.manual_seed(seed)
    except Exception:
        pass


# --------------------------------------------------------------------------- timing
TIMING_DEFINITIONS = {
    "preprocessing_seconds": "fitting and applying the common preprocessing",
    "model_init_seconds": "model construction and device set-up",
    "generator_fit_seconds": "all IPF/IMF stages, coupling/cache refreshes, graph learning and baseline-internal "
                             "learned transforms executed inside fit; EXCLUDES common preprocessing, initialisation, "
                             "validation metrics, checkpoint output and final sampling; accumulated over resumed segments",
    "checkpoint_io_seconds": "writing (and verifying the reload of) the checkpoint",
    "generation_seconds": "sampling in the model representation",
    "inverse_transform_seconds": "decoding model output back to the common schema",
    "training_total_seconds": "preprocessing + model_init + generator_fit (end-to-end training)",
    "stage_wall_seconds": "total wall time of the stage for this unit of work",
}


def _cuda_sync() -> None:
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.synchronize()
    except Exception:
        pass


class Timer:
    """perf_counter intervals with CUDA synchronisation before and after each one."""

    def __init__(self):
        self.seconds: Dict[str, float] = {}

    @contextmanager
    def measure(self, name: str) -> Iterator[None]:
        _cuda_sync()
        t0 = time.perf_counter()
        try:
            yield
        finally:
            _cuda_sync()
            self.seconds[name] = self.seconds.get(name, 0.0) + (time.perf_counter() - t0)

    def add(self, name: str, seconds: float) -> None:
        self.seconds[name] = self.seconds.get(name, 0.0) + float(seconds)

    def to_dict(self) -> dict:
        return dict(self.seconds)


# --------------------------------------------------------------------------- artifact layout
def claim_output_root(output_root, protocol: "Protocol") -> Path:
    """
    ``output_root`` is the protocol-level directory (``<root>/<protocol>``), e.g.
    ``artifacts/sbtab_8515_hpo100_cv5_v1``. Its ``protocol.json`` pins the protocol
    id and hash, so artifacts of two protocols (production vs smoke, or a changed
    constant) can never share a namespace.
    """
    root = Path(output_root)
    marker = root / "protocol.json"
    want = {"protocol_id": protocol.id, "kind": protocol.kind, "protocol_hash": protocol.hash()}
    if marker.exists():
        have = read_json(marker)
        if have.get("protocol_hash") != want["protocol_hash"]:
            raise StageError("undefined", f"{root} belongs to protocol {have.get('protocol_id')!r} "
                             f"(hash {str(have.get('protocol_hash'))[:12]}); refusing to write protocol {protocol.id!r} "
                             f"(hash {want['protocol_hash'][:12]}) into it. Use a different --output-root.")
    else:
        write_json(marker, {**want, "protocol": protocol.data})
    return root


@dataclass(frozen=True)
class ArtifactPaths:
    """<output-root = root/protocol>/<dataset>/<model>/<run-id>/..."""
    root: Path
    dataset: str

    @property
    def dataset_dir(self) -> Path:
        return Path(self.root) / self.dataset

    @property
    def splits(self) -> Path:
        return self.dataset_dir / "splits.json"

    @property
    def schema(self) -> Path:
        return self.dataset_dir / "schema.json"

    @property
    def data(self) -> Path:
        return self.dataset_dir / "data.parquet"

    def run_dir(self, model: str, run_id: str) -> Path:
        return self.dataset_dir / model / run_id


def dataset_dir_from_splits(splits_path) -> Path:
    return Path(splits_path).resolve().parent


class StageError(RuntimeError):
    """A stage cannot proceed; carries a machine-readable status."""

    def __init__(self, status: str, message: str, details: Optional[dict] = None):
        super().__init__(message)
        self.status = status
        self.details = details or {}
