"""Check the pinned cluster environment and cache the two default TabPFN v2 priors."""
from __future__ import annotations

import argparse
from importlib import import_module, metadata
import json
import sys

from sbtab.experiments.experiment_common import REPO_ROOT, StageError, file_hash


def check_cuda() -> dict:
    import torch

    if not torch.cuda.is_available():
        raise StageError("training_failed", "CUDA is required: use CUDA PyTorch and a GPU allocation; CPU fallback is disabled")
    try:
        x = torch.ones((16, 16), device="cuda")
        assert float((x @ x).sum()) == 4096.0
        torch.cuda.synchronize()
    except Exception as error:
        raise StageError("training_failed", f"CUDA kernel check failed: {error}") from error
    return {"torch": torch.__version__, "cuda_runtime": torch.version.cuda,
            "device": torch.cuda.get_device_name(0), "capability": list(torch.cuda.get_device_capability(0))}


def check_packages() -> dict:
    from packaging.requirements import Requirement

    if sys.version_info[:2] != (3, 11):
        raise RuntimeError("The cluster environment requires Python 3.11")
    versions = {}
    names = {"scikit-learn": "sklearn", "PyYAML": "yaml"}
    for line in (REPO_ROOT / "requirements-cluster.txt").read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        requirement = Requirement(line)
        version = metadata.version(requirement.name)
        if version not in requirement.specifier:
            raise RuntimeError(f"Expected {requirement}, found {version}; run scripts/setup_cluster_env.sh")
        import_module(names.get(requirement.name, requirement.name))
        versions[requirement.name] = version
    return versions


def check_tabpfn_cache(download=False) -> dict:
    from tabpfn.model.loading import download_model, resolve_model_path

    weights = {}
    for which in ("classifier", "regressor"):
        path, _, _, _ = resolve_model_path(None, which, "v2")
        if not path.is_file() and download:
            path.parent.mkdir(parents=True, exist_ok=True)
            result = download_model(path, version="v2", which=which)
            if result != "ok":
                raise RuntimeError(f"TabPFN {which} download failed: {result}")
        if not path.is_file() or path.stat().st_size == 0:
            raise RuntimeError(f"Missing TabPFN weights: {path}; run with --download-tabpfn on a node with Internet access")
        weights[which] = {"path": str(path), "sha256": file_hash(path), "bytes": path.stat().st_size}
    return weights


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--require-cuda", action="store_true", help="run on an allocated GPU node")
    parser.add_argument("--download-tabpfn", action="store_true", help="download missing weights before batch submission")
    args = parser.parse_args(argv)
    try:
        report = {"packages": check_packages(), "tabpfn_weights": check_tabpfn_cache(args.download_tabpfn)}
        if args.require_cuda:
            report["cuda"] = check_cuda()
        report["status"] = "ok"
        print(json.dumps(report, indent=2))
        return 0
    except Exception as error:
        print(json.dumps({"status": "failed", "error": str(error)}), file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
