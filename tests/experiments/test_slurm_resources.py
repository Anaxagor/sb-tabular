"""Exercise the real submitter against a recording scheduler, without launching jobs."""
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

import pytest


@pytest.fixture
def submit(tmp_path):
    scripts = Path(__file__).resolve().parents[2] / "scripts/slurm"
    batch_dir = tmp_path / "local batch scripts"
    batch_dir.mkdir()
    for name in ("prepare", "experiment", "metrics", "aggregate"):
        (batch_dir / f"{name}.sbatch").write_text("#!/usr/bin/env bash\nexit 0\n")
    fakebin = tmp_path / "bin"
    fakebin.mkdir()
    scheduler = fakebin / "sbatch"
    scheduler.write_text(
        f"#!{sys.executable}\n"
        "import json,os,sys\nfrom pathlib import Path\n"
        "p=Path(os.environ['SBATCH_CALLS'])\n"
        "rows=p.read_text().splitlines() if p.exists() else []\n"
        "with p.open('a') as f: f.write(json.dumps(sys.argv[1:])+'\\n')\n"
        "print(str(100+len(rows))+';cluster')\n")
    scheduler.chmod(0o755)
    calls_file = tmp_path / "calls.jsonl"

    def run(*, stage="all", device="auto", task_ids="", smoke=False):
        config = tmp_path / "cluster settings.sh"
        config.write_text(
            f"SBTAB_PYTHON={shlex.quote(sys.executable)}\n"
            f"SBTAB_STAGE={shlex.quote(stage)}\n"
            f"SBTAB_DEVICE={shlex.quote(device)}\n"
            f"SBTAB_TASK_IDS={shlex.quote(task_ids)}\n"
            "SBTAB_MAX_CONCURRENT=3\n")
        env = {**os.environ, "PATH": str(fakebin) + os.pathsep + os.environ["PATH"],
               "SBTAB_BATCH_SCRIPT_DIR": str(batch_dir), "SBATCH_CALLS": str(calls_file)}
        args = ["bash", str(scripts / "submit.sh"), str(config), "--output-root", str(tmp_path / "output"),
                "--datasets", "diabetes", "--models", "forestdiffusion", "tabbyflow"]
        if smoke:
            args.append("--smoke")
        result = subprocess.run(args, env=env, capture_output=True, text=True)
        calls = [json.loads(line) for line in calls_file.read_text().splitlines()] if calls_file.exists() else []
        return result, calls

    return run


def option(call, name):
    return [arg.split("=", 1)[1] for arg in call if arg.startswith(name + "=")][-1]


def assert_cpu(call):
    assert option(call, "--gpus") == "0"
    assert option(call, "--constraint") == "type_d"


def test_production_separates_forest_cpu_tabby_gpu_and_per_task_cpu_metrics(submit):
    result, calls = submit()
    assert result.returncode == 0, result.stderr
    assert len(calls) == 6
    prep, forest, forest_metrics, tabby, tabby_metrics, aggregate = calls
    for call in (prep, forest, forest_metrics, tabby_metrics, aggregate):
        assert_cpu(call)
    assert option(tabby, "--gpus") == "1"
    assert option(tabby, "--constraint") == "type_a|type_b|type_c"
    for call in (forest, tabby):
        assert option(call, "--time") == "3-00:00:00"
        assert option(call, "--cpus-per-task") == "4"
        assert option(call, "--dependency") == "afterok:100"
        assert call[-2:] == ["generate", "0"]
    assert option(forest, "--array") == option(forest_metrics, "--array") == "0-0%3"
    assert option(tabby, "--array") == option(tabby_metrics, "--array") == "1-1%3"
    assert option(forest_metrics, "--dependency") == "aftercorr:101"
    assert option(tabby_metrics, "--dependency") == "aftercorr:103"
    for call in (forest_metrics, tabby_metrics):
        assert option(call, "--time") == "12:00:00"
        assert option(call, "--kill-on-invalid-dep") == "yes"
    # Even if evaluators are canceled early, aggregate waits for their generators.
    assert option(aggregate, "--dependency") == "afterany:100:101:102:103:104"
    assert option(prep, "--array") == option(aggregate, "--array") == "0-0"
    assert not any(arg.startswith(("--mem", "--nodelist")) for call in calls for arg in call)
    plan = json.loads(Path(prep[-1]).read_text())
    assert plan["model_devices"] == {"forestdiffusion": "cpu", "tabbyflow": "cuda"}


def test_metrics_only_never_reserves_a_gpu_for_cuda_plan(submit):
    result, calls = submit(stage="metrics")
    assert result.returncode == 0, result.stderr
    assert len(calls) == 3
    for call in calls:
        assert_cpu(call)
    assert option(calls[1], "--array") == "0-1%3"
    assert option(calls[1], "--dependency") == "afterok:100"
    assert Path(calls[1][-4]).name == "metrics.sbatch"
    assert option(calls[2], "--dependency") == "afterany:100:101"


def test_cpu_smoke_combines_models_and_limits_debug_time(submit):
    result, calls = submit(device="cpu", smoke=True)
    assert result.returncode == 0, result.stderr
    assert len(calls) == 4
    for call in calls:
        assert_cpu(call)
    assert option(calls[1], "--array") == option(calls[2], "--array") == "0-1%3"
    assert [option(call, "--time") for call in calls] == ["00:15:00", "00:30:00", "00:30:00", "00:15:00"]


def test_selected_gpu_generation_does_not_submit_unused_cpu_or_metrics_tasks(submit):
    result, calls = submit(stage="generate", task_ids="1")
    assert result.returncode == 0, result.stderr
    assert len(calls) == 3
    assert option(calls[1], "--gpus") == "1"
    assert option(calls[1], "--array") == "1-1%3"
    assert option(calls[2], "--dependency") == "afterany:100:101"
    assert all(not any(arg.endswith("metrics.sbatch") for arg in call) for call in calls)


def test_invalid_task_selection_submits_no_jobs(submit):
    result, calls = submit(task_ids="2")
    assert result.returncode != 0
    assert "valid comma-separated array indices" in result.stderr
    assert calls == []
