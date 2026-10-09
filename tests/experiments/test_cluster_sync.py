"""Exercise real rsync transfers without a network or cluster connection."""
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys

import pytest


@pytest.fixture(params=["path", "system"])
def sync_tree(tmp_path, request):
    rsync = shutil.which("rsync")
    if not rsync:
        pytest.skip("rsync is required for the transfer integration check")
    if request.param == "system":
        system = Path("/usr/bin/rsync")
        if not system.exists() or system.resolve() == Path(rsync).resolve():
            pytest.skip("No separate system rsync to check")
        rsync = str(system)
    project = Path(__file__).resolve().parents[2]
    source = tmp_path / "local project"
    target = tmp_path / "cluster"
    target.mkdir()
    (source / "scripts/slurm").mkdir(parents=True)
    shutil.copyfile(project / "scripts/sync_cluster.sh", source / "scripts/sync_cluster.sh")
    shutil.copyfile(project / "scripts/slurm/cluster.example.sh", source / "scripts/slurm/cluster.local.sh")
    # rsync's remote-shell protocol runs the real receiver on this machine.
    transport = tmp_path / "fake ssh"
    transport.write_text(
        f"#!{sys.executable}\nimport subprocess,sys\n"
        "assert sys.argv[1] == 'test-cluster'\n"
        "sys.exit(subprocess.call(sys.argv[2:]))\n"
    )
    transport.chmod(0o755)
    binaries = tmp_path / "bin"
    binaries.mkdir()
    (binaries / "rsync").symlink_to(rsync)
    env = {**os.environ, "RSYNC_RSH": shlex.quote(str(transport)),
           "PATH": str(binaries) + os.pathsep + os.environ["PATH"]}
    return source, target, env


def put(root, relative, text):
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def upload(source, target, env, *flags):
    return subprocess.run(
        ["bash", str(source / "scripts/sync_cluster.sh"), *flags, f"test-cluster:{target}"],
        env=env, text=True, capture_output=True, check=True,
    )


def snapshot(root):
    return {str(path.relative_to(root)): path.read_bytes()
            for path in root.rglob("*") if path.is_file()}


def test_sync_updates_equal_timestamp_files_and_preserves_cluster_state(sync_tree):
    source, target, env = sync_tree
    old = put(target, "sbtab/model.py", "value = 1\n")
    new = put(source, "sbtab/model.py", "value = 2\n")
    os.utime(old, ns=(new.stat().st_atime_ns, new.stat().st_mtime_ns))
    put(source, "configs/model.yaml", "steps: 2\n")
    put(source, "sbtab/data/datasets/example.csv", "x,y\n1,2\n")
    put(source, ".gitignore", "*.csv\ncluster.local.sh\n")
    put(source, "new_notes.txt", "new uncommitted file\n")
    put(target, "scripts/slurm/cluster.local.sh", "SBATCH_SITE_ARGS=(--account=proj_1825)\n")
    obsolete = ["sbtab/removed.py", "configs/removed.yaml", "scripts/removed.sh",
                "tests/removed.py", "docs/removed.md", "examples/removed.py"]
    for name in obsolete:
        put(target, name, "old source\n")
    # Entire formerly managed directories must also be removed when absent locally.
    protected = ["artifacts/run/result.json", "slurm_logs/job.out", ".venv-cluster/bin/python",
                 ".venv/bin/python", "venv/bin/python", "env/bin/python", ".cache/weights.bin",
                 "catboost_info/events.json", ".git/config", ".env", ".env.local",
                 ".pytest_cache/v/cache", "sbtab/__pycache__/model.pyc",
                 "scripts/slurm/cluster.local.sh.bak", "study.sqlite3", "study.sqlite3-wal"]
    for name in protected:
        put(target, name, "cluster state\n")
        put(source, name, "local state must not replace cluster state\n")
    put(target, "custom-output/results.json", "keep remote-only root data\n")
    put(target, "remote-note.txt", "keep remote-only root file\n")
    put(source, ".aws/credentials", "must not upload\n")
    # Excluded environments can also be symlinks to a shared cluster location.
    (target / ".venv-shared").symlink_to("/cluster/shared/environment", target_is_directory=True)
    (source / ".venv-shared").symlink_to("/local/environment", target_is_directory=True)
    (source / "examples").mkdir(exist_ok=True)
    (source / "examples/sbtab").symlink_to(source / "sbtab", target_is_directory=True)
    completed = upload(source, target, env)
    assert "Upload verified" in completed.stdout
    assert (target / "sbtab/model.py").read_bytes() == new.read_bytes()
    assert (target / "configs/model.yaml").read_bytes() == (source / "configs/model.yaml").read_bytes()
    assert (target / "sbtab/data/datasets/example.csv").exists()
    assert (target / "new_notes.txt").exists()
    config = (target / "scripts/slurm/cluster.local.sh").read_text()
    assert "--account=proj_1752" in config and "proj_1825" not in config
    assert all(not (target / name).exists() for name in obsolete)
    assert all((target / name).read_text() == "cluster state\n" for name in protected)
    assert (target / "custom-output/results.json").exists()
    assert (target / "remote-note.txt").exists()
    assert not (target / ".aws").exists()
    assert os.readlink(target / ".venv-shared") == "/cluster/shared/environment"
    assert not (target / "examples/sbtab").is_symlink()
    second = upload(source, target, env)
    assert "Upload verified" in second.stdout
    assert "*deleting" not in second.stdout


def test_sync_preview_makes_no_changes_and_missing_config_fails(sync_tree):
    source, target, env = sync_tree
    put(source, "sbtab/new.py", "new source\n")
    put(target, "sbtab/old.py", "old source\n")
    before = snapshot(target)
    preview = upload(source, target, env, "--dry-run")
    assert "Preview only" in preview.stdout
    assert "sbtab/new.py" in preview.stdout and "sbtab/old.py" in preview.stdout
    assert snapshot(target) == before
    (source / "scripts/slurm/cluster.local.sh").unlink()
    with pytest.raises(subprocess.CalledProcessError) as error:
        upload(source, target, env)
    assert "Missing scripts/slurm/cluster.local.sh" in error.value.stderr
    assert snapshot(target) == before


@pytest.mark.parametrize("destination", ["test-cluster:/", "test-cluster:/tmp/../home",
                                         "test-cluster:/tmp/./project", "test-cluster:/tmp//project",
                                         "test-cluster:/tmp/project;touch_bad", "test-cluster:relative",
                                         "-host:/tmp/project", "test-cluster:/path with spaces"])
def test_sync_rejects_ambiguous_destinations(destination):
    script = Path(__file__).resolve().parents[2] / "scripts/sync_cluster.sh"
    result = subprocess.run(["bash", str(script), destination], capture_output=True, text=True)
    assert result.returncode == 2
