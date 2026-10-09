"""Cluster text I/O must work even after a library switches the process to C locale."""
import os
from pathlib import Path
import subprocess
import sys


def test_utf8_protocol_and_artifacts_under_ascii_locale(tmp_path):
    code = r'''
import json
import locale
from pathlib import Path
import sys
from sbtab.experiments.experiment_common import (load_protocol, load_yaml, read_json, write_json,
                                                atomic_write_text, append_jsonl)
from sbtab.data.registry import load_dataset_config

assert sys.flags.utf8_mode == 0
locale.setlocale(locale.LC_CTYPE, "C")
root = Path(sys.argv[1])
protocol = Path("configs/protocols/sbtab_8515_hpo100_cv5_v2.yaml")
try:
    with protocol.open() as f:
        f.read()
except UnicodeDecodeError as error:
    assert error.start == 415
else:
    raise AssertionError("test did not reproduce the cluster ASCII locale")
assert load_protocol()["tuning"]["n_trials"] == 100
value = {"label": "\u0414\u0430\u043d\u043d\u044b\u0435 \u2014 caf\u00e9"}
atomic_write_text(root / "example.yaml", "label: " + value["label"] + "\n")
assert load_yaml(root / "example.yaml") == value
atomic_write_text(root / "example.json", json.dumps(value, ensure_ascii=False))
assert read_json(root / "example.json") == value
write_json(root / "written.json", value)
assert read_json(root / "written.json") == value
append_jsonl(root / "events.jsonl", value)
assert json.loads((root / "events.jsonl").read_text(encoding="utf-8")) == value
atomic_write_text(root / "toy.yaml", "name: toy\nnotes: " + value["label"] + "\n")
assert load_dataset_config("toy", root)["notes"] == value["label"]
'''
    env = {**os.environ, "LC_ALL": "C", "LANG": "C", "PYTHONUTF8": "0", "PYTHONCOERCECLOCALE": "0"}
    result = subprocess.run([sys.executable, "-c", code, str(tmp_path)],
                            cwd=Path(__file__).resolve().parents[2], env=env, capture_output=True)
    assert result.returncode == 0, result.stderr.decode("utf-8", errors="replace")
