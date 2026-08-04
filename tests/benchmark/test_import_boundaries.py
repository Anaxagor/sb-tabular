"""Dependency-direction tests for the greenfield benchmark core."""

from __future__ import annotations

import ast
import unittest
from pathlib import Path


BENCHMARK_ROOT = Path(__file__).resolve().parents[2] / "sbtab" / "benchmark"
FORBIDDEN_PREFIXES = (
    "sbtab.data",
    "sbtab.transforms",
    "sbtab.experiments",
)


def _is_forbidden(module: str) -> bool:
    return any(
        module == prefix or module.startswith(f"{prefix}.")
        for prefix in FORBIDDEN_PREFIXES
    )


class BenchmarkImportBoundaryTests(unittest.TestCase):
    """Keep legacy orchestration out of the new runtime dependency graph."""

    def test_benchmark_modules_do_not_import_legacy_orchestration(self) -> None:
        violations: list[str] = []
        for path in sorted(BENCHMARK_ROOT.rglob("*.py")):
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom) and node.module:
                    if _is_forbidden(node.module):
                        violations.append(f"{path}:{node.lineno}: {node.module}")
                elif isinstance(node, ast.Import):
                    for alias in node.names:
                        if _is_forbidden(alias.name):
                            violations.append(
                                f"{path}:{node.lineno}: {alias.name}"
                            )

        self.assertEqual(violations, [])


if __name__ == "__main__":
    unittest.main()
