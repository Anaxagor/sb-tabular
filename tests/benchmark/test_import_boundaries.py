"""Dependency-direction tests for the greenfield benchmark core."""

from __future__ import annotations

import ast
import importlib.util
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


def _imported_modules(node: ast.Import | ast.ImportFrom, package: str) -> list[str]:
    """Resolve absolute and relative imports without importing their targets."""

    if isinstance(node, ast.Import):
        return [alias.name for alias in node.names]

    if node.level:
        relative_name = "." * node.level + (node.module or "")
        base = importlib.util.resolve_name(relative_name, package)
    else:
        base = node.module or ""
    if node.module:
        return [base]
    return [f"{base}.{alias.name}" for alias in node.names]


class BenchmarkImportBoundaryTests(unittest.TestCase):
    """Keep legacy orchestration out of the new runtime dependency graph."""

    def test_benchmark_modules_do_not_import_legacy_orchestration(self) -> None:
        violations: list[str] = []
        for path in sorted(BENCHMARK_ROOT.rglob("*.py")):
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            relative_parent = path.parent.relative_to(BENCHMARK_ROOT.parent)
            package = ".".join(("sbtab", *relative_parent.parts))
            for node in ast.walk(tree):
                if not isinstance(node, (ast.Import, ast.ImportFrom)):
                    continue
                for module in _imported_modules(node, package):
                    if _is_forbidden(module):
                        violations.append(f"{path}:{node.lineno}: {module}")

        self.assertEqual(violations, [])

    def test_relative_legacy_import_is_resolved_before_checking(self) -> None:
        tree = ast.parse("from ..data import DataModule")
        node = tree.body[0]

        self.assertIsInstance(node, ast.ImportFrom)
        modules = _imported_modules(node, "sbtab.benchmark")  # type: ignore[arg-type]
        self.assertEqual(modules, ["sbtab.data"])
        self.assertTrue(_is_forbidden(modules[0]))


if __name__ == "__main__":
    unittest.main()
