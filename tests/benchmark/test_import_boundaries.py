"""Dependency-direction tests for the greenfield benchmark core."""

from __future__ import annotations

import ast
import importlib.util
import unittest
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
BENCHMARK_ROOT = REPOSITORY_ROOT / "sbtab" / "benchmark"
NATIVE_ROOTS = tuple(
    REPOSITORY_ROOT / "sbtab" / name for name in ("bridge", "models", "solvers")
)
LEGACY_PREFIXES = (
    "sbtab.data",
    "sbtab.transforms",
    "sbtab.experiments",
)
UPWARD_DEPENDENCY_PREFIXES = (
    "sbtab.benchmark",
    "sbtab.evaluation",
)


def _has_prefix(module: str, prefixes: tuple[str, ...]) -> bool:
    return any(
        module == prefix or module.startswith(f"{prefix}.")
        for prefix in prefixes
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
    # ``from sbtab import data`` names ``sbtab`` in ``node.module`` even though
    # the imported object may be the forbidden ``sbtab.data`` submodule. AST
    # cannot distinguish a submodule from an attribute, so inspect both the
    # base and each qualified imported name. False positives are acceptable at
    # this architecture boundary: an explicit import can then be reviewed.
    imported = [base] if base else []
    imported.extend(
        f"{base}.{alias.name}" if base else alias.name
        for alias in node.names
        if alias.name != "*"
    )
    return imported


def _find_forbidden_imports(
    roots: tuple[Path, ...],
    forbidden_prefixes: tuple[str, ...],
) -> list[str]:
    """Return source locations whose imports violate a dependency boundary."""

    violations: list[str] = []
    for root in roots:
        for path in sorted(root.rglob("*.py")):
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            package = ".".join(path.parent.relative_to(REPOSITORY_ROOT).parts)
            for node in ast.walk(tree):
                if not isinstance(node, (ast.Import, ast.ImportFrom)):
                    continue
                for module in _imported_modules(node, package):
                    if _has_prefix(module, forbidden_prefixes):
                        relative_path = path.relative_to(REPOSITORY_ROOT)
                        violations.append(
                            f"{relative_path}:{node.lineno}: {module}"
                        )
    return violations


class BenchmarkImportBoundaryTests(unittest.TestCase):
    """Keep legacy orchestration out of the new runtime dependency graph."""

    def test_benchmark_modules_do_not_import_legacy_orchestration(self) -> None:
        self.assertEqual(
            _find_forbidden_imports((BENCHMARK_ROOT,), LEGACY_PREFIXES),
            [],
        )

    def test_native_layers_do_not_import_benchmark_or_evaluation(self) -> None:
        self.assertEqual(
            _find_forbidden_imports(NATIVE_ROOTS, UPWARD_DEPENDENCY_PREFIXES),
            [],
        )

    def test_relative_legacy_import_is_resolved_before_checking(self) -> None:
        tree = ast.parse("from ..data import DataModule")
        node = tree.body[0]

        self.assertIsInstance(node, ast.ImportFrom)
        modules = _imported_modules(node, "sbtab.benchmark")  # type: ignore[arg-type]
        self.assertIn("sbtab.data", modules)
        self.assertTrue(any(_has_prefix(module, LEGACY_PREFIXES) for module in modules))

    def test_absolute_from_import_cannot_hide_forbidden_submodule(self) -> None:
        tree = ast.parse("from sbtab import data")
        node = tree.body[0]

        self.assertIsInstance(node, ast.ImportFrom)
        modules = _imported_modules(node, "sbtab.benchmark")  # type: ignore[arg-type]
        self.assertIn("sbtab.data", modules)
        self.assertTrue(any(_has_prefix(module, LEGACY_PREFIXES) for module in modules))


if __name__ == "__main__":
    unittest.main()
