"""Python package facades and test separation for maintained development tools."""

from __future__ import annotations

import ast
from pathlib import Path

from .common import Finding, ROOT


def tool_source_findings(text: str, path: str) -> list[Finding]:
    """Inspect Python syntax and executable items without importing optional backends."""
    try:
        tree = ast.parse(text, filename=path)
    except SyntaxError as error:
        return [Finding("invalid_python_source", path, error.lineno or 1, error.msg)]
    findings = []
    test_file = "tests" in Path(path).parts
    registry = Path(path).name == "__init__.py"
    for node in tree.body:
        if registry and not test_file:
            doc = isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant) and isinstance(node.value.value, str)
            export = isinstance(node, ast.Assign) and all(
                isinstance(target, ast.Name) and target.id == "__all__" for target in node.targets
            ) and isinstance(node.value, (ast.List, ast.Tuple)) and all(
                isinstance(item, ast.Constant) and isinstance(item.value, str) for item in node.value.elts
            )
            if not (doc or export or isinstance(node, (ast.Import, ast.ImportFrom))):
                findings.append(Finding(
                    "implementation_in_python_facade", path, node.lineno,
                    "package facades contain documentation, imports and explicit __all__ exports",
                ))
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and any(alias.name == "*" for alias in node.names):
            findings.append(Finding("python_wildcard_import", path, node.lineno,
                                    "name package dependencies explicitly"))
        if not test_file:
            is_case = isinstance(node, ast.ClassDef) and any(
                ast.unparse(base).rsplit(".", 1)[-1] == "TestCase" for base in node.bases
            )
            if is_case or isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name.startswith("test_"):
                findings.append(Finding("python_test_in_production", path, node.lineno,
                                        "test implementations belong in the owning package's tests/ directory"))
    return findings


def tool_tree_findings(root: Path = ROOT) -> list[Finding]:
    """Exclude generated data captures and bytecode caches."""
    findings = []
    for path in sorted((root / "tools").rglob("*.py")):
        relative = path.relative_to(root)
        if {"data", "__pycache__"}.intersection(relative.parts):
            continue
        findings.extend(tool_source_findings(path.read_text(encoding="utf-8"), relative.as_posix()))
    return findings
