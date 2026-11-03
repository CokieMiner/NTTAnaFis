"""Repository source discovery shared by Rust audits."""

from __future__ import annotations

from pathlib import Path

from .common import ROOT

SOURCE_DIRECTORIES = ("src", "benches", "tests")


def rust_source_paths(root: Path = ROOT, *, production_only: bool = False) -> list[Path]:
    """Find maintained Rust sources without Cargo or report artifacts."""
    directories = ("src",) if production_only else SOURCE_DIRECTORIES
    return sorted(path for relative in directories for path in (root / relative).rglob("*.rs"))
