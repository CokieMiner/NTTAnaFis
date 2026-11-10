"""Audits Rust source-file structure, visibility gates, module registry purity, and production completeness."""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from .common import Finding, ROOT
from .function_graph import FunctionGraph
from .function_report import write_function_report
from .function_rules import function_reviews
from .import_parser import is_test_path
from .rust_source import clean_rust_code
from .sources import rust_source_paths
from .structure_checks import structural_findings
from .visibility import VisibilityGraph, visibility_findings

# The 500-line cohesion target is advisory; architecture backends are exempt.
COHESION_LINE_LIMIT = 500
ARCHITECTURE_ROOT = "src/arch/"
FORBIDDEN_VISIBILITY_RE = re.compile(r"\bpub\s*\(\s*(?:super\s*\)|in\b[^)]*\))")
PLACEHOLDER_RE = re.compile(r"\b(?:todo|unimplemented)\s*!\s*\(")
PRIVATE_LIB_USE_RE = re.compile(r"^\s*use\s+")


@dataclass(frozen=True)
class SizeReview:
    path: str
    lines: int


def line_number(text: str, offset: int) -> int:
    return text.count("\n", 0, offset) + 1


def run_structure_audit(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Audit Rust source-file structure, visibility constraints, module registry purity, and production completeness.")
    parser.add_argument("--json", action="store_true", help="Print findings as JSON.")
    parser.add_argument("--deny-oversized", action="store_true", help="Treat the 500-line cohesion target as an error (architecture backends exempt).")
    parser.add_argument("--function-report", type=Path, help="Write the function graph as DOT and review evidence as JSON in this directory.")
    parser.add_argument("--deny-function-reviews", action="store_true", help="Fail on advisory function ownership, ordering and visibility candidates.")
    args = parser.parse_args(argv)

    try:
        findings, oversized_files = collect_structure_findings(deny_oversized=args.deny_oversized)
        graph = FunctionGraph()
        reviews = function_reviews(graph)
        report = write_function_report(graph, reviews, args.function_report) if args.function_report else {}
    except (OSError, ValueError) as error:
        print(f"structure audit: {error}", file=sys.stderr)
        return 2

    if args.json:
        payload = {
            "findings": [f.__dict__ for f in findings],
            "oversized_files": [o.__dict__ for o in oversized_files],
            "function_analysis": graph.summary(),
            "function_reviews": [r.__dict__ for r in reviews],
            "function_report": report,
        }
        print(json.dumps(payload, indent=2))
    else:
        print(f"Structure findings: {len(findings)}")
        print(f"Files > 500 lines (architecture backends exempt): {len(oversized_files)}")
        print(f"Function review candidates: {len(reviews)} ({graph.summary()['functions']} declarations)")
        for kind, count in sorted(Counter(r.kind for r in reviews).items()):
            print(f"  {kind}: {count}")
        for finding in findings:
            print(f"  {finding.path}:{finding.line} [{finding.kind}] {finding.detail}")
        if report:
            print(f"Function graph: {report['dot']}")
    return int(bool(findings) or args.deny_function_reviews and bool(reviews))


def collect_structure_findings(*, root=ROOT, deny_oversized: bool = False) -> tuple[list[Finding], list[SizeReview]]:
    """Inspect maintained sources; test logic is exempt from registry purity and size targets."""
    findings: List[Finding] = []
    oversized_files: List[SizeReview] = []
    visibility = VisibilityGraph(root)
    reachable = visibility.reachable_types()

    for path in rust_source_paths(root):
        is_benchmark = "benches" in path.relative_to(root).parts
        rel = str(path.relative_to(root)).replace("\\", "/")
        text = path.read_text(encoding="utf-8")
        cleaned = clean_rust_code(text, scrub_attributes=False)
        findings.extend(visibility_findings(text, rel, visibility, reachable))

        for match in FORBIDDEN_VISIBILITY_RE.finditer(cleaned):
            findings.append(Finding("forbidden_visibility", rel, line_number(cleaned, match.start()), match.group(0)))

        if is_benchmark or not is_test_path(path):
            for match in PLACEHOLDER_RE.finditer(cleaned):
                findings.append(Finding("placeholder_in_production", rel, line_number(cleaned, match.start()), match.group(0)))

        findings.extend(structural_findings(text, rel,
                                            registry=path.name in {"mod.rs", "lib.rs"} and (is_benchmark or not is_test_path(path)),
                                            test_file=is_test_path(path)))
        if path.name == "lib.rs":
            for number, line in enumerate(cleaned.splitlines(), 1):
                if PRIVATE_LIB_USE_RE.match(line):
                    findings.append(Finding("private_import_in_library_facade", rel, number, line.strip()))

        line_count = len(text.splitlines())
        if (line_count > COHESION_LINE_LIMIT and (is_benchmark or not is_test_path(path))
                and not rel.startswith(ARCHITECTURE_ROOT)):
            oversized_files.append(SizeReview(path=rel, lines=line_count))
            if deny_oversized:
                findings.append(Finding("oversized_source_file", rel, 1, f"{line_count} lines exceeds the 500-line cohesion target"))

    return findings, oversized_files
