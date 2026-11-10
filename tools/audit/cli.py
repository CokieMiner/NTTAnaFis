"""Run source audits and optional Python regression suites from one entry point."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from collections import Counter
from pathlib import Path

from .common import ROOT
from .function_graph import FunctionGraph
from .function_report import write_function_report
from .function_rules import function_reviews
from .import_rules import collect_findings
from .structure_rules import collect_structure_findings
from .tool_checks import tool_tree_findings


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true", help="Print one structured audit result")
    parser.add_argument("--tests", action="store_true", help="Run the audit Python regression suite")
    parser.add_argument("--deny-oversized", action="store_true", help="Promote the Rust 500-line cohesion target to an error")
    parser.add_argument("--function-report", type=Path, help="Export the function graph as DOT and review evidence as JSON")
    parser.add_argument("--deny-function-reviews", action="store_true", help="Fail on advisory function visibility, ordering and ownership candidates")
    args = parser.parse_args(argv)
    try:
        structural, oversized = collect_structure_findings(deny_oversized=args.deny_oversized)
        imports, _, _, _ = collect_findings()
        findings = [*structural, *imports, *tool_tree_findings()]
        graph = FunctionGraph()
        reviews = function_reviews(graph)
        report = write_function_report(graph, reviews, args.function_report) if args.function_report else {}
        suites = run_test_suites() if args.tests else {}
    except (OSError, ValueError) as error:
        print(f"audit: {error}", file=sys.stderr)
        return 2
    payload = {
        "findings": [finding.__dict__ for finding in findings],
        "oversized_files": [item.__dict__ for item in oversized],
        "test_suites": suites,
        "function_analysis": graph.summary(),
        "function_reviews": [r.__dict__ for r in reviews],
        "function_report": report,
    }
    if args.json:
        print(json.dumps(payload, indent=2))
    else:
        print(f"Project findings: {len(findings)}")
        print(f"Rust files over the cohesion target: {len(oversized)}")
        print(f"Function review candidates: {len(reviews)} ({graph.summary()['functions']} declarations)")
        for kind, count in sorted(Counter(r.kind for r in reviews).items()):
            print(f"  {kind}: {count}")
        for finding in findings:
            print(f"  {finding.path}:{finding.line} [{finding.kind}] {finding.detail}")
        for name, code in suites.items():
            print(f"  {name}: {'passed' if code == 0 else 'failed'}")
        if report:
            print(f"Function graph: {report['dot']}")
    return int(bool(findings) or any(suites.values()) or args.deny_function_reviews and bool(reviews))


def run_test_suites() -> dict[str, int]:
    """Use a fresh interpreter and the documented tools import root."""
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT / "tools") + os.pathsep + str(ROOT)
    results = {}
    for package in ("audit",):
        tests = ROOT / "tools" / package / "tests"
        if not tests.is_dir():
            print(f"missing regression suite: {tests}", file=sys.stderr)
            results[package] = 1
            continue
        print(f"Running {package} tests...", file=sys.stderr, flush=True)
        try:
            result = subprocess.run(
                [sys.executable, "-m", "unittest", "discover", "-s", str(tests)],
                cwd=ROOT, env=env, capture_output=True, text=True, timeout=300,
            )
        except subprocess.TimeoutExpired:
            print(f"{package} tests exceeded 300 seconds", file=sys.stderr)
            results[package] = 1
            continue
        print(result.stdout + result.stderr, file=sys.stderr, end="")
        results[package] = result.returncode
    return results
