"""Inventory and validate Rust lint allowances and expectations."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple

from .common import Finding, ROOT
from .lint_attributes import lint_attributes, lint_findings
from .sources import rust_source_paths
from .test_regions import find_test_ranges, is_whole_file_test


@dataclass(frozen=True)
class AllowEntry:
    kind: str
    path: str
    line: int
    lints: Tuple[str, ...]
    in_test: bool


def extract_allows_expects(
    text: str,
    rel_path: str,
    test_ranges: List[Tuple[int, int]],
    whole_file_test: bool,
    *,
    kinds: Tuple[str, ...] = ("allow", "expect"),
) -> List[AllowEntry]:
    entries: List[AllowEntry] = []
    for attribute in lint_attributes(text):
        attr_kind = attribute.kind
        if attr_kind not in kinds:
            continue
        lints = attribute.lints
        if not lints:
            continue
        start = attribute.start
        if attribute.inner:
            kind = f"inner #![{attr_kind}]"
        else:
            kind = f"outer #[{attr_kind}]"
        entries.append(
            AllowEntry(
                kind=kind,
                path=rel_path,
                line=text.count("\n", 0, start) + 1,
                lints=lints,
                in_test=(
                    whole_file_test
                    or any(range_start <= start < range_end for range_start, range_end in test_ranges)
                ),
            )
        )
    return entries


def select_lints(
    entries: List[AllowEntry],
    *,
    lint: Optional[str],
    mode: str,
    production_only: bool,
    test_only: bool,
) -> List[AllowEntry]:
    wanted = None
    if lint is not None:
        wanted = {lint}
        if "::" not in lint:
            wanted.add(f"clippy::{lint}")

    selected: List[AllowEntry] = []
    for entry in entries:
        if production_only and entry.in_test:
            continue
        if test_only and not entry.in_test:
            continue
        lints = entry.lints
        if mode == "clippy":
            lints = tuple(name for name in lints if name.startswith("clippy::"))
        elif mode == "non_clippy":
            lints = tuple(name for name in lints if not name.startswith("clippy::"))
        if wanted is not None:
            lints = tuple(name for name in lints if name in wanted)
        if lints:
            selected.append(AllowEntry(entry.kind, entry.path, entry.line, lints, entry.in_test))
    return selected


@dataclass
class Summary:
    entries: List[AllowEntry]
    prod_counts: Counter[str]
    test_counts: Counter[str]
    prod_files: Dict[str, Set[str]]
    test_files: Dict[str, Set[str]]
    kind_counts: Counter[str]
    all_lints: List[str]
    total_allow: int
    total_expect: int


def summarize(entries: List[AllowEntry]) -> Summary:
    prod_counts: Counter[str] = Counter()
    test_counts: Counter[str] = Counter()
    prod_files: Dict[str, Set[str]] = defaultdict(set)
    test_files: Dict[str, Set[str]] = defaultdict(set)
    kind_counts: Counter[str] = Counter()

    for entry in entries:
        attr = "expect" if "expect" in entry.kind else "allow"
        kind_counts[attr] += 1
        kind_counts[entry.kind] += 1
        counts = test_counts if entry.in_test else prod_counts
        files = test_files if entry.in_test else prod_files
        for lint in entry.lints:
            counts[lint] += 1
            files[lint].add(entry.path)

    all_lints = sorted(set(prod_counts) | set(test_counts))
    total_allow = sum(1 for e in entries if "allow" in e.kind)
    total_expect = sum(1 for e in entries if "expect" in e.kind)
    return Summary(
        entries=entries,
        total_allow=total_allow,
        total_expect=total_expect,
        prod_counts=prod_counts,
        test_counts=test_counts,
        prod_files=prod_files,
        test_files=test_files,
        kind_counts=kind_counts,
        all_lints=all_lints,
    )


def render_text(summary: Summary, kinds: Tuple[str, ...], mode: str) -> str:
    lines: List[str] = []
    lines.append(f"Inventory ({mode}) — kinds: {', '.join(kinds)}")
    lines.append(f"Entries: {len(summary.entries)}  (allow: {summary.total_allow}, expect: {summary.total_expect}; both require a descriptive reason)")
    lines.append(f"Prod lints: {sum(summary.prod_counts.values())}  Test lints: {sum(summary.test_counts.values())}")
    lines.append("")

    if not summary.all_lints:
        lines.append("No matching entries.")
        return "\n".join(lines)

    for lint in summary.all_lints:
        prod_n = summary.prod_counts.get(lint, 0)
        test_n = summary.test_counts.get(lint, 0)
        files = sorted(summary.prod_files.get(lint, set()) | summary.test_files.get(lint, set()))
        header = f"{lint}  (prod:{prod_n} test:{test_n} files:{len(files)})"
        lines.append(header)
        for path in files:
            only_test = path in summary.test_files.get(lint, set()) and path not in summary.prod_files.get(lint, set())
            suffix = "  [test]" if only_test else ""
            lines.append(f"  | {path}{suffix}")
        lines.append("")

    lines.append("-" * 80)
    lines.append(f"Kind breakdown: {dict(summary.kind_counts)}")
    return "\n".join(lines)


def render_json(summary: Summary, kinds: Tuple[str, ...], mode: str) -> str:
    by_lint = {
        lint: {
            "prod_count": summary.prod_counts.get(lint, 0),
            "test_count": summary.test_counts.get(lint, 0),
            "prod_files": sorted(summary.prod_files.get(lint, set())),
            "test_files": sorted(summary.test_files.get(lint, set())),
        }
        for lint in summary.all_lints
    }
    payload = {
        "mode": mode,
        "kinds": list(kinds),
        "total_entries": len(summary.entries),
        "total_allows": summary.total_allow,
        "total_expects": summary.total_expect,
        "prod_total": sum(summary.prod_counts.values()),
        "test_total": sum(summary.test_counts.values()),
        "kind_counts": dict(summary.kind_counts),
        "by_lint": by_lint,
    }
    return json.dumps(payload, indent=2)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Inventory Rust lint allowances (#[allow]) and expectations (#[expect]).")
    parser.add_argument("--json", action="store_true", help="Output JSON")
    parser.add_argument("--check", action="store_true", help="Validate all scanned allow/expect attributes; fail on missing reasons or dead-code suppression")
    parser.add_argument("--lint", help="Filter to a single lint (e.g. clippy::as_conversions)")
    loc = parser.add_mutually_exclusive_group()
    loc.add_argument("--prod-only", action="store_true", help="Only production library code")
    loc.add_argument("--test-only", action="store_true", help="Only test code")
    lint_mode = parser.add_mutually_exclusive_group()
    lint_mode.add_argument("--non-clippy", action="store_true", help="Only non-Clippy lints")
    lint_mode.add_argument("--all-lints", action="store_true", help="Both Clippy and non-Clippy lints")
    kind_grp = parser.add_mutually_exclusive_group()
    kind_grp.add_argument("--allow-only", action="store_true", help="Only #[allow]")
    kind_grp.add_argument("--expect-only", action="store_true", help="Only #[expect]")
    return parser


def run_check_allows(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    kinds: Tuple[str, ...]
    if args.allow_only:
        kinds = ("allow",)
    elif args.expect_only:
        kinds = ("expect",)
    else:
        kinds = ("allow", "expect")

    entries: List[AllowEntry] = []
    findings: List[Finding] = []
    for path in rust_source_paths(production_only=args.prod_only):
        text = path.read_text(encoding="utf-8")
        rel = str(path.relative_to(ROOT))
        if args.check:
            findings.extend(lint_findings(text, rel))
        entries.extend(
            extract_allows_expects(text, rel, find_test_ranges(text), is_whole_file_test(rel), kinds=kinds)
        )

    mode = "all" if args.all_lints else "non_clippy" if args.non_clippy else "clippy"
    entries = select_lints(entries, lint=args.lint, mode=mode, production_only=args.prod_only, test_only=args.test_only)
    summary = summarize(entries)

    if args.json:
        payload = json.loads(render_json(summary, kinds, mode))
        if args.check:
            payload["findings"] = [finding.__dict__ for finding in findings]
        print(json.dumps(payload, indent=2))
    else:
        print(render_text(summary, kinds, mode))
        if args.check:
            print(f"Lint findings: {len(findings)}")
            for finding in findings:
                print(f"  {finding.path}:{finding.line} [{finding.kind}] {finding.detail}")
    return 1 if findings else 0
