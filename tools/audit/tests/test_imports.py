"""Import grouping, facade boundaries and permitted test dependencies."""

import unittest
from pathlib import Path

from tools.audit.common import ROOT
from tools.audit.import_checks import comparison_dependency_findings, import_structure_findings
from tools.audit.import_parser import (
    EXTERNAL_CRATE_ROOTS,
    LIBRARY_CRATE,
    import_sort_key,
    module_name,
    project_import_violation,
    resolve_relative_path,
)
from tools.audit.import_rules import FileAnalyzer, collect_findings
from tools.audit.items import top_level_items


class ImportTests(unittest.TestCase):
    def test_import_structure_covers_order_duplicates_cfgs_and_parent_forwarding(self):
        for text, parent, kind, expected in (
            ("use core::ops::{Add, Sub};\nuse core::fmt::{Debug, Display};", None, "grouped_import_path_order", True),
            ("use super::Z;\nuse super::{A, B};", None, "grouped_import_path_order", False),
            ("use core::{fmt::Debug, fmt::Debug};", None, "duplicate_import_in_tree", True),
            ("use core::fmt::Debug;\nuse core::fmt::Debug;", None, "duplicate_import", True),
            ('#[cfg(feature="a")] use core::fmt::Debug;\n#[cfg(not(feature="a"))] use core::fmt::Debug;', None, "duplicate_import", False),
            ('#[cfg(target_arch="arm")] use core::fmt::Debug;\n#[cfg(target_arch="x86")] use core::fmt::Debug;', None, "duplicate_import", False),
            ("use super::Thing;", "mod child;\nuse child::Thing;", "private_sibling_prelude", True),
            ("use super::Thing;", "mod child;\npub use child::Thing;", "private_sibling_prelude", False),
            ("use core::fmt::Debug; use core::fmt::Display;", None, "multiple_imports_on_line", True),
        ):
            with self.subTest(text=text, parent=parent):
                kinds = {finding.kind for finding in import_structure_findings(text, "src/demo/value.rs", parent)}
                self.assertEqual(kind in kinds, expected)

    def test_keywords_sort_at_each_path_segment(self):
        for first, second in (("super::super::Thing", "super::A"), ("self::item2", "self::item10"),
                              ("super::item4", "super::item4_more")):
            with self.subTest(first=first, second=second):
                self.assertLess(import_sort_key(first), import_sort_key(second))

    def test_sibling_file_does_not_bypass_the_parent_facade(self):
        self.assertEqual(project_import_violation(
            "super::types::Value", source_path="src/crt/coefficient.rs",
            source_mod="src::crt::coefficient", plumbing_file=False,
        ), "non_parent_implementation_import")

    def test_test_imports_bypass_boundaries_but_keep_group_rules(self):
        analyzer = FileAnalyzer(ROOT / "src/tests/mod.rs", set())
        analyzer.raw_text = "use crate::arch::NttFloatKernels;\n\nuse core::fmt::Debug;"
        analyzer.raw_lines = analyzer.raw_text.splitlines()
        analyzer.lines = analyzer.code_lines = analyzer.raw_lines
        for line, text in enumerate(analyzer.lines, 1):
            analyzer._handle_line(text, line)
        analyzer._check_import_order()
        kinds = {kind for kind, _path, _line, _detail in analyzer.finding_keys}
        self.assertIn("import_group_order", kinds)
        self.assertNotIn("deep_project_import", kinds)
        self.assertNotIn("non_parent_implementation_import", kinds)

    def test_comparators_are_restricted_to_benchmarks_and_tests(self):
        for source in ("use mp_anafis::Integer;", "use num_bigint::BigUint;",
                       "fn f() { mp_anafis::Integer::new(); }", "use mp_anafis as oracle;",
                       "use {::mp_anafis as oracle};", "extern crate mp_anafis as oracle;"):
            for path in ("src/demo/tests.rs", "src/multiply/primes.rs"):
                with self.subTest(source=source, path=path):
                    findings = comparison_dependency_findings(source, path)
                    expected = 0 if "num_bigint" in source else 1
                    self.assertEqual(len(findings), expected)
                    if expected:
                        self.assertEqual(findings[0].kind, "comparison_dependency_outside_bench_or_test")
            self.assertEqual(comparison_dependency_findings(source, "benches/internal_improvement/main.rs"), [])
            self.assertEqual(comparison_dependency_findings(source, "tests/correctness/compare.rs"), [])
        self.assertEqual(comparison_dependency_findings('// mp_anafis::Integer\nconst X: &str = "mp_anafis::Integer";', "src/demo.rs"), [])
        self.assertEqual(comparison_dependency_findings("use num_traits::Zero; use rayon::ThreadPool;", "benches/internal_improvement/main.rs"), [])

    def test_use_tree_does_not_end_at_its_brace(self):
        items = top_level_items("use core::{fmt::{Debug, Display}, ops::Add};\nfn value() {}")
        self.assertEqual([item.kind for item in items], ["use", "fn"])

    def test_paths_resolve_within_the_library_crate(self):
        self.assertEqual(LIBRARY_CRATE, "ntt_anafis")
        self.assertEqual(module_name(ROOT / "src/lib.rs"), "src")
        self.assertEqual(module_name(ROOT / "src/crt/mod.rs"), "src::crt")
        self.assertEqual(
            resolve_relative_path("src::multiply::primes", "crate::arch::NttFloatKernels"),
            "src::arch::NttFloatKernels",
        )
        self.assertEqual(resolve_relative_path("src::multiply::schedule", "super::Ntt"), "src::multiply::Ntt")
        self.assertIn("core_anafis", EXTERNAL_CRATE_ROOTS)
        self.assertIn("ntt_anafis", EXTERNAL_CRATE_ROOTS)

    def test_multiline_imports_preserve_header_group_checks(self):
        analyzer = FileAnalyzer(ROOT / "src/multiply/schedule.rs", set())
        analyzer.raw_lines = ["use crate::{", "    arch::NttFloatKernels,", "};", "", "use core::hint::black_box;"]
        analyzer.lines = analyzer.raw_lines
        analyzer.code_lines = analyzer.raw_lines
        for line, text in enumerate(analyzer.lines, start=1):
            analyzer._handle_line(text, line)
        analyzer._check_import_order()
        self.assertEqual(len(analyzer.private_imports), 2)
        self.assertIn("import_group_order", {kind for kind, _path, _line, _detail in analyzer.finding_keys})

    def test_imports_follow_subsystem_and_header_rules(self):
        for target, expected in (
            ("crate::arch::NttFloatKernels", None),
            ("crate::multiply::Multiplication", "non_parent_implementation_import"),
            ("crate::Cofactors", None),
            ("crate::workers::Worker", "deep_project_import"),
            ("super::Ntt", None),
            ("super::execute::Multiplication", "non_parent_implementation_import"),
        ):
            with self.subTest(target=target):
                self.assertEqual(
                    project_import_violation(
                        target,
                        source_path="src/multiply/schedule.rs",
                        source_mod="src::multiply::schedule",
                        plumbing_file=False,
                    ),
                    expected,
                )
        analyzer = FileAnalyzer(ROOT / "src/multiply/schedule.rs", set())
        analyzer.lines = ["fn entry() {", "    use std::env;", "    crate::arch::NttFloatKernels::run();", "}"]
        analyzer.code_lines = analyzer.lines
        for line, text in enumerate(analyzer.lines, start=1):
            analyzer._handle_line(text, line)
        self.assertEqual(
            {kind for kind, _path, _line, _detail in analyzer.finding_keys},
            {"use_not_at_top", "inline_qualified_path"},
        )

    def test_repository_import_audit_includes_sources_and_tests(self):
        _findings, edges, boundaries, _aliases = collect_findings()
        paths = {Path(edge.path) for edge in edges}
        self.assertIn(Path("src/multiply/schedule.rs"), paths)
        self.assertIn(Path("src/tests/plans.rs"), paths)
        self.assertIn("src::multiply", boundaries)
        self.assertIn("src::arch", boundaries)


if __name__ == "__main__":
    unittest.main()
