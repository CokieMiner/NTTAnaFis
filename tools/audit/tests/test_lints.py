"""Lint inventories and checks share one parser for reasons and cfg branches."""

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from tools.audit.allow_rules import extract_allows_expects, run_check_allows
from tools.audit.lint_attributes import lint_attributes, lint_findings
from tools.audit.sources import rust_source_paths
from tools.audit.test_regions import find_test_ranges, is_whole_file_test


class LintTests(unittest.TestCase):
    def test_reason_validation_and_inventory_agree(self):
        for kind in ("allow", "expect"):
            for inner in ("#", "#!"):
                for wrapper in ("{}", "cfg_attr(test, {})", "cfg_attr(any(test, feature = \"x\"), {})"):
                    for reason in ('', ', reason = ""', ', reason = "  "', ' /* reason = "decoy" */', ', reason = "bounded ring reduction"', ', reason = r##"comma, and nested (text)"##'):
                        text = f"{inner}[{wrapper.format(f'{kind}(clippy::arithmetic_side_effects{reason})')}] fn f() {{}}"
                        valid = "bounded" in reason or "comma" in reason
                        with self.subTest(text=text):
                            parsed = lint_attributes(text)
                            self.assertEqual(len(parsed), 1)
                            self.assertEqual(parsed[0].lints, ("clippy::arithmetic_side_effects",))
                            self.assertEqual(bool(lint_findings(text, "src/demo.rs")), not valid)
                            entries = extract_allows_expects(text, "src/demo.rs", [], False)
                            self.assertEqual(len(entries), 1)
                            self.assertEqual(entries[0].lints, parsed[0].lints)
        self.assertEqual(lint_attributes('// #[allow(dead_code)]\nconst X: &str = "#[expect(dead_code)]";'), [])

    def test_dead_code_requires_gating_even_with_a_reason(self):
        for kind in ("allow", "expect"):
            findings = lint_findings(f'#[{kind}(dead_code, reason = "other target")] fn f() {{}}', "src/demo.rs")
            self.assertEqual({finding.kind for finding in findings}, {"dead_code_suppression"})

    def test_test_ranges_cover_entire_items_without_classifying_optional_test_gates(self):
        for predicate, expected in (("test", True), ("all(test, feature = \"x\")", True), ("any(test, feature = \"x\")", False)):
            for item in ("fn f<T: Into<u8>>() { let _ = '{'; }", "const VALUE: u8 = 1 << 2;", "use crate::{A, B};"):
                for before in (True, False):
                    allowance = '#[allow(clippy::arithmetic_side_effects, reason = "fixture")]'
                    gate = f"#[cfg({predicate})]"
                    text = f"{allowance if before else gate}\n{gate if before else allowance}\n{item}"
                    with self.subTest(text=text):
                        entries = extract_allows_expects(text, "src/demo.rs", find_test_ranges(text), False)
                        self.assertEqual([entry.in_test for entry in entries], [expected])
        text = 'mod tests { #[allow(clippy::arithmetic_side_effects, reason = "fixture")] fn f() {} }'
        self.assertTrue(extract_allows_expects(text, "src/demo.rs", find_test_ranges(text), False)[0].in_test)
        for path, expected in (("src/tests/ops.rs", True), ("src/tests.rs", True), ("benches/run.rs", True), ("src/api.rs", False)):
            self.assertEqual(is_whole_file_test(path), expected)

    def test_discovery_and_cli_include_source_bench_and_test_sources(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            relatives = ("src/api.rs", "src/tests.rs", "benches/demo.rs", "tests/correctness.rs")
            for relative in relatives:
                path = root / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text('#[cfg_attr(test, allow(clippy::arithmetic_side_effects))] fn f() {}')
            paths = rust_source_paths(root)
            self.assertEqual({path.relative_to(root).as_posix() for path in paths}, set(relatives))
            output = io.StringIO()
            with patch("tools.audit.allow_rules.ROOT", root), patch("tools.audit.allow_rules.rust_source_paths", return_value=paths), contextlib.redirect_stdout(output):
                self.assertEqual(run_check_allows(["--check", "--json"]), 1)
            payload = json.loads(output.getvalue())
            self.assertEqual(len(payload["findings"]), len(relatives))
            self.assertEqual(payload["total_allows"], len(relatives))


if __name__ == "__main__":
    unittest.main()
