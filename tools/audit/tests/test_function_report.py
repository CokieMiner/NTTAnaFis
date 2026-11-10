"""Graph evidence exports and advisory-versus-failing CLI behavior."""

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from tools.audit.cli import main
from tools.audit.function_graph import FunctionGraph
from tools.audit.function_report import write_function_report
from tools.audit.function_rules import function_reviews
from tools.audit.structure_rules import run_structure_audit


class FunctionReportTests(unittest.TestCase):
    def test_graph_exports_preserve_calls_values_uncertainty_and_source_metadata(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "src").mkdir()
            (root / "src/lib.rs").write_text("fn helper() {} fn driver() { helper(); opaque!(helper); } const PTR: fn() = helper;")
            graph = FunctionGraph(root)
            reviews = function_reviews(graph)
            paths = write_function_report(graph, reviews, root / "target/report")
            self.assertEqual(set(paths), {"json", "dot"})
            self.assertEqual({path.name for path in (root / "target/report").iterdir()}, {"functions.json", "functions.dot"})
            payload = json.loads(Path(paths["json"]).read_text())
            self.assertEqual(payload["summary"]["resolved_calls"], 1)
            self.assertEqual(len(payload["sources"]["src/lib.rs"]), 64)
            self.assertTrue(any(r["kind"] == "reference" for r in payload["references"]))
            self.assertTrue(payload["possible_references"])
            self.assertIn("callee_precedes_caller", payload["review_counts"])
            dot = Path(paths["dot"]).read_text()
            self.assertIn("style=solid", dot)
            self.assertIn("style=dotted", dot)

    def test_both_cli_entry_points_include_reviews_and_only_fail_when_requested(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "src").mkdir()
            (root / "src/lib.rs").write_text("fn helper() {} fn driver() { helper(); }")
            graph = FunctionGraph(root)
            reviews = function_reviews(graph)
            for entry, module in ((run_structure_audit, "tools.audit.structure_rules"), (main, "tools.audit.cli")):
                for deny, expected in ((False, 0), (True, 1)):
                    with self.subTest(entry=module, deny=deny):
                        output = io.StringIO()
                        with patch(f"{module}.collect_structure_findings", return_value=([], [])), \
                             patch(f"{module}.FunctionGraph", return_value=graph), \
                             patch("tools.audit.cli.collect_findings", return_value=([], [], set(), [])), \
                             patch("tools.audit.cli.tool_tree_findings", return_value=[]), contextlib.redirect_stdout(output):
                            args = ["--json", "--function-report", str(root / "target/report")]
                            if deny:
                                args.append("--deny-function-reviews")
                            self.assertEqual(entry(args), expected)
                        payload = json.loads(output.getvalue())
                        self.assertEqual(len(payload["function_reviews"]), len(reviews))
                        self.assertEqual(set(payload["function_report"]), {"json", "dot"})
                        self.assertTrue(Path(payload["function_report"]["dot"]).is_file())

    def test_generated_reports_cannot_replace_tracked_source_through_directory_or_file_symlinks(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "src").mkdir()
            (root / "src/lib.rs").write_text("fn entry() {}")
            graph = FunctionGraph(root)
            reviews = function_reviews(graph)
            for directory in (root, root / "src", root / "tools/report"):
                with self.subTest(directory=directory), self.assertRaisesRegex(ValueError, "target/"):
                    write_function_report(graph, reviews, directory)
            output = root / "target/report"
            output.mkdir(parents=True)
            (root / "target/alias").symlink_to(root / "src", target_is_directory=True)
            with self.assertRaisesRegex(ValueError, "target/"):
                write_function_report(graph, reviews, root / "target/alias")
            (output / "functions.json").symlink_to(root / "src/lib.rs")
            with self.assertRaisesRegex(ValueError, "output directory"):
                write_function_report(graph, reviews, output)
            self.assertEqual((root / "src/lib.rs").read_text(), "fn entry() {}")


if __name__ == "__main__":
    unittest.main()
