"""Python facade and test-layout controls for maintained tool sources."""

import contextlib
import io
import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from tools.audit.cli import main, run_test_suites
from tools.audit.common import Finding
from tools.audit.tool_checks import tool_source_findings, tool_tree_findings


class ToolTests(unittest.TestCase):
    def test_facades_tests_and_comment_decoys(self):
        for text, path, expected in (
            ('"""Facade."""\nfrom .entry import run\n__all__ = ["run"]', "tools/example/__init__.py", set()),
            ("def run(): return 0", "tools/example/__init__.py", {"implementation_in_python_facade"}),
            ("CACHE = {}", "tools/example/__init__.py", {"implementation_in_python_facade"}),
            ("__all__ = make_exports()", "tools/example/__init__.py", {"implementation_in_python_facade"}),
            ("from .entry import *", "tools/example/cli.py", {"python_wildcard_import"}),
            ("if enabled:\n    from .entry import *", "tools/example/cli.py", {"python_wildcard_import"}),
            ("def test_case(): pass", "tools/example/entry.py", {"python_test_in_production"}),
            ("class Cases(unittest.TestCase): pass", "tools/example/entry.py", {"python_test_in_production"}),
            ("class Cases(unittest.TestCase): pass", "tools/example/tests/test_entry.py", set()),
            ('"""def test_fake(): pass"""\n# def test_decoy(): pass', "tools/example/entry.py", set()),
            ("def broken(", "tools/example/entry.py", {"invalid_python_source"}),
        ):
            with self.subTest(text=text, path=path):
                self.assertEqual({finding.kind for finding in tool_source_findings(text, path)}, expected)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for relative in ("tools/example/entry.py", "tools/audit/data/capture.py", "tools/audit/__pycache__/cache.py"):
                path = root / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("def test_case(): pass")
            self.assertEqual([finding.path for finding in tool_tree_findings(root)], ["tools/example/entry.py"])

    def test_combined_json_keeps_findings_and_suite_failures(self):
        for findings, suites, expected in (([], {}, 0), ([Finding("fixture", "source.rs", 3, "detail")], {}, 1), ([], {"audit": 1}, 1)):
            output = io.StringIO()
            with patch("tools.audit.cli.collect_structure_findings", return_value=(findings, [])), \
                 patch("tools.audit.cli.collect_findings", return_value=([], [], set(), [])), \
                 patch("tools.audit.cli.tool_tree_findings", return_value=[]), \
                 patch("tools.audit.cli.FunctionGraph") as graph, \
                 patch("tools.audit.cli.function_reviews", return_value=[]), \
                 patch("tools.audit.cli.run_test_suites", return_value=suites), contextlib.redirect_stdout(output):
                graph.return_value.summary.return_value = {"functions": 0}
                self.assertEqual(main(["--json", "--tests"]), expected)
            self.assertEqual(json.loads(output.getvalue())["test_suites"], suites)

    def test_suite_execution_keeps_import_roots_and_cannot_pass_missing_or_timed_out_tests(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            tests = root / "tools" / "audit" / "tests"
            tests.mkdir(parents=True)
            passed = subprocess.CompletedProcess([], 0, "", "passed")
            with patch("tools.audit.cli.ROOT", root), \
                 patch("tools.audit.cli.subprocess.run", side_effect=[passed, subprocess.TimeoutExpired([], 300)]) as execution, \
                 contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(run_test_suites(), {"audit": 0})
                self.assertEqual(run_test_suites(), {"audit": 1})
                tests.rmdir()
                self.assertEqual(run_test_suites(), {"audit": 1})
            self.assertEqual(execution.call_count, 2)
            for invocation in execution.call_args_list:
                self.assertEqual(invocation.kwargs["cwd"], root)
                self.assertEqual(invocation.kwargs["env"]["PYTHONPATH"], str(root / "tools") + os.pathsep + str(root))


if __name__ == "__main__":
    unittest.main()
