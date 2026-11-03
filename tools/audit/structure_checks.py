"""Structural rules that require item boundaries rather than line regexes."""

from __future__ import annotations

import re

from .common import Finding
from .items import attributes, cfg_requires_test, top_level_items
from .lint_attributes import lint_findings
from .rust_source import clean_rust_code


def structural_findings(text: str, path: str, *, registry: bool, test_file: bool) -> list[Finding]:
    findings = []
    def add(kind: str, offset: int, detail: str) -> None:
        findings.append(Finding(kind, path, text.count("\n", 0, offset) + 1, detail))

    try:
        attrs = attributes(text)
        items = top_level_items(text)
        findings.extend(lint_findings(text, path))
    except ValueError as error:
        add("unparsed_rust_structure", 0, str(error))
        return findings
    for attribute in attrs:
        if not test_file and re.fullmatch(r"(?:test|(?:\w+::)+test)(?:\s*\(.*\))?", attribute.code, re.DOTALL):
            add("test_in_production_file", attribute.start, "test functions belong in a dedicated tests.rs or tests/ directory")
    if registry:
        stage = 0
        for item in items:
            is_test_module = item.kind == "mod" and (item.name == "tests" or item.name.endswith("_tests"))
            test_gate = any(cfg_requires_test(attribute.code) for attribute in item.attributes)
            if test_gate and not (is_test_module and item.name == "tests"):
                add("test_item_in_module_registry", item.start,
                    "the only test-specific registry item is the final #[cfg(test)] mod tests; declaration")
            if is_test_module:
                if not any(cfg_requires_test(attribute.code) for attribute in item.attributes):
                    add("test_module_without_test_gate", item.start, "test module must be disabled outside cfg(test)")
                current = 3
            elif item.kind == "use":
                current = 2 if re.match(r"pub\b", item.header) else 0
            elif item.kind in {"mod", "extern crate"}:
                current = 1
            else:
                add("implementation_in_module_registry", item.start, item.header)
                continue
            if current < stage:
                add("module_registry_order", item.start, "required order: parent imports, modules, reexports, test modules")
            stage = max(stage, current)
            if item.kind == "mod" and not text[item.start:item.end].rstrip().endswith(";"):
                add("inline_module_in_registry", item.start, "module registries contain external module declarations only")
    if not test_file:
        cleaned = clean_rust_code(text, scrub_attributes=False)
        if not path.startswith("src/arch/"):
            for attribute in attrs:
                if re.search(r"\btarget_(?:arch|feature)\b", attribute.code):
                    add("architecture_selection_outside_arch", attribute.start,
                        "generic modules delegate ISA and CPU-feature selection to src/arch")
            for match in re.finditer(r"\bis_[A-Za-z0-9_]+_feature_detected\s*!", cleaned):
                add("architecture_selection_outside_arch", match.start(),
                    "CPU-feature detection belongs in src/arch")
        for item in items:
            if item.kind == "mod" and item.name == "tests" and not registry:
                add("test_module_outside_registry", item.start, "declare tests only as the final item of mod.rs")
        # Include test modules with nonstandard names and multiline attributes.
        for item in items:
            if item.kind == "mod" and any(cfg_requires_test(attribute.code) for attribute in item.attributes):
                if cleaned[item.start:item.end].rstrip().endswith("}"):
                    add("inline_test_module", item.start, "test implementations must be in separate files")
    return findings
