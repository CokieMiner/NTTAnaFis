"""Additional import checks over complete use statements and direct sibling dependencies."""

from __future__ import annotations

import re

from .common import Finding
from .import_parser import EXTERNAL_CRATE_ROOTS, expand_use_tree, import_sort_key
from .items import top_level_items
from .rust_source import clean_rust_code

COMPARISON_CRATES = {"mp_anafis"}


def comparison_dependency_findings(text: str, path: str) -> list[Finding]:
    """Restricts the production comparator to benchmark and integration sources."""
    if path.startswith(("benches/", "tests/")):
        return []
    cleaned = clean_rust_code(text)
    roots = "|".join(sorted(COMPARISON_CRATES))
    pattern = re.compile(rf"\b({roots})\s*::|\bextern\s+crate\s+({roots})\b")
    locations = {
        (cleaned.count("\n", 0, match.start()) + 1, match.group(1) or match.group(2))
        for match in pattern.finditer(cleaned)
    }
    for declaration in re.finditer(r"\buse\s+([^;]+);", cleaned):
        for target, _alias in expand_use_tree(declaration.group(1)):
            root = target.lstrip(":").split("::", 1)[0]
            if root in COMPARISON_CRATES:
                offset = cleaned.find(root, declaration.start(), declaration.end())
                locations.add((cleaned.count("\n", 0, offset) + 1, root))
    return [
        Finding(
            "comparison_dependency_outside_bench_or_test",
            path,
            line,
            root,
        )
        for line, root in sorted(locations)
    ]


def import_structure_findings(text: str, path: str, parent_text: str | None = None) -> list[Finding]:
    result = []
    def add(kind: str, offset: int, detail: str) -> None:
        result.append(Finding(kind, path, text.count("\n", 0, offset) + 1, detail))
    try:
        items = top_level_items(text)
    except ValueError as error:
        add("unparsed_import_structure", 0, str(error))
        return result
    private_sibling_names = set()
    if parent_text:
        for parent_item in top_level_items(parent_text):
            if parent_item.kind == "use" and parent_item.header.startswith("use "):
                for target, alias in expand_use_tree(parent_item.header[4:]):
                    root = target.split("::", 1)[0]
                    if root not in EXTERNAL_CRATE_ROOTS | {"super", "crate"}:
                        private_sibling_names.add(alias or target.split("::")[-1])
    seen = set()
    previous = None
    for item in items:
        if item.kind != "use":
            continue
        declaration = re.sub(r"^(?:pub\s*(?:\([^)]*\))?\s+)?use\s+", "", item.header)
        entries = expand_use_tree(declaration)
        # Identical imports under different cfg predicates are valid. Duplicate
        # entries within a single use tree are never required by target selection.
        local_seen = set()
        for target, alias in entries:
            identity = (target, alias)
            if identity in local_seen:
                add("duplicate_import_in_tree", item.start, target)
            local_seen.add(identity)
            if target.startswith("super::") and target.count("::") == 1 and target[7:] in private_sibling_names:
                add("private_sibling_prelude", item.start, f"{target} requires an explicit reexport from the parent facade")
        # Literal values distinguish target/feature branches. The masked code
        # deliberately erases those values and cannot be used as their identity.
        cfg = tuple(attribute.raw.strip() for attribute in item.attributes
                    if re.match(r"cfg(?:_attr)?\s*\(", attribute.code))
        public = bool(re.match(r"pub\b", item.header))
        for target, alias in entries:
            identity = (target, alias, cfg, public)
            if identity in seen:
                add("duplicate_import", item.start, target)
            seen.add(identity)
        # Do not let grouped use trees bypass sorting between declarations.
        # cfg-gated alternatives retain their source order when roots coincide.
        # rustfmt puts a single-path declaration before a grouped declaration
        # of the same parent. Compare the declared group prefixes, not their
        # first leaves: `use super::Z; use super::{A, B};` is valid formatting.
        prefix = declaration.split("{", 1)[0].rstrip(":").strip()
        group_root = prefix.split("::", 1)[0]
        if previous is not None:
            old_root, old_prefix, old_public = previous
            same_branch = prefix.startswith(old_prefix + "::") or old_prefix.startswith(prefix + "::")
            if public == old_public and group_root == old_root and prefix != old_prefix and not same_branch:
                if import_sort_key(prefix) < import_sort_key(old_prefix):
                    add("grouped_import_path_order", item.start, f"{prefix} must precede {old_prefix}")
        previous = (group_root, prefix, public)
    cleaned = clean_rust_code(text)
    for match in re.finditer(r";[^\n]*\buse\s+", cleaned):
        add("multiple_imports_on_line", match.start(), "each import declaration occupies its own source line")
    return result
