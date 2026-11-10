"""Source reachability through explicit Rust module declarations and reexports.

All cfg branches are inspected. Macros are not expanded; unresolved crate-visible
methods require an explicit source declaration rather than an assumed export.
"""

from __future__ import annotations

import re
from pathlib import Path

from .common import Finding, ROOT
from .import_parser import expand_use_tree, is_test_path, resolve_relative_path
from .items import top_level_items
from .rust_source import clean_rust_code


class VisibilityGraph:
    """Resolve library type ownership without treating sealed pub items as exports."""

    def __init__(self, root: Path = ROOT) -> None:
        self.paths = {}
        for path in (root / "src").rglob("*.rs"):
            if is_test_path(path):
                continue
            parts = list(path.relative_to(root).with_suffix("").parts)
            if parts[-1] in {"mod", "lib"}:
                parts.pop()
            self.paths["::".join(parts)] = path
        self.surfaces = {}

    def reachable_types(self) -> set[str]:
        """Follow public modules and reexports, including feature-gated surfaces."""
        types = set()
        pending = ["src"]
        visited = set()
        while pending:
            module = pending.pop()
            if module in visited:
                continue
            visited.add(module)
            for target in self.exports(module):
                if target in self.paths:
                    pending.append(target)
                else:
                    owner, _, name = target.rpartition("::")
                    definitions, _, _, _ = self.surface(owner)
                    if definitions.get(name) in {"struct", "enum", "union", "type"}:
                        types.add(target)
        return types

    def exports(self, module: str, visited: frozenset[str] = frozenset()) -> set[str]:
        if module in visited:
            return set()
        _, _, names, globs = self.surface(module)
        result = {target for name in names for target in self.resolve(f"{module}::{name}")}
        for target, public in globs:
            if public:
                for owner in self.resolve(target):
                    result.update(self.exports(owner, visited | {module}))
        return result

    def resolve(self, target: str, visited: frozenset[str] = frozenset()) -> set[str]:
        """Resolve a local name, import alias, or reexport to its source owner."""
        if target in visited:
            return set()
        if target in self.paths:
            return {target}
        module, _, name = target.rpartition("::")
        if not module:
            return set()
        if module not in self.paths:
            return {resolved for owner in self.resolve(module, visited | {target})
                    for resolved in self.resolve(f"{owner}::{name}", visited | {target})}
        definitions, bindings, _, globs = self.surface(module)
        if name in definitions and (definitions[name] != "type" or name not in bindings):
            return {target}
        result = {resolved for imported in bindings.get(name, ())
                  for resolved in self.resolve(imported, visited | {target})}
        for owner, _ in globs:
            for resolved_owner in self.resolve(owner, visited | {target}):
                result.update(self.resolve(f"{resolved_owner}::{name}", visited | {target}))
        return result

    def surface(self, module: str) -> tuple[dict, dict, set, list]:
        if module in self.surfaces:
            return self.surfaces[module]
        definitions, bindings, public_names, globs = {}, {}, set(), []
        path = self.paths.get(module)
        if path is not None:
            for item in top_level_items(path.read_text(encoding="utf-8")):
                public = bool(re.match(r"pub\s+(?!\()", item.header))
                if item.kind == "use":
                    declaration = re.sub(r"^(?:pub\s*(?:\([^)]*\))?\s+)?use\s+", "", item.header)
                    for target, alias in expand_use_tree(declaration):
                        qualified = resolve_relative_path(module, target)
                        if qualified.endswith("::*"):
                            globs.append((qualified[:-3], public))
                            continue
                        name = alias or target.rsplit("::", 1)[-1]
                        bindings.setdefault(name, []).append(qualified)
                        if public:
                            public_names.add(name)
                elif item.kind == "mod":
                    bindings.setdefault(item.name, []).append(f"{module}::{item.name}")
                    if public:
                        public_names.add(item.name)
                elif item.name and item.kind != "impl":
                    definitions[item.name] = item.kind
                    if item.kind == "type":
                        target = re.match(r"\s*([A-Za-z_]\w*(?:::[A-Za-z_]\w*)*)", item.header.partition("=")[2])
                        if target:
                            bindings.setdefault(item.name, []).append(resolve_relative_path(module, target[1]))
                    if public:
                        public_names.add(item.name)
        self.surfaces[module] = definitions, bindings, public_names, globs
        return self.surfaces[module]


def visibility_findings(text: str, path: str, graph: VisibilityGraph, reachable: set[str]) -> list[Finding]:
    """Permit pub(crate) only for explicit facade reexports in src/lib.rs."""
    del graph, reachable
    cleaned = clean_rust_code(text, scrub_attributes=False)
    declarations = list(re.finditer(r"\bpub\s*\(\s*crate\s*\)", cleaned))
    if not declarations:
        return []
    permitted = []
    if path == "src/lib.rs":
        permitted = [(item.start, item.end) for item in top_level_items(text) if item.kind == "use"]
    return [Finding(
        "forbidden_crate_visibility", path, text.count("\n", 0, match.start()) + 1,
        "pub(crate) is restricted to explicit reexports in src/lib.rs",
    ) for match in declarations if not any(start <= match.start() < end for start, end in permitted)]
