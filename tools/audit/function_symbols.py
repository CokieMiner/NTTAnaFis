"""Rust function declarations and module bindings for source-level review."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

from .import_parser import LIBRARY_CRATE, expand_use_tree, is_test_path
from .items import attributes, cfg_requires_test, top_level_items
from .rust_source import clean_rust_code, matching_delimiter
from .sources import rust_source_paths
from .visibility import VisibilityGraph


@dataclass
class Function:
    id: str
    name: str
    path: str
    module: str
    line: int
    start: int
    body_start: int
    end: int
    header: str
    visibility: str
    owner: str = ""
    trait: str = ""
    receiver: bool = False
    test: bool = False
    conditional: bool = False
    exported: bool = False
    namespace: bool = False
    opaque: bool = False
    cycle: int = -1

    @property
    def symbol(self) -> str:
        return f"{self.owner or self.module}::{self.name}"


@dataclass
class Source:
    path: str
    module: str
    text: str
    cleaned: str
    exclusions: list[tuple[int, int]] = field(default_factory=list)
    scopes: list[tuple[int, int, str]] = field(default_factory=list)
    macros: list[tuple[int, int]] = field(default_factory=list)


@dataclass
class Type:
    symbol: str
    namespace: bool
    fields: dict[str, str]
    private_fields: set[str]


class FunctionIndex:
    """Index explicit declarations, preserving cfg alternatives and source spans."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.visibility = VisibilityGraph(root)
        self.sources: dict[str, Source] = {}
        self.functions: dict[str, Function] = {}
        self.types: dict[str, Type] = {}
        self.symbols: dict[str, list[str]] = {}
        self.names: dict[str, list[str]] = {}
        self.notes: list[dict] = []
        self.test_modules: set[str] = set()
        self.conditional_modules: set[str] = set()
        self.conditional_types: set[str] = set()
        for path in rust_source_paths(root):
            relative = path.relative_to(root).as_posix()
            parts = list(Path(relative).with_suffix("").parts)
            if parts[-1] in {"mod", "lib", "main"}:
                parts.pop()
            module = "::".join(parts)
            text = path.read_bytes().decode("utf-8")
            source = Source(relative, module, text, clean_rust_code(text))
            self.sources[relative] = source
            self.visibility.paths[module] = path
            inner = [a for a in attributes(text) if a.inner]
            test = is_test_path(Path(relative)) or any(cfg_requires_test(a.code) for a in inner)
            conditional = any(re.search(r"\bcfg\s*\(", a.code) for a in inner)
            self._declarations(source, text, 0, module, test, conditional=conditional)
        self._owners_and_exports()

    def _declarations(self, source: Source, fragment: str, base: int, module: str,
                      test: bool, owner: str = "", trait: str = "", conditional: bool = False) -> None:
        definitions, bindings, public_names, globs = self.visibility.surfaces.setdefault(module, ({}, {}, set(), []))
        for item in top_level_items(fragment):
            start, end = base + item.start, base + item.end
            gated_test = test or any(cfg_requires_test(a.code) for a in item.attributes)
            gated = conditional or any(re.search(r"\bcfg\s*\(", a.code) for a in item.attributes)
            opening = source.cleaned.find("{", start, end)
            has_body = opening >= 0 and source.cleaned[end - 1] == "}"
            body_start = opening + 1 if has_body else end
            public = bool(re.match(r"pub\s+(?!\()", item.header))
            if any(a.code.startswith("path") for a in item.attributes):
                self.notes.append({"path": source.path, "line": source.text.count("\n", 0, start) + 1,
                                   "detail": "custom module paths require compiler-assisted name resolution"})
            if item.kind == "use":
                source.exclusions.append((start, end))
                declaration = re.sub(r"^(?:pub\s*(?:\([^)]*\)\s*)?)?use\s+", "", item.header)
                for target, alias in expand_use_tree(declaration):
                    qualified = self.qualify(module, target)
                    if qualified.endswith("::*"):
                        globs.append((qualified[:-3], public))
                    else:
                        name = alias or target.rsplit("::", 1)[-1]
                        bindings.setdefault(name, []).append(qualified)
                        if public:
                            public_names.add(name)
            elif item.kind == "mod":
                child = f"{module}::{item.name}"
                bindings.setdefault(item.name, []).append(child)
                if gated_test:
                    self.test_modules.add(child)
                if gated:
                    self.conditional_modules.add(child)
                if public:
                    public_names.add(item.name)
                if has_body:
                    self.visibility.paths[child] = self.root / source.path
                    source.scopes.append((body_start, end - 1, child))
                    self._declarations(source, source.text[body_start:end - 1], body_start, child, gated_test,
                                       conditional=gated)
                source.exclusions.append((start, body_start))
            elif item.kind in {"impl", "trait"}:
                if item.kind == "trait":
                    impl_owner, impl_trait = f"{module}::{item.name}", item.name
                    definitions[item.name] = "trait"
                    if public:
                        public_names.add(item.name)
                else:
                    impl_owner, impl_trait = impl_target(item.header)
                source.exclusions.append((start, body_start))
                if has_body and impl_owner:
                    self._declarations(source, source.text[body_start:end - 1], body_start, module,
                                       gated_test, impl_owner, impl_trait, gated)
            elif item.kind == "fn":
                source.exclusions.append((start, body_start))
                if not has_body:
                    continue
                line = source.text.count("\n", 0, start) + 1
                visibility = re.match(r"pub(?:\s*\([^)]*\))?", item.header)
                function = Function(f"{source.path}:{line}:{start}", item.name, source.path, module,
                                    line, start, body_start, end - 1, item.header,
                                    visibility[0] if visibility else "private", owner, trait,
                                    bool(re.search(r"\bself\b(?!\s*::)", item.header)), gated_test, gated)
                function.exported = any(re.search(r"\b(?:no_mangle|export_name)\b", a.code) for a in item.attributes)
                body = source.cleaned[body_start:end - 1]
                # Nested items and local imports require lexical scopes beyond
                # this module-level index; they prevent absence-based reviews.
                function.opaque = bool(re.search(r"\b(?:fn|use)\s+\w", body))
                self.functions[function.id] = function
                if not owner:
                    definitions[item.name] = "fn"
                    if public:
                        public_names.add(item.name)
            elif item.kind in {"macro_rules", "macro"}:
                # Inspect macro tokens as possible references, never declarations.
                source.exclusions.append((start, body_start))
                source.macros.append((body_start, end - 1))
            elif item.name:
                definitions[item.name] = item.kind
                if public:
                    public_names.add(item.name)
                if item.kind == "type":
                    alias = re.match(r"\s*([\w:]+)", item.header.partition("=")[2])
                    if alias:
                        bindings.setdefault(item.name, []).append(self.qualify(module, alias[1]))
                if item.kind in {"struct", "enum", "union"}:
                    content = source.cleaned[body_start:end - 1] if has_body else ""
                    declarations = re.findall(r"\b(?:(pub(?:\s*\([^)]*\))?)\s+)?([a-z_]\w*)\s*:\s*(?:&\s*(?:'\w+\s*)?(?:mut\s+)?)?([A-Za-z_]\w*(?:::\w+)*)", content)
                    fields = {name: typename for _, name, typename in declarations}
                    private_fields = {name for visibility, name, _ in declarations if not visibility}
                    namespace = item.kind == "struct" and (not has_body and "(" not in item.header
                                                             or has_body and not source.cleaned[body_start:end - 1].strip())
                    symbol = f"{module}::{item.name}"
                    self.types[symbol] = Type(symbol, namespace, fields, private_fields)
                    if gated:
                        self.conditional_types.add(symbol)
                # Initializers can store function pointers; only mask the header.
                equal = source.cleaned.find("=", start, body_start)
                source.exclusions.append((start, equal + 1 if equal >= 0 else body_start))

    def _owners_and_exports(self) -> None:
        reachable = self.visibility.reachable_types()
        exports, pending, seen = set(), ["src"], set()
        while pending:
            module = pending.pop()
            if module in seen:
                continue
            seen.add(module)
            for target in self.visibility.exports(module):
                if target in self.visibility.paths:
                    pending.append(target)
                else:
                    exports.add(target)
        for function in self.functions.values():
            function.test |= any(function.module == m or function.module.startswith(m + "::") for m in self.test_modules)
            function.conditional |= any(function.module == m or function.module.startswith(m + "::") for m in self.conditional_modules)
            if function.owner:
                if function.owner not in self.types and not function.owner.startswith("src::"):
                    targets = self.visibility.resolve(self.qualify(function.module, function.owner))
                    function.owner = next(iter(targets)) if len(targets) == 1 else self.qualify(function.module, function.owner)
                function.namespace = bool(self.types.get(function.owner) and self.types[function.owner].namespace)
                function.conditional |= function.owner in self.conditional_types
                function.exported |= function.owner in reachable and function.visibility == "pub" and not function.trait
            else:
                function.exported |= function.symbol in exports
            self.symbols.setdefault(function.symbol, []).append(function.id)
            self.names.setdefault(function.name, []).append(function.id)

    def qualify(self, module: str, target: str) -> str:
        """Keep consumer crate roots separate from the library's crate root."""
        target = target.strip().lstrip(":").replace("$crate", "crate")
        parts = module.split("::")
        if target == "self" or target.startswith("self::"):
            return module + target[4:]
        if target.startswith("crate::"):
            prefix = "src" if parts[0] == "src" else module
            return f"{prefix}::{target[7:]}"
        while target.startswith("super::"):
            target = target[7:]
            if len(parts) > 1:
                parts.pop()
        if target.startswith(f"{LIBRARY_CRATE}::"):
            return "src::" + target[len(LIBRARY_CRATE) + 2:]
        if target.startswith(("core::", "std::", "alloc::")):
            return target
        return f"{'::'.join(parts)}::{target}"


def impl_target(header: str) -> tuple[str, str]:
    """Extract the owning type after impl generics and before where predicates."""
    value = re.sub(r"^(?:unsafe\s+)?impl\s*", "", header)
    if value.startswith("<"):
        closing = matching_delimiter(value, 0, "<", ">")
        if closing is None:
            return "", ""
        value = value[closing + 1:].strip()
    value = value.split("where", 1)[0].strip()
    parts = re.split(r"\s+for\s+", value, maxsplit=1)
    owner = re.match(r"([\w:]+)", parts[-1])
    return (owner[1] if owner else "", parts[0] if len(parts) == 2 else "")
