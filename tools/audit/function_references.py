"""Resolve explicit function paths; retain uncertain method and macro references."""

from __future__ import annotations

import re
from dataclasses import dataclass

from .function_symbols import Function, FunctionIndex, Source
from .rust_source import matching_delimiter

PATH = re.compile(r"(?<![\w$])(?:r#)?[A-Za-z_]\w*(?:\s*::\s*(?:r#)?[A-Za-z_]\w*)*")


@dataclass(frozen=True)
class Reference:
    caller: str | None
    callee: str
    path: str
    line: int
    kind: str
    confidence: str
    spelling: str
    offset: int


@dataclass(frozen=True)
class Unresolved:
    caller: str | None
    path: str
    line: int
    spelling: str
    kind: str
    offset: int


class ReferenceIndex:
    """Separate syntactic calls, function values and opaque macro tokens."""

    def __init__(self, index: FunctionIndex) -> None:
        self.index = index
        self.references: list[Reference] = []
        self.unresolved: list[Unresolved] = []
        self.cache: dict[tuple[str, str, str], tuple[str, ...]] = {}
        for source in index.sources.values():
            self._source(source)

    def _source(self, source: Source) -> None:
        functions = sorted((f for f in self.index.functions.values() if f.path == source.path), key=lambda f: f.start)
        cleaned = list(source.cleaned)
        for start, end in source.exclusions:
            cleaned[start:end] = ["\n" if c == "\n" else " " for c in cleaned[start:end]]
        code = "".join(cleaned)
        # Turbofish arguments are types, not path segments or executable uses.
        for match in re.finditer(r"::\s*<", code):
            opening = code.index("<", match.start(), match.end())
            closing = matching_delimiter(code, opening, "<", ">")
            if closing is not None:
                code = code[:match.start()] + " " * (closing + 1 - match.start()) + code[closing + 1:]
        macros = list(source.macros)
        for match in re.finditer(r"\b\w+(?:::\w+)*\s*!\s*([({\[])", code):
            opening = match.end() - 1
            closing = matching_delimiter(code, opening, match[1], {"(": ")", "{": "}", "[": "]"}[match[1]])
            if closing is not None:
                macros.append((opening, closing))
        contexts = {f.id: self._binding_context(f, source) for f in functions}
        cursor = 0
        for match in PATH.finditer(code):
            while cursor < len(functions) and functions[cursor].end <= match.start():
                cursor += 1
            function = functions[cursor] if cursor < len(functions) and functions[cursor].body_start <= match.start() < functions[cursor].end else None
            module = function.module if function else source.module
            for start, end, child in source.scopes:
                if start <= match.start() < end:
                    module = child
            spelling = re.sub(r"\s+|r#", "", match[0])
            following, preceding = match.end(), match.start()
            while following < len(code) and code[following].isspace():
                following += 1
            while preceding and code[preceding - 1].isspace():
                preceding -= 1
            suffix = code[following:following + 1]
            prefix = code[max(0, preceding - 300):preceding]
            if suffix == ":":
                continue
            method = prefix.endswith(".") and not prefix.endswith("..")
            in_macro = any(start <= match.start() < end for start, end in macros)
            # Rust method values use Type::method. A bare receiver.field is a
            # field access even when an inherent method has the same name.
            if method and not in_macro and suffix != "(":
                continue
            kind = "macro" if in_macro else "call" if suffix.startswith("(") else "reference"
            line = source.text.count("\n", 0, match.start()) + 1
            candidates = ()
            if method and function:
                receiver = re.search(r"\b([A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*)\.$", prefix)
                if receiver:
                    owner = contexts[function.id][0].get(receiver[1])
                    if owner:
                        candidates = tuple(self.index.symbols.get(f"{owner}::{spelling}", ()))
            elif (not method and not (prefix.endswith("::") and "::" not in spelling)
                  and not (function and "::" not in spelling and spelling in contexts[function.id][1])):
                candidates = self.resolve(module, spelling, function.owner if function else "")
            confidence = "resolved" if len(candidates) == 1 and not in_macro and not (function and function.opaque) else "possible"
            if not candidates:
                name = spelling.rsplit("::", 1)[-1]
                # Name matches preserve possible users without asserting that
                # an untyped receiver or generated selector calls this function.
                if method or in_macro or kind == "call":
                    candidates = tuple(self.index.names.get(name, ()))
                confidence = "possible"
            for callee in candidates:
                self.references.append(Reference(function.id if function else None, callee, source.path,
                                                 line, kind, confidence, spelling, match.start()))
            if (kind == "call" and confidence != "resolved") or suffix.startswith("!"):
                self.unresolved.append(Unresolved(function.id if function else None, source.path, line, spelling,
                                                   "macro" if suffix.startswith("!") else "method" if method else "call", match.start()))

    def resolve(self, module: str, spelling: str, owner: str = "") -> tuple[str, ...]:
        key = module, spelling, owner
        if key in self.cache:
            return self.cache[key]
        if spelling.startswith("Self::"):
            qualified = owner + spelling[4:]
        else:
            qualified = self.index.qualify(module, spelling)
        targets = {qualified}
        targets.update(self.index.visibility.resolve(qualified))
        parent, separator, name = qualified.rpartition("::")
        if separator:
            targets.update(f"{resolved}::{name}" for resolved in self.index.visibility.resolve(parent))
        candidates = tuple(sorted({function for target in targets for function in self.index.symbols.get(target, ())
                                   if not self.index.functions[function].trait}))
        self.cache[key] = candidates
        return candidates

    def _binding_context(self, function: Function, source: Source) -> tuple[dict[str, str], set[str]]:
        """Use explicit parameter types and Self fields; do not infer arbitrary expressions."""
        result = {"self": function.owner} if function.receiver else {}
        declarations = re.findall(r"\b([a-z_]\w*)\s*:\s*(?:&\s*(?:'\w+\s*)?(?:mut\s+)?)?([A-Z]\w*(?:::\w+)*)", function.header)
        body = source.cleaned[function.body_start:function.end]
        # Shadowed bindings and destructuring are left to Rust type resolution.
        patterns = [match[1] for match in re.finditer(r"\b(?:let|for)\s+([^;=]+?)(?:=|\bin\b)", body)]
        patterns.extend(match[1] for match in re.finditer(r"(?:\{|,)\s*([^=;\n]+)=>", body))
        patterns.extend(match[1] for match in re.finditer(r"(?<!\|)\|([^|\n]+)\|", body))
        rebound = {name for pattern in patterns for name in re.findall(r"\b[a-z_]\w*\b", pattern)}
        bindings = rebound | set(re.findall(r"\b([a-z_]\w*)\s*:", function.header))
        for name, typename in declarations:
            if name in rebound:
                continue
            owners = {function.owner} if typename == "Self" else self.index.visibility.resolve(self.index.qualify(function.module, typename))
            if len(owners) == 1:
                result[name] = next(iter(owners))
        owner = self.index.types.get(function.owner)
        if owner:
            owner_module = function.owner.rpartition("::")[0]
            for name, typename in owner.fields.items():
                targets = self.index.visibility.resolve(self.index.qualify(owner_module, typename))
                if len(targets) == 1:
                    result[f"self.{name}"] = next(iter(targets))
        return result, bindings
