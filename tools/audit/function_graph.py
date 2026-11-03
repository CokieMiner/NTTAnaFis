"""Function dependency graph, explicit call cycles and review evidence."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import asdict
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path

from .common import ROOT
from .function_references import ReferenceIndex
from .function_symbols import FunctionIndex


class FunctionGraph:
    """Collect source relationships without claiming compiler-level reachability."""

    def __init__(self, root: Path = ROOT) -> None:
        self.index = FunctionIndex(root)
        references = ReferenceIndex(self.index)
        self.references = references.references
        self.unresolved = references.unresolved
        self.incoming = defaultdict(list)
        self.outgoing = defaultdict(list)
        self.calls = defaultdict(set)
        for reference in self.references:
            self.incoming[reference.callee].append(reference)
            if reference.caller:
                self.outgoing[reference.caller].append(reference)
                if reference.confidence == "resolved" and reference.kind == "call":
                    self.calls[reference.caller].add(reference.callee)
        self._cycles()

    def summary(self) -> dict:
        functions = self.index.functions.values()
        return {
            "files": len(self.index.sources),
            "functions": len(self.index.functions),
            "library_functions": sum(f.path.startswith("src/") and not f.test for f in functions),
            "exported_functions": sum(f.exported for f in functions),
            "resolved_calls": sum(r.kind == "call" and r.confidence == "resolved" for r in self.references),
            "function_values": sum(r.kind == "reference" and r.confidence == "resolved" for r in self.references),
            "possible_references": sum(r.confidence == "possible" for r in self.references),
            "unresolved_sites": len(self.unresolved),
            "source_notes": len(self.index.notes),
        }

    def payload(self, reviews: list) -> dict:
        """Keep declaration roles, evidence and uncertainty in the exported graph."""
        grouped = {}
        for reference in self.references:
            if reference.confidence == "possible":
                key = reference.caller, reference.path, reference.line, reference.kind, reference.spelling, reference.offset
                grouped.setdefault(key, set()).add(reference.callee)
        return {
            "schema": 1,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "root": str(self.index.root.resolve()),
            "sources": {path: sha256(source.text.encode("utf-8")).hexdigest() for path, source in self.index.sources.items()},
            "summary": self.summary(),
            "review_counts": dict(sorted(Counter(r.kind for r in reviews).items())),
            "functions": [asdict(f) | {"symbol": f.symbol} for f in self.index.functions.values()],
            "references": [asdict(r) for r in self.references if r.confidence == "resolved"],
            "possible_references": [dict(zip(("caller", "path", "line", "kind", "spelling", "offset"), key)) | {"callees": sorted(callees)}
                                    for key, callees in grouped.items()],
            "unresolved": [asdict(r) for r in self.unresolved],
            "source_notes": self.index.notes,
            "reviews": [asdict(r) for r in reviews],
            "limits": [
                "Source declarations and all cfg alternatives are inspected; macros are not expanded.",
                "Explicit paths, Self and simple typed parameters resolve syntactically; dynamic dispatch and arbitrary receiver expressions remain uncertain.",
                "An unresolved site can belong to an external crate, a primitive, a closure, a generated function or an untyped project method.",
                "Placement, visibility and ordering reviews are candidates, not proofs of dead code or incorrect behavior.",
                "Source call order does not establish runtime stage order; signatures and custom module paths need compiler-assisted review.",
            ],
        }

    def _cycles(self) -> None:
        """Assign strongly connected components without recursive Python traversal."""
        seen, finished = set(), []
        for function in sorted(self.index.functions):
            if function in seen:
                continue
            stack = [(function, False)]
            while stack:
                node, exiting = stack.pop()
                if exiting:
                    finished.append(node)
                elif node not in seen:
                    seen.add(node)
                    stack.append((node, True))
                    stack.extend((child, False) for child in sorted(self.calls[node], reverse=True) if child not in seen)
        reverse = defaultdict(set)
        for caller, callees in self.calls.items():
            for callee in callees:
                reverse[callee].add(caller)
        seen = set()
        component = 0
        for node in reversed(finished):
            if node in seen:
                continue
            stack = [node]
            while stack:
                current = stack.pop()
                if current in seen:
                    continue
                seen.add(current)
                self.index.functions[current].cycle = component
                stack.extend(reverse[current] - seen)
            component += 1
