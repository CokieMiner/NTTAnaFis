"""Review visibility, local call ordering and helper ownership conservatively."""

from __future__ import annotations

import re
from dataclasses import dataclass

from .function_graph import FunctionGraph

SUBSYSTEMS = (
    "src/api", "src/arch", "src/conversion", "src/crt", "src/plan",
    "src/prime", "src/profile", "src/support", "src/transform",
)


@dataclass(frozen=True)
class FunctionReview:
    kind: str
    function: str
    path: str
    line: int
    detail: str
    evidence: tuple[str, ...]
    uncertain_uses: int = 0


def function_reviews(graph: FunctionGraph) -> list[FunctionReview]:
    """Expose source-supported candidates separately from enforceable violations."""
    result = []
    functions = graph.index.functions
    for function in functions.values():
        if not function.path.startswith("src/") or function.test or function.trait or function.exported:
            continue
        incoming = [r for r in graph.incoming[function.id] if r.caller != function.id]
        definite = [r for r in incoming if r.confidence == "resolved"]
        possible = [r for r in incoming if r.confidence == "possible"]
        production = [r for r in definite if r.path.startswith("src/") and
                      (r.caller is None or not functions[r.caller].test) and "/tests/" not in r.path and not r.path.endswith("/tests.rs")]
        local = [r for r in production if r.path == function.path]
        consumers = {r.path for r in production}
        evidence = tuple(sorted({f"{r.path}:{r.line} ({r.kind})" for r in definite}))
        architecture = function.path.startswith("src/arch/")
        source = graph.index.sources[function.path]
        body = source.cleaned[function.body_start:function.end]
        coupled = bool(re.search(r"\bSelf\b", function.header + body) or
                       function.owner and re.search(rf"\b{re.escape(function.owner.rsplit('::', 1)[-1])}\b", function.header + body))
        protected = architecture or function.conditional or function.opaque
        owner = graph.index.types.get(function.owner)
        private_access = owner and any(re.search(rf"\b(?:self|Self)\s*\.\s*{re.escape(name)}\b", body)
                                               for name in owner.private_fields)
        owner_file = graph.index.visibility.paths.get(function.owner.rpartition("::")[0])
        owns_private_fields = private_access and owner_file == graph.index.root / function.path
        constructor = bool(function.owner and not function.receiver and coupled and re.search(r"->\s*Self\b", function.header))
        if function.visibility != "private" and definite and not possible and not protected:
            # A method's privacy is attached to its implementation module.
            # Test users outside that subtree also require a visibility boundary.
            user_modules = [functions[r.caller].module if r.caller else graph.index.sources[r.path].module for r in definite]
            if all(m == function.module or m.startswith(function.module + "::") for m in user_modules):
                result.append(FunctionReview("possibly_unnecessary_pub", function.id, function.path, function.line,
                                             "All resolved users are inside the declaration module or its descendants; review facade bindings before making it private.", evidence))
        if not incoming and not protected and not function.namespace:
            result.append(FunctionReview("no_observed_users", function.id, function.path, function.line,
                                         "No function call or value use resolves to this declaration. Check generated uses and compiler diagnostics before removing it.", ()))
        if not local and consumers and not architecture and not function.namespace and not owns_private_fields and not constructor:
            if len(consumers) == 1 and next(iter(consumers)) != function.path:
                # Moving engine operations into API consumers would violate
                # the subsystem boundary even with a single observed consumer.
                own_boundary = next((b for b in SUBSYSTEMS if function.path.startswith(b)), "")
                consumer = next(iter(consumers))
                if not own_boundary or consumer.startswith(own_boundary):
                    result.append(FunctionReview("single_consumer_other_file", function.id, function.path, function.line,
                                                 "The only resolved production consumer is another file in the same subsystem; review ownership and the existing module boundary.", evidence, len(possible)))
            elif len(consumers) > 1 and graph.calls[function.id]:
                result.append(FunctionReview("non_leaf_without_local_users", function.id, function.path, function.line,
                                             "This function calls project routines but has no resolved consumer in its own file. Shared leaves are exempt; review whether this is an intentional subsystem entry point.", evidence, len(possible)))
        if function.owner and not function.receiver and not function.namespace and not coupled and not protected:
            result.append(FunctionReview("static_helper_without_type_dependency", function.id, function.path, function.line,
                                         "This associated function has no receiver or explicit owning-type dependency. Review whether the impl expresses a necessary ownership boundary.", evidence))
    # Recursion components are exempt. Compare only resolved local calls to
    # helpers, not public or trait entry points and not function-pointer values.
    ordering = {}
    for caller_id, callees in graph.calls.items():
        caller = functions[caller_id]
        for callee_id in callees:
            callee = functions[callee_id]
            if (caller.path != callee.path or caller.test or callee.test or caller.start <= callee.start
                    or not caller.path.startswith("src/")
                    or caller.cycle == callee.cycle or callee.exported or callee.trait
                    or caller.conditional or callee.conditional):
                continue
            ordering.setdefault(callee_id, []).append(caller)
    for callee_id, callers in ordering.items():
        callee = functions[callee_id]
        result.append(FunctionReview("callee_precedes_caller", callee.id, callee.path, callee.line,
                                     "A resolved caller appears below this helper; review the driver-to-leaf order. This is textual dependency order, not a runtime sequencing proof.",
                                     tuple(f"{f.path}:{f.line} ({f.symbol})" for f in sorted(callers, key=lambda f: f.start))))
    return sorted(result, key=lambda r: (r.path, r.line, r.kind))
