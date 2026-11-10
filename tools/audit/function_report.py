"""Write a function graph in DOT and its review evidence in JSON."""

from __future__ import annotations

import json
from pathlib import Path

from .function_graph import FunctionGraph
from .function_rules import FunctionReview


def write_function_report(graph: FunctionGraph, reviews: list[FunctionReview], directory: Path) -> dict[str, str]:
    """Export all function relationships, source hashes, and review candidates."""
    directory = directory.resolve()
    try:
        relative = directory.relative_to(graph.index.root.resolve())
    except ValueError:
        relative = None
    if relative is not None and (not relative.parts or relative.parts[0] != "target"):
        raise ValueError("working function reports inside the repository belong under target/")
    directory.mkdir(parents=True, exist_ok=True)
    payload = graph.payload(reviews)
    serialized = json.dumps(payload, ensure_ascii=False, indent=2)
    json_path, dot_path = (directory / name for name in ("functions.json", "functions.dot"))
    if any(path.resolve().parent != directory for path in (json_path, dot_path)):
        raise ValueError("function report files must stay in their output directory")
    json_path.write_text(serialized + "\n", encoding="utf-8")
    dot = ["digraph functions {", "  rankdir=TB;", '  node [shape=box, fontname="monospace"];']
    for function in graph.index.functions.values():
        label = f"{function.symbol}\n{function.path}:{function.line}"
        dot.append(f"  {json.dumps(function.id)} [label={json.dumps(label)}];")
    edges = sorted({(r.caller, r.callee, r.kind, r.confidence) for r in graph.references if r.caller})
    for caller, callee, kind, confidence in edges:
        style = "solid" if confidence == "resolved" and kind == "call" else "dashed" if confidence == "resolved" else "dotted"
        dot.append(f"  {json.dumps(caller)} -> {json.dumps(callee)} [style={style}, label={json.dumps(kind + ':' + confidence)}];")
    dot_path.write_text("\n".join([*dot, "}", ""]), encoding="utf-8")
    return {"json": str(json_path.resolve()), "dot": str(dot_path.resolve())}
