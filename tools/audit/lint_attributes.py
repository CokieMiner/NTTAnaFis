"""Lint attributes and their literal reasons, including nested cfg_attr branches."""

from __future__ import annotations

import re
from dataclasses import dataclass

from .common import Finding
from .items import attributes
from .rust_source import clean_rust_code, matching_delimiter, split_top_level


@dataclass(frozen=True)
class LintAttribute:
    kind: str
    start: int
    inner: bool
    lints: tuple[str, ...]
    reason: str | None


def lint_attributes(text: str) -> list[LintAttribute]:
    """Read allow/expect calls outside comments and strings, preserving offsets."""
    result = []
    for attribute in attributes(text):
        cleaned = clean_rust_code(attribute.raw, scrub_attributes=False)
        for match in re.finditer(r"\b(allow|expect)\s*\(", cleaned):
            closing = matching_delimiter(cleaned, match.end() - 1, "(", ")")
            if closing is None:
                raise ValueError(f"unclosed lint attribute at offset {attribute.start}")
            body = cleaned[match.end():closing]
            lints = tuple(part for part in split_top_level(body) if not re.match(r"reason\s*=", part))
            reason_key = re.search(r"\breason\s*=", body)
            reason = None
            if reason_key:
                raw_value = attribute.raw[match.end() + reason_key.end():closing].lstrip()
                literal = re.match(r'"((?:\\.|[^"\\])*)"', raw_value, re.DOTALL)
                raw_literal = re.match(r'r(\#*)"(.*?)"\1(?=\s*(?:,|$))', raw_value, re.DOTALL)
                if literal:
                    reason = literal[1]
                elif raw_literal:
                    reason = raw_literal[2]
            result.append(LintAttribute(match[1], attribute.start, attribute.inner, lints, reason))
    return result


def lint_findings(text: str, path: str) -> list[Finding]:
    """Require reasons and reject dead-code suppression rather than cfg gating."""
    findings = []
    for attribute in lint_attributes(text):
        line = text.count("\n", 0, attribute.start) + 1
        if not attribute.reason or not attribute.reason.strip():
            findings.append(Finding(
                f"{attribute.kind}_without_reason", path, line,
                "allow and expect attributes require a nonempty literal reason, including cfg_attr branches",
            ))
        if "dead_code" in attribute.lints:
            findings.append(Finding(
                "dead_code_suppression", path, line,
                "gate the item, imports and reexports for the configurations that use it",
            ))
    return findings
