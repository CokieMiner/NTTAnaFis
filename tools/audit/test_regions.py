"""Locate test-only Rust items for lint inventory classification."""

from __future__ import annotations

import re
from pathlib import Path

from .items import attributes, cfg_requires_test
from .rust_source import clean_rust_code, matching_delimiter

TEST_PATH_PARTS = {"benches", "tests", "fuzz", "fuzz_targets"}
TEST_MODULE_RE = re.compile(r"\bmod\s+(?:tests|[A-Za-z_][A-Za-z0-9_]*_tests)\s*\{")


def is_whole_file_test(rel_path: str) -> bool:
    parts = rel_path.replace("\\", "/").split("/")
    stem = Path(parts[-1]).stem
    return any(part in TEST_PATH_PARTS for part in parts) or stem == "tests" or stem.endswith("_tests")


def find_test_ranges(text: str) -> list[tuple[int, int]]:
    cleaned = clean_rust_code(text, scrub_attributes=False)
    ranges = []
    for match in TEST_MODULE_RE.finditer(cleaned):
        opening = cleaned.find("{", match.start(), match.end())
        closing = matching_delimiter(cleaned, opening, "{", "}")
        if closing is not None:
            ranges.append((match.start(), closing + 1))
    parsed = attributes(text)
    for index, attribute in enumerate(parsed):
        if not attribute.inner and cfg_requires_test(attribute.code):
            item_start = skip_attributes_and_whitespace(cleaned, attribute.end)
            item_end = find_item_end(cleaned, item_start)
            if item_end is not None:
                while index > 0 and not parsed[index - 1].inner and not cleaned[parsed[index - 1].end:parsed[index].start].strip():
                    index -= 1
                ranges.append((parsed[index].start, item_end))
    merged = []
    for start, end in sorted(ranges):
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


def skip_attributes_and_whitespace(text: str, offset: int) -> int:
    while offset < len(text):
        if text[offset].isspace():
            offset += 1
            continue
        if text.startswith("#![", offset):
            opening = offset + 2
        elif text.startswith("#[", offset):
            opening = offset + 1
        else:
            break
        closing = matching_delimiter(text, opening, "[", "]")
        if closing is None:
            break
        offset = closing + 1
    return offset


def find_item_end(text: str, offset: int) -> int | None:
    depths = {"(": 0, "[": 0, "<": 0}
    closing = {")": "(", "]": "[", ">": "<"}
    in_initializer = False
    while offset < len(text):
        char = text[offset]
        if char in {"(", "["}:
            depths[char] += 1
        elif char == "<" and not in_initializer:
            depths[char] += 1
        elif char in closing and depths[closing[char]] > 0:
            depths[closing[char]] -= 1
        elif char == "=" and not any(depths.values()):
            in_initializer = True
        elif char in "{;" and not any(depths.values()):
            if char == ";":
                return offset + 1
            body_end = matching_delimiter(text, offset, "{", "}")
            return len(text) if body_end is None else body_end + 1
        offset += 1
    return None
