"""Post-processing: robust JSON parsing, chunk merging and output shaping."""

from __future__ import annotations

import json
import re
from typing import Any

import json_repair

JsonValue = Any

_FENCE_START = re.compile(r"^\s*```(?:json|JSON)?\s*")
_FENCE_END = re.compile(r"\s*```\s*$")


class JsonParseError(ValueError):
    """The model did not return parseable JSON."""


def strip_code_fences(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = _FENCE_START.sub("", text)
        text = _FENCE_END.sub("", text)
    return text.strip()


def extract_json_candidate(text: str) -> str | None:
    """Return the first balanced JSON object/array found in ``text``.

    Naive brace counting would break on braces inside strings, so this scanner
    tracks string state and escapes.
    """
    start = -1
    opener = ""
    for index, char in enumerate(text):
        if char in "{[":
            start = index
            opener = char
            break
    if start == -1:
        return None

    closer = "}" if opener == "{" else "]"
    depth = 0
    in_string = False
    escaped = False
    for index in range(start, len(text)):
        char = text[index]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == opener:
            depth += 1
        elif char == closer:
            depth -= 1
            if depth == 0:
                return text[start : index + 1]
    return None


def parse_json(text: str) -> JsonValue:
    """Parse model output, tolerating code fences and small syntax errors."""
    if not text or not text.strip():
        raise JsonParseError("Model returned an empty response")

    cleaned = strip_code_fences(text)

    for candidate in (cleaned, extract_json_candidate(cleaned)):
        if not candidate:
            continue
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            continue

    # Last resort: repair common VLM mistakes (trailing commas, quotes...).
    try:
        repaired = json_repair.loads(cleaned)
    except Exception as exc:  # json_repair raises various exceptions
        raise JsonParseError(f"Could not parse JSON from model output: {text[:400]!r}") from exc

    if repaired in ("", None):
        raise JsonParseError(f"Could not parse JSON from model output: {text[:400]!r}")
    return repaired


def merge_structured_results(results: list[JsonValue]) -> JsonValue:
    """Merge per-chunk extraction results.

    * objects: recursive key merge (first non-null scalar wins)
    * arrays: concatenation (e.g. line items across pages)
    """
    values = [value for value in results if value is not None]
    if not values:
        return None

    first = values[0]
    if isinstance(first, dict):
        merged: dict[str, Any] = {}
        for value in values:
            if isinstance(value, dict):
                merged = _merge_dict(merged, value)
        return merged

    if isinstance(first, list):
        merged_list: list[Any] = []
        for value in values:
            if isinstance(value, list):
                merged_list.extend(value)
            elif value is not None:
                merged_list.append(value)
        return merged_list

    return first


def _merge_dict(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    merged = dict(left)
    for key, right_value in right.items():
        if key not in merged:
            merged[key] = right_value
        else:
            merged[key] = _merge_value(merged[key], right_value)
    return merged


def _merge_value(left: JsonValue, right: JsonValue) -> JsonValue:
    if left is None:
        return right
    if right is None:
        return left

    if isinstance(left, dict) and isinstance(right, dict):
        return _merge_dict(left, right)
    if isinstance(left, list) and isinstance(right, list):
        return [*left, *right]
    if isinstance(left, list):
        return [*left, right]
    if isinstance(right, list):
        return [left, *right]

    if isinstance(left, str) and not left.strip():
        return right
    return left
