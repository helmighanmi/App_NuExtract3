"""NuExtract template handling.

A NuExtract template is a JSON document shaped like the expected output, whose
leaves describe *types* instead of values::

    {
      "invoice_number": "verbatim-string",
      "total": "number",
      "currency": ["USD", "EUR", "Other"],
      "line_items": [{"description": "verbatim-string", "quantity": "integer"}]
    }

This module provides:
* a registry for the bundled templates,
* a JSON-Schema / Pydantic -> NuExtract template converter,
* light structural validation and output shaping.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pydantic import BaseModel

JsonValue = Any

SCALAR_TYPES = {
    "verbatim-string",
    "string",
    "integer",
    "number",
    "boolean",
    "date",
    "time",
    "date-time",
    "country",
    "currency",
    "email",
    "phone",
    "url",
    "object",
}

STRING_FORMATS = {
    "date": "date",
    "time": "time",
    "date-time": "date-time",
    "email": "email",
    "uri": "url",
    "url": "url",
    "phone": "phone",
}

META_KEY = "_description"


class TemplateError(ValueError):
    """Invalid template or schema."""


class TemplateNotFoundError(TemplateError):
    pass


@dataclass(frozen=True)
class TemplateInfo:
    name: str
    description: str
    template: JsonValue


class TemplateRegistry:
    """Loads the ``configs/templates/*.json`` catalogue."""

    def __init__(self, directory: Path) -> None:
        self.directory = directory

    def _paths(self) -> list[Path]:
        if not self.directory.exists():
            return []
        return sorted(self.directory.glob("*.json"))

    def names(self) -> list[str]:
        return [path.stem for path in self._paths()]

    def list_templates(self) -> list[TemplateInfo]:
        infos: list[TemplateInfo] = []
        for path in self._paths():
            try:
                template, description = _load_template_file(path)
            except TemplateError:
                continue
            infos.append(TemplateInfo(path.stem, description, template))
        return infos

    def get(self, name: str) -> JsonValue:
        path = self.directory / f"{name}.json"
        if not path.exists():
            raise TemplateNotFoundError(
                f"Unknown template {name!r}. Available: {', '.join(self.names()) or 'none'}"
            )
        template, _ = _load_template_file(path)
        return template

    def get_info(self, name: str) -> TemplateInfo:
        path = self.directory / f"{name}.json"
        if not path.exists():
            raise TemplateNotFoundError(
                f"Unknown template {name!r}. Available: {', '.join(self.names()) or 'none'}"
            )
        template, description = _load_template_file(path)
        return TemplateInfo(name, description, template)

    def resolve(self, ref: str | dict[str, Any] | list[Any] | None) -> JsonValue:
        """Resolve a name, file path, JSON string or in-memory template."""
        if ref is None:
            raise TemplateError("A template name, path or JSON object is required")
        if isinstance(ref, (dict, list)):
            return ref

        candidate = ref.strip()
        if candidate.startswith(("{", "[")):
            try:
                return json.loads(candidate)
            except json.JSONDecodeError as exc:
                raise TemplateError(f"Template is not valid JSON: {exc}") from exc

        path = Path(candidate)
        if path.suffix == ".json" and path.exists():
            template, _ = _load_template_file(path)
            return template

        return self.get(candidate)


def _load_template_file(path: Path) -> tuple[JsonValue, str]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise TemplateError(f"Could not load template {path}: {exc}") from exc

    description = ""
    if isinstance(payload, dict) and isinstance(payload.get(META_KEY), str):
        description = payload[META_KEY]
        payload = {key: value for key, value in payload.items() if key != META_KEY}

    validate_template_shape(payload)
    return payload, description


def validate_template_shape(template: JsonValue, path: str = "$") -> None:
    """Reject clearly malformed templates before spending GPU time."""
    if isinstance(template, dict):
        if not template:
            raise TemplateError(f"Template object at {path} is empty")
        for key, value in template.items():
            if not isinstance(key, str):
                raise TemplateError(f"Template keys must be strings at {path}")
            validate_template_shape(value, f"{path}.{key}")
        return

    if isinstance(template, list):
        if not template:
            raise TemplateError(f"Template array at {path} must describe its items")
        if len(template) == 1:
            validate_template_shape(template[0], f"{path}[]")
            return
        if all(isinstance(item, str) for item in template):
            return  # enum
        raise TemplateError(
            f"Template array at {path} must be a one-item array or a list of enum values"
        )

    if not isinstance(template, str) or not template:
        raise TemplateError(f"Template leaf at {path} must be a non-empty string")
    if template not in SCALAR_TYPES:
        # Custom type names are allowed by NuExtract, but flag likely typos.
        return


def convert_json_schema(schema: dict[str, Any]) -> tuple[JsonValue, list[str]]:
    """Convert a JSON Schema (e.g. ``Model.model_json_schema()``).

    Returns the NuExtract template and a list of dropped/approximated branches.
    """
    if not isinstance(schema, dict):
        raise TemplateError("JSON Schema must be an object")
    dropped: list[str] = []
    defs = schema.get("$defs", {}) or schema.get("definitions", {}) or {}
    template = _convert_node(schema, defs, dropped, "$")
    validate_template_shape(template)
    return template, dropped


def convert_pydantic(model: type[BaseModel]) -> tuple[JsonValue, list[str]]:
    return convert_json_schema(model.model_json_schema())


def _convert_node(
    node: dict[str, Any],
    defs: dict[str, Any],
    dropped: list[str],
    path: str,
) -> JsonValue:
    if not isinstance(node, dict):
        dropped.append(f"{path}: non-object schema")
        return "string"

    if "$ref" in node:
        ref = str(node["$ref"])
        name = ref.rsplit("/", 1)[-1]
        target = defs.get(name)
        if target is None:
            dropped.append(f"{path}: unresolved $ref {ref}")
            return "string"
        return _convert_node(target, defs, dropped, path)

    for combinator in ("allOf",):
        if combinator in node:
            merged: dict[str, Any] = {}
            for item in node[combinator]:
                converted = _convert_node(item, defs, dropped, path)
                if isinstance(converted, dict):
                    merged.update(converted)
            if merged:
                return merged

    for combinator in ("anyOf", "oneOf"):
        if combinator in node:
            variants = [item for item in node[combinator] if item.get("type") != "null"]
            if len(variants) == 1:
                return _convert_node(variants[0], defs, dropped, path)
            if variants and all("const" in item for item in variants):
                return [str(item["const"]) for item in variants]
            if variants and all("enum" in item for item in variants):
                values: list[str] = []
                for item in variants:
                    values.extend(str(value) for value in item["enum"])
                return values
            dropped.append(f"{path}: union type approximated as string")
            return "string"

    if "const" in node:
        return [str(node["const"])]

    if "enum" in node:
        return [str(value) for value in node["enum"]]

    node_type = node.get("type")
    if isinstance(node_type, list):
        non_null = [item for item in node_type if item != "null"]
        node_type = non_null[0] if len(non_null) == 1 else None

    if node_type == "object" or "properties" in node:
        properties = node.get("properties", {})
        if not properties:
            dropped.append(f"{path}: free-form object approximated as string")
            return "string"
        return {
            key: _convert_node(value, defs, dropped, f"{path}.{key}")
            for key, value in properties.items()
        }

    if node_type == "array":
        items = node.get("items", {})
        if isinstance(items, list):  # tuple-style; not supported by NuExtract
            dropped.append(f"{path}: tuple arrays approximated")
            items = items[0] if items else {}
        return [_convert_node(items, defs, dropped, f"{path}[]")]

    if node_type == "integer":
        return "integer"
    if node_type == "number":
        return "number"
    if node_type == "boolean":
        return "boolean"
    if node_type == "string":
        fmt = node.get("format")
        return STRING_FORMATS.get(fmt, "string") if isinstance(fmt, str) else "string"
    if node_type == "null":
        return "string"

    dropped.append(f"{path}: unsupported schema node, defaulted to string")
    return "string"


def _is_array_template(template: JsonValue) -> bool:
    return isinstance(template, list) and len(template) == 1


def ensure_template_shape(
    value: JsonValue,
    template: JsonValue,
    path: str = "$",
) -> tuple[JsonValue, list[str]]:
    """Fill missing keys / arrays so downstream consumers get a stable shape."""
    warnings: list[str] = []

    if isinstance(template, dict):
        if not isinstance(value, dict):
            warnings.append(
                f"{path}: expected an object, got {type(value).__name__}; reset to object"
            )
            value = {}
        for key, child_template in template.items():
            if key not in value:
                value[key] = [] if _is_array_template(child_template) else None
                warnings.append(f"{path}.{key}: missing in model output")
                continue
            value[key], child_warnings = ensure_template_shape(
                value[key], child_template, f"{path}.{key}"
            )
            warnings.extend(child_warnings)
        return value, warnings

    if _is_array_template(template):
        if value is None:
            return [], warnings
        if not isinstance(value, list):
            warnings.append(f"{path}: expected an array, got {type(value).__name__}; wrapped")
            value = [value]
        shaped: list[JsonValue] = []
        for index, item in enumerate(value):
            shaped_item, child_warnings = ensure_template_shape(
                item, template[0], f"{path}[{index}]"
            )
            shaped.append(shaped_item)
            warnings.extend(child_warnings)
        return shaped, warnings

    # Scalar / enum leaf: leave the model value untouched (validation is done
    # downstream if the caller supplies a Pydantic model).
    return value, warnings
