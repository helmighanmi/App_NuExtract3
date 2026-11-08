from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

import pytest
from pydantic import BaseModel

from nuextract_app.core.postprocess import (
    JsonParseError,
    merge_structured_results,
    parse_json,
)
from nuextract_app.core.templates import (
    TemplateError,
    TemplateNotFoundError,
    TemplateRegistry,
    convert_json_schema,
    ensure_template_shape,
    validate_template_shape,
)


def test_registry_lists_bundled_templates() -> None:
    registry = TemplateRegistry(Path("configs/templates"))
    names = registry.names()
    assert {"invoice", "receipt", "contract"}.issubset(set(names))
    info = registry.get_info("invoice")
    assert info.description
    assert "_description" not in info.template


def test_registry_resolves_inline_json_and_unknown_name() -> None:
    registry = TemplateRegistry(Path("configs/templates"))
    assert registry.resolve('{"total": "number"}') == {"total": "number"}
    with pytest.raises(TemplateNotFoundError):
        registry.get("does-not-exist")


def test_json_schema_conversion_handles_nested_and_refs() -> None:
    schema = {
        "$defs": {
            "Party": {
                "type": "object",
                "properties": {"name": {"type": "string"}},
            }
        },
        "type": "object",
        "properties": {
            "invoice_date": {"type": "string", "format": "date"},
            "total": {"type": "number"},
            "vendor": {"$ref": "#/$defs/Party"},
            "status": {"enum": ["paid", "open"]},
            "note": {"anyOf": [{"type": "string"}, {"type": "null"}]},
            "items": {
                "type": "array",
                "items": {"type": "object", "properties": {"qty": {"type": "integer"}}},
            },
        },
    }
    template, dropped = convert_json_schema(schema)
    assert template["invoice_date"] == "date"
    assert template["total"] == "number"
    assert template["vendor"] == {"name": "string"}
    assert template["status"] == ["paid", "open"]
    assert template["note"] == "string"
    assert template["items"] == [{"qty": "integer"}]
    assert dropped == []


def test_pydantic_literal_becomes_enum() -> None:
    class Booking(BaseModel):
        room: Literal["single", "double"]
        guests: int

    template, _ = convert_json_schema(Booking.model_json_schema())
    assert template["room"] == ["single", "double"]
    assert template["guests"] == "integer"


def test_invalid_templates_are_rejected() -> None:
    with pytest.raises(TemplateError):
        validate_template_shape({})
    with pytest.raises(TemplateError):
        validate_template_shape({"items": []})
    with pytest.raises(TemplateError):
        validate_template_shape({"weird": [1, 2]})


def test_ensure_template_shape_fills_missing_keys() -> None:
    template = {"total": "number", "items": [{"name": "string"}]}
    value, warnings = ensure_template_shape({"items": [{"name": "coffee"}]}, template)
    assert value["total"] is None
    assert value["items"] == [{"name": "coffee"}]
    assert any("total" in warning for warning in warnings)


def test_parse_json_tolerates_fences_and_trailing_commas() -> None:
    assert parse_json('```json\n{"a": 1}\n```') == {"a": 1}
    assert parse_json('Here is the result: {"a": 1, "b": [2,]}') == {"a": 1, "b": [2]}
    with pytest.raises(JsonParseError):
        parse_json("   ")


def test_parse_json_ignores_braces_inside_strings() -> None:
    payload = '{"note": "value } inside", "ok": true}'
    assert parse_json(f"prose before {payload} prose after")["ok"] is True


def test_merge_chunks_concatenates_arrays_and_keeps_first_scalar() -> None:
    merged = merge_structured_results(
        [
            {"invoice_number": "INV-1", "items": [{"name": "a"}], "total": 10},
            {"invoice_number": None, "items": [{"name": "b"}], "total": 99},
        ]
    )
    assert merged == {
        "invoice_number": "INV-1",
        "items": [{"name": "a"}, {"name": "b"}],
        "total": 10,
    }


def test_bundled_templates_are_valid_json() -> None:
    for path in Path("configs/templates").glob("*.json"):
        payload = json.loads(path.read_text("utf-8"))
        payload.pop("_description", None)
        validate_template_shape(payload)
