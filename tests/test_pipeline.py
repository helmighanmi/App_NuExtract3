from __future__ import annotations

from pathlib import Path

import pytest

from nuextract_app.config import Settings
from nuextract_app.core.pipeline import ExtractionPipeline
from nuextract_app.core.templates import TemplateRegistry


@pytest.fixture
def registry(tmp_path: Path) -> TemplateRegistry:
    (tmp_path / "invoice.json").write_text(
        '{"invoice_number": "verbatim-string", "items": [{"name": "string"}], "total": "number"}',
        encoding="utf-8",
    )
    return TemplateRegistry(tmp_path)


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return Settings(
        templates_dir=tmp_path,
        dpi=80,
        max_pages=10,
        pages_per_request=1,
        native_text_min_chars=20,
    )


async def test_extract_text_pdf_uses_text_route(
    settings: Settings, registry: TemplateRegistry, fake_client, text_pdf: bytes
) -> None:
    pipeline = ExtractionPipeline(settings, fake_client, registry)
    result = await pipeline.extract(
        text_pdf,
        "invoice.pdf",
        template="invoice",
        thinking=False,
    )

    assert result.result["ok"] is True
    assert result.result["invoice_number"] is None
    assert result.is_scanned is False
    assert result.pages[0].route == "text"
    assert result.prompt_tokens == 100
    assert fake_client.requests[0].images == []
    assert "INV-2026-0042" in fake_client.requests[0].text
    assert fake_client.requests[0].template == registry.get("invoice")


async def test_scanned_pdf_uses_image_route(
    settings: Settings, registry: TemplateRegistry, fake_client, scanned_pdf: bytes
) -> None:
    pipeline = ExtractionPipeline(settings, fake_client, registry)
    result = await pipeline.extract(scanned_pdf, "scan.pdf", template="invoice")

    assert result.is_scanned is True
    assert result.pages[0].route == "image"
    assert len(fake_client.requests[0].images) == 1
    assert fake_client.requests[0].text is None


async def test_force_vision_on_digital_pdf(
    settings: Settings, registry: TemplateRegistry, fake_client, text_pdf: bytes
) -> None:
    pipeline = ExtractionPipeline(settings, fake_client, registry)
    await pipeline.extract(text_pdf, "invoice.pdf", template="invoice", force_vision=True)
    assert len(fake_client.requests[0].images) == 1


async def test_multi_page_chunks_are_merged(
    settings: Settings,
    registry: TemplateRegistry,
    multi_page_pdf: bytes,
    fake_client_factory,
) -> None:
    client = fake_client_factory(
        [
            '{"invoice_number": "INV-1", "items": [{"name": "a"}], "total": 10}',
            '{"invoice_number": null, "items": [{"name": "b"}], "total": null}',
            '{"items": [{"name": "c"}], "total": 30}',
        ]
    )
    pipeline = ExtractionPipeline(settings, client, registry)
    result = await pipeline.extract(multi_page_pdf, "multi.pdf", template="invoice")

    assert len(client.requests) == 3
    assert result.result["invoice_number"] == "INV-1"
    assert [item["name"] for item in result.result["items"]] == ["a", "b", "c"]
    assert result.result["total"] == 10


async def test_unparseable_chunks_warn_but_partial_result_survives(
    settings: Settings,
    registry: TemplateRegistry,
    multi_page_pdf: bytes,
    fake_client_factory,
) -> None:
    client = fake_client_factory(
        [
            "this is not json",
            '{"invoice_number": "INV-2", "items": [], "total": 5}',
            '{"invoice_number": null, "items": [], "total": null}',
        ]
    )
    pipeline = ExtractionPipeline(settings, client, registry)
    result = await pipeline.extract(multi_page_pdf, "multi.pdf", template="invoice")

    assert result.result["invoice_number"] == "INV-2"
    assert any("Could not parse response" in warning for warning in result.warnings)


async def test_markdown_is_joined(
    settings: Settings,
    registry: TemplateRegistry,
    multi_page_pdf: bytes,
    fake_client_factory,
) -> None:
    client = fake_client_factory(["# Page one", "# Page two", "# Page three"])
    pipeline = ExtractionPipeline(settings, client, registry)
    result = await pipeline.markdown(multi_page_pdf, "multi.pdf")

    assert result.markdown == "# Page one\n\n---\n\n# Page two\n\n---\n\n# Page three"
    assert result.page_count == 3


async def test_template_generation(
    settings: Settings, registry: TemplateRegistry, fake_client_factory
) -> None:
    client = fake_client_factory(['{"name": "string", "date": "date"}'])
    pipeline = ExtractionPipeline(settings, client, registry)
    result = await pipeline.generate_template("Extract name and date")

    assert result.template == {"name": "string", "date": "date"}
    assert client.requests[0].mode == "template-generation"
