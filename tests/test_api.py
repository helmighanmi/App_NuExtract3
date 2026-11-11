from __future__ import annotations

from typing import Any

import pytest
from fastapi.testclient import TestClient

from nuextract_app.api.deps import get_pipeline
from nuextract_app.api.main import create_app
from nuextract_app.config import Settings, get_settings
from nuextract_app.core.client import BackendInfo
from nuextract_app.core.pipeline import (
    ExtractionResult,
    MarkdownResult,
    PageTrace,
    TemplateGenerationResult,
)


class FakePipeline:
    async def extract(self, data: bytes, filename: str, **kwargs: Any) -> ExtractionResult:
        return ExtractionResult(
            result={"invoice_number": "INV-2026-0042", "total": 5533.5},
            template={"invoice_number": "verbatim-string", "total": "number"},
            page_count=1,
            is_scanned=False,
            pages=[PageTrace(number=1, route="text", chars=200)],
            prompt_tokens=100,
            completion_tokens=20,
            latency_s=0.42,
            raw='{"invoice_number": "INV-2026-0042", "total": 5533.5}',
        )

    async def markdown(self, data: bytes, filename: str, **kwargs: Any) -> MarkdownResult:
        return MarkdownResult(
            markdown="# Invoice\n\nTotal **5533.50 EUR**",
            page_count=1,
            is_scanned=False,
            prompt_tokens=80,
            completion_tokens=10,
            latency_s=0.2,
        )

    async def generate_template(
        self, prompt: str, *, thinking: bool = False
    ) -> TemplateGenerationResult:
        return TemplateGenerationResult(
            template={"name": "string"},
            raw='{"name": "string"}',
            prompt_tokens=10,
            completion_tokens=5,
            latency_s=0.1,
        )

    async def health(self) -> bool:
        return True

    def backend_info(self) -> BackendInfo:
        return BackendInfo(
            provider="fake",
            model="fake-model",
            device="cpu",
            dtype="float32",
            endpoint="http://fake",
        )

    async def close(self) -> None:
        pass


@pytest.fixture
def client(text_pdf: bytes) -> TestClient:
    app = create_app()
    app.dependency_overrides[get_pipeline] = lambda: FakePipeline()
    with TestClient(app) as test_client:
        test_client.pdf_bytes = text_pdf  # type: ignore[attr-defined]
        yield test_client


def test_health(client: TestClient) -> None:
    response = client.get("/health")
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ok"
    assert payload["provider"]
    assert "torch_available" in payload
    assert response.headers["x-request-id"]


def test_ready_reports_resolved_backend(client: TestClient) -> None:
    response = client.get("/ready")
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ready"
    assert payload["backend"] is True
    assert payload["resolved_provider"] == "fake"
    assert payload["device"] == "cpu"
    assert payload["dtype"] == "float32"


def test_extract_endpoint(client: TestClient) -> None:
    response = client.post(
        "/v1/extract",
        files={"file": ("invoice.pdf", client.pdf_bytes, "application/pdf")},  # type: ignore[attr-defined]
        data={"template_name": "invoice", "thinking": "false"},
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["result"]["invoice_number"] == "INV-2026-0042"
    assert payload["page_count"] == 1
    assert payload["pages"][0]["route"] == "text"
    assert payload["usage"]["total_tokens"] == 120


def test_extract_rejects_empty_file(client: TestClient) -> None:
    response = client.post(
        "/v1/extract",
        files={"file": ("empty.pdf", b"", "application/pdf")},
        data={"template_name": "invoice"},
    )
    assert response.status_code == 400


def test_extract_enforces_upload_limit(text_pdf: bytes) -> None:
    app = create_app()
    app.dependency_overrides[get_pipeline] = lambda: FakePipeline()
    app.dependency_overrides[get_settings] = lambda: Settings(max_upload_mb=0)
    with TestClient(app) as client:
        response = client.post(
            "/v1/extract",
            files={"file": ("invoice.pdf", text_pdf, "application/pdf")},
            data={"template_name": "invoice"},
        )
    assert response.status_code == 413


def test_markdown_endpoint(client: TestClient) -> None:
    response = client.post(
        "/v1/markdown",
        files={"file": ("invoice.pdf", client.pdf_bytes, "application/pdf")},  # type: ignore[attr-defined]
    )
    assert response.status_code == 200
    assert "Invoice" in response.json()["markdown"]


def test_template_generation_endpoint(client: TestClient) -> None:
    response = client.post(
        "/v1/templates/generate",
        json={"prompt": "Extract a name", "thinking": False},
    )
    assert response.status_code == 200
    assert response.json()["template"] == {"name": "string"}


def test_schema_conversion_endpoint(client: TestClient) -> None:
    response = client.post(
        "/v1/templates/convert",
        json={
            "schema": {
                "type": "object",
                "properties": {
                    "total": {"type": "number"},
                    "date": {"type": "string", "format": "date"},
                },
            }
        },
    )
    assert response.status_code == 200
    assert response.json()["template"] == {"total": "number", "date": "date"}


def test_templates_catalogue(client: TestClient) -> None:
    response = client.get("/v1/templates")
    assert response.status_code == 200
    names = {item["name"] for item in response.json()}
    assert "invoice" in names
