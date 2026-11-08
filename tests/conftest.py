"""Shared fixtures. No GPU or network access required."""

from __future__ import annotations

import io

import pymupdf as fitz
import pytest
from PIL import Image, ImageDraw

from nuextract_app.core.client import GenerationResult


@pytest.fixture
def text_pdf() -> bytes:
    doc = fitz.open()
    page = doc.new_page(width=595, height=842)
    page.insert_text((72, 72), "INVOICE INV-2026-0042", fontsize=16)
    page.insert_text((72, 110), "Date: 2026-03-14", fontsize=11)
    page.insert_text((72, 130), "Total: 5533.50 EUR", fontsize=11)
    page.insert_text((72, 160), "Filler text " * 40, fontsize=10)
    data = doc.tobytes()
    doc.close()
    return data


@pytest.fixture
def multi_page_pdf() -> bytes:
    doc = fitz.open()
    for index in range(3):
        page = doc.new_page()
        page.insert_text((72, 72), f"Page {index + 1} " + "content " * 40, fontsize=11)
    data = doc.tobytes()
    doc.close()
    return data


@pytest.fixture
def scanned_pdf() -> bytes:
    """A PDF with no text layer (blank page = worst-case scan)."""
    doc = fitz.open()
    page = doc.new_page()
    page.draw_rect((100, 100, 400, 300), color=(0, 0, 0), width=2)
    data = doc.tobytes()
    doc.close()
    return data


@pytest.fixture
def image_bytes() -> bytes:
    image = Image.new("RGB", (640, 400), "white")
    draw = ImageDraw.Draw(image)
    draw.text((40, 40), "RECEIPT\nCoffee 3.50\nTotal 3.50", fill="black")
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


class FakeClient:
    """Deterministic model backend for pipeline/API tests."""

    def __init__(self, content: str | list[str] = '{"ok": true}') -> None:
        self.contents = content if isinstance(content, list) else [content]
        self.requests: list[object] = []
        self.closed = False

    async def generate(self, request):  # type: ignore[no-untyped-def]
        self.requests.append(request)
        index = min(len(self.requests) - 1, len(self.contents) - 1)
        return GenerationResult(
            content=self.contents[index],
            prompt_tokens=100,
            completion_tokens=20,
            latency_s=0.01,
        )

    async def health(self) -> bool:
        return True

    async def close(self) -> None:
        self.closed = True


@pytest.fixture
def fake_client() -> FakeClient:
    return FakeClient()


@pytest.fixture
def fake_client_factory() -> type[FakeClient]:
    return FakeClient
