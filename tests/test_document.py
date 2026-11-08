from __future__ import annotations

import pytest

from nuextract_app.core.document import (
    UnsupportedDocumentError,
    load_document,
)

PNG_MAGIC = b"\x89PNG\r\n\x1a\n"


def test_text_pdf_uses_native_text(text_pdf: bytes) -> None:
    doc = load_document(text_pdf, "invoice.pdf", dpi=120, max_pages=10)
    assert doc.kind == "pdf"
    assert doc.page_count == 1
    assert doc.is_scanned is False
    assert doc.has_native_text is True
    page = doc.pages[0]
    assert "INV-2026-0042" in page.text
    assert page.png is not None and page.png.startswith(PNG_MAGIC)
    assert page.width > 0 and page.height > 0


def test_scanned_pdf_is_detected(scanned_pdf: bytes) -> None:
    doc = load_document(scanned_pdf, "scan.pdf", dpi=100, max_pages=10)
    assert doc.page_count == 1
    assert doc.is_scanned is True
    assert doc.pages[0].is_scanned is True
    assert doc.pages[0].png is not None


def test_image_is_treated_as_scan(image_bytes: bytes) -> None:
    doc = load_document(image_bytes, "receipt.png")
    assert doc.kind == "image"
    assert doc.page_count == 1
    assert doc.is_scanned is True
    assert doc.pages[0].png is not None


def test_page_limit_truncates_with_warning(multi_page_pdf: bytes) -> None:
    doc = load_document(multi_page_pdf, "multi.pdf", dpi=80, max_pages=2)
    assert doc.page_count == 2
    assert any("only the first 2" in warning for warning in doc.warnings)


def test_unknown_format_is_rejected() -> None:
    with pytest.raises(UnsupportedDocumentError):
        load_document(b"not a document", "file.txt")


def test_empty_file_is_rejected() -> None:
    with pytest.raises(ValueError):
        load_document(b"", "empty.pdf")
