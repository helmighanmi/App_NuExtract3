"""Document ingestion: PDF / image loading, page rendering and scan detection.

Design notes
------------
* PyMuPDF is used instead of Poppler because it ships wheels (no system deps).
* Every PDF page is rendered to PNG so the same code path works for digital,
  scanned and mixed documents.
* Native text is kept when available: sending text to the model is cheaper and
  more accurate than OCR-ing a rendered page.
* Image-heavy pages are tagged ``is_scanned`` so the pipeline can route them to
  the vision path.
"""

from __future__ import annotations

import io
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import pymupdf as fitz  # PyMuPDF
from PIL import Image, ImageOps, UnidentifiedImageError

DocumentKind = Literal["pdf", "image"]

# Above this size images are downscaled before inference (tokens are expensive).
MAX_IMAGE_SIDE = 3200


class DocumentError(ValueError):
    """Base error for unreadable/unsupported documents."""


class UnsupportedDocumentError(DocumentError):
    pass


@dataclass(frozen=True)
class DocumentPage:
    number: int  # 1-based
    text: str
    png: bytes | None
    width: int
    height: int
    is_scanned: bool


@dataclass
class LoadedDocument:
    filename: str
    kind: DocumentKind
    pages: list[DocumentPage]
    warnings: list[str] = field(default_factory=list)

    @property
    def page_count(self) -> int:
        return len(self.pages)

    @property
    def is_scanned(self) -> bool:
        """A document is considered scanned when no page has a usable text layer."""
        return all(page.is_scanned for page in self.pages)

    @property
    def has_native_text(self) -> bool:
        return any(not page.is_scanned for page in self.pages)


def load_document(
    data: bytes,
    filename: str,
    *,
    dpi: int = 200,
    max_pages: int = 40,
    native_text_min_chars: int = 120,
) -> LoadedDocument:
    """Load an image or PDF into normalised pages.

    Raises:
        UnsupportedDocumentError: unknown/corrupt format.
        DocumentError: encrypted or empty document.
    """
    if not data:
        raise DocumentError("Empty file")

    kind = _detect_kind(data, filename)
    if kind == "pdf":
        return _load_pdf(data, filename, dpi, max_pages, native_text_min_chars)
    return _load_image(data, filename)


def _detect_kind(data: bytes, filename: str) -> DocumentKind:
    if data[:1024].lstrip().startswith(b"%PDF-"):
        return "pdf"
    suffix = Path(filename).suffix.lower()
    if suffix == ".pdf":
        # Trust the extension but verify the header; truncated PDFs are common.
        if b"%PDF-" not in data[:2048]:
            raise UnsupportedDocumentError(f"{filename!r} does not look like a valid PDF")
        return "pdf"
    try:
        with Image.open(io.BytesIO(data)) as image:
            image.verify()
    except (UnidentifiedImageError, OSError) as exc:
        raise UnsupportedDocumentError(
            f"Unsupported file type for {filename!r}. Upload a PDF or an image."
        ) from exc
    return "image"


def _load_pdf(
    data: bytes,
    filename: str,
    dpi: int,
    max_pages: int,
    native_text_min_chars: int,
) -> LoadedDocument:
    try:
        doc = fitz.open(stream=data, filetype="pdf")
    except Exception as exc:  # PyMuPDF raises generic exceptions
        raise UnsupportedDocumentError(f"Could not open PDF {filename!r}: {exc}") from exc

    warnings: list[str] = []
    try:
        if doc.needs_pass:
            raise DocumentError("Password-protected PDFs are not supported")
        if doc.page_count == 0:
            raise DocumentError("PDF has no pages")

        page_count = doc.page_count
        if page_count > max_pages:
            warnings.append(
                f"Document has {page_count} pages; only the first {max_pages} are processed."
            )
            page_count = max_pages

        zoom = dpi / 72.0
        matrix = fitz.Matrix(zoom, zoom)
        pages: list[DocumentPage] = []

        for index in range(page_count):
            page = doc.load_page(index)
            text = page.get_text("text").strip()
            pixmap = page.get_pixmap(matrix=matrix, colorspace=fitz.csRGB, alpha=False)
            png = pixmap.tobytes("png")
            is_scanned = len(text) < native_text_min_chars
            pages.append(
                DocumentPage(
                    number=index + 1,
                    text=text,
                    png=png,
                    width=pixmap.width,
                    height=pixmap.height,
                    is_scanned=is_scanned,
                )
            )
    finally:
        doc.close()

    return LoadedDocument(filename=filename, kind="pdf", pages=pages, warnings=warnings)


def _load_image(data: bytes, filename: str) -> LoadedDocument:
    try:
        with Image.open(io.BytesIO(data)) as opened:
            # Honour EXIF orientation (phone photos), then normalise to RGB.
            image = ImageOps.exif_transpose(opened)
            image = image.convert("RGB")
            image = _downscale(image)

            buffer = io.BytesIO()
            image.save(buffer, format="PNG", optimize=True)
            png = buffer.getvalue()
            width, height = image.size
    except (UnidentifiedImageError, OSError) as exc:
        raise UnsupportedDocumentError(f"Could not read image {filename!r}: {exc}") from exc

    page = DocumentPage(
        number=1,
        text="",
        png=png,
        width=width,
        height=height,
        is_scanned=True,
    )
    return LoadedDocument(filename=filename, kind="image", pages=[page])


def _downscale(image: Image.Image) -> Image.Image:
    longest = max(image.size)
    if longest <= MAX_IMAGE_SIDE:
        return image
    ratio = MAX_IMAGE_SIDE / longest
    new_size = (max(1, int(image.width * ratio)), max(1, int(image.height * ratio)))
    return image.resize(new_size, Image.Resampling.LANCZOS)
