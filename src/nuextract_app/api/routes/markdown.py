"""Document-to-Markdown endpoint (OCR / RAG preprocessing)."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile

from nuextract_app.api.deps import get_pipeline
from nuextract_app.api.routes.extract import HANDLED_ERRORS, read_upload
from nuextract_app.api.schemas import MarkdownResponse, Usage
from nuextract_app.config import Settings, get_settings
from nuextract_app.core.pipeline import ExtractionPipeline
from nuextract_app.logging import current_request_id, get_logger

logger = get_logger(__name__)
router = APIRouter(prefix="/v1", tags=["markdown"])


@router.post("/markdown", response_model=MarkdownResponse, summary="Convert a document to Markdown")
async def convert_to_markdown(
    pipeline: Annotated[ExtractionPipeline, Depends(get_pipeline)],
    settings: Annotated[Settings, Depends(get_settings)],
    file: Annotated[UploadFile, File(description="PDF or image file")],
    thinking: Annotated[bool | None, Form(description="Enable reasoning mode")] = None,
    dpi: Annotated[int | None, Form(ge=72, le=400)] = None,
    max_pages: Annotated[int | None, Form(ge=1, le=200)] = None,
) -> MarkdownResponse:
    data = await read_upload(file, settings)
    filename = file.filename or "upload"

    try:
        result = await pipeline.markdown(
            data,
            filename,
            thinking=thinking,
            dpi=dpi,
            max_pages=max_pages,
        )
    except HANDLED_ERRORS as exc:
        logger.warning("Markdown conversion failed for %s: %s", filename, exc)
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    return MarkdownResponse(
        request_id=current_request_id(),
        filename=filename,
        markdown=result.markdown,
        page_count=result.page_count,
        is_scanned=result.is_scanned,
        warnings=result.warnings,
        usage=Usage(
            prompt_tokens=result.prompt_tokens,
            completion_tokens=result.completion_tokens,
            total_tokens=result.prompt_tokens + result.completion_tokens,
        ),
        latency_s=round(result.latency_s, 3),
    )
