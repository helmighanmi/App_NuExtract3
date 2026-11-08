"""Structured extraction and batch extraction endpoints."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile

from nuextract_app.api.deps import get_pipeline
from nuextract_app.api.schemas import (
    BatchExtractionResponse,
    BatchItem,
    ExtractionResponse,
    Usage,
)
from nuextract_app.config import Settings, get_settings
from nuextract_app.core.client import ModelBackendError
from nuextract_app.core.document import DocumentError
from nuextract_app.core.pipeline import ExtractionError, ExtractionPipeline
from nuextract_app.core.templates import TemplateError
from nuextract_app.logging import current_request_id, get_logger

logger = get_logger(__name__)
router = APIRouter(prefix="/v1", tags=["extraction"])

MAX_BATCH_FILES = 20
HANDLED_ERRORS = (DocumentError, TemplateError, ExtractionError, ModelBackendError)


async def read_upload(file: UploadFile, settings: Settings) -> bytes:
    data = await file.read(settings.max_upload_bytes + 1)
    if not data:
        raise HTTPException(status_code=400, detail=f"{file.filename!r} is empty")
    if len(data) > settings.max_upload_bytes:
        raise HTTPException(
            status_code=413,
            detail=f"{file.filename!r} exceeds the {settings.max_upload_mb} MB upload limit",
        )
    return data


def _template_ref(
    template: str | None,
    template_name: str | None,
    settings: Settings,
) -> str:
    return template or template_name or settings.default_template


@router.post("/extract", response_model=ExtractionResponse, summary="Extract structured data")
async def extract_document(
    pipeline: Annotated[ExtractionPipeline, Depends(get_pipeline)],
    settings: Annotated[Settings, Depends(get_settings)],
    file: Annotated[UploadFile, File(description="PDF or image file")],
    template: Annotated[str | None, Form(description="Inline JSON template")] = None,
    template_name: Annotated[str | None, Form(description="Bundled template name")] = None,
    instructions: Annotated[str | None, Form(description="Extraction instructions")] = None,
    thinking: Annotated[bool | None, Form(description="Enable reasoning mode")] = None,
    dpi: Annotated[int | None, Form(ge=72, le=400)] = None,
    max_pages: Annotated[int | None, Form(ge=1, le=200)] = None,
    force_vision: Annotated[bool, Form()] = False,
    include_raw: Annotated[bool, Form()] = False,
) -> ExtractionResponse:
    data = await read_upload(file, settings)
    filename = file.filename or "upload"

    try:
        extracted = await pipeline.extract(
            data,
            filename,
            template=_template_ref(template, template_name, settings),
            instructions=instructions,
            thinking=thinking,
            dpi=dpi,
            max_pages=max_pages,
            force_vision=force_vision,
        )
    except HANDLED_ERRORS as exc:
        logger.warning("Extraction failed for %s: %s", filename, exc)
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    payload = extracted.to_dict(include_raw=include_raw)
    return ExtractionResponse(request_id=current_request_id(), filename=filename, **payload)


@router.post(
    "/extract/batch",
    response_model=BatchExtractionResponse,
    summary="Extract structured data from multiple files",
)
async def extract_batch(
    pipeline: Annotated[ExtractionPipeline, Depends(get_pipeline)],
    settings: Annotated[Settings, Depends(get_settings)],
    files: Annotated[list[UploadFile], File(description="PDF or image files")],
    template: Annotated[str | None, Form()] = None,
    template_name: Annotated[str | None, Form()] = None,
    instructions: Annotated[str | None, Form()] = None,
    thinking: Annotated[bool | None, Form()] = None,
    dpi: Annotated[int | None, Form(ge=72, le=400)] = None,
    max_pages: Annotated[int | None, Form(ge=1, le=200)] = None,
    force_vision: Annotated[bool, Form()] = False,
) -> BatchExtractionResponse:
    if len(files) > MAX_BATCH_FILES:
        raise HTTPException(status_code=400, detail=f"At most {MAX_BATCH_FILES} files per batch")

    template_ref = _template_ref(template, template_name, settings)
    items: list[BatchItem] = []
    for file in files:
        filename = file.filename or "upload"
        try:
            data = await read_upload(file, settings)
            extracted = await pipeline.extract(
                data,
                filename,
                template=template_ref,
                instructions=instructions,
                thinking=thinking,
                dpi=dpi,
                max_pages=max_pages,
                force_vision=force_vision,
            )
        except HTTPException as exc:
            items.append(BatchItem(filename=filename, ok=False, error=str(exc.detail)))
            continue
        except HANDLED_ERRORS as exc:
            logger.warning("Batch extraction failed for %s: %s", filename, exc)
            items.append(BatchItem(filename=filename, ok=False, error=str(exc)))
            continue

        items.append(
            BatchItem(
                filename=filename,
                ok=True,
                result=extracted.result,
                page_count=extracted.page_count,
                is_scanned=extracted.is_scanned,
                warnings=extracted.warnings,
                usage=Usage(
                    prompt_tokens=extracted.prompt_tokens,
                    completion_tokens=extracted.completion_tokens,
                    total_tokens=extracted.prompt_tokens + extracted.completion_tokens,
                ),
                latency_s=round(extracted.latency_s, 3),
            )
        )

    return BatchExtractionResponse(request_id=current_request_id(), items=items)
