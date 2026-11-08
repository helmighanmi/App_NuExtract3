"""Template catalogue, generation and JSON-Schema conversion endpoints."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException

from nuextract_app.api.deps import get_pipeline, get_registry
from nuextract_app.api.schemas import (
    SchemaConversionRequest,
    SchemaConversionResponse,
    TemplateGenerationRequest,
    TemplateGenerationResponse,
    TemplateInfoModel,
    Usage,
)
from nuextract_app.core.pipeline import ExtractionPipeline
from nuextract_app.core.templates import (
    TemplateError,
    TemplateNotFoundError,
    TemplateRegistry,
    convert_json_schema,
)
from nuextract_app.logging import current_request_id, get_logger

logger = get_logger(__name__)
router = APIRouter(prefix="/v1/templates", tags=["templates"])


@router.get("", response_model=list[TemplateInfoModel], summary="List bundled templates")
def list_templates(
    registry: Annotated[TemplateRegistry, Depends(get_registry)],
) -> list[TemplateInfoModel]:
    return [
        TemplateInfoModel(name=info.name, description=info.description, template=info.template)
        for info in registry.list_templates()
    ]


@router.post(
    "/generate",
    response_model=TemplateGenerationResponse,
    summary="Generate a template from a natural-language description",
)
async def generate_template(
    payload: TemplateGenerationRequest,
    pipeline: Annotated[ExtractionPipeline, Depends(get_pipeline)],
) -> TemplateGenerationResponse:
    try:
        result = await pipeline.generate_template(payload.prompt, thinking=payload.thinking)
    except (TemplateError, ValueError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    return TemplateGenerationResponse(
        request_id=current_request_id(),
        template=result.template,
        raw=result.raw,
        usage=Usage(
            prompt_tokens=result.prompt_tokens,
            completion_tokens=result.completion_tokens,
            total_tokens=result.prompt_tokens + result.completion_tokens,
        ),
        latency_s=round(result.latency_s, 3),
    )


@router.post(
    "/convert",
    response_model=SchemaConversionResponse,
    summary="Convert a JSON Schema / Pydantic schema to a NuExtract template",
)
def convert_schema(payload: SchemaConversionRequest) -> SchemaConversionResponse:
    try:
        template, dropped = convert_json_schema(payload.schema_)
    except TemplateError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return SchemaConversionResponse(template=template, dropped=dropped)


@router.get("/{name}", response_model=TemplateInfoModel, summary="Get a bundled template")
def get_template(
    name: str,
    registry: Annotated[TemplateRegistry, Depends(get_registry)],
) -> TemplateInfoModel:
    try:
        info = registry.get_info(name)
    except TemplateNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return TemplateInfoModel(name=info.name, description=info.description, template=info.template)
