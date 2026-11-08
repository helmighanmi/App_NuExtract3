"""Health and readiness endpoints."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, Response

from nuextract_app import __version__
from nuextract_app.api.deps import get_pipeline
from nuextract_app.api.schemas import HealthResponse, ReadyResponse
from nuextract_app.config import Settings, get_settings
from nuextract_app.core.pipeline import ExtractionPipeline

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse, summary="Liveness probe")
def health(settings: Annotated[Settings, Depends(get_settings)]) -> HealthResponse:
    return HealthResponse(
        status="ok",
        app=settings.app_name,
        version=__version__,
        provider=settings.provider,
        model=settings.model_name,
    )


@router.get(
    "/ready",
    response_model=ReadyResponse,
    summary="Readiness probe (checks the model backend)",
)
async def ready(
    response: Response,
    settings: Annotated[Settings, Depends(get_settings)],
    pipeline: Annotated[ExtractionPipeline, Depends(get_pipeline)],
) -> ReadyResponse:
    backend_ready = await pipeline.health()
    if not backend_ready:
        response.status_code = 503
    return ReadyResponse(
        status="ready" if backend_ready else "not_ready",
        backend=backend_ready,
        provider=settings.provider,
    )
