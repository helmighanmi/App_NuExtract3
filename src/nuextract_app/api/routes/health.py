"""Health and readiness endpoints."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, Response

from nuextract_app import __version__
from nuextract_app.api.deps import get_pipeline
from nuextract_app.api.schemas import HealthResponse, ReadyResponse
from nuextract_app.config import Settings, get_settings
from nuextract_app.core.pipeline import ExtractionPipeline
from nuextract_app.core.runtime import detect_runtime, torch_installed

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse, summary="Liveness probe")
def health(settings: Annotated[Settings, Depends(get_settings)]) -> HealthResponse:
    # Keep liveness cheap: do not import torch here. /ready reports the
    # resolved device/dtype after probing the backend.
    return HealthResponse(
        status="ok",
        app=settings.app_name,
        version=__version__,
        provider=settings.provider,
        model=settings.model_name,
        backend=settings.provider,
        device=None if settings.device == "auto" else settings.device,
        dtype=None if settings.dtype == "auto" else settings.dtype,
        torch_available=torch_installed(),
    )


@router.get(
    "/ready",
    response_model=ReadyResponse,
    summary="Readiness probe (checks and describes the model backend)",
)
async def ready(
    response: Response,
    settings: Annotated[Settings, Depends(get_settings)],
    pipeline: Annotated[ExtractionPipeline, Depends(get_pipeline)],
) -> ReadyResponse:
    backend_ready = await pipeline.health()
    if not backend_ready:
        response.status_code = 503

    getter = getattr(pipeline, "backend_info", None)
    info = getter() if callable(getter) else None
    runtime = detect_runtime(settings.device, settings.dtype)

    return ReadyResponse(
        status="ready" if backend_ready else "not_ready",
        backend=backend_ready,
        provider=settings.provider,
        resolved_provider=getattr(info, "provider", None),
        device=getattr(info, "device", None) or runtime.device,
        dtype=getattr(info, "dtype", None) or runtime.dtype,
        endpoint=getattr(info, "endpoint", None),
        note=getattr(info, "note", None) or runtime.note,
    )
