"""FastAPI application factory."""

from __future__ import annotations

import time
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, RedirectResponse

from nuextract_app import __version__
from nuextract_app.api.deps import close_pipeline
from nuextract_app.api.routes import extract, health, markdown, templates
from nuextract_app.config import Settings, get_settings
from nuextract_app.core.client import ModelBackendError
from nuextract_app.core.document import DocumentError
from nuextract_app.core.pipeline import ExtractionError
from nuextract_app.core.runtime import detect_runtime
from nuextract_app.core.templates import TemplateError
from nuextract_app.logging import configure_logging, get_logger, request_id_var

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings = get_settings()
    configure_logging(settings.log_level)
    runtime = detect_runtime(settings.device, settings.dtype)
    logger.info(
        "Starting %s (provider=%s, model=%s, device=%s/%s, endpoint=%s)",
        settings.app_name,
        settings.provider,
        settings.model_name,
        runtime.device,
        runtime.dtype,
        settings.openai_base_url,
    )
    yield
    await close_pipeline()
    logger.info("Shutdown complete")


def create_app() -> FastAPI:
    settings: Settings = get_settings()
    app = FastAPI(
        title=settings.app_name,
        version=__version__,
        description=(
            "Document intelligence API powered by **NuExtract3**. Supports vLLM on GPU "
            "and Ollama, llama.cpp or in-process Transformers on CPU. "
            "Structured extraction, document-to-Markdown and template generation."
        ),
        lifespan=lifespan,
    )

    if settings.cors_origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=settings.cors_origins,
            allow_credentials=False,
            allow_methods=["*"],
            allow_headers=["*"],
        )

    @app.middleware("http")
    async def request_context(request: Request, call_next):  # type: ignore[no-untyped-def]
        request_id = request.headers.get("x-request-id") or uuid.uuid4().hex[:12]
        token = request_id_var.set(request_id)
        started = time.perf_counter()
        try:
            response = await call_next(request)
        finally:
            request_id_var.reset(token)
        response.headers["X-Request-ID"] = request_id
        response.headers["X-Response-Time-Ms"] = f"{(time.perf_counter() - started) * 1000:.1f}"
        return response

    @app.exception_handler(DocumentError)
    async def document_error_handler(request: Request, exc: DocumentError) -> JSONResponse:
        return JSONResponse(status_code=422, content={"detail": str(exc)})

    @app.exception_handler(TemplateError)
    async def template_error_handler(request: Request, exc: TemplateError) -> JSONResponse:
        return JSONResponse(status_code=422, content={"detail": str(exc)})

    @app.exception_handler(ExtractionError)
    async def extraction_error_handler(request: Request, exc: ExtractionError) -> JSONResponse:
        return JSONResponse(status_code=422, content={"detail": str(exc)})

    @app.exception_handler(ModelBackendError)
    async def backend_error_handler(request: Request, exc: ModelBackendError) -> JSONResponse:
        logger.error("Inference backend error: %s", exc)
        return JSONResponse(status_code=502, content={"detail": str(exc)})

    app.include_router(health.router)
    app.include_router(extract.router)
    app.include_router(markdown.router)
    app.include_router(templates.router)

    @app.get("/", include_in_schema=False)
    async def root() -> RedirectResponse:
        return RedirectResponse(url="/docs")

    return app


app = create_app()
