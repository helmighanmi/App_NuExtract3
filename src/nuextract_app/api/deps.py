"""Process-wide dependencies (settings, template registry, pipeline)."""

from __future__ import annotations

from functools import lru_cache

from nuextract_app.config import Settings, get_settings
from nuextract_app.core.client import build_client
from nuextract_app.core.pipeline import ExtractionPipeline
from nuextract_app.core.templates import TemplateRegistry


@lru_cache
def get_registry() -> TemplateRegistry:
    return TemplateRegistry(get_settings().templates_dir)


@lru_cache
def get_pipeline() -> ExtractionPipeline:
    settings: Settings = get_settings()
    return ExtractionPipeline(settings, build_client(settings), get_registry())


async def close_pipeline() -> None:
    if get_pipeline.cache_info().currsize:
        await get_pipeline().close()
