"""Extraction orchestration: document -> model -> validated structure."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from nuextract_app.config import Settings
from nuextract_app.core.client import (
    BackendInfo,
    GenerationRequest,
    ImagePart,
    ModelBackendError,
    VisionLanguageClient,
)
from nuextract_app.core.document import DocumentPage, load_document
from nuextract_app.core.postprocess import (
    JsonParseError,
    merge_structured_results,
    parse_json,
)
from nuextract_app.core.templates import (
    TemplateRegistry,
    ensure_template_shape,
    validate_template_shape,
)
from nuextract_app.logging import get_logger

logger = get_logger(__name__)


class ExtractionError(RuntimeError):
    """The pipeline could not produce a usable result."""


@dataclass
class PageTrace:
    number: int
    route: str  # "text" | "image"
    chars: int

    def to_dict(self) -> dict[str, Any]:
        return {"number": self.number, "route": self.route, "chars": self.chars}


@dataclass
class ExtractionResult:
    result: Any
    template: Any
    page_count: int
    is_scanned: bool
    pages: list[PageTrace]
    warnings: list[str] = field(default_factory=list)
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_s: float = 0.0
    raw: str | None = None

    def to_dict(self, include_raw: bool = False) -> dict[str, Any]:
        payload = {
            "result": self.result,
            "template": self.template,
            "page_count": self.page_count,
            "is_scanned": self.is_scanned,
            "pages": [page.to_dict() for page in self.pages],
            "warnings": self.warnings,
            "usage": {
                "prompt_tokens": self.prompt_tokens,
                "completion_tokens": self.completion_tokens,
                "total_tokens": self.prompt_tokens + self.completion_tokens,
            },
            "latency_s": round(self.latency_s, 3),
        }
        if include_raw:
            payload["raw"] = self.raw
        return payload


@dataclass
class MarkdownResult:
    markdown: str
    page_count: int
    is_scanned: bool
    warnings: list[str] = field(default_factory=list)
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_s: float = 0.0


@dataclass
class TemplateGenerationResult:
    template: Any
    raw: str
    warnings: list[str] = field(default_factory=list)
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_s: float = 0.0


class ExtractionPipeline:
    def __init__(
        self,
        settings: Settings,
        client: VisionLanguageClient,
        registry: TemplateRegistry,
    ) -> None:
        self.settings = settings
        self.client = client
        self.registry = registry

    # ------------------------------------------------------------------ extract
    async def extract(
        self,
        data: bytes,
        filename: str,
        *,
        template: str | dict[str, Any] | list[Any] | None = None,
        instructions: str | None = None,
        thinking: bool | None = None,
        dpi: int | None = None,
        max_pages: int | None = None,
        force_vision: bool = False,
    ) -> ExtractionResult:
        started = time.perf_counter()
        thinking = self.settings.enable_thinking if thinking is None else thinking

        loaded = load_document(
            data,
            filename,
            dpi=dpi or self.settings.dpi,
            max_pages=max_pages or self.settings.max_pages,
            native_text_min_chars=self.settings.native_text_min_chars,
        )
        template_ref = template if template is not None else self.settings.default_template
        template_obj = self.registry.resolve(template_ref)
        validate_template_shape(template_obj)

        warnings = list(loaded.warnings)
        parsed_chunks: list[Any] = []
        raw_chunks: list[str] = []
        prompt_tokens = completion_tokens = 0
        traces: list[PageTrace] = []

        for chunk in _chunk(loaded.pages, self.settings.pages_per_request):
            request, chunk_traces = self._build_extraction_request(
                chunk,
                template_obj,
                instructions=instructions,
                thinking=thinking,
                force_vision=force_vision,
            )
            traces.extend(chunk_traces)
            generation = await self.client.generate(request)
            prompt_tokens += generation.prompt_tokens
            completion_tokens += generation.completion_tokens
            raw_chunks.append(generation.content)

            try:
                parsed_chunks.append(parse_json(generation.content))
            except JsonParseError as exc:
                warnings.append(f"Could not parse response for pages {_page_range(chunk)}: {exc}")

        if not parsed_chunks:
            raise ExtractionError(
                "The model did not return any parseable JSON. Check vLLM logs and the model name."
            )

        merged = merge_structured_results(parsed_chunks)
        merged, shape_warnings = ensure_template_shape(merged, template_obj)
        warnings.extend(shape_warnings)

        return ExtractionResult(
            result=merged,
            template=template_obj,
            page_count=loaded.page_count,
            is_scanned=loaded.is_scanned,
            pages=traces,
            warnings=warnings,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            latency_s=time.perf_counter() - started,
            raw="\n\n".join(raw_chunks),
        )

    def _build_extraction_request(
        self,
        pages: list[DocumentPage],
        template: Any,
        *,
        instructions: str | None,
        thinking: bool,
        force_vision: bool,
    ) -> tuple[GenerationRequest, list[PageTrace]]:
        use_images = force_vision or any(page.is_scanned for page in pages)
        images: list[ImagePart] = []
        text_sections: list[str] = []
        traces: list[PageTrace] = []

        for page in pages:
            if use_images and page.png is not None:
                images.append(ImagePart(page.png))
                traces.append(PageTrace(number=page.number, route="image", chars=len(page.text)))
            elif page.text:
                text_sections.append(f'<page number="{page.number}">\n{page.text}\n</page>')
                traces.append(PageTrace(number=page.number, route="text", chars=len(page.text)))
            else:
                traces.append(PageTrace(number=page.number, route="text", chars=0))

        request = GenerationRequest(
            images=images,
            text="\n\n".join(text_sections) or None,
            template=template,
            instructions=instructions,
            mode="structured",
            thinking=thinking,
            temperature=(
                self.settings.thinking_temperature if thinking else self.settings.temperature
            ),
            max_tokens=self.settings.max_output_tokens,
        )
        return request, traces

    # ----------------------------------------------------------------- markdown
    async def markdown(
        self,
        data: bytes,
        filename: str,
        *,
        thinking: bool | None = None,
        dpi: int | None = None,
        max_pages: int | None = None,
    ) -> MarkdownResult:
        started = time.perf_counter()
        thinking = self.settings.enable_thinking if thinking is None else thinking
        loaded = load_document(
            data,
            filename,
            dpi=dpi or self.settings.dpi,
            max_pages=max_pages or self.settings.max_pages,
            native_text_min_chars=self.settings.native_text_min_chars,
        )

        parts: list[str] = []
        warnings = list(loaded.warnings)
        prompt_tokens = completion_tokens = 0

        for chunk in _chunk(loaded.pages, self.settings.pages_per_request):
            images = [ImagePart(page.png) for page in chunk if page.png is not None]
            fallback_text = "\n\n".join(
                f'<page number="{page.number}">\n{page.text}\n</page>'
                for page in chunk
                if page.png is None and page.text
            )
            request = GenerationRequest(
                images=images,
                text=fallback_text or None,
                mode="markdown",
                thinking=thinking,
                temperature=self.settings.markdown_temperature,
                max_tokens=self.settings.max_output_tokens,
            )
            generation = await self.client.generate(request)
            prompt_tokens += generation.prompt_tokens
            completion_tokens += generation.completion_tokens
            if generation.content.strip():
                parts.append(generation.content.strip())
            else:
                warnings.append(f"Empty markdown for pages {_page_range(chunk)}")

        if not parts:
            raise ExtractionError("The model did not return any markdown content.")

        return MarkdownResult(
            markdown="\n\n---\n\n".join(parts),
            page_count=loaded.page_count,
            is_scanned=loaded.is_scanned,
            warnings=warnings,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            latency_s=time.perf_counter() - started,
        )

    # ---------------------------------------------------------------- templates
    async def generate_template(
        self,
        prompt: str,
        *,
        thinking: bool = False,
    ) -> TemplateGenerationResult:
        request = GenerationRequest(
            text=prompt,
            mode="template-generation",
            thinking=thinking,
            temperature=(
                self.settings.thinking_temperature if thinking else self.settings.temperature
            ),
            max_tokens=self.settings.max_output_tokens,
        )
        generation = await self.client.generate(request)
        template = parse_json(generation.content)
        validate_template_shape(template)
        return TemplateGenerationResult(
            template=template,
            raw=generation.content,
            prompt_tokens=generation.prompt_tokens,
            completion_tokens=generation.completion_tokens,
            latency_s=generation.latency_s,
        )

    def backend_info(self) -> BackendInfo:
        getter = getattr(self.client, "info", None)
        if callable(getter):
            return getter()
        return BackendInfo(provider=self.settings.provider, model=self.settings.model_name)

    async def health(self) -> bool:
        try:
            return await self.client.health()
        except ModelBackendError:
            return False

    async def close(self) -> None:
        await self.client.close()


def _chunk(pages: list[DocumentPage], size: int) -> list[list[DocumentPage]]:
    if size <= 0 or size >= len(pages):
        return [pages]
    return [pages[index : index + size] for index in range(0, len(pages), size)]


def _page_range(pages: list[DocumentPage]) -> str:
    if len(pages) == 1:
        return str(pages[0].number)
    return f"{pages[0].number}-{pages[-1].number}"
