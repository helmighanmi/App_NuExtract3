"""API request/response models."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class Usage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class PageTraceModel(BaseModel):
    number: int
    route: str
    chars: int


class ExtractionResponse(BaseModel):
    request_id: str
    filename: str
    result: Any
    template: Any
    page_count: int
    is_scanned: bool
    pages: list[PageTraceModel]
    warnings: list[str] = Field(default_factory=list)
    usage: Usage
    latency_s: float
    raw: str | None = None


class BatchItem(BaseModel):
    filename: str
    ok: bool
    result: Any = None
    error: str | None = None
    page_count: int = 0
    is_scanned: bool = False
    warnings: list[str] = Field(default_factory=list)
    usage: Usage = Field(default_factory=Usage)
    latency_s: float = 0.0


class BatchExtractionResponse(BaseModel):
    request_id: str
    items: list[BatchItem]


class MarkdownResponse(BaseModel):
    request_id: str
    filename: str
    markdown: str
    page_count: int
    is_scanned: bool
    warnings: list[str] = Field(default_factory=list)
    usage: Usage
    latency_s: float


class TemplateInfoModel(BaseModel):
    name: str
    description: str = ""
    template: Any


class TemplateGenerationRequest(BaseModel):
    prompt: str = Field(min_length=3, max_length=4000)
    thinking: bool = False


class TemplateGenerationResponse(BaseModel):
    request_id: str
    template: Any
    raw: str
    usage: Usage
    latency_s: float


class SchemaConversionRequest(BaseModel):
    schema_: dict[str, Any] = Field(alias="schema")


class SchemaConversionResponse(BaseModel):
    template: Any
    dropped: list[str] = Field(default_factory=list)


class HealthResponse(BaseModel):
    status: str
    app: str
    version: str
    provider: str
    model: str


class ReadyResponse(BaseModel):
    status: str
    backend: bool
    provider: str
