"""Inference backends for NuExtract3.

The production path is vLLM (OpenAI-compatible). A Transformers backend is
provided as a fallback for laptops / notebooks, but it is single-request and
not intended for serving.
"""

from __future__ import annotations

import asyncio
import base64
import json
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol

import httpx

from nuextract_app.config import Settings
from nuextract_app.logging import get_logger

logger = get_logger(__name__)

GenerationMode = Literal["structured", "markdown", "content", "template-generation"]

RETRYABLE_STATUS = {408, 409, 429, 500, 502, 503, 504}


class ModelBackendError(RuntimeError):
    """The inference backend failed or is unreachable."""


@dataclass(frozen=True)
class ImagePart:
    data: bytes
    mime_type: str = "image/png"


@dataclass
class GenerationRequest:
    images: list[ImagePart] = field(default_factory=list)
    text: str | None = None
    template: Any = None
    instructions: str | None = None
    mode: GenerationMode = "structured"
    thinking: bool = False
    temperature: float = 0.2
    max_tokens: int = 16384


@dataclass
class GenerationResult:
    content: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_s: float = 0.0


class VisionLanguageClient(Protocol):
    async def generate(self, request: GenerationRequest) -> GenerationResult: ...

    async def health(self) -> bool: ...

    async def close(self) -> None: ...


def _chat_template_kwargs(request: GenerationRequest) -> dict[str, Any]:
    kwargs: dict[str, Any] = {"enable_thinking": request.thinking}
    if request.template is not None:
        kwargs["template"] = json.dumps(request.template, ensure_ascii=False)
    if request.instructions:
        kwargs["instructions"] = request.instructions
    if request.mode != "structured":
        kwargs["mode"] = request.mode
    return kwargs


class VLLMClient:
    """Async client for a vLLM OpenAI-compatible server."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._base_url = settings.openai_base_url
        self._client = httpx.AsyncClient(
            timeout=httpx.Timeout(settings.request_timeout_s, connect=10.0),
            limits=httpx.Limits(max_connections=16, max_keepalive_connections=8),
        )

    async def generate(self, request: GenerationRequest) -> GenerationResult:
        content: list[dict[str, Any]] = []
        for image in request.images:
            encoded = base64.b64encode(image.data).decode("ascii")
            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:{image.mime_type};base64,{encoded}"},
                }
            )
        if request.text:
            content.append({"type": "text", "text": request.text})
        if not content:
            raise ModelBackendError("Generation request contains no text or images")

        payload: dict[str, Any] = {
            "model": self._settings.model_name,
            "messages": [{"role": "user", "content": content}],
            "temperature": request.temperature,
            "max_tokens": request.max_tokens,
            "chat_template_kwargs": _chat_template_kwargs(request),
        }

        started = time.perf_counter()
        response = await self._post_with_retries("/chat/completions", payload)
        latency = time.perf_counter() - started

        try:
            body = response.json()
            choice = body["choices"][0]["message"]
            text = choice.get("content") or choice.get("reasoning_content") or ""
            usage = body.get("usage", {})
        except (KeyError, IndexError, ValueError) as exc:
            raise ModelBackendError(
                f"Unexpected response from vLLM: {response.text[:500]}"
            ) from exc

        return GenerationResult(
            content=text,
            prompt_tokens=int(usage.get("prompt_tokens", 0) or 0),
            completion_tokens=int(usage.get("completion_tokens", 0) or 0),
            latency_s=latency,
        )

    async def _post_with_retries(self, path: str, payload: dict[str, Any]) -> httpx.Response:
        url = f"{self._base_url}{path}"
        headers = {"Authorization": f"Bearer {self._settings.model_api_key}"}
        attempts = max(0, self._settings.max_retries)

        for attempt in range(attempts + 1):
            try:
                response = await self._client.post(url, json=payload, headers=headers)
                if response.status_code in RETRYABLE_STATUS and attempt < attempts:
                    await asyncio.sleep(min(2**attempt, 8))
                    continue
                if response.status_code >= 400:
                    raise ModelBackendError(
                        f"vLLM returned HTTP {response.status_code}: {response.text[:500]}"
                    )
                return response
            except httpx.HTTPError as exc:
                if attempt < attempts:
                    logger.warning("vLLM request failed (%s); retrying", exc)
                    await asyncio.sleep(min(2**attempt, 8))
                    continue
                raise ModelBackendError(f"Could not reach vLLM at {url}: {exc}") from exc

        raise ModelBackendError(f"vLLM request to {url} exhausted retries")

    async def health(self) -> bool:
        try:
            response = await self._client.get(
                f"{self._base_url}/models",
                headers={"Authorization": f"Bearer {self._settings.model_api_key}"},
                timeout=5.0,
            )
            return response.status_code < 500
        except httpx.HTTPError:
            return False

    async def close(self) -> None:
        await self._client.aclose()


class TransformersClient:
    """Local, in-process fallback. Enable with ``NUEXTRACT_PROVIDER=transformers``.

    Requires the ``local`` extra: ``uv sync --extra local``.
    """

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._lock = threading.Lock()
        self._processor: Any = None
        self._model: Any = None
        self._torch: Any = None

    def _load(self) -> None:
        if self._model is not None:
            return
        with self._lock:
            if self._model is not None:
                return
            try:
                import torch
                from transformers import AutoProcessor
            except ImportError as exc:
                raise ModelBackendError(
                    "Transformers backend requires the 'local' extra: uv sync --extra local"
                ) from exc

            try:
                from transformers import AutoModelForMultimodalLM
            except ImportError:  # older transformers naming
                from transformers import AutoModelForImageTextToText as AutoModelForMultimodalLM

            logger.info(
                "Loading %s with Transformers (this can take a while)", self._settings.model_name
            )
            self._processor = AutoProcessor.from_pretrained(self._settings.model_name)
            self._model = AutoModelForMultimodalLM.from_pretrained(
                self._settings.model_name,
                device_map="auto",
                torch_dtype="auto",
            )
            self._model.eval()
            self._torch = torch

    async def generate(self, request: GenerationRequest) -> GenerationResult:
        return await asyncio.to_thread(self._generate_sync, request)

    def _generate_sync(self, request: GenerationRequest) -> GenerationResult:
        import io

        from PIL import Image

        self._load()
        torch = self._torch

        content: list[dict[str, Any]] = []
        for image in request.images:
            with Image.open(io.BytesIO(image.data)) as img:
                content.append({"type": "image", "image": img.convert("RGB").copy()})
        if request.text:
            content.append({"type": "text", "text": request.text})

        messages = [{"role": "user", "content": content}]
        started = time.perf_counter()
        inputs = self._processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            **_chat_template_kwargs(request),
        ).to(self._model.device)

        with torch.inference_mode():
            generation_kwargs: dict[str, Any] = {"max_new_tokens": request.max_tokens}
            if request.temperature > 0:
                generation_kwargs["do_sample"] = True
                generation_kwargs["temperature"] = request.temperature
            else:
                generation_kwargs["do_sample"] = False
            outputs = self._model.generate(**inputs, **generation_kwargs)
        generated = outputs[0][inputs["input_ids"].shape[-1] :]
        text = self._processor.decode(generated, skip_special_tokens=True)

        return GenerationResult(
            content=text,
            prompt_tokens=int(inputs["input_ids"].shape[-1]),
            completion_tokens=int(generated.shape[-1]),
            latency_s=time.perf_counter() - started,
        )

    async def health(self) -> bool:
        try:
            self._load()
            return True
        except Exception as exc:  # model load errors are environment issues
            logger.warning("Transformers backend not ready: %s", exc)
            return False

    async def close(self) -> None:
        self._model = None
        self._processor = None


def build_client(settings: Settings) -> VisionLanguageClient:
    if settings.provider == "transformers":
        return TransformersClient(settings)
    return VLLMClient(settings)
