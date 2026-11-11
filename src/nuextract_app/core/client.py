"""Inference backends for NuExtract3.

Three deployment shapes are supported behind one interface:

* ``vllm`` / ``llamacpp`` — any OpenAI-compatible server that forwards
  ``chat_template_kwargs`` (vLLM on GPU, llama.cpp ``llama-server`` on CPU).
* ``ollama`` — the official ``numind/nuextract3`` Modelfile. Ollama exposes an
  OpenAI-compatible API and maps the template/instructions/mode to message roles.
* ``transformers`` — in-process inference on CPU, CUDA or Apple Metal (MPS).

``auto`` probes vLLM, then Ollama, then falls back to Transformers.
"""

from __future__ import annotations

import asyncio
import base64
import io
import json
import os
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol

import httpx

from nuextract_app.config import Settings
from nuextract_app.core.runtime import RuntimeInfo, detect_runtime, torch_dtype
from nuextract_app.logging import get_logger

logger = get_logger(__name__)

GenerationMode = Literal["structured", "markdown", "content", "template-generation"]

RETRYABLE_STATUS = {408, 409, 429, 500, 502, 503, 504}


class ModelBackendError(RuntimeError):
    """The inference backend failed or is unreachable."""


@dataclass(frozen=True)
class BackendInfo:
    """Human-readable description of the backend currently in use."""

    provider: str
    model: str
    device: str | None = None
    dtype: str | None = None
    endpoint: str | None = None
    note: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "provider": self.provider,
            "model": self.model,
            "device": self.device,
            "dtype": self.dtype,
            "endpoint": self.endpoint,
            "note": self.note,
        }


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

    def info(self) -> BackendInfo: ...


def _chat_template_kwargs(request: GenerationRequest) -> dict[str, Any]:
    kwargs: dict[str, Any] = {"enable_thinking": request.thinking}
    if request.template is not None:
        kwargs["template"] = json.dumps(request.template, ensure_ascii=False)
    if request.instructions:
        kwargs["instructions"] = request.instructions
    if request.mode != "structured":
        kwargs["mode"] = request.mode
    return kwargs


def _image_data_url(image: ImagePart) -> str:
    encoded = base64.b64encode(image.data).decode("ascii")
    return f"data:{image.mime_type};base64,{encoded}"


async def _post_with_retries(
    http: httpx.AsyncClient,
    url: str,
    payload: dict[str, Any],
    *,
    headers: dict[str, str] | None = None,
    attempts: int = 2,
    backend: str = "model server",
) -> httpx.Response:
    for attempt in range(attempts + 1):
        try:
            response = await http.post(url, json=payload, headers=headers)
            if response.status_code in RETRYABLE_STATUS and attempt < attempts:
                await asyncio.sleep(min(2**attempt, 8))
                continue
            if response.status_code >= 400:
                raise ModelBackendError(
                    f"{backend} returned HTTP {response.status_code}: {response.text[:500]}"
                )
            return response
        except httpx.HTTPError as exc:
            if attempt < attempts:
                logger.warning("%s request failed (%s); retrying", backend, exc)
                await asyncio.sleep(min(2**attempt, 8))
                continue
            raise ModelBackendError(f"Could not reach {backend} at {url}: {exc}") from exc

    raise ModelBackendError(f"{backend} request to {url} exhausted retries")


def _parse_chat_completion(response: httpx.Response, backend: str) -> GenerationResult:
    try:
        body = response.json()
        choice = body["choices"][0]["message"]
        text = (
            choice.get("content")
            or choice.get("reasoning_content")
            or choice.get("reasoning")
            or ""
        )
        usage = body.get("usage") or {}
    except (KeyError, IndexError, TypeError, ValueError) as exc:
        raise ModelBackendError(
            f"Unexpected response from {backend}: {response.text[:500]}"
        ) from exc

    return GenerationResult(
        content=text,
        prompt_tokens=int(usage.get("prompt_tokens", 0) or 0),
        completion_tokens=int(usage.get("completion_tokens", 0) or 0),
    )


class VLLMClient:
    """Async client for an OpenAI-compatible model server (vLLM, llama.cpp)."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._provider = "llamacpp" if settings.provider == "llamacpp" else "vllm"
        self._base_url = settings.openai_base_url
        self._client = httpx.AsyncClient(
            timeout=httpx.Timeout(settings.request_timeout_s, connect=10.0),
            limits=httpx.Limits(max_connections=16, max_keepalive_connections=8),
        )

    def info(self) -> BackendInfo:
        return BackendInfo(
            provider=self._provider,
            model=self._settings.model_name,
            device=None,
            endpoint=self._base_url,
            note="remote OpenAI-compatible server",
        )

    async def generate(self, request: GenerationRequest) -> GenerationResult:
        content: list[dict[str, Any]] = []
        for image in request.images:
            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": _image_data_url(image)},
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
        response = await _post_with_retries(
            self._client,
            f"{self._base_url}/chat/completions",
            payload,
            headers={"Authorization": f"Bearer {self._settings.model_api_key}"},
            attempts=max(0, self._settings.max_retries),
            backend="OpenAI-compatible server",
        )
        result = _parse_chat_completion(response, "OpenAI-compatible server")
        result.latency_s = time.perf_counter() - started
        return result

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


class OllamaClient:
    """Client for Ollama's OpenAI-compatible API using the official Modelfile.

    Ollama cannot receive ``chat_template_kwargs``, so the NuExtract3 Modelfile
    maps them to chat roles: ``template``, ``instructions``, ``mode`` and
    ``examples.*``. Thinking is toggled with ``reasoning_effort``.
    """

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._base_url = settings.ollama_openai_base_url
        self._native_url = settings.ollama_base_url
        self._client = httpx.AsyncClient(
            timeout=httpx.Timeout(settings.request_timeout_s, connect=10.0),
            limits=httpx.Limits(max_connections=8, max_keepalive_connections=4),
        )

    def info(self) -> BackendInfo:
        return BackendInfo(
            provider="ollama",
            model=self._settings.ollama_model,
            device="auto",
            dtype=None,
            endpoint=self._base_url,
            note="Ollama picks CPU/Metal/CUDA automatically",
        )

    def _messages(self, request: GenerationRequest) -> list[dict[str, Any]]:
        messages: list[dict[str, Any]] = []
        if request.template is not None:
            messages.append(
                {
                    "role": "template",
                    "content": json.dumps(request.template, ensure_ascii=False),
                }
            )
        if request.instructions:
            messages.append({"role": "instructions", "content": request.instructions})
        if request.mode != "structured":
            messages.append({"role": "mode", "content": request.mode})

        if request.images:
            content: list[dict[str, Any]] = [
                {"type": "image_url", "image_url": {"url": _image_data_url(image)}}
                for image in request.images
            ]
            if request.text:
                content.append({"type": "text", "text": request.text})
            messages.append({"role": "user", "content": content})
        else:
            messages.append({"role": "user", "content": request.text or ""})
        return messages

    def _payload(self, request: GenerationRequest) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self._settings.ollama_model,
            "messages": self._messages(request),
            "temperature": request.temperature,
            "max_tokens": request.max_tokens,
            "reasoning_effort": "medium" if request.thinking else "none",
        }
        if self._settings.ollama_num_ctx > 0:
            # Newer Ollama builds forward this to options.num_ctx; older builds
            # ignore it and use the Modelfile / OLLAMA_CONTEXT_LENGTH value.
            payload["num_ctx"] = self._settings.ollama_num_ctx
        return payload

    async def generate(self, request: GenerationRequest) -> GenerationResult:
        payload = self._payload(request)
        started = time.perf_counter()
        response = await _post_with_retries(
            self._client,
            f"{self._base_url}/chat/completions",
            payload,
            headers={"Authorization": f"Bearer {self._settings.model_api_key}"},
            attempts=max(0, self._settings.max_retries),
            backend="Ollama",
        )
        result = _parse_chat_completion(response, "Ollama")
        result.latency_s = time.perf_counter() - started
        return result

    async def health(self) -> bool:
        try:
            response = await self._client.get(f"{self._native_url}/api/tags", timeout=5.0)
        except httpx.HTTPError:
            return False
        if response.status_code >= 400:
            return False
        try:
            models = [str(item.get("name", "")) for item in response.json().get("models", [])]
        except (ValueError, AttributeError):
            return True
        if not models:
            return True

        wanted = self._settings.ollama_model
        families = {name.split(":")[0] for name in models}
        if wanted in models or wanted.split(":")[0] in families:
            return True
        logger.warning(
            "Ollama is reachable but %s is not pulled (available: %s)",
            wanted,
            ", ".join(models) or "none",
        )
        return False

    async def close(self) -> None:
        await self._client.aclose()


class TransformersClient:
    """Local, in-process backend for CPU, CUDA and Apple Metal (MPS).

    Requires the ``local`` extra (PyPI PyTorch: CUDA/MPS) or ``cpu`` extra
    (CPU-only wheels): ``uv sync --extra local`` / ``uv sync --extra cpu``.
    Device and dtype are resolved from ``NUEXTRACT_DEVICE`` / ``NUEXTRACT_DTYPE``.
    """

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._lock = threading.Lock()
        self._processor: Any = None
        self._model: Any = None
        self._torch: Any = None
        self._device: str | None = None
        self._runtime: RuntimeInfo | None = None

    def info(self) -> BackendInfo:
        runtime = self._runtime or detect_runtime(self._settings.device, self._settings.dtype)
        note = runtime.note
        if runtime.device == "cpu":
            note = (note + "; " if note else "") + (
                "CPU inference is functional but slow; prefer Ollama with a GGUF build"
            )
        return BackendInfo(
            provider="transformers",
            model=self._settings.model_name,
            device=runtime.device,
            dtype=runtime.dtype,
            endpoint=None,
            note=note,
        )

    def _load(self) -> None:
        if self._model is not None:
            return
        with self._lock:
            if self._model is not None:
                return

            # Must be set before torch is imported for unsupported MPS ops.
            if self._settings.device in {"auto", "mps"}:
                os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

            try:
                import torch
                from transformers import AutoProcessor
            except ImportError as exc:
                raise ModelBackendError(
                    "Transformers backend requires the 'local' or 'cpu' extra: "
                    "uv sync --extra local (CPU-only Linux: --extra cpu)"
                ) from exc

            try:
                from transformers import AutoModelForMultimodalLM
            except ImportError:  # older transformers naming
                from transformers import AutoModelForImageTextToText as AutoModelForMultimodalLM

            runtime = detect_runtime(self._settings.device, self._settings.dtype)
            self._runtime = runtime

            if runtime.device == "cpu" and self._settings.cpu_threads > 0:
                torch.set_num_threads(self._settings.cpu_threads)

            dtype = torch_dtype(torch, runtime.dtype)
            load_kwargs: dict[str, Any] = {
                "dtype": dtype,
                "trust_remote_code": self._settings.trust_remote_code,
            }
            # device_map shards across GPUs only when the user did not pin a device.
            if (
                runtime.device == "cuda"
                and self._settings.device == "auto"
                and runtime.cuda_devices > 1
            ):
                load_kwargs["device_map"] = "auto"

            logger.info(
                "Loading %s with Transformers (device=%s, dtype=%s)",
                self._settings.model_name,
                runtime.device,
                runtime.dtype,
            )
            self._processor = AutoProcessor.from_pretrained(
                self._settings.model_name,
                trust_remote_code=self._settings.trust_remote_code,
            )
            try:
                model = AutoModelForMultimodalLM.from_pretrained(
                    self._settings.model_name, **load_kwargs
                )
            except TypeError:
                # transformers < 4.56 used ``torch_dtype``.
                load_kwargs.pop("dtype", None)
                load_kwargs["torch_dtype"] = dtype
                model = AutoModelForMultimodalLM.from_pretrained(
                    self._settings.model_name, **load_kwargs
                )

            if "device_map" not in load_kwargs:
                model = model.to(runtime.device)
            model.eval()

            self._model = model
            self._torch = torch
            self._device = runtime.device

    async def generate(self, request: GenerationRequest) -> GenerationResult:
        return await asyncio.to_thread(self._generate_sync, request)

    def _generate_sync(self, request: GenerationRequest) -> GenerationResult:
        from PIL import Image

        self._load()
        torch = self._torch
        device = self._device or "cpu"

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
        ).to(device)

        max_new_tokens = request.max_tokens
        if self._settings.local_max_output_tokens > 0:
            max_new_tokens = min(max_new_tokens, self._settings.local_max_output_tokens)

        with torch.inference_mode():
            generation_kwargs: dict[str, Any] = {"max_new_tokens": max_new_tokens}
            if request.temperature > 0:
                generation_kwargs["do_sample"] = True
                generation_kwargs["temperature"] = request.temperature
            else:
                generation_kwargs["do_sample"] = False
            outputs = self._model.generate(**inputs, **generation_kwargs)

        generated = outputs[0][inputs["input_ids"].shape[-1] :]
        text = self._processor.batch_decode(
            [generated], skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
        text = text.strip()
        if request.thinking and "</think>" in text:
            text = text.rsplit("</think>", 1)[-1].strip()

        return GenerationResult(
            content=text,
            prompt_tokens=int(inputs["input_ids"].shape[-1]),
            completion_tokens=int(generated.shape[-1]),
            latency_s=time.perf_counter() - started,
        )

    async def health(self) -> bool:
        try:
            await asyncio.to_thread(self._load)
            return True
        except Exception as exc:  # model load errors are environment issues
            logger.warning("Transformers backend not ready: %s", exc)
            return False

    async def close(self) -> None:
        self._model = None
        self._processor = None


class AutoClient:
    """Resolve the best available backend at first use.

    Order: OpenAI-compatible server (vLLM/llama.cpp) → Ollama → Transformers.
    Resolution is cached, so it happens once per process.
    """

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._client: VisionLanguageClient | None = None
        self._lock = asyncio.Lock()

    async def _probe(self, url: str, headers: dict[str, str] | None = None) -> bool:
        try:
            async with httpx.AsyncClient(timeout=self._settings.auto_probe_timeout_s) as probe:
                response = await probe.get(url, headers=headers)
                return response.status_code < 500
        except httpx.HTTPError:
            return False

    async def resolve(self) -> VisionLanguageClient:
        if self._client is not None:
            return self._client
        async with self._lock:
            if self._client is not None:
                return self._client

            if await self._probe(
                f"{self._settings.openai_base_url}/models",
                headers={"Authorization": f"Bearer {self._settings.model_api_key}"},
            ):
                logger.info(
                    "auto backend: OpenAI-compatible server at %s", self._settings.openai_base_url
                )
                self._client = VLLMClient(self._settings)
            elif await self._probe(f"{self._settings.ollama_base_url}/api/tags"):
                logger.info("auto backend: Ollama at %s", self._settings.ollama_base_url)
                self._client = OllamaClient(self._settings)
            else:
                logger.info("auto backend: in-process Transformers (no server reachable)")
                self._client = TransformersClient(self._settings)
            return self._client

    def info(self) -> BackendInfo:
        if self._client is not None:
            return self._client.info()
        runtime = detect_runtime(self._settings.device, self._settings.dtype)
        return BackendInfo(
            provider="auto",
            model=self._settings.model_name,
            device=runtime.device,
            dtype=runtime.dtype,
            endpoint=self._settings.openai_base_url,
            note="not resolved yet; call /ready",
        )

    async def generate(self, request: GenerationRequest) -> GenerationResult:
        client = await self.resolve()
        return await client.generate(request)

    async def health(self) -> bool:
        client = await self.resolve()
        return await client.health()

    async def close(self) -> None:
        if self._client is not None:
            await self._client.close()


def build_client(settings: Settings) -> VisionLanguageClient:
    if settings.provider == "transformers":
        return TransformersClient(settings)
    if settings.provider == "ollama":
        return OllamaClient(settings)
    if settings.provider in {"vllm", "llamacpp"}:
        return VLLMClient(settings)
    return AutoClient(settings)
