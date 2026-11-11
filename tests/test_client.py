from __future__ import annotations

import json
from typing import Any

import httpx
import pytest

from nuextract_app.config import Settings
from nuextract_app.core.client import (
    AutoClient,
    BackendInfo,
    GenerationRequest,
    ImagePart,
    OllamaClient,
    TransformersClient,
    VLLMClient,
    _chat_template_kwargs,
    build_client,
)
from nuextract_app.core.runtime import detect_runtime


def _mock_http(handler: Any) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


def _chat_response(content: str) -> dict[str, Any]:
    return {
        "choices": [{"message": {"content": content}}],
        "usage": {"prompt_tokens": 11, "completion_tokens": 7},
    }


def test_chat_template_kwargs_structured() -> None:
    request = GenerationRequest(
        template={"total": "number"},
        instructions="Use ISO dates",
        mode="structured",
        thinking=True,
    )
    assert _chat_template_kwargs(request) == {
        "enable_thinking": True,
        "template": '{"total": "number"}',
        "instructions": "Use ISO dates",
    }


def test_chat_template_kwargs_markdown_mode() -> None:
    kwargs = _chat_template_kwargs(GenerationRequest(mode="markdown", thinking=False))
    assert kwargs == {"enable_thinking": False, "mode": "markdown"}


def test_build_client_selects_backend() -> None:
    assert isinstance(build_client(Settings(provider="vllm")), VLLMClient)
    assert isinstance(build_client(Settings(provider="llamacpp")), VLLMClient)
    assert isinstance(build_client(Settings(provider="ollama")), OllamaClient)
    assert isinstance(build_client(Settings(provider="transformers")), TransformersClient)
    assert isinstance(build_client(Settings(provider="auto")), AutoClient)


def test_vllm_client_reports_provider() -> None:
    llamacpp = build_client(
        Settings(provider="llamacpp", model_base_url="http://localhost:8080/v1")
    )
    assert llamacpp.info() == BackendInfo(
        provider="llamacpp",
        model="numind/NuExtract3",
        device=None,
        endpoint="http://localhost:8080/v1",
        note="remote OpenAI-compatible server",
    )


async def test_vllm_payload_carries_template_kwargs() -> None:
    settings = Settings(provider="vllm", model_base_url="http://server:8000/v1")
    client = VLLMClient(settings)
    seen: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["auth"] = request.headers.get("authorization")
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json=_chat_response('{"total": 12.4}'))

    await client._client.aclose()
    client._client = _mock_http(handler)
    try:
        result = await client.generate(
            GenerationRequest(
                images=[ImagePart(b"png-bytes", "image/png")],
                template={"total": "number"},
                thinking=False,
                max_tokens=512,
            )
        )
    finally:
        await client.close()

    assert result.content == '{"total": 12.4}'
    assert result.prompt_tokens == 11
    assert seen["url"] == "http://server:8000/v1/chat/completions"
    assert seen["auth"] == "Bearer EMPTY"
    body = seen["body"]
    assert body["chat_template_kwargs"]["template"] == '{"total": "number"}'
    assert body["chat_template_kwargs"]["enable_thinking"] is False
    assert body["messages"][0]["content"][0]["type"] == "image_url"
    assert body["messages"][0]["content"][0]["image_url"]["url"].startswith(
        "data:image/png;base64,"
    )


def test_ollama_messages_use_nuextract_roles() -> None:
    client = OllamaClient(Settings(provider="ollama"))
    messages = client._messages(
        GenerationRequest(
            template={"store": "string"},
            instructions="Keep names verbatim",
            mode="structured",
            text="Receipt text",
        )
    )

    assert [message["role"] for message in messages] == ["template", "instructions", "user"]
    assert messages[0]["content"] == '{"store": "string"}'
    assert messages[1]["content"] == "Keep names verbatim"
    assert messages[2]["content"] == "Receipt text"


def test_ollama_messages_put_images_in_user_content() -> None:
    client = OllamaClient(Settings(provider="ollama"))
    messages = client._messages(
        GenerationRequest(images=[ImagePart(b"png")], mode="markdown", text=None)
    )

    assert messages[0] == {"role": "mode", "content": "markdown"}
    content = messages[1]["content"]
    assert content[0]["type"] == "image_url"
    assert content[0]["image_url"]["url"].startswith("data:image/png;base64,")


async def test_ollama_generate_and_health() -> None:
    settings = Settings(
        provider="ollama",
        ollama_base_url="http://ollama:11434",
        ollama_model="nuextract3:q4_k_m",
    )
    client = OllamaClient(settings)
    seen: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        if request.url.path.endswith("/api/tags"):
            return httpx.Response(200, json={"models": [{"name": "nuextract3:q4_k_m"}]})
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json=_chat_response('{"store": "ACME"}'))

    await client._client.aclose()
    client._client = _mock_http(handler)
    try:
        healthy = await client.health()
        result = await client.generate(
            GenerationRequest(
                template={"store": "string"},
                text="ACME receipt",
                thinking=True,
                temperature=0.6,
                max_tokens=1024,
            )
        )
    finally:
        await client.close()

    assert healthy is True
    assert result.content == '{"store": "ACME"}'
    assert seen["url"] == "http://ollama:11434/v1/chat/completions"
    assert seen["body"]["model"] == "nuextract3:q4_k_m"
    assert seen["body"]["reasoning_effort"] == "medium"
    assert seen["body"]["num_ctx"] == 32768


async def test_ollama_health_fails_when_model_missing() -> None:
    client = OllamaClient(Settings(provider="ollama", ollama_model="nuextract3:q4_k_m"))

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"models": [{"name": "llama3:8b"}]})

    await client._client.aclose()
    client._client = _mock_http(handler)
    try:
        assert await client.health() is False
    finally:
        await client.close()


async def test_auto_client_prefers_openai_server(monkeypatch: pytest.MonkeyPatch) -> None:
    async def probe(self: AutoClient, url: str, headers: Any = None) -> bool:
        return "/v1/models" in url

    monkeypatch.setattr(AutoClient, "_probe", probe)
    client = AutoClient(Settings(provider="auto"))
    resolved = await client.resolve()
    try:
        assert isinstance(resolved, VLLMClient)
        assert client.info().provider == "vllm"
    finally:
        await client.close()


async def test_auto_client_falls_back_to_ollama(monkeypatch: pytest.MonkeyPatch) -> None:
    async def probe(self: AutoClient, url: str, headers: Any = None) -> bool:
        return url.endswith("/api/tags")

    monkeypatch.setattr(AutoClient, "_probe", probe)
    client = AutoClient(Settings(provider="auto"))
    resolved = await client.resolve()
    try:
        assert isinstance(resolved, OllamaClient)
        assert client.info().provider == "ollama"
    finally:
        await client.close()


async def test_auto_client_falls_back_to_transformers(monkeypatch: pytest.MonkeyPatch) -> None:
    async def probe(self: AutoClient, url: str, headers: Any = None) -> bool:
        return False

    monkeypatch.setattr(AutoClient, "_probe", probe)
    client = AutoClient(Settings(provider="auto"))
    resolved = await client.resolve()
    try:
        assert isinstance(resolved, TransformersClient)
        assert client.info().provider == "transformers"
    finally:
        await client.close()


def test_runtime_never_requires_torch_for_cpu() -> None:
    runtime = detect_runtime("cpu", "auto")
    assert runtime.device == "cpu"


def test_runtime_downgrades_float16_on_cpu() -> None:
    runtime = detect_runtime("cpu", "float16")
    assert runtime.dtype == "float32"
