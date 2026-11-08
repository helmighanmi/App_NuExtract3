"""Thin synchronous client for the NuExtract3 Studio API (used by Streamlit)."""

from __future__ import annotations

import json
import mimetypes
from typing import Any

import httpx


class ApiError(RuntimeError):
    def __init__(self, message: str, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code


class ApiClient:
    def __init__(self, base_url: str, timeout_s: float = 300.0) -> None:
        self.base_url = base_url.rstrip("/")
        self._client = httpx.Client(timeout=httpx.Timeout(timeout_s, connect=10.0))

    def close(self) -> None:
        self._client.close()

    def _url(self, path: str) -> str:
        return f"{self.base_url}{path}"

    def _raise(self, response: httpx.Response) -> None:
        try:
            detail = response.json().get("detail", response.text)
        except ValueError:
            detail = response.text
        raise ApiError(f"HTTP {response.status_code}: {detail}", response.status_code)

    def health(self) -> dict[str, Any]:
        response = self._client.get(self._url("/health"))
        if response.status_code >= 400:
            self._raise(response)
        return response.json()

    def ready(self) -> tuple[bool, dict[str, Any]]:
        response = self._client.get(self._url("/ready"))
        try:
            payload = response.json()
        except ValueError:
            payload = {"status": "unknown"}
        return response.status_code < 400, payload

    def list_templates(self) -> list[dict[str, Any]]:
        response = self._client.get(self._url("/v1/templates"))
        if response.status_code >= 400:
            self._raise(response)
        return response.json()

    def extract(
        self,
        filename: str,
        data: bytes,
        *,
        template: dict[str, Any] | list[Any],
        instructions: str | None = None,
        thinking: bool = False,
        force_vision: bool = False,
        dpi: int = 200,
        max_pages: int = 40,
    ) -> dict[str, Any]:
        mime = mimetypes.guess_type(filename)[0] or "application/octet-stream"
        form: dict[str, str] = {
            "template": json.dumps(template, ensure_ascii=False),
            "thinking": str(thinking).lower(),
            "force_vision": str(force_vision).lower(),
            "dpi": str(dpi),
            "max_pages": str(max_pages),
        }
        if instructions:
            form["instructions"] = instructions

        response = self._client.post(
            self._url("/v1/extract"),
            files={"file": (filename, data, mime)},
            data=form,
        )
        if response.status_code >= 400:
            self._raise(response)
        return response.json()

    def markdown(
        self,
        filename: str,
        data: bytes,
        *,
        thinking: bool = False,
        dpi: int = 200,
        max_pages: int = 40,
    ) -> dict[str, Any]:
        mime = mimetypes.guess_type(filename)[0] or "application/octet-stream"
        response = self._client.post(
            self._url("/v1/markdown"),
            files={"file": (filename, data, mime)},
            data={
                "thinking": str(thinking).lower(),
                "dpi": str(dpi),
                "max_pages": str(max_pages),
            },
        )
        if response.status_code >= 400:
            self._raise(response)
        return response.json()

    def generate_template(self, prompt: str, *, thinking: bool = False) -> dict[str, Any]:
        response = self._client.post(
            self._url("/v1/templates/generate"),
            json={"prompt": prompt, "thinking": thinking},
        )
        if response.status_code >= 400:
            self._raise(response)
        return response.json()

    def convert_schema(self, schema: dict[str, Any]) -> dict[str, Any]:
        response = self._client.post(
            self._url("/v1/templates/convert"),
            json={"schema": schema},
        )
        if response.status_code >= 400:
            self._raise(response)
        return response.json()
