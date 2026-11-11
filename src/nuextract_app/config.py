"""Typed application configuration (12-factor, env-driven)."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# ``auto`` probes vLLM, then Ollama, then falls back to in-process Transformers.
Provider = Literal["auto", "vllm", "ollama", "llamacpp", "transformers"]
Device = Literal["auto", "cpu", "cuda", "mps"]
DType = Literal["auto", "float32", "bfloat16", "float16"]


class Settings(BaseSettings):
    """Runtime settings for the API, CLI and pipeline.

    All values can be overridden with ``NUEXTRACT_<FIELD_NAME>`` env vars.
    """

    model_config = SettingsConfigDict(
        env_prefix="NUEXTRACT_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- Application --------------------------------------------------------
    app_name: str = "NuExtract3 Document Studio"
    environment: Literal["local", "staging", "production"] = "local"
    log_level: str = "INFO"
    api_host: str = "0.0.0.0"
    api_port: int = 8080
    cors_origins: list[str] = Field(default_factory=lambda: ["*"])

    # --- Inference backend --------------------------------------------------
    # auto: probe vLLM -> Ollama -> in-process Transformers.
    provider: Provider = "auto"
    model_base_url: str = "http://localhost:8000/v1"
    model_api_key: str = "EMPTY"
    model_name: str = "numind/NuExtract3"
    request_timeout_s: float = 180.0
    max_retries: int = 2
    auto_probe_timeout_s: float = 1.5

    # --- Local inference (transformers provider) ----------------------------
    device: Device = "auto"
    dtype: DType = "auto"
    cpu_threads: int = 0  # 0 = let PyTorch decide
    local_max_output_tokens: int = 4096
    trust_remote_code: bool = True

    # --- Ollama (CPU/GPU, GGUF) --------------------------------------------
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "nuextract3:q4_k_m"
    # Forwarded as ``num_ctx`` on newer Ollama builds; also set the server-wide
    # OLLAMA_CONTEXT_LENGTH env var for older versions (see docker-compose.cpu.yml).
    ollama_num_ctx: int = 32768

    # --- Document handling --------------------------------------------------
    max_upload_mb: int = 50
    max_pages: int = 40
    dpi: int = 200
    native_text_min_chars: int = 120
    pages_per_request: int = 8

    # --- Generation ---------------------------------------------------------
    max_output_tokens: int = 16384
    temperature: float = 0.2
    thinking_temperature: float = 0.6
    markdown_temperature: float = 1.0
    enable_thinking: bool = False
    default_template: str = "invoice"

    # --- Templates ----------------------------------------------------------
    templates_dir: Path = Path("configs/templates")

    @field_validator("model_base_url", "ollama_base_url")
    @classmethod
    def _strip_trailing_slash(cls, value: str) -> str:
        return value.rstrip("/")

    @field_validator("templates_dir")
    @classmethod
    def _resolve_templates_dir(cls, value: Path) -> Path:
        # Relative paths are resolved from the current working directory so the
        # same config works locally and inside the container (WORKDIR=/app).
        return value if value.is_absolute() else (Path.cwd() / value).resolve()

    @property
    def max_upload_bytes(self) -> int:
        return self.max_upload_mb * 1024 * 1024

    @property
    def openai_base_url(self) -> str:
        """Normalise to an OpenAI-compatible ``/v1`` base URL."""
        if self.model_base_url.endswith("/v1"):
            return self.model_base_url
        return f"{self.model_base_url}/v1"

    @property
    def ollama_openai_base_url(self) -> str:
        """Ollama's OpenAI-compatible endpoint (``/v1``)."""
        if self.ollama_base_url.endswith("/v1"):
            return self.ollama_base_url
        return f"{self.ollama_base_url}/v1"


@lru_cache
def get_settings() -> Settings:
    return Settings()
