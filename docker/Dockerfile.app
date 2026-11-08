# syntax=docker/dockerfile:1.7
# CPU image: FastAPI orchestrator + Streamlit demo. No model weights inside.
FROM python:3.12-slim AS runtime

COPY --from=ghcr.io/astral-sh/uv:0.9.5 /uv /uvx /bin/

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    PATH="/app/.venv/bin:$PATH"

WORKDIR /app

# Dependency layer (cached unless pyproject/lock change).
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-dev --no-install-project

# Application layer.
COPY src ./src
COPY apps ./apps
COPY configs ./configs
COPY eval ./eval
COPY scripts ./scripts
RUN uv sync --frozen --no-dev

# Non-root runtime user.
RUN useradd --create-home --uid 10001 appuser \
    && chown -R appuser:appuser /app
USER appuser
ENV HOME=/home/appuser

EXPOSE 8080 8501

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8080/health', timeout=3)"

CMD ["uvicorn", "nuextract_app.api.main:app", "--host", "0.0.0.0", "--port", "8080"]
