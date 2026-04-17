FROM python:3.12-slim-bookworm AS builder

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy everything first
COPY . .

# Single sync — installs all deps including the project itself
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev

FROM python:3.12-slim-bookworm

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY --from=builder /app/.venv /app/.venv
COPY --from=builder /app /app

ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app \
    HF_HOME=/app/.cache/huggingface \
    SENTENCE_TRANSFORMERS_HOME=/app/.cache/sentence_transformers \
    HF_HUB_DISABLE_PROGRESS_BARS=1

RUN mkdir -p data/uploads logs .cache/huggingface .cache/sentence_transformers

HEALTHCHECK --interval=30s --timeout=10s --start-period=180s --retries=5 \
    CMD /app/.venv/bin/python -c "import httpx; r=httpx.get('http://localhost:8000/health', timeout=5); r.raise_for_status()" || exit 1

EXPOSE 8000

CMD ["/app/.venv/bin/uvicorn", "app:app", \
    "--host", "0.0.0.0", \
    "--port", "8000", \
    "--workers", "2", \
    "--timeout-keep-alive", "600", \
    "--log-level", "warning"]