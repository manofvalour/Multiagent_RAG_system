# ---------- Builder ----------
FROM python:3.13-slim-bookworm AS builder

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/app/.venv

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/* \
    && pip install uv

COPY . .

RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev


# ---------- Runtime ----------
FROM python:3.13-slim-bookworm

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && pip install uv

COPY --from=builder /app /app

ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app \
    HF_HOME=/app/.cache/huggingface \
    SENTENCE_TRANSFORMERS_HOME=/app/.cache/sentence_transformers \
    HF_HUB_DISABLE_PROGRESS_BARS=1

RUN mkdir -p data/uploads logs \
    .cache/huggingface \
    .cache/sentence_transformers

HEALTHCHECK --interval=30s --timeout=10s --start-period=180s --retries=5 \
    CMD python -c "import httpx; r=httpx.get('http://localhost:8000/health', timeout=10); r.raise_for_status()" || exit 1

EXPOSE 8000

CMD ["uvicorn", "app:app", \ 
    "--host", "0.0.0.0", \
    "--port", "8000", \
     "--workers", "1", \
     "--timeout-keep-alive", "600", \
     "--log-level", "warning"]