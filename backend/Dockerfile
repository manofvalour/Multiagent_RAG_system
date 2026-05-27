# ---------- Builder ----------
FROM python:3.13-slim-bookworm AS builder

WORKDIR /app

# Install build deps and uv
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/* \
    && pip install --no-cache-dir uv

# Copy only dependency files first (better layer caching)
COPY pyproject.toml uv.lock* ./

# Install dependencies (cached by uv)
ENV UV_HTTP_TIMEOUT=300 \
    UV_CONCURRENT_DOWNLOADS=50
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev

# Copy source code AFTER dependencies (only re-runs on code changes)
COPY . .

# ---------- Runtime ----------
FROM python:3.13-slim-bookworm

WORKDIR /app

# Install runtime deps
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy venv from builder
COPY --from=builder /app/.venv /app/.venv
COPY --from=builder /app/pyproject.toml /app/pyproject.toml
COPY --from=builder /app/multiagent_rag_system /app/multiagent_rag_system
COPY --from=builder /app/app.py /app/app.py

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

# Simpler healthcheck using wget (available in base image)
HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
    CMD wget --no-verbose --tries=1 --spider http://localhost:8000/health || exit 1

EXPOSE 8000

CMD ["uvicorn", "app:app", \
    "--host", "0.0.0.0", \
    "--port", "8000", \
    "--workers", "1", \
    "--timeout-keep-alive", "60", \
    "--log-level", "warning"]