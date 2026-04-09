FROM python:3.13-slim AS builder

WORKDIR /build

# Copy requirements and install Python dependencies
# No build-essential needed if your packages are pure Python or have wheels
COPY requirements.txt .
RUN pip install --user --no-cache-dir -r requirements.txt

# Runtime stage
FROM python:3.13-slim

WORKDIR /app

# Copy Python dependencies from builder
COPY --from=builder /root/.local /root/.local

# Set environment variables
ENV PATH=/root/.local/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app

# Copy application code
COPY . .

# Create directories
RUN mkdir -p data/uploads logs

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD python -c "import httpx; httpx.get('http://localhost:8000/health', timeout=5)" || exit 1

EXPOSE 8000

CMD ["uvicorn", "multiagent_rag_system.app:app", "--host", "0.0.0.0", "--port", "8000"]