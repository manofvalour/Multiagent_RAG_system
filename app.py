"""
app.py -- FastAPI application
Routes: /query,/ingest_text, /ingest_file, /delete_documents, /health, /metrics, /analytics
Middleware: CORS, rate-limiting, request-id injection, structured logging
"""
from __future__ import annotations
import asyncio
import time
import uuid
from contextlib import asynccontextmanager
from typing import Optional
import datetime
import json

from fastapi import FastAPI, HTTPException, Request, Response, Depends, Header, status, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, PlainTextResponse, FileResponse
from multiagent_rag_system.src.observability.observability import setup_observability, setup_otel

import structlog
import os

from multiagent_rag_system.agent.agents.doc_ingestion import DocumentIngestionPipeline
from multiagent_rag_system.agent.pipeline.pipeline import RAGOrchestrator
from multiagent_rag_system.src.cache.cache import CacheClient
from multiagent_rag_system.src.utils.config_loader import get_settings
from multiagent_rag_system.src.logger import GLOBAL_LOGGER as logger
from multiagent_rag_system.src.models.models import (
    HealthComponent, HealthResponse, IngestRequest, IngestResponse,
    QueryRequest, QueryResponse, QueryMetrics, DocumentChunk,
)
from multiagent_rag_system.src.database.vector_store import get_vector_store
from multiagent_rag_system.src.embedding.embedding import get_embedder
from multiagent_rag_system.src.utils.metrics import (
    get_metrics_output, record_ingestion, record_query, 
    track_request, update_store_size
)


settings = get_settings()
config = settings.server

#File upload configuration
UPLOAD_DIR = "data/uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

#Singletons — initialized once in lifespan (avoids double init)
_pipeline: Optional[RAGOrchestrator] = None
_ingestion: Optional[DocumentIngestionPipeline] = None
_cache: Optional[CacheClient] = None
_start_time: float = 0.0


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _pipeline, _ingestion, _cache, _paper_fetcher, _paper_reader, _start_time
    _start_time = time.perf_counter()
    logger.info("startup", env=settings.environment, version=settings.app_version)

    setup_observability()  # Configures both OpenTelemetry and LangSmith based on settings
    # Enable console exporter if OTEL_CONSOLE_EXPORT=true (for debugging)
    use_console = os.getenv("OTEL_CONSOLE_EXPORT", "").lower() == "true"
    setup_otel(use_console_exporter=use_console)

    # Warm up embedder and vector store
    await get_embedder()
    await get_vector_store()

    # Initialize singletons once here (not at module level)
    _pipeline  = RAGOrchestrator()
    _ingestion = DocumentIngestionPipeline()
    _cache     = CacheClient()

    logger.info("startup_complete")
    yield
    
    # Graceful shutdown: clean up resources
    try:
        logger.info("shutdown_cleanup", message="Cleaning up resources")
        # Give any pending tasks a short window to complete
        await asyncio.sleep(0.5)
    except Exception as e:
        logger.error("shutdown_cleanup_error", error=str(e))
    
    logger.info("shutdown")


app = FastAPI(
    title="Multi-Agent RAG API",
    version=settings.app_version,
    description="Production-grade RAG system with 7-agent hallucination reduction pipeline",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

#Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=config.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def request_middleware(request: Request, call_next):
    request_id = str(uuid.uuid4())[:8]
    structlog.contextvars.bind_contextvars(request_id=request_id)
    t0 = time.perf_counter()
    try:
        response = await call_next(request)
        latency = round((time.perf_counter() - t0) * 1000, 2)
        response.headers["X-Request-Id"] = request_id
        response.headers["X-Response-Time"] = f"{latency}ms"
        logger.info("http_request",
                    method=request.method, path=request.url.path,
                    status=response.status_code, latency_ms=latency)
        return response
    
    except Exception as e:
        logger.error("unhandled_error", error=str(e))
        return JSONResponse(status_code=500, content={"detail": "Internal server error"})
    finally:
        structlog.contextvars.unbind_contextvars("request_id")


#Rate limiting dependency
async def rate_limit(request: Request):
    identifier = request.headers.get("X-Forwarded-For", request.client.host if request.client else "unknown")
    allowed, remaining = await _cache.check_rate_limit(identifier)
    if not allowed:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=f"Rate limit exceeded. Max {settings.cache.requests_per_minute} requests per {settings.cache.window_seconds}s",
            headers={"Retry-After": str(settings.cache.window_seconds)},
        )


# API Routes
@app.post("/query", response_model=QueryResponse, dependencies=[Depends(rate_limit)])
async def query(req: QueryRequest):
    """
    Run the full 7-agent RAG pipeline.

    - Checks query result cache first (Redis)
    - Retrieves from Qdrant vector store
    - Runs: Retrieval -> Reranker -> Consensus -> Claim Verification -> Confidence Scoring
    - Returns grounded answer with full provenance
    """
    async with track_request():
        # Run pipeline
        result = await _pipeline.run(req)

        # Metrics
        is_cached = getattr(result, 'cached', False)
        n_supported = sum(1 for c in result.claims if c.supported)
        n_chunks = len(getattr(result, 'reranked_chunks', []))
        record_query(
            latency_ms=result.latency_ms,
            confidence=result.confidence.final,
            risk=result.hallucination_risk.value,
            cached=is_cached,
            n_claims=len(result.claims),
            n_supported=n_supported,
            n_chunks=n_chunks,
        )

        # Store analytics data in Redis for /analytics endpoint
        analytics_entry = {
            "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "query": req.query,
            "latency_ms": result.latency_ms,
            "confidence": {"final": result.confidence.final},
            "hallucination_risk": result.hallucination_risk.value,
            "cached": is_cached,
            "claims_count": len(result.claims),
            "supported_claims": n_supported,
            "retrieved_chunks": [
                {"chunk": {"source": chunk.chunk.source, "text": chunk.chunk.content[:200] if chunk.chunk.content else None, "score": chunk.reranker_score}}
                for chunk in getattr(result, 'reranked_chunks', [])
            ],
        }
        await _cache.lpush_bounded("rag:index", analytics_entry, max_len=1000)

        return result


@app.post("/ingest_text", response_model=IngestResponse, status_code=status.HTTP_201_CREATED)
async def ingest_text(req: IngestRequest):
    """Ingest a text document: chunk -> embed -> index in vector store."""
    result = await _ingestion.ingest_text(req)
    record_ingestion()
    store = await get_vector_store()
    update_store_size(await store.count())
    return result

@app.post("/ingest_file", response_model=IngestResponse, status_code=status.HTTP_201_CREATED)
async def ingest_file(file: UploadFile = File(...)):
    """Ingest a file(pdf, docx, images, or ppt): chunk -> embed -> index in vector store."""
    # Read file bytes
    file_content = await file.read()
    
    # Ingest the file
    result = await _ingestion.ingest_file(content=file_content, filename=file.filename, metadata={"source": file.filename})
    record_ingestion()
    store = await get_vector_store()
    update_store_size(await store.count())
    return result


@app.get("/documents", response_model=list[str])
async def get_documents():
    """
    Retrieve the list of all unique document ids in the vector store
    """
    _store = await get_vector_store()
    try:
        doc_ids = await _store.get_all_document_ids()
        return doc_ids

    except Exception as e:
        logger.error(f"Error fetching documents: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@app.delete("/documents/{doc_id}")
async def delete_document(doc_id: str):
    """Remove all chunks for a document from the vector store."""
    store = await get_vector_store()
    removed = await store.delete_document(doc_id)
    if removed == 0:
        raise HTTPException(status_code=404, detail=f"Document {doc_id} not found")
    await _cache.delete(f"rag:doc:{doc_id}")
    update_store_size(await store.count())
    return {"document_id": doc_id, "chunks_removed": removed}


@app.get("/health", response_model=HealthResponse)
async def health():
    """Deep health check: vector store, Redis, LLM provider."""
    components: list[HealthComponent] = []

    # Vector store
    try:
        store = await get_vector_store()
        count = await store.count()
        components.append(HealthComponent(name="Vector_Store", healthy=True, detail=f"{count} chunks"))
    except Exception as e:
        components.append(HealthComponent(name="Vector_Store", healthy=False, detail=str(e)))

    # Redis
    try:
        lat = await _cache.ping()
        components.append(HealthComponent(name="Redis", healthy=True, latency_ms=round(lat, 2)))
    except Exception as e:
        components.append(HealthComponent(name="Redis", healthy=False, detail=str(e)))

    # LLM
    try:
        from multiagent_rag_system.src.llm.llms import get_llm_client
        llm = get_llm_client()
        ok = await llm.health_check()
        components.append(HealthComponent(name="llm", healthy=ok, detail=type(llm).__name__))
    except Exception as e:
        components.append(HealthComponent(name="llm", healthy=False, detail=str(e)))

    all_healthy = all(c.healthy for c in components)
    any_healthy = any(c.healthy for c in components)

    return HealthResponse(
        status="healthy" if all_healthy else ("degraded" if any_healthy else "unhealthy"),
        version=settings.app_version,
        components=components,
        uptime_s=round(time.perf_counter() - _start_time, 1),
    )


@app.get("/metrics", response_class=PlainTextResponse, include_in_schema=False)
async def metrics():
    """Prometheus metrics endpoint."""
    return PlainTextResponse(get_metrics_output(), media_type="text/plain; version=0.0.4")


def _parse_history_timestamp(entry):
    timestamp = entry.get("timestamp") or entry.get("created_at") or entry.get("ts")
    if timestamp is None:
        return None

    if isinstance(timestamp, (int, float)):
        return datetime.datetime.fromtimestamp(timestamp, tz=datetime.timezone.utc)

    if isinstance(timestamp, str):
        iso = timestamp.strip()
        for fmt in ("%Y-%m-%dT%H:%M:%S.%fZ", "%Y-%m-%dT%H:%M:%S.%f", "%Y-%m-%dT%H:%M:%SZ",
                    "%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M:%S.%f"):
            try:
                return datetime.datetime.strptime(iso, fmt).replace(tzinfo=datetime.timezone.utc)
            except ValueError:
                pass
        try:
            # Parse ISO format with timezone and normalize to UTC
            parsed = datetime.datetime.fromisoformat(iso.replace("Z", "+00:00"))
            if parsed.tzinfo is None:
                parsed = parsed.replace(tzinfo=datetime.timezone.utc)
            return parsed
        except ValueError:
            pass

    return None


@app.get("/analytics", response_model=QueryMetrics)
async def analytics(window_minutes: int = 60):
    """Query performance and hallucination analytics over rolling window."""
    history = await _cache.lrange("rag:index", 0, -1)
   # history = [_normalize_history_item(item) for item in raw_history]

    cutoff = datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(minutes=window_minutes)
    filtered_history = []
    for item in history:
        ts = _parse_history_timestamp(item)
        if ts is None or ts >= cutoff:
            filtered_history.append(item)

    history = filtered_history

    if not history:
        return QueryMetrics(
            window_minutes=window_minutes, total_queries=0, avg_confidence=0.0,
            avg_latency_ms=0.0, risk_distribution={"LOW": 0, "MEDIUM": 0, "HIGH": 0},
            cache_hit_rate=0.0, top_sources=[],
        )

    confidences = [h.get("confidence", {}).get("final", 0) for h in history]
    latencies   = [h.get("latency_ms", 0) for h in history]
    risks       = [h.get("hallucination_risk", "MEDIUM") for h in history]
    cached_hits = sum(1 for h in history if h.get("cached"))

    source_counts: dict[str, int] = {}
    for h in history:
        for chunk in h.get("retrieved_chunks", []):
            src = chunk.get("chunk", {}).get("source", "unknown")
            source_counts[src] = source_counts.get(src, 0) + 1

    top_sources = sorted(
        [{"source": k, "count": v} for k, v in source_counts.items()],
        key=lambda x: x["count"], reverse=True
    )[:10]

    return QueryMetrics(
        window_minutes=window_minutes,
        total_queries=len(history),
        avg_confidence=round(sum(confidences) / max(len(confidences), 1), 3),
        avg_latency_ms=round(sum(latencies) / max(len(latencies), 1), 1),
        risk_distribution={
            "LOW":    risks.count("LOW"),
            "MEDIUM": risks.count("MEDIUM"),
            "HIGH":   risks.count("HIGH"),
        },
        cache_hit_rate=round(cached_hits / max(len(history), 1), 3),
        top_sources=top_sources,
    )

@app.get("/")
async def root():
    return {
        "name": settings.app_name,
        "version": settings.app_version,
        "docs": "/docs",
        "health": "/health"
    }



