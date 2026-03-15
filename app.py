"""
api/app.py — FastAPI application
Routes: /query, /ingest, /documents, /health, /metrics, /analytics
Middleware: CORS, rate-limiting, request-id injection, structured logging
"""
from __future__ import annotations
import time
import uuid
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException, Request, Response, Depends, Header, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, PlainTextResponse

import structlog

from multiagent_rag_system.agent.ingestion import DocumentIngestionPipeline
from multiagent_rag_system.agent.pipeline import MultiAgentRAGPipeline
from multiagent_rag_system.src.cache.cache import CacheClient
from multiagent_rag_system.src.utils.config_loader import get_settings
from multiagent_rag_system.src.logger.logger import GLOBAL_LOGGER as logger
from multiagent_rag_system.src.models.models import (
    HealthComponent, HealthResponse, IngestRequest, IngestResponse,
    QueryRequest, QueryResponse, QueryMetrics,
)
from multiagent_rag_system.src.database.vector_store import get_embedder, get_vector_store
from multiagent_rag_system.src.utils.metrics import (
    get_metrics_output, record_ingestion, record_query, track_request, update_store_size
)


settings = get_settings()

# ─── Singletons initialised at startup ───────────────────────────────────────
_pipeline: Optional[MultiAgentRAGPipeline] = None
_ingestion: Optional[DocumentIngestionPipeline] = None
_cache: Optional[CacheClient] = None
_start_time: float = 0.0


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _pipeline, _ingestion, _cache, _start_time
    _start_time = time.perf_counter()
    logger.info("startup", env=settings.environment, version=settings.app_version)

    # Warm up embedder and vector store
    await get_embedder()
    await get_vector_store()

    _pipeline  = MultiAgentRAGPipeline()
    _ingestion = DocumentIngestionPipeline()
    _cache     = CacheClient()

    # Seed knowledge base in dev mode
    if settings.environment == "development":
        await _seed_demo_data()

    logger.info("startup_complete")
    yield
    logger.info("shutdown")


app = FastAPI(
    title="Multi-Agent RAG API",
    version=settings.app_version,
    description="Production-grade RAG system with 5-agent hallucination reduction pipeline",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# ─── Middleware ───────────────────────────────────────────────────────────────

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
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
            detail=f"Rate limit exceeded. Max {settings.rate_limit_requests} requests per {settings.rate_limit_window_seconds}s",
            headers={"Retry-After": str(settings.rate_limit_window_seconds)},
        )


# ─── Routes ───────────────────────────────────────────────────────────────────

@app.post("/query", response_model=QueryResponse, dependencies=[Depends(rate_limit)])
async def query(req: QueryRequest, request: Request):
    """
    Run the full 5-agent RAG pipeline.

    - Checks query result cache first (Redis)
    - Retrieves from FAISS vector store
    - Runs: Validation → Consensus → Claim Verification → Confidence Scoring
    - Returns grounded answer with full provenance
    """
    async with track_request():
        # Cache check
        cached_result = await _cache.get_query_result(req.query, req.top_k)
        if cached_result:
            logger.info("cache_hit", query=req.query[:60])
            resp = QueryResponse(**cached_result)
            resp.cached = True
            return resp

        # Retrieve from vector store
        store = await get_vector_store()
        embedder = await get_embedder()
        query_emb = (await embedder.embed([req.query]))[0]
        raw_chunks = await store.search(query_emb, top_k=req.top_k * 2)

        # Run pipeline
        result = await _pipeline.run(req, raw_chunks)

        # Metrics
        n_supported = sum(1 for c in result.claims if c.supported)
        record_query(
            latency_ms=result.latency_ms,
            confidence=result.confidence.final,
            risk=result.hallucination_risk.value,
            cached=False,
            n_claims=len(result.claims),
            n_supported=n_supported,
            n_chunks=len(result.retrieved_chunks),
        )

        # Cache and history
        await _cache.set_query_result(req.query, req.top_k, result.model_dump())
        await _cache.lpush_bounded("rag:history", result.model_dump())

        return result


@app.post("/ingest", response_model=IngestResponse, status_code=status.HTTP_201_CREATED)
async def ingest(req: IngestRequest):
    """Ingest a document: chunk → embed → index in vector store."""
    result = await _ingestion.ingest(req)
    record_ingestion()
    store = await get_vector_store()
    update_store_size(await store.count())
    return result


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
        components.append(HealthComponent(name="vector_store", healthy=True, detail=f"{count} chunks"))
    except Exception as e:
        components.append(HealthComponent(name="vector_store", healthy=False, detail=str(e)))

    # Redis
    try:
        lat = await _cache.ping()
        components.append(HealthComponent(name="redis", healthy=True, latency_ms=round(lat, 2)))
    except Exception as e:
        components.append(HealthComponent(name="redis", healthy=False, detail=str(e)))

    # LLM
    try:
        from .multiagent_rag_system.src.llm.llm import get_llm_client
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


@app.get("/analytics", response_model=QueryMetrics)
async def analytics(window_minutes: int = 60):
    """Query performance and hallucination analytics over rolling window."""
    history = await _cache.lrange("rag:history", 0, 999)

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
        "health": "/health",
    }


# ─── Demo seeding ─────────────────────────────────────────────────────────────

DEMO_DOCS = [
    ("Retrieval-Augmented Generation (RAG) combines a retriever with a large language model. The retriever fetches relevant documents from an external corpus. The generator uses these documents to produce factually grounded answers, significantly reducing hallucination.", "RAG Overview"),
    ("Hallucination in large language models refers to the confident generation of factually incorrect or unsupported content. Grounding model responses in retrieved documents reduces hallucination rates by 35-45% compared to closed-book generation.", "LLM Hallucination Research"),
    ("Async Python using asyncio enables concurrent execution of I/O-bound tasks without multi-threading. The asyncio.gather function runs coroutines concurrently, making it ideal for parallel agent orchestration in multi-agent systems.", "Python Async Guide"),
    ("Multi-agent systems distribute complex tasks across specialised agents. Each agent handles a narrow sub-problem, and their outputs are aggregated. This division of labour improves accuracy, reduces individual agent errors, and provides interpretable audit trails.", "Multi-Agent Architecture"),
    ("Vector databases use dense numerical embeddings to enable semantic similarity search. Documents are converted to high-dimensional vectors, and queries are matched by cosine or dot-product similarity. Popular options include FAISS, pgvector, Pinecone, and Qdrant.", "Vector Search Systems"),
    ("Claim verification is a post-generation quality-control step. An LLM or lexical checker evaluates each atomic assertion in the generated answer against source documents. Claims without source support are flagged, substantially reducing user-facing hallucinations.", "Fact Verification Methods"),
    ("Consensus-based generation runs multiple LLM instances in parallel with slight temperature variation. The most internally consistent output is selected via voting. This ensemble approach reduces variance and single-model hallucinations by approximately 20-30%.", "Ensemble LLM Methods"),
    ("FAISS (Facebook AI Similarity Search) is an open-source library for efficient similarity search on dense vectors. It supports exact and approximate nearest neighbour search and scales to billions of vectors with GPU acceleration.", "FAISS Documentation"),
    ("Sentence-BERT (SBERT) produces fixed-size sentence embeddings optimised for semantic similarity tasks. Embeddings from SBERT models outperform averaged word vectors on most retrieval benchmarks and are widely used in RAG pipelines.", "Sentence Embeddings"),
    ("Confidence scoring aggregates multiple quality signals: claim support rate, average retrieval relevance, and source-answer word overlap. Scores below 0.4 indicate high hallucination risk and should trigger escalation or human review.", "RAG Quality Metrics"),
]


async def _seed_demo_data():
    store = await get_vector_store()
    if await store.count() > 0:
        return
    logger.info("seeding_demo_data", n_docs=len(DEMO_DOCS))
    for content, source in DEMO_DOCS:
        req = IngestRequest(content=content, source=source, chunk_strategy="sentence")
        await _ingestion.ingest(req)
    logger.info("demo_data_seeded")

