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

from fastapi import FastAPI, HTTPException, Request, Response, Depends, Header, status, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, PlainTextResponse, FileResponse
from multiagent_rag_system.src.observability.observability import setup_observability

import structlog
import os

from multiagent_rag_system.agent.agents.doc_ingestion import DocumentIngestionPipeline
from multiagent_rag_system.agent.pipeline.pipeline import RAGOrchestrator
from multiagent_rag_system.agent.agents.paper_reader_agent import PaperReaderAgent
from multiagent_rag_system.src.fetcher.paper_fetcher_service import PaperFetcherService
from multiagent_rag_system.src.cache.cache import CacheClient
from multiagent_rag_system.src.utils.config_loader import get_settings
from multiagent_rag_system.src.logger import GLOBAL_LOGGER as logger
from multiagent_rag_system.src.models.models import (
    HealthComponent, HealthResponse, IngestRequest, IngestResponse,
    QueryRequest, QueryResponse, QueryMetrics,
    PaperSource, PaperMetadata, PaperAnalysis, PaperSearchRequest,
    PaperSearchResponse, DocumentChunk,
)
from multiagent_rag_system.src.database.vector_store import get_vector_store, get_paper_vector_store
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
_paper_fetcher: Optional[PaperFetcherService] = None
_paper_reader: Optional[PaperReaderAgent] = None
_start_time: float = 0.0


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _pipeline, _ingestion, _cache, _paper_fetcher, _paper_reader, _start_time
    _start_time = time.perf_counter()
    logger.info("startup", env=settings.environment, version=settings.app_version)

    setup_observability()  # Configures both OpenTelemetry and LangSmith based on settings

    # Warm up embedder and vector store
    await get_embedder()
    await get_vector_store()
    await get_paper_vector_store()

    # Initialize singletons once here (not at module level)
    _pipeline  = RAGOrchestrator()
    _ingestion = DocumentIngestionPipeline()
    _cache     = CacheClient()
    _paper_fetcher = PaperFetcherService()
    _paper_reader = PaperReaderAgent()

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
        n_supported = sum(1 for c in result.claims if c.supported)
        record_query(
            latency_ms=result.latency_ms,
            confidence=result.confidence.final,
            risk=result.hallucination_risk.value,
            cached=False,
            n_claims=len(result.claims),
            n_supported=n_supported,
            n_chunks=len(result.reranked_chunks),
        )

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
    result = await _ingestion.ingest_file(content=file_content, filename=file.filename)
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
    cached_hits  = sum(1 for h in history if h.get("cached"))

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


# ─── Paper Reader Endpoints ────────────────────────────────────────────────────

@app.get("/papers/search", response_model=PaperSearchResponse)
async def search_papers(
    q: str,
   # sources: list[str]= ["arxiv"],
    max_results: int = 10,
    topic_filter: Optional[str] = None,
):
    """Search for research papers across multiple sources."""
    #source_list = [PaperSource(s.strip()) for s in sources.split(",")]

    papers = await _paper_fetcher.fetch_arxiv_papers(q, max_results)

    return PaperSearchResponse(papers=papers, total=len(papers))


@app.get("/papers/{paper_id}", response_model=PaperMetadata)
async def get_paper(paper_id: str):
    """Get paper metadata by ID. Checks cache, then vector store."""
    # Try cache
    cached = await _cache.get(f"paper:{paper_id}")
    if cached:
        return PaperMetadata(**cached)

    # Try to get from paper vector store
    try:
        store = await get_paper_vector_store()
        results = await store.search_paper_by_id(paper_id)
        if results:
            return results[0]
    except Exception:
        pass

    raise HTTPException(status_code=404, detail=f"Paper {paper_id} not found")


@app.post("/papers/{paper_id}/read", response_model=PaperAnalysis)
async def read_paper(
    paper_id: str,
    include_math: bool = True,
    include_code: bool = True,
):
    """Full AI-powered paper reading with intuition, math, and code breakdown."""
    # Get paper metadata
    paper = await get_paper(paper_id)

    # Scrape PDF content for this single paper
    if paper.pdf_url:
        content = await _paper_fetcher.scrape_paper_pdf(paper.pdf_url)
    else:
        content = None

    # Build chunks from scraped content (split into sections)
    chunks = []
    if content:
        # Simple chunking: split on double newlines or by character count
        sections = content.split('\n\n')
        for i, section in enumerate(sections):
            if len(section.strip()) > 50:  # Skip tiny sections
                chunks.append(DocumentChunk(
                    id=str(uuid.uuid4()),
                    doc_id=f"paper-{paper_id}",
                    content=section.strip()[:2000],  # Cap at 2000 chars per chunk
                    chunk_index=i,
                ))

    # If no content from PDF, fall back to abstract as single chunk
    if not chunks and paper.abstract:
        chunks.append(DocumentChunk(
            id=str(uuid.uuid4()),
            doc_id=f"paper-{paper_id}",
            content=paper.abstract,
            chunk_index=0,
        ))

    # Run paper reader agent
    analysis, event = await _paper_reader.run(
        paper=paper,
        chunks=chunks,
        include_math=include_math,
        include_code=include_code,
    )

    # Cache analysis result
    await _cache.set(
        f"paper:analysis:{paper_id}",
        analysis.model_dump_json(),
        ex=86400,
    )

    return analysis


@app.get("/")
async def root():
    return {
        "name": settings.app_name,
        "version": settings.app_version,
        "docs": "/docs",
        "health": "/health"
    }



