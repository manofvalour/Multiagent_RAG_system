from __future__ import annotations
import time
from contextlib import asynccontextmanager
from typing import Callable

from ..utils.config_loader import get_settings
from ..exception.custom_exception import MulitagentragException
from ..logger import GLOBAL_LOGGER

settings = get_settings()
config = settings.observability

## setting up prometheus
try:
    from prometheus_client import Counter, Histogram, Gauge, CollectorRegistry
    REGISTRY = CollectorRegistry()

    QUERY_TOTAL = Counter(
        "rag_queries_total", "Total RAG queries",
        ["risk_level", 'cached'], registry=REGISTRY,
    )
    QUERY_LATENCY = Histogram(
        "rag_query_latency_ms", "End-to-end query latency",
        buckets=[50, 100, 200, 500, 1000, 2000,5000], registry=REGISTRY,
    )
    CONFIDENCE_HIST = Histogram(
        "rag_confidence_score", "Distribution of confidence scores",
        buckets = [0.1, 0.2,0.3,0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0], registry=REGISTRY,
    )

    CLAIM_VERIFIED = Counter(
        "rag_claim_verified_total", "Total claim verified",
        ['supported'], registry = REGISTRY,
    )
    CHUNKS_RETRIEVED = Histogram(
        "rag_chunks_retrieved", "Chunks per query after validation",
        buckets = [0,1,2,3,4,5,8,10], registry = REGISTRY
    )

    INGESTION_TOTAL = Counter(
        "rag_ingestion_total", "Total documents ingested", registry=REGISTRY
    )
    VECTOR_STORE_SIZE = Gauge(
        "rag_vector_store_chunks", "Current number of chunks in vector store", registry=REGISTRY
    )
    ACTIVE_REQUESTS = Gauge(
        "rag_active_reqests", "Current in_flight requests", registry=REGISTRY
    )
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False

def record_query(latency_ms: float, confidence:float, risk:str,
                 cached:bool, n_claims: int, n_supported:int,
                 n_chunks:int):
    if not PROMETHEUS_AVAILABLE or not config.enable_metrics:
        return
    QUERY_TOTAL.labels(risk_level =risk, cached=str(cached)).inc()
    QUERY_LATENCY.observe(latency_ms)
    CONFIDENCE_HIST.observe(confidence)
    CLAIM_VERIFIED.labels(supported = "true").inc(n_supported)
    CLAIM_VERIFIED.labels(supported='false').inc(n_claims - n_supported)
    CHUNKS_RETRIEVED.observe(n_chunks)

def record_ingestion():
    if not PROMETHEUS_AVAILABLE or not config.enable_metrics:
        return
    
    INGESTION_TOTAL.inc()

def update_store_size(size:int):
    if not PROMETHEUS_AVAILABLE or not config.enable_metrics:
        return
    VECTOR_STORE_SIZE.set(size)

@asynccontextmanager
async def track_request():
    if PROMETHEUS_AVAILABLE and config.enable_metrics:
        ACTIVE_REQUESTS.inc()
    try:
        yield
    finally:
        if PROMETHEUS_AVAILABLE and config.enable_metrics:
            ACTIVE_REQUESTS.dec()

def get_metrics_output()-> bytes:
    if not PROMETHEUS_AVAILABLE:
        return b"# Prometheus client not installed\n"
    
    from prometheus_client import generate_latest
    return generate_latest(REGISTRY)