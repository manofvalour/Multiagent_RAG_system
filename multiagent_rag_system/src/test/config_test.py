"""
tests/conftest.py
Shared pytest fixtures.

Strategy:
  - VectorStore is mocked so tests never need a running Qdrant server
  - SemanticCache is mocked so tests never need Redis
  - Settings are loaded from a test-specific config dict
"""
from __future__ import annotations

from typing import Optional
from unittest.mock import AsyncMock, MagicMock, patch
import pytest

from ..utils.config_loader import (
    AuthConfig,
    CacheConfig,
    ChunkingConfig,
    EmbeddingsConfig,
    EvaluationConfig,
    LLMProviderConfig,
    ObservabilityConfig,
    QueryExpansionConfig,
    RateLimitConfig,
    RerankerConfig,
    RetrieverConfig,
    Settings,
    VectorStoreConfig,
)
from ..models.models import DocumentChunk, ContentType, RetrievedChunk, RerankedChunk


# ── Settings fixture ─────────────────────────────────────────────────────────

@pytest.fixture
def test_settings() -> Settings:
    """Minimal Settings for tests — no real API keys needed."""
    return Settings(
        OPENAI_API_KEY="sk-test",
        ANTHROPIC_API_KEY="",
        GROQ_API_KEY="",
        REDIS_URL="redis://localhost:6379/0",
        LANGSMITH_API_KEY="",
        QDRANT_API_KEY="",
        retriever=RetrieverConfig(top_k=3, similarity_threshold=0.5),
        reranker=RerankerConfig(enabled=False),        # disabled — no model download in CI
        query_expansion=QueryExpansionConfig(enabled=False),
        generator=LLMProviderConfig(provider="openai", model="gpt-4o-mini", stream=False),
        embeddings=EmbeddingsConfig(model="sentence-transformers/all-MiniLM-L6-v2", dimension=384),
        chunking=ChunkingConfig(chunk_size=256, chunk_overlap=32),
        cache=CacheConfig(enabled=False),
        vector_store=VectorStoreConfig(url="", local_path="/tmp/qdrant_test"),
        auth=AuthConfig(enabled=False),
        rate_limit=RateLimitConfig(requests_per_minute=1000),
        observability=ObservabilityConfig(otel_enabled=False, langsmith_enabled=False),
        evaluation=EvaluationConfig(enabled=False),
    )


# ── Sample data fixtures ──────────────────────────────────────────────────────

@pytest.fixture
def sample_chunks() -> list[DocumentChunk]:
    return [
        DocumentChunk(
            id="aaaaaaaa-0000-0000-0000-000000000001",
            content="RAG combines retrieval with generation to reduce hallucinations.",
            source="rag.txt", chunk_index=0, content_type=ContentType.PROSE,
            metadata={"doc_id": "doc-001"},
        ),
        DocumentChunk(
            id="aaaaaaaa-0000-0000-0000-000000000002",
            content="Qdrant is a vector database with native payload filtering.",
            source="qdrant.txt", chunk_index=0, content_type=ContentType.PROSE,
            metadata={"doc_id": "doc-002"},
        ),
        DocumentChunk(
            id="aaaaaaaa-0000-0000-0000-000000000003",
            content="HNSW offers the best query latency at scale with minimal accuracy loss.",
            source="hnsw.txt", chunk_index=0, content_type=ContentType.PROSE,
            metadata={"doc_id": "doc-003"},
        ),
    ]


@pytest.fixture
def sample_retrieved(sample_chunks) -> list[RetrievedChunk]:
    return [
        RetrievedChunk(chunk=sample_chunks[0], similarity_score=0.92),
        RetrievedChunk(chunk=sample_chunks[1], similarity_score=0.85),
        RetrievedChunk(chunk=sample_chunks[2], similarity_score=0.78),
    ]


@pytest.fixture
def sample_reranked(sample_retrieved) -> list[RerankedChunk]:
    return [
        RerankedChunk(chunk=r.chunk, similarity_score=r.similarity_score, rerank_score=r.similarity_score)
        for r in sample_retrieved
    ]


# ── Mock VectorStore ──────────────────────────────────────────────────────────

@pytest.fixture
def mock_vector_store(sample_retrieved):
    vs = MagicMock()
    vs.search   = AsyncMock(return_value=sample_retrieved)
    vs.add_chunks = AsyncMock()
    vs.count    = AsyncMock(return_value=3)
    vs.delete_document = AsyncMock(return_value=1)
    vs.collection_info = AsyncMock(return_value={
        "name": "rag_chunks", "points_count": 3, "status": "green"
    })
    vs.connect  = AsyncMock()
    return vs


# ── Mock SemanticCache ────────────────────────────────────────────────────────

@pytest.fixture
def mock_cache():
    cache = MagicMock()
    cache.get               = AsyncMock(return_value=None)
    cache.set               = AsyncMock()
    cache.connect           = AsyncMock()
    cache.close             = AsyncMock()
    cache.check_rate_limit  = AsyncMock(return_value=(True, 59))
    cache._client           = MagicMock()
    cache._client.ping      = AsyncMock()
    return cache


# ── Mock embed model ─────────────────────────────────────────────────────────

@pytest.fixture
def mock_embed_model():
    import numpy as np
    model = MagicMock()
    model.encode = MagicMock(
        return_value=np.random.rand(1, 384).astype("float32")
    )
    return model



if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v", "--asyncio-mode=auto"])