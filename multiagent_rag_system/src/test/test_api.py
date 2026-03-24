"""
tests/test_api.py
FastAPI endpoint tests using TestClient.
All agents and infrastructure are mocked — no Qdrant, Redis, or LLM calls.
"""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch
import pytest
from fastapi.testclient import TestClient

from app import app
from src.models.models import IngestResponse, QueryResponse

@pytest.fixture
def mock_pipeline(sample_reranked):
    resp = QueryResponse(
        request_id="test-query-id",
        query = "query",
        answer="Test answer from pipeline.",
        claims=["doc.txt"],
        retrieved_chunks=[],
        reranked_chunks=sample_reranked,
        expanded_queries=["original"],
        latency_ms=123.4,
        confidence= MagicMock(),
        hallucination_risk=MagicMock(),
        agent_trace=[]
    )
    pipeline = MagicMock()
    pipeline.run           = AsyncMock(return_value=resp)
    pipeline.run_streaming = AsyncMock(return_value=iter([]))
    return pipeline


@pytest.fixture
def mock_ingestion():
    ingestion = MagicMock()
    ingestion.ingest_text = AsyncMock(return_value=IngestResponse(
        document_id="doc-001", source="test.txt",
        num_chunks=3, content_type="prose", latency_ms=50.0,
    ))
    ingestion.ingest_file = AsyncMock(return_value=IngestResponse(
        document_id="doc-002", source="report.pdf",
        num_chunks=5, content_type="pdf", latency_ms=200.0,
    ))
    return ingestion


@pytest.fixture
def mock_vs():
    vs = MagicMock()
    vs.connect         = AsyncMock()
    vs.count           = AsyncMock(return_value=10)
    vs.delete_document = AsyncMock(return_value=3)
    vs.collection_info = AsyncMock(return_value={"name": "rag_chunks", "points_count": 10, "status": "green"})
    return vs


@pytest.fixture
def client(test_settings, mock_pipeline, mock_ingestion, mock_vs, mock_cache):
    """
    Build a test FastAPI app with all infrastructure mocked out via
    dependency injection on app.state.
    """
    app = app(test_settings)

    # Patch the lifespan so it injects mocks instead of real components
    async def mock_lifespan(app):
        app.state.pipeline     = mock_pipeline
        app.state.ingestion    = mock_ingestion
        app.state.vector_store = mock_vs
        app.state.cache        = mock_cache
        app.state.settings     = test_settings
        yield

    app.router.lifespan_context = mock_lifespan

    with TestClient(app, raise_server_exceptions=True) as c:
        yield c


class TestQueryEndpoint:
    def test_query_returns_200(self, client):
        r = client.post("/query", json={"text": "What is RAG?"})
        assert r.status_code == 200
        body = r.json()
        assert body["answer"] == "Test answer from pipeline."
        assert body["query_id"] == "test-query-id"

    def test_query_empty_text_returns_422(self, client):
        r = client.post("/query", json={"text": ""})
        assert r.status_code == 422

    def test_query_too_long_returns_422(self, client):
        r = client.post("/query", json={"text": "x" * 5000})
        assert r.status_code == 422

    def test_query_missing_body_returns_422(self, client):
        r = client.post("/query")
        assert r.status_code == 422


class TestIngestEndpoint:
    def test_ingest_text_returns_201(self, client):
        r = client.post("/ingest", json={
            "content": "RAG is a technique that combines retrieval with generation.",
            "source": "test.txt",
        })
        assert r.status_code == 201
        body = r.json()
        assert body["document_id"] == "doc-001"
        assert body["num_chunks"] == 3

    def test_ingest_file_pdf_returns_201(self, client):
        r = client.post(
            "/ingest/file",
            files={"file": ("report.pdf", b"%PDF-1.4 fake", "application/pdf")},
        )
        assert r.status_code == 201
        assert r.json()["document_id"] == "doc-002"

    def test_ingest_file_unsupported_type_returns_400(self, client):
        r = client.post(
            "/ingest/file",
            files={"file": ("data.xlsx", b"fake", "application/octet-stream")},
        )
        assert r.status_code == 400


class TestDeleteEndpoint:
    def test_delete_existing_document(self, client):
        r = client.delete("/documents/doc-001")
        assert r.status_code == 200
        assert r.json()["chunks_removed"] == 3

    def test_delete_nonexistent_returns_404(self, client, mock_vs):
        mock_vs.delete_document = AsyncMock(return_value=0)
        r = client.delete("/documents/does-not-exist")
        assert r.status_code == 404


class TestHealthEndpoint:
    def test_health_returns_healthy(self, client):
        r = client.get("/health")
        assert r.status_code == 200
        body = r.json()
        assert "status" in body
        assert "components" in body
        assert "qdrant" in body["components"]


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--asyncio-mode=auto"])
    #test_retrieval_validation_sorts_by_relevance()