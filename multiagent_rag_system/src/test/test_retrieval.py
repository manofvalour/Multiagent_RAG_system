"""
tests/test_agents.py
Unit tests for the ChunkRetrieval — multi-query ANN retrieval via Qdrant
"""
from __future__ import annotations

import json
from typing import Optional
from unittest.mock import AsyncMock, MagicMock, patch
import pytest

from ..models.models import (ContentType, DocumentChunk, 
                                 QueryResponse,
                                 AgentStatus, Claim,
                                 RerankedChunk, RetrievedChunk
)


def _make_doc(i: int, content: str = None) -> DocumentChunk:
    return DocumentChunk(
        id=f"aaaaaaaa-0000-0000-0000-00000000000{i}",
        content=content or f"RAG is a technique that combines retrieval and generation. Chunk {i}.",
        source=f"doc{i}.txt",
        chunk_index=i,
        metadata={},
        doc_id=f"doc-00{i}"
    )


@pytest.fixture
def docs():
    return [_make_doc(i) for i in range(1, 4)]


@pytest.fixture
def retrieved_chunks(docs) -> list[RetrievedChunk]:
    scores = [0.92, 0.85, 0.78]
    return [RetrievedChunk(chunk=d, vector_score=s) for d, s in zip(docs, scores)]


@pytest.fixture
def reranked_chunks(docs) -> list[RerankedChunk]:
    return [
        RerankedChunk(chunk=docs[0], similarity_score=0.92, reranker_score=0.95),
        RerankedChunk(chunk=docs[1], similarity_score=0.85, reranker_score=0.82),
        RerankedChunk(chunk=docs[2], similarity_score=0.78, reranker_score=0.70),
    ]


@pytest.fixture
def sample_claims(docs) -> list[Claim]:
    return [
        Claim(text="RAG reduces hallucinations.", supported=True,  confidence=0.9),
        Claim(text="Qdrant is a graph database.", supported=False, confidence=0.2),
        Claim(text="HNSW has sub-linear query time.", supported=True, confidence=0.85),
    ]


@pytest.fixture
def mock_llm():
    llm = MagicMock()
    llm.complete = AsyncMock(
        return_value=QueryResponse(
            answer="RAG reduces hallucinations by grounding answers.",
            query="test query",
            claims=[],
            retrieved_chunks=[],
            reranked_chunks=[],
            expanded_queries=[],
            confidence=0.8,
            hallucination_risk="LOW",
            latency_ms=100.0
        )
    )
    return llm

class TestChunkRetrieval:
    """
    ChunkRetrieval.retrieve(queries, filters) ->
        tuple[list[RetrievedChunk], AgentEvent]

    Tests:
      - Returns merged, deduplicated chunks sorted by vector_score desc
      - Deduplication keeps highest score per chunk id
      - Concurrent search: one task per query variant
      - Empty results returns empty list with DONE event
      - Filters are forwarded to vector store
    """

    def _make_agent(self, mock_vs, mock_embedder):
        from multiagent_rag_system.agent.retrieval_agent import ChunkRetrieval
        agent = ChunkRetrieval.__new__(ChunkRetrieval)
        agent.config = MagicMock()
        agent.config.top_k = 5
        agent.config.similarity_threshold = 0.5
        agent.config.hnsw_ef_search = 64

        import multiagent_rag_system.agent.retrieval_agent as ra_module
        ra_module.get_vector_store = mock_vs
        ra_module.get_embedder = mock_embedder

        return agent

    def _chunk_result(self, chunk_id: str, score: float, content: str = "RAG is a technique."):
        doc = DocumentChunk(
            id=chunk_id,
            content=content,
            source="test.txt",
            chunk_index=0,
            doc_id="doc-001",
            metadata={}
        )
        return RetrievedChunk(chunk=doc, vector_score=score)

    @pytest.mark.asyncio
    async def test_returns_merged_results(self):
        results = [
            self._chunk_result("aaaaaaaa-0000-0000-0000-000000000001", 0.92),
            self._chunk_result("aaaaaaaa-0000-0000-0000-000000000002", 0.85),
        ]
        mock_vs = MagicMock()
        mock_vs.search = AsyncMock(return_value=results)

        import numpy as np
        mock_embedder = MagicMock()
        mock_embedder.embed = MagicMock(
            return_value=np.random.rand(2, 4).astype("float32")
        )

        agent = self._make_agent(mock_vs, mock_embedder)
        merged, event = await agent.retrieve(["query 1", "query 2"])

        assert len(merged) >= 1
        assert event.status == AgentStatus.DONE

    @pytest.mark.asyncio
    async def test_deduplication_keeps_highest_score(self):
        """
        Same chunk id returned by two queries with different scores —
        only the higher score should survive.
        """
        chunk_id = "aaaaaaaa-0000-0000-0000-000000000001"
        results_q1 = [self._chunk_result(chunk_id, 0.92)]
        results_q2 = [self._chunk_result(chunk_id, 0.70)]

        mock_vs = MagicMock()
        mock_vs.search = AsyncMock(side_effect=[results_q1, results_q2])

        import numpy as np
        mock_embedder = MagicMock()
        mock_embedder.embed = MagicMock(
            return_value=np.random.rand(2, 4).astype("float32")
        )

        agent = self._make_agent(mock_vs, mock_embedder)
        merged, _ = await agent.retrieve(["q1", "q2"])

        assert len(merged) == 1
        assert merged[0].vector_score == pytest.approx(0.92)

    @pytest.mark.asyncio
    async def test_sorted_by_vector_score_descending(self):
        results = [
            self._chunk_result("aaaaaaaa-0000-0000-0000-000000000001", 0.60),
            self._chunk_result("aaaaaaaa-0000-0000-0000-000000000002", 0.95),
            self._chunk_result("aaaaaaaa-0000-0000-0000-000000000003", 0.80),
        ]
        mock_vs = MagicMock()
        mock_vs.search = AsyncMock(return_value=results)

        import numpy as np
        mock_embedder = MagicMock()
        mock_embedder.embed = MagicMock(
            return_value=np.random.rand(1, 4).astype("float32")
        )

        agent = self._make_agent(mock_vs, mock_embedder)
        merged, _ = await agent.retrieve(["query"])

        for i in range(len(merged) - 1):
            assert merged[i].vector_score >= merged[i + 1].vector_score

    @pytest.mark.asyncio
    async def test_empty_results_returns_empty_list(self):
        mock_vs = MagicMock()
        mock_vs.search = AsyncMock(return_value=[])

        import numpy as np
        mock_embedder = MagicMock()
        mock_embedder.embed = MagicMock(
            return_value=np.random.rand(1, 4).astype("float32")
        )

        agent = self._make_agent(mock_vs, mock_embedder)
        merged, event = await agent.retrieve(["obscure query"])

        assert merged == []
        assert event.status == AgentStatus.DONE

    @pytest.mark.asyncio
    async def test_filters_forwarded_to_vector_store(self):
        mock_vs = MagicMock()
        mock_vs.search = AsyncMock(return_value=[])

        import numpy as np
        mock_embedder = MagicMock()
        mock_embedder.embed = MagicMock(
            return_value=np.random.rand(1, 4).astype("float32")
        )

        agent = self._make_agent(mock_vs, mock_embedder)
        filters = {"must": [{"key": "source", "match": {"value": "report.pdf"}}]}
        await agent.retrieve(["query"], filters=filters)

        call_kwargs = mock_vs.search.call_args[1]
        assert call_kwargs.get("filters") == filters

if __name__=="__main__":
    import pytest
    pytest.main([__file__, "-v", "--asyncio-mode=auto"])
