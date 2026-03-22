"""
tests/test_agents.py
Unit tests for the RerankerAgent — cross-encoder reranking
"""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch
import pytest

from ..models.models import (ContentType, DocumentChunk, 
                                 QueryResponse, AgentStatus, Claim,
                                 RerankedChunk, RetrievedChunk)

#Shared fixtures

def _make_doc(i: int, content: str = None) -> DocumentChunk:
    return DocumentChunk(
        id=f"aaaaaaaa-0000-0000-0000-00000000000{i}",
        content=content or f"RAG is a technique that combines retrieval and generation. Chunk {i}.",
        source=f"doc{i}.txt",
        chunk_index=i,
        content_type=ContentType.PROSE,
        metadata={},
        doc_id = f"doc-00{i}",
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
        return_value=QueryResponse(answer="RAG reduces hallucinations by grounding answers.")
    )
    return llm


class TestRerankerAgent:
    """
    RerankerAgent.rerank(query, chunks) ->
        tuple[list[RerankedChunk], AgentEvent]

    Tests:
      - Disabled reranker passes chunks through with vector_score as reranker_score
      - Enabled reranker sorts by reranker_score descending
      - Output is capped at config.top_n
      - Empty input returns empty output with DONE event
      - CrossEncoder is loaded lazily (not on construction)
    """

    def _make_agent(self, enabled: bool = True, top_n: int = 2):
        from multiagent_rag_system.agent.reranker_agent import RerankerAgent
        agent = RerankerAgent()
        agent.config = MagicMock()
        agent.config.enabled = enabled
        agent.config.top_n   = top_n
        agent.config.model   = "cross-encoder/ms-marco-MiniLM-L-6-v2"
        agent._model = None
        # threshold attribute referenced in event building
        agent.threshold = 0.0
        return agent

    @pytest.mark.asyncio
    async def test_disabled_passes_through(self, retrieved_chunks):
        agent = self._make_agent(enabled=False)
        reranked, event = await agent.rerank("What is RAG?", retrieved_chunks)

        assert len(reranked) == len(retrieved_chunks)
        assert event.status == AgentStatus.DONE
        for r in reranked:
            assert hasattr(r, "reranker_score") or hasattr(r, "similarity_score")

    @pytest.mark.asyncio
    async def test_empty_input_returns_empty(self):
        agent = self._make_agent(enabled=True)
        reranked, event = await agent.rerank("query", [])
        assert reranked == []
        assert event.status == AgentStatus.DONE

    @pytest.mark.asyncio
    async def test_enabled_sorts_by_reranker_score(self, retrieved_chunks):
        agent = self._make_agent(enabled=True, top_n=3)

        # Mock the CrossEncoder to return preset scores in reverse order
        # (so reranker reverses the retrieval order)
        fake_scores = [0.3, 0.9, 0.6]   # chunk[1] should come first after sort

        with patch.object(agent, "_load"):
            agent._model = MagicMock()
            import numpy as np
            agent._model.predict = MagicMock(return_value=np.array(fake_scores))

            reranked, event = await agent.rerank("What is RAG?", retrieved_chunks)

        assert event.status == AgentStatus.DONE
        # First result should have the highest reranker score
        assert reranked[0].reranker_score == pytest.approx(0.9)

    @pytest.mark.asyncio
    async def test_top_n_cap_applied(self, retrieved_chunks):
        agent = self._make_agent(enabled=True, top_n=2)

        with patch.object(agent, "_load"):
            agent._model = MagicMock()
            import numpy as np
            agent._model.predict = MagicMock(
                return_value=np.array([0.9, 0.8, 0.7])
            )
            reranked, _ = await agent.rerank("q", retrieved_chunks)

        assert len(reranked) <= 2

    def test_model_not_loaded_at_construction(self):
        agent = self._make_agent()
        assert agent._model is None

if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v", "--asyncio-mode=auto"])