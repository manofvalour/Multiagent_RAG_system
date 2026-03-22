"""
tests/test_agents.py
Unit tests for the agent orchestrator
"""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch
import pytest

from ..models.models import (DocumentChunk, 
                                 QueryRequest, QueryResponse,
                                 AgentEvent, AgentStatus, Claim, ConfidenceBreakdown,
                                 RerankedChunk, RetrievedChunk, HallucinationRisk)

def _make_doc(i: int, content: str = None) -> DocumentChunk:
    return DocumentChunk(
        id=f"aaaaaaaa-0000-0000-0000-00000000000{i}",
        content=content or f"RAG is a technique that combines retrieval and generation. Chunk {i}.",
        source=f"doc{i}.txt",
        chunk_index=i,
        doc_id=f"doc-00{i}",
        metadata={},
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

class TestRAGOrchestrator:
    """
    RAGOrchestrator.run(query) -> QueryResponse

    Tests use fully mocked agents so no real LLM/Qdrant/Redis calls are made.
    The orchestrator is tested as a black box: given mocked inputs, assert
    the output response shape and control flow.

    Agents mocked:
      expansion      — returns (["original"], None)
      retriever      — returns (sample_retrieved, event)
      reranker       — returns (sample_reranked, event)
      consensus      — returns ("Generated answer.", [...], event)
      claim_verifier — returns ([Claim(...)], event)
      confidence     — returns (ConfidenceBreakdown(...), LOW, event)
      evaluator      — returns None (sampled out)
      cache          — get=None (miss), set=noop
    """

    def _make_event(self, message: str = "ok") -> AgentEvent:
        return AgentEvent(agent="mock", status=AgentStatus.DONE, message=message)

    def _make_pipeline(self, retrieved_chunks, reranked_chunks, mock_cache):
        from multiagent_rag_system.agent.pipeline import RAGOrchestrator

        expansion = MagicMock()
        expansion.expand = AsyncMock(return_value=(["original query"], None))

        retriever = MagicMock()
        retriever.retrieve = AsyncMock(return_value=(retrieved_chunks, self._make_event("retrieved")))

        reranker = MagicMock()
        reranker.rerank = AsyncMock(return_value=(reranked_chunks, self._make_event("reranked")))

        consensus = MagicMock()
        consensus.run = AsyncMock(
            return_value=("Generated answer.", ["Generated answer."], self._make_event("consensus"))
        )

        claim_verifier = MagicMock()
        claim_verifier.run = AsyncMock(
            return_value=([Claim(text ="Generated answer.", supported=True)], self._make_event("claims"))
        )

        confidence_scorer = MagicMock()
        confidence_scorer.run = AsyncMock(
            return_value=(
                ConfidenceBreakdown(claim_support=0.9, avg_relevance=0.85, source_overlap=0.7, final=0.85),
                HallucinationRisk.LOW,
                self._make_event("confidence"),
            )
        )

        evaluator = MagicMock()
        evaluator.evaluate = AsyncMock(return_value=None)

        return RAGOrchestrator(
            expansion=expansion,
            retriever=retriever,
            reranker=reranker,
            consensus=consensus,
            cache=mock_cache,
            evaluator=evaluator,
            confidence_score=confidence_scorer,
            claim_verification=claim_verifier,
        )

    @pytest.mark.asyncio
    async def test_full_pipeline_returns_response(self, retrieved_chunks, reranked_chunks, mock_cache):
        pipeline = self._make_pipeline(retrieved_chunks, reranked_chunks, mock_cache)
        q = QueryRequest(query="What is RAG?")
        response = await pipeline.run(q)

        assert response.answer == "Generated answer."
        assert response.query_id == q.id
        assert len(response.sources) >= 1
        assert response.latency_ms > 0

    @pytest.mark.asyncio
    async def test_cache_hit_returns_early(self, retrieved_chunks, reranked_chunks):
        mock_cache = MagicMock()
        cached = MagicMock()
        cached.latency_ms = 0.0
        mock_cache.get = AsyncMock(return_value=cached)
        mock_cache.set = AsyncMock()

        pipeline = self._make_pipeline(retrieved_chunks, reranked_chunks, mock_cache)
        q = QueryRequest(query="What is RAG?")
        result = await pipeline.run(q)

        assert result is cached
        pipeline.retriever.retrieve.assert_not_awaited()
        pipeline.consensus.run.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_empty_retrieval_returns_graceful_message(self, reranked_chunks):
        mock_cache = MagicMock()
        mock_cache.get = AsyncMock(return_value=None)
        mock_cache.set = AsyncMock()

        from multiagent_rag_system.agent.pipeline import RAGOrchestrator

        expansion  = MagicMock(); expansion.expand   = AsyncMock(return_value=(["query"], None))
        retriever  = MagicMock(); retriever.retrieve = AsyncMock(
            return_value=([], AgentEvent(agent="r", status=AgentStatus.DONE, message="0 found", duration_ms=5.0))
        )
        reranker   = MagicMock()
        consensus  = MagicMock()
        verifier   = MagicMock()
        scorer     = MagicMock()
        evaluator  = MagicMock(); evaluator.evaluate = AsyncMock(return_value=None)

        pipeline = RAGOrchestrator(
            expansion=expansion, retriever=retriever, reranker=reranker,
            consensus=consensus, cache=mock_cache, evaluator=evaluator,
            confidence_score=scorer, claim_verification=verifier,
        )
        q = QueryRequest(query="obscure question")
        response = await pipeline.run(q)

        assert "could not find" in response.answer.lower()
        reranker.rerank.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_sources_deduplicated(self, retrieved_chunks, reranked_chunks, mock_cache):
        for c in reranked_chunks:
            c.chunk.source = "same.txt"

        pipeline = self._make_pipeline(retrieved_chunks, reranked_chunks, mock_cache)
        q = QueryRequest(query="What is it?")
        response = await pipeline.run(q)

        assert len(response.sources) == 1
        assert response.sources[0] == "same.txt"

    @pytest.mark.asyncio
    async def test_cache_set_after_generation(self, retrieved_chunks, reranked_chunks, mock_cache):
        pipeline = self._make_pipeline(retrieved_chunks, reranked_chunks, mock_cache)
        q = QueryRequest(query="What is Qdrant?")
        await pipeline.run(q)
        mock_cache.set.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_evaluator_called_fire_and_forget(self, retrieved_chunks, reranked_chunks, mock_cache):
        pipeline = self._make_pipeline(retrieved_chunks, reranked_chunks, mock_cache)
        q = QueryRequest(query="What is HNSW?")
        await pipeline.run(q)
        pipeline.evaluator.evaluate.assert_awaited_once()

    @pytest.fixture
    def mock_cache(self):
        cache = MagicMock()
        cache.get = AsyncMock(return_value=None)
        cache.set = AsyncMock()
        return cache


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v", "--asyncio-mode=auto"])