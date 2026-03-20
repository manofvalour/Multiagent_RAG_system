"""
tests/test_pipeline.py
Integration tests for the full RAG pipeline (orchestrator).
All external calls (LLM, Qdrant, Redis) are mocked.
"""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch
import pytest

from .agents.evaluator import RAGASEvaluator
from src.agents.generator import GeneratorAgent
from src.agents.orchestrator import RAGOrchestrator
from src.agents.query_expansion import QueryExpansionAgent
from src.agents.reranker import RerankerAgent
from src.agents.retriever import RetrieverAgent
from src.core.models import RAGQuery


@pytest.fixture
def pipeline(
    test_settings,
    mock_vector_store,
    mock_embed_model,
    mock_cache,
    sample_reranked,
):
    expansion = MagicMock(spec=QueryExpansionAgent)
    expansion.expand = AsyncMock(return_value=(["original query"], None))

    retriever = MagicMock(spec=RetrieverAgent)
    retriever.retrieve = AsyncMock(
        return_value=[MagicMock(chunk=r.chunk, similarity_score=r.similarity_score)
                      for r in sample_reranked]
    )

    reranker = MagicMock(spec=RerankerAgent)
    reranker.rerank = AsyncMock(return_value=sample_reranked)

    generator = MagicMock(spec=GeneratorAgent)
    generator.generate = AsyncMock(return_value="This is the generated answer.")

    evaluator = MagicMock(spec=RAGASEvaluator)
    evaluator.evaluate = AsyncMock(return_value=None)

    return RAGOrchestrator(expansion, retriever, reranker, generator, mock_cache, evaluator)


class TestRAGOrchestrator:
    @pytest.mark.asyncio
    async def test_full_pipeline_returns_response(self, pipeline):
        q = RAGQuery(text="What is RAG?")
        result = await pipeline.run(q)

        assert result.query_id == q.id
        assert result.answer == "This is the generated answer."
        assert len(result.sources) >= 1
        assert result.latency_ms > 0

    @pytest.mark.asyncio
    async def test_cache_hit_returns_early(self, pipeline, mock_cache):
        from src.core.models import RAGResponse
        cached_resp = RAGResponse(
            query_id="cached-id",
            answer="Cached answer",
            sources=["cached.txt"],
            retrieved_chunks=[],
            reranked_chunks=[],
        )
        mock_cache.get = AsyncMock(return_value=cached_resp)

        q = RAGQuery(text="What is RAG?")
        result = await pipeline.run(q)

        assert result.answer == "Cached answer"
        assert result.cached is True
        # Retriever should NOT have been called
        pipeline.retriever.retrieve.assert_not_called()

    @pytest.mark.asyncio
    async def test_empty_retrieval_returns_no_info_message(self, pipeline, mock_cache):
        mock_cache.get = AsyncMock(return_value=None)
        pipeline.retriever.retrieve = AsyncMock(return_value=[])

        q = RAGQuery(text="Something completely obscure")
        result = await pipeline.run(q)

        assert "could not find" in result.answer.lower()
        assert result.sources == []

    @pytest.mark.asyncio
    async def test_response_sources_are_deduplicated(self, pipeline, sample_reranked, mock_cache):
        # Make all reranked chunks point to the same source
        for chunk in sample_reranked:
            chunk.chunk.source = "same_source.txt"
        pipeline.reranker.rerank = AsyncMock(return_value=sample_reranked)
        mock_cache.get = AsyncMock(return_value=None)

        q = RAGQuery(text="Any question")
        result = await pipeline.run(q)

        assert len(result.sources) == 1
        assert result.sources[0] == "same_source.txt"

    @pytest.mark.asyncio
    async def test_cache_is_set_after_generation(self, pipeline, mock_cache):
        mock_cache.get = AsyncMock(return_value=None)
        q = RAGQuery(text="What is Qdrant?")
        await pipeline.run(q)
        mock_cache.set.assert_called_once()


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v", "--asyncio-mode=auto"])